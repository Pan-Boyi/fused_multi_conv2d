"""Verify both installed CANN fusion stages; this module implements no fusion."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re

FIRST_PASS_NAME = 'Conv2DFixPipeToExtendConv2DFusionPass'
PASS_NAME = 'ExtendConv2DPairToFusedMultiConv2DFusionPass'
PR_URL = 'https://gitcode.com/cann/canndev/pull/82567/diffs'
FUSED_TYPE = 'FusedMultiConv2d'
CONV_TYPES = {'Conv', 'Conv2D', 'Conv2DV2', 'ExtendConv2D', 'Convolution',
              'DepthwiseConv2D', 'QuantConv2D'}
SOURCE_COMPUTE_TYPES = CONV_TYPES | {'AscendRequant', 'AscendDequant', 'AscendQuant', 'Relu', 'FixPipe'}


def check_input_model(model):
    """Require the real two-Conv source, with neither fusion target present."""
    def walk(nodes):
        for node in nodes:
            if node.op_type in (FUSED_TYPE, 'ExtendConv2D'):
                raise ValueError('input model already contains %s; fusion was bypassed' % node.op_type)
            for attr in node.attribute:
                if attr.type == 5:
                    walk(attr.g.node)
                elif attr.type == 10:
                    for graph in attr.graphs:
                        walk(graph.node)
    walk(model.graph.node)
    for function in model.functions:
        walk(function.node)
    convs = [node for node in model.graph.node if node.op_type == 'Conv']
    if len(convs) != 2:
        raise ValueError('source graph must contain exactly two Conv nodes')
    producers = {output: node for node in model.graph.node for output in node.output}
    current = convs[1].input[0]
    seen = set()
    while current != convs[0].output[0]:
        node = producers.get(current)
        if node is None or node.op_type not in ('Relu', 'AscendRequant', 'AscendDequant') or current in seen:
            raise ValueError('source Conv nodes do not form one consecutive computation')
        seen.add(current)
        current = node.input[0]
    # Only the second layer's result is a model output: no extra mid consumer.
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers[name] = consumers.get(name, 0) + 1
    for node in model.graph.node:
        if node is convs[1]:
            continue
        for name in node.output:
            if name in seen or name == convs[0].output[0]:
                if consumers.get(name) != 1 or any(out.name == name for out in model.graph.output):
                    raise ValueError('intermediate result has additional consumers')
    return {'source_conv_count': 2}


def operators(document):
    if not isinstance(document, dict) or not isinstance(document.get('graph'), list) or not document['graph']:
        raise ValueError('invalid OM graph JSON: expected a nonempty graph list')
    result = []
    for graph in document['graph']:
        if not isinstance(graph, dict) or not isinstance(graph.get('op'), list):
            raise ValueError('invalid OM graph JSON: expected an operator list')
        for op in graph['op']:
            if not isinstance(op, dict) or not isinstance(op.get('type'), str):
                raise ValueError('invalid OM graph JSON: operator type is missing')
            result.append(op)
    return result


def check_inputs(ops, expected):
    names = [op.get('name') for op in ops if op['type'] == 'Data']
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError('OM Data inputs differ from case: got %s, expected %s' % (names, expected))


def check_stage1_graph(document, input_names):
    ops = operators(document)
    check_inputs(ops, input_names)
    convs = [op for op in ops if op['type'] == 'ExtendConv2D']
    if len(convs) != 2 or any(op['type'] in SOURCE_COMPUTE_TYPES - {'ExtendConv2D'} or op['type'] == FUSED_TYPE for op in ops):
        raise ValueError('stage 1 must produce exactly two ExtendConv2D nodes; check the installed Conv/Fixpipe pass')
    connected = []
    for first, second in ((convs[0], convs[1]), (convs[1], convs[0])):
        inputs = second.get('input', [])
        if inputs and inputs[0] in (first['name'] + ':0', first['name'] + ':y0'):
            connected.append((first['name'], second['name']))
    if len(connected) != 1:
        raise ValueError('stage 1 ExtendConv2D nodes are not connected directly on y0 -> x')
    return {'stage1_extend_count': 2, 'stage1_nodes': list(connected[0])}


def check_compiled_graph(document, input_names):
    ops = operators(document)
    check_inputs(ops, input_names)
    fused = [op for op in ops if op['type'] == FUSED_TYPE]
    if len(fused) != 1:
        raise ValueError('stage 2 requires one FusedMultiConv2d node, found %d; check PR 82567 and its switch/SoC constraints' % len(fused))
    remaining = [op for op in ops if op['type'] in SOURCE_COMPUTE_TYPES]
    if remaining:
        raise ValueError('unfused source computations remain: %s' % ', '.join(op.get('name', op['type']) for op in remaining))
    return {'fused_node_count': 1, 'fused_node_name': fused[0].get('name', ''),
            'unfused_convolution_count': 0, 'input_names': input_names}


@contextlib.contextmanager
def graph_dump_environment(directory):
    """Collect both fusion rounds from the same ATC invocation."""
    values = {'DUMP_GE_GRAPH': '2', 'DUMP_GRAPH_LEVEL': '1',
              'DUMP_GRAPH_FORMAT': 'ge_proto', 'DUMP_GRAPH_PATH': str(Path(directory).resolve()),
              'ASCEND_DUMP_PATH': None, 'NPU_COLLECT_PATH': None}
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_ge_proto_snapshot(path):
    """Read the topology projection of GE's ModelDef protobuf text dump."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format
    descriptor = descriptor_pb2.FileDescriptorProto(name='fc2d_graph_dump.proto', package='fc2d_dump')
    for name, fields in (
        ('OpDef', [('name', 1, 9, 1, ''), ('type', 2, 9, 1, ''), ('input', 3, 9, 3, '')]),
        ('GraphDef', [('name', 1, 9, 1, ''), ('op', 2, 11, 3, '.fc2d_dump.OpDef')]),
        ('ModelDef', [('graph', 1, 11, 3, '.fc2d_dump.GraphDef')]),
    ):
        message = descriptor.message_type.add(name=name)
        for field_name, number, kind, label, type_name in fields:
            field = message.field.add(name=field_name, number=number, type=kind, label=label)
            if type_name:
                field.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    model_type = pool.FindMessageTypeByName('fc2d_dump.ModelDef')
    if hasattr(message_factory, 'GetMessageClass'):
        model = message_factory.GetMessageClass(model_type)()
    else:
        model = message_factory.MessageFactory(pool).GetPrototype(model_type)()
    try:
        text_format.Parse(Path(path).read_text(), model, allow_unknown_field=True)
    except (text_format.ParseError, UnicodeError) as exc:
        raise ValueError('invalid GE graph dump %s: %s' % (path, exc)) from exc
    return {'graph': [{'name': graph.name, 'op': [
        {'name': op.name, 'type': op.type, 'input': list(op.input)} for op in graph.op
    ]} for graph in model.graph]}


def _snapshots(directory):
    snapshots, errors = [], []
    for path in sorted(Path(directory).rglob('ge_proto*.txt')):
        sequence = re.match(r'ge_proto_(\d+)_', path.name)
        if sequence is None:
            continue
        try:
            document = load_ge_proto_snapshot(path)
            for graph in document['graph']:
                snapshots.append((path, int(sequence[1]), graph))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return snapshots, errors


def check_graph_dumps(directory, input_names):
    """Require an Extend pair followed by one Fused node in the same graph/process."""
    snapshots, errors = _snapshots(directory)
    first_round, second_round = [], []
    for path, sequence, graph in snapshots:
        for checker, target in ((check_stage1_graph, first_round), (check_compiled_graph, second_round)):
            try:
                report = checker({'graph': [graph]}, input_names)
                target.append((path, sequence, graph, report))
            except ValueError:
                pass
    for first in first_round:
        for second in second_round:
            if (first[0].parent == second[0].parent and first[2]['name'] == second[2]['name']
                    and first[1] < second[1]):
                report = dict(first[3], evidence_mode='single_atc_graph_dump', compile_count=1)
                for stage, snapshot in (('stage1', first), ('stage2', second)):
                    report[stage + '_snapshot_file'] = str(snapshot[0].relative_to(directory))
                    report[stage + '_snapshot_sha256'] = hashlib.sha256(snapshot[0].read_bytes()).hexdigest()
                return report
    detail = '; '.join(errors[:2]) if errors else 'no ordered ExtendConv2D pair -> FusedMultiConv2d snapshots in one graph/process'
    raise ValueError('single-compilation fusion evidence incomplete: ' + detail)


def diagnose_compilation(directory, input_names, tiling_directory, log):
    """Describe the last observed compilation stage without recompiling a diagnostic OM."""
    snapshots, errors = _snapshots(directory)
    types = {op['type'] for _, _, graph in snapshots for op in graph['op']}
    fused_graph = False
    for _, _, graph in snapshots:
        try:
            check_compiled_graph({'graph': [graph]}, input_names)
            fused_graph = True
        except ValueError:
            pass
    if fused_graph:
        captured = False
        for path in Path(tiling_directory).glob('tiling.*.jsonl'):
            for line in path.read_text(errors='replace').splitlines():
                try:
                    record = json.loads(line)
                    result = record.get('result')
                    captured |= (record.get('event') == 'tiling' and record.get('optype') == FUSED_TYPE
                                 and isinstance(result, dict) and bool(result.get('tiling_data')))
                except (ValueError, AttributeError):
                    continue
        status = 'fused_kernel_compile_failed' if captured else 'fused_before_native_tiling'
    elif 'ExtendConv2D' in types:
        status = 'stage2_not_fused'
    elif types & CONV_TYPES:
        status = 'stage1_not_fused'
    else:
        status = 'evidence_incomplete'
    text = Path(log).read_text(errors='replace') if Path(log).is_file() else ''
    return {'status': status, 'observed_operator_types': sorted(types), 'dump_errors': errors,
            'set_m1_dequant_reported': 'SetM1Dequant' in text}
