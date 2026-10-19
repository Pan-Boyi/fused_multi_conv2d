"""Verify both installed CANN fusion stages; this module implements no fusion."""
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
