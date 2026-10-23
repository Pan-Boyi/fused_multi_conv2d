"""Synthetic graph/tiling evidence for transport tests; never a production result."""
import hashlib
import json
import os
from pathlib import Path
import struct
import run_fused_multi_conv2d as runner
from fc2d_fusion import PASS_NAME, FIRST_PASS_NAME


def synthetic_proof(info):
    digest = hashlib.sha256(b'UNIT TEST ONLY: synthetic graph').hexdigest()
    return dict(pass_name=PASS_NAME, stage1_pass_name=FIRST_PASS_NAME, source_conv_count=2, stage1_extend_count=2,
                fused_node_count=1, unfused_convolution_count=0,
                input_names=runner.model_input_slots(info), source_graph_sha256=digest,
                evidence_mode='single_atc_graph_dump', compile_count=1,
                stage1_snapshot_sha256=digest, stage2_snapshot_sha256=digest, model_graph_sha256=digest,
                native_tiling=dict(tiling_data_bytes=312, tiling_data_sha256=digest))


def write_test_binding(case, model, **kwargs):
    return runner.write_model_binding(str(case), str(model),
        fusion_report=synthetic_proof(runner._binding_spec(str(case))), **kwargs)


def fake_graph(names, stage1=False):
    ops = [dict(name=n, type='Data') for n in names]
    if stage1:
        ops += [dict(name='extend1', type='ExtendConv2D', input=['x:0']),
                dict(name='extend2', type='ExtendConv2D', input=['extend1:0'])]
    else:
        ops += [dict(name='fused', type='FusedMultiConv2d')]
    return dict(graph=[dict(op=ops)])


def graph_dump_text(names, stage1=False, graph_name='pair'):
    """Build synthetic ModelDef text with the nesting of a GE graph dump."""
    nodes = fake_graph(names, stage1)['graph'][0]['op']
    lines = ['name: "synthetic_model"', 'version: 1', 'graph {', '  name: ' + json.dumps(graph_name)]
    for node in nodes:
        lines += ['  op {', '    name: ' + json.dumps(node['name']), '    type: ' + json.dumps(node['type'])]
        lines += ['    input: ' + json.dumps(value) for value in node.get('input', [])]
        lines += ['    output_desc {', '      name: "y"', '      dtype: DT_FLOAT16',
                  '      shape { dim: 1 dim: 32 dim: 8 dim: 9 }', '      layout: "NCHW"',
                  '      attr { key: "origin_shape" value { list { i: 1 i: 32 i: 8 i: 9 } } }',
                  '    }', '    attr { key: "_dump_note" value { s: "type: \\"Conv2D\\" { ignored }" } }', '  }']
    lines += ['}']
    return '\n'.join(lines) + '\n'


def emit_graph_dumps(names, failure=None):
    """Emit two synthetic pass snapshots inside this compilation's dump root."""
    directory = Path(os.environ['DUMP_GRAPH_PATH']) / '12345_20261023'
    directory.mkdir(parents=True, exist_ok=True)
    first = directory / 'ge_proto_00001_pair_stage1.txt'
    second = directory / 'ge_proto_00002_pair_stage2.txt'
    if failure != 'missing_stage1':
        first.write_text(graph_dump_text(names, True))
    if failure != 'missing_stage2':
        second.write_text(graph_dump_text(names))
    if failure == 'stage1_graph':
        first.write_text(first.read_text().replace('type: "ExtendConv2D"', 'type: "Conv2D"', 1))
    elif failure == 'stage2_graph':
        second.write_text(graph_dump_text(names, True))
    elif failure == 'stage2_residual':
        text = second.read_text().rsplit('}', 1)[0]
        second.write_text(text + '  op { name: "leftover" type: "ExtendConv2D" }\n}\n')
    elif failure == 'reversed_snapshots':
        first.write_text(graph_dump_text(names))
        second.write_text(graph_dump_text(names, True))
    elif failure == 'different_graph':
        second.write_text(graph_dump_text(names, graph_name='other_pair'))
    elif failure == 'different_process':
        other = directory.parent / '54321_20261023'
        other.mkdir()
        second.rename(other / second.name)
    elif failure == 'malformed_dump':
        first.write_text('graph { op { name: "broken"')
    return first, second


def emit_tiling(spec):
    words = [0]*78
    keys = ('n','ci','hi','wi','cout1','cout2','kh','kw','kh2','kw2','s1','s2','ph1','pw1','ph2','pw2')
    words[:16] = [spec[k] for k in keys]
    words[-7:] = [1,spec['shift1'],spec['shift2'],spec['bias'],spec['bias'],spec['relu1'],spec['relu2']]
    names = ('bias_mask','dtype','strides','pads','dilations','groups','data_format','offset_x','round_mode','pad_mode','enable_hf32','enable_relu','ascendc_op_para_size','fixed_shift_value')
    attrs = [dict(name=n) for n in names]
    for attr in attrs:
        if attr['name'] in ('data_format','round_mode','pad_mode'):
            attr.update(dtype='list_int',value=[0,0])
    record = dict(event='tiling',optype='FusedMultiConv2d',attrs=attrs,
                  result=dict(block_dim=1,tiling_data=struct.pack('<78I',*words).hex()))
    (Path(os.environ['FC2D_ATC_TILING_CAPTURE'])/'tiling.synthetic.jsonl').write_text(json.dumps(record)+'\n')
