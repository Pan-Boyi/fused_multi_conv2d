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
                stage1_graph_sha256=digest, model_graph_sha256=digest,
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
