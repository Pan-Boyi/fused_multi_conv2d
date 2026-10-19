"""Two real source convolutions, stage gates, and failure invalidation (no CANN)."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fc2d_onnx as frontend
import fc2d_fusion as fusion
import run_fused_multi_conv2d as runner
from fusion_test_support import fake_graph, emit_tiling, synthetic_proof, write_test_binding


class FusionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def case(self, mode='int8', bias=False, relu=(0, 1)):
        case = fc2d.normalise(dict(name='pair', dtype=mode, hi=8, wi=9, cout2=2,
            kernel=[1,3,3,1], strides=[1,1,2,2], pads=[0,0,1,1,1,1,0,0],
            bias=bias, enable_relu=list(relu), fixed_shift_value=[29 if mode=='a16w8' else 42,42]),0)
        path=self.work/(mode+str(bias)+str(relu[0])+'.bin')
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.do_case(case,dict(bin=str(path)))
        return path

    def test_source_graph_contract_for_all_modes_and_optional_inputs(self):
        for mode in ('fp16','int8','s8f16','a16w8'):
            for bias in (False, True):
                for relu in ((0,0),(0,1),(1,0),(1,1)):
                    with self.subTest(mode=mode,bias=bias,relu=relu):
                        path=self.case(mode,bias,relu)
                        spec=fc2d.read_spec(str(path)); tensors,*_=runner.load_case(str(path))
                        plan=frontend.tensor_plan(spec)
                        model=frontend.build_model(plan,tensors)
                        nodes=list(model.graph.node)
                        self.assertEqual(fusion.check_input_model(model),dict(source_conv_count=2))
                        self.assertEqual([v.name for v in model.graph.input],runner.model_input_slots(spec))
                        self.assertEqual(len(model.graph.input),1+2*bias+2*(mode!='fp16'))
                        self.assertEqual([n.name for n in nodes if n.op_type=='Relu'],
                                         ['relu%d'%(i+1) for i,v in enumerate(relu) if v])
                        convs=[n for n in nodes if n.op_type=='Conv']
                        for conv,kernel,pads in zip(convs,([1,3],[3,1]),([0,1,0,1],[1,0,1,0])):
                            attrs={a.name:helper.get_attribute_value(a) for a in conv.attribute}
                            self.assertEqual(attrs['kernel_shape'],kernel)
                            self.assertEqual(attrs['pads'],pads)
                            self.assertNotIn('fixed_shift_value',attrs)
                        quant=[n for n in nodes if n.op_type in ('AscendRequant','AscendDequant')]
                        self.assertEqual([n.op_type for n in quant], [] if mode=='fp16' else
                            ['AscendRequant', 'AscendRequant' if mode=='int8' else 'AscendDequant'])
                        self.assertEqual(convs[1].input[0],'mid')
                        if mode=='fp16':
                            onnx.checker.check_model(model)

    def test_quantized_source_graph_computation_matches_unchanged_integer_golden(self):
        # Interpret the generated graph independently of tensor_plan and the C++
        # oracle. A16W8 S=29 makes its fp16 inputs exact integers.
        for mode in ('int8','s8f16','a16w8'):
            with self.subTest(mode=mode):
                path=self.case(mode,True)
                spec=fc2d.read_spec(str(path)); tensors,*_=runner.load_case(str(path))
                model=frontend.build_model(frontend.tensor_plan(spec),tensors)
                dtype={1:'<f2',2:'i1',3:'<i4',10:'<u8'}
                env={n:np.frombuffer(raw,dtype=dtype[dt]).reshape(shape)
                     for n,(dt,shape,raw) in tensors.items() if n in runner.model_input_slots(spec)}
                env.update({t.name:numpy_helper.to_array(t) for t in model.graph.initializer})
                for node in model.graph.node:
                    attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
                    args=[env[n] for n in node.input]
                    if node.op_type=='Conv':
                        x,w=args[:2]; kh,kw=w.shape[2:]; sh,sw=attrs['strides']; t,l,b,r=attrs['pads']
                        padded=np.pad(x.astype(np.int64),((0,0),(0,0),(t,b),(l,r)))
                        ho=(padded.shape[2]-kh)//sh+1; wo=(padded.shape[3]-kw)//sw+1
                        y=np.zeros((x.shape[0],w.shape[0],ho,wo),dtype=np.int64)
                        for row in range(ho):
                            for col in range(wo):
                                y[:,:,row,col]=np.einsum('nchw,ochw->no',
                                    padded[:,:,row*sh:row*sh+kh,col*sw:col*sw+kw],w.astype(np.int64))
                        if len(args)==3:y+=args[2][None,:,None,None]
                    elif node.op_type in ('AscendRequant','AscendDequant'):
                        scale=(args[1]&np.uint64(0xffffffff)).astype('<u4').view('<f4')
                        y=args[0].astype(np.float64)*scale[None,:,None,None].astype(np.float64)
                        if node.op_type=='AscendRequant':y=np.clip(np.rint(y),-128,127).astype('i1')
                        else:y=y.astype('<f2')
                    elif node.op_type=='Relu':y=np.maximum(args[0],0)
                    else:self.fail('unexpected source operator '+node.op_type)
                    env[node.output[0]]=y
                expected=np.frombuffer(tensors['y_expect'][2],dtype='i1' if mode=='int8' else '<f2').reshape(env['y'].shape)
                # Standalone ONNX Relu uses +0; the unchanged Fixpipe oracle
                # preserves -0 for negative fp16 values. Compare numeric zero
                # equality here and retain exact nonzero bit patterns.
                np.testing.assert_array_equal(env['y'],expected)
                self.assertEqual(env['y'][env['y']!=0].tobytes(),expected[expected!=0].tobytes())

    def test_source_rejects_bypassed_fusion_and_nonconsecutive_pair(self):
        path=self.case(); tensors,*_=runner.load_case(str(path))
        source=frontend.build_model(frontend.tensor_plan(fc2d.read_spec(str(path))),tensors)
        for target in ('FusedMultiConv2d','ExtendConv2D'):
            model=copy.deepcopy(source);model.graph.node[0].op_type=target
            with self.assertRaisesRegex(ValueError,'bypassed'):fusion.check_input_model(model)
        model=copy.deepcopy(source)
        next(n for n in model.graph.node if n.name=='conv2').input[0]='x'
        with self.assertRaisesRegex(ValueError,'consecutive'):fusion.check_input_model(model)
        model=copy.deepcopy(source)
        model.graph.output.append(helper.make_tensor_value_info('mid',onnx.TensorProto.INT8,[1,64,8,9]))
        with self.assertRaisesRegex(ValueError,'additional consumers'):fusion.check_input_model(model)

    def test_stage1_checks_count_topology_and_inputs(self):
        valid=fake_graph(['x'],True)
        self.assertEqual(fusion.check_stage1_graph(valid,['x'])['stage1_extend_count'],2)
        for edit in ('count','edge','type','inputs'):
            document=copy.deepcopy(valid);ops=document['graph'][0]['op']
            if edit=='count':ops.pop()
            elif edit=='edge':ops[-1]['input']=['x:0']
            elif edit=='type':ops[-1]['type']='Conv2D'
            else:ops[0]['name']='wrong_input'
            with self.subTest(edit=edit),self.assertRaises(ValueError):fusion.check_stage1_graph(document,['x'])

    def test_stage2_requires_exactly_one_fused_node_and_no_independent_convolution(self):
        valid=fake_graph(['x'])
        self.assertEqual(fusion.check_compiled_graph(valid,['x'])['fused_node_count'],1)
        for op in ('Conv2D','ExtendConv2D','FusedMultiConv2d','Relu','AscendRequant','AscendDequant'):
            document=copy.deepcopy(valid);document['graph'][0]['op'].append(dict(name='extra',type=op))
            with self.subTest(op=op),self.assertRaises(ValueError):fusion.check_compiled_graph(document,['x'])
        with self.assertRaises(ValueError):fusion.check_compiled_graph(fake_graph(['unexpected']),['x'])

    def test_no_proof_or_incomplete_proof_cannot_create_binding(self):
        case=self.case();model=self.work/'model.om';model.write_bytes(b'synthetic')
        info=runner._binding_spec(str(case))
        proof=synthetic_proof(info)
        for bad in (None, {}, dict(proof,source_conv_count=0),dict(proof,stage1_extend_count=0),
                    dict(proof,native_tiling={}),dict(proof,stage1_graph_sha256='invalid')):
            with self.subTest(proof=bad),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
                runner.write_model_binding(str(case),str(model),fusion_report=bad)
            self.assertFalse(Path(str(model)+'.fc2d.json').exists())

    def test_failed_compilation_or_fusion_invalidates_previous_binding(self):
        for failure in ('stage1_compile','stage1_graph','stage2_compile','stage2_graph','missing_tiling','wrong_shift','source_changed'):
            with self.subTest(failure=failure):
                case=self.case('fp16',True,(1,1))
                paths=dict(bin=str(case),onnx=str(self.work/'source.onnx'),omdir=str(self.work/'om'))
                omdir=Path(paths['omdir']);omdir.mkdir(exist_ok=True)
                old=omdir/'pair.om';old.write_bytes(b'old synthetic model')
                write_test_binding(case,old)
                spec=fc2d.read_spec(str(case));names=runner.model_input_slots(spec)
                def atc(command,**kwargs):
                    if '--mode=1' in command:
                        model=next(a.split('=',1)[1] for a in command if a.startswith('--om='))
                        target=next(a.split('=',1)[1] for a in command if a.startswith('--json='))
                        first=Path(model).stem=='stage1'
                        graph=fake_graph(names,first)
                        if (first and failure=='stage1_graph') or (not first and failure=='stage2_graph'):
                            graph['graph'][0]['op'][-1]['type']='Conv2D'
                        Path(target).write_text(json.dumps(graph));return 0
                    prefix=next(a.split('=',1)[1] for a in command if a.startswith('--output='))
                    first=Path(prefix).name=='stage1'
                    if (first and failure=='stage1_compile') or (not first and failure=='stage2_compile'):return 9
                    Path(prefix+'.om').write_bytes(b'synthetic ATC only')
                    if first and failure=='source_changed':
                        Path(paths['onnx']).write_bytes(b'source changed during ATC')
                    if not first and failure!='missing_tiling':
                        actual=dict(spec)
                        if failure=='wrong_shift':actual['shift1']=0
                        emit_tiling(actual)
                    return 0
                with mock.patch.object(fc2d.shutil,'which',return_value='/synthetic/atc'), \
                        mock.patch.object(fc2d,'run',side_effect=atc),contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    fc2d.do_om(dict(name='pair'),paths,'MC62CM12AA')
                self.assertFalse(Path(str(old)+'.fc2d.json').exists())

    def test_atc_log_preserves_failure_output(self):
        log=self.work/'atc.log'
        with contextlib.redirect_stdout(io.StringIO()):
            rc=fc2d.run([sys.executable,'-c',"print('compiler diagnostic');raise SystemExit(17)"],log_path=str(log))
        self.assertEqual(rc,17)
        self.assertIn('compiler diagnostic',log.read_text())


if __name__=='__main__':unittest.main()
