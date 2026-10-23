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
from fusion_test_support import (fake_graph, emit_tiling, emit_graph_dumps, graph_dump_text,
                                 synthetic_proof, write_test_binding)


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
            bias=bias, enable_relu=list(relu), fixed_shift_value=[13 if mode=='a16w8' else 42,42]),0)
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

    def test_cann_fixedpoint_source_reference_matches_unchanged_integer_golden(self):
        # CANN's mixed Conv2D uses fixed-point input conversion. This reference
        # follows that contract rather than standard ONNX float Conv semantics.
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
                        if mode=='a16w8' and node.name=='conv1':
                            x=np.rint(np.ldexp(x.astype(np.float64),29-spec['shift1'])).astype(np.int64)
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
                    dict(proof,native_tiling={}),dict(proof,stage1_snapshot_sha256='invalid'),
                    dict(proof,stage2_snapshot_sha256='invalid'),dict(proof,compile_count=2),
                    dict(proof,evidence_mode='two_atc_compilations')):
            with self.subTest(proof=bad),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
                runner.write_model_binding(str(case),str(model),fusion_report=bad)
            self.assertFalse(Path(str(model)+'.fc2d.json').exists())

    def test_same_compilation_dump_sequence_preserves_nesting_and_stage_order(self):
        dump_root=self.work/'dumps'
        with mock.patch.dict(os.environ, {'DUMP_GRAPH_PATH':str(dump_root)}):
            first,second=emit_graph_dumps(['x','bias1','bias2'])
        proof=fusion.check_graph_dumps(dump_root,['x','bias1','bias2'])
        self.assertEqual(proof['stage1_extend_count'],2)
        self.assertEqual(proof['stage1_nodes'],['extend1','extend2'])
        self.assertEqual(proof['evidence_mode'],'single_atc_graph_dump')
        self.assertEqual(proof['compile_count'],1)
        self.assertEqual(proof['stage1_snapshot_sha256'],runner._file_sha256(first))
        self.assertEqual(proof['stage2_snapshot_sha256'],runner._file_sha256(second))

    def test_dump_evidence_rejects_missing_unordered_or_unrelated_snapshots(self):
        for failure in ('missing_stage1','missing_stage2','reversed_snapshots',
                        'different_graph','different_process','stage1_graph','stage2_graph','stage2_residual','malformed_dump'):
            with self.subTest(failure=failure):
                dump_root=self.work/failure
                with mock.patch.dict(os.environ, {'DUMP_GRAPH_PATH':str(dump_root)}):
                    emit_graph_dumps(['x'],failure)
                with self.assertRaises(ValueError):
                    fusion.check_graph_dumps(dump_root,['x'])

    def compile_synthetic(self, failure=None):
        case=self.case('fp16',True,(1,1))
        paths=dict(bin=str(case),onnx=str(self.work/'source.onnx'),omdir=str(self.work/'om'))
        omdir=Path(paths['omdir']);omdir.mkdir(exist_ok=True)
        old=omdir/'pair.om';old.write_bytes(b'old synthetic model')
        write_test_binding(case,old)
        spec=fc2d.read_spec(str(case));names=runner.model_input_slots(spec)
        def atc(command,**kwargs):
            if '--mode=1' in command:
                if failure=='export_failure':return 9
                target=next(a.split('=',1)[1] for a in command if a.startswith('--json='))
                if failure=='missing_export':return 0
                graph=fake_graph(names)
                if failure=='final_graph':
                    graph['graph'][0]['op'].append(dict(name='unfused',type='ExtendConv2D'))
                Path(target).write_text(json.dumps(graph));return 0
            if failure=='compile_failure':return 9
            prefix=next(a.split('=',1)[1] for a in command if a.startswith('--output='))
            self.assertNotEqual(Path(prefix).name,'stage1')
            switch=next(a.split('=',1)[1] for a in command if a.startswith('--fusion_switch_file='))
            switches=json.loads(Path(switch).read_text())['Switch']['GraphFusion']
            self.assertEqual(switches[fusion.FIRST_PASS_NAME],'on')
            self.assertEqual(switches[fusion.PASS_NAME],'on')
            self.assertEqual(os.environ['DUMP_GE_GRAPH'],'2')
            if failure=='set_m1_dequant':
                emit_graph_dumps(names,'missing_stage2')
                Path(kwargs['log_path']).write_text('compiler error: SetM1Dequant is unavailable\n')
                return 9
            Path(prefix+'.om').write_bytes(b'synthetic ATC only')
            emit_graph_dumps(names,failure)
            if failure=='source_changed':Path(paths['onnx']).write_bytes(b'source changed during ATC')
            if failure!='missing_tiling':
                actual=dict(spec)
                if failure=='wrong_shift':actual['shift1']=0
                emit_tiling(actual)
            return 0
        with mock.patch.object(fc2d.shutil,'which',return_value='/synthetic/atc'), \
                mock.patch.object(fc2d,'run',side_effect=atc) as calls,contextlib.redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaises(SystemExit):fc2d.do_om(dict(name='pair'),paths,'MC62CM12AA')
            else:fc2d.do_om(dict(name='pair'),paths,'MC62CM12AA')
        return case,old,calls.call_args_list

    def test_set_m1_dequant_failure_preserves_stage_diagnosis_without_binding(self):
        _,model,calls=self.compile_synthetic('set_m1_dequant')
        self.assertEqual(len(calls),1)
        self.assertFalse(Path(str(model)+'.fc2d.json').exists())
        reports=list(self.work.glob('fusion_*/failure_report.json'))
        self.assertEqual(len(reports),1)
        report=json.loads(reports[0].read_text())
        self.assertEqual(report['phase'],'model_compilation')
        self.assertEqual(report['status'],'stage2_not_fused')
        self.assertTrue(report['set_m1_dequant_reported'])
        self.assertIn('ATC returned 9',report['error'])

    def test_only_one_final_compile_proves_both_passes_and_binds_model(self):
        case,model,calls=self.compile_synthetic()
        compile_calls=[call for call in calls if '--mode=1' not in call.args[0]]
        self.assertEqual(len(compile_calls),1)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(list(self.work.rglob('stage1.om'))),0)
        binding=runner.validate_model_binding(str(case),str(model))
        self.assertEqual(binding['schema_version'],5)
        self.assertEqual(binding['fusion']['evidence_mode'],'single_atc_graph_dump')
        self.assertEqual(binding['fusion']['compile_count'],1)
        self.assertNotIn('stage1_graph_sha256',binding['fusion'])

    def test_failed_compilation_or_fusion_invalidates_previous_binding(self):
        for failure in ('compile_failure','export_failure','missing_export','missing_stage1','missing_stage2',
                        'stage1_graph','stage2_graph','stage2_residual','reversed_snapshots','different_graph','different_process',
                        'malformed_dump','final_graph','missing_tiling','wrong_shift','source_changed'):
            with self.subTest(failure=failure):
                _,model,calls=self.compile_synthetic(failure)
                self.assertFalse(Path(str(model)+'.fc2d.json').exists())
                self.assertEqual(sum('--mode=1' not in call.args[0] for call in calls),1)
                self.assertEqual(len(list(self.work.rglob('stage1.om'))),0)

    def test_atc_log_preserves_failure_output(self):
        log=self.work/'atc.log'
        with contextlib.redirect_stdout(io.StringIO()):
            rc=fc2d.run([sys.executable,'-c',"print('compiler diagnostic');raise SystemExit(17)"],log_path=str(log))
        self.assertEqual(rc,17)
        self.assertIn('compiler diagnostic',log.read_text())


class FusionDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work=Path(self.temp.name)

    def test_real_sdk_modeldef_dump_parses_data_attributes_and_tensor_descriptors(self):
        # Captured from a generated two-Conv probe on CANN 9.2. Data metadata is
        # retained; other nodes are reduced to their topology fields.
        snapshot=ROOT/'tests/fixtures/ge_graph_dump_probe.pbtxt'
        document=fusion.load_ge_proto_snapshot(snapshot)
        self.assertEqual(len(document['graph']),1)
        graph=document['graph'][0]
        self.assertEqual(graph['name'],'dump_probe')
        self.assertEqual([op['type'] for op in graph['op']],
                         ['Data','Const','Conv2D','Relu','Const','Conv2D','Relu','NetOutput'])
        self.assertEqual(graph['op'][0],dict(name='x',type='Data',input=['']))
        self.assertEqual(graph['op'][5]['input'],['relu1:0','filter2_Initializer_1:0','',''])

    def test_graph_dump_environment_restores_parent_values_on_success_and_exception(self):
        keys=('DUMP_GE_GRAPH','DUMP_GRAPH_LEVEL','DUMP_GRAPH_FORMAT','DUMP_GRAPH_PATH',
              'ASCEND_DUMP_PATH','NPU_COLLECT_PATH')
        for parent_exists in (False,True):
            for fail in (False,True):
                with self.subTest(parent_exists=parent_exists,fail=fail),mock.patch.dict(os.environ):
                    for key in keys:
                        if parent_exists:os.environ[key]='parent-'+key
                        else:os.environ.pop(key,None)
                    previous={key:os.environ.get(key) for key in keys}
                    try:
                        with fusion.graph_dump_environment(self.work/'graphs'):
                            self.assertEqual(os.environ['DUMP_GE_GRAPH'],'2')
                            self.assertEqual(os.environ['DUMP_GRAPH_LEVEL'],'1')
                            self.assertEqual(os.environ['DUMP_GRAPH_FORMAT'],'ge_proto')
                            self.assertEqual(os.environ['DUMP_GRAPH_PATH'],str((self.work/'graphs').resolve()))
                            self.assertNotIn('ASCEND_DUMP_PATH',os.environ)
                            self.assertNotIn('NPU_COLLECT_PATH',os.environ)
                            if fail:raise RuntimeError('synthetic ATC failure')
                    except RuntimeError:
                        if not fail:raise
                    self.assertEqual({key:os.environ.get(key) for key in keys},previous)

    def test_set_m1_dequant_diagnosis_distinguishes_observed_stages(self):
        for stage,capture_record,expected in (
                ('none',None,'evidence_incomplete'),
                ('source',None,'stage1_not_fused'),
                ('extend',None,'stage2_not_fused'),
                ('fused',None,'fused_before_native_tiling'),
                ('fused',{'event':'capture-unavailable','error':'adapter unavailable'},'fused_before_native_tiling'),
                ('fused',{'event':'tiling','optype':'FusedMultiConv2d','result':{
                    'block_dim':1,'tiling_data':'00'*312}},'fused_kernel_compile_failed')):
            with self.subTest(stage=stage,capture_record=capture_record):
                root=Path(tempfile.mkdtemp(dir=self.work))
                dumps=root/'graphs';dumps.mkdir()
                capture=root/'capture';capture.mkdir()
                if stage!='none':
                    text=graph_dump_text(['x'],stage=='extend')
                    if stage=='source':text=text.replace('type: "FusedMultiConv2d"','type: "Conv2D"')
                    (dumps/'ge_proto_00001_pair.txt').write_text(text)
                if capture_record is not None:
                    (capture/'tiling.synthetic.jsonl').write_text(json.dumps(capture_record)+'\n')
                log=root/'atc.log';log.write_text('error: SetM1Dequant is unavailable\n')
                report=fusion.diagnose_compilation(dumps,['x'],capture,log)
                self.assertEqual(report['status'],expected)
                self.assertTrue(report['set_m1_dequant_reported'])


if __name__=='__main__':unittest.main()
