"""Public attribute shape, unsupported geometry, and unified calibration checks."""
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fc2d_onnx
import run_fused_multi_conv2d as runner
from fusion_test_support import write_test_binding


class PrototypeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')

    def test_public_defaults_and_internal_geometry_conversion(self):
        case = fc2d.normalise({'dtype':'int8'}, 0)
        self.assertEqual(case['strides'], [1,1,1,1])
        self.assertEqual(case['pads'], [0]*8)
        self.assertEqual(case['fixed_shift_value'], [0,0])
        case.update(strides=[2,2,3,3], pads=[0,0,1,1,2,2,0,0])
        argv=fc2d.gen_case_argv(case, 'unused.bin')
        self.assertEqual([argv[argv.index(k)+1] for k in ('--s1','--s2','--ph1','--pw1','--ph2','--pw2')],
                         ['2','3','0','1','2','0'])

    def test_unsupported_or_legacy_geometry_and_shifts_are_rejected(self):
        invalid = [dict(strides=[1,2]), dict(strides=[1,2,2,2]), dict(strides=[1,1,2,3]),
                   dict(strides=[0,0,1,1]), dict(pads=[0]*4), dict(pads=[0,1,0,0,0,0,0,0]),
                   dict(pads=[0,0,0,0,0,0,1,0]),dict(pads=[-1]*8),
                   dict(enable_relu=[0,2]),dict(fixed_shift_value=[-1,0]),
                   dict(fixed_shift_value=[0,59]),dict(dtype='a16w8',fixed_shift_value=[30,0]),
                   dict(relu=[1,1]),dict(fixed_shift=[42,42])]
        for values in invalid:
            with self.subTest(values=values), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit): fc2d.normalise(values,0)

    def test_a16w8_unified_shift_has_exact_nonzero_inputs_at_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            for shift in (0,5,27,29):
                path=Path(temp)/('shift%d.bin'%shift)
                case=fc2d.normalise(dict(dtype='a16w8',hi=8,wi=9,cout2=2,
                                         fixed_shift_value=[shift,0]),0)
                subprocess.run(fc2d.gen_case_argv(case,str(path)),check=True,stdout=subprocess.DEVNULL)
                spec=fc2d.read_spec(str(path)); tensors,*_=runner.load_case(str(path))
                plan=fc2d_onnx.tensor_plan(spec)
                self.assertEqual(plan['attrs']['fixed_shift_value'],[shift,0])
                self.assertNotIn('a16w8_shift',plan['attrs'])
                values=struct.unpack('<%de'%(len(tensors['x'][2])//2),tensors['x'][2])
                integers=[math.ldexp(v,29-shift) for v in values]
                self.assertTrue(any(integers))
                self.assertTrue(all(v==int(v) and -128<=v<=127 for v in integers))
                self.assertTrue(any(tensors['y_expect'][2]))

    def test_mapped_enum_defaults_reach_onnx_as_two_integer_values(self):
        with tempfile.TemporaryDirectory() as temp:
            case=fc2d.normalise(dict(dtype='int8',hi=8,wi=9,cout2=2),0)
            path=Path(temp)/'case.bin'
            subprocess.run(fc2d.gen_case_argv(case,str(path)),check=True,stdout=subprocess.DEVNULL)
            spec=fc2d.read_spec(str(path));tensors,*_=runner.load_case(str(path))
            plan=fc2d_onnx.tensor_plan(spec); model=fc2d_onnx.build_model(plan,tensors)
            for name in fc2d.MAPPED_ENUM_VALUES:
                self.assertEqual(case[name],[0,0]);self.assertEqual(plan['attrs'][name],[0,0])
                # Mapped enums belong to the target IR. Source nodes keep the
                # standard ONNX Conv geometry and are never directly fused targets.
                self.assertEqual(sum(n.op_type == 'Conv' for n in model.graph.node),2)
                self.assertNotIn('FusedMultiConv2d',[n.op_type for n in model.graph.node])

    def test_mapped_enum_lengths_types_and_unsupported_modes_are_rejected(self):
        for name, mapping in fc2d.MAPPED_ENUM_VALUES.items():
            for values in ([],[0],[0,0,0],0,mapping[0],[False,0],[0.0,0]):
                with self.subTest(name=name,values=values),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):fc2d.normalise({name:values},0)
            for value in range(-1,len(mapping)+1):
                if value==0:continue
                for layer in range(2):
                    values=[0,0];values[layer]=value
                    with self.subTest(name=name,values=values),contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit):fc2d.normalise({name:values},0)
        self.assertEqual(fc2d.MAPPED_ENUM_VALUES['pad_mode'][5],'NOTSET')

    def test_schema1_model_bindings_are_rejected_after_enum_contract_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'case.bin';model=Path(temp)/'case.om'
            case=fc2d.normalise(dict(dtype='int8',hi=8,wi=9,cout2=2),0)
            subprocess.run(fc2d.gen_case_argv(case,str(path)),check=True,stdout=subprocess.DEVNULL)
            model.write_bytes(b'synthetic model')
            binding_path=write_test_binding(str(path),str(model))
            binding=json.loads(Path(binding_path).read_text())
            self.assertEqual(binding['schema_version'],4)
            binding['schema_version']=1
            Path(str(model)+'.fc2d.json').write_text(json.dumps(binding))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
                runner.validate_model_binding(str(path),str(model))

    def test_abi7_is_rejected_by_readers_and_model_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'case.bin';case=fc2d.normalise(dict(dtype='int8',hi=8,wi=9,cout2=2),0)
            subprocess.run(fc2d.gen_case_argv(case,str(path)),check=True,stdout=subprocess.DEVNULL)
            raw=bytearray(path.read_bytes()); struct.pack_into('<I',raw,8,7);path.write_bytes(raw)
            model=Path(temp)/'case.om';model.write_bytes(b'synthetic test model')
            for call in (lambda:fc2d.read_spec(str(path)),lambda:runner.load_case(str(path)),
                         lambda:write_test_binding(str(path),str(model))):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit): call()

if __name__=='__main__': unittest.main()
