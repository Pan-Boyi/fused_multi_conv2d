# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Check that Python-computed references reach the standalone device runner."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fused_multi_conv2d_golden as golden
import run_fused_multi_conv2d as runner
import run_profile as profile
from fusion_test_support import write_test_binding
from test_profile_workflow import LocalTransport


class PythonGoldenWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')
        cls.shared_temp = tempfile.TemporaryDirectory()
        cls.shared = Path(cls.shared_temp.name)
        cls.stub = cls.shared / 'stub'
        cls.stub.mkdir()
        subprocess.run(['gcc', '-shared', '-fPIC', '-O1',
                        str(ROOT / 'tests/stub_ascendcl/stub.c'),
                        '-o', str(cls.stub / 'libascendcl.so')], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.shared_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.case = self.work / 'case.bin'
        self.sidecar = Path(str(self.case) + '.golden.json')
        self.config = fc2d.normalise(dict(name='python_reference', dtype='int8', n=1, ci=32,
                                          cout1=32, cout2=2, hi=4, wi=5, kernel=[1, 1],
                                          strides=[1, 1, 1, 1], pads=[0] * 8,
                                          enable_relu=[False, False], bias=False), 0)
        self.python_output = bytes([7] * 40)
        self.python_result = {'y_expect': (2, [1, 2, 4, 5], self.python_output),
                              'stats': {'nonzero': 40, 'sat': 0}}

    def raw_cpp_case(self):
        subprocess.run(fc2d.gen_case_argv(self.config, str(self.case)), check=True,
                       stdout=subprocess.DEVNULL)

    def create_mock_python_case(self):
        with mock.patch.object(golden, 'compute_golden', return_value=self.python_result), \
                contextlib.redirect_stdout(io.StringIO()):
            fc2d.do_case(self.config, {'bin': str(self.case)})
        return json.loads(self.sidecar.read_text())

    def test_recompute_repairs_corrupt_cpp_reference_and_preserves_inputs(self):
        self.raw_cpp_case()
        original_tensors, *_, info = runner.load_case(str(self.case))
        blob = bytearray(self.case.read_bytes())
        offset = runner.HDR_LEN
        for _ in range(struct.unpack_from('<I', blob, 12)[0]):
            record = struct.unpack_from(runner.REC_FMT, blob, offset)
            offset += runner.REC_LEN
            if record[0].rstrip(b'\0') == b'y_expect':
                blob[offset:offset + record[-1]] = bytes([42]) * record[-1]
            offset += record[-1]
        struct.pack_into('<QQ', blob, 16, 1, 999)
        self.case.write_bytes(blob)
        report = fc2d.write_python_golden(str(self.case))
        repaired, nonzero, _, sat, *_ = runner.load_case(str(self.case))
        self.assertEqual(repaired['y_expect'], original_tensors['y_expect'])
        for name in original_tensors:
            if name != 'y_expect':
                self.assertEqual(repaired[name], original_tensors[name], name)
        expected = golden.compute_golden(info, original_tensors)
        self.assertEqual((nonzero, sat), (expected['stats']['nonzero'], expected['stats']['sat']))
        self.assertEqual(report['generator'], 'python')
        self.assertFalse(report['outputs']['y_expect']['matches_cpp'])
        self.assertEqual(report['case_sha256'], hashlib.sha256(self.case.read_bytes()).hexdigest())

    def test_case_generation_uses_python_result_and_binds_its_provenance(self):
        report = self.create_mock_python_case()
        tensors, nonzero, _, sat, *_ = runner.load_case(str(self.case))
        self.assertEqual(tensors['y_expect'][2], self.python_output)
        self.assertEqual((nonzero, sat), (40, 0))
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['generator'], 'python')
        self.assertFalse(report['outputs']['y_expect']['matches_cpp'])
        self.assertEqual(report['outputs']['y_expect']['sha256'], hashlib.sha256(self.python_output).hexdigest())
        self.assertEqual(report['implementation_sha256'], hashlib.sha256(Path(golden.__file__).read_bytes()).hexdigest())
        self.assertEqual(report['case_sha256'], hashlib.sha256(self.case.read_bytes()).hexdigest())
        self.assertEqual(list(self.work.glob('.case-python-*')), [])

    def test_python_failure_preserves_existing_case_and_report(self):
        self.create_mock_python_case()
        original_case, original_report = self.case.read_bytes(), self.sidecar.read_bytes()
        with mock.patch.object(golden, 'compute_golden', side_effect=ValueError('synthetic golden failure')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'synthetic golden failure'):
            fc2d.do_case(self.config, {'bin': str(self.case)})
        self.assertEqual(self.case.read_bytes(), original_case)
        self.assertEqual(self.sidecar.read_bytes(), original_report)
        self.assertEqual(list(self.work.glob('.case-python-*')), [])

    def test_uploaded_python_reference_is_compared_without_numpy_on_device(self):
        self.create_mock_python_case()
        model = self.work / 'python_reference.om'
        model.write_bytes(b'synthetic model for ACL runner workflow')
        write_test_binding(str(self.case), str(model))
        conf = self.work / 'stub.conf'
        tensors, *_, info = runner.load_case(str(self.case))
        conf.write_text(''.join('%s %d\n' % (name, len(tensors[name][2]))
                                for name in runner.model_input_slots(info)) + 'OUT 40\n')
        device_output = self.work / 'device_output.bin'
        device_output.write_bytes(self.python_output)
        env = dict(os.environ, FC2D_STUB_CONF=str(conf), FC2D_STUB_GOLDEN=str(device_output),
                   LD_LIBRARY_PATH=str(self.stub), DYLD_LIBRARY_PATH=str(self.stub), REPEAT='1', WARMUP='0')
        remote = LocalTransport(self.work / 'remote', env)
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(remote, self.config, {'bin': str(self.case)}, str(model))
        device_dir = Path(remote.dir) / self.config['name']
        self.assertEqual({Path(source).name for source, _ in remote.uploads},
                         {'case.bin', model.name, model.name + '.fc2d.json', 'run_fused_multi_conv2d.py'})
        self.assertEqual((device_dir / 'case.bin').read_bytes(), self.case.read_bytes())
        bootstrap = (
            "import sys,runpy\n"
            "class RejectMathImports:\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname.split('.')[0] in ('numpy','fused_multi_conv2d_golden'):\n"
            "   raise ImportError('device runner must not import '+fullname)\n"
            "sys.meta_path.insert(0,RejectMathImports())\n"
            "sys.argv=sys.argv[1:]\n"
            "runpy.run_path(sys.argv[0],run_name='__main__')\n")
        command = [sys.executable, '-S', '-c', bootstrap,
                   str(device_dir / 'run_fused_multi_conv2d.py'), str(device_dir / 'case.bin'),
                   'FusedMultiConv2d', '0', str(device_dir / model.name)]
        script = 'DYLD_LIBRARY_PATH=%s %s' % (shlex.quote(str(self.stub)),
                                             ' '.join(shlex.quote(argument) for argument in command))
        rc, output = remote.ssh(script)
        self.assertEqual(rc, 0, output)
        self.assertIn('Precision: 100.000000%', output)
        self.assertIn('[PASS]', output)
        device_output.write_bytes(bytes([8] * 40))
        rc, output = remote.ssh(script)
        self.assertNotEqual(rc, 0, output)
        self.assertIn('Precision: 0.000000%', output)
        self.assertIn('[FAIL]', output)


if __name__ == '__main__':
    unittest.main()
