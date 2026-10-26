"""Reject stale constant-weight models before ACL or remote transport."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_fused_multi_conv2d as runner
from fusion_test_support import write_test_binding
import run_profile
import fc2d
import fc2d_onnx


class ModelBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        cls.case = cls.work / 'case.bin'
        generator = cls.work / 'gen_case'
        subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off',
                        str(ROOT / 'gen_case.cpp'), '-o', str(generator)], check=True)
        subprocess.run([str(generator), str(cls.case), '--dtype', 'int8',
                        '--ci', '32', '--cout1', '64', '--cout2', '2', '--hi', '8', '--wi', '9',
                        '--kh', '1', '--kw', '3', '--kh2', '3', '--kw2', '1',
                        '--s1', '1', '--s2', '2', '--ph1', '0', '--pw1', '1',
                        '--ph2', '1', '--pw2', '0', '--bias', '1'], check=True,
                       stdout=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_model(self):
        om = self.work / (self._testMethodName + '.om')
        om.write_bytes(b'unit-test fake model; no device execution')
        sidecar = write_test_binding(str(self.case), str(om))
        return om, Path(sidecar)

    def assert_rejected(self, case, model):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.validate_model_binding(str(case), str(model))

    def test_binding_matches_full_case_and_model(self):
        model, sidecar = self.make_model()
        binding = runner.validate_model_binding(str(self.case), str(model))
        self.assertEqual(binding['model_input_count'], 3)
        self.assertEqual(binding['scale_mode'], 'constant')
        self.assertEqual(binding['weight_mode'], 'constant')
        self.assertEqual(binding, json.loads(sidecar.read_text()))

    def test_binding_accepts_known_tiling_layouts_only(self):
        model, sidecar = self.make_model()
        binding = json.loads(sidecar.read_text())
        for size in (312, 316):
            binding['fusion']['native_tiling']['tiling_data_bytes'] = size
            sidecar.write_text(json.dumps(binding))
            self.assertEqual(runner.validate_model_binding(str(self.case), str(model)), binding)
        for size in (308, 313, 315, 317, 320):
            binding['fusion']['native_tiling']['tiling_data_bytes'] = size
            sidecar.write_text(json.dumps(binding))
            with self.subTest(size=size):
                self.assert_rejected(self.case, model)

    def test_missing_binding_or_changed_model_is_rejected(self):
        model, sidecar = self.make_model()
        model.write_bytes(model.read_bytes() + b'changed')
        self.assert_rejected(self.case, model)
        sidecar.unlink()
        self.assert_rejected(self.case, model)

    def test_binding_rejects_previous_schema_and_wrong_operator(self):
        model, sidecar = self.make_model()
        binding = json.loads(sidecar.read_text())
        self.assertEqual(binding['op_type'], 'FusedMultiConv2d')
        for old_schema in (1, 2, 3, 4, 5):
            previous = dict(binding, schema_version=old_schema)
            sidecar.write_text(json.dumps(previous))
            self.assert_rejected(self.case, model)
        for scale_mode in (None, 'runtime'):
            sidecar.write_text(json.dumps(dict(binding, scale_mode=scale_mode)))
            self.assert_rejected(self.case, model)
        for op_type in (None, 'OtherOperator'):
            sidecar.write_text(json.dumps(dict(binding, op_type=op_type)))
            self.assert_rejected(self.case, model)

    def test_changed_case_or_unsupported_contract_is_rejected(self):
        model, sidecar = self.make_model()
        altered = self.work / 'changed_case.bin'
        altered.write_bytes(self.case.read_bytes() + b'changed')
        self.assert_rejected(altered, model)
        binding = json.loads(sidecar.read_text())
        binding['model_input_count'] = 9
        sidecar.write_text(json.dumps(binding))
        self.assert_rejected(self.case, model)

    def test_missing_binding_exits_before_loading_acl(self):
        model, sidecar = self.make_model()
        sidecar.unlink()
        env = dict(os.environ, LD_LIBRARY_PATH='/missing-test-library-directory')
        result = subprocess.run([sys.executable, str(ROOT / 'run_fused_multi_conv2d.py'),
                                 str(self.case), 'FusedMultiConv2d', '0', str(model)],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing or invalid case/OM binding', result.stdout)
        self.assertNotIn('加载 libascendcl.so 失败', result.stdout)

    def test_remote_upload_copies_binding_and_rejects_stale_case_before_mutation(self):
        model, sidecar = self.make_model()

        class RemoteStub:
            dir = '/tmp/fc2d-test-only'
            target = 'test-only'
            def __init__(self):
                self.commands, self.uploads = [], []
            def ssh(self, command):
                self.commands.append(command)
                return 0, ''
            def scp_to(self, source, target):
                self.uploads.append((source, target))
                return 0

        remote = RemoteStub()
        with contextlib.redirect_stdout(io.StringIO()):
            run_profile.do_push(remote, {'name': 'test-only'}, {'bin': str(self.case)}, str(model))
        self.assertTrue(any(source == str(sidecar) for source, _ in remote.uploads))
        altered = self.work / 'stale_upload.bin'
        altered.write_bytes(self.case.read_bytes() + b'changed')
        remote = RemoteStub()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            run_profile.do_push(remote, {'name': 'test-only'}, {'bin': str(altered)}, str(model))
        self.assertEqual(remote.commands, [])
        self.assertEqual(remote.uploads, [])

    def test_direct_graph_input_signature_is_rejected_by_acl_stub(self):
        model, _ = self.make_model()
        stub = self.work / 'acl_stub'
        stub.mkdir()
        subprocess.run(['gcc', '-shared', '-fPIC', '-O1',
                        str(ROOT / 'tests/stub_ascendcl/stub.c'),
                        '-o', str(stub / 'libascendcl.so')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'tests/stub_ascendcl/make_conf.py'),
                        str(self.case), str(stub)], check=True, stdout=subprocess.DEVNULL)
        conf = stub / 'stub.conf'
        runtime_conf=conf.read_text()
        env = dict(os.environ, LD_LIBRARY_PATH=str(stub), DYLD_LIBRARY_PATH=str(stub), FC2D_STUB_CONF=str(conf),
                   FC2D_STUB_GOLDEN=str(stub / 'golden.bin'), REPEAT='1', WARMUP='0')
        for extra in ('filter1 6144\nfilter2 384\n','quant_scale1 512\nquant_scale2 16\n'):
            with self.subTest(extra_inputs=extra):
                conf.write_text(runtime_conf+extra)
                result = subprocess.run([sys.executable, str(ROOT / 'run_fused_multi_conv2d.py'),
                                         str(self.case), 'FusedMultiConv2d', '0', str(model)],
                                        env=env, cwd=str(stub), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('model has 5 inputs', result.stdout)


    def test_case_changed_during_atc_is_not_bound_to_old_weights(self):
        case = self.work / "compile_race.bin"
        shutil.copyfile(self.case, case)
        paths = {"bin": str(case), "onnx": str(self.work / "compile_race.onnx"),
                 "omdir": str(self.work / "compile_race_om")}

        def mutate_during_atc(command, **kwargs):
            prefix = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
            Path(prefix + ".om").write_bytes(b"compiled old weights")
            case.write_bytes(case.read_bytes() + b"new case during compilation")
            return 0

        with mock.patch.object(fc2d, "run", side_effect=mutate_during_atc) as atc_run, \
                mock.patch.object(fc2d.shutil, "which", return_value="/mock/atc"), \
                contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            fc2d.do_om({"name": "compile_race"}, paths, "MC62CM12AA")
        atc_run.assert_called_once()
        self.assertIn("case SHA256 changed during compilation", output.getvalue())
        self.assertFalse((Path(paths["omdir"]) / "compile_race.om.fc2d.json").exists())


if __name__ == '__main__':
    unittest.main()
