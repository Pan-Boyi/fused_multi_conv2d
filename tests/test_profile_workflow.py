"""Local workflow gates with real cases/ONNX/ACL stub and synthetic transport/ATC."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fc2d_onnx
import run_profile as profile
import run_fused_conv2d as runner
sys.path.insert(0, str(ROOT / 'tests/stub_ascendcl'))
import run_stub_test as stub_test


class LocalTransport:
    """Use only temporary local directories; never connect to a machine."""
    target = 'local-test-only'
    cann_env = ''
    repeat = 1
    warmup = 0
    device = 0
    auth_missing = []

    def __init__(self, directory, env):
        self.dir, self.env = str(directory), env
        self.uploads = []
        self.downloads = []

    def describe(self):
        return 'local synthetic transport'

    def ssh(self, command, stdin_script=None, capture=False):
        result = subprocess.run(['bash', '-s'] if stdin_script is not None else ['bash', '-c', command],
                                input=stdin_script, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.returncode, result.stdout

    def scp_to(self, source, target):
        self.uploads.append((source, target))
        shutil.copyfile(source, target)
        return 0

    def scp_from(self, source, target, recursive=False):
        self.downloads.append((source, target))
        shutil.copytree(source, Path(target) / Path(source).name)
        return 0


class ProfileWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')
            fc2d.ensure_tool('fc2d_geom')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shared = Path(cls.tmp.name)
        cls.stub = cls.shared / 'stub'
        cls.stub.mkdir()
        subprocess.run(['gcc', '-shared', '-fPIC', '-O1', str(ROOT / 'tests/stub_ascendcl/stub.c'),
                        '-o', str(cls.stub / 'libascendcl.so')], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.out = self.work / 'out'
        self.profs = self.work / 'prof_out'
        self.case = fc2d.normalise({'name': 'contract', 'dtype': 'int8',
                                    'hi': 8, 'wi': 9, 'cout2': 2}, 0)
        self.paths = {'bin': str(self.out / 'contract/case.bin'),
                      'omdir': str(self.out / 'contract/om'),
                      'onnx': str(self.out / 'contract/contract.onnx')}
        Path(self.paths['bin']).parent.mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.do_case(self.case, self.paths)
        Path(self.paths['omdir']).mkdir()
        self.model = Path(self.paths['omdir']) / 'contract.om'
        self.model.write_bytes(b'synthetic OM for runner testing; not an ATC result')
        runner.write_model_binding(self.paths['bin'], str(self.model))
        self.conf = self.work / 'stub_conf'
        subprocess.run([sys.executable, str(ROOT / 'tests/stub_ascendcl/make_conf.py'),
                        self.paths['bin'], str(self.conf)], check=True, stdout=subprocess.DEVNULL)
        env = dict(os.environ, FC2D_STUB_CONF=str(self.conf / 'stub.conf'),
                   FC2D_STUB_GOLDEN=str(self.conf / 'golden.bin'))
        for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
            env[key] = str(self.stub)
        test_bin = self.work / 'test_bin'
        test_bin.mkdir()
        python = test_bin / 'python3'
        python.write_text('#!/bin/bash\nexport DYLD_LIBRARY_PATH=%s\nexec %s "$@"\n' %
                          (shlex.quote(str(self.stub)), shlex.quote(sys.executable)))
        python.chmod(0o755)
        env['PATH'] = str(test_bin) + os.pathsep + env['PATH']
        self.remote = LocalTransport(self.work / 'remote', env)
        self.msprof = self.work / 'fake_msprof'
        # Intentionally swallow the application RC, reproducing the failure mode.
        self.msprof.write_text('#!/bin/bash\n"$@"\nmkdir "PROF_$(python3 -c "import uuid; print(uuid.uuid4().hex)")"\nexit 0\n')
        self.msprof.chmod(0o755)
        self.suite = self.work / 'suite'
        exporter = self.suite / 'analysis/msprof/msprof.py'
        exporter.parent.mkdir(parents=True)
        exporter.write_text("import pathlib,sys\nd=pathlib.Path(sys.argv[-1])/'mindstudio_profiler_output'\nd.mkdir(exist_ok=True)\n(d/'op_summary_test.csv').write_text('total_exe_time(us)\\n12\\n10\\n')\n")
        self.config = self.work / 'synthetic.json'
        self.config.write_text(json.dumps({'soc_version': 'MC62CM12AA',
                                          'outdir': str(self.out), 'profdir': str(self.profs),
                                          'profile': {'msprof': str(self.msprof),
                                                      'tbe_work_test_suit': str(self.suite),
                                                      'columns': ['total_exe_time(us)']},
                                          'cases': [self.case]}))

    def cli(self, steps):
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config), '--steps', steps]), \
                mock.patch.object(profile, 'Remote', return_value=self.remote), \
                contextlib.redirect_stdout(output):
            result = profile.main()
        return result, output.getvalue()

    def push(self):
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(self.remote, self.case, self.paths, str(self.model))

    def test_all_stages_generate_graph_upload_run_download_export_and_parse(self):
        original_run = fc2d.run

        def synthetic_atc(command):
            if command[0] == 'atc':
                import onnx
                model_path = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--model='))
                graph = onnx.load(model_path).graph
                self.assertEqual(len(graph.input), 7)
                self.assertEqual(len(graph.node[0].input), 9)
                self.assertEqual({t.name for t in graph.initializer}, {'filter1', 'filter2'})
                self.assertIn('--input_format=NCHW', command)
                prefix = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--output='))
                Path(prefix + '.om').write_bytes(b'synthetic ATC result')
                return 0
            return original_run(command)

        with mock.patch.object(fc2d.shutil, 'which', return_value='/synthetic/atc'), \
                mock.patch.object(fc2d, 'run', side_effect=synthetic_atc) as calls:
            rc, output = self.cli(','.join(profile.ALL_STEPS))
        self.assertEqual(rc, 0, output)
        self.assertTrue(any(call.args[0][0] == 'atc' for call in calls.call_args_list))
        state = profile.load_profile_state(self.paths, str(self.model))
        self.assertEqual(state['device_rc'], 0)
        self.assertEqual(state['summary']['rows'], 2)
        self.assertEqual(len(self.remote.uploads), 4)
        self.assertTrue(Path(state['execution_log']).is_file())
        self.assertIn('[PASS]', Path(state['execution_log']).read_text())
        self.assertEqual(len(self.remote.downloads), 1)

    def test_resume_pull_export_parse_independently_without_profiling_again(self):
        self.assertEqual(self.cli('push,prof')[0], 0)
        self.assertEqual(self.cli('pull')[0], 0)
        # Local phases must not construct Remote or ask for authentication.
        for phase in ('export', 'parse'):
            with mock.patch.object(profile, 'Remote', side_effect=AssertionError('remote used by local phase')):
                output = io.StringIO()
                with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config), '--steps', phase]), \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(profile.main(), 0, output.getvalue())
        self.assertEqual(len(self.remote.downloads), 1)
        state = profile.load_profile_state(self.paths, str(self.model))
        self.assertEqual(state['summary']['total_exe_time(us)']['min'], 10)

    def test_missing_state_or_local_prof_is_an_error(self):
        self.assertEqual(self.cli('pull')[0], 1)
        self.assertEqual(self.cli('push,prof')[0], 0)
        self.assertEqual(self.cli('parse')[0], 1)

    def test_case_or_model_change_rejects_old_prof_even_after_rebinding(self):
        self.assertEqual(self.cli('push,prof,pull')[0], 0)
        for changed in ('case', 'model'):
            with self.subTest(changed=changed):
                path = Path(self.paths['bin']) if changed == 'case' else self.model
                path.write_bytes(path.read_bytes() + b'changed')
                runner.write_model_binding(self.paths['bin'], str(self.model))
                self.assertEqual(self.cli('parse')[0], 1)

    def test_remote_old_but_self_consistent_bundle_is_rejected_before_execution(self):
        self.push()
        remote_model = Path(self.remote.dir) / 'contract/contract.om'
        remote_model.write_bytes(b'old remote model')
        runner.write_model_binding(str(remote_model.parent / 'case.bin'), str(remote_model))
        rc, output = self.cli('prof')
        self.assertEqual(rc, 1, output)
        self.assertIn('differs from the selected local binding', output)
        state = json.loads(Path(profile.profile_state_path(self.paths)).read_text())
        self.assertIsNone(state['device_rc'])
        self.assertIsNone(state['prof'])

    def test_prof_only_rejects_local_binding_before_any_remote_execution(self):
        self.model.write_bytes(b'changed model')
        with mock.patch.object(self.remote, 'ssh', wraps=self.remote.ssh) as ssh, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            profile.do_prof(self.remote, self.case, self.paths, str(self.model), True, str(self.msprof))
        ssh.assert_not_called()

    def test_profiler_swallowing_runner_failure_preserves_prof_but_cli_fails(self):
        self.push()
        # Bad model byte sizes make the real runner fail before execute. msprof returns zero.
        conf = self.conf / 'stub.conf'
        original = conf.read_text()
        first_line = original.splitlines()[0]
        self.assertEqual(first_line.split()[0], 'x')
        self.assertGreater(int(first_line.split()[1]), 1)
        conf.write_text(original.replace(first_line, 'x 1', 1))
        self.assertNotEqual(conf.read_text(), original)
        rc, output = self.cli('prof,pull,export,parse')
        self.assertEqual(rc, 1, output)
        state = profile.load_profile_state(self.paths, str(self.model))
        self.assertNotEqual(state['device_rc'], 0)
        self.assertEqual(state['profiler_rc'], 0)
        self.assertEqual(state['ssh_rc'], 1)
        self.assertEqual(state['summary']['rows'], 2)
        self.assertTrue(Path(state['local_prof']).is_dir())
        self.assertNotIn('横向对比', output)
        self.assertEqual(self.cli('parse')[0], 1)

    def test_no_msprof_keeps_functional_return_code_and_log(self):
        self.push()
        with contextlib.redirect_stdout(io.StringIO()):
            state = profile.do_prof(self.remote, self.case, self.paths, str(self.model), False, '')
        self.assertFalse(profile.execution_failed(state))
        self.assertIsNone(state['prof'])
        self.assertIn('[PASS]', Path(state['execution_log']).read_text())


    def test_keep_going_runs_next_case_after_binding_validation_failure(self):
        second = dict(self.case, name='second')
        directory = self.out / 'second'
        directory.mkdir()
        shutil.copyfile(self.paths['bin'], directory / 'case.bin')
        (directory / 'om').mkdir()
        model = directory / 'om/second.om'
        model.write_bytes(b'second valid model')
        runner.write_model_binding(str(directory / 'case.bin'), str(model))
        self.model.write_bytes(b'stale first model')
        document = json.loads(self.config.read_text())
        document['cases'] = [self.case, second]
        self.config.write_text(json.dumps(document))
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config), '--steps', 'push', '--keep-going']), \
                mock.patch.object(profile, 'Remote', return_value=self.remote), \
                contextlib.redirect_stdout(output):
            self.assertEqual(profile.main(), 1)
        self.assertEqual(len(self.remote.uploads), 4)
        self.assertTrue(all('/second/' in target for _, target in self.remote.uploads))

    def test_no_msprof_runner_failure_is_saved_and_cli_returns_failure(self):
        self.push()
        conf = self.conf / 'stub.conf'
        lines = conf.read_text().splitlines()
        lines[0] = 'x 1'
        conf.write_text('\n'.join(lines) + '\n')
        document = json.loads(self.config.read_text())
        document['profile']['enabled'] = False
        self.config.write_text(json.dumps(document))
        rc, output = self.cli('prof')
        self.assertEqual(rc, 1, output)
        state = json.loads(Path(profile.profile_state_path(self.paths)).read_text())
        self.assertNotEqual(state['device_rc'], 0)
        self.assertEqual(state['profiler_rc'], 0)
        self.assertIsNone(state['prof'])
        self.assertIn('[X]', Path(state['execution_log']).read_text())


class DefaultAndContractCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')
            fc2d.ensure_tool('fc2d_geom')

    def test_direct_cpp_defaults_match_python_and_output_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / 'direct.bin'
            args = ['--hi', '8', '--wi', '9', '--cout2', '2']
            subprocess.run([str(ROOT / 'gen_case'), str(case)] + args,
                           check=True, stdout=subprocess.DEVNULL)
            spec = fc2d.read_spec(str(case))
            self.assertEqual([spec['s1'], spec['s2']], fc2d.DEFAULTS['strides'][::2])
            self.assertEqual([spec[k] for k in ('ph1', 'pw1', 'ph2', 'pw2')], fc2d.DEFAULTS['pads'][::2])
            self.assertEqual([spec['ho2'], spec['wo2']], [4, 5])
            geom = subprocess.check_output([str(ROOT / 'fc2d_geom')] + args, text=True)
            self.assertIn('conv1 -> 6x7, conv2 -> 4x5', geom)

    def test_four_dtype_contract_config_graph_and_stub_entry(self):
        import onnx
        cases = profile.load_cases(json.loads((ROOT / 'contract_cases.json').read_text())['cases'])
        self.assertEqual({case['dtype'] for case in cases}, {'fp16', 'int8', 's8f16', 'a16w8'})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'cases'
            for case in cases:
                directory = output / case['name']
                directory.mkdir(parents=True)
                paths = {'bin': str(directory / 'case.bin'), 'onnx': str(directory / 'case.onnx')}
                with contextlib.redirect_stdout(io.StringIO()):
                    fc2d.do_check(case, 1024 * 1024)
                    fc2d.do_case(case, paths)
                    plan, status = fc2d_onnx.write_onnx(paths['bin'], paths['onnx'])
                model = onnx.load(paths['onnx'])
                self.assertEqual([v.name for v in model.graph.input], runner.model_input_slots(fc2d.read_spec(paths['bin'])))
                self.assertEqual(len(model.graph.node[0].input), 9)
                self.assertEqual([list(t.dims) for t in model.graph.initializer],
                                 [[64, 32, *case['kernel'][:2]], [2, 64, *case['kernel'][2:]]])
                self.assertNotIn('kernel_size', plan['attrs'])
                self.assertEqual(plan['attrs']['fixed_shift_value'], case['fixed_shift_value'])
                self.assertEqual(plan['attrs']['strides'], case['strides'])
                self.assertEqual(plan['attrs']['pads'], case['pads'])
                conf_dir = directory / 'conf'
                subprocess.run([sys.executable, str(ROOT / 'tests/stub_ascendcl/make_conf.py'),
                                paths['bin'], str(conf_dir)], check=True, stdout=subprocess.DEVNULL)
                sizes = {onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.INT8: 1,
                         onnx.TensorProto.INT32: 4, onnx.TensorProto.UINT64: 8,
                         onnx.TensorProto.FLOAT: 4}
                actual = []
                for value in model.graph.input:
                    tensor = value.type.tensor_type
                    bytecount = sizes[tensor.elem_type]
                    for dim in tensor.shape.dim:
                        bytecount *= dim.dim_value
                    actual.append((value.name, bytecount))
                reported = [line.split() for line in (conf_dir / 'stub.conf').read_text().splitlines()
                            if not line.startswith('OUT ')]
                self.assertEqual(actual, [(name, int(count)) for name, count in reported])

                (directory / 'om').mkdir()
                fake_om = directory / 'om/test.om'
                fake_om.write_bytes(b'synthetic ACL stub OM, not a compiled device model')
                runner.write_model_binding(paths['bin'], str(fake_om))
            env = dict(os.environ, PYTHON=sys.executable)
            result = subprocess.run(['bash', str(ROOT / 'tests/stub_ascendcl/run_stub_test.sh'),
                                     '--out-root', str(output), '--work-dir', str(Path(temp) / 'stub')],
                                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.count(' PASS; log:'), 8)


class StubSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def make_bundle(self, directory):
        directory.mkdir(parents=True)
        case = directory / 'case.bin'
        subprocess.run([str(ROOT / 'gen_case'), str(case), '--dtype', 'int8', '--hi', '8',
                        '--wi', '9', '--cout2', '2'], check=True, stdout=subprocess.DEVNULL)
        (directory / 'om').mkdir()
        model = directory / 'om/model.om'
        model.write_bytes(b'synthetic stub OM')
        runner.write_model_binding(str(case), str(model))
        return directory.resolve()

    def test_supports_plain_and_config_directory_without_pairing_mistakes(self):
        first = self.make_bundle(self.work / 'plain')
        second = self.make_bundle(self.work / 'config/nested')
        selected = stub_test.select_cases(self.work, [], [])
        self.assertEqual({item[0] for item in selected}, {first, second})
        for directory, case, model in selected:
            self.assertEqual(case.parent, directory)
            self.assertEqual(model.parent.parent, directory)

    def test_same_name_ambiguity_requires_explicit_directory(self):
        first = self.make_bundle(self.work / 'one/same')
        self.make_bundle(self.work / 'two/same')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            stub_test.select_cases(self.work, ['same'], [])
        self.assertEqual(stub_test.select_cases(self.work, [], [str(first)])[0][0], first)

    def test_requested_missing_case_and_multiple_oms_fail(self):
        with self.assertRaisesRegex(ValueError, 'found 0'):
            stub_test.select_cases(self.work, ['missing'], [])
        directory = self.make_bundle(self.work / 'case')
        (directory / 'om/extra.om').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            stub_test.select_cases(self.work, ['case'], [])

    def test_pass_text_does_not_override_nonzero_return_code(self):
        result = subprocess.CompletedProcess([], 7, '[PASS] stale text\n')
        with mock.patch.object(stub_test.subprocess, 'run', return_value=result):
            self.assertFalse(stub_test.run_mode(['fake'], {}, self.work, self.work / 'run.log'))


if __name__ == '__main__':
    unittest.main()
