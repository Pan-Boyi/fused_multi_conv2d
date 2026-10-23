# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Verify the device-reference path with independent synthetic ACL outputs."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
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
import fc2d_fusion as fusion
import run_fused_multi_conv2d as runner
import run_profile as profile
from fusion_test_support import fake_graph, write_test_binding, emit_graph_dumps, emit_tiling
from test_profile_workflow import LocalTransport


class OutputComparison(unittest.TestCase):
    def test_int8_exact_values_and_nchw_coordinates(self):
        reference = struct.pack('8b', -128, 0, 1, 127, -4, 3, 2, 1)
        fused = reference[:-1] + b'\x02'
        exact = runner.compare_device_outputs(reference, reference, 2, [1, 2, 2, 2])
        self.assertTrue(exact['passed'])
        report = runner.compare_device_outputs(fused, reference, 2, [1, 2, 2, 2], rel_tol=1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['bit_mismatch_count'], 1)
        self.assertEqual(report['tolerance_mismatch_count'], 1)
        self.assertEqual(report['first_mismatch']['coordinates'], [0, 1, 1, 1])

    def test_fp16_rounding_and_first_outside_tolerance_are_distinct(self):
        reference = struct.pack('<3e', 1, 1, 1)
        fused = struct.pack('<3e', 1 + 2**-10, 1, 1 + 2**-9)
        report = runner.compare_device_outputs(fused, reference, 1, [1, 1, 1, 3])
        self.assertFalse(report['passed'])
        self.assertEqual(report['bit_mismatch_count'], 2)
        self.assertEqual(report['tolerance_mismatch_count'], 1)
        self.assertEqual(report['first_mismatch']['flat_index'], 0)
        self.assertEqual(report['first_tolerance_mismatch']['flat_index'], 2)
        near = runner.compare_device_outputs(fused[:2], reference[:2], 1, [1])
        self.assertTrue(near['passed'])
        self.assertFalse(near['bitwise_equal'])

    def test_zero_reference_is_exact_but_signed_zero_is_numerically_equal(self):
        zero = struct.pack('<e', 0)
        tiny = struct.pack('<e', 2**-24)
        report = runner.compare_device_outputs(tiny, zero, 1, [1])
        self.assertFalse(report['passed'])
        self.assertEqual(report['zero_reference_absolute_tolerance'], 0)
        self.assertEqual(report['max_relative_error'], 'inf')
        signed = runner.compare_device_outputs(struct.pack('<e', -0.0), zero, 1, [1])
        self.assertTrue(signed['passed'])
        self.assertFalse(signed['bitwise_equal'])

    def test_nonfinite_values_fail_even_for_identical_bits(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                raw = struct.pack('<e', value)
                report = runner.compare_device_outputs(raw, raw, 1, [1])
                self.assertFalse(report['passed'])
                self.assertTrue(report['bitwise_equal'])
                self.assertEqual(report['nonfinite_count'], {'reference': 1, 'fused': 1})
                json.dumps(report, allow_nan=False)

    def test_mismatched_output_shapes_bytes_and_dtype_are_rejected(self):
        for args in ((b'\0', b'\0\0', 2, [1]), (b'\0', b'\0', 2, [2]),
                     (b'\0\0', b'\0\0', 3, [1])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.compare_device_outputs(*args)
        for tolerance in (-1, float('nan'), float('inf')):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                runner.compare_device_outputs(b'\0\0', b'\0\0', 1, [1], tolerance)


class ReferenceComparison(unittest.TestCase):
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
        self.case_config = fc2d.normalise(dict(name='pair', dtype='a16w8', hi=8, wi=9,
                                               cout2=2, bias=True, fixed_shift_value=[13, 0]), 0)
        self.case = self.work / 'case.bin'
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.do_case(self.case_config, {'bin': str(self.case)})
        self._zero_cpu_expected()
        self.fused = self.work / 'om/pair.om'
        self.reference = self.work / 'reference/pair.om'
        for path, payload in ((self.fused, b'synthetic fused OM'),
                              (self.reference, b'synthetic two-Extend OM')):
            path.parent.mkdir()
            path.write_bytes(payload)
        write_test_binding(self.case, self.fused)
        self.reference_sidecar = Path(runner.write_reference_model_binding(
            str(self.case), str(self.reference), str(self.fused), self.reference_report()))
        self.output = self.work / 'comparison'
        self.tensors, *_, self.info = runner.load_case(str(self.case))
        self.output_dtype, self.output_shape, expected = self.tensors['y_expect']
        count = len(expected) // 2
        self.reference_data = struct.pack('<%de' % count, *[0.5 + (i % 4) / 4 for i in range(count)])
        self.reference_output = self.work / 'reference_output.bin'
        self.fused_output = self.work / 'fused_output.bin'
        self.reference_output.write_bytes(self.reference_data)
        self.fused_output.write_bytes(self.reference_data)
        self.conf = self.work / 'stub.conf'
        self.conf.write_text(''.join('%s %d\n' % (name, len(self.tensors[name][2]))
                                    for name in runner.model_input_slots(self.info)) +
                             'OUT %d\n' % len(expected))
        self.trace = self.work / 'trace.txt'
        self.env = dict(os.environ, FC2D_STUB_CONF=str(self.conf),
                        FC2D_STUB_REFERENCE_OM=str(self.reference),
                        FC2D_STUB_REFERENCE_OUTPUT=str(self.reference_output),
                        FC2D_STUB_FUSED_OUTPUT=str(self.fused_output),
                        FC2D_STUB_OUTPUT_DTYPE=str(self.output_dtype),
                        FC2D_STUB_OUTPUT_SHAPE=','.join(map(str, self.output_shape)),
                        FC2D_STUB_TRACE=str(self.trace), REPEAT='1', WARMUP='0')
        for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
            self.env[key] = str(self.stub)

    def _zero_cpu_expected(self):
        blob = bytearray(self.case.read_bytes())
        offset = runner.HDR_LEN
        for _ in range(struct.unpack_from('<I', blob, 12)[0]):
            record = struct.unpack_from(runner.REC_FMT, blob, offset)
            offset += runner.REC_LEN
            if record[0].rstrip(b'\0') == b'y_expect':
                blob[offset:offset + record[-1]] = bytes(record[-1])
            offset += record[-1]
        struct.pack_into('<Q', blob, 16, 0)
        self.case.write_bytes(blob)

    def reference_report(self):
        fused = runner.validate_model_binding(str(self.case), str(self.fused))
        return dict(pipeline='extend_pair_reference', source_conv_count=2,
                    stage1_extend_count=2, unfused_convolution_count=2, fused_node_count=0,
                    input_names=runner.model_input_slots(runner._binding_spec(str(self.case))),
                    source_graph_sha256=fused['fusion']['source_graph_sha256'],
                    model_graph_sha256='1' * 64, disabled_passes=[fusion.PASS_NAME],
                    enabled_passes=[fusion.FIRST_PASS_NAME])

    def run_pair(self, extra_env=None, extra_args=None):
        env = dict(self.env, **(extra_env or {}))
        command = [sys.executable, str(ROOT / 'run_fused_multi_conv2d.py'), str(self.case),
                   'FusedMultiConv2d', '0', str(self.fused), '--reference-om', str(self.reference),
                   '--comparison-output', str(self.output)] + (extra_args or [])
        result = subprocess.run(command, env=env, cwd=str(self.work), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report_path = self.output / 'comparison.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        return result, report

    def test_reference_binding_rejects_changed_artifacts_and_contracts(self):
        baseline = json.loads(self.reference_sidecar.read_text())
        self.assertEqual(runner.validate_reference_model_binding(str(self.case), str(self.reference),
                         str(self.fused)), baseline)
        alterations = (dict(baseline, schema_version=0), dict(baseline, role='onnx_conv_pair_unfused'),
                       dict(baseline, model_sha256='0' * 64), dict(baseline, fused_model_sha256='0' * 64),
                       dict(baseline, case_sha256='0' * 64), dict(baseline, model_input_names=['x']),
                       dict(baseline, reference=dict(baseline['reference'], source_graph_sha256='0' * 64)),
                       dict(baseline, reference=dict(baseline['reference'], stage1_extend_count=0)))
        for altered in alterations:
            with self.subTest(altered=altered):
                self.reference_sidecar.write_text(json.dumps(altered))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    runner.validate_reference_model_binding(str(self.case), str(self.reference), str(self.fused))
        self.reference_sidecar.write_text(json.dumps(baseline))
        self.reference.write_bytes(self.reference.read_bytes() + b'changed')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.validate_reference_model_binding(str(self.case), str(self.reference), str(self.fused))

    def test_reference_writer_rejects_same_model_and_source_or_input_mismatch(self):
        for report in (dict(self.reference_report(), source_graph_sha256='0' * 64),
                       dict(self.reference_report(), input_names=['x'])):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                runner.write_reference_model_binding(str(self.case), str(self.reference), str(self.fused), report)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.write_reference_model_binding(str(self.case), str(self.fused), str(self.fused), self.reference_report())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.write_reference_model_binding(str(self.case), str(self.reference), str(self.fused),
                                                  self.reference_report(), expected_case_sha256='0' * 64)

    def test_device_outputs_are_independent_of_cpu_golden_and_both_execute(self):
        result, report = self.run_pair()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report['passed'])
        self.assertFalse(report['cpu_golden_used'])
        self.assertEqual(report['golden_source'], 'extend_pair_device_output')
        self.assertEqual((self.output / 'reference.bin').read_bytes(), self.reference_data)
        self.assertEqual((self.output / 'fused.bin').read_bytes(), self.reference_data)
        self.assertNotEqual(self.reference_data, self.tensors['y_expect'][2])
        trace = [line.split() for line in self.trace.read_text().splitlines()]
        self.assertEqual([line[1] for line in trace], ['reference', 'reference', 'fused', 'fused'])
        self.assertTrue(all(line[2:] == trace[0][2:] for line in trace))
        self.assertEqual(report['models']['reference']['model_sha256'], runner._file_sha256(str(self.reference)))

    def test_device_difference_fails_and_keeps_both_raw_outputs(self):
        changed = bytearray(self.reference_data)
        changed[-2:] = struct.pack('<e', 8)
        self.fused_output.write_bytes(changed)
        result, report = self.run_pair({'RATIO_MIN': '0'})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(report['passed'])
        self.assertEqual(report['comparison']['tolerance_mismatch_count'], 1)
        self.assertEqual((self.output / 'reference.bin').read_bytes(), self.reference_data)
        self.assertEqual((self.output / 'fused.bin').read_bytes(), changed)

    def test_acl_failure_and_output_contract_failures_do_not_reuse_stale_results(self):
        for failure in ({'FC2D_STUB_REFERENCE_EXECUTE_FAIL': '1'},
                        {'FC2D_STUB_REFERENCE_SHAPE': '1,1,1,1'},
                        {'FC2D_STUB_REFERENCE_DTYPE': '2'},
                        {'FC2D_STUB_REFERENCE_SIZE': '1'},
                        {'FC2D_STUB_REFERENCE_SKIP_WRITE': '1', 'FC2D_STUB_FUSED_SKIP_WRITE': '1'}):
            with self.subTest(failure=failure):
                self.output.mkdir(exist_ok=True)
                for filename in ('reference.bin', 'fused.bin', 'comparison.json'):
                    (self.output / filename).write_bytes(b'stale')
                result, report = self.run_pair(failure)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(report['passed'])
                self.assertEqual(report['models']['reference']['status'], 'failed')
                self.assertFalse((self.output / 'reference.bin').exists())
                if 'FC2D_STUB_FUSED_SKIP_WRITE' not in failure:
                    self.assertEqual(report['models']['fused']['status'], 'success')
                    self.assertEqual((self.output / 'fused.bin').read_bytes(), self.reference_data)

    def test_identical_nan_device_outputs_fail(self):
        raw = struct.pack('<e', float('nan')) + self.reference_data[2:]
        self.reference_output.write_bytes(raw)
        self.fused_output.write_bytes(raw)
        result, report = self.run_pair()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report['comparison']['bitwise_equal'])
        self.assertEqual(report['comparison']['nonfinite_count'], {'reference': 1, 'fused': 1})

    def test_reference_cli_rejects_dry_run_and_malformed_arguments(self):
        result, report = self.run_pair(extra_args=['--dry-run'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires actual device execution', result.stdout)
        self.assertIsNone(report)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner._extract_pair_arguments(['--comparison-output', '/tmp/output'])

    def test_reference_graph_requires_two_direct_extend_nodes_and_no_runtime_helper(self):
        names = runner.model_input_slots(self.info)
        report = fusion.check_reference_graph(fake_graph(names, True), names)
        self.assertEqual(report['pipeline'], 'extend_pair_reference')
        self.assertEqual(report['enabled_passes'], [fusion.FIRST_PASS_NAME])
        self.assertEqual(report['disabled_passes'], [fusion.PASS_NAME])
        for target in ('FusedMultiConv2d', 'SetM1Dequant', 'Conv2D'):
            graph = copy.deepcopy(fake_graph(names, True))
            graph['graph'][0]['op'].append(dict(name='unexpected', type=target))
            with self.subTest(target=target), self.assertRaises(ValueError):
                fusion.check_reference_graph(graph, names)

    def test_compile_uses_same_source_and_separate_stage2_off_reference(self):
        paths = dict(bin=str(self.case), onnx=str(self.work / 'source.onnx'),
                     omdir=str(self.fused.parent))
        compiles = []

        def atc(command, **kwargs):
            if '--mode=1' in command:
                model = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--om=')))
                target = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--json=')))
                target.write_text(json.dumps(fake_graph(runner.model_input_slots(self.info),
                                                       model.parent.name == 'reference')))
                return 0
            source = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--model=')))
            switch = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--fusion_switch_file=')))
            output = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--output=')))
            switches = json.loads(switch.read_text())['Switch']['GraphFusion']
            compiles.append((source.read_bytes(), switches, output.parent.name))
            output.with_suffix('.om').write_bytes(b'synthetic OM ' + output.parent.name.encode())
            if switches[fusion.PASS_NAME] == 'on':
                emit_graph_dumps(runner.model_input_slots(self.info))
                emit_tiling(fc2d.read_spec(str(self.case)))
            return 0

        with mock.patch.object(fc2d.shutil, 'which', return_value='/synthetic/atc'), \
                mock.patch.object(fc2d, 'run', side_effect=atc), \
                contextlib.redirect_stdout(io.StringIO()):
            fc2d.do_om(self.case_config, paths, 'MC62CM12AA', compare_unfused=True)
        self.assertEqual(len(compiles), 2)
        self.assertEqual(compiles[0][0], compiles[1][0])
        self.assertEqual(compiles[0][1], {fusion.FIRST_PASS_NAME: 'on', fusion.PASS_NAME: 'on'})
        self.assertEqual(compiles[1][1], {fusion.FIRST_PASS_NAME: 'on', fusion.PASS_NAME: 'off'})
        self.assertEqual([entry[2] for entry in compiles], ['om', 'reference'])
        runner.validate_reference_model_binding(str(self.case), str(self.reference), str(self.fused))

    def local_remote(self):
        scripts = self.work / 'scripts'
        scripts.mkdir(exist_ok=True)
        python = scripts / 'python3'
        python.write_text('#!/bin/bash\nexport DYLD_LIBRARY_PATH=%s\nexec %s "$@"\n' %
                          (shlex.quote(str(self.stub)), shlex.quote(sys.executable)))
        python.chmod(0o755)
        env = dict(self.env, PATH=str(scripts) + os.pathsep + self.env['PATH'],
                   FC2D_STUB_REFERENCE_OM='reference.om')
        return LocalTransport(self.work / 'remote', env)

    def profile_paths(self):
        return dict(bin=str(self.case), omdir=str(self.fused.parent))

    def test_profile_uploads_and_runs_pair_and_downloads_failure_evidence(self):
        remote = self.local_remote()
        paths = self.profile_paths()
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(remote, self.case_config, paths, str(self.fused), reference_om=str(self.reference))
        self.assertEqual(len(remote.uploads), 6)
        self.assertEqual({Path(target).name for _, target in remote.uploads},
                         {'case.bin', 'pair.om', 'pair.om.fc2d.json', 'reference.om',
                          'reference.om.fc2d.json', 'run_fused_multi_conv2d.py'})
        for success in (True, False):
            with self.subTest(success=success):
                raw = self.reference_data if success else struct.pack('<e', 8) + self.reference_data[2:]
                self.fused_output.write_bytes(raw)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    state = profile.do_prof(remote, self.case_config, paths, str(self.fused), False, '',
                                            reference_om=str(self.reference))
                self.assertEqual(state['device_rc'], 0 if success else 1, output.getvalue())
                self.assertEqual(state['comparison_passed'], success, output.getvalue())
                self.assertEqual(profile.execution_failed(state), not success)
                report = Path(state['comparison_report'])
                self.assertEqual((report.parent / 'reference.bin').read_bytes(), self.reference_data)
                self.assertEqual((report.parent / 'fused.bin').read_bytes(), raw)
                self.assertEqual(json.loads(report.read_text())['passed'], success)
        self.assertEqual(len(remote.downloads), 2)

    def test_remote_stale_reference_is_rejected_before_device_execution(self):
        remote = self.local_remote()
        paths = self.profile_paths()
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(remote, self.case_config, paths, str(self.fused), reference_om=str(self.reference))
        reference = Path(remote.dir) / self.case_config['name'] / 'reference.om'
        reference.write_bytes(reference.read_bytes() + b'changed after upload')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'without a result'):
            profile.do_prof(remote, self.case_config, paths, str(self.fused), False, '',
                            reference_om=str(self.reference))
        self.assertFalse(self.trace.exists())
        self.assertEqual(remote.downloads, [])

    def test_profile_rejects_tampered_download_and_profiling_pair(self):
        remote = self.local_remote()
        paths = self.profile_paths()
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(remote, self.case_config, paths, str(self.fused), reference_om=str(self.reference))
        download = remote.scp_from

        def tamper(source, target, recursive=False):
            result = download(source, target, recursive)
            raw = Path(target) / 'comparison/fused.bin'
            raw.write_bytes(b'\0\0' + raw.read_bytes()[2:])
            return result

        with mock.patch.object(remote, 'scp_from', side_effect=tamper), \
                contextlib.redirect_stdout(io.StringIO()):
            state = profile.do_prof(remote, self.case_config, paths, str(self.fused), False, '',
                                    reference_om=str(self.reference))
        self.assertEqual(state['collection_rc'], 2)
        self.assertIn('output hash differs', state['comparison_error'])
        self.assertTrue(profile.execution_failed(state))
        with self.assertRaisesRegex(RuntimeError, '--no-msprof'):
            profile.do_prof(remote, self.case_config, paths, str(self.fused), True, '/not-used',
                            reference_om=str(self.reference))

    def test_profile_comparison_flag_requires_no_msprof_before_loading_config(self):
        with mock.patch.object(sys, 'argv', ['run_profile.py', '/missing.json', '--compare-unfused']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            profile.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
