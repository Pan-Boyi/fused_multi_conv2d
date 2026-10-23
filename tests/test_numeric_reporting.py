# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Verify numerical diagnostics independently of the device and CPU oracles."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import run_fused_multi_conv2d as runner
import run_profile as profile
import test_reference_compare as reference_tests


class NumericSummary(unittest.TestCase):
    def test_first_fifty_differences_do_not_truncate_full_output_statistics(self):
        reference = struct.pack('<64e', *([1] * 64))
        actual = struct.pack('<64e', *([2] * 64))
        report = runner.compare_device_outputs(actual, reference, 1, [1, 2, 4, 8])
        self.assertEqual(report['elements'], 64)
        self.assertEqual(report['bit_mismatch_count'], 64)
        self.assertEqual(report['tolerance_mismatch_count'], 64)
        self.assertEqual(len(report['mismatch_samples']), 50)
        self.assertEqual([point['flat_index'] for point in report['mismatch_samples']], list(range(50)))
        last = report['mismatch_samples'][-1]
        self.assertEqual(last['coordinates'], [0, 1, 2, 1])
        self.assertEqual((last['fused'], last['reference']), (2, 1))
        self.assertEqual((last['absolute_error'], last['relative_error']), (1, 1))
        precision = report['precision_0_001']
        self.assertEqual(precision, dict(relative_tolerance=0.001, passed_count=0,
                                        failed_count=64, total_count=64, pass_ratio=0,
                                        pass_percent=0))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.print_output_comparison(report, 'actual', 'reference')
        self.assertIn('first 50 mismatches', output.getvalue())
        self.assertIn('index=49 ', output.getvalue())
        self.assertNotIn('index=50 ', output.getvalue())
        self.assertIn('0 / 64', output.getvalue())

    def test_fixed_precision_statistics_are_independent_of_configured_tolerance(self):
        reference = struct.pack('<3e', 1, -1, 1)
        actual = struct.pack('<3e', 1 + 2**-10, -1.25, 1)
        report = runner.compare_device_outputs(actual, reference, 1, [3], rel_tol=0.5)
        self.assertTrue(report['passed'])
        self.assertEqual(report['tolerance_mismatch_count'], 0)
        self.assertEqual(report['precision_0_001']['passed_count'], 2)
        self.assertEqual(report['precision_0_001']['failed_count'], 1)
        self.assertAlmostEqual(report['precision_0_001']['pass_percent'], 200 / 3)
        self.assertEqual([point['precision_0_001_passed'] for point in report['mismatch_samples']],
                         [True, False])

    def test_first_outside_tolerance_is_retained_after_fifty_rounding_differences(self):
        reference = struct.pack('<64e', *([1] * 64))
        values = [1 + 2**-10] * 64
        values[-1] = 2
        report = runner.compare_device_outputs(struct.pack('<64e', *values), reference, 1, [64])
        self.assertEqual(len(report['mismatch_samples']), 50)
        self.assertTrue(all(point['precision_0_001_passed'] for point in report['mismatch_samples']))
        self.assertEqual(report['first_tolerance_mismatch']['flat_index'], 63)
        self.assertEqual(report['tolerance_mismatch_count'], 1)
        self.assertEqual(report['precision_0_001']['passed_count'], 63)

    def test_zero_negative_ulp_and_nonfinite_values_are_reported_without_invalid_json(self):
        reference = struct.pack('<8e', 1, -1, 0, 0, 1, -1, float('inf'), float('nan'))
        actual = struct.pack('<8e', 1 + 2**-10, -(1 + 2**-9), 2**-24,
                             -0.0, 1, -1, float('inf'), float('nan'))
        report = runner.compare_device_outputs(actual, reference, 1, [1, 1, 2, 4])
        self.assertFalse(report['passed'])
        self.assertEqual(report['precision_0_001']['passed_count'], 4)
        self.assertEqual(report['precision_0_001']['failed_count'], 4)
        samples = {sample['flat_index']: sample for sample in report['mismatch_samples']}
        self.assertTrue(samples[0]['precision_0_001_passed'])
        self.assertFalse(samples[1]['precision_0_001_passed'])
        self.assertEqual(samples[2]['relative_error'], 'inf')
        self.assertTrue(samples[3]['precision_0_001_passed'])
        self.assertEqual(samples[3]['absolute_error'], 0)
        self.assertEqual(samples[6]['reference'], 'inf')
        self.assertEqual(samples[7]['fused'], 'nan')
        json.dumps(report, allow_nan=False)

    def test_integer_diagnostics_preserve_exact_comparison(self):
        report = runner.compare_device_outputs(struct.pack('4b', -128, -2, 0, 126),
                                                struct.pack('4b', -128, -1, 0, 127),
                                                2, [4], rel_tol=1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['tolerance_mismatch_count'], 2)
        self.assertEqual(report['precision_0_001']['passed_count'], 2)
        self.assertEqual([point['flat_index'] for point in report['mismatch_samples']], [1, 3])

    def test_legacy_precision_does_not_accept_identical_nonfinite_values(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                decoded = runner.decode_f16(struct.pack('<e', value))
                report = runner.evaluate(decoded, decoded, 0.001, 0x7F7F)
                self.assertEqual(report['prec_ok'], 0)

    def test_printed_values_have_distinct_actual_and_reference_labels(self):
        report = runner.compare_device_outputs(struct.pack('<2e', 2, -3),
                                                struct.pack('<2e', 1, -2), 1, [2])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.print_output_comparison(report, 'actual_test', 'reference_test')
        text = output.getvalue()
        self.assertIn('actual_test', text)
        self.assertIn('reference_test', text)
        self.assertIn('0.001', text)
        self.assertIn('2', text)
        self.assertIn('-3', text)
        self.assertIn('-2', text)


class NumericalDeviceDiagnostics(unittest.TestCase):
    setUpClass = classmethod(reference_tests.ReferenceComparison.setUpClass.__func__)
    tearDownClass = classmethod(reference_tests.ReferenceComparison.tearDownClass.__func__)
    setUp = reference_tests.ReferenceComparison.setUp
    _zero_cpu_expected = reference_tests.ReferenceComparison._zero_cpu_expected
    reference_report = reference_tests.ReferenceComparison.reference_report
    run_pair = reference_tests.ReferenceComparison.run_pair
    local_remote = reference_tests.ReferenceComparison.local_remote
    profile_paths = reference_tests.ReferenceComparison.profile_paths

    def test_pair_reports_values_counts_and_preserves_independent_device_oracle(self):
        actual = bytearray(self.reference_data)
        actual[-2:] = struct.pack('<e', 8)
        self.fused_output.write_bytes(actual)
        result, report = self.run_pair()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        summary = report['comparison']
        self.assertEqual(summary['precision_0_001']['passed_count'], len(actual) // 2 - 1)
        self.assertEqual(summary['precision_0_001']['failed_count'], 1)
        self.assertEqual(summary['mismatch_samples'][0]['fused'], 8)
        self.assertEqual(summary['mismatch_samples'][0]['reference'],
                         struct.unpack('<e', self.reference_data[-2:])[0])
        self.assertIn('0.001', result.stdout)
        self.assertIn('8', result.stdout)
        self.assertFalse(report['cpu_golden_used'])

    def test_reference_golden_switch_changes_result_when_devices_agree_but_cpu_does_not(self):
        default, default_report = self.run_pair()
        self.assertEqual(default.returncode, 0, default.stdout)
        self.assertTrue(default_report['passed'])
        self.assertFalse(default_report['cpu_golden_used'])
        self.assertNotIn('reference_vs_golden', default_report)
        result, report = self.run_pair(extra_args=['--compare-reference-golden'])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report['comparison']['passed'])
        self.assertFalse(report['passed'])
        self.assertTrue(report['cpu_golden_used'])
        summary = report['reference_vs_golden']
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['precision_0_001']['failed_count'], len(self.reference_data) // 2)
        self.assertEqual(summary['mismatch_samples'][0]['reference'], 0)
        self.assertEqual(summary['mismatch_samples'][0]['fused'], 0.5)
        self.assertEqual((self.output / 'reference.bin').read_bytes(), self.reference_data)
        self.assertEqual((self.output / 'fused.bin').read_bytes(), self.reference_data)
        for role in ('reference', 'fused'):
            self.assertEqual(report['models'][role]['output_sha256'],
                             hashlib.sha256(self.reference_data).hexdigest())
        self.assertIn('0.001', result.stdout)

    def test_reference_golden_switch_passes_when_all_three_outputs_agree(self):
        cpu_golden = self.tensors['y_expect'][2]
        self.reference_output.write_bytes(cpu_golden)
        self.fused_output.write_bytes(cpu_golden)
        result, report = self.run_pair(extra_args=['--compare-reference-golden'])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report['passed'])
        self.assertTrue(report['comparison']['passed'])
        self.assertTrue(report['reference_vs_golden']['passed'])
        self.assertEqual(report['cpu_golden_sha256'], hashlib.sha256(cpu_golden).hexdigest())
        self.assertEqual(report['reference_vs_golden']['precision_0_001']['pass_percent'], 100)

    def test_legacy_cpu_comparison_prints_actual_device_differences(self):
        blob = bytearray(self.case.read_bytes())
        offset = runner.HDR_LEN
        for _ in range(struct.unpack_from('<I', blob, 12)[0]):
            record = struct.unpack_from(runner.REC_FMT, blob, offset)
            offset += runner.REC_LEN
            if record[0].rstrip(b'\0') == b'y_expect':
                count = record[-1] // 2
                blob[offset:offset + record[-1]] = struct.pack('<%de' % count, *([1] * count))
            offset += record[-1]
        struct.pack_into('<Q', blob, 16, count)
        self.case.write_bytes(blob)
        reference_tests.write_test_binding(self.case, self.fused)
        command = [sys.executable, str(ROOT / 'run_fused_multi_conv2d.py'), str(self.case),
                   'FusedMultiConv2d', '0', str(self.fused)]
        for tolerance, expected_status in (('0.001', 1), ('0.5', 0)):
            with self.subTest(tolerance=tolerance):
                env = dict(self.env, REL_TOL=tolerance)
                result = subprocess.run(command, env=env, cwd=str(self.work), text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, expected_status, result.stdout)
                self.assertIn('relative error <= 0.001', result.stdout)
                self.assertIn('0.5', result.stdout)
                self.assertIn('golden', result.stdout)
                self.assertIn('first 30 mismatches', result.stdout)
                self.assertIn('10 / 40', result.stdout)

    def test_reference_golden_flag_reaches_remote_runner_and_downloaded_report(self):
        remote, paths = self.local_remote(), self.profile_paths()
        with contextlib.redirect_stdout(io.StringIO()):
            profile.do_push(remote, self.case_config, paths, str(self.fused),
                            reference_om=str(self.reference))
            state = profile.do_prof(remote, self.case_config, paths, str(self.fused), False, '',
                                    reference_om=str(self.reference), compare_reference_golden=True)
        self.assertEqual(state['device_rc'], 1)
        self.assertEqual(state['collection_rc'], 0)
        self.assertFalse(state['comparison_passed'])
        report_path = Path(state['comparison_report'])
        report = json.loads(report_path.read_text())
        self.assertTrue(report['cpu_golden_used'])
        self.assertTrue(report['comparison']['passed'])
        self.assertFalse(report['reference_vs_golden']['passed'])
        self.assertEqual((report_path.parent / 'reference.bin').read_bytes(), self.reference_data)

    def test_reference_golden_flag_reaches_local_runner(self):
        with mock.patch.object(fc2d, 'run', return_value=0) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            result = fc2d.do_run(self.case_config, self.profile_paths(), 0, [],
                                  compare_unfused=True, compare_reference_golden=True)
        self.assertEqual(result, 0)
        command = run.call_args[0][0]
        self.assertIn('--compare-reference-golden', command)
        self.assertEqual(command[command.index('--reference-om') + 1], str(self.reference))


class NumericalDiagnosticArguments(unittest.TestCase):
    def test_reference_golden_requires_reference_om(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner._extract_pair_arguments(['--compare-reference-golden'])

    def test_high_level_reference_golden_requires_compare_unfused(self):
        for module, argv in ((fc2d, ['fc2d.py', '/missing.json', '--compare-reference-golden']),
                             (profile, ['run_profile.py', '--cases', '/missing.json',
                                        '--compare-reference-golden', '--no-msprof'])):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stderr(io.StringIO()) as error, \
                    self.assertRaises(SystemExit) as exit_status:
                module.main()
            self.assertEqual(exit_status.exception.code, 2)
            self.assertIn('--compare-unfused', error.getvalue())


if __name__ == '__main__':
    unittest.main()
