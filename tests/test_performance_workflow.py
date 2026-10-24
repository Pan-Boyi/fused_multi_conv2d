# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Validate paired device profiling orchestration independently of device timings."""
import contextlib
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import run_profile as profile
import test_reference_compare as reference_tests


class PerformanceArguments(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.config = self.work / 'performance.json'
        self.document = dict(soc_version='MC62CM12AA', outdir=str(self.work / 'out'),
                             profdir=str(self.work / 'prof_out'),
                             cases=[dict(name='performance_case', dtype='a16w8', hi=8,
                                         wi=9, cout2=2, bias=True)])
        self.config.write_text(json.dumps(self.document))

    def invoke(self, *arguments):
        with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config)] + list(arguments)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = profile.main()
        return result, output.getvalue()

    def test_performance_mode_builds_the_same_reference_pair_as_precision_mode(self):
        with mock.patch.object(fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'source_env'):
            result, output = self.invoke('--compare-performance', '--steps', 'om')
        self.assertEqual(result, 0, output)
        self.assertEqual(compile_model.call_count, 1)
        self.assertTrue(compile_model.call_args.kwargs['compare_unfused'])

    def test_performance_mode_accepts_optional_reference_cpu_comparison(self):
        with mock.patch.object(fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'source_env'):
            result, output = self.invoke('--compare-performance', '--compare-reference-golden',
                                         '--steps', 'om')
        self.assertEqual(result, 0, output)
        self.assertTrue(compile_model.call_args.kwargs['compare_unfused'])

    def test_performance_mode_rejects_no_msprof_before_build_or_remote_work(self):
        with mock.patch.object(fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'Remote') as remote, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.invoke('--compare-performance', '--no-msprof', '--steps', 'om')
        self.assertEqual(error.exception.code, 2)
        compile_model.assert_not_called()
        remote.assert_not_called()

    def test_performance_mode_rejects_disabled_profile_configuration(self):
        self.document['profile'] = dict(enabled=False)
        self.config.write_text(json.dumps(self.document))
        with mock.patch.object(fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'Remote') as remote, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke('--compare-performance', '--steps', 'om')
        compile_model.assert_not_called()
        remote.assert_not_called()


class PerformanceDeviceWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference_tests.ReferenceComparison.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        reference_tests.ReferenceComparison.tearDownClass.__func__(cls)

    def setUp(self):
        self.outer = tempfile.TemporaryDirectory()
        self.addCleanup(self.outer.cleanup)
        case_directory = tempfile.TemporaryDirectory(dir=self.outer.name)
        self.fixture = reference_tests.ReferenceComparison(
            'test_device_outputs_are_independent_of_cpu_golden_and_both_execute')
        self.fixture.stub = self.stub
        with mock.patch.object(reference_tests.tempfile, 'TemporaryDirectory',
                               return_value=case_directory):
            self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.work = self.fixture.work
        fused_sidecar = Path(str(self.fixture.fused) + '.fc2d.json')
        fused_binding = json.loads(fused_sidecar.read_text())
        fused_binding['fusion']['fused_node_name'] = 'fused'
        fused_sidecar.write_text(json.dumps(fused_binding))
        reference_binding = json.loads(self.fixture.reference_sidecar.read_text())
        reference_binding['reference']['stage1_nodes'] = ['extend1', 'extend2']
        self.fixture.reference_sidecar.write_text(json.dumps(reference_binding))
        self.case = dict(self.fixture.case_config, name=self.work.name)
        self.remote = self.fixture.local_remote()
        self.remote.repeat, self.remote.warmup = 3, 1
        self.msprof = self.work / 'fake_msprof'
        self.msprof.write_text('#!/bin/bash\n"$@"\n'
                               'mkdir "PROF_$(python3 -c "import uuid; print(uuid.uuid4().hex)")"\n'
                               'exit 0\n')
        self.msprof.chmod(0o755)
        self.suite = self.work / 'suite'
        exporter = self.suite / 'analysis/msprof/msprof.py'
        exporter.parent.mkdir(parents=True)
        exporter.write_text(
            'import pathlib,sys\n'
            "directory=pathlib.Path(sys.argv[-1])/'mindstudio_profiler_output'\n"
            'directory.mkdir(exist_ok=True)\n'
            "rows=['OP Type,Op Name,total_exe_time(us),Task Start Time(us)']\n"
            'timestamp=0\n'
            'for first,second in zip([1000,6,8,4,999],[1000,4,2,7,999]):\n'
            '    for name,duration in [("extend1",first),("extend2",second)]:\n'
            '        timestamp+=1\n'
            '        rows.append("ExtendConv2D,%s,%s,%s"%(name,duration,timestamp))\n'
            'for duration in [1000,5,6,4,999]:\n'
            '    timestamp+=1\n'
            '    rows.append("FusedMultiConv2d,fused,%s,%s"%(duration,timestamp))\n'
            "(directory/'op_summary_test.csv').write_text('\\n'.join(rows)+'\\n')\n")
        self.config = self.work / 'performance.json'
        self.config.write_text(json.dumps(dict(
            outdir=str(self.work.parent), profdir=str(self.work / 'prof_out'),
            profile=dict(msprof=str(self.msprof), tbe_work_test_suit=str(self.suite)),
            cases=[self.case])))

    def invoke(self, steps):
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config),
                                            '--compare-performance', '--steps', steps]), \
                mock.patch.object(profile, 'Remote', return_value=self.remote), \
                contextlib.redirect_stdout(output):
            result = profile.main()
        return result, output.getvalue()

    def test_same_profile_executes_both_models_and_excludes_warmup_and_probe(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        state = profile.load_profile_state(self.fixture.profile_paths(), str(self.fixture.fused))
        self.assertTrue(state['comparison_passed'])
        self.assertEqual(state['device_rc'], 0)
        trace = [line.split()[1] for line in self.fixture.trace.read_text().splitlines()]
        self.assertEqual(trace, ['reference'] * 5 + ['fused'] * 5)
        self.assertIn('50.00%', output)
        self.assertEqual(len(self.remote.uploads), 6)
        self.assertEqual(len(self.remote.downloads), 2)
        self.assertEqual(state['performance']['fused_us'], 5)
        self.assertEqual(state['performance']['extend_total_us'], 10)
        self.assertEqual(state['performance']['samples']['extend_total_us'], [10, 10, 11])
        summary = json.loads((self.work.parent / 'performance_comparison.json').read_text())
        self.assertEqual(summary['units'], 'us')
        self.assertEqual(len(summary['cases']), 1)
        self.assertEqual(summary['cases'][0]['case'], self.case['name'])
        self.assertEqual(summary['cases'][0]['improvement_percent'], 50)
        with (self.work.parent / 'performance_comparison.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]['saved_us']), 5)
        self.assertEqual(float(rows[0]['improvement_percent']), 50)

    def test_parse_only_resumes_the_same_validated_device_execution(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        before = self.fixture.trace.read_text()
        with mock.patch.object(profile, 'Remote', side_effect=AssertionError('local parse used remote')):
            output = io.StringIO()
            with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.config),
                                                '--compare-performance', '--steps', 'parse']), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(profile.main(), 0, output.getvalue())
        self.assertEqual(self.fixture.trace.read_text(), before)
        self.assertIn('50.00%', output.getvalue())

    def test_parse_uses_bound_case_shape_when_configuration_has_changed(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        paths = self.fixture.profile_paths()
        original = profile.load_profile_state(paths, str(self.fixture.fused))['performance']
        document = json.loads(self.config.read_text())
        document['cases'][0].update(hi=16, wi=17)
        self.config.write_text(json.dumps(document))
        result, output = self.invoke('parse')
        self.assertEqual(result, 0, output)
        current = profile.load_profile_state(paths, str(self.fixture.fused))['performance']
        self.assertEqual(current['shape'], original['shape'])
        self.assertEqual(current['shape'], 'a16w8 1x32x8x9 -> 1x64x6x7 -> 1x2x4x5')
        self.assertEqual(current['case_spec'], fc2d.read_spec(str(self.fixture.case)))
        self.assertEqual((current['case_spec']['hi'], current['case_spec']['wi']), (8, 9))

    def test_parse_uses_captured_counts_when_remote_configuration_has_changed(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        before = self.fixture.trace.read_text()
        document = json.loads(self.config.read_text())
        document['remote'] = dict(warmup=9, repeat=7)
        self.config.write_text(json.dumps(document))
        self.remote.warmup, self.remote.repeat = 9, 7
        result, output = self.invoke('parse')
        self.assertEqual(result, 0, output)
        state = profile.load_profile_state(self.fixture.profile_paths(), str(self.fixture.fused))
        self.assertEqual(state['profiling_execution'],
                         dict(warmup=1, repeat=3, execution_count_per_model=5))
        self.assertEqual((state['performance']['warmup'], state['performance']['repeat']), (1, 3))
        self.assertEqual(state['performance']['samples']['extend_total_us'], [10, 10, 11])
        self.assertEqual(self.fixture.trace.read_text(), before)

    def test_numerical_failure_never_enters_the_performance_table(self):
        self.fixture.fused_output.write_bytes(struct.pack('<e', 8) + self.fixture.reference_data[2:])
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 1, output)
        state = profile.load_profile_state(self.fixture.profile_paths(), str(self.fixture.fused))
        self.assertFalse(state['comparison_passed'])
        self.assertNotIn('50.00%', output)
        summary = json.loads((self.work.parent / 'performance_comparison.json').read_text())
        self.assertEqual(summary['cases'], [])

    def test_parse_rejects_tampered_execution_counts(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        paths = self.fixture.profile_paths()
        state = profile.load_profile_state(paths, str(self.fixture.fused))
        state['profiling_execution']['repeat'] = 4
        profile.save_profile_state(paths, state)
        result, output = self.invoke('parse')
        self.assertEqual(result, 1, output)
        self.assertNotIn('50.00%', output)
        summary = json.loads((self.work.parent / 'performance_comparison.json').read_text())
        self.assertEqual(summary['cases'], [])


if __name__ == '__main__':
    unittest.main()
