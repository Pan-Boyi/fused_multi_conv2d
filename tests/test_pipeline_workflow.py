# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""验证流水明细沿用同一组设备执行、节点绑定和性能结果。"""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import test_performance_workflow as workflow

profile = workflow.profile


class PipelineArguments(unittest.TestCase):
    def setUp(self):
        self.fixture = workflow.PerformanceArguments(
            'test_performance_mode_builds_the_same_reference_pair_as_precision_mode')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_pipeline_mode_compiles_the_paired_reference_without_extra_flag(self):
        with mock.patch.object(workflow.fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'source_env'):
            result, output = self.fixture.invoke('--pipeline-details', '--steps', 'om')
        self.assertEqual(result, 0, output)
        self.assertEqual(compile_model.call_count, 1)
        self.assertTrue(compile_model.call_args.kwargs['compare_unfused'])

    def test_pipeline_mode_rejects_no_msprof_before_build_or_remote_work(self):
        with mock.patch.object(workflow.fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'Remote') as remote, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.fixture.invoke('--pipeline-details', '--no-msprof', '--steps', 'om')
        self.assertEqual(error.exception.code, 2)
        compile_model.assert_not_called()
        remote.assert_not_called()

    def test_pipeline_mode_rejects_disabled_profile_before_build(self):
        self.fixture.document['profile'] = dict(enabled=False)
        self.fixture.config.write_text(json.dumps(self.fixture.document))
        with mock.patch.object(workflow.fc2d, 'do_om') as compile_model, \
                mock.patch.object(profile, 'Remote') as remote, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.fixture.invoke('--pipeline-details', '--steps', 'om')
        self.assertEqual(error.exception.code, 2)
        compile_model.assert_not_called()
        remote.assert_not_called()


class PipelineDeviceWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow.PerformanceDeviceWorkflow.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        workflow.PerformanceDeviceWorkflow.tearDownClass.__func__(cls)

    def setUp(self):
        self.fixture = workflow.PerformanceDeviceWorkflow(
            'test_same_profile_executes_both_models_and_excludes_warmup_and_probe')
        self.fixture.stub = self.stub
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        exporter = self.fixture.suite / 'analysis/msprof/msprof.py'
        exporter.write_text(
            'import csv,pathlib,sys\n'
            "directory=pathlib.Path(sys.argv[-1])/'mindstudio_profiler_output'\n"
            'directory.mkdir(exist_ok=True)\n'
            'columns=%r\n' % (['OP Type', 'Op Name'] + profile.DEFAULT_COLUMNS +
                              ['Task Start Time(us)']) +
            'rows=[]\n'
            'timestamp=0\n'
            'def add(name,duration,iteration):\n'
            '    global timestamp\n'
            '    timestamp+=1\n'
            '    row={"OP Type":"FusedMultiConv2d" if name=="fused" else "ExtendConv2D",\n'
            '         "Op Name":name,"total_exe_time(us)":duration,\n'
            '         "Task Start Time(us)":timestamp}\n'
            '    cube={"fused":[1000,1,2,3,999],\n'
            '          "extend1":[1000,1,1,9,999],\n'
            '          "extend2":[1000,9,1,1,999]}\n'
            '    row["mac_exe_time(us)"]=cube[name][iteration]\n'
            '    for factor,column in enumerate(columns[4:-1],2):\n'
            '        measured=iteration if name=="fused" else iteration+(3 if name=="extend2" else 0)\n'
            '        scale=0.1 if name=="fused" else 0.01\n'
            '        row[column]=1000 if iteration in (0,4) else measured*factor*scale\n'
            '    rows.append(row)\n'
            'for iteration,(first,second) in enumerate(zip([1000,6,8,4,999],[1000,4,2,7,999])):\n'
            '    add("extend1",first,iteration)\n'
            '    add("extend2",second,iteration)\n'
            'for iteration,duration in enumerate([1000,5,6,4,999]):\n'
            '    add("fused",duration,iteration)\n'
            "with (directory/'op_summary_test.csv').open('w',newline='') as stream:\n"
            '    writer=csv.DictWriter(stream,fieldnames=columns)\n'
            '    writer.writeheader()\n'
            '    writer.writerows(rows)\n')

    def invoke(self, steps, pipeline=True):
        flags = ['--pipeline-details'] if pipeline else ['--compare-performance']
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run_profile.py', str(self.fixture.config)] +
                               flags + ['--steps', steps]), \
                mock.patch.object(profile, 'Remote', return_value=self.fixture.remote), \
                contextlib.redirect_stdout(output):
            result = profile.main()
        return result, output.getvalue()

    def performance(self):
        state = profile.load_profile_state(self.fixture.fixture.profile_paths(),
                                           str(self.fixture.fixture.fused))
        return state['performance']

    def test_prints_and_persists_each_pipeline_for_the_same_paired_execution(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        measured = self.performance()
        details = measured['pipeline_details']
        self.assertEqual(details['units'], 'us')
        self.assertEqual(details['method'], 'median_of_per_execution_pipeline_durations')
        self.assertEqual(details['columns']['cube'], 'mac_exe_time(us)')
        self.assertEqual(details['columns']['fixpipe'], 'fixpipe_time(us)')
        operators = details['operators']
        labels = ['FusedMultiConv2d', 'ExtendConv2D[1]', 'ExtendConv2D[2]', '2 x ExtendConv2D']
        self.assertEqual([row['label'] for row in operators], labels)
        self.assertEqual([row['name'] for row in operators],
                         ['fused', 'extend1', 'extend2', ['extend1', 'extend2']])
        self.assertEqual([row['total_us'] for row in operators], [5, 6, 4, 10])
        self.assertEqual([row['pipeline_us']['cube'] for row in operators], [2, 1, 1, 10])
        self.assertEqual(operators[0]['samples']['cube'], [1, 2, 3])
        self.assertEqual(operators[3]['samples']['cube'], [10, 2, 10])
        for label in labels + ['Cube', 'Scalar', 'MTE1', 'MTE2', 'MTE3', 'Fixpipe']:
            self.assertIn(label, output)
        trace = [line.split()[1] for line in self.fixture.fixture.trace.read_text().splitlines()]
        self.assertEqual(trace, ['reference'] * 5 + ['fused'] * 5)
        summary_path = self.fixture.work.parent / 'performance_comparison.json'
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary['cases'][0]['pipeline_details'], details)
        with (self.fixture.work.parent / 'performance_comparison.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]['fused_us']), 5)
        self.assertEqual(float(rows[0]['extend_total_us']), 10)
        self.assertEqual(float(rows[0]['improvement_percent']), 50)

    def test_parse_only_adds_pipeline_details_without_executing_models_again(self):
        result, output = self.invoke('push,prof,pull,export,parse', pipeline=False)
        self.assertEqual(result, 0, output)
        self.assertNotIn('pipeline_details', self.performance())
        trace = self.fixture.fixture.trace.read_text()
        output = io.StringIO()
        with mock.patch.object(profile, 'Remote', side_effect=AssertionError('local parse used remote')), \
                mock.patch.object(sys, 'argv', ['run_profile.py', str(self.fixture.config),
                                               '--pipeline-details', '--steps', 'parse']), \
                contextlib.redirect_stdout(output):
            result = profile.main()
        self.assertEqual(result, 0, output.getvalue())
        self.assertEqual(self.fixture.fixture.trace.read_text(), trace)
        self.assertEqual(self.performance()['pipeline_details']['operators'][0]['pipeline_us']['cube'], 2)
        self.assertIn('Fixpipe', output.getvalue())

    def test_uncollected_pipeline_values_print_na_without_invalidating_total(self):
        result, output = self.invoke('push,prof,pull,export,parse')
        self.assertEqual(result, 0, output)
        state = profile.load_profile_state(self.fixture.fixture.profile_paths(),
                                           str(self.fixture.fixture.fused))
        csv_path = Path(profile.find_op_summary(state['local_prof']))
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.pop('mte3_exe_time(us)')
            row['fixpipe_time(us)'] = 'N/A'
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result, output = self.invoke('parse')
        self.assertEqual(result, 0, output)
        details = self.performance()['pipeline_details']
        self.assertIsNone(details['columns']['mte3'])
        for row in details['operators']:
            self.assertIsNone(row['pipeline_us']['mte3'])
            self.assertIsNone(row['pipeline_us']['fixpipe'])
        self.assertIn('N/A', output)
        self.assertEqual(self.performance()['extend_total_us'], 10)


if __name__ == '__main__':
    unittest.main()
