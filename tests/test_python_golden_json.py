# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Check standalone JSON input without changing the model validation workflow."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fused_multi_conv2d_golden as golden


class PythonGoldenJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def spec(self, dtype='a16w8', name='json_case'):
        return dict(name=name, dtype=dtype, n=1, ci=32, hi=5, wi=7,
                    cout1=32, cout2=7, kernel=[3, 3, 1, 1],
                    strides=[2, 2, 1, 1], pads=[1, 1, 1, 1, 0, 0, 0, 0],
                    bias=True, enable_relu=[False, True])

    def write_json(self, doc, name='cases.json'):
        path = self.work / name
        path.write_text(json.dumps(doc))
        return path

    def cpp_case(self, spec, name='case.bin'):
        config = fc2d.normalise(spec, 0)
        path = self.work / name
        subprocess.run(fc2d.gen_case_argv(config, str(path)), check=True,
                       stdout=subprocess.DEVNULL)
        return path, golden.read_case(str(path))[1]

    def cli(self, source, output, *options):
        return subprocess.run([sys.executable, str(ROOT / 'fused_multi_conv2d_golden.py'),
                               str(source), '--output', str(output), *options],
                              cwd=str(self.work), text=True, capture_output=True)

    def test_json_and_binary_cli_match_cpp_for_all_four_paths(self):
        for dtype in ('fp16', 'int8', 's8f16', 'a16w8'):
            with self.subTest(dtype=dtype):
                spec = self.spec(dtype)
                source = self.write_json(spec, dtype + '.json')
                case, tensors = self.cpp_case(spec, dtype + '.bin')
                json_output = self.work / (dtype + '_json_output.bin')
                binary_output = self.work / (dtype + '_binary_output.bin')
                json_exact = self.work / (dtype + '_json_exact.bin')
                binary_exact = self.work / (dtype + '_binary_exact.bin')
                json_options = ('--exact-output', str(json_exact)) if dtype == 'fp16' else ()
                binary_options = ('--exact-output', str(binary_exact)) if dtype == 'fp16' else ()
                json_result = self.cli(source, json_output, '--workspace-mb', '0.01', *json_options)
                binary_result = self.cli(case, binary_output, '--workspace-mb', '0.01', *binary_options)
                self.assertEqual(json_result.returncode, 0, json_result.stdout + json_result.stderr)
                self.assertEqual(binary_result.returncode, 0, binary_result.stdout + binary_result.stderr)
                self.assertEqual(json_output.read_bytes(), tensors['y_expect'][2])
                self.assertEqual(binary_output.read_bytes(), json_output.read_bytes())
                if dtype == 'fp16':
                    self.assertEqual(json_exact.read_bytes(), tensors['y_exact'][2])
                    self.assertEqual(binary_exact.read_bytes(), json_exact.read_bytes())

    def test_single_case_list_and_cases_wrapper(self):
        spec = self.spec('int8')
        _, tensors = self.cpp_case(spec)
        for index, doc in enumerate((spec, [spec], {'cases': [spec]})):
            with self.subTest(wrapper=index):
                source = self.write_json(doc, 'wrapper_%d.json' % index)
                output = self.work / ('wrapper_%d.bin' % index)
                result = self.cli(source, output)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), tensors['y_expect'][2])

    def test_multiple_cases_require_a_single_explicit_selection(self):
        first, second = self.spec('fp16', 'first'), self.spec('a16w8', 'second')
        source = self.write_json({'cases': [first, second]})
        output = self.work / 'selected.bin'
        missing = self.cli(source, output)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('--only', missing.stdout + missing.stderr)
        self.assertFalse(output.exists())
        unknown = self.cli(source, output, '--only', 'unknown')
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn('unknown', unknown.stdout + unknown.stderr)
        self.assertFalse(output.exists())
        selected = self.cli(source, output, '--only', 'second')
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        _, tensors = self.cpp_case(second)
        self.assertEqual(output.read_bytes(), tensors['y_expect'][2])

    def test_both_expands_to_the_existing_harness_names(self):
        spec = self.spec('both', 'pair')
        source = self.write_json(spec)
        output = self.work / 'both_output.bin'
        missing = self.cli(source, output)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('--only', missing.stdout + missing.stderr)
        for dtype in ('fp16', 'int8'):
            with self.subTest(dtype=dtype):
                result = self.cli(source, output, '--only', 'pair_' + dtype)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                expanded = fc2d.normalise(spec, 0)
                expanded.update(dtype=dtype, name='pair_' + dtype)
                _, tensors = self.cpp_case(expanded)
                self.assertEqual(output.read_bytes(), tensors['y_expect'][2])

    def test_json_preserves_existing_inputs_and_does_not_enter_validation_steps(self):
        spec = self.spec('a16w8')
        source = self.write_json(spec)
        out_dir = self.work / 'out' / 'fusion_cases' / spec['name']
        out_dir.mkdir(parents=True)
        old_case, old_model = out_dir / 'case.bin', out_dir / 'model.om'
        old_case.write_bytes(b'keep existing compiled case')
        old_model.write_bytes(b'keep existing model')
        output = self.work / 'standalone_output.bin'
        before = {path: path.read_bytes() for path in (source, old_case, old_model)}
        with mock.patch.object(fc2d, 'do_check', side_effect=AssertionError('check step called')), \
                mock.patch.object(fc2d, 'do_case', side_effect=AssertionError('case step called')), \
                mock.patch.object(fc2d, 'do_om', side_effect=AssertionError('OM step called')), \
                mock.patch.object(fc2d, 'do_run', side_effect=AssertionError('run step called')), \
                mock.patch.object(fc2d, 'run', side_effect=AssertionError('workflow run called')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(golden.main([str(source), '--output', str(output)]), 0)
        self.assertGreater(output.stat().st_size, 0)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(set(out_dir.iterdir()), {old_case, old_model})
        self.assertEqual({path for path in self.work.iterdir()}, {source, output, self.work / 'out'})

    def test_invalid_json_case_does_not_generate_output(self):
        spec = self.spec()
        spec['unexpected_option'] = True
        source = self.write_json(spec)
        original = source.read_bytes()
        output = self.work / 'invalid_output.bin'
        result = self.cli(source, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unexpected_option', result.stdout + result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
