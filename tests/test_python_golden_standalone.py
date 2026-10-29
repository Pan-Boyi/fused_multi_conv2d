# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Verify the copied golden script needs only Python, NumPy and its input."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fused_multi_conv2d_golden as golden


ISOLATED_DRIVER = r'''
import builtins
import hashlib
import json
import os
import runpy
import subprocess
import sys
import numpy

repository, action, script, source = sys.argv[1:5]
options = sys.argv[5:]
original_import = builtins.__import__

def isolated_import(name, *args, **kwargs):
    if name.split('.')[0] in ('fc2d', 'run_fused_multi_conv2d'):
        raise AssertionError('Repository module dependency: ' + name)
    return original_import(name, *args, **kwargs)

def forbidden_process(*args, **kwargs):
    raise AssertionError('Standalone golden attempted to run an external process')

builtins.__import__ = isolated_import
for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
    setattr(subprocess, name, forbidden_process)
os.system = os.popen = forbidden_process
if action == 'inspect':
    module = runpy.run_path(script, run_name='isolated_golden')
    info, tensors = module['read_json_case'](source, options[0] if options else None)
    inputs = {name: [dtype, shape, hashlib.sha256(raw).hexdigest()]
              for name, (dtype, shape, raw) in tensors.items() if not name.startswith('y_')}
    print(json.dumps({'info': info, 'inputs': inputs}, sort_keys=True))
else:
    sys.argv = [script, source] + options
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as exc:
        if exc.code:
            raise
for name, module in tuple(sys.modules.items()):
    filename = getattr(module, '__file__', None)
    if filename and os.path.realpath(filename).startswith(repository + os.sep):
        raise AssertionError('Repository import in isolated execution: ' + name)
'''


class PythonGoldenStandalone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle_temp = tempfile.TemporaryDirectory()
        cls.generator = Path(cls.oracle_temp.name) / 'gen_case'
        subprocess.run(['g++', '-std=c++17', '-O2', str(ROOT / 'gen_case.cpp'),
                        '-o', str(cls.generator)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.oracle_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.script = self.work / 'fused_multi_conv2d_golden.py'
        shutil.copyfile(ROOT / self.script.name, self.script)

    def descriptor(self, mode='a16w8', name='standalone'):
        return dict(name=name,
                    input=dict(shape=[2, 32, 7, 9],
                               dtype='fp16' if mode in ('fp16', 'a16w8') else 'int8'),
                    kernels=[dict(shape=[32, 32, 3, 1], dtype='fp16' if mode == 'fp16' else 'int8'),
                             dict(shape=[7, 32, 1, 3], dtype='fp16' if mode == 'fp16' else 'int8')],
                    output=dict(shape=[2, 7, 4, 5], dtype='int8' if mode == 'int8' else 'fp16'),
                    attrs=dict(strides=[2, 2, 1, 1], pads=[1, 1, 0, 0, 0, 0, 1, 1],
                               bias_mask=[True, True], enable_relu=[False, True],
                               fixed_shift_value=[43, 44] if mode == 'fp16' else
                               [17, 0] if mode == 'a16w8' else [0, 0]))

    def oracle_case(self, descriptor, mode):
        batch, channels, height, width = descriptor['input']['shape']
        first, second = [kernel['shape'] for kernel in descriptor['kernels']]
        attrs = descriptor.get('attrs', {})
        strides, pads = attrs.get('strides', [1] * 4), attrs.get('pads', [0] * 8)
        shifts = attrs.get('fixed_shift_value', [42, 42] if mode == 'fp16' else
                           [13, 0] if mode == 'a16w8' else [0, 0])
        relu, mask = attrs.get('enable_relu', [True, True]), attrs.get('bias_mask', [False, False])
        values = dict(dtype=mode, n=batch, ci=channels, hi=height, wi=width,
                      cout1=first[0], cout2=second[0], kh=first[2], kw=first[3],
                      kh2=second[2], kw2=second[3], s1=strides[0], s2=strides[2],
                      ph1=pads[0], pw1=pads[2], ph2=pads[4], pw2=pads[6],
                      bias=int(any(mask)), relu1=int(relu[0]), relu2=int(relu[1]),
                      shift1=shifts[0], shift2=shifts[1])
        path = Path(self.oracle_temp.name) / (descriptor['name'] + '.bin')
        command = [str(self.generator), str(path)]
        for key, value in values.items():
            command.extend(['--' + key, str(value)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return path, golden.read_case(str(path))

    def isolated(self, source, *options, action='cli'):
        environment = dict(os.environ, PATH='', OPENBLAS_NUM_THREADS='1')
        environment.pop('PYTHONPATH', None)
        return subprocess.run([sys.executable, '-I', '-B', '-c', ISOLATED_DRIVER,
                               str(ROOT.resolve()), action, str(self.script), str(source), *options],
                              cwd=str(self.work), env=environment, text=True, capture_output=True)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_isolated_json_inputs_scales_and_outputs_match_cpp_for_four_paths(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            with self.subTest(mode=mode):
                descriptor = self.descriptor(mode)
                source = self.work / 'case.json'
                source.write_text(json.dumps(descriptor))
                _, (info, tensors) = self.oracle_case(descriptor, mode)
                inspected = self.isolated(source, action='inspect')
                self.assert_success(inspected)
                expected_inputs = {name: [dtype, shape, hashlib.sha256(raw).hexdigest()]
                                   for name, (dtype, shape, raw) in tensors.items()
                                   if not name.startswith('y_')}
                self.assertEqual(json.loads(inspected.stdout), {'info': info, 'inputs': expected_inputs})
                output, exact = self.work / 'output.bin', self.work / 'exact.bin'
                options = ['--output', str(output), '--workspace-mb', '0.01']
                if mode == 'fp16':
                    options.extend(['--exact-output', str(exact)])
                self.assert_success(self.isolated(source, *options))
                self.assertEqual(output.read_bytes(), tensors['y_expect'][2])
                if mode == 'fp16':
                    self.assertEqual(exact.read_bytes(), tensors['y_exact'][2])

    def test_checked_in_tensor_json_runs_when_copied_without_harness(self):
        source = self.work / 'golden_tensor_cases.json'
        shutil.copyfile(ROOT / source.name, source)
        original = source.read_bytes()
        descriptor = json.loads(original)['cases'][0]
        _, (_, tensors) = self.oracle_case(descriptor, 'a16w8')
        output = self.work / 'output.bin'
        self.assert_success(self.isolated(source, '--output', str(output)))
        self.assertEqual(output.read_bytes(), tensors['y_expect'][2])
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(set(path.name for path in self.work.iterdir()),
                         {self.script.name, source.name, output.name})

    def test_isolated_binary_entry_matches_cpp_output_without_json_generator(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            with self.subTest(mode=mode):
                fixture, (_, tensors) = self.oracle_case(self.descriptor(mode), mode)
                source = self.work / 'case.bin'
                shutil.copyfile(fixture, source)
                output = self.work / 'output.bin'
                self.assert_success(self.isolated(source, '--output', str(output)))
                self.assertEqual(output.read_bytes(), tensors['y_expect'][2])

    def test_isolated_case_selection_and_errors_do_not_create_output(self):
        first, second = self.descriptor('fp16', 'first'), self.descriptor('a16w8', 'second')
        source, output = self.work / 'cases.json', self.work / 'output.bin'
        source.write_text(json.dumps({'cases': [first, second]}))
        for options, error in (([], '--only'), (['--only', 'missing'], 'missing')):
            result = self.isolated(source, '--output', str(output), *options)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stdout + result.stderr)
            self.assertFalse(output.exists())
        self.assert_success(self.isolated(source, '--output', str(output), '--only', 'second'))
        _, (_, tensors) = self.oracle_case(second, 'a16w8')
        self.assertEqual(output.read_bytes(), tensors['y_expect'][2])
        output.unlink()
        second['attrs']['strides'] = [1, 2, 1, 1]
        source.write_text(json.dumps(second))
        result = self.isolated(source, '--output', str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('strides', result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_script_remains_python37_syntax_compatible(self):
        ast.parse(self.script.read_text(), feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
