# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Validate tensor-oriented JSON against the existing C++ reference."""
import contextlib
import copy
import ctypes
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
import test_python_golden as cpp_reference


class PythonGoldenTensorJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            fc2d.ensure_tool('gen_case')
        cls.oracle_work = tempfile.TemporaryDirectory()
        work = Path(cls.oracle_work.name)
        source = work / 'oracle.cpp'
        source.write_text(cpp_reference.ORACLE_SOURCE)
        library = work / 'oracle.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I', str(ROOT),
                        str(source), '-o', str(library)], check=True)
        cls.oracle = ctypes.CDLL(str(library))
        cls.oracle.Evaluate.argtypes = [ctypes.POINTER(ctypes.c_int)] + [ctypes.c_void_p] * 9
        cls.oracle.Evaluate.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.oracle_work.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def descriptor(self, mode='a16w8', variant=0, name='tensor_case'):
        input_dtype = 'fp16' if mode in ('fp16', 'a16w8') else 'int8'
        kernel_dtype = 'fp16' if mode == 'fp16' else 'int8'
        output_dtype = 'int8' if mode == 'int8' else 'fp16'
        first_kernel, second_kernel = (([3, 1], [1, 3]) if variant else ([3, 3], [3, 3]))
        attrs = dict(strides=[2, 2, 1, 1] if variant else [1, 1, 1, 1],
                     pads=[1, 1, 0, 0, 0, 0, 1, 1] if variant else [1] * 8,
                     enable_relu=[False, True], bias_mask=[True, True])
        return dict(name=name, input=dict(shape=[2 if variant else 1, 32, 7, 9], dtype=input_dtype),
                    kernels=[dict(shape=[32, 32] + first_kernel, dtype=kernel_dtype),
                             dict(shape=[7, 32] + second_kernel, dtype=kernel_dtype)],
                    output=dict(dtype=output_dtype), attrs=attrs)

    def flat_spec(self, descriptor, mode):
        batch, channels, height, width = descriptor['input']['shape']
        filters = descriptor['kernels']
        attrs = descriptor.get('attrs', {})
        flat = dict(name=descriptor['name'], dtype=mode, n=batch, ci=channels,
                    hi=height, wi=width, cout1=filters[0]['shape'][0], cout2=filters[1]['shape'][0],
                    kernel=filters[0]['shape'][2:] + filters[1]['shape'][2:],
                    strides=attrs.get('strides', [1] * 4), pads=attrs.get('pads', [0] * 8),
                    enable_relu=attrs.get('enable_relu', [True, True]),
                    bias=any(attrs.get('bias_mask', [False, False])))
        if 'fixed_shift_value' in attrs:
            flat['fixed_shift_value'] = attrs['fixed_shift_value']
        return flat

    def write_json(self, value, name='case.json'):
        path = self.work / name
        path.write_text(json.dumps(value))
        return path

    def read_descriptor(self, descriptor, only=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return golden.read_json_case(str(self.write_json(descriptor)), only)

    def cpp_case(self, flat):
        path = self.work / 'cpp_case.bin'
        config = fc2d.normalise(flat, 0)
        subprocess.run(fc2d.gen_case_argv(config, str(path)), check=True,
                       stdout=subprocess.DEVNULL)
        return golden.read_case(str(path))

    def assert_outputs_equal(self, actual, expected):
        cpp_reference.PythonGolden.assert_outputs_equal(self, actual, expected)

    def test_four_paths_and_geometry_match_legacy_json_and_cpp(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            for variant in (0, 1):
                with self.subTest(mode=mode, variant=variant):
                    descriptor = self.descriptor(mode, variant)
                    info, tensors = self.read_descriptor(descriptor)
                    cpp_info, cpp_tensors = self.cpp_case(self.flat_spec(descriptor, mode))
                    self.assertEqual(info, cpp_info)
                    for name in ('x', 'filter1', 'filter2', 'bias1', 'bias2',
                                 'quant_scale1', 'quant_scale2', 'dequant_scale2'):
                        if name in cpp_tensors:
                            self.assertEqual(tensors[name], cpp_tensors[name], name)
                    expected = {name: cpp_tensors[name] for name in ('y_expect', 'y_exact')
                                if name in cpp_tensors}
                    self.assert_outputs_equal(golden.compute_golden(info, tensors), expected)
                    legacy_info, legacy_tensors = self.read_descriptor(self.flat_spec(descriptor, mode))
                    self.assert_outputs_equal(golden.compute_golden(legacy_info, legacy_tensors), expected)

    def test_partial_bias_masks_and_each_relu_match_cpp_oracle(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            for mask, relu in (([True, False], [False, True]),
                               ([False, True], [True, False])):
                with self.subTest(mode=mode, mask=mask, relu=relu):
                    descriptor = self.descriptor(mode, variant=1)
                    descriptor['attrs'].update(bias_mask=mask, enable_relu=relu)
                    info, tensors = self.read_descriptor(descriptor)
                    _, all_bias_tensors = self.cpp_case(self.flat_spec(descriptor, mode))
                    self.assertEqual(info['bias'], 1)
                    self.assertEqual([info['relu1'], info['relu2']], [int(value) for value in relu])
                    for layer, enabled in enumerate(mask, 1):
                        tensor = tensors['bias%d' % layer]
                        if enabled:
                            self.assertEqual(tensor, all_bias_tensors['bias%d' % layer])
                        else:
                            self.assertEqual(tensor[2], bytes(len(tensor[2])))
                    expected = cpp_reference.PythonGolden.cpp_output(self, info, tensors)
                    self.assert_outputs_equal(golden.compute_golden(info, tensors), expected)

    def test_omitted_attrs_retain_existing_mode_defaults(self):
        for mode, shift in (('fp16', [42, 42]), ('int8', [0, 0]),
                            ('s8f16', [0, 0]), ('a16w8', [13, 0])):
            with self.subTest(mode=mode):
                descriptor = self.descriptor(mode)
                descriptor.pop('attrs')
                info, tensors = self.read_descriptor(descriptor)
                self.assertEqual([info['shift1'], info['shift2']], shift)
                self.assertEqual([info['relu1'], info['relu2']], [1, 1])
                self.assertEqual(info['bias'], 0)
                cpp_info, cpp_tensors = self.cpp_case(self.flat_spec(descriptor, mode))
                self.assertEqual(info, cpp_info)
                self.assert_outputs_equal(golden.compute_golden(info, tensors),
                                          {name: cpp_tensors[name] for name in ('y_expect', 'y_exact')
                                           if name in cpp_tensors})

    def test_supported_default_only_attrs_match_omitted_defaults(self):
        descriptor = self.descriptor('a16w8')
        base_info, base_tensors = self.read_descriptor(descriptor)
        defaults = dict(dilations=[1] * 8, groups=[1, 1], offset_x=[0, 0],
                        enable_hf32=[False, False])
        for enum_defaults in (dict(data_format=[0, 0], round_mode=[0, 0], pad_mode=[0, 0]),
                              dict(data_format=['NCHW', 'nchw'], round_mode=['rint', 'RINT'],
                                   pad_mode=['SPECIFIC', 'specific'])):
            with self.subTest(enums=enum_defaults):
                configured = copy.deepcopy(descriptor)
                configured['attrs'].update(defaults, **enum_defaults)
                info, tensors = self.read_descriptor(configured)
                self.assertEqual(info, base_info)
                self.assertEqual(tensors, base_tensors)

    def test_float16_dtype_alias_and_output_shape(self):
        descriptor = self.descriptor('fp16', variant=1)
        expected_info, expected_tensors = self.read_descriptor(descriptor)
        descriptor['input']['dtype'] = 'float16'
        for tensor in descriptor['kernels']:
            tensor['dtype'] = 'float16'
        descriptor['output'].update(dtype='float16', shape=[2, 7, 4, 5])
        info, tensors = self.read_descriptor(descriptor)
        self.assertEqual(info, expected_info)
        self.assertEqual(tensors, expected_tensors)

    def test_explicit_shift_values_and_integer_relu(self):
        for mode, shift in (('fp16', [43, 44]), ('int8', [0, 58]),
                            ('s8f16', [7, 17]), ('a16w8', [29, 58])):
            with self.subTest(mode=mode):
                descriptor = self.descriptor(mode)
                descriptor['attrs'].update(fixed_shift_value=shift, enable_relu=[1, 0])
                info, tensors = self.read_descriptor(descriptor)
                self.assertEqual([info['shift1'], info['shift2']], shift)
                self.assertEqual([info['relu1'], info['relu2']], [1, 0])
                expected = cpp_reference.PythonGolden.cpp_output(self, info, tensors)
                self.assert_outputs_equal(golden.compute_golden(info, tensors), expected)

    def test_case_wrappers_and_selection_accept_tensor_descriptors(self):
        first = self.descriptor('int8', name='first')
        second = self.descriptor('a16w8', variant=1, name='second')
        expected = self.read_descriptor(second)
        for document in ([second], {'cases': [second]}, [first, second], {'cases': [first, second]}):
            with self.subTest(document_type=type(document).__name__, count=len(document)):
                self.assertEqual(self.read_descriptor(document, 'second'), expected)

    def test_cli_exports_tensor_descriptor_reference_without_workflow(self):
        descriptor = self.descriptor('fp16', variant=1)
        source = self.write_json(descriptor)
        original = source.read_bytes()
        output, exact = self.work / 'output.bin', self.work / 'exact.bin'
        _, tensors = self.cpp_case(self.flat_spec(descriptor, 'fp16'))
        with mock.patch.object(fc2d, 'do_check', side_effect=AssertionError('check step called')), \
                mock.patch.object(fc2d, 'do_case', side_effect=AssertionError('case step called')), \
                mock.patch.object(fc2d, 'do_om', side_effect=AssertionError('OM step called')), \
                mock.patch.object(fc2d, 'do_run', side_effect=AssertionError('run step called')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = golden.main([str(source), '--output', str(output), '--exact-output', str(exact)])
        self.assertEqual(result, 0)
        self.assertEqual(output.read_bytes(), tensors['y_expect'][2])
        self.assertEqual(exact.read_bytes(), tensors['y_exact'][2])
        self.assertEqual(source.read_bytes(), original)

    def test_invalid_shapes_dtypes_and_descriptors_are_rejected(self):
        invalid = []
        for shape in ([1, 32, 7], [1, 0, 7, 9], [1, 32, -1, 9],
                      [1, 32, 7.0, 9], [True, 32, 7, 9]):
            case = self.descriptor()
            case['input']['shape'] = shape
            invalid.append(case)
        for tensor_path in ('input', 'output'):
            case = self.descriptor()
            case[tensor_path]['dtype'] = 'float32'
            invalid.append(case)
        for shape in ([32, 31, 3, 3], [32, 32, 0, 3], [32, 32, 3], [32, 32, 3, 1.5]):
            case = self.descriptor()
            case['kernels'][0]['shape'] = shape
            invalid.append(case)
        case = self.descriptor()
        case['kernels'][1]['shape'][1] = 31
        invalid.append(case)
        case = self.descriptor()
        case['kernels'][1]['dtype'] = 'fp16'
        invalid.append(case)
        case = self.descriptor('fp16')
        case['output']['dtype'] = 'int8'
        invalid.append(case)
        case = self.descriptor()
        case['output']['shape'] = [1, 7, 7, 8]
        invalid.append(case)
        case = self.descriptor()
        case['kernels'].append(copy.deepcopy(case['kernels'][1]))
        invalid.append(case)
        for field in ('input', 'kernels', 'output'):
            case = self.descriptor()
            case.pop(field)
            invalid.append(case)
        case = self.descriptor()
        case['unknown_option'] = 1
        invalid.append(case)
        for index, case in enumerate(invalid):
            with self.subTest(invalid=index), self.assertRaises((ValueError, SystemExit)):
                self.read_descriptor(case)

    def test_unsupported_attribute_values_are_rejected(self):
        invalid = dict(strides=[[1, 2, 1, 1], [0, 0, 1, 1], [1, 1]],
                       pads=[[0, 1, 0, 0, 0, 0, 0, 0], [-1] * 8, [0] * 4],
                       enable_relu=[[0, 2], [True], [1.0, 0]],
                       bias_mask=[[1, 0], [True], 'false'],
                       fixed_shift_value=[[-1, 0], [30, 0], [13, 59], [True, 0]],
                       dilations=[[1, 1, 2, 1, 1, 1, 1, 1]], groups=[[2, 1]],
                       data_format=[[1, 0], ['NHWC', 'NCHW']],
                       round_mode=[[1, 0], ['round', 'rint']],
                       pad_mode=[[1, 0], ['SAME', 'SPECIFIC']],
                       offset_x=[[1, 0]], enable_hf32=[[True, False]],
                       unknown_attribute=[[0, 0]])
        for key, values in invalid.items():
            for value in values:
                with self.subTest(attribute=key, value=value):
                    descriptor = self.descriptor()
                    descriptor['attrs'][key] = value
                    with self.assertRaises((ValueError, SystemExit)):
                        self.read_descriptor(descriptor)


if __name__ == '__main__':
    unittest.main()
