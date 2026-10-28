# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Check Python reference values against the original C++ CPU reference."""
import ctypes
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fused_multi_conv2d_golden as golden
import run_fused_multi_conv2d as runner


ORACLE_SOURCE = r'''
#include <algorithm>
#include "fused_multi_conv2d_golden.h"
using namespace fc2d_golden;

static float Scale(uint64_t entry)
{
    const uint32_t bits = static_cast<uint32_t>(entry);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return int8path::EffectiveScale(value);
}

extern "C" void Evaluate(const int* s, const void* x, const void* w1, const void* b1,
    const void* w2, const void* b2, const uint64_t* scale1, const uint64_t* scale2,
    void* expected, uint16_t* exact)
{
    const int mode=s[0], n=s[1], ci=s[2], hi=s[3], wi=s[4], c1=s[5], c2=s[6];
    const int kh1=s[7], kw1=s[8], kh2=s[9], kw2=s[10], st1=s[11], st2=s[12];
    const int ph1=s[13], pw1=s[14], ph2=s[15], pw2=s[16];
    const int shift1=s[17], shift2=s[18], relu1=s[19], relu2=s[20], bias=s[21];
    const int h1=(hi+2*ph1-kh1)/st1+1, wmid=(wi+2*pw1-kw1)/st1+1;
    const int h2=(h1+2*ph2-kh2)/st2+1, wo=(wmid+2*pw2-kw2)/st2+1;
    for (int batch=0; batch<n; ++batch) {
        std::vector<int32_t> acc1, acc2;
        if (mode==0) {
            std::vector<double> raw1, raw2;
            std::vector<uint16_t> mid, out, outExact;
            f16path::ConvRaw(static_cast<const uint16_t*>(x)+batch*ci*hi*wi,
                static_cast<const uint16_t*>(w1),ci,hi,wi,c1,h1,wmid,kh1,kw1,
                st1,ph1,pw1,raw1,nullptr);
            long sat=0;
            f16path::FixedEpilogue(raw1,bias?static_cast<const uint16_t*>(b1):nullptr,
                c1,h1*wmid,shift1,relu1,mid,&sat);
            f16path::ConvRaw(mid.data(),static_cast<const uint16_t*>(w2),c1,h1,wmid,
                c2,h2,wo,kh2,kw2,st2,ph2,pw2,raw2,nullptr);
            f16path::FixedEpilogue(raw2,bias?static_cast<const uint16_t*>(b2):nullptr,
                c2,h2*wo,shift2,relu2,out,&sat);
            f16path::ExactEpilogue(raw2,bias?static_cast<const uint16_t*>(b2):nullptr,
                c2,h2*wo,relu2,outExact);
            std::copy(out.begin(),out.end(),static_cast<uint16_t*>(expected)+batch*c2*h2*wo);
            std::copy(outExact.begin(),outExact.end(),exact+batch*c2*h2*wo);
            continue;
        }
        if (mode==3) {
            std::vector<int8_t> restored(ci*hi*wi);
            const auto* input=static_cast<const uint16_t*>(x)+batch*ci*hi*wi;
            for (size_t i=0;i<restored.size();++i) {
                restored[i]=static_cast<int8_t>(std::ldexp(F16BitsToF32(input[i]),29-shift1));
            }
            int8path::ConvInt8(restored.data(),static_cast<const int8_t*>(w1),
                bias?static_cast<const int32_t*>(b1):nullptr,ci,hi,wi,c1,h1,wmid,
                kh1,kw1,st1,ph1,pw1,acc1);
        } else {
            int8path::ConvInt8(static_cast<const int8_t*>(x)+batch*ci*hi*wi,
                static_cast<const int8_t*>(w1),bias?static_cast<const int32_t*>(b1):nullptr,
                ci,hi,wi,c1,h1,wmid,kh1,kw1,st1,ph1,pw1,acc1);
        }
        std::vector<int8_t> mid(acc1.size());
        for (int co=0;co<c1;++co) {
            for (int p=0;p<h1*wmid;++p) {
                mid[co*h1*wmid+p]=int8path::Req8(acc1[co*h1*wmid+p],Scale(scale1[co]),relu1);
            }
        }
        int8path::ConvInt8(mid.data(),static_cast<const int8_t*>(w2),
            bias?static_cast<const int32_t*>(b2):nullptr,c1,h1,wmid,c2,h2,wo,
            kh2,kw2,st2,ph2,pw2,acc2);
        for (int co=0;co<c2;++co) {
            for (int p=0;p<h2*wo;++p) {
                const int local=co*h2*wo+p, offset=batch*c2*h2*wo+local;
                if (mode==1) {
                    static_cast<int8_t*>(expected)[offset]=int8path::Req8(acc2[local],Scale(scale2[co]),relu2);
                } else {
                    float value=static_cast<float>(acc2[local])*Scale(scale2[co]);
                    if (relu2 && value<0) value=-0.0f;
                    static_cast<uint16_t*>(expected)[offset]=F32ToF16Bits(value);
                }
            }
        }
    }
}
'''


class PythonGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temp.name)
        cls.generator = cls.work / 'gen_case'
        subprocess.run(['g++', '-std=c++17', '-O2', str(ROOT / 'gen_case.cpp'),
                        '-o', str(cls.generator)], check=True)
        source = cls.work / 'oracle.cpp'
        source.write_text(ORACLE_SOURCE)
        library = cls.work / 'oracle.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I', str(ROOT),
                        str(source), '-o', str(library)], check=True)
        cls.oracle = ctypes.CDLL(str(library))
        cls.oracle.Evaluate.argtypes = [ctypes.POINTER(ctypes.c_int)] + [ctypes.c_void_p] * 9
        cls.oracle.Evaluate.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def generated_case(self, mode, variant=0, shift1=None):
        path = self.work / ('%s_%d_%s.bin' % (mode, variant, shift1))
        if shift1 is None:
            shift1 = 13 if mode == 'a16w8' else 42
        geometry = ([1, 3, 3, 1, 1, 2, 0, 1, 1, 0] if variant else
                    [3, 3, 1, 1, 2, 1, 1, 1, 0, 0])
        command = [str(self.generator), str(path), '--dtype', mode, '--n', '2',
                   '--ci', '32', '--cout1', '32', '--cout2', '7', '--hi', '5', '--wi', '7',
                   '--bias', str(variant), '--relu1', str(variant), '--relu2', str(1-variant),
                   '--shift1', str(shift1), '--shift2', '42']
        for key, value in zip(('kh', 'kw', 'kh2', 'kw2', 's1', 's2', 'ph1', 'pw1', 'ph2', 'pw2'), geometry):
            command.extend(['--' + key, str(value)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        tensors, *_, info = runner.load_case(str(path))
        return info, tensors, path

    def cpp_output(self, info, tensors):
        fields = [info[k] for k in ('dtype_mode', 'n', 'ci', 'hi', 'wi', 'cout1', 'cout2',
                  'kh', 'kw', 'kh2', 'kw2', 's1', 's2', 'ph1', 'pw1', 'ph2', 'pw2',
                  'shift1', 'shift2', 'relu1', 'relu2', 'bias')]
        fields[9] = fields[9] or fields[7]
        fields[10] = fields[10] or fields[8]
        spec = (ctypes.c_int * len(fields))(*fields)
        mode = info['dtype_mode']
        names = ['x', 'filter1', 'bias1', 'filter2', 'bias2', 'quant_scale1',
                 'quant_scale2' if mode == 1 else 'dequant_scale2']
        buffers = [ctypes.create_string_buffer(tensors[name][2]) if name in tensors else None
                   for name in names]
        shape = (info['n'], info['cout2'], info['ho2'], info['wo2'])
        elements = int(np.prod(shape))
        expected = ctypes.create_string_buffer(elements * (1 if mode == 1 else 2))
        exact = ctypes.create_string_buffer(elements * 2)
        self.oracle.Evaluate(spec, *buffers, expected, exact)
        result = {'y_expect': (2 if mode == 1 else 1, shape, expected.raw)}
        if mode == 0:
            result['y_exact'] = (1, shape, exact.raw)
        return result

    def assert_outputs_equal(self, actual, expected):
        for name in ('y_expect', 'y_exact'):
            if name not in expected:
                continue
            self.assertEqual(actual[name][0], expected[name][0], name)
            self.assertEqual(tuple(actual[name][1]), tuple(expected[name][1]), name)
            self.assertEqual(actual[name][2], expected[name][2], name)

    def test_original_cpp_generator_four_paths_and_mixed_geometry(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            for variant in (0, 1):
                with self.subTest(mode=mode, variant=variant):
                    info, tensors, path = self.generated_case(mode, variant)
                    expected = {key: tensors[key] for key in ('y_expect', 'y_exact') if key in tensors}
                    actual = golden.compute_golden(info, tensors)
                    self.assert_outputs_equal(actual, expected)
                    header = struct.unpack_from('<8sIIQQQ', path.read_bytes())
                    self.assertEqual(actual['stats'], dict(nonzero=header[3], sat=header[4]))
                    parsed_info, parsed_tensors = golden.read_case(str(path))
                    self.assert_outputs_equal(golden.compute_golden(parsed_info, parsed_tensors), expected)
                    self.assert_outputs_equal(self.cpp_output(info, tensors), expected)

    def test_stored_golden_and_duplicate_weight_do_not_supply_python_answer(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            with self.subTest(mode=mode):
                info, tensors, _ = self.generated_case(mode)
                expected = {key: tensors[key] for key in ('y_expect', 'y_exact') if key in tensors}
                changed = dict(tensors)
                for name in ('y_expect', 'y_exact', 'w1_nchw'):
                    if name in changed:
                        dtype, dims, raw = changed[name]
                        changed[name] = (dtype, dims, b'\xa5' * len(raw))
                self.assert_outputs_equal(golden.compute_golden(info, changed), expected)
                for name in ('y_expect', 'y_exact', 'w1_nchw'):
                    changed.pop(name, None)
                self.assert_outputs_equal(golden.compute_golden(info, changed), expected)

    def controlled_case(self, mode, relu=False):
        info, tensors, _ = self.generated_case(mode)
        info.update(n=1, hi=1, wi=8, kh=1, kw=1, kh2=1, kw2=1, s1=1, s2=1,
                    ph1=0, pw1=0, ph2=0, pw2=0, ho2=1, wo2=8, cout2=7,
                    bias=1, relu1=int(relu), relu2=int(relu), shift1=58 if mode == 'fp16' else 29,
                    shift2=57)
        float_mode = mode == 'fp16'
        inputs = np.tile(np.array([-2.5, -1.5, -.5, -0., 0., .5, 1.5, 2.5] if float_mode
                                  else [-128, -3, -1, 0, 1, 3, 126, 127],
                                  dtype='<f2' if float_mode or mode == 'a16w8' else 'i1'), (1, 32, 1, 1))
        weight_dtype = '<f2' if float_mode else 'i1'
        w1 = np.eye(32, dtype=weight_dtype).reshape(32, 32, 1, 1)
        w2 = np.zeros((7, 32, 1, 1), dtype=weight_dtype)
        for co in range(7):
            w2[co, co, 0, 0] = (-1 if co % 2 else 1) * (co + 1)
        bias_dtype = '<f2' if float_mode else '<i4'
        arrays = {'x': inputs, 'filter1': w1, 'filter2': w2,
                  'bias1': np.array([((i % 3)-1) * (.5 if float_mode else 1) for i in range(32)], dtype=bias_dtype),
                  'bias2': np.array([((i % 3)-1) * (.5 if float_mode else 1) for i in range(7)], dtype=bias_dtype)}
        changed = {name: (1 if array.dtype == np.dtype('<f2') else (3 if array.dtype == np.dtype('<i4') else 2),
                          tuple(array.shape), array.tobytes()) for name, array in arrays.items()}
        if not float_mode:
            for name, count in (('quant_scale1', 32), ('quant_scale2' if mode == 'int8' else 'dequant_scale2', 7)):
                scales = [(.5, .25, .75019, 1.00019)[co % 4] for co in range(count)]
                entries = [struct.unpack('<I', struct.pack('<f', scale))[0] | (1 << 46) for scale in scales]
                changed[name] = (10, (count,), struct.pack('<%dQ' % count, *entries))
        return info, changed

    def test_per_channel_scale_halfway_rounding_saturation_and_relu(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            for relu in (False, True):
                with self.subTest(mode=mode, relu=relu):
                    info, tensors = self.controlled_case(mode, relu)
                    self.assert_outputs_equal(golden.compute_golden(info, tensors), self.cpp_output(info, tensors))

    def test_fp16_fixed_accumulator_saturation(self):
        info, tensors = self.controlled_case('fp16')
        info.update(shift1=42, shift2=42)
        dtype, dims, raw = tensors['x']
        tensors['x'] = (dtype, dims, np.full(dims, 256, dtype='<f2').tobytes())
        weights = np.eye(32, dtype='<f2').reshape(32, 32, 1, 1) * np.float16(256)
        tensors['filter1'] = (1, weights.shape, weights.tobytes())
        actual = golden.compute_golden(info, tensors)
        self.assert_outputs_equal(actual, self.cpp_output(info, tensors))
        self.assertTrue(np.isfinite(np.frombuffer(actual['y_expect'][2], dtype='<f2')).all())

    def test_fp16_subnormal_and_signed_zero(self):
        info, tensors = self.controlled_case('fp16', relu=True)
        info.update(shift1=34, shift2=34, bias=0)
        values = np.array([-2**-24, -0., 0., 2**-24, 2**-14, 1+2**-10, 1, -1], dtype='<f2')
        tensors['x'] = (1, (1, 32, 1, 8), np.tile(values, 32).tobytes())
        self.assert_outputs_equal(golden.compute_golden(info, tensors), self.cpp_output(info, tensors))

    def test_small_workspace_keeps_output_values(self):
        for mode in ('fp16', 'int8', 's8f16', 'a16w8'):
            with self.subTest(mode=mode):
                info, tensors, _ = self.generated_case(mode, 1)
                expected = golden.compute_golden(info, tensors)
                self.assert_outputs_equal(golden.compute_golden(info, tensors, workspace_mb=.02), expected)

    def test_a16w8_non_lattice_input_is_not_rounded_to_invent_a_reference(self):
        info, tensors = self.controlled_case('a16w8')
        dtype, dims, raw = tensors['x']
        inputs = np.frombuffer(raw, dtype='<f2').copy()
        inputs[0] = np.float16(.5)
        tensors['x'] = (dtype, dims, inputs.tobytes())
        with self.assertRaises(ValueError):
            golden.compute_golden(info, tensors)

    def test_a16w8_shift_endpoints_reproduce_original_integer_reference(self):
        for shift1 in (0, 5, 29):
            with self.subTest(shift1=shift1):
                info, tensors, _ = self.generated_case('a16w8', shift1=shift1)
                actual = golden.compute_golden(info, tensors)
                self.assert_outputs_equal(actual, {'y_expect': tensors['y_expect']})
                for shift2 in (0, 58):
                    self.assert_outputs_equal(golden.compute_golden(dict(info, shift2=shift2), tensors), actual)


if __name__ == '__main__':
    unittest.main()
