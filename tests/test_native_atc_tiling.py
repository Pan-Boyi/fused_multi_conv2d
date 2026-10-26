"""Check native ATC wire captures; compiling a new OM requires the separate tool."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location('native_atc_check', ROOT / 'tools/verify_fc2d_atc_tiling.py')
tool = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(tool)


class NativeTilingCapture(unittest.TestCase):
    def fixture(self, mode):
        return json.loads((ROOT / 'tests/fixtures' / ('native_tiling_before_%s.json' % mode)).read_text())

    def corrected(self, mode):
        fixture = self.fixture(mode)
        record = copy.deepcopy(fixture['record'])
        record['attrs'] = [{'name': name} for name in ('bias_mask','dtype','strides','pads','dilations','groups','data_format','offset_x','round_mode','pad_mode','enable_hf32','enable_relu','ascendc_op_para_size','fixed_shift_value')]
        for attr in record['attrs']:
            if attr['name'] in ('data_format','round_mode','pad_mode'):
                attr.update(dtype='list_int', value=[0,0])
        words = list(struct.unpack('<78I', bytes.fromhex(record['result']['tiling_data'])))
        words[-6] = 29 if mode == 'a16w8' else 42
        record['result']['tiling_data'] = struct.pack('<78I', *words).hex()
        spec = copy.deepcopy(fixture['spec'])
        if mode == 'a16w8':
            spec['shift1'] = 29
        return spec, record

    def test_real_pre_fix_captures_reveal_wrong_private_shifts(self):
        for mode, got, expected in (('fp16', 0, 42), ('a16w8', 2, 29)):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, 'operator attribute layout'):
                    spec = copy.deepcopy(fixture['spec'])
                    if mode == 'a16w8': spec['shift1'] = 29
                    tool.check_tiling(spec, fixture['record'])

    def test_decode_correct_shifts_and_independent_overrides(self):
        for mode in ('fp16', 'a16w8'):
            with self.subTest(mode=mode):
                spec, record = self.corrected(mode)
                report = tool.check_tiling(spec, record)
                self.assertEqual(report['fields']['fixedShift1'], 29 if mode == 'a16w8' else 42)
                self.assertEqual(report['tiling_data_bytes'], 312)
        spec, record = self.corrected('fp16')
        spec.update(shift1=34, shift2=37)
        words = list(struct.unpack('<78I', bytes.fromhex(record['result']['tiling_data'])))
        words[-6:-4] = [34, 37]
        record['result']['tiling_data'] = struct.pack('<78I', *words).hex()
        self.assertEqual(tool.check_tiling(spec, record)['fields']['fixedShift2'], 37)

    def test_only_shifts_consumed_by_each_dtype_path_are_compared(self):
        spec, record = self.corrected('a16w8')
        spec['shift2'] = 0
        self.assertEqual(tool.check_tiling(spec, record)['fields']['fixedShift1'],29)
        spec, record = self.corrected('fp16')
        spec.update(dtype_mode=1, shift1=0, shift2=0)
        self.assertEqual(tool.check_tiling(spec, record)['tiling_data_bytes'],312)
        spec['dtype_mode'] = 0
        with self.assertRaisesRegex(ValueError,'fixedShift1'):
            tool.check_tiling(spec,record)

    def test_second_l0c_stride_layout_preserves_shifted_fields(self):
        for mode in ('fp16', 'a16w8'):
            spec, record = self.corrected(mode)
            words = list(struct.unpack('<78I', bytes.fromhex(record['result']['tiling_data'])))
            words[52:55] = [16384, 1, 0]
            words[-4:] = [1, 1, 0, 1]
            spec.update(bias=1, relu1=0, relu2=1)
            record['result']['tiling_data'] = struct.pack('<78I', *words).hex()
            old = tool.check_tiling(spec, record)
            words.insert(53, 24576)
            record['result']['tiling_data'] = struct.pack('<79I', *words).hex()
            with self.subTest(mode=mode):
                new = tool.check_tiling(spec, record)
                self.assertEqual(new['tiling_data_bytes'], 316)
                self.assertEqual(new['fields']['l0cStride2'], 24576)
                self.assertEqual(old['fields']['l0cStride2'], 0)
                for name in tool.TAIL_FIELDS + ('l0cStride', 'w1Stage', 'w2Stage'):
                    self.assertEqual(new['fields'][name], old['fields'][name])
                self.assertEqual(new['fields']['w1Stage'], 1)
                self.assertEqual(new['fields']['w2Stage'], 0)

    def test_unknown_nearby_tiling_sizes_are_rejected(self):
        spec, record = self.corrected('fp16')
        for size in (308, 313, 315, 317, 320):
            malformed = copy.deepcopy(record)
            malformed['result']['tiling_data'] = '00' * size
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'tiling ABI'):
                tool.check_tiling(spec, malformed)

    def test_rejects_old_string_contract_wrong_enum_types_and_nonzero_codes(self):
        spec, good = self.corrected('fp16')
        for name, old in (('data_format','NCHW'),('round_mode','rint'),('pad_mode','SPECIFIC')):
            for dtype, value in (('str',old),('int',0),('list_int',[0]),('list_int',[0,1]),('list_int',None)):
                record=copy.deepcopy(good)
                attr=next(a for a in record['attrs'] if a['name']==name)
                attr.update(dtype=dtype,value=value)
                with self.subTest(name=name,dtype=dtype,value=value), self.assertRaisesRegex(ValueError,'enum attribute contract'):
                    tool.check_tiling(spec,record)

    def test_missing_sdk_slot_wrong_wire_size_or_geometry_is_rejected(self):
        spec, record = self.corrected('fp16')
        missing = copy.deepcopy(record)
        missing['attrs'].pop(12)
        with self.assertRaisesRegex(ValueError, 'attribute layout'):
            tool.check_tiling(spec, missing)
        malformed = copy.deepcopy(record)
        malformed['result']['tiling_data'] = '00000000'
        with self.assertRaisesRegex(ValueError, 'tiling ABI'):
            tool.check_tiling(spec, malformed)
        spec['hi'] += 1
        with self.assertRaisesRegex(ValueError, 'different case geometry'):
            tool.check_tiling(spec, record)

    def test_capture_is_required_and_conflicting_runs_cannot_pass(self):
        spec, record = self.corrected('fp16')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(ValueError, 'no native'):
                tool.check_capture(spec, directory)
            path = directory / 'tiling.1.jsonl'
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(tool.check_capture(spec, directory)['fields']['fixedShift1'], 42)
            other = copy.deepcopy(record)
            words = list(struct.unpack('<78I', bytes.fromhex(other['result']['tiling_data'])))
            words[6] += 1
            other['result']['tiling_data'] = struct.pack('<78I', *words).hex()
            path.write_text(path.read_text() + json.dumps(other) + '\n')
            with self.assertRaisesRegex(ValueError, 'multiple different'):
                tool.check_capture(spec, directory)

    def test_unverified_compilation_loses_binding_and_preserves_original_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'case.bin'
            source.write_bytes(b'original case content')
            output = root / 'verified'

            def fake_compile(case, paths, soc):
                omdir = Path(paths['omdir'])
                omdir.mkdir()
                (omdir / 'fake.om').write_bytes(b'synthetic test model')
                (omdir / 'fake.om.fc2d.json').write_text('{}')

            with mock.patch.object(tool.fc2d, 'read_spec', return_value={}), \
                    mock.patch.object(tool.fc2d, 'do_om', side_effect=fake_compile) as compile_om:
                with self.assertRaisesRegex(ValueError, 'no native'):
                    tool.compile_and_verify(source, output, 'MC62CM12AA')
            compile_om.assert_called_once()
            self.assertFalse((output / 'om/fake.om.fc2d.json').exists())
            self.assertEqual(source.read_bytes(), b'original case content')
            self.assertTrue((output / 'om/fake.om').exists())


if __name__ == '__main__':
    unittest.main()
