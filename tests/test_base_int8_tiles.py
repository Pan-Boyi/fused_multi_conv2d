"""Fault-injection controls for offline base_int8 output diagnostics."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/analyze_base_int8_tiles.py'
spec = importlib.util.spec_from_file_location('tile_analysis', path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class TileAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = bytes((i * 37 + (i // 56) * 17) % 127 for i in range(774144))

    def test_clean_output(self):
        result = tool.analyze(self.reference, self.reference, self.reference)
        self.assertEqual(result['mismatches'], 0)
        self.assertEqual(result['exact_other_tile_matches'], [])
        self.assertEqual(result['repeat_comparison']['different_elements'], 0)

    def test_first_tile_overwritten_by_second_tile(self):
        actual = bytearray(self.reference)
        for channel in range(96):
            for band in range(8):
                pos = channel * 144 * 56 + band * 18 * 56
                actual[pos:pos + 504] = self.reference[pos + 504:pos + 1008]
        result = tool.analyze(self.reference, bytes(actual))
        self.assertGreater(result['mismatches'], 0)
        self.assertEqual(result['exact_other_tile_matches'],
                         [{'actual_tile': 0, 'matches_reference_tile': 1}])
        self.assertTrue(all(n == 0 for h, n in enumerate(result['row_mismatches']) if h % 18 >= 9))

    def test_one_value_error_is_not_a_tile_copy(self):
        actual = bytearray(self.reference)
        actual[0] ^= 1
        result = tool.analyze(self.reference, bytes(actual), self.reference)
        self.assertEqual(result['mismatches'], 1)
        self.assertEqual(result['repeat_comparison']['different_elements'], 1)
        self.assertEqual(result['exact_other_tile_matches'], [])

    def test_rejects_incomplete_output(self):
        with self.assertRaises(ValueError):
            tool.analyze(self.reference, self.reference[:-1])

    def test_case_reference_and_version_check(self):
        shape = [1, 32, 288, 112, 64, 96, 3, 3, 1, 2, 1, 1, 1, 1,
                 1, 0, 1, 1, 42, 42, 144, 56] + [0] * 10
        header = struct.pack('<8sIIQQQ32iII', b'FC2DCASE', 6, 1, 0, 0,
                             len(self.reference), *shape, 0, 0)
        tensor = struct.pack('<16sII4qQ', b'y_expect', 2, 4, 1, 96, 144, 56,
                             len(self.reference))
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case.bin'
            case.write_bytes(header + tensor + self.reference)
            self.assertEqual(tool.read_reference(case), self.reference)
            broken = bytearray(case.read_bytes())
            struct.pack_into('<I', broken, 8, 5)
            case.write_bytes(broken)
            with self.assertRaises(ValueError):
                tool.read_reference(case)


if __name__ == '__main__':
    unittest.main()
