# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""验证设备性能配对、预热排除和汇总，避免把主机耗时当作算子耗时。"""
import ast
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d_performance as performance


def profile_rows(first=(999, 2, 20, 777), second=(999, 30, 3, 777), fused=(999, 12, 10, 777), timestamps=True):
    rows = []
    for iteration, (one, two) in enumerate(zip(first, second)):
        for layer, duration in enumerate((one, two)):
            row = {"OP Type": "ExtendConv2D", "Op Name": "conv%d" % (layer + 1),
                   "total_exe_time(us)": str(duration)}
            if timestamps:
                row["Task Start Time(us)"] = str(iteration * 10 + layer)
            rows.append(row)
    for iteration, duration in enumerate(fused):
        row = {"OP Type": "FusedMultiConv2d", "Op Name": "fused",
               "total_exe_time(us)": str(duration)}
        if timestamps:
            row["Task Start Time(us)"] = str(100 + iteration * 10)
        rows.append(row)
    return rows


class PerformanceMetrics(unittest.TestCase):
    def test_sums_same_execution_before_median_and_discards_warmup_probe(self):
        result = performance.parse_profile_rows(profile_rows(), 1, 2)
        self.assertEqual(result["samples"]["extend_total_us"], [32, 23])
        self.assertEqual(result["extend_total_us"], 27.5)
        self.assertEqual(result["fused_us"], 11)
        self.assertEqual(result["saved_us"], 16.5)
        self.assertAlmostEqual(result["improvement_percent"], 60)
        self.assertEqual(result["execution_count_per_model"], 4)
        self.assertEqual(result["operators"]["extend"], ["conv1", "conv2"])

    def test_total_is_not_the_sum_of_independent_layer_medians(self):
        rows = profile_rows(first=(100, 1, 1, 9, 200), second=(100, 9, 1, 1, 200),
                            fused=(100, 5, 5, 5, 200))
        result = performance.parse_profile_rows(rows, 1, 3)
        self.assertEqual(result["samples"]["extend_total_us"], [10, 2, 10])
        self.assertEqual(result["extend_total_us"], 10)
        self.assertEqual(result["improvement_percent"], 50)

    def test_timestamp_order_handles_shuffled_csv_rows(self):
        rows = profile_rows()
        rows.reverse()
        result = performance.parse_profile_rows(rows, 1, 2)
        self.assertEqual(result["samples"]["extend_total_us"], [32, 23])
        self.assertEqual(result["ordering"], "task_timestamp")

    def test_no_timestamp_uses_original_order_within_each_operator(self):
        rows = profile_rows(timestamps=False)
        grouped = rows[::2][:4] + rows[1::2][:4] + rows[8:]
        result = performance.parse_profile_rows(grouped, 1, 2)
        self.assertEqual(result["samples"]["extend_total_us"], [32, 23])
        self.assertEqual(result["ordering"], "csv_order_per_operator")

    def test_standard_msprof_columns_and_helper_rows(self):
        rows = profile_rows()
        for row in rows:
            row["Task Duration(us)"] = row.pop("total_exe_time(us)")
            row["Op Type"] = row.pop("OP Type")
            row["Node Name"] = row.pop("Op Name")
        rows.insert(0, {"Op Type": "Cast", "Node Name": "fused/Cast", "Task Duration(us)": "8",
                        "Task Start Time(us)": "-1"})
        result = performance.parse_profile_rows(rows, 1, 2)
        self.assertEqual(result["duration_column"], "Task Duration(us)")
        self.assertEqual(result["fused_us"], 11)

    def test_compiled_node_names_identify_opaque_nodes_without_type_column(self):
        rows = profile_rows()
        for row in rows:
            row.pop("OP Type")
        expected = {"fused": "fused", "extend": ["conv1", "conv2"]}
        result = performance.parse_profile_rows(rows, 1, 2, expected_nodes=expected)
        self.assertEqual(result["extend_total_us"], 27.5)

    def test_bound_node_names_allow_uncollected_operator_types(self):
        rows = profile_rows()
        for row in rows:
            row["OP Type"] = "N/A"
        result = performance.parse_profile_rows(rows, 1, 2,
                 expected_nodes={"fused": "fused", "extend": ["conv1", "conv2"]})
        self.assertEqual(result["extend_total_us"], 27.5)
        self.assertEqual(result["fused_us"], 11)

    def test_compiled_node_names_reject_type_mismatch_foreign_or_missing_targets(self):
        expected = {"fused": "fused", "extend": ["conv1", "conv2"]}
        for mutation in (lambda rows: rows[0].update({"OP Type": "Cast"}),
                         lambda rows: rows[0].update({"Op Name": "foreign_conv"}),
                         lambda rows: rows.pop(0)):
            rows = profile_rows()
            mutation(rows)
            with self.assertRaises(ValueError):
                performance.parse_profile_rows(rows, 1, 2, expected_nodes=expected)

    def test_binding_names_still_filter_helpers_and_reject_wrong_binding(self):
        rows = profile_rows()
        rows.append({"OP Type": "Cast", "Op Name": "helper", "total_exe_time(us)": "900",
                     "Task Start Time(us)": "250"})
        result = performance.parse_profile_rows(rows, 1, 2,
                 expected_nodes={"fused": "fused", "extend": ["conv1", "conv2"]})
        self.assertEqual(result["fused_us"], 11)
        for expected in ({"fused": "wrong", "extend": ["conv1", "conv2"]},
                         {"fused": "fused", "extend": ["conv1", "conv1"]}):
            with self.assertRaises(ValueError):
                performance.parse_profile_rows(rows, 1, 2, expected_nodes=expected)

    def test_missing_column_error_includes_available_export_headers(self):
        with self.assertRaisesRegex(ValueError, "actual columns: Name, Host Duration"):
            performance.parse_profile_rows([{"Name": "conv1", "Host Duration": "100"}], 0, 1)

    def test_name_only_export_can_identify_full_operator_tokens(self):
        rows = profile_rows()
        for row in rows:
            family = row.pop("OP Type")
            row["Name"] = "graph/%s_%s" % (family, {"conv1": "1", "conv2": "2", "fused": "0"}[row.pop("Op Name")])
        result = performance.parse_profile_rows(rows, 1, 2)
        self.assertEqual(result["fused_us"], 11)

    def test_rejects_unnamed_or_unidentifiable_operators(self):
        for identity in ("", "conv"):
            rows = profile_rows()
            if identity:
                for row in rows:
                    row.pop("OP Type")
                    row["Op Name"] = identity
            else:
                rows[0]["Op Name"] = ""
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                performance.parse_profile_rows(rows, 1, 2)

    def test_rejects_missing_probe_extra_execution_or_ambiguous_layer_identity(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[-1])),
                         lambda rows: rows[0].update({"Op Name": "third_extend"})):
            rows = profile_rows()
            mutation(rows)
            with self.assertRaises(ValueError):
                performance.parse_profile_rows(rows, 1, 2)

    def test_rejects_nonfinite_zero_negative_and_missing_durations(self):
        for value in ("nan", "inf", "-1", "0", "", None):
            rows = profile_rows()
            rows[2]["total_exe_time(us)"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "duration"):
                performance.parse_profile_rows(rows, 1, 2)

    def test_rejects_partial_or_ambiguous_timestamps(self):
        for change in (lambda rows: rows[0].pop("Task Start Time(us)"),
                       lambda rows: rows[2].update({"Task Start Time(us)": "0"}),
                       lambda rows: rows[-1].update({"Task Start Time(us)": "0"}),
                       lambda rows: rows[2].update({"Task Start Time(us)": "0.5"})):
            rows = profile_rows()
            change(rows)
            with self.assertRaises(ValueError):
                performance.parse_profile_rows(rows, 1, 2)

    def test_negative_improvement_is_not_hidden(self):
        result = performance.parse_profile_rows(profile_rows(first=(1, 1), second=(2, 2), fused=(5, 5)), 0, 1)
        self.assertEqual(result["saved_us"], -2)
        self.assertAlmostEqual(result["improvement_percent"], -200 / 3)

    def test_rejects_nonfinite_derived_pair_totals(self):
        rows = profile_rows(first=(1e308, 1), second=(1e308, 1), fused=(1, 1))
        with self.assertRaisesRegex(ValueError, "paired device duration"):
            performance.parse_profile_rows(rows, 0, 1)

    def test_rejects_invalid_sampling_configuration(self):
        for warmup, repeat in ((-1, 2), (0, 0), (True, 2), (0, 2.0)):
            with self.assertRaises(ValueError):
                performance.parse_profile_rows(profile_rows(), warmup, repeat)

    def test_csv_bom_and_hash_bind_raw_profile_evidence(self):
        rows = profile_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_summary.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = performance.parse_profile_csv(path, 1, 2)
            self.assertEqual(result["op_summary_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_shape_and_table_include_contract_and_signed_saving(self):
        case = {"dtype": "a16w8", "n": 1, "ci": 32, "hi": 8, "wi": 9, "cout1": 64, "cout2": 2,
                "kernel": [3, 3], "strides": [1, 1, 1, 1], "pads": [0] * 8}
        shape = performance.format_case_shape(case)
        self.assertIn("1x32x8x9 -> 1x64x6x7 -> 1x2x4x5", shape)
        self.assertEqual(shape, "a16w8 1x32x8x9 -> 1x64x6x7 -> 1x2x4x5")
        table = performance.render_performance_table([{"case": "fusion_a16w8", "shape": shape,
                    "fused_us": 5, "extend_total_us": 3, "saved_us": -2, "improvement_percent": -200 / 3}])
        self.assertIn("fusion_a16w8", table)
        self.assertIn("2 x ExtendConv2D (us)", table)
        self.assertIn("-2.000", table)
        self.assertIn("-66.67%", table)

    def test_module_is_python37_compatible(self):
        ast.parse((ROOT / "fc2d_performance.py").read_text(), feature_version=(3, 7))


if __name__ == "__main__":
    unittest.main()
