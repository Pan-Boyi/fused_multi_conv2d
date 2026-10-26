# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""验证流水耗时与算子执行轮次对齐，并保留缺失指标。"""
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


def profile_rows():
    rows = []
    durations = {"conv1": (100, 1, 1, 9, 200), "conv2": (100, 9, 1, 1, 200),
                 "fused": (100, 3, 4, 5, 200)}
    for iteration in range(5):
        for layer, name in enumerate(("conv1", "conv2")):
            rows.append({"Op Type": "ExtendConv2D", "Op Name": name,
                         "Task Start Time(us)": str(iteration * 10 + layer),
                         "total_exe_time(us)": str(durations[name][iteration])})
    for iteration, duration in enumerate(durations["fused"]):
        rows.append({"Op Type": "FusedMultiConv2d", "Op Name": "fused",
                     "Task Start Time(us)": str(100 + iteration * 10),
                     "total_exe_time(us)": str(duration)})
    for row in rows:
        for index, (_, aliases) in enumerate(performance.PIPELINE_COLUMNS):
            row[aliases[0]] = str(float(row["total_exe_time(us)"]) * (index + 1) / 10)
    return rows


def parse(rows):
    return performance.parse_profile_rows(rows, 1, 3, pipeline_details=True)


class PipelineMetrics(unittest.TestCase):
    def test_optional_metrics_do_not_change_existing_results(self):
        rows = profile_rows()
        rows[2]["mte1_exe_time(us)"] = "bad"
        result = performance.parse_profile_rows(rows, 1, 3)
        self.assertNotIn("pipeline_details", result)
        self.assertEqual(result["extend_total_us"], 10)

    def test_per_operator_medians_use_same_selected_executions(self):
        result = parse(profile_rows())
        detail = result["pipeline_details"]
        fused, first, second, pair = detail["operators"]
        self.assertEqual(detail["units"], "us")
        self.assertEqual(detail["columns"]["cube"], "mac_exe_time(us)")
        self.assertEqual([op["name"] for op in detail["operators"]],
                         ["fused", "conv1", "conv2", ["conv1", "conv2"]])
        self.assertEqual(fused["samples"]["cube"], [0.3, 0.4, 0.5])
        self.assertEqual(first["samples"]["cube"], [0.1, 0.1, 0.9])
        self.assertEqual(second["samples"]["cube"], [0.9, 0.1, 0.1])
        self.assertEqual(fused["total_us"], result["fused_us"])
        self.assertEqual(pair["total_us"], result["extend_total_us"])
        self.assertAlmostEqual(fused["pipeline_us"]["fixpipe"], 2.4)

    def test_pair_pipeline_median_sums_each_execution_first(self):
        first, second, pair = parse(profile_rows())["pipeline_details"]["operators"][1:]
        self.assertEqual(pair["samples"]["cube"], [1.0, 0.2, 1.0])
        self.assertEqual(pair["pipeline_us"]["cube"], 1.0)
        self.assertNotEqual(pair["pipeline_us"]["cube"],
                            first["pipeline_us"]["cube"] + second["pipeline_us"]["cube"])

    def test_timestamp_sorting_preserves_pipeline_values(self):
        rows = profile_rows()
        rows.reverse()
        self.assertEqual(parse(rows)["pipeline_details"], parse(profile_rows())["pipeline_details"])

    def test_missing_columns_are_unavailable_not_zero(self):
        rows = profile_rows()
        for row in rows:
            row["fixpipe_time(ns)"] = row.pop("fixpipe_time(us)")
        detail = parse(rows)["pipeline_details"]
        self.assertIsNone(detail["columns"]["fixpipe"])
        for operator in detail["operators"]:
            self.assertIsNone(operator["pipeline_us"]["fixpipe"])
            self.assertEqual(operator["samples"]["fixpipe"], [None, None, None])

    def test_missing_selected_sample_disables_complete_median_and_pair(self):
        for value in (None, "", "  ", "N/A", "n/a", "NA", "-"):
            with self.subTest(value=value):
                rows = profile_rows()
                rows[2]["mte2_exe_time(us)"] = value
                _, first, second, pair = parse(rows)["pipeline_details"]["operators"]
                self.assertIsNone(first["pipeline_us"]["mte2"])
                self.assertIsNone(pair["pipeline_us"]["mte2"])
                self.assertIsNone(pair["samples"]["mte2"][0])
                self.assertIsNotNone(second["pipeline_us"]["mte2"])
                self.assertIsNotNone(first["pipeline_us"]["mte1"])

    def test_zero_pipeline_time_is_valid(self):
        rows = profile_rows()
        for row in rows:
            row["mte3_exe_time(us)"] = "0"
        detail = parse(rows)["pipeline_details"]
        for operator in detail["operators"]:
            self.assertEqual(operator["pipeline_us"]["mte3"], 0)
        self.assertIn("0.000", performance.render_pipeline_table({"pipeline_details": detail}))

    def test_malformed_selected_samples_name_metric_and_operator(self):
        for value in ("nan", "inf", "-inf", "-0.01", "bad"):
            with self.subTest(value=value):
                rows = profile_rows()
                rows[2]["mte1_exe_time(us)"] = value
                with self.assertRaisesRegex(ValueError, "mte1 pipeline duration for operator conv1"):
                    parse(rows)

    def test_excluded_warmup_and_probe_metrics_are_not_validated(self):
        rows = profile_rows()
        for index in (0, 1, 8, 9, 10, 14):
            rows[index]["mac_exe_time(us)"] = "bad"
        self.assertEqual(parse(rows)["pipeline_details"]["operators"][0]["pipeline_us"]["cube"], 0.4)

    def test_non_target_helpers_do_not_affect_pipeline_selection(self):
        rows = profile_rows()
        rows.insert(0, {"Op Type": "Cast", "Op Name": "helper", "mte1_exe_time(us)": "bad",
                        "total_exe_time(us)": "bad", "Task Start Time(us)": "bad"})
        result = performance.parse_profile_rows(rows, 1, 3, pipeline_details=True,
                 expected_nodes={"fused": "fused", "extend": ["conv1", "conv2"]})
        self.assertEqual(result["pipeline_details"], parse(profile_rows())["pipeline_details"])

    def test_cube_alias_and_normalized_us_headers(self):
        rows = profile_rows()
        for row in rows:
            row["Cube_Exe_Time (us)"] = row.pop("mac_exe_time(us)")
            row["MTE1 Exe Time (us)"] = row.pop("mte1_exe_time(us)")
        detail = parse(rows)["pipeline_details"]
        self.assertEqual(detail["columns"]["cube"], "Cube_Exe_Time (us)")
        self.assertEqual(detail["columns"]["mte1"], "MTE1 Exe Time (us)")
        self.assertEqual(detail["operators"][0]["pipeline_us"]["cube"], 0.4)

    def test_mac_column_takes_priority_when_both_aliases_exist(self):
        rows = profile_rows()
        for row in rows:
            row["cube_exe_time(us)"] = "900"
        self.assertEqual(parse(rows)["pipeline_details"]["operators"][0]["pipeline_us"]["cube"], 0.4)

    def test_aic_headers_keep_vector_pipeline_times_separate(self):
        rows = profile_rows()
        aliases = {"mac_exe_time(us)": "aic_mac_time(us)",
                   "scalar_exe_time(us)": "aic_scalar_time(us)",
                   "mte1_exe_time(us)": "aic_mte1_time(us)",
                   "mte2_exe_time(us)": "aic_mte2_time(us)",
                   "fixpipe_time(us)": "aic_fixpipe_time(us)"}
        for row in rows:
            for old, new in aliases.items():
                row[new] = row.pop(old)
            row.pop("mte3_exe_time(us)")
            row["aiv_mte2_time(us)"] = "900"
            row["aiv_mte3_time(us)"] = "800"
        detail = parse(rows)["pipeline_details"]
        self.assertEqual(detail["columns"]["cube"], "aic_mac_time(us)")
        self.assertEqual(detail["columns"]["scalar"], "aic_scalar_time(us)")
        self.assertEqual(detail["columns"]["mte1"], "aic_mte1_time(us)")
        self.assertEqual(detail["columns"]["mte2"], "aic_mte2_time(us)")
        self.assertEqual(detail["columns"]["fixpipe"], "aic_fixpipe_time(us)")
        self.assertAlmostEqual(detail["operators"][0]["pipeline_us"]["mte2"], 1.6)
        self.assertIsNone(detail["columns"]["mte3"])
        self.assertIsNone(detail["operators"][0]["pipeline_us"]["mte3"])

    def test_existing_headers_precede_aic_aliases_and_ns_is_excluded(self):
        rows = profile_rows()
        for row in rows:
            row["aic_mte2_time(us)"] = "900"
            row["aic_fixpipe_time(ns)"] = row.pop("fixpipe_time(us)")
        detail = parse(rows)["pipeline_details"]
        self.assertEqual(detail["columns"]["mte2"], "mte2_exe_time(us)")
        self.assertAlmostEqual(detail["operators"][0]["pipeline_us"]["mte2"], 1.6)
        self.assertIsNone(detail["columns"]["fixpipe"])

    def test_nonfinite_pair_sum_reports_metric_and_both_nodes(self):
        rows = profile_rows()
        rows[2]["mte1_exe_time(us)"] = "1e308"
        rows[3]["mte1_exe_time(us)"] = "1e308"
        with self.assertRaisesRegex(ValueError, "mte1 pipeline duration for operator conv1 \\+ conv2"):
            parse(rows)

    def test_csv_bom_hash_and_pipeline_samples(self):
        rows = profile_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_summary.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = performance.parse_profile_csv(path, 1, 3, pipeline_details=True)
            self.assertEqual(result["op_summary_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["pipeline_details"], parse(rows)["pipeline_details"])

    def test_render_all_pipelines_and_unavailable_markers(self):
        rows = profile_rows()
        rows[2]["fixpipe_time(us)"] = "N/A"
        table = performance.render_pipeline_table(parse(rows))
        for label in ("FusedMultiConv2d", "ExtendConv2D[1]", "ExtendConv2D[2]", "2 x ExtendConv2D",
                      "Total (us)", "Cube (us)", "Scalar (us)", "MTE1 (us)", "MTE2 (us)",
                      "MTE3 (us)", "Fixpipe (us)", "N/A", "Pipeline times may overlap"):
            self.assertIn(label, table)

    def test_python37_compatible(self):
        ast.parse((ROOT / "fc2d_performance.py").read_text(), feature_version=(3, 7))


if __name__ == "__main__":
    unittest.main()
