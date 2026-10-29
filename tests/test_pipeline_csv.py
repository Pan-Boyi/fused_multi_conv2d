# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""验证批量性能 CSV 与已选中的设备流水采样保持一致。"""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d_performance as performance
import run_profile as profile
from test_pipeline_metrics import profile_rows


BASE_COLUMNS = ["case", "shape", "fused_us", "extend_total_us", "saved_us", "improvement_percent"]
ROLES = ["fused", "extend1", "extend2", "extend_total"]
METRICS = ["cube", "scalar", "mte1", "mte2", "mte3", "fixpipe"]
SAMPLE_COLUMNS = ["case", "shape", "operator", "op_name", "sample_index", "total_us"] + [
    metric + "_us" for metric in METRICS]


def measured_case(name="case_a", rows=None, pipeline=True):
    result = performance.parse_profile_rows(profile_rows() if rows is None else rows, 1, 3,
                                             pipeline_details=pipeline)
    result.update(case=name, shape="a16w8 1x32x8x9 -> 1x64x6x7 -> 1x2x4x5")
    return result


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


class PipelineCsv(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def save(self, rows):
        profile.save_performance_results(str(self.output), rows, display=False)
        return (read_csv(self.output / "performance_comparison.csv"),
                read_csv(self.output / "pipeline_samples.csv"))

    def test_batch_summary_keeps_original_columns_and_all_pipeline_medians(self):
        second_rows = profile_rows()
        for row in second_rows:
            row["total_exe_time(us)"] = str(float(row["total_exe_time(us)"]) * 2)
            for _, aliases in performance.PIPELINE_COLUMNS:
                row[aliases[0]] = str(float(row[aliases[0]]) * 3)
        results = [measured_case(), measured_case("case_b,第二条", second_rows)]
        results[1]["shape"] = "shape, with comma"
        (columns, summary), _ = self.save(results)
        expected_columns = BASE_COLUMNS + ["extend1_us", "extend2_us"] + [
            role + "_" + metric + "_us" for role in ROLES for metric in METRICS]
        self.assertEqual(columns, expected_columns)
        self.assertEqual(len(summary), 2)
        for source, row in zip(results, summary):
            self.assertEqual(row["case"], source["case"])
            self.assertEqual(row["shape"], source["shape"])
            for column in BASE_COLUMNS[2:]:
                self.assertEqual(float(row[column]), source[column])
            operators = source["pipeline_details"]["operators"]
            self.assertEqual(float(row["extend1_us"]), operators[1]["total_us"])
            self.assertEqual(float(row["extend2_us"]), operators[2]["total_us"])
            for role, operator in zip(ROLES, operators):
                for metric in METRICS:
                    self.assertEqual(float(row[role + "_" + metric + "_us"]),
                                     operator["pipeline_us"][metric])

    def test_pair_median_is_taken_after_per_execution_addition(self):
        (columns, summary), _ = self.save([measured_case()])
        row = summary[0]
        self.assertEqual(float(row["extend_total_us"]), 10)
        self.assertEqual(float(row["extend1_us"]), 1)
        self.assertEqual(float(row["extend2_us"]), 1)
        self.assertEqual(float(row["extend_total_cube_us"]), 1)
        self.assertNotEqual(float(row["extend_total_cube_us"]),
                            float(row["extend1_cube_us"]) + float(row["extend2_cube_us"]))

    def test_samples_preserve_all_selected_values_for_every_case_and_operator(self):
        results = [measured_case(), measured_case("case_b")]
        _, (columns, samples) = self.save(results)
        self.assertEqual(columns, SAMPLE_COLUMNS)
        self.assertEqual(len(samples), 2 * 4 * 3)
        for source in results:
            details = source["pipeline_details"]["operators"]
            total_samples = [source["samples"]["fused_us"],
                             source["samples"]["extend_layer_us"][0],
                             source["samples"]["extend_layer_us"][1],
                             source["samples"]["extend_total_us"]]
            for role, operator, totals in zip(ROLES, details, total_samples):
                selected = [row for row in samples if row["case"] == source["case"] and
                            row["operator"] == role]
                self.assertEqual([int(row["sample_index"]) for row in selected], [1, 2, 3])
                expected_name = (" + ".join(operator["name"]) if isinstance(operator["name"], list)
                                 else operator["name"])
                for index, row in enumerate(selected):
                    self.assertEqual(row["shape"], source["shape"])
                    self.assertEqual(row["op_name"], expected_name)
                    self.assertEqual(float(row["total_us"]), totals[index])
                    for metric in METRICS:
                        self.assertEqual(float(row[metric + "_us"]), operator["samples"][metric][index])
        self.assertEqual([float(row["total_us"]) for row in samples
                          if row["case"] == "case_a" and row["operator"] == "fused"], [3, 4, 5])

    def test_missing_metrics_are_blank_while_zero_samples_remain_numeric(self):
        rows = profile_rows()
        for row in rows:
            row.pop("fixpipe_time(us)")
            row["mte3_exe_time(us)"] = "0"
        rows[2]["mte1_exe_time(us)"] = "N/A"
        (columns, summary), (_, samples) = self.save([measured_case(rows=rows)])
        row = summary[0]
        for role in ROLES:
            self.assertEqual(row[role + "_fixpipe_us"], "")
            self.assertEqual(float(row[role + "_mte3_us"]), 0)
        self.assertEqual(row["extend1_mte1_us"], "")
        self.assertEqual(row["extend_total_mte1_us"], "")
        self.assertGreater(float(row["extend2_mte1_us"]), 0)
        for row in samples:
            self.assertEqual(row["fixpipe_us"], "")
            self.assertEqual(float(row["mte3_us"]), 0)
            if row["operator"] in ("extend1", "extend_total") and row["sample_index"] == "1":
                self.assertEqual(row["mte1_us"], "")
            else:
                self.assertGreater(float(row["mte1_us"]), 0)

    def test_without_pipeline_details_keeps_original_summary_and_no_sample_rows(self):
        result = measured_case(pipeline=False)
        (columns, summary), (sample_columns, samples) = self.save([result])
        self.assertEqual(columns, BASE_COLUMNS)
        self.assertEqual(len(summary), 1)
        self.assertEqual(float(summary[0]["fused_us"]), result["fused_us"])
        self.assertEqual(sample_columns, SAMPLE_COLUMNS)
        self.assertEqual(samples, [])

    def test_mixed_batch_keeps_empty_columns_for_cases_without_pipeline_details(self):
        detailed, total_only = measured_case(), measured_case("case_b", pipeline=False)
        (columns, summary), (_, samples) = self.save([detailed, total_only])
        self.assertGreater(len(columns), len(BASE_COLUMNS))
        for column in columns[len(BASE_COLUMNS):]:
            self.assertEqual(summary[1][column], "")
        self.assertEqual({row["case"] for row in samples}, {"case_a"})
        self.assertEqual(len(samples), 12)

    def test_empty_batch_and_rerun_replace_previous_csv_data(self):
        self.save([measured_case()])
        (columns, summary), (sample_columns, samples) = self.save([])
        self.assertEqual(columns, BASE_COLUMNS)
        self.assertEqual(summary, [])
        self.assertEqual(sample_columns, SAMPLE_COLUMNS)
        self.assertEqual(samples, [])
        (_, summary), (_, samples) = self.save([measured_case("rerun", pipeline=False)])
        self.assertEqual([row["case"] for row in summary], ["rerun"])
        self.assertEqual(samples, [])

    def test_saving_does_not_modify_json_or_input_results(self):
        results = [measured_case(), measured_case("case_b", pipeline=False)]
        original = copy.deepcopy(results)
        self.save(results)
        self.assertEqual(results, original)
        document = json.loads((self.output / "performance_comparison.json").read_text())
        self.assertEqual(document, {"schema_version": 1, "source": "msprof_op_summary",
                                    "units": "us", "cases": original})


if __name__ == "__main__":
    unittest.main()
