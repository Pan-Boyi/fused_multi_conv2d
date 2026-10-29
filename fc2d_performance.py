# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""从设备侧 op_summary 比较融合算子与两个 ExtendConv2D 的耗时。"""
import csv
import hashlib
import math
import re
import statistics


PIPELINE_COLUMNS = (
    ("cube", ("mac_exe_time(us)", "cube_exe_time(us)", "aic_mac_time(us)")),
    ("scalar", ("scalar_exe_time(us)", "aic_scalar_time(us)")),
    ("mte1", ("mte1_exe_time(us)", "aic_mte1_time(us)")),
    ("mte2", ("mte2_exe_time(us)", "aic_mte2_time(us)")),
    ("mte3", ("mte3_exe_time(us)",)),
    ("fixpipe", ("fixpipe_time(us)", "aic_fixpipe_time(us)")),
)
PERFORMANCE_CSV_COLUMNS = ("case", "shape", "fused_us", "extend_total_us", "saved_us", "improvement_percent")
PIPELINE_ROLES = (("fused", "FusedMultiConv2d"), ("extend1", "ExtendConv2D[1]"),
                  ("extend2", "ExtendConv2D[2]"), ("extend_total", "2 x ExtendConv2D"))
PIPELINE_SAMPLE_COLUMNS = ("case", "shape", "operator", "op_name", "sample_index", "total_us") + tuple(
    metric + "_us" for metric, _ in PIPELINE_COLUMNS)


def performance_csv_columns(include_pipeline=False):
    """有流水明细时，在原性能列后追加两层耗时及各流水中位数列。"""
    if not include_pipeline:
        return PERFORMANCE_CSV_COLUMNS
    return PERFORMANCE_CSV_COLUMNS + ("extend1_us", "extend2_us") + tuple(
        role + "_" + metric + "_us" for role, _ in PIPELINE_ROLES for metric, _ in PIPELINE_COLUMNS)


def performance_csv_row(result):
    """将已统计的流水中位数展平为 CSV 行，保留原结果。"""
    row = dict(result)
    if not result.get("pipeline_details"):
        return row
    operators = {operator["label"]: operator for operator in result["pipeline_details"]["operators"]}
    for role, label in PIPELINE_ROLES:
        operator = operators[label]
        if role in ("extend1", "extend2"):
            row[role + "_us"] = operator["total_us"]
        row.update((role + "_" + metric + "_us", operator["pipeline_us"][metric])
                   for metric, _ in PIPELINE_COLUMNS)
    return row


def pipeline_sample_rows(result):
    """逐算子导出正式执行轮次的原始耗时，轮次从 1 开始。"""
    if not result.get("pipeline_details"):
        return
    operators = {operator["label"]: operator for operator in result["pipeline_details"]["operators"]}
    samples = result["samples"]
    total_samples = (samples["fused_us"], samples["extend_layer_us"][0],
                     samples["extend_layer_us"][1], samples["extend_total_us"])
    for (role, label), totals in zip(PIPELINE_ROLES, total_samples):
        operator = operators[label]
        name = " + ".join(operator["name"]) if isinstance(operator["name"], list) else operator["name"]
        for index, total in enumerate(totals):
            row = {"case": result["case"], "shape": result["shape"], "operator": role,
                   "op_name": name, "sample_index": index + 1, "total_us": total}
            row.update((metric + "_us", operator["samples"][metric][index]) for metric, _ in PIPELINE_COLUMNS)
            yield row


def _column_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(columns, aliases):
    actual = {_column_key(column): column for column in columns}
    return next((actual[_column_key(alias)] for alias in aliases
                 if _column_key(alias) in actual), None)


def _operator_family(value):
    return {"fusedmulticonv2d": "fused", "extendconv2d": "extend"}.get(_column_key(value))


def _name_family(value):
    # 无类型列时，仅接受名称中完整的算子字段。
    parts = re.split(r"[/.:]", str(value))
    families = set()
    for part in parts:
        match = re.fullmatch(r"(FusedMultiConv2[dD]|ExtendConv2[dD]|fused_multi_conv2d|extend_conv2d)(?:_\d+)?",
                             part, re.IGNORECASE)
        if match:
            families.add(_operator_family(match.group(1)))
    if len(families) > 1:
        raise ValueError("ambiguous operator family in name: %s" % value)
    return next(iter(families)) if families else None


def _number(value, label, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("invalid %s: %r" % (label, value))
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("invalid %s: %r" % (label, value))
    return result


def _expected_operator_names(expected_nodes):
    if expected_nodes is None:
        return {}
    fused, extend = expected_nodes.get("fused"), expected_nodes.get("extend")
    if (not isinstance(fused, str) or not fused.strip() or not isinstance(extend, (list, tuple)) or
            len(extend) != 2 or any(not isinstance(name, str) or not name.strip() for name in extend) or
            len(set([fused] + list(extend))) != 3):
        raise ValueError("expected_nodes requires one fused name and two distinct ExtendConv2D names")
    return dict([(fused, "fused")] + [(name, "extend") for name in extend])


def _profile_tasks(rows, expected_nodes):
    if not rows:
        raise ValueError("op_summary contains no rows")
    columns = list(dict.fromkeys(column for row in rows for column in row))
    duration = _find_column(columns, ("total_exe_time(us)", "Task Duration(us)", "Duration(us)"))
    op_type = _find_column(columns, ("Op Type", "OP Type", "OpType", "op_type"))
    name = _find_column(columns, ("Op Name", "OpName", "Node Name", "NodeName", "Name"))
    timestamp = _find_column(columns, ("Task Start Time(us)", "Start Time(us)", "Begin Time(us)"))
    if duration is None or name is None:
        raise ValueError("op_summary requires device duration and operator identity columns; actual columns: %s" %
                         ", ".join(map(str, columns)))
    expected = _expected_operator_names(expected_nodes)
    tasks = []
    for index, row in enumerate(rows):
        identity = str(row.get(name, "")).strip()
        type_value = str(row.get(op_type, "")).strip() if op_type else ""
        if type_value.lower() in ("n/a", "na", "-"):
            type_value = ""
        family = _operator_family(type_value) if type_value else _name_family(identity)
        if expected:
            if identity in expected:
                if type_value and family != expected[identity]:
                    raise ValueError("operator %s has type %s inconsistent with its compiled graph binding" %
                                     (identity, type_value))
                family = expected[identity]
            elif family:
                raise ValueError("unexpected target operator %s absent from compiled graph bindings" % identity)
        if family is None:
            continue
        if not identity:
            raise ValueError("target operator has no identity at row %d" % (index + 1))
        tasks.append({"family": family, "name": identity, "duration_us":
                      _number(row.get(duration), "device duration", positive=True), "row": index,
                      "timestamp_us": _number(row.get(timestamp), "task timestamp") if timestamp else None})
    if not tasks:
        raise ValueError("op_summary contains no FusedMultiConv2d or ExtendConv2D tasks")
    return tasks, duration, timestamp


def _group_tasks(tasks, execution_count, timestamp):
    operators = {"fused": {}, "extend": {}}
    for task in tasks:
        operators[task["family"]].setdefault(task["name"], []).append(task)
    if len(operators["fused"]) != 1 or len(operators["extend"]) != 2:
        raise ValueError("expected one FusedMultiConv2d and exactly two distinct ExtendConv2D operators")
    for group in operators.values():
        for name, samples in group.items():
            if len(samples) != execution_count:
                raise ValueError("operator %s has %d executions; expected %d including warmup and write probe" %
                                 (name, len(samples), execution_count))
            if timestamp:
                samples.sort(key=lambda task: task["timestamp_us"])
                times = [task["timestamp_us"] for task in samples]
                if len(set(times)) != len(times):
                    raise ValueError("operator %s has ambiguous duplicate task timestamps" % name)
    return operators


def _order_extend_pair(operators, timestamp):
    key = "timestamp_us" if timestamp else "row"
    extend = sorted(operators["extend"], key=lambda name: operators["extend"][name][0][key])
    if timestamp:
        ordered = sorted((task for group in operators["extend"].values() for task in group),
                         key=lambda task: task[key])
        expected = extend * len(operators["extend"][extend[0]])
        if [task["name"] for task in ordered] != expected:
            raise ValueError("ExtendConv2D task timestamps do not describe sequential paired executions")
        fused = next(iter(operators["fused"].values()))
        if ordered[-1][key] >= fused[0][key]:
            raise ValueError("reference and fused execution phases overlap or are reversed")
    return extend


def _pipeline_number(value, metric, name):
    """解析指定算子的流水耗时，缺失值保留为空。"""
    if value is None or str(value).strip().lower() in ("", "n/a", "na", "-"):
        return None
    label = "%s pipeline duration for operator %s" % (metric, name)
    result = _number(value, label)
    if result < 0:
        raise ValueError("invalid %s: %r" % (label, value))
    return result


def _complete_median(samples):
    """仅在全部采样齐全时计算中位数。"""
    return None if any(value is None for value in samples) else statistics.median(samples)


def _pipeline_operator(label, name, tasks, rows, columns):
    """按已选执行轮次汇总一个算子的各条流水。"""
    samples = {metric: [_pipeline_number(rows[task["row"]].get(column), metric, name)
                        for task in tasks] if column else [None] * len(tasks)
               for metric, column in columns.items()}
    return {"label": label, "name": name,
            "total_us": statistics.median([task["duration_us"] for task in tasks]),
            "pipeline_us": {metric: _complete_median(values) for metric, values in samples.items()},
            "samples": samples}


def _pipeline_details(rows, operators, fused, extend, selected, extend_us):
    """汇总三个算子，并逐轮相加得到两个 Extend 的流水总耗时。"""
    columns = list(dict.fromkeys(column for row in rows for column in row))
    pipeline_columns = {metric: _find_column(columns, aliases) for metric, aliases in PIPELINE_COLUMNS}
    result = [_pipeline_operator("FusedMultiConv2d", fused, operators["fused"][fused][selected],
                                 rows, pipeline_columns)]
    result.extend(_pipeline_operator("ExtendConv2D[%d]" % (index + 1), name,
                                    operators["extend"][name][selected], rows, pipeline_columns)
                  for index, name in enumerate(extend))
    pair_samples = {}
    for metric in pipeline_columns:
        pair_samples[metric] = [None if first is None or second is None else
                                _pipeline_number(first + second, metric, " + ".join(extend))
                                for first, second in zip(result[1]["samples"][metric],
                                                         result[2]["samples"][metric])]
    result.append({"label": "2 x ExtendConv2D", "name": extend, "total_us": extend_us,
                   "pipeline_us": {metric: _complete_median(values)
                                   for metric, values in pair_samples.items()},
                   "samples": pair_samples})
    return {"units": "us", "method": "median_of_per_execution_pipeline_durations",
            "pair_method": "sum_per_execution_before_median", "columns": pipeline_columns,
            "operators": result}


def parse_profile_rows(rows, warmup, repeat, expected_nodes=None, pipeline_details=False):
    """排除预热和末次写出探针，逐轮相加两个 Extend，再分别取中位数。"""
    if type(warmup) is not int or warmup < 0 or type(repeat) is not int or repeat < 1:
        raise ValueError("warmup must be nonnegative and repeat must be positive integers")
    rows = list(rows)
    tasks, duration, timestamp = _profile_tasks(rows, expected_nodes)
    execution_count = warmup + repeat + 1
    operators = _group_tasks(tasks, execution_count, timestamp)
    extend = _order_extend_pair(operators, timestamp)
    fused = next(iter(operators["fused"]))
    selected = slice(warmup, warmup + repeat)
    fused_samples = [task["duration_us"] for task in operators["fused"][fused]][selected]
    extend_samples = [[task["duration_us"] for task in operators["extend"][name]][selected]
                      for name in extend]
    totals = [_number(first + second, "paired device duration", positive=True)
              for first, second in zip(*extend_samples)]
    fused_us = _number(statistics.median(fused_samples), "median fused duration", positive=True)
    extend_us = _number(statistics.median(totals), "median Extend duration", positive=True)
    saved_us = extend_us - fused_us
    improvement = _number(100.0 * (saved_us / extend_us), "improvement percentage")
    result = {"schema_version": 1, "source": "msprof_op_summary", "units": "us",
            "method": "median_of_per_execution_device_durations", "duration_column": duration,
            "ordering": "task_timestamp" if timestamp else "csv_order_per_operator",
            "warmup": warmup, "repeat": repeat, "write_probe_executions": 1,
            "execution_count_per_model": execution_count,
            "operators": {"fused": fused, "extend": extend},
            "samples": {"fused_us": fused_samples, "extend_layer_us": extend_samples,
                        "extend_total_us": totals},
            "fused_us": fused_us, "extend_total_us": extend_us, "saved_us": saved_us,
            "improvement_percent": improvement}
    if pipeline_details:
        result["pipeline_details"] = _pipeline_details(rows, operators, fused, extend, selected, extend_us)
    return result


def sha256_file(path):
    """记录实际参与解析的设备性能文件。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_profile_csv(path, warmup, repeat, expected_nodes=None, pipeline_details=False):
    """读取 msprof 导出文件，并在指标中绑定原始 CSV 的 SHA256。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("op_summary has no CSV header")
        result = parse_profile_rows(reader, warmup, repeat, expected_nodes=expected_nodes,
                                    pipeline_details=pipeline_details)
    result["op_summary_sha256"] = sha256_file(path)
    return result


def format_case_shape(case):
    """展示数据类型及输入、中间结果、输出的 NCHW 形状。"""
    kernel = list(case["kernel"])
    if len(kernel) == 2:
        kernel *= 2
    strides, pads = case["strides"], case["pads"]
    n, ci, hi, wi = (case[key] for key in ("n", "ci", "hi", "wi"))
    mid_h = (hi + pads[0] + pads[1] - kernel[0]) // strides[0] + 1
    mid_w = (wi + pads[2] + pads[3] - kernel[1]) // strides[1] + 1
    out_h = (mid_h + pads[4] + pads[5] - kernel[2]) // strides[2] + 1
    out_w = (mid_w + pads[6] + pads[7] - kernel[3]) // strides[3] + 1
    shape = "%dx%dx%dx%d -> %dx%dx%dx%d -> %dx%dx%dx%d" % (
        n, ci, hi, wi, n, case["cout1"], mid_h, mid_w, n, case["cout2"], out_h, out_w)
    return "%s %s" % (case["dtype"], shape)


def render_performance_table(rows):
    """用同一张表展示所有成功上板用例，节省耗时和提升百分比允许为负。"""
    headers = ("Case", "Shape", "FusedMultiConv2d (us)", "2 x ExtendConv2D (us)", "Saved (us)", "Improvement")
    data = []
    for row in rows:
        data.append((str(row["case"]), str(row["shape"]), "%.3f" % row["fused_us"],
                     "%.3f" % row["extend_total_us"], "%.3f" % row["saved_us"],
                     "%.2f%%" % row["improvement_percent"]))
    widths = [max(len(headers[index]), max((len(row[index]) for row in data), default=0))
              for index in range(len(headers))]
    def line(values):
        return " | ".join(value.ljust(widths[index]) if index < 2 else value.rjust(widths[index])
                          for index, value in enumerate(values))
    return "\n".join([line(headers), "-+-".join("-" * width for width in widths)] +
                     [line(row) for row in data])


def render_pipeline_table(result):
    """逐算子展示采样中位数，缺失的流水指标保留为 N/A。"""
    headers = ("Operator", "Total (us)", "Cube (us)", "Scalar (us)", "MTE1 (us)",
               "MTE2 (us)", "MTE3 (us)", "Fixpipe (us)")
    data = []
    for operator in result["pipeline_details"]["operators"]:
        values = [operator["total_us"]] + [operator["pipeline_us"][metric]
                                         for metric, _ in PIPELINE_COLUMNS]
        data.append([operator["label"]] + ["N/A" if value is None else "%.3f" % value for value in values])
    widths = [max(len(header), max((len(row[index]) for row in data), default=0))
              for index, header in enumerate(headers)]
    def line(values):
        return " | ".join(value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
                          for index, value in enumerate(values))
    return "\n".join([line(headers), "-+-".join("-" * width for width in widths)] +
                     [line(row) for row in data] +
                     ["Pipeline times may overlap and need not sum to Total. N/A: unavailable or incomplete."])
