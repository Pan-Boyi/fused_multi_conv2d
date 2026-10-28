# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""从 case.bin 或 JSON 用例描述计算两层卷积参考。"""

import argparse
import contextlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


CASE_VERSION = 8
SPEC_KEYS = (
    "n", "ci", "hi", "wi", "cout1", "cout2", "kh", "kw", "s1", "s2",
    "ph1", "pw1", "ph2", "pw2", "elem_bytes", "bias", "relu1", "relu2",
    "shift1", "shift2", "ho2", "wo2", "safe_f1", "safe_f2", "out_fp16",
    "dtype_mode", "kh2", "kw2",
)
HEADER_FORMAT = "<8sIIQQQ32iII"
RECORD_FORMAT = "<16sII4qQ"
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT64 = 10
NUMPY_DTYPES = {
    ACL_FLOAT16: np.dtype("<f2"),
    ACL_INT8: np.dtype("i1"),
    ACL_INT32: np.dtype("<i4"),
    ACL_UINT64: np.dtype("<u8"),
}
INT32_MIN = -2147483648
INT32_MAX = 2147483647


def _element_count(shape):
    """计算张量元素数，兼容没有 math.prod 的 Python。"""
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def read_case(path):
    """读取 ABI 8 用例，返回形状属性和 (dtype, dims, raw) 张量表。"""
    with open(path, "rb") as source:
        blob = source.read()
    header_size = struct.calcsize(HEADER_FORMAT)
    if len(blob) < header_size:
        raise ValueError("Truncated case header")
    header = struct.unpack_from(HEADER_FORMAT, blob)
    if header[0] != b"FC2DCASE" or header[1] != CASE_VERSION:
        raise ValueError("Expected FC2DCASE ABI %d" % CASE_VERSION)
    info = dict(zip(SPEC_KEYS, header[6:38]))
    tensors = {}
    offset = header_size
    record_size = struct.calcsize(RECORD_FORMAT)
    for _ in range(header[2]):
        if offset + record_size > len(blob):
            raise ValueError("Truncated tensor header")
        name, dtype, rank, *record = struct.unpack_from(RECORD_FORMAT, blob, offset)
        offset += record_size
        name = name.rstrip(b"\0").decode("ascii")
        dims, byte_count = list(record[:rank]), record[4]
        if rank > 4 or dtype not in NUMPY_DTYPES or name in tensors:
            raise ValueError("Invalid tensor descriptor: %s" % name)
        expected = _element_count(dims) * NUMPY_DTYPES[dtype].itemsize
        if expected != byte_count or offset + byte_count > len(blob):
            raise ValueError("Invalid tensor data length: %s" % name)
        tensors[name] = (dtype, dims, blob[offset:offset + byte_count])
        offset += byte_count
    if offset != len(blob):
        raise ValueError("Unexpected bytes after case tensors")
    return info, tensors


def _json_object(value, name, fields, required=()):
    """检查 JSON 对象的字段范围和必填字段。"""
    if not isinstance(value, dict):
        raise ValueError("%s must be an object" % name)
    unknown = sorted(set(value) - set(fields))
    if unknown:
        raise ValueError("Unknown %s fields: %s" % (name, ", ".join(unknown)))
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError("Missing %s fields: %s" % (name, ", ".join(missing)))
    return value


def _json_shape(value, name):
    """校验四维逻辑形状，每个维度使用正整数。"""
    if (not isinstance(value, list) or len(value) != 4 or
            any(type(item) is not int or not 0 < item <= INT32_MAX for item in value)):
        raise ValueError("%s must contain four positive integers" % name)
    return value


def _json_dtype(value, name):
    """统一 JSON 中 fp16、float16 和 int8 的写法。"""
    aliases = {"fp16": "fp16", "float16": "fp16", "int8": "int8"}
    if not isinstance(value, str) or value.lower() not in aliases:
        raise ValueError("Unsupported %s: %r; expected fp16, float16 or int8" % (name, value))
    return aliases[value.lower()]


def _json_int_list(value, name, length, minimum=0):
    """校验逐层整型属性的长度和范围。"""
    if (not isinstance(value, list) or len(value) != length or
            any(type(item) is not int or not minimum <= item <= INT32_MAX for item in value)):
        raise ValueError("%s requires %d integers in [%d, %d]" %
                         (name, length, minimum, INT32_MAX))
    return value


def _json_default_attrs(attrs):
    """拒绝当前 golden 未实现的格式、膨胀和分组等配置。"""
    enum_defaults = {"data_format": "NCHW", "round_mode": "rint", "pad_mode": "SPECIFIC"}
    for name, default in enum_defaults.items():
        values = attrs.get(name, [0, 0])
        if (not isinstance(values, list) or len(values) != 2 or
                any(not ((type(item) is int and item == 0) or
                         (isinstance(item, str) and item.lower() == default.lower()))
                    for item in values)):
            raise ValueError("attrs.%s supports only [0, 0] (%s for both layers)" % (name, default))
    defaults = {"dilations": [1] * 8, "groups": [1, 1], "offset_x": [0, 0]}
    for name, default in defaults.items():
        values = _json_int_list(attrs.get(name, default), "attrs." + name, len(default))
        if values != default:
            raise ValueError("attrs.%s supports only %s" % (name, default))
    hf32 = attrs.get("enable_hf32", [False, False])
    if (not isinstance(hf32, list) or len(hf32) != 2 or
            any(type(item) is not bool or item for item in hf32)):
        raise ValueError("attrs.enable_hf32 supports only [false, false]")


def _json_tensor_attrs(attrs):
    """将可配置属性映射成原输入生成器的逐层参数。"""
    fields = ("strides", "pads", "enable_relu", "bias_mask", "fixed_shift_value",
              "data_format", "round_mode", "pad_mode", "dilations", "groups",
              "offset_x", "enable_hf32")
    _json_object(attrs, "attrs", fields)
    _json_default_attrs(attrs)
    strides = _json_int_list(attrs.get("strides", [1] * 4), "attrs.strides", 4, 1)
    pads = _json_int_list(attrs.get("pads", [0] * 8), "attrs.pads", 8)
    if strides[0] != strides[1] or strides[2] != strides[3]:
        raise ValueError("attrs.strides requires equal H/W strides within each convolution")
    if any(pads[index] != pads[index + 1] for index in range(0, 8, 2)):
        raise ValueError("attrs.pads requires symmetric padding on each axis")
    relu = attrs.get("enable_relu", [True, True])
    if (not isinstance(relu, list) or len(relu) != 2 or
            any(type(item) not in (bool, int) or item not in (0, 1) for item in relu)):
        raise ValueError("attrs.enable_relu requires two bool values or integer modes 0/1")
    mask = attrs.get("bias_mask", [False, False])
    if (not isinstance(mask, list) or len(mask) != 2 or
            any(type(item) is not bool for item in mask)):
        raise ValueError("attrs.bias_mask requires two bool values")
    mapped = dict(strides=strides, pads=pads, enable_relu=relu, bias=any(mask))
    if "fixed_shift_value" in attrs:
        shifts = _json_int_list(attrs["fixed_shift_value"], "attrs.fixed_shift_value", 2)
        if any(item > 58 for item in shifts):
            raise ValueError("attrs.fixed_shift_value requires two integers in [0, 58]")
        mapped["fixed_shift_value"] = shifts
    return mapped, mask


def _json_tensor_case(raw):
    """按 NCHW 输入和 OIHW 双权重描述推导原生成器参数及输出形状。"""
    _json_object(raw, "case", ("name", "input", "kernels", "output", "attrs"),
                 ("input", "kernels", "output"))
    input_desc = _json_object(raw["input"], "input", ("shape", "dtype"), ("shape", "dtype"))
    output_desc = _json_object(raw["output"], "output", ("shape", "dtype"), ("dtype",))
    kernels = raw["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 2:
        raise ValueError("kernels must contain exactly two kernel objects")
    kernel_shapes, kernel_dtypes = [], []
    for index, kernel in enumerate(kernels):
        name = "kernels[%d]" % index
        _json_object(kernel, name, ("shape", "dtype"), ("shape", "dtype"))
        kernel_shapes.append(_json_shape(kernel["shape"], name + ".shape"))
        kernel_dtypes.append(_json_dtype(kernel["dtype"], name + ".dtype"))
    input_shape = _json_shape(input_desc["shape"], "input.shape")
    if kernel_shapes[0][1] != input_shape[1] or kernel_shapes[1][1] != kernel_shapes[0][0]:
        raise ValueError("Kernel input channels must match input channels and the preceding kernel output")
    if kernel_dtypes[0] != kernel_dtypes[1]:
        raise ValueError("Both kernels must have the same dtype")
    signature = (_json_dtype(input_desc["dtype"], "input.dtype"), kernel_dtypes[0],
                 _json_dtype(output_desc["dtype"], "output.dtype"))
    modes = {("fp16", "fp16", "fp16"): "fp16", ("int8", "int8", "int8"): "int8",
             ("int8", "int8", "fp16"): "s8f16", ("fp16", "int8", "fp16"): "a16w8"}
    if signature not in modes:
        raise ValueError("Unsupported input/kernel/output dtype combination: %s" % (signature,))
    mapped, mask = _json_tensor_attrs(raw.get("attrs", {}))
    mapped.update(dtype=modes[signature], n=input_shape[0], ci=input_shape[1],
                  hi=input_shape[2], wi=input_shape[3], cout1=kernel_shapes[0][0],
                  cout2=kernel_shapes[1][0], kernel=kernel_shapes[0][2:] + kernel_shapes[1][2:])
    if "name" in raw:
        if not isinstance(raw["name"], str) or not raw["name"]:
            raise ValueError("name must be a nonempty string")
        mapped["name"] = raw["name"]
    _json_output_shape(input_shape, kernel_shapes, output_desc, mapped)
    return mapped, mask


def _json_output_shape(input_shape, kernels, output_desc, case):
    """按两层卷积公式推导空间尺寸并校验可选的输出形状。"""
    height, width = input_shape[2:]
    for index, kernel in enumerate(kernels):
        top, bottom, left, right = case["pads"][index * 4:index * 4 + 4]
        stride_h, stride_w = case["strides"][index * 2:index * 2 + 2]
        height = (height + top + bottom - kernel[2]) // stride_h + 1
        width = (width + left + right - kernel[3]) // stride_w + 1
        if min(height, width) <= 0:
            raise ValueError("Convolution %d has an empty output" % (index + 1))
    expected = [input_shape[0], kernels[1][0], height, width]
    if "shape" in output_desc and _json_shape(output_desc["shape"], "output.shape") != expected:
        raise ValueError("output.shape does not match inferred shape %s" % expected)


def _select_json_case(document, only):
    """沿用 harness 的参数默认值和 both 展开规则，选择一条 JSON 用例。"""
    import fc2d

    if isinstance(document, dict):
        raw_cases = document.get("cases", [document])
    else:
        raw_cases = document
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("JSON must contain a case object or a nonempty cases array")
    cases = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError("Each JSON case must be an object")
        mask = None
        if any(field in raw for field in ("input", "kernels", "output", "attrs")):
            raw, mask = _json_tensor_case(raw)
        with contextlib.redirect_stdout(sys.stderr):
            case = fc2d.normalise(raw, index)
        if mask is not None:
            case["_bias_mask"] = mask
        if case["dtype"] == "both":
            for dtype in ("fp16", "int8"):
                variant = dict(case, dtype=dtype, name=case["name"] + "_" + dtype)
                cases.append(variant)
        else:
            cases.append(case)
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("JSON case names must be unique")
    if only is not None:
        cases = [case for case in cases if case["name"] == only]
        if not cases:
            raise ValueError("Unknown JSON case: %s" % only)
    if len(cases) != 1:
        raise ValueError("JSON contains multiple cases; select one with --only NAME")
    return cases[0]


def read_json_case(path, only=None):
    """在临时目录生成 JSON 用例的确定性输入，不调用原验证流程。"""
    import fc2d

    with open(path, encoding="utf-8") as source:
        case = _select_json_case(json.load(source), only)
    with contextlib.redirect_stdout(sys.stderr):
        fc2d.ensure_tool("gen_case")
    with tempfile.TemporaryDirectory(prefix="fc2d-golden-json-") as directory:
        case_path = os.path.join(directory, "case.bin")
        process = subprocess.run(fc2d.gen_case_argv(case, case_path),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode:
            raise ValueError("JSON input generation failed:\n%s" % process.stdout)
        info, tensors = read_case(case_path)
    for index, enabled in enumerate(case.get("_bias_mask", [True, True])):
        name = "bias%d" % (index + 1)
        if not enabled and name in tensors:
            dtype, dims, raw = tensors[name]
            tensors[name] = (dtype, dims, bytes(len(raw)))
    return info, tensors


def _tensor(tensors, name, dtype, shape=None):
    """按逻辑形状读取一个输入张量，并检查其类型和长度。"""
    actual_dtype, dims, raw = tensors[name]
    if actual_dtype != dtype or (shape is not None and tuple(dims) != tuple(shape)):
        raise ValueError("Invalid dtype or logical shape for %s" % name)
    expected = _element_count(dims) * NUMPY_DTYPES[dtype].itemsize
    if len(raw) != expected:
        raise ValueError("Invalid tensor data length: %s" % name)
    return np.frombuffer(raw, dtype=NUMPY_DTYPES[dtype]).reshape(dims)


def _half_bits(values):
    """将 float32 按最近偶数舍入到 fp16，并沿用 C++ 的 NaN 位型。"""
    values = np.asarray(values, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        bits = values.astype(np.float16).view(np.uint16)
    if np.any(np.isnan(values)):
        signs = (values.view(np.uint32) >> 16) & 0x8000
        bits = np.where(np.isnan(values), signs | 0x7e00, bits).astype(np.uint16)
    return bits


def _conv(input_values, weights, stride, pad_height, pad_width, workspace_mb):
    """按 NCHW 执行分块卷积；整数累加用 int64，浮点累加用 double。"""
    batch, channels, height, width = input_values.shape
    outputs, filter_channels, kernel_height, kernel_width = weights.shape
    if channels != filter_channels or stride <= 0 or min(pad_height, pad_width) < 0:
        raise ValueError("Invalid convolution geometry")
    out_height = (height + 2 * pad_height - kernel_height) // stride + 1
    out_width = (width + 2 * pad_width - kernel_width) // stride + 1
    if min(out_height, out_width) <= 0:
        raise ValueError("Convolution has an empty output")
    integer = input_values.dtype.kind in "iu" and weights.dtype.kind in "iu"
    accumulator_dtype = np.int64 if integer else np.float64
    reduction_size = channels * kernel_height * kernel_width
    weight_matrix = weights.reshape(outputs, reduction_size).astype(accumulator_dtype)
    bytes_per_position = reduction_size * (input_values.itemsize + 8) + outputs * 8
    positions_per_block = max(1, int(workspace_mb * 1024 * 1024) // bytes_per_position)
    result = np.empty((batch, outputs, out_height * out_width), dtype=accumulator_dtype)
    for batch_index in range(batch):
        padded = np.pad(input_values[batch_index],
                        ((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                        mode="constant")
        channel_step, row_step, column_step = padded.strides
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(channels, out_height, out_width, kernel_height, kernel_width),
            strides=(channel_step, stride * row_step, stride * column_step,
                     row_step, column_step), writeable=False)
        for start in range(0, out_height * out_width, positions_per_block):
            stop = min(start + positions_per_block, out_height * out_width)
            positions = np.arange(start, stop)
            patch = windows[:, positions // out_width, positions % out_width, :, :]
            patch = patch.transpose(0, 2, 3, 1).reshape(reduction_size, stop - start)
            result[batch_index, :, start:stop] = np.dot(
                weight_matrix, patch.astype(accumulator_dtype))
    return result.reshape(batch, outputs, out_height, out_width)


def _fixed_epilogue(accumulators, bias, shift, relu):
    """分别落格卷积和 bias，限幅 int32 后反定标成 fp16。"""
    exponent = 58 - shift
    quantized = np.rint(np.ldexp(accumulators, exponent))
    if bias is not None:
        quantized += np.rint(np.ldexp(bias.astype(np.float64), exponent))[None, :, None, None]
    saturated = int(np.count_nonzero((quantized > INT32_MAX) | (quantized < INT32_MIN)))
    quantized = np.clip(quantized, INT32_MIN, INT32_MAX)
    values = np.ldexp(quantized, -exponent).astype(np.float32)
    bits = _half_bits(values)
    if relu:
        bits = np.where(values > 0, bits, bits & 0x8000).astype(np.uint16)
    return bits, saturated


def _exact_epilogue(accumulators, bias, relu):
    """沿用 C++ 的 fixed mid，仅在第二层出口省略定点落格。"""
    values = accumulators
    if bias is not None:
        values = values + bias.astype(np.float64)[None, :, None, None]
    values = values.astype(np.float32)
    bits = _half_bits(values)
    if relu:
        bits = np.where(values > 0, bits, bits & 0x8000).astype(np.uint16)
    return bits


def _scales(tensors, name, channels):
    """读取 uint64 逐通道表的 float32 低字，并清除低 13 位尾数。"""
    table = _tensor(tensors, name, ACL_UINT64).reshape(-1)
    if table.size not in (1, channels):
        raise ValueError("Invalid per-channel scale length: %s" % name)
    bits = (table & np.uint64(0xffffe000)).astype(np.uint32)
    return bits.view(np.float32)[None, :, None, None]


def _requantize(accumulators, scales, relu):
    """使用 double 乘法、最近偶数舍入和有符号 int8 饱和完成 REQ8。"""
    values = np.rint(accumulators.astype(np.float64) * scales.astype(np.float64))
    values = np.clip(values, -128, 127).astype(np.int8)
    if relu:
        values = np.maximum(values, np.int8(0))
    return values


def _dequantize(accumulators, scales, relu):
    """先将 int32 转成 float32，再按通道乘法并窄化到 fp16。"""
    values = accumulators.astype(np.float32) * scales
    if relu:
        values = np.where(values < 0, np.float32(-0.0), values)
    return _half_bits(values)


def _a16w8_integer_input(input_values, shift):
    """恢复现有 C++ A16W8 golden 对应的精确 int8 输入网格。"""
    restored = np.ldexp(input_values.astype(np.float64), 29 - shift)
    representable = (np.isfinite(restored) & (restored == np.rint(restored)) &
                     (restored >= -128) & (restored <= 127))
    if not np.all(representable):
        raise ValueError("A16W8 CPU golden requires x * 2^(29-shift1) to be exact int8 values")
    return restored.astype(np.int8)


def _result_tensor(bits_or_int8):
    """将 NCHW 结果编码为用例张量记录需要的类型、形状和字节。"""
    if bits_or_int8.dtype == np.uint16:
        return ACL_FLOAT16, list(bits_or_int8.shape), bits_or_int8.astype("<u2").tobytes()
    return ACL_INT8, list(bits_or_int8.shape), bits_or_int8.tobytes()


def compute_golden(info, tensors, workspace_mb=64):
    """从输入独立计算 y_expect；fp16 另返回 y_exact，并附 nonzero/sat 统计。"""
    if not math.isfinite(workspace_mb) or workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive and finite")
    mode = info.get("dtype_mode", 0 if info["elem_bytes"] == 2 else 1)
    if mode not in (0, 1, 2, 3):
        raise ValueError("Unsupported dtype mode: %s" % mode)
    shift1, shift2 = info["shift1"], info["shift2"]
    if mode == 0 and not (0 <= shift1 <= 58 and 0 <= shift2 <= 58):
        raise ValueError("FP16 fixed_shift_value is out of range")
    if mode == 3 and not (0 <= shift1 <= 29 and 0 <= shift2 <= 58):
        raise ValueError("A16W8 fixed_shift_value is out of range")
    x_shape = (info["n"], info["ci"], info["hi"], info["wi"])
    kernel2 = (info.get("kh2") or info["kh"], info.get("kw2") or info["kw"])
    weight1_shape = (info["cout1"], info["ci"], info["kh"], info["kw"])
    weight2_shape = (info["cout2"], info["cout1"]) + kernel2
    input_dtype = ACL_FLOAT16 if mode in (0, 3) else ACL_INT8
    weight_dtype = ACL_FLOAT16 if mode == 0 else ACL_INT8
    x = _tensor(tensors, "x", input_dtype, x_shape)
    weight1 = _tensor(tensors, "filter1", weight_dtype, weight1_shape)
    weight2 = _tensor(tensors, "filter2", weight_dtype, weight2_shape)
    bias_dtype = ACL_FLOAT16 if mode == 0 else ACL_INT32
    bias1 = _tensor(tensors, "bias1", bias_dtype, (info["cout1"],)) if info["bias"] else None
    bias2 = _tensor(tensors, "bias2", bias_dtype, (info["cout2"],)) if info["bias"] else None
    if mode == 3:
        x = _a16w8_integer_input(x, shift1)
    acc1 = _conv(x, weight1, info["s1"], info["ph1"], info["pw1"], workspace_mb)
    saturated = 0
    if mode == 0:
        mid_bits, saturated = _fixed_epilogue(acc1, bias1, shift1, info["relu1"])
        mid = mid_bits.view(np.float16)
    else:
        if bias1 is not None:
            acc1 = acc1 + bias1.astype(acc1.dtype)[None, :, None, None]
        acc1 = acc1.astype(np.int32)
        mid = _requantize(acc1, _scales(tensors, "quant_scale1", info["cout1"]), info["relu1"])
    acc2 = _conv(mid, weight2, info["s2"], info["ph2"], info["pw2"], workspace_mb)
    result = {}
    if mode == 0:
        output, layer2_saturated = _fixed_epilogue(acc2, bias2, shift2, info["relu2"])
        saturated += layer2_saturated
        result["y_exact"] = _result_tensor(_exact_epilogue(acc2, bias2, info["relu2"]))
    else:
        if bias2 is not None:
            acc2 += bias2.astype(np.int64)[None, :, None, None]
        acc2 = acc2.astype(np.int32)
        if mode == 1:
            output = _requantize(acc2, _scales(tensors, "quant_scale2", info["cout2"]), info["relu2"])
            saturated = int(np.count_nonzero((output == -128) | (output == 127)))
        else:
            output = _dequantize(acc2, _scales(tensors, "dequant_scale2", info["cout2"]), info["relu2"])
    expected_shape = (info["n"], info["cout2"], info["ho2"], info["wo2"])
    if output.shape != expected_shape:
        raise ValueError("Output shape does not match case spec")
    result["y_expect"] = _result_tensor(output)
    magnitude = output & 0x7fff if output.dtype == np.uint16 else output
    result["stats"] = {"nonzero": int(np.count_nonzero(magnitude)), "sat": saturated}
    return result


def main(argv=None):
    """独立读取用例并输出 NCHW golden 原始数据。"""
    parser = argparse.ArgumentParser(description="Compute FusedMultiConv2d CPU golden from case inputs")
    parser.add_argument("case", help="ABI 8 case.bin or JSON case description path")
    parser.add_argument("--only", help="Select one JSON case by name")
    parser.add_argument("--output", required=True, help="Raw NCHW y_expect output path")
    parser.add_argument("--exact-output", help="Raw NCHW y_exact output path, FP16 only")
    parser.add_argument("--workspace-mb", type=float, default=64, help="Convolution patch workspace in MiB")
    args = parser.parse_args(argv)
    if args.case.lower().endswith(".json"):
        try:
            info, tensors = read_json_case(args.case, args.only)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    else:
        if args.only is not None:
            parser.error("--only is supported only for JSON input")
        info, tensors = read_case(args.case)
    result = compute_golden(info, tensors, args.workspace_mb)
    if args.exact_output and "y_exact" not in result:
        parser.error("--exact-output is supported only for FP16 cases")
    outputs = [("y_expect", args.output)]
    if args.exact_output:
        outputs.append(("y_exact", args.exact_output))
    if os.path.abspath(args.case) in [os.path.abspath(path) for _, path in outputs]:
        parser.error("Output path must not overwrite the input file")
    for name, path in outputs:
        with open(path, "wb") as target:
            target.write(result[name][2])
    print(json.dumps({"output": args.output, "dtype": result["y_expect"][0],
                      "shape": result["y_expect"][1], "stats": result["stats"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
