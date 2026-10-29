# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""从 case.bin 或 JSON 用例描述计算两层卷积参考。"""

import argparse
import json
import math
import os
import struct

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
    """将可配置属性映射成两层卷积的计算参数。"""
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
    """按 NCHW 输入和 OIHW 双权重描述推导计算参数及输出形状。"""
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


def _normalise_json_case(raw, index):
    """补全旧格式的默认参数，并校验两层卷积的形状和定标范围。"""
    defaults = dict(dtype="fp16", n=1, ci=32, hi=288, wi=112, cout1=64, cout2=96,
                    kernel=[3, 3], strides=[1] * 4, pads=[0] * 8, bias=False,
                    enable_relu=[True, True], fixed_shift_value=None,
                    data_format=[0, 0], round_mode=[0, 0], pad_mode=[0, 0])
    _json_object(raw, "case", tuple(defaults) + ("name",))
    case = dict(defaults, **raw)
    case["name"] = raw.get("name", "case%02d" % index)
    if not isinstance(case["name"], str) or not case["name"]:
        raise ValueError("name must be a nonempty string")
    mode = case["dtype"]
    if mode not in ("fp16", "int8", "s8f16", "a16w8", "both"):
        raise ValueError("dtype must be fp16, int8, s8f16, a16w8 or both")
    if "fixed_shift_value" not in raw:
        case["fixed_shift_value"] = list({"fp16": (42, 42), "both": (42, 42),
                                          "a16w8": (13, 0)}.get(mode, (0, 0)))
    _json_shape([case[key] for key in ("n", "ci", "hi", "wi")], "input.shape")
    _json_int_list([case["cout1"], case["cout2"]], "output channels", 2, 1)
    kernel = case["kernel"]
    if not isinstance(kernel, list) or len(kernel) not in (2, 4):
        raise ValueError("kernel requires [Kh,Kw] or [Kh1,Kw1,Kh2,Kw2]")
    case["kernel"] = _json_int_list(kernel, "kernel", len(kernel), 1) * (2 if len(kernel) == 2 else 1)
    if type(case["bias"]) not in (bool, int) or case["bias"] not in (0, 1):
        raise ValueError("bias must be a bool or integer 0/1")
    attrs = {key: case[key] for key in ("strides", "pads", "enable_relu", "fixed_shift_value",
                                      "data_format", "round_mode", "pad_mode")}
    attrs["bias_mask"] = [bool(case["bias"])] * 2
    mapped, _ = _json_tensor_attrs(attrs)
    case.update(mapped)
    if mode == "a16w8" and case["fixed_shift_value"][0] > 29:
        raise ValueError("A16W8 fixed_shift_value[0] must be in [0, 29]")
    return case


def _select_json_case(document, only):
    """按原默认参数和 both 展开规则，选择一条 JSON 用例。"""
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
        case = _normalise_json_case(raw, index)
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


def read_json_case(path, only=None, workspace_mb=64):
    """仅用 Python 和 NumPy 生成 JSON 用例的确定性输入。"""
    with open(path, encoding="utf-8") as source:
        case = _select_json_case(json.load(source), only)
    if not math.isfinite(workspace_mb) or workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive and finite")
    info, tensors = _generate_json_inputs(case, workspace_mb)
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


def _json_random_values(shape, salt, half=False):
    """按原 Mix64 序列生成逻辑张量，固定分块限制随机数临时空间。"""
    values = np.empty(_element_count(shape), dtype=np.float16 if half else np.int8)
    for start in range(0, values.size, 1024 * 1024):
        stop = min(start + 1024 * 1024, values.size)
        mixed = np.arange(start, stop, dtype=np.uint64)
        mixed += np.uint64(salt + 0x9e3779b97f4a7c15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        mixed ^= mixed >> np.uint64(31)
        if half:
            grid = (mixed % np.uint64(512)).astype(np.int16) - 256
            values[start:stop] = grid.astype(np.float32) / np.float32(256)
        else:
            values[start:stop] = (mixed % np.uint64(201)).astype(np.int16) - 100
    return values.reshape(shape)


def _json_tensor_record(dtype, values):
    """将生成值保存为兼容 case.bin 的小端逻辑张量记录。"""
    values = np.asarray(values, dtype=NUMPY_DTYPES[dtype])
    return dtype, list(values.shape), values.tobytes()


def _json_generation_info(case):
    """由规范化 JSON 参数构造 ABI 8 形状信息，并检查通道对齐。"""
    mode = {"fp16": 0, "int8": 1, "s8f16": 2, "a16w8": 3}[case["dtype"]]
    info = dict.fromkeys(SPEC_KEYS, 0)
    info.update({name: case[name] for name in ("n", "ci", "hi", "wi", "cout1", "cout2")})
    info.update(kh=case["kernel"][0], kw=case["kernel"][1],
                kh2=case["kernel"][2], kw2=case["kernel"][3],
                s1=case["strides"][0], s2=case["strides"][2],
                ph1=case["pads"][0], pw1=case["pads"][2],
                ph2=case["pads"][4], pw2=case["pads"][6],
                bias=int(bool(case["bias"])), relu1=int(bool(case["enable_relu"][0])),
                relu2=int(bool(case["enable_relu"][1])),
                shift1=case["fixed_shift_value"][0], shift2=case["fixed_shift_value"][1],
                elem_bytes=2 if mode in (0, 3) else 1,
                out_fp16=int(mode in (2, 3)), dtype_mode=mode)
    x_alignment = 16 if info["elem_bytes"] == 2 else 32
    weight_alignment = 16 if mode == 0 else 32
    if info["ci"] % x_alignment or info["ci"] % weight_alignment or info["cout1"] % weight_alignment:
        raise ValueError("Input and intermediate channels do not meet tensor alignment requirements")
    height, width = info["hi"], info["wi"]
    for layer in (1, 2):
        kernel_height = info["kh"] if layer == 1 else info["kh2"]
        kernel_width = info["kw"] if layer == 1 else info["kw2"]
        height = (height + 2 * info["ph%d" % layer] - kernel_height) // info["s%d" % layer] + 1
        width = (width + 2 * info["pw%d" % layer] - kernel_width) // info["s%d" % layer] + 1
        if min(height, width) <= 0:
            raise ValueError("Convolution %d has an empty output" % layer)
    info.update(ho2=height, wo2=width)
    return info


def _json_effective_scale(values):
    """按硬件表项精度清除 float32 scale 的低 13 位尾数。"""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32) & np.uint32(0xffffe000)
    return bits.view(np.float32)


def _json_pick_scale(accumulators):
    """根据整个 batch 的累加峰值选择原生成器相同的 int8 scale。"""
    peak = int(np.max(np.abs(accumulators.astype(np.int64))))
    scale = np.float32(1) if peak == 0 else np.float32(127) / np.float32(peak)
    return _json_effective_scale(scale)


def _json_scale_record(values, quantized=False):
    """编码逐通道 scale；重量化表额外设置有符号饱和 bit 46。"""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    if quantized:
        bits |= np.uint64(1 << 46)
    return _json_tensor_record(ACL_UINT64, bits)


def _json_safe_exponent(input_values, weights, bias, stride, pad_height, pad_width, workspace_mb):
    """以绝对乘积和加 bias 为部分和上界，求最大安全定标指数。"""
    absolute_sum = _conv(np.abs(input_values), np.abs(weights), stride,
                         pad_height, pad_width, workspace_mb)
    absolute_sum += np.abs(bias.astype(np.float64))[None, :, None, None]
    peak = float(np.max(absolute_sum))
    if peak <= 0:
        return 58
    exponent = 0
    while exponent < 58 and math.ldexp(peak, exponent + 1) < 2147483000.0:
        exponent += 1
    return exponent


def _json_fp16_inputs(info, tensors, arrays, workspace_mb):
    """生成两层 FP16 定点结果、安全指数，并保留原工作点自检。"""
    inputs, weight1, bias1, weight2, bias2 = arrays
    acc1 = _conv(inputs, weight1, info["s1"], info["ph1"], info["pw1"], workspace_mb)
    info["safe_f1"] = _json_safe_exponent(inputs, weight1, bias1, info["s1"],
                                         info["ph1"], info["pw1"], workspace_mb)
    mid_bits, _ = _fixed_epilogue(acc1, bias1, info["shift1"], info["relu1"])
    mid = mid_bits.view(np.float16)
    acc2 = _conv(mid, weight2, info["s2"], info["ph2"], info["pw2"], workspace_mb)
    info["safe_f2"] = _json_safe_exponent(mid, weight2, bias2, info["s2"],
                                         info["ph2"], info["pw2"], workspace_mb)
    output, _ = _fixed_epilogue(acc2, bias2, info["shift2"], info["relu2"])
    exact = _exact_epilogue(acc2, bias2, info["relu2"])
    expected_values = output.view(np.float16).astype(np.float64)
    exact_values = exact.view(np.float16).astype(np.float64)
    denominator = np.where(np.abs(exact_values) > 1e-6, np.abs(exact_values), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = np.count_nonzero(np.abs(expected_values - exact_values) / denominator > 1e-2)
    if bad * 1000 > output.size:
        raise ValueError("FP16 fixed-point and exact golden disagree; choose valid fixed_shift_value")
    tensors["y_expect"] = _result_tensor(output)
    tensors["y_exact"] = _result_tensor(exact)


def _json_integer_inputs(info, tensors, arrays, workspace_mb):
    """按精确整数卷积选择量化表，支持 int8、s8f16 和 A16W8。"""
    inputs, weight1, bias1, weight2, bias2 = arrays
    acc1 = _conv(inputs, weight1, info["s1"], info["ph1"], info["pw1"], workspace_mb)
    acc1 = (acc1 + bias1[None, :, None, None]).astype(np.int32)
    scale1 = _json_pick_scale(acc1)
    scale1_channels = np.full(info["cout1"], scale1, dtype=np.float32)
    tensors["quant_scale1"] = _json_scale_record(scale1_channels, True)
    mid = _requantize(acc1, scale1_channels[None, :, None, None], info["relu1"])
    acc2 = _conv(mid, weight2, info["s2"], info["ph2"], info["pw2"], workspace_mb)
    acc2 = (acc2 + bias2[None, :, None, None]).astype(np.int32)
    if info["dtype_mode"] == 1:
        scale2 = np.full(info["cout2"], _json_pick_scale(acc2), dtype=np.float32)
        tensors["quant_scale2"] = _json_scale_record(scale2, True)
        output = _requantize(acc2, scale2[None, :, None, None], info["relu2"])
    else:
        peak = np.max(np.abs(acc2.astype(np.int64)), axis=(0, 2, 3))
        base = np.ones(info["cout2"], dtype=np.float32)
        np.divide(np.float32(8), peak.astype(np.float32), out=base, where=peak != 0)
        factor = np.float32(1) + np.float32(0.05) * (np.arange(info["cout2"]) % 7).astype(np.float32)
        scale2 = _json_effective_scale(base * factor)
        tensors["dequant_scale2"] = _json_scale_record(scale2)
        output = _dequantize(acc2, scale2[None, :, None, None], info["relu2"])
    tensors["y_expect"] = _result_tensor(output)


def _generate_json_inputs(case, workspace_mb=64):
    """直接从 JSON 参数生成原 C++ 序列的输入和参考，不依赖编译器。"""
    if not math.isfinite(workspace_mb) or workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive and finite")
    info = _json_generation_info(case)
    mode = info["dtype_mode"]
    shapes = [(info["n"], info["ci"], info["hi"], info["wi"]),
              (info["cout1"], info["ci"], info["kh"], info["kw"]),
              (info["cout2"], info["cout1"], info["kh2"], info["kw2"])]
    salts = (0x41, 0x42, 0x43) if mode == 0 else (0x11, 0x22, 0x33)
    inputs, weight1, weight2 = [_json_random_values(shape, salt, mode == 0)
                                for shape, salt in zip(shapes, salts)]
    if mode == 3:
        multiple = 1 << max(0, 5 - info["shift1"])
        integers = inputs.astype(np.int16)
        inputs = (np.sign(integers) * (np.abs(integers) // multiple) * multiple).astype(np.int8)
        scaled = np.ldexp(inputs.astype(np.float32), info["shift1"] - 29)
        half = scaled.astype(np.float16)
        if not np.all(half.astype(np.float32) == scaled):
            raise ValueError("A16W8 exact golden requires representable scaled inputs")
        input_record = _json_tensor_record(ACL_FLOAT16, half)
    else:
        input_record = _json_tensor_record(ACL_FLOAT16 if mode == 0 else ACL_INT8, inputs)
    bias_dtype = np.float16 if mode == 0 else np.int32
    bias1 = np.zeros(info["cout1"], dtype=bias_dtype)
    bias2 = np.zeros(info["cout2"], dtype=bias_dtype)
    if info["bias"]:
        for layer, bias in enumerate((bias1, bias2)):
            grid = np.arange(bias.size, dtype=np.int64) - bias.size // 2
            bias[:] = grid.astype(np.float32) / np.float32(64) if mode == 0 else grid * (137 if layer == 0 else 91)
    weight_dtype = ACL_FLOAT16 if mode == 0 else ACL_INT8
    tensors = {"x": input_record, "filter1": _json_tensor_record(weight_dtype, weight1),
               "filter2": _json_tensor_record(weight_dtype, weight2),
               "bias1": _json_tensor_record(ACL_FLOAT16 if mode == 0 else ACL_INT32, bias1),
               "bias2": _json_tensor_record(ACL_FLOAT16 if mode == 0 else ACL_INT32, bias2)}
    arrays = inputs, weight1, bias1, weight2, bias2
    if mode == 0:
        _json_fp16_inputs(info, tensors, arrays, workspace_mb)
    else:
        _json_integer_inputs(info, tensors, arrays, workspace_mb)
    tensors["w1_nchw"] = tensors["filter1"]
    return info, tensors


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
            info, tensors = read_json_case(args.case, args.only, args.workspace_mb)
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
