# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""从 case.bin 的真实输入计算两层卷积参考，不读取已有的 golden 张量。"""

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
    parser.add_argument("case", help="ABI 8 case.bin path")
    parser.add_argument("--output", required=True, help="Raw NCHW y_expect output path")
    parser.add_argument("--exact-output", help="Raw NCHW y_exact output path, FP16 only")
    parser.add_argument("--workspace-mb", type=float, default=64, help="Convolution patch workspace in MiB")
    args = parser.parse_args(argv)
    info, tensors = read_case(args.case)
    result = compute_golden(info, tensors, args.workspace_mb)
    if args.exact_output and "y_exact" not in result:
        parser.error("--exact-output is supported only for FP16 cases")
    outputs = [("y_expect", args.output)]
    if args.exact_output:
        outputs.append(("y_exact", args.exact_output))
    if os.path.abspath(args.case) in [os.path.abspath(path) for _, path in outputs]:
        parser.error("Output path must not overwrite the case file")
    for name, path in outputs:
        with open(path, "wb") as target:
            target.write(result[name][2])
    print(json.dumps({"output": args.output, "dtype": result["y_expect"][0],
                      "shape": result["y_expect"][1], "stats": result["stats"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
