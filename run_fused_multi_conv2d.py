#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""FusedMultiConv2d @ 5102 单算子验证 —— 目标机器上不需要编译器。

用 ctypes 直接调 libascendcl.so 里的 aclmdl*（模型 API），输入和 golden 全部来自
gen_case 生成的 .bin。所以这台机器只要有 python3 + CANN 就够了，不需要 g++、
不需要 aclnn 头、不需要 numpy。

    python3 run_fused_multi_conv2d.py [case.bin] [op_type] [device_id] [om]
    默认  fused_multi_conv2d_case.bin  FusedMultiConv2d  0  (无 om)

第 4 个参数 om 可以是**单个 .om 文件**，也可以是装着 .om 的目录：
    ... 0 om_out/base_int8.om           加载这一个（走 aclmdlLoadFromFile）
    ... 0 om_out                        目录里只有一个 .om 时用它，多于一个就报错

环境变量：
    REL_TOL=1e-3     相对误差判据（默认 1e-3）
    RATIO_MIN=1.0    达标比例的下限，低于它算 FAIL（默认 1.0，即要求 100%）
    REPEAT=1         计时的下发次数。想避开冷启动就调大，比如 REPEAT=10
    WARMUP=0         正式计时前先空跑多少次（不计入统计）。总次数 = WARMUP + REPEAT

设备上没装带 FusedMultiConv2d 的算子包时，传第 4 个参数：那是在有算子包的机器上用
fc2d.py / run_profile.py 的 om 步骤编出来的单算子离线模型。运行时就从它里面找算子，
不再要求本机的算子信息库里有这个算子。

om 是 atc --framework=5 --model=<x.onnx> 编出来的**图** om，所以下发走
aclmdlLoadFromFile + aclmdlExecute，而不是单算子的 aclopExecuteV2。为什么不能走
单算子：filters 是 DYNAMIC list，要两个实例（conv1 的核 + conv2 的核），而
atc --singleop 给 DYNAMIC 输入只实例化一个。

图 om 必须**指名到文件** —— 没有「把目录里所有 om 都装上、让 ACL 自己匹配」这回事，
那是单算子 API（aclopSetModelDir）的玩法，也正是它造成过「目录里留着上一个形状的
om，旧的反而先匹配上」。

判据和 C++ 版完全一致，输出行可以逐字对比。
"""

import ctypes
from functools import reduce
import hashlib
import json
import math
import os
from operator import mul
import statistics
import struct
import sys
import time

# ---------------------------------------------------------------- ACL 常量
ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_FORMAT_ND = 2
ACL_DT_UNDEFINED = -1
ACL_FORMAT_UNDEFINED = -1

DTYPE_NAME = {0: "float32", 1: "float16", 2: "int8", 3: "int32", 10: "uint64"}
DTYPE_SIZE = {0: 4, 1: 2, 2: 1, 3: 4, 10: 8}

# y 改成 fp16（fixpipe 走 VDEQF16）之后，哨兵按字节填 0x7F，两字节拼起来是
# 0x7F7F —— 一个 fp16 NaN（阶码全 1、尾数非零）。golden 永远算不出 NaN，所以
# 「有没有被写过」这个判据不需要再论证合法输出取不到哨兵值。
Y_SENTINEL_BYTE = 0x7F
Y_SENTINEL_U16 = 0x7F7F   # fp16 输出的哨兵（两字节）
Y_SENTINEL_I8 = 0x7F      # int8 输出的哨兵（一字节，正好是 int8 的 127）
Y_ELEM_BYTES = 2          # fp16 那条；int8 那条用 1，见 decode_y()

# case 文件的头（version 8）。**形状全在 spec 里**，脚本不再写死任何一个维度 ——
# gen_case 写它，fc2d.py 从它生成 .onnx（再 atc 编成 om），本脚本从它取输入数据，
# 三处同源。属性不再下发 —— 图 om 的属性是编译期烘进去的。
CASE_VERSION = 8
# 末尾的 kh2 / kw2 是 conv2 的核，**只能往后加**：老的 .bin 在这两个位置上是 0，
# 而那时两层必然同核，所以退回 kh / kw 就是对的 —— 相同核可按此回退；
# ABI 8 stores filters as logical NCHW, so version 6 packed cases must be regenerated.
# 注意这张表是被 _check_spec_keys_in_sync() 正则逐字比对的，**列表里不能写注释**。
SPEC_KEYS = [
    "n", "ci", "hi", "wi", "cout1", "cout2", "kh", "kw", "s1", "s2",
    "ph1", "pw1", "ph2", "pw2", "elem_bytes", "bias", "relu1", "relu2",
    "shift1", "shift2", "ho2", "wo2", "safe_f1", "safe_f2", "out_fp16",
    "dtype_mode", "kh2", "kw2",
]
SPEC_N = 32
HDR_FMT = "<8sIIQQQ" + "%di" % SPEC_N + "II"
HDR_LEN = struct.calcsize(HDR_FMT)
REC_FMT = "<16sII4qQ"           # name, dtype, ndim, dims[4], nbytes
REC_LEN = struct.calcsize(REC_FMT)

# Case existence lists include constant filters/scales for reference generation. They do
# not define graph runtime order; model_input_slots supplies the consumed inputs.
ORDER = ["x", "filter1", "bias1", "filter2", "bias2"]
INPUT_SLOTS = ["x", "filter1", "filter2", "bias1", "bias2",
               "quant_scale1", "quant_scale2", "dequant_scale2",
               "relu_weight1", "relu_weight2"]
ORDER_INT8 = ORDER + ["quant_scale1", "quant_scale2"]
ORDER_S8F16 = ORDER + ["dequant_scale2", "quant_scale1"]
ORDER_A16W8 = ORDER_S8F16
# 两个 golden：y_expect 是定点模型（假设成立时应逐位相等），y_exact 是纯 fp32
# 参考（判「有没有在算这个卷积」，不受定点假设影响）。见 golden.h 顶部那段。
GOLDENS = ["y_expect", "y_exact"]

# 硬件移位域宽度。属性 S 和实际定标指数 F 的关系恒为 F = FIX_SHIFT_LEN - S，
# 见 dav_m510/kernel_operator_fixpipe_impl.h:77 SetDeqScalarDepOnMode。
FIX_SHIFT_LEN = 58
A16W8_SHIFT_LEN = 29

DTYPE_FP16 = 0
DTYPE_INT8 = 1
DTYPE_S8F16 = 2
DTYPE_A16W8 = 3


class AclModelDimensions(ctypes.Structure):
    """aclmdlIODims 的公开 ABI。"""
    _fields_ = [("name", ctypes.c_char * 128), ("dimCount", ctypes.c_size_t),
                ("dims", ctypes.c_int64 * 128)]


def model_input_slots(info):
    """Actual external Data nodes; missing list instances are Const nodes made by the pass."""
    slots = ["x"]
    if info["bias"]:
        slots += ["bias1", "bias2"]
    return slots


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_spec(path):
    with open(path, "rb") as stream:
        header = stream.read(HDR_LEN)
    if len(header) != HDR_LEN:
        die("truncated case header")
    values = struct.unpack(HDR_FMT, header)
    if values[0] != b"FC2DCASE" or values[1] != CASE_VERSION:
        die("binding requires case ABI 8")
    return dict(zip(SPEC_KEYS, values[6:6 + SPEC_N]))


def _valid_fusion_report(report, names):
    if not isinstance(report, dict):
        return False
    expected = {"pass_name": "ExtendConv2DPairToFusedMultiConv2DFusionPass",
                "stage1_pass_name": "Conv2DFixPipeToExtendConv2DFusionPass",
                "fused_node_count": 1, "unfused_convolution_count": 0, "input_names": names,
                "source_conv_count": 2, "stage1_extend_count": 2,
                "evidence_mode": "single_atc_graph_dump", "compile_count": 1}
    if any(report.get(key) != value for key, value in expected.items()):
        return False
    for key in ("source_graph_sha256", "stage1_snapshot_sha256", "stage2_snapshot_sha256", "model_graph_sha256"):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return False
    tiling = report.get("native_tiling")
    if not isinstance(tiling, dict) or tiling.get("tiling_data_bytes") not in (312, 316):
        return False
    digest = tiling.get("tiling_data_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return False
    return True


def write_model_binding(case_path, om_path, expected_case_sha256=None, fusion_report=None):
    """Bind constant weights/scales to the exact case used during compilation."""
    with open(case_path, "rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or struct.unpack_from("<I", header, 8)[0] != CASE_VERSION:
        die("constant-weight OM binding requires case ABI 8; regenerate case and OM")
    case_sha = _file_sha256(case_path)
    if expected_case_sha256 is not None and case_sha != expected_case_sha256:
        die("case SHA256 changed during compilation; regenerate case and OM")
    info = _binding_spec(case_path)
    names = model_input_slots(info)
    if not _valid_fusion_report(fusion_report, names):
        die("verified production fusion report is required; rebuild through canndev PR 82567")
    binding = {"schema_version": 6, "op_type": "FusedMultiConv2d", "case_abi": CASE_VERSION,
               "pipeline": "onnx_conv_pair_fusion", "weight_mode": "constant", "scale_mode": "constant", "model_input_count": len(names),
               "model_input_names": names, "fusion": fusion_report,
               "model_sha256": _file_sha256(om_path), "case_sha256": expected_case_sha256 or case_sha}
    path = om_path + ".fc2d.json"
    with open(path + ".tmp", "w") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(path + ".tmp", path)
    return path


def validate_model_binding(case_path, om_path):
    """Validate before loading ACL: weights and attributes are fixed in OM."""
    path = om_path + ".fc2d.json"
    try:
        with open(path) as stream:
            binding = json.load(stream)
    except (OSError, ValueError) as exc:
        die("missing or invalid case/OM binding %s: %s; regenerate case and OM" % (path, exc))
    contract = {"schema_version": 6, "op_type": "FusedMultiConv2d", "case_abi": CASE_VERSION,
                "weight_mode": "constant", "scale_mode": "constant", "pipeline": "onnx_conv_pair_fusion"}
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in contract.items()):
        die("unsupported binding; regenerate OM through the production GE fusion pass")
    info = _binding_spec(case_path)
    names = model_input_slots(info)
    proof = binding.get("fusion", {})
    if binding.get("model_input_count") != len(names) or binding.get("model_input_names") != names or not _valid_fusion_report(proof, names):
        die("invalid fusion/input contract; regenerate OM")
    if binding.get("model_sha256") != _file_sha256(om_path):
        die("OM SHA256 differs from its binding; regenerate case and OM")
    if binding.get("case_sha256") != _file_sha256(case_path):
        die("case SHA256 differs from the compiled OM binding; weights/attributes changed, regenerate OM")
    return binding


def _reference_report_valid(report, names, source_sha256):
    """确认参考模型保留两个 ExtendConv2D，只关闭第二阶段融合 Pass。"""
    expected = {"pipeline": "extend_pair_reference", "source_conv_count": 2,
                "unfused_convolution_count": 2, "fused_node_count": 0,
                "stage1_extend_count": 2, "input_names": names,
                "source_graph_sha256": source_sha256,
                "enabled_passes": ["Conv2DFixPipeToExtendConv2DFusionPass"],
                "disabled_passes": ["ExtendConv2DPairToFusedMultiConv2DFusionPass"]}
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        return False
    digest = report.get("model_graph_sha256")
    return isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def write_reference_model_binding(case_path, reference_om_path, fused_om_path, report,
                                  expected_case_sha256=None):
    """将未融合 OM 绑定到同一 case、源图及待测融合 OM。"""
    if os.path.realpath(reference_om_path) == os.path.realpath(fused_om_path):
        die("reference OM must be a separate file from fused OM")
    fused = validate_model_binding(case_path, fused_om_path)
    case_sha = _file_sha256(case_path)
    if expected_case_sha256 is not None and case_sha != expected_case_sha256:
        die("case SHA256 changed during reference compilation; regenerate case and both OMs")
    tensors, _, _, _, _, _, _, info = load_case(case_path)
    names = model_input_slots(info)
    if not _reference_report_valid(report, names, fused["fusion"]["source_graph_sha256"]):
        die("verified unfused reference graph report is required")
    dtype, shape, _ = tensors["y_expect"]
    binding = {"schema_version": 1, "role": "extend_pair_reference", "case_abi": CASE_VERSION,
               "case_sha256": case_sha, "model_sha256": _file_sha256(reference_om_path),
               "fused_model_sha256": fused["model_sha256"], "model_input_names": names,
               "output": {"dtype": dtype, "shape": shape}, "reference": report}
    path = reference_om_path + ".fc2d.json"
    with open(path + ".tmp", "w") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(path + ".tmp", path)
    return path


def validate_reference_model_binding(case_path, reference_om_path, fused_om_path):
    """在访问设备前验证参考模型与待测模型的 case、源图和输出契约。"""
    fused = validate_model_binding(case_path, fused_om_path)
    try:
        with open(reference_om_path + ".fc2d.json") as stream:
            binding = json.load(stream)
    except (OSError, ValueError) as exc:
        die("missing or invalid reference binding: %s" % exc)
    tensors, _, _, _, _, _, _, info = load_case(case_path)
    dtype, shape, _ = tensors["y_expect"]
    names = model_input_slots(info)
    expected = {"schema_version": 1, "role": "extend_pair_reference", "case_abi": CASE_VERSION,
                "case_sha256": _file_sha256(case_path), "model_sha256": _file_sha256(reference_om_path),
                "fused_model_sha256": fused["model_sha256"], "model_input_names": names,
                "output": {"dtype": dtype, "shape": shape}}
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected.items()):
        die("reference binding differs from case/fused OM; rebuild both models")
    if not _reference_report_valid(binding.get("reference"), names, fused["fusion"]["source_graph_sha256"]):
        die("invalid unfused reference graph contract; rebuild reference OM")
    return binding



def runtime_input_slots(info):
    """Validate case tensor names and return the consumed graph runtime inputs."""
    unknown = set(info["order"]) - set(INPUT_SLOTS)
    if unknown:
        die("case 里有不属于 FusedMultiConv2d 的输入: %s" % sorted(unknown))
    return model_input_slots(info)


# ACL 把失败的细节（AI Core 异常、哪条 task、PC）攒在一个进程级的字符串里，
# 只有 aclGetRecentErrMsg() 能取出来 —— 它不走 slog，所以 ASCEND_SLOG_PRINT_TO_STDOUT
# 打不打开都不影响。返回码本身（比如 507014）只说"哪一类"，这个字符串才说"哪一个"。
_ACL_HANDLE = None


def acl_err_detail():
    if _ACL_HANDLE is None:
        return ""
    fn = getattr(_ACL_HANDLE, "aclGetRecentErrMsg", None)
    if fn is None:
        return "\n    （这个 CANN 的 libascendcl.so 里没有 aclGetRecentErrMsg）"
    try:
        msg = fn()
    except Exception as e:  # noqa: BLE001 - 诊断路径，不能再抛
        return "\n    （aclGetRecentErrMsg 调用失败: %s）" % e
    if not msg:
        return "\n    （aclGetRecentErrMsg 是空的 —— 错误可能发生在 runtime 更下层，看 ~/ascend/log/）"
    return "\n\n    ---- aclGetRecentErrMsg ----\n    " + msg.decode(errors="replace").replace("\n", "\n    ")


def die(msg):
    print("\n[X] %s%s" % (msg, acl_err_detail()))
    sys.exit(1)


# ---------------------------------------------------------------- 读 .bin
def load_case(path):
    if not os.path.isfile(path):
        die("找不到 %s —— 在能编译的机器上跑 ./gen_case（或 fc2d.py --steps case）生成，再拷过来" % path)
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < HDR_LEN:
        die("%s 只有 %d 字节，文件不完整（传输中断？）" % (path, len(blob)))

    vals = struct.unpack_from(HDR_FMT, blob, 0)
    magic, version, ntensors, nonzero, sat, y_elems = vals[:6]
    spec_vals = vals[6:6 + SPEC_N]
    qs1_bits, qs2_bits = vals[6 + SPEC_N], vals[7 + SPEC_N]
    if magic != b"FC2DCASE":
        die("%s 不是 case 文件（magic = %r）" % (path, magic))
    if version != CASE_VERSION:
        die("case 文件版本 %d，本脚本认版本 %d。\n"
            "        gen_case 和 run_fused_multi_conv2d.py 必须配套 —— 重新生成一次 .bin。"
            % (version, CASE_VERSION))
    spec = dict(zip(SPEC_KEYS, spec_vals))

    tensors, off = {}, HDR_LEN
    for i in range(ntensors):
        if off + REC_LEN > len(blob):
            die("第 %d 个张量的头越界，文件被截断了" % i)
        name, dtype, ndim, d0, d1, d2, d3, nbytes = struct.unpack_from(REC_FMT, blob, off)
        off += REC_LEN
        if off + nbytes > len(blob):
            die("张量 %s 的数据越界（需要 %d 字节，只剩 %d）—— 文件被截断了"
                % (name.rstrip(b"\0").decode(), nbytes, len(blob) - off))
        name = name.rstrip(b"\0").decode()
        dims = [d0, d1, d2, d3][:ndim]
        tensors[name] = (dtype, dims, blob[off:off + nbytes])
        off += nbytes

    if off != len(blob):
        die("文件尾部多出 %d 字节，格式对不上" % (len(blob) - off))

    # 元素数和 dims 必须自洽。这一条能挡住「传了半个文件却刚好没越界」。
    for name, (dtype, dims, data) in tensors.items():
        want = DTYPE_SIZE[dtype]
        for d in dims:
            want *= d
        if want != len(data):
            die("张量 %s: dims=%s dtype=%s 应为 %d 字节，实际 %d"
                % (name, dims, DTYPE_NAME[dtype], want, len(data)))

    mode = spec.get("dtype_mode", DTYPE_FP16 if spec["elem_bytes"] == 2 else DTYPE_INT8)
    isInt8 = mode in (DTYPE_INT8, DTYPE_S8F16)   # **入口 x** 是 int8
    isS8F16 = mode == DTYPE_S8F16
    isA16W8 = mode == DTYPE_A16W8
    isFp16 = mode == DTYPE_FP16
    outInt8 = mode == DTYPE_INT8
    order = ORDER_A16W8 if isA16W8 else (ORDER_S8F16 if isS8F16 else
                                         (ORDER_INT8 if outInt8 else ORDER))
    # y_exact（纯 fp32 参考）只有 fp16 定点通路有 —— 那边的定点模型带假设，需要
    # 第二个不依赖模型的参照。两条 int8 入口的 golden 都是精确整数运算，没有这个概念。
    required = list(order) + ["y_expect"] + (["y_exact"] if isFp16 else [])
    for name in required:
        if name not in tensors:
            die("case 文件里缺张量 %s" % name)

    # ABI 8 graph weights must retain exact logical shape; packed FZ buffers
    # with an accidentally rewritten header cannot masquerade as NCHW inputs.
    logical_filters = {
        "filter1": [spec["cout1"], spec["ci"], spec["kh"], spec["kw"]],
        "filter2": [spec["cout2"], spec["cout1"], spec.get("kh2") or spec["kh"],
                    spec.get("kw2") or spec["kw"]],
    }
    for name, expected in logical_filters.items():
        if tensors[name][1] != expected:
            die("ABI 8 requires logical NCHW %s=%s, got %s; regenerate the case file"
                % (name, expected, tensors[name][1]))

    shift1, shift2 = spec["shift1"], spec["shift2"]
    if isA16W8 and not (0 <= shift1 <= 29 and 0 <= shift2 <= 58):
        die("A16W8 fixed_shift_value out of range")
    if isFp16 and not (0 <= shift1 <= FIX_SHIFT_LEN and 0 <= shift2 <= FIX_SHIFT_LEN):
        die("header 里的定点定标 %d / %d 超出 [0,%d]" % (shift1, shift2, FIX_SHIFT_LEN))
    info = dict(spec)
    info["isInt8"] = isInt8      # 入口
    info["outInt8"] = outInt8    # 出口。s8f16 这条两者不同，别混用
    info["isS8F16"] = isS8F16
    info["isA16W8"] = isA16W8
    info["isFp16"] = isFp16
    info["order"] = order
    info["hasBias"] = bool(spec["bias"])
    info["relu"] = bool(spec["relu1"])   # 兼容旧调用点；两层各自的开关见 relu1/relu2
    info["qs1"] = struct.unpack("<f", struct.pack("<I", qs1_bits))[0]
    info["qs2"] = struct.unpack("<f", struct.pack("<I", qs2_bits))[0]
    info["shape_text"] = (
        "n%d %d->%d->%d %dx%d k%dx%d s%d/%d p%d,%d/%d,%d -> %dx%d %s"
        % (spec["n"], spec["ci"], spec["cout1"], spec["cout2"], spec["hi"], spec["wi"],
           spec["kh"], spec["kw"], spec["s1"], spec["s2"], spec["ph1"], spec["pw1"],
           spec["ph2"], spec["pw2"], spec["ho2"], spec["wo2"],
           "a16w8" if isA16W8 else ("int8->fp16" if isS8F16 else
           ("int8" if outInt8 else "fp16"))))
    # probe 这一位在版本 4 里没有了：探针权重是运行时输入，要探就直接改 gen_case
    # 生成的权重，不需要在 header 里留一个字段。
    return tensors, nonzero, 0, sat, y_elems, shift1, shift2, info


# ---------------------------------------------------------------- 绑 libascendcl
def load_acl():
    try:
        acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        die("加载 libascendcl.so 失败：%s\n"
            "    先 source <CANN安装路径>/set_env.sh（它会设 LD_LIBRARY_PATH）" % e)

    c_vp, c_i32, c_i, c_sz, c_cp = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_int,
                                    ctypes.c_size_t, ctypes.c_char_p)
    sig = [
        ("aclInit", [c_cp], c_i),
        ("aclFinalize", [], c_i),
        ("aclrtSetDevice", [c_i32], c_i),
        ("aclrtResetDevice", [c_i32], c_i),
        ("aclrtCreateStream", [ctypes.POINTER(c_vp)], c_i),
        ("aclrtDestroyStream", [c_vp], c_i),
        ("aclrtSynchronizeStream", [c_vp], c_i),
        ("aclrtMalloc", [ctypes.POINTER(c_vp), c_sz, c_i], c_i),
        ("aclrtFree", [c_vp], c_i),
        ("aclrtMemcpy", [c_vp, c_sz, c_vp, c_sz, c_i], c_i),
        ("aclCreateTensorDesc", [c_i, c_i, ctypes.POINTER(ctypes.c_int64), c_i], c_vp),
        ("aclDestroyTensorDesc", [c_vp], None),
        ("aclCreateDataBuffer", [c_vp, c_sz], c_vp),
        ("aclDestroyDataBuffer", [c_vp], c_i),
        ("aclopCreateAttr", [], c_vp),
        ("aclopSetAttrInt", [c_vp, ctypes.c_char_p, ctypes.c_int64], ctypes.c_int),
        ("aclopDestroyAttr", [c_vp], None),
    ]
    # 这两个只有用离线模型时才需要，缺了不该在这里就把脚本打死 ——
    # 用到的时候再报，那时的报错还能顺带说清楚该退回哪条路。
    opt = [
        # 这两个只有 int8 通路用。放**可选**里：fp16 那条已经在跑了，不该因为
        # 我新增的依赖而在缺这两个函数的 CANN 上直接 die。真要用时再报缺。
        # 注意签名必须注册，否则 ctypes 不知道怎么转 float/bool，报
        # "Don't know how to convert parameter 3"。
        ("aclopSetAttrFloat", [c_vp, ctypes.c_char_p, ctypes.c_float], ctypes.c_int),
        ("aclopSetAttrBool", [c_vp, ctypes.c_char_p, ctypes.c_uint8], ctypes.c_int),
        # 卷积超参（strides / pads）是 list_int。不注册签名的话
        # ctypes 会按 int 传指针，报 "Don't know how to convert parameter 4"。
        ("aclopSetAttrListInt", [c_vp, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
         ctypes.c_int),
        ("aclGetRecentErrMsg", [], c_cp),
        # ---- 模型 API。om 现在是 atc --framework=5 编出来的**图** om，不是
        # 单算子 om，所以下发走 aclmdl* 而不是 aclopExecuteV2。放可选里的理由同上：
        # 缺哪个在真正用到的地方集中报一次（见 MODEL_API_SYMS），那时的报错
        # 还能顺带说清楚是 CANN 太老还是 runtime 不完整。
        ("aclmdlLoadFromFile", [c_cp, ctypes.POINTER(ctypes.c_uint32)], c_i),
        ("aclmdlUnload", [ctypes.c_uint32], c_i),
        ("aclmdlCreateDesc", [], c_vp),
        ("aclmdlDestroyDesc", [c_vp], c_i),
        ("aclmdlGetDesc", [c_vp, ctypes.c_uint32], c_i),
        ("aclmdlGetNumInputs", [c_vp], c_sz),
        ("aclmdlGetNumOutputs", [c_vp], c_sz),
        ("aclmdlGetInputSizeByIndex", [c_vp, c_sz], c_sz),
        ("aclmdlGetOutputSizeByIndex", [c_vp, c_sz], c_sz),
        ("aclmdlGetOutputDataType", [c_vp, c_sz], c_i),
        ("aclmdlGetOutputDims", [c_vp, c_sz, ctypes.POINTER(AclModelDimensions)], c_i),
        ("aclmdlCreateDataset", [], c_vp),
        ("aclmdlDestroyDataset", [c_vp], c_i),
        ("aclmdlAddDatasetBuffer", [c_vp, c_vp], c_i),
        ("aclmdlExecute", [ctypes.c_uint32, c_vp, c_vp], c_i),
        # 按名字取输入下标。有它就不用赌模型的输入顺序 —— 这是老 run_om.py
        # 没做而吃过亏的地方（它只用 GetInputNameByIndex 打印给人看，实际按下标喂）。
        ("aclmdlGetInputIndexByName", [c_vp, c_cp, ctypes.POINTER(c_sz)], c_i),
        ("aclmdlGetInputNameByIndex", [c_vp, c_sz], c_cp),
        ("aclmdlGetOutputNameByIndex", [c_vp, c_sz], c_cp),
    ]
    for name, argtypes, restype in sig:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            die("libascendcl.so 里没有 %s —— CANN 版本太老？" % name)
        fn.argtypes = argtypes
        fn.restype = restype
    for name, argtypes, restype in opt:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            continue
        fn.argtypes = argtypes
        fn.restype = restype
    global _ACL_HANDLE
    _ACL_HANDLE = acl
    return acl


# ---------------------------------------------------------------- 比对
def report_times(durs):
    """把一组耗时报出来，并把第 1 次（冷启动）单独拎出来。"""
    def line(tag, v):
        v = sorted(v)
        print("  %-14s n=%-4d min=%9.3f  p50=%9.3f  mean=%9.3f  max=%9.3f  stdev=%8.3f  us"
              % (tag, len(v), v[0], statistics.median(v), sum(v) / len(v), v[-1],
                 statistics.pstdev(v) if len(v) > 1 else 0.0))

    line("全部", durs)
    if len(durs) < 2:
        return
    print("  %-14s %.3f us" % ("第 1 次", durs[0]))
    rest = durs[1:]
    line("去掉第 1 次", rest)
    warm = statistics.median(sorted(rest))
    print("  %-14s %.3f us  (第 1 次 %.3f - 稳态中位数 %.3f)"
          % ("冷启动开销≈", durs[0] - warm, durs[0], warm))


def decode_y(buf, is_int8):
    """按输出 dtype 解码成 (值列表, 位模式列表)。

    两条通路的比对逻辑是共用的，只有解码不同：
      fp16   struct 'e'，位模式是 uint16；哨兵 0x7F7F
      int8   直接就是整数；位模式就是值本身（无符号形式），哨兵 0x7F

    int8 的哨兵 0x7F 同时也是合法值 127，所以"没写过"的判定在 int8 上不如
    fp16 强 —— 好在 golden 的 scale 是按 127/峰值 挑的，饱和为 0，正常结果里
    出现 127 的概率很低。gen_case 会把饱和数打出来，为 0 时这个哨兵才可信。
    """
    if not is_int8:
        return decode_f16(buf)
    vals = list(struct.unpack("<%db" % len(buf), buf))
    bits = [v & 0xFF for v in vals]
    return vals, bits


def decode_f16(buf):
    """字节串 -> (值列表 float, 位模式列表 uint16)。

    值用来算相对误差，位模式用来做逐位比较和哨兵判定 —— 两者缺一不可：
    NaN != NaN，只看值会把「没写过」误判成「写错了」。
    struct 的 'e' 就是 IEEE754 binary16，Python 3.6 起自带。
    """
    n = len(buf) // Y_ELEM_BYTES
    return list(struct.unpack("<%de" % n, buf)), list(struct.unpack("<%dH" % n, buf))


def report_ratio(got, want, n):
    """got / want 的比值分布，只提供乘性偏差线索。

    分布包含已经正确的元素；局部块出错时，正确部分就能制造 2^0 主峰。
    主峰不能证明卷积正确，也不能单独确定定标改法。int8 的缩放走 per-channel
    scale 表，不使用 fixed_shift_value；需结合逐位误差及空间分布判断。
    """
    import math
    hist, sign_flip, sat_like, both_nz = {}, 0, 0, 0
    for i in range(n):
        g, w = got[0][i], want[0][i]
        if g == 0.0 or w == 0.0:
            continue
        if not math.isfinite(g) or not math.isfinite(w):
            continue
        both_nz += 1
        if (g > 0) != (w > 0):
            sign_flip += 1
            continue
        k = int(round(math.log(abs(g) / abs(w), 2)))
        hist[k] = hist.get(k, 0) + 1
    print("\n[比值] 两边都非零的 %d 个点，log2(got/want) 的分布" % both_nz)
    if both_nz == 0:
        print("       （没有可比的点）")
        return
    print("       符号相反 %d 个 (%.4f%%)" % (sign_flip, sign_flip * 100.0 / both_nz))
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:12]
    for k, c in top:
        print("         2^%-4d %9d  (%.4f%%)" % (k, c, c * 100.0 / both_nz))
    if top and top[0][1] > both_nz * 0.5:
        if top[0][0] == 0:
            print("       => 主峰 2^0 包含已正确元素；不能据此判定卷积正确或建议修改 shift。")
        else:
            print("       => 主峰 2^%d 是乘性偏差线索，需结合错误分布验证；int8 检查 scale 表，不改 fixed_shift_value。" % top[0][0])
    else:
        print("       => 没有主峰：不是单纯的定标问题。")


def dump_raw(path, raw, tag):
    """把设备原始输出落盘，供离线分析。

    上板一次很贵，而「到底哪一级错了」往往要拿原始数据反复试假设。存下来之后
    所有分析都能在本地做，不用再占板子。
    """
    try:
        with open(path, "wb") as f:
            f.write(raw)
        print("\n[dump] %s 已写入 %s (%d 字节)" % (tag, path, len(raw)))
    except IOError as e:
        print("\n[dump] 写 %s 失败: %s" % (path, e))


def probe_acc_scale(got, exact, n, shift2, dims):
    """在设备上把定标推出来，只打印结论 —— 原始数据传不出去。

    已确证的那半：out = acc_int32 * 2^-F，F = 58 - S（fixpipe 的 DEQF16 分支只按
    shift 现搭一个 2^-F 的 float，见 kernel_operator_fixpipe_impl.h:77）。
    未知的那半：mmad 拿 fixShiftVal 干了什么，也就是 acc 相对真值的定标 2^E。

    合起来 out = 真值 * 2^(E-F)，饱和阈值 |真值| * 2^E >= 2^31。两件事都可测：

      (1) 比值。未饱和的点上 log2(|got| / |真值|) 应当是常数 E-F。有单一主峰就
          说明整条链只差一个 2 的幂，改 shift 即可。
      (2) 夹逼。饱和的点给出 E >= log2(2^31/|真值|)，未饱和的点给出 E < 同式。
          两边一夹，E 就定到 1 位以内 —— 不依赖 (1) 的主峰是否干净。

    真值取 y_exact（纯 fp32 参考）。它是 relu 之后的，所以只有正半边可观测；
    负半边饱和会被 relu 抹成 0，看不见，这不影响结论。
    """
    import math
    deq_exp = FIX_SHIFT_LEN - shift2
    sat_out = math.ldexp(1.0, 31 - deq_exp)      # acc 顶到 2^31 时的输出值
    print("\n[定标反推] F = 58 - %d = %d，累加器饱和对应的输出值 = %.6g" % (shift2, deq_exp, sat_out))

    hist = {}
    nsat = nz = flip = 0
    sat_min_true = float("inf")     # 饱和点里最小的 |真值| -> E 的下界
    unsat_max_true = 0.0            # 未饱和点里最大的 |真值| -> E 的上界
    sat_idx = unsat_idx = -1
    for i in range(n):
        g, t = got[0][i], exact[0][i]
        if t == 0.0 or g == 0.0:
            continue
        if not math.isfinite(g) or not math.isfinite(t):
            continue
        nz += 1
        if (g > 0) != (t > 0):
            flip += 1
        # 饱和判据要**精确**。acc 顶到 2^31-1 时输出是 sat_out*(1-2^-31)，fp16 舍成
        # 恰好 sat_out；而未饱和的最大可能输出是它下面那个 fp16（sat_out 的 1 个 ULP
        # 之下）。所以 >= sat_out 是干净的分界。留松一点（比如 0.999）会把边界附近
        # 被 fp16 上舍的点误判成饱和，夹逼就会差 1 位 —— 合成数据上踩到过。
        if abs(g) >= sat_out:
            nsat += 1
            if abs(t) < sat_min_true:
                sat_min_true, sat_idx = abs(t), i
        else:
            if abs(t) > unsat_max_true:
                unsat_max_true, unsat_idx = abs(t), i
            k = int(round(math.log(abs(g) / abs(t), 2)))
            hist[k] = hist.get(k, 0) + 1

    print("       两边都非零 %d 个；其中疑似饱和 %d 个 (%.4f%%)；符号相反 %d 个 (%.4f%%)"
          % (nz, nsat, nsat * 100.0 / max(nz, 1), flip, flip * 100.0 / max(nz, 1)))

    tot = sum(hist.values())
    print("       (1) 未饱和点上 log2(|got|/|真值|) 的分布，共 %d 个:" % tot)
    for k, c in sorted(hist.items(), key=lambda kv: -kv[1])[:10]:
        print("             2^%-4d %9d  (%.4f%%)   => E-F=%d, 即 E=%d" % (k, c, c * 100.0 / max(tot, 1), k, k + deq_exp))
    if tot and max(hist.values()) > tot * 0.5:
        kbest = max(hist, key=lambda k: hist[k])
        print("           有主峰: E = %d。想让 out == 真值，需要 E-F = 0，" % (kbest + deq_exp))
        print("           即把 F 从 %d 改成 %d，也就是属性 S 从 %d 改成 %d。"
              % (deq_exp, kbest + deq_exp, shift2, FIX_SHIFT_LEN - (kbest + deq_exp)))
    else:
        print("           没有主峰 —— 不是单纯的定标问题（多半是上一层就错了）。")

    print("       (2) 夹逼:")
    lo = math.log(math.ldexp(1.0, 31) / sat_min_true, 2) if nsat and sat_min_true > 0 else None
    hi = math.log(math.ldexp(1.0, 31) / unsat_max_true, 2) if unsat_max_true > 0 else None
    if lo is not None:
        print("             饱和点里最小真值 %.6g @ %d (%s)  =>  E >= %.2f"
              % (sat_min_true, sat_idx, idx_label(sat_idx, dims), lo))
    else:
        print("             没有饱和点 —— 拿不到 E 的下界")
    if hi is not None:
        print("             未饱和点里最大真值 %.6g @ %d (%s)  =>  E <  %.2f"
              % (unsat_max_true, unsat_idx, idx_label(unsat_idx, dims), hi))
    if lo is not None and hi is not None:
        if lo < hi:
            print("             => E 落在 [%.2f, %.2f)，取整 E = %d" % (lo, hi, int(math.ceil(lo))))
        else:
            print("             => 区间是空的（%.2f >= %.2f）：饱和与否不只由 |真值| 决定，" % (lo, hi))
            print("                说明上一层（conv1）就已经错了，不是 conv2 单级定标的问题。")


def print_sample(got, want, exact, n, dims, count=32):
    """定点抽样。原始数据传不出去，就打一小撮出来供离线核对。

    取等间距的 count 个点，位模式和值都打 —— 位模式才能看出 -0 和舍入的最后一位。
    """
    # 步长要和 HO2*WO2 互质，否则会一直落在同一个空间位置上（上一版就是，32 个点
    # 全在 h0 w0 那个 padding 角落）。这里直接取一个不整除 plane 的奇数步长。
    print("\n[抽样] %d 个点，跨通道跨空间铺开   (idx  c/h/w   got | y_expect | y_exact)" % count)
    step = max(1, n // count) | 1
    while step > 1 and (dims[2] * dims[3]) % step == 0:
        step += 2
    for j in range(count):
        i = (j * step) % n
        print("   %7d %-14s 0x%04x %-12.6g | 0x%04x %-12.6g | 0x%04x %.6g"
              % (i, idx_label(i, dims), got[1][i], got[0][i],
                 want[1][i], want[0][i], exact[1][i], exact[0][i]))


# 输出是 NCHW [1, COUT2, HO2, WO2]，所以下标先除 HO2*WO2 得通道，再拆 h/w。
# 之前这里按 i // cout2 拆，那是把 [空间][通道] 的排布当成了真的 —— 标签是错的，
# 而且等间距抽样会全落在同一个空间位置上（padding 角落），看不出问题。
def idx_label(i, dims):
    plane = dims[2] * dims[3]
    c, sp = i // plane, i % plane
    return "c%d h%d w%d" % (c, sp // dims[3], sp % dims[3])


def report_idmap(got, dims, probe):
    """编号斜坡探针的读数表。

    x 里每个位置的值只编码它的通道号（或列号），恒等权重下输出应当是同一个编号。
    所以设备打出来的值除以最小的那个，就是「实际落到这里的是几号」—— 不用从错误里
    反推排布，直接读。比例是自归一的，所以不受未知的 2^E 定标影响。

    每个输出通道取众数（同一通道内所有位置本该是同一个值），并报告：
      modal      该通道出现最多的非零值
      id         modal / 全局最小 modal - 1，即实际落到这里的编号
      一致率     取到众数的位置占比。远小于 1 说明同一通道内部就不一致，
                 那不是通道置换，是空间上也错位了。
      饱和       该通道里顶到 32768 的位置数；饱和会毁掉读数，要单独看
    """
    plane = dims[2] * dims[3]
    nch = dims[1]
    sat = 32768.0
    rows = []
    gmin = None
    for c in range(nch):
        cnt = {}
        nsat = 0
        for i in range(c * plane, (c + 1) * plane):
            v = got[0][i]
            if v >= sat:
                nsat += 1
                continue
            if v == 0.0:
                continue
            cnt[v] = cnt.get(v, 0) + 1
        if cnt:
            modal = max(cnt, key=lambda k: cnt[k])
            agree = cnt[modal] / float(plane)
            if gmin is None or modal < gmin:
                gmin = modal
        else:
            modal, agree = 0.0, 0.0
        rows.append((c, modal, agree, nsat, len(cnt)))

    label = "输入通道" if probe == 3 else "列号"
    print("\n[编号读数] 每个输出通道的众数值 -> 实际落在这里的%s" % label)
    print("       全局最小众数 = %s（当作编号 0 的刻度）" % ("%.6g" % gmin if gmin else "无"))
    print("       %-4s %-14s %-6s %-8s %-7s %s" % ("c", "众数", "实测id", "一致率", "饱和数", "应为"))
    for c, modal, agree, nsat, ndist in rows:
        want = str(c) if c < 32 else "0(空)"
        if modal == 0.0:
            got_id = "-" if nsat == 0 else "全饱和"
        else:
            got_id = "%.2f" % (modal / gmin - 1.0) if gmin else "?"
        flag = ""
        if modal == 0.0 and nsat == 0 and c >= 32:
            flag = "  OK(空)"
        elif gmin and modal and abs(modal / gmin - 1.0 - c) < 0.01 and c < 32 and nsat == 0:
            flag = "  OK"
        print("       %-4d %-14.6g %-6s %-8.4f %-7d %s%s"
              % (c, modal, got_id, agree, nsat, want, flag))


def report_lsb(got, want, r, mismatches, dims, shift2):
    """把偏差同时按**输出 ULP** 和累加器 LSB 报出来。

    以前这里只按累加器 LSB 判，是错的。累加器的 LSB 是 2^-F（F = 58 - S），而
    fp16 输出在数值 v 附近的间隔是 v * 2^-10。F=16、v=142 时，1 个输出 ULP =
    0.125 = **8192 个累加器 LSB** —— 于是「差 1 个 ULP」这种最良性的情况会被报成
    「远超 LSB 量级」。判据必须用输出 ULP，累加器 LSB 只作参考。

    读法：
      偏差 <= 1 ULP    定点链算对了，差的只是 round 发生在哪一级（每个乘积 /
                       每条 mmad / 整条 K 累完），属于建模精度，不是 kernel 的问题；
      偏差 >> 1 ULP    才是真错了或者累加器溢出了。
    """
    import math
    deq_exp = FIX_SHIFT_LEN - shift2
    lsb = 2.0 ** (-deq_exp)
    print("\n[偏差] conv2 定标 2^%d（属性 S=%d）；累加器 1 LSB = %.6g" % (deq_exp, shift2, lsb))
    if not mismatches:
        print("       逐位全等，无偏差")
        return

    def ulp_of(v):
        """v 附近 fp16 的间隔。次正规区间恒为 2^-24。"""
        a = abs(v)
        if a == 0.0 or a < 2.0 ** -14:
            return 2.0 ** -24
        e = math.floor(math.log(a, 2))
        return 2.0 ** (e - 10)

    edges = [1.0, 2.0, 4.0, 16.0, 256.0]
    names = ["<=1 ULP", "<=2", "<=4", "<=16", "<=256", ">256"]
    hist = dict.fromkeys(names, 0)
    max_ulp, max_i, nonfinite = 0.0, -1, 0
    for i in range(r["n"]):
        if got[1][i] == want[1][i]:
            continue
        d = got[0][i] - want[0][i]
        if d != d or d in (float("inf"), float("-inf")):
            nonfinite += 1
            continue
        # 用两者中较大的那个定 ULP —— relu 边界上一边是 0，用 0 的 ULP 没意义
        scale = max(abs(got[0][i]), abs(want[0][i]))
        d = abs(d) / ulp_of(scale)
        for e, nm in zip(edges, names):
            if d <= e:
                hist[nm] += 1
                break
        else:
            hist[">256"] += 1
        if d > max_ulp:
            max_ulp, max_i = d, i
    if max_i >= 0:
        print("       最大偏差 %.2f ULP @ %d (%s): got %.6g, want %.6g   （= %.0f 个累加器 LSB）"
              % (max_ulp, max_i, idx_label(max_i, dims), got[0][max_i], want[0][max_i],
                 abs(got[0][max_i] - want[0][max_i]) / lsb))
    for nm in names:
        if hist[nm]:
            print("         %-9s %9d  (%.4f%%)" % (nm, hist[nm], hist[nm] * 100.0 / mismatches))
    if nonfinite:
        print("         非有限差 %d 个" % nonfinite)
    within1 = hist["<=1 ULP"]
    if max_ulp <= 2.0 and nonfinite == 0:
        print("       => 全部在 1~2 个输出 ULP 以内：卷积算对了，差的只是 round 的级别。")
    elif within1 * 1.0 / mismatches >= 0.9:
        print("       => %.2f%% 在 1 个 ULP 以内，尾部 %d 个更大 —— 多半是 relu 边界附近"
              % (within1 * 100.0 / mismatches, mismatches - within1))
        print("          （真值贴近 0 时，累加的一点点差异就会改变 relu 的取舍）。")
    else:
        print("       => 偏差远超 1 个 ULP：这不是 round 的问题。")


def report_error_map(got, want, dims):
    """**错在哪里** —— 按输出行 / 通道 / 线性位移三个维度给分布。

    原来的报告只说「有多少个不对」，不说「错在哪」。而这三张图各自指向不同的根因：

      按行     dirty 的行成连续带 -> 分核或行带相关。kernel 按「行带 x 列段」分
               chunk，一个核算坏了就表现成几条连续的输出行整段不对。
      按通道   dirty 的通道成片或等间隔 -> per-channel 的东西取错了
               （quant_scale 表、bias、或者权重的 Cout 维错位）。
      线性位移 某个非零位移 k 的吻合率明显高于 k=0 -> 值算对了但**放错位置**，
               而 k 直接告诉你错了多少个元素。

    三张都平（到处错一点、没有结构）才是「算错了」而不是「放错了」，那时才该
    去查定标和累加。
    """
    gv = got[0]
    wv = want[0]
    n = min(len(gv), len(wv))
    if n == 0 or len(dims) < 4:
        return
    C, H, W = int(dims[1]), int(dims[2]), int(dims[3])
    plane = H * W
    if C * plane == 0 or C * plane > n:
        return

    bad_h = [0] * H
    bad_c = [0] * C
    tot_h = C * W
    for i in range(C * plane):
        if gv[i] != wv[i]:
            bad_c[i // plane] += 1
            bad_h[(i % plane) // W] += 1

    print("\n[错误分布] 按输出行（共 %d 行，每行 %d 个元素）" % (H, tot_h))
    runs = []
    for hh in range(H):
        state = "干净" if bad_h[hh] == 0 else ("全错" if bad_h[hh] == tot_h else "部分")
        if runs and runs[-1][0] == state:
            runs[-1][2] = hh
        else:
            runs.append([state, hh, hh])
    for state, a, z in runs[:24]:
        span = "h%d" % a if a == z else "h%d-%d" % (a, z)
        worst = max(bad_h[a:z + 1])
        print("    %-10s %-6s 最差一行 %d/%d (%.1f%%)" % (span, state, worst, tot_h,
                                                        worst * 100.0 / tot_h))
    if len(runs) > 24:
        print("    ...（还有 %d 段）" % (len(runs) - 24))
    if len(runs) == 1 and runs[0][0] != "干净":
        print("    -> 所有行都受影响，不是分核/行带的问题")

    nbadc = sum(1 for x in bad_c if x)
    print("[错误分布] 按通道：%d/%d 个通道有错" % (nbadc, C))
    if 0 < nbadc < C:
        idx = [c for c in range(C) if bad_c[c]]
        step = sorted(set(idx[i + 1] - idx[i] for i in range(len(idx) - 1))) if len(idx) > 1 else []
        print("    出错的通道: %s%s" % (", ".join(str(c) for c in idx[:16]),
                                       " ..." if len(idx) > 16 else ""))
        if len(step) == 1:
            print("    -> 等间隔 %d，像是 per-channel 的表按错的步长取了" % step[0])
    elif nbadc == C:
        print("    -> 每个通道都有错，不是 per-channel 的东西取错了")

    # 线性位移：got[i] 和 want[i+k] 比。k=0 就是正常比对。
    #
    # **候选要连续扫，不能只挑几个"像样"的数。** 第一版只试了 0/±1/±2/±4/±8/±16/±W/
    # ±2W/±plane，注入 +3 的位移时它一个都没命中，于是拿 -8 的 30.68% 对 k=0 的
    # 30.51% —— 0.17% 的噪声差 —— 断言"位移 -8，值放错了"。**错的结论比没有结论更坏。**
    # 所以现在两件事：连续扫 -64..64，并且要赢得够多才敢下结论。
    lim = 64
    cand = list(range(-lim, lim + 1))
    for s in (W, 2 * W, 4 * W, plane):
        cand += [s, -s, s + 1, -(s + 1)]
    cand = sorted(set(cand))

    def match_rate(k, stride):
        hit = tot = 0
        for i in range(0, C * plane, stride):
            j = i + k
            if 0 <= j < n:
                tot += 1
                if gv[i] == wv[j]:
                    hit += 1
        return (hit * 100.0 / tot) if tot else -1.0

    coarse_stride = max(1, (C * plane) // 8000)
    rough = sorted(((match_rate(k, coarse_stride), k) for k in cand), reverse=True)
    fine_stride = max(1, (C * plane) // 60000)
    refine = sorted(set([k for _, k in rough[:6]] + [0]))
    rows = sorted(((match_rate(k, fine_stride), k) for k in refine), reverse=True)
    base = dict((k, v) for v, k in rows).get(0, -1.0)
    print("[错误分布] 线性位移探针（got[i] vs want[i+k]，扫 k=-%d..%d 及行/面步长）" % (lim, lim))
    for rate, k in rows[:5]:
        print("    k=%-7d 吻合 %.2f%%%s" % (k, rate, "   <- 正常比对" if k == 0 else ""))
    best_rate, best_k = rows[0]
    # 要赢得够多才算"位移"：噪声级的领先什么都不说明。
    if best_k != 0 and best_rate >= max(base * 1.5, base + 15.0):
        print("    -> **位移 %d 的吻合率 %.2f%% 显著高于正常比对的 %.2f%%**："
              "值算对了但放错位置。" % (best_k, best_rate, base))
    else:
        print("    -> 没有显著胜出的位移（最好的 k=%d 只有 %.2f%%，正常比对 %.2f%%）——"
              % (best_k, best_rate, base))
        print("       不是整体错位，是算出来的值本身不对。")

def report_vs_exact(got, exact, n):
    """对纯 fp32 参考。不设门槛，只报数。

    上面那一路拿定点模型当判据；这一路完全不依赖定点模型，只回答「这个算子到底
    有没有在算这个卷积」。定点格本身有噪声，所以这里不当失败判据 —— 它过而定点
    模型不过，问题在模型，不在 kernel。
    """
    worst, worst_i, ok1, ok2 = 0.0, -1, 0, 0
    for i in range(n):
        g, w = got[0][i], exact[0][i]
        if w == 0.0:
            e = 0.0 if g == 0.0 else float("inf")
        else:
            e = abs(g - w) / abs(w)
        if e <= 1e-2:
            ok1 += 1
        if e <= 1e-1:
            ok2 += 1
        if e > worst:
            worst, worst_i = e, i
    print("\n[对纯 fp32 参考] 不设门槛，只报数（定点格本身有噪声）")
    print("       相对误差 <= 1e-2: %d / %d = %.4f%%    <= 1e-1: %d / %d = %.4f%%"
          % (ok1, n, ok1 * 100.0 / n, ok2, n, ok2 * 100.0 / n))
    if worst_i >= 0:
        print("       最大相对误差 %s @ %d: got %.6g, 纯 fp32 %.6g"
              % (("%.4g" % worst) if worst != float("inf") else "inf",
                 worst_i, got[0][worst_i], exact[0][worst_i]))



def evaluate(got, want, rel_tol, sentinel=Y_SENTINEL_U16):
    """一趟扫完所有判据。返回 dict。

    三个"写没写对"的判据，各挡一类失败，缺一不可：
      unwritten  kernel 一个字节都没写（哨兵 0x7F7F 还在）
      nonzero    kernel 只写了一部分（前一半对、后一半空白）
      mismatches 数值错

    外加精度判据：逐元素相对误差 <= rel_tol 的比例。
      golden 非 0: |got-want| / |want|
      golden 为 0: 退化成绝对误差 |got-want|（即要求 got 也为 0）
    """
    n = len(got[1])   # got / want 都是 (值, 位模式) 二元组
    buckets = ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]
    r = {
        "n": n,
        "nonzero": sum(1 for x in got[0] if x != 0.0),
        "unwritten": 0,
        "mismatches": 0,
        "first_bad": -1,
        "off_by_one": 0,
        "prec_ok": n,
        "max_rel": 0.0,
        "max_rel_idx": -1,
        "zero_golden": 0,
        "hist": dict.fromkeys(buckets, 0),
        "sentinel": sentinel,
    }
    gv, gb = got
    wv, wb = want
    if gb == wb and all(math.isfinite(value) for values in (gv, wv) for value in values):
        # 逐位一致：相对误差全是 0
        r["hist"]["=0"] = n
        r["zero_golden"] = sum(1 for x in wv if x == 0.0)
        return r

    zero_golden = 0
    prec_ok = 0
    for i in range(n):
        g, w = gv[i], wv[i]
        g8, w8 = gb[i], wb[i]
        if g8 == r["sentinel"] and w8 != r["sentinel"]:
            r["unwritten"] += 1
        if g8 != w8:
            if r["first_bad"] < 0:
                r["first_bad"] = i
            r["mismatches"] += 1
            # fp16 上「差 1」就是差 1 个 ULP（同号有限值的位模式相邻）。
            if abs(g8 - w8) == 1:
                r["off_by_one"] += 1
        if not math.isfinite(g) or not math.isfinite(w):
            err = float("inf")
        elif w == 0.0:
            zero_golden += 1
            err = 0.0 if g == 0.0 else float("inf")
        else:
            err = abs(g - w) / float(abs(w))
        if err <= rel_tol:
            prec_ok += 1
        if err > r["max_rel"]:
            r["max_rel"] = err
            r["max_rel_idx"] = i
        if err == 0.0:
            r["hist"]["=0"] += 1
        elif err == float("inf"):
            r["hist"]["inf(golden=0)"] += 1
        elif err <= 1e-3:
            r["hist"]["(0,1e-3]"] += 1
        elif err <= 1e-2:
            r["hist"]["(1e-3,1e-2]"] += 1
        elif err <= 1e-1:
            r["hist"]["(1e-2,1e-1]"] += 1
        elif err <= 1.0:
            r["hist"]["(1e-1,1]"] += 1
        else:
            r["hist"][">1"] += 1
    r["prec_ok"] = prec_ok
    r["zero_golden"] = zero_golden
    return r


# ---------------------------------------------------------------- 融合前后设备输出对比
def _require_acl_success(result, operation):
    if result != ACL_SUCCESS:
        raise RuntimeError("%s failed: %d %s" % (operation, result, acl_err_detail()))


def execute_bound_model(acl, om_path, tensors, input_names, output_dtype, output_shape, repeat=1, warmup=0):
    """在已有 ACL context 中执行一个 OM，并释放该模型占用的全部资源。"""
    model_id, desc, datasets = None, None, []
    allocations, buffers, host_buffers = [], [], []
    try:
        model = ctypes.c_uint32()
        _require_acl_success(acl.aclmdlLoadFromFile(os.fsencode(om_path), ctypes.byref(model)), "aclmdlLoadFromFile")
        model_id = model.value
        desc = acl.aclmdlCreateDesc()
        if not desc:
            raise RuntimeError("aclmdlCreateDesc returned null")
        _require_acl_success(acl.aclmdlGetDesc(desc, model_id), "aclmdlGetDesc")
        count = int(acl.aclmdlGetNumInputs(desc))
        if count != len(input_names) or int(acl.aclmdlGetNumOutputs(desc)) != 1:
            raise RuntimeError("model input/output count differs from bound source graph")
        shape = AclModelDimensions()
        _require_acl_success(acl.aclmdlGetOutputDims(desc, 0, ctypes.byref(shape)), "aclmdlGetOutputDims")
        if shape.dimCount > 128 or list(shape.dims[:shape.dimCount]) != list(output_shape):
            raise RuntimeError("model output shape differs from bound NCHW output")
        if int(acl.aclmdlGetOutputDataType(desc, 0)) != output_dtype:
            raise RuntimeError("model output dtype differs from bound output dtype")
        ordered_inputs = [None] * count
        for name in input_names:
            index = ctypes.c_size_t()
            _require_acl_success(acl.aclmdlGetInputIndexByName(desc, name.encode(), ctypes.byref(index)),
                                 "aclmdlGetInputIndexByName(%s)" % name)
            if index.value >= count or ordered_inputs[index.value] is not None:
                raise RuntimeError("model input names are not a one-to-one mapping")
            ordered_inputs[index.value] = name
        output_size = reduce(mul, output_shape, 1) * DTYPE_SIZE[output_dtype]
        if int(acl.aclmdlGetOutputSizeByIndex(desc, 0)) != output_size:
            raise RuntimeError("model output byte size differs from bound output")
        payloads = []
        for index, name in enumerate(ordered_inputs):
            data = tensors[name][2]
            if int(acl.aclmdlGetInputSizeByIndex(desc, index)) != len(data):
                raise RuntimeError("model input byte size differs for %s" % name)
            payloads.append(data)
        payloads.append(bytes([Y_SENTINEL_BYTE]) * output_size)
        for payload in payloads:
            device = ctypes.c_void_p()
            _require_acl_success(acl.aclrtMalloc(ctypes.byref(device), len(payload), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")
            allocations.append(device)
            host = ctypes.create_string_buffer(payload, len(payload))
            host_buffers.append(host)
            _require_acl_success(acl.aclrtMemcpy(device, len(payload), ctypes.cast(host, ctypes.c_void_p),
                                               len(payload), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D")
            buffer = acl.aclCreateDataBuffer(device, len(payload))
            if not buffer:
                raise RuntimeError("aclCreateDataBuffer returned null")
            buffers.append(buffer)
        for _ in range(2):
            dataset = acl.aclmdlCreateDataset()
            if not dataset:
                raise RuntimeError("aclmdlCreateDataset returned null")
            datasets.append(dataset)
        for buffer in buffers[:-1]:
            _require_acl_success(acl.aclmdlAddDatasetBuffer(datasets[0], buffer), "aclmdlAddDatasetBuffer input")
        _require_acl_success(acl.aclmdlAddDatasetBuffer(datasets[1], buffers[-1]), "aclmdlAddDatasetBuffer output")
        durations = []
        for iteration in range(warmup + repeat):
            start = time.perf_counter()
            _require_acl_success(acl.aclmdlExecute(model_id, datasets[0], datasets[1]), "aclmdlExecute")
            if iteration >= warmup:
                durations.append((time.perf_counter() - start) * 1e6)
        output = ctypes.create_string_buffer(output_size)
        _require_acl_success(acl.aclrtMemcpy(ctypes.cast(output, ctypes.c_void_p), output_size, allocations[-1],
                                           output_size, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H")
        raw = output.raw[:output_size]
        # 换用另一种预填值再执行一次，识别未写输出，且允许合法的全 127 整数结果。
        probe_pattern = ctypes.create_string_buffer(bytes([0x80]) * output_size, output_size)
        _require_acl_success(acl.aclrtMemcpy(allocations[-1], output_size, ctypes.cast(probe_pattern, ctypes.c_void_p),
                                           output_size, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy output write probe")
        _require_acl_success(acl.aclmdlExecute(model_id, datasets[0], datasets[1]), "aclmdlExecute output write probe")
        _require_acl_success(acl.aclrtMemcpy(ctypes.cast(output, ctypes.c_void_p), output_size, allocations[-1],
                                           output_size, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy write probe D2H")
        if output.raw[:output_size] != raw:
            raise RuntimeError("output depends on initial buffer contents or is nondeterministic; possible unwritten output")
        return raw, durations
    finally:
        for dataset in datasets:
            acl.aclmdlDestroyDataset(dataset)
        for buffer in buffers:
            acl.aclDestroyDataBuffer(buffer)
        for allocation in allocations:
            acl.aclrtFree(allocation)
        if model_id is not None:
            acl.aclmdlUnload(model_id)
        if desc:
            acl.aclmdlDestroyDesc(desc)


def _comparison_number(value):
    """保留有限数值，将非有限值转换为可写入 JSON 的字符串。"""
    return value if math.isfinite(value) else str(value)


def _comparison_sample(index, actual, reference, shape, absolute, relative, outside):
    """记录单个差异元素的数值、误差和张量坐标。"""
    remainder, coordinates = index, []
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder //= size
    return {"flat_index": index, "coordinates": list(reversed(coordinates)),
            "reference": _comparison_number(reference), "fused": _comparison_number(actual),
            "actual": _comparison_number(actual), "outside_tolerance": outside,
            "absolute_error": _comparison_number(absolute), "relative_error": _comparison_number(relative),
            "precision_0_001_passed": relative <= 0.001}


def summarize_output_comparison(actual, reference, output_shape, output_dtype, rel_tol=1e-3):
    """汇总已解码输出的差异；固定 0.001 统计不改变原有验收阈值。"""
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("comparison tolerance must be finite and nonnegative")
    count = reduce(mul, output_shape, 1)
    if output_dtype not in (1, 2) or any(len(values) != count for values in actual + reference):
        raise ValueError("comparison requires equally sized fp16 or int8 outputs matching the shape")
    bit_mismatches, tolerance_mismatches, nonfinite = 0, 0, {"reference": 0, "fused": 0}
    max_abs, max_rel, first, first_outside, precision_passed = 0.0, 0.0, None, None, 0
    samples = []
    for index, (got, want) in enumerate(zip(actual[0], reference[0])):
        bit_diff = actual[1][index] != reference[1][index]
        bit_mismatches += int(bit_diff)
        finite = math.isfinite(got) and math.isfinite(want)
        nonfinite["reference"] += int(not math.isfinite(want))
        nonfinite["fused"] += int(not math.isfinite(got))
        absolute = abs(got - want) if finite else float("inf")
        relative = absolute / abs(want) if finite and want != 0 else (0.0 if absolute == 0 else float("inf"))
        precision_passed += int(relative <= 0.001)
        max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
        outside = not finite or (bit_diff if output_dtype == 2 else relative > rel_tol)
        tolerance_mismatches += int(outside)
        if (bit_diff or outside) and (len(samples) < 50 or (first_outside is None and outside)):
            mismatch = _comparison_sample(index, got, want, output_shape, absolute, relative, outside)
            if len(samples) < 50:
                samples.append(mismatch)
            if first is None:
                first = mismatch
            if first_outside is None and outside:
                first_outside = mismatch
    precision_ratio = precision_passed / float(count) if count else 0.0
    return {"dtype": DTYPE_NAME[output_dtype], "shape": list(output_shape), "elements": count,
            "bitwise_equal": bit_mismatches == 0, "bit_mismatch_count": bit_mismatches,
            "tolerance_mismatch_count": tolerance_mismatches, "first_mismatch": first,
            "first_tolerance_mismatch": first_outside,
            "max_absolute_error": _comparison_number(max_abs),
            "max_relative_error": _comparison_number(max_rel), "nonfinite_count": nonfinite,
            "mismatch_samples": samples,
            "precision_0_001": {"relative_tolerance": 0.001, "passed_count": precision_passed,
                                "failed_count": count - precision_passed, "total_count": count,
                                "pass_ratio": precision_ratio, "pass_percent": precision_ratio * 100.0},
            "relative_tolerance": 0.0 if output_dtype == 2 else rel_tol,
            "zero_reference_absolute_tolerance": 0.0,
            "required_pass_ratio": 1.0, "passed": tolerance_mismatches == 0}


def print_output_comparison(summary, actual_label, reference_label):
    """用六列表格展示前 50 个差异，表后打印全量 0.001 达标百分比。"""
    headers = ['index', 'position', 'Output', 'Reference', 'Absolute Error', 'Relative Error']
    rows = []
    for sample in summary['mismatch_samples']:
        position = '[' + ','.join(str(value) for value in sample['coordinates']) + ']'
        numbers = [sample[field] for field in ('actual', 'reference', 'absolute_error', 'relative_error')]
        rows.append([str(sample['flat_index']), position] + [
            '%.9g' % value if isinstance(value, (int, float)) else str(value) for value in numbers])
    widths = [max([len(header)] + [len(row[index]) for row in rows]) for index, header in enumerate(headers)]
    pattern = ' | '.join(('{:<%d}' if index == 1 else '{:>%d}') % width for index, width in enumerate(widths))
    print('\n%s vs %s' % (actual_label, reference_label))
    print(pattern.format(*headers))
    print('-+-'.join('-' * width for width in widths))
    for row in rows:
        print(pattern.format(*row))
    precision = summary['precision_0_001']
    print('Precision: %.6f%% (%d / %d, relative error <= 0.001)' %
          (precision['pass_percent'], precision['passed_count'], precision['total_count']))


def compare_device_outputs(fused_raw, reference_raw, output_dtype, output_shape, rel_tol=1e-3):
    """以参考输出比较原始字节；整数逐位相等，浮点应用既有误差阈值。"""
    if output_dtype not in (1, 2) or len(fused_raw) != len(reference_raw):
        raise ValueError("comparison requires equally sized fp16 or int8 outputs")
    if len(fused_raw) != reduce(mul, output_shape, 1) * DTYPE_SIZE[output_dtype]:
        raise ValueError("raw output size differs from output shape")
    return summarize_output_comparison(decode_y(fused_raw, output_dtype == 2),
                                       decode_y(reference_raw, output_dtype == 2),
                                       output_shape, output_dtype, rel_tol)


def run_reference_comparison(case_path, fused_om_path, reference_om_path, device_id=0,
                             output_dir=None, rel_tol=1e-3, repeat=1, warmup=0,
                             compare_reference_golden=False):
    """在同一设备依次运行两个绑定模型，保存原始输出及可追溯的精度报告。"""
    fused_binding = validate_model_binding(case_path, fused_om_path)
    reference_binding = validate_reference_model_binding(case_path, reference_om_path, fused_om_path)
    tensors, _, _, _, _, _, _, info = load_case(case_path)
    dtype, shape, _ = tensors["y_expect"]
    if not math.isfinite(rel_tol) or rel_tol < 0 or repeat < 1 or warmup < 0:
        die("invalid comparison tolerance/repeat/warmup")
    suffix = "_".join(binding[key][:12] for binding, key in
                      ((fused_binding, "case_sha256"), (fused_binding, "model_sha256"),
                       (reference_binding, "model_sha256")))
    output_dir = os.path.abspath(output_dir or os.path.join(os.path.dirname(fused_om_path), "comparison_" + suffix))
    os.makedirs(output_dir, exist_ok=True)
    for filename in ("reference.bin", "fused.bin", "comparison.json"):
        stale = os.path.join(output_dir, filename)
        if os.path.isfile(stale):
            os.unlink(stale)
    names = model_input_slots(info)
    report = {"schema_version": 1, "run_id": str(time.time_ns()),
              "warmup": warmup, "repeat": repeat,
              "golden_source": "extend_pair_device_output", "cpu_golden_used": compare_reference_golden, "device_id": device_id,
              "case_sha256": fused_binding["case_sha256"], "source_graph_sha256": fused_binding["fusion"]["source_graph_sha256"],
              "runtime_input_sha256": {name: hashlib.sha256(tensors[name][2]).hexdigest() for name in names},
              "output": {"dtype": DTYPE_NAME[dtype], "shape": shape},
              "models": {"reference": {"model_sha256": reference_binding["model_sha256"], "status": "not_run"},
                         "fused": {"model_sha256": fused_binding["model_sha256"], "status": "not_run"}},
              "passed": False}
    raw_outputs = {}
    acl, initialized, device_set, stream = None, False, False, ctypes.c_void_p()
    try:
        acl = load_acl()
        required = ("aclmdlLoadFromFile", "aclmdlUnload", "aclmdlCreateDesc", "aclmdlDestroyDesc",
                    "aclmdlGetDesc", "aclmdlGetNumInputs", "aclmdlGetNumOutputs", "aclmdlGetInputIndexByName",
                    "aclmdlGetInputSizeByIndex", "aclmdlGetOutputSizeByIndex", "aclmdlGetOutputDataType",
                    "aclmdlGetOutputDims", "aclmdlCreateDataset", "aclmdlDestroyDataset", "aclmdlAddDatasetBuffer", "aclmdlExecute")
        missing = [name for name in required if getattr(acl, name, None) is None]
        if missing:
            raise RuntimeError("comparison requires ACL model APIs: %s" % ", ".join(missing))
        _require_acl_success(acl.aclInit(None), "aclInit")
        initialized = True
        _require_acl_success(acl.aclrtSetDevice(device_id), "aclrtSetDevice")
        device_set = True
        _require_acl_success(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
        for role, path, binding in (("reference", reference_om_path, reference_binding),
                                     ("fused", fused_om_path, fused_binding)):
            entry = {"model_sha256": binding["model_sha256"], "status": "failed"}
            report["models"][role] = entry
            try:
                raw, durations = execute_bound_model(acl, path, tensors, names, dtype, shape, repeat, warmup)
                raw_outputs[role] = raw
                raw_path = os.path.join(output_dir, role + ".bin")
                with open(raw_path + ".tmp", "wb") as output:
                    output.write(raw)
                os.replace(raw_path + ".tmp", raw_path)
                entry.update(status="success", output_file=os.path.basename(raw_path),
                             output_sha256=hashlib.sha256(raw).hexdigest(), durations_us=durations,
                             output_write_probe="two_initial_patterns_equal", execution_count=warmup + repeat + 1)
            except (RuntimeError, OSError) as exc:
                entry["error"] = str(exc)
        if len(raw_outputs) == 2:
            report["comparison"] = compare_device_outputs(raw_outputs["fused"], raw_outputs["reference"], dtype, shape, rel_tol)
            report["passed"] = report["comparison"]["passed"]
        if compare_reference_golden and "reference" in raw_outputs:
            report["reference_vs_golden"] = compare_device_outputs(raw_outputs["reference"], tensors["y_expect"][2],
                                                                  dtype, shape, rel_tol)
            report["cpu_golden_sha256"] = hashlib.sha256(tensors["y_expect"][2]).hexdigest()
            report["passed"] = report["passed"] and report["reference_vs_golden"]["passed"]
    except (RuntimeError, OSError, SystemExit) as exc:
        report["error"] = str(exc)
    finally:
        if acl is not None:
            if stream.value:
                acl.aclrtDestroyStream(stream)
            if device_set:
                acl.aclrtResetDevice(device_id)
            if initialized:
                acl.aclFinalize()
        report_path = os.path.join(output_dir, "comparison.json")
        with open(report_path + ".tmp", "w") as output:
            json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        os.replace(report_path + ".tmp", report_path)
    print("[%s] numerical comparison run; report: %s" % ("PASS" if report["passed"] else "FAIL", report_path))
    if "comparison" in report:
        print_output_comparison(report["comparison"], "FusedMultiConv2d", "ExtendConv2D pair")
    if "reference_vs_golden" in report:
        print_output_comparison(report["reference_vs_golden"], "ExtendConv2D pair", "CPU golden")
    return 0 if report["passed"] else 1


def _extract_pair_arguments(argv):
    """提取具名对比选项，避免路径被既有位置参数解析器误认为主 OM。"""
    remaining, options, index = [], {}, 0
    while index < len(argv):
        argument = argv[index]
        flag, separator, inline = argument.partition("=")
        if flag == "--compare-reference-golden":
            if flag in options or separator:
                die("%s is a boolean option and may only be specified once" % flag)
            options[flag] = True
        elif flag in ("--reference-om", "--comparison-output"):
            if flag in options:
                die("duplicate option %s" % flag)
            if separator:
                value = inline
            else:
                index += 1
                value = argv[index] if index < len(argv) else ""
            if not value or value.startswith("--"):
                die("%s requires a path" % flag)
            options[flag] = value
        else:
            remaining.append(argument)
        index += 1
    if "--comparison-output" in options and "--reference-om" not in options:
        die("--comparison-output requires --reference-om")
    if "--compare-reference-golden" in options and "--reference-om" not in options:
        die("--compare-reference-golden requires --reference-om")
    return remaining, options


# ---------------------------------------------------------------- main
def main():
    # 参数是位置式的，而最自然的写法 `脚本 case.bin om_out/fp16` 会把 om 目录
    # 撞到「算子名」那一位上 —— 于是去找一个名叫 "om_out/fp16/" 的算子，报
    # MatchOpModel fail，看着像算子没装。真踩过，所以这里按**形状**认参数，
    # 不强求顺序：带 / 或者是个已存在的目录，那就是 om 目录；纯数字是 device。
    argv, pair_options = _extract_pair_arguments(sys.argv[1:])
    dry_run = "--dry-run" in argv
    if dry_run:
        argv = [a for a in argv if a != "--dry-run"]
    case_path, op_type, device_id, om_arg = "fused_multi_conv2d_case.bin", "FusedMultiConv2d", 0, None
    positional = []
    for a in argv:
        # 按**后缀和类型**认，不按「带不带斜杠」。上一版是「带 / 就当 om 目录」，
        # 而 case 文件现在都在 out/<名字>/case.bin 这样的子目录里 —— 一进来就被
        # 当成 om 目录，然后 "FusedMultiConv2d" 落到 case 文件那一位上，报「找不到
        # case 文件 FusedMultiConv2d」。
        if a.endswith(".bin"):
            if case_path != "fused_multi_conv2d_case.bin":
                die("给了两个 .bin：%s 和 %s" % (case_path, a))
            case_path = a
        elif a.endswith(".om") or os.path.isdir(a):
            if om_arg is None:
                om_arg = a
            else:
                die("给了两个 om 参数：%s 和 %s" % (om_arg, a))
        elif a.isdigit():
            device_id = int(a)
        else:
            positional.append(a)
    if positional:
        op_type = positional[0]
    if len(positional) > 1:
        die("多余的参数：%s\n        用法: %s <case.bin> [算子名] [device_id] [om目录]\n"
            "        顺序随意 —— .bin 结尾的当 case，.om 结尾或是目录的当离线模型，"
            "纯数字当 device_id，剩下的当算子名。"
            % (positional[1:], os.path.basename(sys.argv[0])))
    # --dry-run：不碰硬件，用 golden 冒充设备输出把主流程走一遍。加它是因为
    # 「只在板上才炸」的问题（比如 int8 没有 y_exact，而下面按 ORDER + GOLDENS
    # 遍历会 KeyError）本地发现不了 —— 单独测 load_case 和 evaluate 不够，
    # 得真的走完 main()。
    if not os.path.isfile(case_path):
        die("找不到 case 文件 %s\n        用法: %s <case.bin> [算子名] [device_id] [om目录]"
            % (case_path, os.path.basename(sys.argv[0])))

    def env_num(name, default, cast):
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        try:
            return cast(v)
        except ValueError:
            die("环境变量 %s=%r 不是合法数字" % (name, v))

    rel_tol = env_num("REL_TOL", 1e-3, float)
    ratio_min = env_num("RATIO_MIN", 1.0, float)
    repeat = env_num("REPEAT", 1, int)
    warmup = env_num("WARMUP", 0, int)
    if repeat < 1:
        die("REPEAT 至少是 1")
    if warmup < 0:
        die("WARMUP 不能为负")
    if not math.isfinite(rel_tol) or rel_tol < 0:
        die("REL_TOL must be finite and nonnegative")
    if not (0.0 <= ratio_min <= 1.0):
        die("RATIO_MIN 要在 [0,1] 之间，传的是 %g" % ratio_min)
    if repeat < 0:
        die("REPEAT 不能为负")

    print('FusedMultiConv2d @ 5102 验证 —— ctypes + aclmdl* 图 om（目标机不需要编译器、不需要算子包）')
    print('  case = "%s"   opType = "%s"   device = %d' % (case_path, op_type, device_id))
    print('  om     = %s' % (om_arg if om_arg else "(不用离线模型，走本机算子信息库)"))
    print('  REL_TOL = %g   RATIO_MIN = %g   repeat = %d   warmup = %d\n'
          % (rel_tol, ratio_min, repeat, warmup))

    om_file = om_dir = None
    if om_arg is not None:
        if os.path.isfile(om_arg):
            om_file = os.path.abspath(om_arg)
            if os.path.getsize(om_file) == 0:
                die("%s 是空文件" % om_file)
            if not om_file.endswith(".om"):
                print("  [!] %s 不是 .om 结尾，确认没传错文件" % os.path.basename(om_file))
            print("  离线模型: %s (%d 字节，只加载这一个)"
                  % (os.path.basename(om_file), os.path.getsize(om_file)))
        elif os.path.isdir(om_arg):
            om_dir = os.path.abspath(om_arg)
            oms = [f for f in os.listdir(om_dir) if f.endswith(".om")]
            if not oms:
                die("%s 下没有 .om —— 在有算子包的机器上跑\n"
                    "        python3 fc2d.py <配置>.json --steps om --soc <soc_version>\n"
                    "        生成，再拷过来（run_profile.py 会自动拷）" % om_dir)
            print("  离线模型目录: %s，共 %d 个 .om: %s"
                  % (om_dir, len(oms), ", ".join(sorted(oms))))
            if len(oms) != 1:
                die("OM directory must contain exactly one model")
            om_file = os.path.join(om_dir, oms[0])
        else:
            die("%s 既不是文件也不是目录" % om_arg)

    tensors, gold_nonzero, probe, sat, y_elems, shift1, shift2, info = load_case(case_path)
    if pair_options:
        if dry_run:
            die("--reference-om requires actual device execution; --dry-run cannot validate fusion precision")
        if om_file is None:
            die("--reference-om requires a fused OM")
        return run_reference_comparison(case_path, om_file, pair_options["--reference-om"], device_id,
                                        pair_options.get("--comparison-output"), rel_tol, repeat, warmup,
                                        compare_reference_golden=pair_options.get("--compare-reference-golden", False))
    print("形状（来自 case 文件的 spec）: %s" % info["shape_text"])
    want_raw = tensors["y_expect"][2]
    yElemBytes = 1 if info["outInt8"] else 2
    # **定义在这里，不要放进 ACL 分支里。** 它既被 ACL 那段用（核对模型输出字节数、
    # 读回设备输出），也被分支外的落盘用。原来它在 ACL 分支的后半段才赋值，而我
    # 在前半段加了一处「模型输出字节数核对」去读它 —— 于是板上第一次跑就
    # UnboundLocalError。dry-run 不进 ACL 分支，没有 NPU 的机器又在 aclInit 就退出，
    # 两种本地测法都盖不到那一行。
    y_bytes = y_elems * yElemBytes
    ySentinel = Y_SENTINEL_I8 if info["outInt8"] else Y_SENTINEL_U16
    if len(want_raw) != y_bytes:
        die("golden 输出 %d 字节，按 %d 个元素应为 %d"
            % (len(want_raw), y_elems, y_bytes))
    want = decode_y(want_raw, info["outInt8"])
    print("case 文件 OK：")
    # 只列**存在**的张量：int8 通路没有 y_exact（它的 golden 是精确整数模型，
    # 不需要 fp16 那边的"纯 fp32 参考"）。写死成 ORDER + GOLDENS 会 KeyError。
    for name in info["order"] + GOLDENS:
        if name not in tensors:
            continue
        dtype, dims, data = tensors[name]
        print("  %-9s %-6s %-16s %9d 字节" % (name, DTYPE_NAME[dtype], dims, len(data)))
    slots = runtime_input_slots(info)
    print("模型运行输入（%d 个；filter 和 scale 是 OM 常量）:" % len(model_input_slots(info)))
    for i, name in enumerate(slots):
        print("  %d: %s" % (i, name))
    if info["outInt8"]:
        print("量化系数（header 调试元数据；实际以 per-channel uint64 张量下发）: "
              "quant_scale1=%.9g quant_scale2=%.9g  relu=%s  bias=%s"
              % (info["qs1"], info["qs2"], "on" if info["relu"] else "off",
                 "on" if info["hasBias"] else "off"))
        print("            out_int8 = saturate( round( acc_int32 * quant_scale ) )，再 relu")
    elif info["isS8F16"]:
        print("量化系数: conv1 quant_scale1=%.9g；conv2 使用 per-channel dequant_scale2；relu=%s bias=%s"
              % (info["qs1"], "on" if info["relu"] else "off",
                 "on" if info["hasBias"] else "off"))
    elif info["isA16W8"]:
        print("A16W8 定标: fixed_shift_value[0]=%d（累加器定标 2^(%d-%d)），"
              "conv1 quant_scale1=%.9g，conv2 使用 per-channel dequant_scale2"
              % (shift1, A16W8_SHIFT_LEN, shift1, info["qs1"]))
    else:
        # 只是打出来给人看。属性在编 om 那一步就烘进 om 了，板侧不下发
        # （见下面 aclmdlAddDatasetBuffer 那段）。算子侧的名字是 fixed_shift_value，
        # 一个 ListInt、每层一项。
        print("定点定标契约（来自 header，编译时核对真实 tiling）fixed_shift_value=[%d, %d]"
              % (shift1, shift2))
        print("            对应累加器定标 2^%d / 2^%d —— 属性越大定标越小，别搞反"
              % (FIX_SHIFT_LEN - shift1, FIX_SHIFT_LEN - shift2))
    print("golden: nonzero=%d/%d sat=%d" % (gold_nonzero, y_elems, sat))
    if gold_nonzero == 0:
        die("golden 全是 0 —— 先别管设备")

    if om_file is not None:
        validate_model_binding(case_path, om_file)
        print("  case/OM SHA256 and production fusion binding verified")
    elif not dry_run:
        die("provide a freshly compiled constant-weight OM and its .fc2d.json binding")

    # dry-run 连库都不加载 —— 它的目的就是在没有 CANN 的机器上验代码路径。
    acl = None if dry_run else load_acl()

    def check(ret, msg):
        if ret != ACL_SUCCESS:
            die(msg % ret if "%d" in msg else "%s (ret=%d)" % (msg, ret))

    # 这两个来自 case 文件本身，和跑不跑硬件无关，所以必须在分叉之前拿到 ——
    # 之前它们在 ACL 那段里，dry-run 走不到，抽样处就 UnboundLocalError。
    y_dtype, y_dims, _ = tensors["y_expect"]
    if dry_run:
        # 用 golden 冒充设备输出，把 ACL 之后的整段（比对、诊断、抽样、判据）
        # 原样走一遍。目的不是验数值 —— 数值必然全对 —— 而是验**代码路径**：
        # 两条通路的解码、属性打印、诊断分支都得能跑到底，不能只在板上才炸。
        print("\n[dry-run] 不碰硬件，用 golden 冒充设备输出走一遍主流程")
        got = want
    else:
        # 模型 API 的符号在这里集中查一次。放在 load_acl 的可选表里是刻意的：
        # 缺哪个要在**用到的地方**报，那时才能说清楚替代方案。
        MODEL_API_SYMS = ("aclmdlLoadFromFile", "aclmdlCreateDesc", "aclmdlGetDesc",
                          "aclmdlGetNumInputs", "aclmdlGetNumOutputs",
                          "aclmdlGetInputSizeByIndex", "aclmdlGetOutputSizeByIndex",
                          "aclmdlCreateDataset", "aclmdlAddDatasetBuffer",
                          "aclmdlExecute", "aclmdlUnload")
        lack = [s for s in MODEL_API_SYMS if getattr(acl, s, None) is None]
        if lack:
            die("这个 CANN 的 libascendcl.so 里没有模型 API: %s\n"
                "    om 是 atc --framework=5 编的**图** om，只能用 aclmdl* 加载执行。\n"
                "    单算子那条（aclopExecuteV2）已经不可用：filters 是 DYNAMIC list，\n"
                "    atc --singleop 表达不了两个实例。" % ", ".join(lack))

        check(acl.aclInit(None), "aclInit = %d")
        # **顺序要紧**：模型 API 必须先 SetDevice 再加载模型。单算子那条 aclopLoad
        # 和 SetDevice 的先后无所谓，这条不是。
        check(acl.aclrtSetDevice(device_id), "aclrtSetDevice(" + str(device_id) + ") = %d —— 芯片被占？")
        # aclmdlExecute 是**同步**的，不吃 stream。这里仍然建一个，只为收尾段对称
        # （aclrtDestroyStream）以及保证有个 context —— 不参与下发。
        stream = ctypes.c_void_p()
        check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream = %d")

        # 图 om 必须指名到**文件**。没有「把目录里所有 om 都装上、让 ACL 自己匹配」
        # 这回事 —— 那是单算子 API 的玩法（aclopSetModelDir），而且正是它导致过
        # 「目录里留着上一个形状的 om，旧的反而先匹配上」。
        if om_file is None:
            if om_dir is None:
                die("没给 om。图 om 要指名到文件：run_fused_multi_conv2d.py case.bin FusedMultiConv2d <device> <xxx.om>")
            oms = [f for f in os.listdir(om_dir) if f.endswith(".om")]
            if len(oms) != 1:
                die("%s 下有 %d 个 .om（%s）。图 om 要指名到文件，不能靠目录匹配。"
                    % (om_dir, len(oms), ", ".join(sorted(oms))))
            om_file = os.path.join(om_dir, oms[0])
            print("  目录里只有一个 .om，用它: %s" % oms[0])
        mid = ctypes.c_uint32()
        check(acl.aclmdlLoadFromFile(om_file.encode(), ctypes.byref(mid)),
              "aclmdlLoadFromFile(%s) = " % os.path.basename(om_file) + "%d\n"
              "        这个 om 是在**编译机**上生成的，板上不需要装算子包（kernel 的 ELF\n"
              "        就在 om 文件里）。加载失败通常是 soc_version 和这块芯片不符。")
        model_id = mid.value
        mdesc = acl.aclmdlCreateDesc()
        if not mdesc:
            die("aclmdlCreateDesc 返回 null")
        check(acl.aclmdlGetDesc(mdesc, model_id), "aclmdlGetDesc = %d")
        n_in = int(acl.aclmdlGetNumInputs(mdesc))
        n_out = int(acl.aclmdlGetNumOutputs(mdesc))
        print("  aclmdlLoadFromFile OK: model_id=%d, %d 个输入, %d 个输出" % (model_id, n_in, n_out))
        if n_in != len(model_input_slots(info)):
            die("model has %d inputs; differs from the verified fusion graph, regenerate OM" % n_in)
        if n_out != 1:
            die("这个模型报了 %d 个输出，FusedMultiConv2d 只有一个 y" % n_out)

        keep = []          # 挡住 GC：host 侧缓冲在 memcpy 之前不能被回收
        dev_ptrs, descs, bufs = [], [], []

        def make_operand(dtype, dims, data):
            dev = ctypes.c_void_p()
            check(acl.aclrtMalloc(ctypes.byref(dev), len(data), ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(%d) 失败, ret = " % len(data) + "%d")
            host = ctypes.create_string_buffer(data, len(data))
            keep.append(host)
            check(acl.aclrtMemcpy(dev, len(data), ctypes.cast(host, ctypes.c_void_p),
                                  len(data), ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D = %d")
            arr = (ctypes.c_int64 * len(dims))(*dims)
            desc = acl.aclCreateTensorDesc(dtype, len(dims), arr, ACL_FORMAT_ND)
            if not desc:
                die("aclCreateTensorDesc 返回 null")
            buf = acl.aclCreateDataBuffer(dev, len(data))
            if not buf:
                die("aclCreateDataBuffer 返回 null")
            dev_ptrs.append(dev)
            descs.append(desc)
            bufs.append(buf)
        # ---- 按**模型自己报的**输入表喂数据 ----
        # 和单算子 om 的本质差别：单算子那套「槽位不能压缩、q1/q2 不能左移」的规则
        # 在这条路上不适用 —— 那条规则是 aclopExecuteV2 按 numInputs 匹配 .om 带来的。
        # Unused bias/scale/relu_weight instances are internal Const nodes
        # created by the production pass; they are not runtime placeholders.
        present = model_input_slots(info)
        missing = [name for name in present if name not in tensors]
        if missing:
            die("case.bin missing graph inputs: %s" % missing)

        # **优先按名字定位下标，不赌模型的输入顺序。** om 的模型输入名就是 onnx 图的
        # 输入名，而那又是 case.bin 里的张量名 —— fc2d_onnx.py 刻意让三者同名，
        # 所以正常情况下每个都查得到。老的 run_om.py 在这里吃过亏：它只按下标喂，
        # GetInputNameByIndex 的结果仅打印给人看。
        feed, how = [None] * n_in, "名字"
        idx_by_name = getattr(acl, "aclmdlGetInputIndexByName", None)
        if idx_by_name is None:
            how = None
        else:
            for slot in present:
                k = ctypes.c_size_t()
                if idx_by_name(mdesc, slot.encode(), ctypes.byref(k)) != ACL_SUCCESS:
                    how = None
                    break
                i = int(k.value)
                if not (0 <= i < n_in) or feed[i] is not None:
                    how = None
                    break
                feed[i] = slot
        if how is None or any(f is None for f in feed):
            # 退回 IR 顺序。**这是猜的**，所以要打出来让人看见；下面那段字节数校验
            # 是唯一的安全网。
            if n_in != len(present):
                die("模型报了 %d 个输入，而这条 dtype 通路有 %d 个有数据的槽（%s），\n"
                    "    而且按名字取下标也没成功 —— 对不上就不能靠顺序猜。\n"
                    "    先确认这个 .om 是用同一个 case.bin 编的。"
                    % (n_in, len(present), ", ".join(present)))
            feed, how = list(present), "IR 顺序（按名字取下标没成功，这是猜的）"

        name_by_idx = getattr(acl, "aclmdlGetInputNameByIndex", None)
        print("\n模型输入表（定位方式: %s）:" % how)
        bad = False
        for i, slot in enumerate(feed):
            nb = int(acl.aclmdlGetInputSizeByIndex(mdesc, i))
            mn = "?"
            if name_by_idx is not None:
                raw = name_by_idx(mdesc, i)
                if raw:
                    mn = raw.decode("utf-8", "replace")
            data = tensors[slot][2]
            mark = "OK" if len(data) == nb else "** 模型要 %d，case.bin 给 %d **" % (nb, len(data))
            if len(data) != nb:
                bad = True
            src = "case.bin 的 %-16s" % slot
            print("  [%d] 模型说 %-16s <- %s %9d 字节  %s" % (i, mn, src, nb, mark))
        if bad:
            die("输入字节数和模型要的对不上。顺序错了也会这样 —— 上面那张表就是模型要的顺序。")

        for i, slot in enumerate(feed):
            dtype, dims, data = tensors[slot]
            make_operand(dtype, dims, data)

        # 输出缓冲预填哨兵 127：kernel 一个字节都没写的话，读回来还是 127，
        # 这是"返回码 0 但什么都没算"唯一能被抓住的地方。
        make_operand(y_dtype, y_dims, bytes([Y_SENTINEL_BYTE]) * y_bytes)

        # NIN 在下面还当"输出操作数的下标"用（读回 dev_ptrs[NIN]、收尾 range(NIN+1)），
        # 所以它必须等于模型的输入个数，不能再是 len(INPUT_SLOTS)。
        NIN = n_in
        out_nb = int(acl.aclmdlGetOutputSizeByIndex(mdesc, 0))
        if out_nb != y_bytes:
            die("模型的输出是 %d 字节，case.bin 的 y 是 %d 字节 —— 形状对不上，"
                "这个 .om 大概不是用这个 case.bin 编的" % (out_nb, y_bytes))

        ds_in = acl.aclmdlCreateDataset()
        ds_out = acl.aclmdlCreateDataset()
        if not ds_in or not ds_out:
            die("aclmdlCreateDataset 返回 null")
        for i in range(NIN):
            check(acl.aclmdlAddDatasetBuffer(ds_in, bufs[i]),
                  "aclmdlAddDatasetBuffer(in %d) = " % i + "%d")
        check(acl.aclmdlAddDatasetBuffer(ds_out, bufs[NIN]), "aclmdlAddDatasetBuffer(out) = %d")

        # **属性不下发。** 图 om 的属性在编译期就烘进去了，aclmdlExecute 不吃属性。
        # 定标 fixed_shift_value 是私有 ListInt，每层一项；A16W8 仅使用第一项。
        # 融合后的真实 native tiling 必须与 case 的有效定标一致，运行器不覆写属性。
        # Geometric attributes and fixed-point calibration below describe the
        # case contract. Compilation verified the actual native tiling before
        # writing the binding; no attributes are changed during execution.
        attr = None
        kh2 = info.get("kh2") or info["kh"]
        kw2 = info.get("kw2") or info["kw"]
        # dtype 和 bias_mask 必须在这里打出来：它们是两个**静默改变行为**的属性
        # —— dtype 给错会把整条通路从「int8 出口 VREQ8」换成「fp16 出口 VDEQF16」
        # 而 om 照样编得过去；bias_mask 给错会静默多加或少加一次 bias。
        # 板上拿到错数时，这一行是唯一能核对「om 到底按什么编的」的地方。
        # 与融合后 tiling 验证采用同一份 case spec。
        dtype_echo = 2 if info["outInt8"] else 1   # ge::DataType: fp16=1, int8=2
        # bias_mask 现在是**每层一项的列表**（算子侧是 ListBool），不再是整数位掩码。
        bias_mask_echo = [1, 1] if info["bias"] else [0, 0]
        # 列表**先格式化好再塞进 %s** —— 直接把列表交给 % 会被当成多个参数展开。
        # （上一版这里是 0b%02b，而 % 没有 %b 这个转换符，在板上抛
        #  ValueError: unsupported format character b。那个错是
        #  tests/stub_ascendcl 的桩抓到的 —— 板侧下发那一段 --dry-run 和没有 NPU
        #  的机器都盖不到，和当初 y_bytes 那个先用后赋同一类。）
        bias_mask_str = "[%d,%d]" % (bias_mask_echo[0], bias_mask_echo[1])
        # relu 的值是硬件的 ReluMode：0=关 1=普通 2=标量斜率 3=逐通道斜率。
        # 目前只会发 0/1 —— 2/3 要用 relu_weight 那张斜率表，算子侧会拒。
        relu_str = "[%d,%d]" % (1 if info["relu1"] else 0, 1 if info["relu2"] else 0)
        print("算子属性（编译期已烘进 om，此处仅供核对）: dtype=%d bias_mask=%s "
              "private fixed_shift_value=[%d,%d] enable_relu=%s "
              "kernel=%dx%d>%dx%d strides=%d/%d pads=%d,%d/%d,%d"
              % (dtype_echo, bias_mask_str, shift1, shift2, relu_str,
                 info["kh"], info["kw"], kh2, kw2, info["s1"], info["s2"],
                 info["ph1"], info["pw1"], info["ph2"], info["pw2"]))
        print("            保留属性（声明了但算子还不支持，全发 IR 缺省）: "
              "dilations=[1,1,1,1,1,1,1,1] groups=[1,1] data_format=[0,0](NCHW) offset_x=[0,0] "
              "round_mode=[0,0](rint) pad_mode=[0,0](SPECIFIC) enable_hf32=[false,false]")

        def launch():
            # aclmdlExecute 是**同步**的：返回时 kernel 已经跑完，不需要
            # aclrtSynchronizeStream。（异步要的是 aclmdlExecuteAsync + stream。）
            return acl.aclmdlExecute(model_id, ds_in, ds_out)

        # 总共下发 warmup + repeat 次；只有后 repeat 次计入统计。
        durs = []
        for k in range(warmup + repeat):
            t0 = time.perf_counter()
            ret = launch()
            if ret != ACL_SUCCESS:
                if k == 0:
                    die("aclmdlExecute(model_id=%d) = %d\n"
                        "        模型已经加载成功了，所以不是「算子没装」—— kernel 的 ELF 就在\n"
                        "        om 文件里。图 om 走到这一步失败，先看这几条：\n"
                        "          * 输入个数/字节数：上面那张模型输入表已经逐项核过了，\n"
                        "            如果那里全 OK，就不是形状问题。\n"
                        "          * device 侧日志（~/ascend/log/ 或 /var/log/npu）里的 AI Core 异常。\n"
                        "          * 这个 .om 的 soc_version 和这块芯片是否一致。\n"
                        "        注意：属性对不上导致的 MatchOpModel fail 在这条路上**不可能**发生，\n"
                        "        属性是编译期烘进 om 的，没有运行期匹配这回事。"
                        % (model_id, ret))
                die("第 %d 次 aclmdlExecute = %d" % (k + 1, ret))
            dt = (time.perf_counter() - t0) * 1e6
            if k >= warmup:
                durs.append(dt)
        if repeat > 1 or warmup:
            print("\n[耗时] host 侧墙钟，含下发和同步；设备侧以 msprof 为准")
            report_times(durs)

        out = ctypes.create_string_buffer(y_bytes)
        check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), y_bytes, dev_ptrs[NIN], y_bytes,   # NIN 号才是输出，别写死
                              ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
        got = decode_y(out.raw[:y_bytes], info["outInt8"])

    # ------------------------------------------------------------ 比对
    r = evaluate(got, want, rel_tol, ySentinel)
    comparison = summarize_output_comparison(got, want, y_dims, y_dtype, rel_tol)
    print_output_comparison(comparison, "FusedMultiConv2d", "CPU golden")
    nonzero, unwritten, mismatches = r["nonzero"], r["unwritten"], r["mismatches"]
    ratio = r["prec_ok"] / float(r["n"])

    # 设置 FC2D_DUMP 保存设备原始输出。
    dump_path = os.environ.get("FC2D_DUMP", "")
    if (not dry_run) and dump_path and dump_path != "-":
        dump_raw(dump_path, out.raw[:y_bytes], "设备输出 (%s NCHW [1,%d,%d,%d])"
                 % ("int8" if info["outInt8"] else "fp16", y_dims[1], y_dims[2], y_dims[3]))

    if not dry_run:
        if attr is not None:
            acl.aclopDestroyAttr(attr)
        acl.aclmdlDestroyDataset(ds_in)
        acl.aclmdlDestroyDataset(ds_out)
        for i in range(NIN + 1):
            acl.aclDestroyDataBuffer(bufs[i])
            acl.aclDestroyTensorDesc(descs[i])
            if dev_ptrs[i]:
                acl.aclrtFree(dev_ptrs[i])
        # 先卸模型再销毁 desc：desc 是从 model_id 取出来的。
        acl.aclmdlUnload(model_id)
        acl.aclmdlDestroyDesc(mdesc)
        acl.aclrtDestroyStream(stream)
        acl.aclrtResetDevice(device_id)
        acl.aclFinalize()

    print()
    if any(comparison["nonfinite_count"].values()):
        print("[FAIL] nonfinite output: actual=%d golden=%d" %
              (comparison["nonfinite_count"]["fused"], comparison["nonfinite_count"]["reference"]))
        return 1
    if unwritten:
        print("[FAIL] %d 个输出从来没被写过 —— kernel 没跑，或只跑了一部分核" % unwritten)
        return 1
    if nonzero != gold_nonzero:
        print("[FAIL] 非零个数 %d != golden 的 %d —— 写的分布不对" % (nonzero, gold_nonzero))
        return 1
    if ratio < ratio_min:
        print("[FAIL] 达标比例 %.6f%% < 下限 %.6f%%（%d 个元素超出相对误差 %g）"
              % (ratio * 100.0, ratio_min * 100.0, r["n"] - r["prec_ok"], rel_tol))
        return 1
    if mismatches:
        # 比例过了但仍有不一致：说明 RATIO_MIN 被放宽过。不当失败，但要说清楚。
        print("[PASS*] 达标比例 %.6f%% 满足下限，但仍有 %d 个元素与 golden 不逐位相等"
              % (ratio * 100.0, mismatches))
        return 0
    print("[PASS] %d 个输出全部写过，与 CPU golden 逐位相等，相对误差达标比例 %.6f%%"
          % (y_elems, ratio * 100.0))
    return 0

if __name__ == "__main__":
    sys.exit(main())
