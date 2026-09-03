#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""FusedConv2d @ 5102 单算子验证 —— 目标机器上不需要编译器。

用 ctypes 直接调 libascendcl.so 里的 aclopExecuteV2，输入和 golden 全部来自
gen_case 生成的 .bin。所以这台机器只要有 python3 + CANN 就够了，不需要 g++、
不需要 aclnn 头、不需要 numpy。

    python3 run_fused_conv2d.py [case.bin] [op_type] [device_id] [om]
    默认  fused_conv2d_case.bin  FusedConv2d  0  (无 om)

第 4 个参数 om 可以是**单个 .om 文件**，也可以是装着 .om 的目录：
    ... 0 om_out/0_FusedConv2d_1.om     只加载这一个（走 aclopLoad）
    ... 0 om_out                        加载这个目录下所有 .om（走 aclopSetModelDir）

环境变量：
    REL_TOL=1e-3     相对误差判据（默认 1e-3）
    RATIO_MIN=1.0    达标比例的下限，低于它算 FAIL（默认 1.0，即要求 100%）
    REPEAT=1         计时的下发次数。想避开冷启动就调大，比如 REPEAT=10
    WARMUP=0         正式计时前先空跑多少次（不计入统计）。总次数 = WARMUP + REPEAT

设备上没装带 FusedConv2d 的算子包时，传第 4 个参数：那是在有算子包的机器上用
build_om.sh 编出来的单算子离线模型。运行时就从它里面找算子，不再要求本机的算子
信息库里有这个算子。

注意 aclopSetModelDir 只吃**目录**，而且会把目录下所有 .om 都加载进来。所以给单个
文件时走的是 aclopLoad(把文件读进内存再注册)，只注册你指定的这一个 —— 目录里有多个
.om 时这条更干净。老版本的 CANN 没有 aclopLoad 的话，会退回"取所在目录 +
aclopSetModelDir"，并明确告诉你。

判据和 C++ 版完全一致，输出行可以逐字对比。
"""

import ctypes
import os
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

DTYPE_NAME = {0: "float32", 1: "float16", 2: "int8", 3: "int32", 10: "uint64"}
DTYPE_SIZE = {0: 4, 1: 2, 2: 1, 3: 4, 10: 8}

# y 改成 fp16（fixpipe 走 VDEQF16）之后，哨兵按字节填 0x7F，两字节拼起来是
# 0x7F7F —— 一个 fp16 NaN（阶码全 1、尾数非零）。golden 永远算不出 NaN，所以
# 「有没有被写过」这个判据不需要再论证合法输出取不到哨兵值。
Y_SENTINEL_BYTE = 0x7F
Y_SENTINEL_U16 = 0x7F7F
Y_ELEM_BYTES = 2

HDR_FMT = "<8sIIQQQQQ"          # magic, version, ntensors, nonzero, probe, sat, y_elems, attrs

# 第 5 个字段原本恒为 0（int8 时代的 ties 计数），现在放探针 id。
# 探针 = 把某一层的权重换成中心抽头恒等，用来把两层拆开单独看。
# 权重是运行时输入不是属性，所以换探针只要换 .bin，不用重编 .om。
PROBES = {
    0: "none —— 随机权重，正常用例",
    1: "passthrough —— conv1/conv2 都恒等、bias 全 0；y 应当 == relu(x[c][2h][2w])，"
       "c>=32 全 0。纯数据搬运测试，和 fixShiftVal 无关",
    2: "mid —— 只有 conv2 恒等；y 直接暴露 conv1 的输出（下采样后）",
    3: "chanid —— 恒等权重 + x[c][h][w] = (c+1)*2^-14；每个输出通道的值直接编码"
       "「哪个输入通道落到了这里」",
    4: "colid —— 恒等权重 + x[c][h][w] = (w+1)*2^-14；值编码「哪一列落到了这里」",
}
# attrs 的低 8 位是 fixed_shift1，次 8 位是 fixed_shift2。放在文件里而不是脚本
# 里写死，是因为它由 golden 按数据算出来，主机和设备必须用同一个值。
HDR_LEN = struct.calcsize(HDR_FMT)
REC_FMT = "<16sII4qQ"           # name, dtype, ndim, dims[4], nbytes
REC_LEN = struct.calcsize(REC_FMT)

ORDER = ["x", "filter1", "bias1", "filter2", "bias2"]
# 两个 golden：y_expect 是定点模型（假设成立时应逐位相等），y_exact 是纯 fp32
# 参考（判「有没有在算这个卷积」，不受定点假设影响）。见 golden.h 顶部那段。
GOLDENS = ["y_expect", "y_exact"]

# 硬件移位域宽度。属性 S 和实际定标指数 F 的关系恒为 F = FIX_SHIFT_LEN - S，
# 见 dav_m510/kernel_operator_fixpipe_impl.h:77 SetDeqScalarDepOnMode。
FIX_SHIFT_LEN = 58


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
        die("找不到 %s —— 在能编译的机器上跑 ./gen_case 生成，再拷过来" % path)
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < HDR_LEN:
        die("%s 只有 %d 字节，文件不完整（传输中断？）" % (path, len(blob)))

    magic, version, ntensors, nonzero, probe, sat, y_elems, attrs = struct.unpack_from(HDR_FMT, blob, 0)
    if magic != b"FC2DCASE":
        die("%s 不是 case 文件（magic = %r）" % (path, magic))
    if version != 2:
        die("case 文件版本 %d，本脚本只认 2 —— gen_case 和 run_fused_conv2d.py 得配套。\n        版本 1 是上一版的量化接口（8 输入），这一版是定点接口（5 输入 + 2 属性）。" % version)

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

    # 元素数和 dims 必须自洽。这一条能挡住"传了半个文件却刚好没越界"。
    for name, (dtype, dims, data) in tensors.items():
        want = DTYPE_SIZE[dtype]
        for d in dims:
            want *= d
        if want != len(data):
            die("张量 %s: dims=%s dtype=%s 应为 %d 字节，实际 %d"
                % (name, dims, DTYPE_NAME[dtype], want, len(data)))

    for name in ORDER + GOLDENS:
        if name not in tensors:
            die("case 文件里缺张量 %s" % name)

    shift1 = int(attrs & 0xFF)
    shift2 = int((attrs >> 8) & 0xFF)
    if not (0 <= shift1 <= 58 and 0 <= shift2 <= 58):
        die("header 里的定点定标 %d / %d 超出 [0,58] —— case 文件版本对不上？" % (shift1, shift2))
    return tensors, nonzero, probe, sat, y_elems, shift1, shift2


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
        ("aclopExecuteV2", [c_cp, c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp),
                            c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp), c_vp, c_vp], c_i),
    ]
    # 这两个只有用离线模型时才需要，缺了不该在这里就把脚本打死 ——
    # 用到的时候再报，那时的报错还能顺带说清楚该退回哪条路。
    opt = [
        ("aclopSetModelDir", [c_cp], c_i),
        ("aclopLoad", [c_vp, c_sz], c_i),
        ("aclGetRecentErrMsg", [], c_cp),
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


def decode_f16(buf):
    """字节串 -> (值列表 float, 位模式列表 uint16)。

    值用来算相对误差，位模式用来做逐位比较和哨兵判定 —— 两者缺一不可：
    NaN != NaN，只看值会把「没写过」误判成「写错了」。
    struct 的 'e' 就是 IEEE754 binary16，Python 3.6 起自带。
    """
    n = len(buf) // Y_ELEM_BYTES
    return list(struct.unpack("<%de" % n, buf)), list(struct.unpack("<%dH" % n, buf))


def report_ratio(got, want, n):
    """got / want 的比值分布。这是判「定标错了」还是「算错了」的直接证据。

    定点链里每一环的误差都是**乘性**的：任何一级的 shift 猜错，输出就整体差一个
    2 的整数次幂。所以只要两边都非零，看 log2(got/want)：
      集中在某个整数 k  -> 定标差 2^k，改 shift 就行，卷积本身是对的；
      散开 / 大量反号   -> 不是定标问题（溢出饱和、或者真算错了）。
    饱和的点要摘出去单独看 —— 它们的比值只反映饱和值，不反映定标。
    """
    import math
    hist, sign_flip, sat_like, both_nz = {}, 0, 0, 0
    for i in range(n):
        g, w = got[0][i], want[0][i]
        if g == 0.0 or w == 0.0:
            continue
        if g != g or w != w or abs(g) == float("inf"):
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
        print("       => 有主峰 2^%d：定标整体差这么多，改 shift 即可，卷积是对的。" % top[0][0])
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
        if g != g or t != t or abs(g) == float("inf"):
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
    """把偏差换算成累加器 LSB —— 这是分开两类失败的判据。

    定点链是 acc_i32 = round(真值 * 2^F)，出口再乘 2^-F，F = 58 - fixed_shift2。
    所以输出的最小可分辨间隔就是 2^-F。换算成 LSB 之后：
      偏差都在个位数 LSB   -> 只是那次 round 发生的级别猜得不同（每个乘积 /
                             每条 mmad / 整条 K 累完），卷积本身是对的；
      偏差成千上万个 LSB   -> 算错了，或者定标错了、累加器溢出了。
    """
    deq_exp2 = FIX_SHIFT_LEN - shift2
    lsb = 2.0 ** (-deq_exp2)
    print("\n[定点] conv2 定标 2^%d（属性 S=%d），累加器 1 LSB = %.6g" % (deq_exp2, shift2, lsb))
    if not mismatches:
        return
    edges = [1, 2, 4, 16, 256, 4096]
    names = ["<=1", "<=2", "<=4", "<=16", "<=256", "<=4096", ">4096"]
    hist = dict.fromkeys(names, 0)
    max_lsb, max_i, nonfinite = 0.0, -1, 0
    for i in range(r["n"]):
        if got[1][i] == want[1][i]:
            continue
        d = got[0][i] - want[0][i]
        if d != d or d in (float("inf"), float("-inf")):
            nonfinite += 1
            continue
        d = abs(d) / lsb
        for e, nm in zip(edges, names):
            if d <= e:
                hist[nm] += 1
                break
        else:
            hist[">4096"] += 1
        if d > max_lsb:
            max_lsb, max_i = d, i
    if max_i >= 0:
        print("       最大偏差 %.1f LSB @ %d (%s): got %.6g, want %.6g"
              % (max_lsb, max_i, idx_label(max_i, dims), got[0][max_i], want[0][max_i]))
    for nm in names:
        if hist[nm]:
            print("         %-8s %9d  (%.6f%%)" % (nm, hist[nm], hist[nm] * 100.0 / mismatches))
    if nonfinite:
        print("         非有限差 %d 个（got 或 want 是 Inf/NaN）" % nonfinite)
    if max_lsb <= 16.0 and nonfinite == 0:
        print("       => 偏差全在个位数 LSB 量级：卷积算对了，差的只是 round 的级别。")
    else:
        print("       => 偏差远超 LSB 量级：这不是 round 的问题，是算错了或累加器溢出了。")


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



def evaluate(got, want, rel_tol):
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
    }
    gv, gb = got
    wv, wb = want
    if gb == wb:
        # 逐位一致：相对误差全是 0
        r["hist"]["=0"] = n
        r["zero_golden"] = sum(1 for x in wv if x == 0.0)
        return r

    zero_golden = 0
    prec_ok = 0
    for i in range(n):
        g, w = gv[i], wv[i]
        g8, w8 = gb[i], wb[i]
        if g8 == Y_SENTINEL_U16 and w8 != Y_SENTINEL_U16:
            r["unwritten"] += 1
        if g8 != w8:
            if r["first_bad"] < 0:
                r["first_bad"] = i
            r["mismatches"] += 1
            # fp16 上「差 1」就是差 1 个 ULP（同号有限值的位模式相邻）。
            if abs(g8 - w8) == 1:
                r["off_by_one"] += 1
        if w == 0.0:
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


# ---------------------------------------------------------------- main
def main():
    case_path = sys.argv[1] if len(sys.argv) > 1 else "fused_conv2d_case.bin"
    op_type = sys.argv[2] if len(sys.argv) > 2 else "FusedConv2d"
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    om_arg = sys.argv[4] if len(sys.argv) > 4 else None

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
    if rel_tol < 0:
        die("REL_TOL 不能为负")
    if not (0.0 <= ratio_min <= 1.0):
        die("RATIO_MIN 要在 [0,1] 之间，传的是 %g" % ratio_min)
    if repeat < 0:
        die("REPEAT 不能为负")

    print('FusedConv2d @ 5102 单算子验证 —— ctypes + aclopExecuteV2（目标机不需要编译器）')
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
                die("%s 下没有 .om —— 在有算子包的机器上跑 build_om.sh 生成，拷过来" % om_dir)
            print("  离线模型目录: %s，共 %d 个 .om: %s"
                  % (om_dir, len(oms), ", ".join(sorted(oms))))
            if len(oms) > 1:
                print("     (目录形式会把这些全部加载。只想用其中一个就直接传那个文件)")
        else:
            die("%s 既不是文件也不是目录" % om_arg)

    tensors, gold_nonzero, probe, sat, y_elems, shift1, shift2 = load_case(case_path)
    want_raw = tensors["y_expect"][2]
    if len(want_raw) != y_elems * Y_ELEM_BYTES:
        die("golden 输出 %d 字节，按 %d 个 fp16 元素应为 %d"
            % (len(want_raw), y_elems, y_elems * Y_ELEM_BYTES))
    want = decode_f16(want_raw)
    print("case 文件 OK：")
    for name in ORDER + GOLDENS:
        dtype, dims, data = tensors[name]
        print("  %-9s %-6s %-16s %9d 字节" % (name, DTYPE_NAME[dtype], dims, len(data)))
    print("定点定标（来自 header，将作为算子属性下发）: fixed_shift1=%d fixed_shift2=%d"
          % (shift1, shift2))
    print("            对应累加器定标 2^%d / 2^%d —— 属性越大定标越小，别搞反"
          % (FIX_SHIFT_LEN - shift1, FIX_SHIFT_LEN - shift2))
    print("golden: nonzero=%d/%d sat=%d" % (gold_nonzero, y_elems, sat))
    print("探针: %s" % PROBES.get(probe, "未知 id %d —— .bin 和脚本版本可能对不上" % probe))
    if gold_nonzero == 0:
        die("golden 全是 0 —— 先别管设备")

    acl = load_acl()

    def check(ret, msg):
        if ret != ACL_SUCCESS:
            die(msg % ret if "%d" in msg else "%s (ret=%d)" % (msg, ret))

    check(acl.aclInit(None), "aclInit = %d")
    # 离线模型的注册。要在 aclopExecuteV2 之前，和 aclrtSetDevice 的先后无所谓。
    om_keep = []
    if om_file is not None:
        fn = getattr(acl, "aclopLoad", None)
        if fn is not None:
            with open(om_file, "rb") as f:
                blob = f.read()
            buf = ctypes.create_string_buffer(blob, len(blob))
            om_keep.append(buf)  # 挡住 GC，ACL 可能还引用着这块内存
            check(fn(ctypes.cast(buf, ctypes.c_void_p), len(blob)),
                  "aclopLoad(%s) = " % os.path.basename(om_file) + "%d")
            print("  aclopLoad OK (%d 字节)" % len(blob))
        else:
            d = os.path.dirname(om_file)
            smd = getattr(acl, "aclopSetModelDir", None)
            if smd is None:
                die("这个 CANN 的 libascendcl.so 里既没有 aclopLoad 也没有 aclopSetModelDir，\n"
                    "    用不了离线模型。只能在设备上装带这个算子的算子包。")
            print("  [!] 这个 CANN 没有 aclopLoad，退回 aclopSetModelDir(%s)。" % d)
            print("      注意这会把该目录下**所有** .om 都加载进来。")
            check(smd(d.encode()), "aclopSetModelDir = %d")
            print("  aclopSetModelDir OK")
    elif om_dir is not None:
        fn = getattr(acl, "aclopSetModelDir", None)
        if fn is None:
            die("这个 CANN 的 libascendcl.so 里没有 aclopSetModelDir，用不了离线模型目录")
        check(fn(om_dir.encode()), "aclopSetModelDir(%s) = " % om_dir + "%d")
        print("  aclopSetModelDir OK")
    check(acl.aclrtSetDevice(device_id), "aclrtSetDevice(" + str(device_id) + ") = %d —— 芯片被占？")
    stream = ctypes.c_void_p()
    check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream = %d")

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

    # ABI 顺序钉死，和 op_host/fused_conv2d_def.cpp 的 Input() 调用顺序一一对应：
    #   x, scale_x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    # 少传一个 ACL 不一定报错，它可能把后面的实参往前挪，于是 filter1 被当成 scale_x。
    for name in ORDER:
        dtype, dims, data = tensors[name]
        make_operand(dtype, dims, data)

    # 输出缓冲预填哨兵 127：kernel 一个字节都没写的话，读回来还是 127，
    # 这是"返回码 0 但什么都没算"唯一能被抓住的地方。
    y_dtype, y_dims, _ = tensors["y_expect"]
    make_operand(y_dtype, y_dims, bytes([Y_SENTINEL_BYTE]) * (y_elems * Y_ELEM_BYTES))

    NIN = len(ORDER)
    in_desc = (ctypes.c_void_p * NIN)(*descs[:NIN])
    in_buf = (ctypes.c_void_p * NIN)(*bufs[:NIN])
    out_desc = (ctypes.c_void_p * 1)(descs[NIN])
    out_buf = (ctypes.c_void_p * 1)(bufs[NIN])

    # 两个定点定标是**必需属性**。顺序和 fused_conv2d_def.cpp 里 Attr() 的调用
    # 顺序一致（fixed_shift1 在前），tiling 侧按 GetInt(0)/GetInt(1) 取。
    attr = acl.aclopCreateAttr()
    if not attr:
        die("aclopCreateAttr 返回 null")
    setattr_fn = getattr(acl, "aclopSetAttrInt", None)
    if setattr_fn is None:
        die("这个 CANN 的 libascendcl.so 里没有 aclopSetAttrInt —— 属性传不下去")
    for nm, v in (("fixed_shift1", shift1), ("fixed_shift2", shift2)):
        rc = setattr_fn(attr, nm.encode(), v)
        if rc != ACL_SUCCESS:
            die("aclopSetAttrInt(%s=%d) = %d" % (nm, v, rc))

    def launch():
        return acl.aclopExecuteV2(op_type.encode(), NIN, in_desc, in_buf,
                                  1, out_desc, out_buf, attr, stream)

    # 总共下发 warmup + repeat 次；只有后 repeat 次计入统计。
    durs = []
    for k in range(warmup + repeat):
        t0 = time.perf_counter()
        ret = launch()
        if ret != ACL_SUCCESS:
            if k == 0:
                die('aclopExecuteV2("%s") = %d\n'
                    "        走 .om（已 aclopSetModelDir）时最常见的其实不是「没装」，而是**匹配不上**：\n"
                    "          ACL 拿 op 类型 + 每个 tensor 的 shape/dtype/format + **全部 attr 的值**\n"
                    "          一起去匹配 .om，任何一项对不上都报成这个「算子没找到」。\n"
                    "          本次下发的属性: fixed_shift1=%d fixed_shift2=%d\n"
                    "          编 .om 时用的值在 om_out/singleop_used.json 里，先比这两个数。\n"
                    "          对不上 -> 重跑 build_om.sh（它会从 .bin 的 header 现读，不用手改）\n"
                    "        真的没装 -> grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/\n"
                    "        装了但选不出 kernel -> shape/dtype 和 binary.json 里登记的组合对不上"
                    % (op_type, ret, shift1, shift2, op_type))
            die("第 %d 次 aclopExecuteV2 = %d" % (k + 1, ret))
        ret = acl.aclrtSynchronizeStream(stream)
        if ret != ACL_SUCCESS:
            die("第 %d 次 aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志"
                % (k + 1, ret))
        dt = (time.perf_counter() - t0) * 1e6
        if k >= warmup:
            durs.append(dt)
    if repeat > 1 or warmup:
        print("\n[耗时] host 侧墙钟，含下发和同步；设备侧以 msprof 为准")
        report_times(durs)

    y_bytes = y_elems * Y_ELEM_BYTES
    out = ctypes.create_string_buffer(y_bytes)
    check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), y_bytes, dev_ptrs[NIN], y_bytes,   # NIN 号才是输出，别写死
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
    got = decode_f16(out.raw[:y_bytes])

    # ------------------------------------------------------------ 比对
    r = evaluate(got, want, rel_tol)
    nonzero, unwritten, mismatches = r["nonzero"], r["unwritten"], r["mismatches"]
    ratio = r["prec_ok"] / float(r["n"])

    print("\n[fused-conv2d-5102-device] out_elems=%d nonzero=%d golden_nonzero=%d "
          "unwritten=%d mismatches=%d" % (y_elems, nonzero, gold_nonzero, unwritten, mismatches))

    if mismatches:
        fb = r["first_bad"]
        print("first mismatch @ %d: got %.6g (0x%04x), want %.6g (0x%04x) (%s)"
              "；差 1 个 ULP 的 %d / 共 %d"
              % (fb, got[0][fb], got[1][fb], want[0][fb], want[1][fb], idx_label(fb, y_dims),
                 r["off_by_one"], mismatches))

    # ------------------------------------------------------------ 精度
    print("\n[精度] 相对误差 <= %g 的比例: %d / %d = %.6f%%   (下限 %.6f%%)"
          % (rel_tol, r["prec_ok"], r["n"], ratio * 100.0, ratio_min * 100.0))
    mr = r["max_rel"]
    if r["max_rel_idx"] >= 0:
        mi = r["max_rel_idx"]
        print("       最大相对误差 %s @ %d (%s): got %.6g, want %.6g"
              % ("inf" if mr == float("inf") else "%.6e" % mr, mi, idx_label(mi, y_dims),
                 got[0][mi], want[0][mi]))
    else:
        print("       最大相对误差 0")
    print("       golden 为 0 的 %d 个元素按绝对误差判" % r["zero_golden"])
    print("       相对误差分布:")
    for k in ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]:
        v = r["hist"][k]
        if v:
            print("         %-14s %9d  (%.6f%%)" % (k, v, v * 100.0 / r["n"]))
    # 这个提醒值得常驻，而且**输出改成 fp16 之后结论反过来了**。
    #
    # int8 时代：|golden| <= 100，最小非零差值是 1，所以任何一个 LSB 的偏差相对
    # 误差都 >= 1e-2，1e-3 的判据等价于逐位相等。
    #
    # fp16 时代：1 个 ULP 的相对误差最大只有 2^-10 ≈ 9.77e-4（正规数，尾数 10 位），
    # **小于 1e-3**。也就是说差 1 个 ULP 的结果照样能通过 1e-3 的判据。
    # 所以现在这两条判据不再等价，宽严关系是：逐位相等 严于 1e-3。
    # 真正的判据是 mismatches == 0；达标比例 100% 但 mismatches != 0 就是 [PASS*]。
    print("       注: 输出是 fp16，1 个 ULP 的相对误差 <= 2^-10 ≈ 9.77e-4 < %g，" % rel_tol)
    print("           所以 %g 的判据比逐位相等**宽**：差 1 ULP 也能达标。" % rel_tol)
    print("           以 mismatches == 0 为准；只达标不逐位相等会报 [PASS*]。")

    report_lsb(got, want, r, mismatches, y_dims, shift2)
    report_ratio(got, want, r["n"])
    if "y_exact" in tensors:
        exact = decode_f16(tensors["y_exact"][2])
        report_vs_exact(got, exact, r["n"])
        # 拿真值反推定标。和 y_expect 比没意义 —— 那是按（可能错的）模型算出来的。
        probe_acc_scale(got, exact, r["n"], shift2, y_dims)
        print_sample(got, want, exact, r["n"], y_dims)
    if probe in (3, 4):
        report_idmap(got, y_dims, probe)

    # 设备原始输出落盘。默认**关**：板子那台机器上的文件多半传不出来，存了也是垃圾。
    # 真要离线分析再 FC2D_DUMP=<路径> 打开。分析所需的结论上面几段已经在设备上算完了。
    dump_path = os.environ.get("FC2D_DUMP", "")
    if dump_path and dump_path != "-":
        dump_raw(dump_path, out.raw[:y_bytes], "设备输出 (fp16 NCHW [1,%d,%d,%d])"
                 % (y_dims[1], y_dims[2], y_dims[3]))

    acl.aclopDestroyAttr(attr)
    for i in range(NIN + 1):
        acl.aclDestroyDataBuffer(bufs[i])
        acl.aclDestroyTensorDesc(descs[i])
        acl.aclrtFree(dev_ptrs[i])
    acl.aclrtDestroyStream(stream)
    acl.aclrtResetDevice(device_id)
    acl.aclFinalize()

    print()
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
