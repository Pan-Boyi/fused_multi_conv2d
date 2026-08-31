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

DTYPE_NAME = {1: "float16", 2: "int8", 3: "int32", 10: "uint64"}
DTYPE_SIZE = {1: 2, 2: 1, 3: 4, 10: 8}

# y 改成 fp16（fixpipe 走 VDEQF16）之后，哨兵按字节填 0x7F，两字节拼起来是
# 0x7F7F —— 一个 fp16 NaN（阶码全 1、尾数非零）。golden 永远算不出 NaN，所以
# 「有没有被写过」这个判据不需要再论证合法输出取不到哨兵值。
Y_SENTINEL_BYTE = 0x7F
Y_SENTINEL_U16 = 0x7F7F
Y_ELEM_BYTES = 2

HDR_FMT = "<8sIIQQQQQ"          # magic, version, ntensors, nonzero, ties, sat, y_elems, reserved
HDR_LEN = struct.calcsize(HDR_FMT)
REC_FMT = "<16sII4qQ"           # name, dtype, ndim, dims[4], nbytes
REC_LEN = struct.calcsize(REC_FMT)

ORDER = ["x", "filter1", "bias1", "scale1", "filter2", "bias2", "scale2"]


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

    magic, version, ntensors, nonzero, ties, sat, y_elems, _ = struct.unpack_from(HDR_FMT, blob, 0)
    if magic != b"FC2DCASE":
        die("%s 不是 case 文件（magic = %r）" % (path, magic))
    if version != 1:
        die("case 文件版本 %d，本脚本只认 1 —— gen_case 和 run_fused_conv2d.py 得配套" % version)

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

    for name in ORDER + ["y_expect"]:
        if name not in tensors:
            die("case 文件里缺张量 %s" % name)

    return tensors, nonzero, ties, sat, y_elems


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

    tensors, gold_nonzero, ties, sat, y_elems = load_case(case_path)
    want_raw = tensors["y_expect"][2]
    if len(want_raw) != y_elems * Y_ELEM_BYTES:
        die("golden 输出 %d 字节，按 %d 个 fp16 元素应为 %d"
            % (len(want_raw), y_elems, y_elems * Y_ELEM_BYTES))
    want = decode_f16(want_raw)
    print("case 文件 OK：")
    for name in ORDER + ["y_expect"]:
        dtype, dims, data = tensors[name]
        print("  %-9s %-6s %-16s %9d 字节" % (name, DTYPE_NAME[dtype], dims, len(data)))
    print("golden: nonzero=%d/%d ties=%d sat=%d" % (gold_nonzero, y_elems, ties, sat))
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

    # ABI 顺序钉死：x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    for name in ORDER:
        dtype, dims, data = tensors[name]
        make_operand(dtype, dims, data)

    # 输出缓冲预填哨兵 127：kernel 一个字节都没写的话，读回来还是 127，
    # 这是"返回码 0 但什么都没算"唯一能被抓住的地方。
    y_dtype, y_dims, _ = tensors["y_expect"]
    make_operand(y_dtype, y_dims, bytes([Y_SENTINEL_BYTE]) * (y_elems * Y_ELEM_BYTES))

    in_desc = (ctypes.c_void_p * 7)(*descs[:7])
    in_buf = (ctypes.c_void_p * 7)(*bufs[:7])
    out_desc = (ctypes.c_void_p * 1)(descs[7])
    out_buf = (ctypes.c_void_p * 1)(bufs[7])

    # 本算子没有属性。传空 attr 而不是 null：有些版本按 attr 指针参与 kernel 选择的哈希。
    attr = acl.aclopCreateAttr()

    def launch():
        return acl.aclopExecuteV2(op_type.encode(), 7, in_desc, in_buf,
                                  1, out_desc, out_buf, attr, stream)

    # 总共下发 warmup + repeat 次；只有后 repeat 次计入统计。
    durs = []
    for k in range(warmup + repeat):
        t0 = time.perf_counter()
        ret = launch()
        if ret != ACL_SUCCESS:
            if k == 0:
                die('aclopExecuteV2("%s") = %d\n'
                    "        算子没找到(161001) -> 本机的算子信息库里没有它，或者类型名拼错\n"
                    "                      查: grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/\n"
                    "                      这台机器没装带 FusedConv2d 的算子包的话，先在有包的机器上\n"
                    "                      跑 build_om.sh 编出 .om，把它当第 4 个参数传进来\n"
                    "        找到但选不出 kernel -> shape/dtype 和 binary.json 里登记的组合对不上"
                    % (op_type, ret, op_type))
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
    check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), y_bytes, dev_ptrs[7], y_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
    got = decode_f16(out.raw[:y_bytes])

    # ------------------------------------------------------------ 比对
    r = evaluate(got, want, rel_tol)
    nonzero, unwritten, mismatches = r["nonzero"], r["unwritten"], r["mismatches"]
    ratio = r["prec_ok"] / float(r["n"])

    print("\n[fused-conv2d-5102-device] out_elems=%d nonzero=%d golden_nonzero=%d "
          "unwritten=%d mismatches=%d" % (y_elems, nonzero, gold_nonzero, unwritten, mismatches))

    cout2 = y_dims[1]
    if mismatches:
        fb = r["first_bad"]
        print("first mismatch @ %d: got %.6g (0x%04x), want %.6g (0x%04x) (row %d, cout %d)"
              "；差 1 个 ULP 的 %d / 共 %d"
              % (fb, got[0][fb], got[1][fb], want[0][fb], want[1][fb], fb // cout2, fb % cout2,
                 r["off_by_one"], mismatches))

    # ------------------------------------------------------------ 精度
    print("\n[精度] 相对误差 <= %g 的比例: %d / %d = %.6f%%   (下限 %.6f%%)"
          % (rel_tol, r["prec_ok"], r["n"], ratio * 100.0, ratio_min * 100.0))
    mr = r["max_rel"]
    if r["max_rel_idx"] >= 0:
        mi = r["max_rel_idx"]
        print("       最大相对误差 %s @ %d (row %d, cout %d): got %.6g, want %.6g"
              % ("inf" if mr == float("inf") else "%.6e" % mr, mi, mi // cout2, mi % cout2,
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

    acl.aclopDestroyAttr(attr)
    for i in range(8):
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
