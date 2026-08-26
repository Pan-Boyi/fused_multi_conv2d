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

    python3 run_fused_conv2d.py [case.bin] [op_type] [device_id] [om_dir]
    默认  fused_conv2d_case.bin  FusedConv2d  0  (无 om_dir)

环境变量：
    REL_TOL=1e-3     相对误差判据（默认 1e-3）
    RATIO_MIN=1.0    达标比例的下限，低于它算 FAIL（默认 1.0，即要求 100%）
    REPEAT=1         计时的下发次数。想避开冷启动就调大，比如 REPEAT=10
    WARMUP=0         正式计时前先空跑多少次（不计入统计）。总次数 = WARMUP + REPEAT

设备上没装带 FusedConv2d 的算子包时，传第 4 个参数 om_dir：那是在有算子包的机器上
用 build_om.sh 编出来的单算子离线模型目录。脚本会先 aclopSetModelDir(om_dir)，
运行时就从那里找算子，不再要求本机的算子信息库里有它。

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

DTYPE_NAME = {2: "int8", 3: "int32", 10: "uint64"}
DTYPE_SIZE = {2: 1, 3: 4, 10: 8}

Y_SENTINEL = 127

HDR_FMT = "<8sIIQQQQQ"          # magic, version, ntensors, nonzero, ties, sat, y_elems, reserved
HDR_LEN = struct.calcsize(HDR_FMT)
REC_FMT = "<16sII4qQ"           # name, dtype, ndim, dims[4], nbytes
REC_LEN = struct.calcsize(REC_FMT)

ORDER = ["x", "filter1", "bias1", "scale1", "filter2", "bias2", "scale2"]


def die(msg):
    print("\n[X] %s" % msg)
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
        ("aclopSetModelDir", [c_cp], c_i),
        ("aclopCreateAttr", [], c_vp),
        ("aclopDestroyAttr", [c_vp], None),
        ("aclopExecuteV2", [c_cp, c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp),
                            c_i, ctypes.POINTER(c_vp), ctypes.POINTER(c_vp), c_vp, c_vp], c_i),
    ]
    for name, argtypes, restype in sig:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            die("libascendcl.so 里没有 %s —— CANN 版本太老？" % name)
        fn.argtypes = argtypes
        fn.restype = restype
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


def as_i8(b):
    """.bin 和 memcpy 回来的都是无符号字节；判据全按 int8 解释。"""
    return b - 256 if b > 127 else b


def evaluate(got, want, rel_tol):
    """一趟扫完所有判据。返回 dict。

    三个"写没写对"的判据，各挡一类失败，缺一不可：
      unwritten  kernel 一个字节都没写（哨兵 127 还在）
      nonzero    kernel 只写了一部分（前一半对、后一半空白）
      mismatches 数值错

    外加精度判据：逐元素相对误差 <= rel_tol 的比例。
      golden 非 0: |got-want| / |want|
      golden 为 0: 退化成绝对误差 |got-want|（int8 下即要求 got 也为 0）
    """
    n = len(got)
    buckets = ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]
    r = {
        "n": n,
        "nonzero": n - got.count(0),
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
    if got == want:
        # 完全一致：不用逐元素算，相对误差全是 0
        r["hist"]["=0"] = n
        r["zero_golden"] = want.count(0)
        return r

    zero_golden = 0
    prec_ok = 0
    for i, (g8, w8) in enumerate(zip(got, want)):
        g, w = as_i8(g8), as_i8(w8)
        if g8 == Y_SENTINEL and w8 != Y_SENTINEL:
            r["unwritten"] += 1
        if g8 != w8:
            if r["first_bad"] < 0:
                r["first_bad"] = i
            r["mismatches"] += 1
            if abs(g - w) == 1:
                r["off_by_one"] += 1
        if w == 0:
            zero_golden += 1
            err = 0.0 if g == 0 else float("inf")
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
    om_dir = sys.argv[4] if len(sys.argv) > 4 else None

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
    print('  om_dir = %s' % (om_dir if om_dir else "(不用离线模型，走本机算子信息库)"))
    print('  REL_TOL = %g   RATIO_MIN = %g   repeat = %d   warmup = %d\n'
          % (rel_tol, ratio_min, repeat, warmup))

    if om_dir is not None:
        if not os.path.isdir(om_dir):
            die("om_dir %s 不是目录" % om_dir)
        oms = [f for f in os.listdir(om_dir) if f.endswith(".om")]
        if not oms:
            die("%s 下没有 .om —— 在有算子包的机器上跑 build_om.sh 生成，整个目录拷过来" % om_dir)
        print("  离线模型: %s" % ", ".join(sorted(oms)))

    tensors, gold_nonzero, ties, sat, y_elems = load_case(case_path)
    want = tensors["y_expect"][2]
    if len(want) != y_elems:
        die("golden 输出 %d 字节，头里写的是 %d" % (len(want), y_elems))
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
    if om_dir is not None:
        # 必须在 aclrtSetDevice 之前/之后都可以，但要在 aclopExecuteV2 之前。
        check(acl.aclopSetModelDir(os.path.abspath(om_dir).encode()),
              "aclopSetModelDir(%s) = " % om_dir + "%d")
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
    make_operand(y_dtype, y_dims, bytes([Y_SENTINEL]) * y_elems)

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
                    "                      跑 build_om.sh 编出 .om，把目录拷过来当第 4 个参数传进来\n"
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

    out = ctypes.create_string_buffer(y_elems)
    check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), y_elems, dev_ptrs[7], y_elems,
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
    got = out.raw[:y_elems]

    # ------------------------------------------------------------ 比对
    r = evaluate(got, want, rel_tol)
    nonzero, unwritten, mismatches = r["nonzero"], r["unwritten"], r["mismatches"]
    ratio = r["prec_ok"] / float(r["n"])

    print("\n[fused-conv2d-5102-device] out_elems=%d nonzero=%d golden_nonzero=%d "
          "unwritten=%d mismatches=%d" % (y_elems, nonzero, gold_nonzero, unwritten, mismatches))

    cout2 = y_dims[1]
    if mismatches:
        fb = r["first_bad"]
        print("first mismatch @ %d: got %d, want %d (row %d, cout %d)；差 ±1 的 %d / 共 %d"
              % (fb, as_i8(got[fb]), as_i8(want[fb]), fb // cout2, fb % cout2,
                 r["off_by_one"], mismatches))

    # ------------------------------------------------------------ 精度
    print("\n[精度] 相对误差 <= %g 的比例: %d / %d = %.6f%%   (下限 %.6f%%)"
          % (rel_tol, r["prec_ok"], r["n"], ratio * 100.0, ratio_min * 100.0))
    mr = r["max_rel"]
    if r["max_rel_idx"] >= 0:
        mi = r["max_rel_idx"]
        print("       最大相对误差 %s @ %d (row %d, cout %d): got %d, want %d"
              % ("inf" if mr == float("inf") else "%.6e" % mr, mi, mi // cout2, mi % cout2,
                 as_i8(got[mi]), as_i8(want[mi])))
    else:
        print("       最大相对误差 0")
    print("       golden 为 0 的 %d 个元素按绝对误差判" % r["zero_golden"])
    print("       相对误差分布:")
    for k in ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]:
        v = r["hist"][k]
        if v:
            print("         %-14s %9d  (%.6f%%)" % (k, v, v * 100.0 / r["n"]))
    # 这个提醒值得常驻：int8 输出下 |golden| <= 100，最小的非零差值是 1，
    # 所以任何一个 LSB 的偏差相对误差都 >= 1/100 = 1e-2，早就越过 1e-3 了。
    # 也就是说在这个算子上，1e-3 的相对误差判据**等价于逐位相等**，
    # 达标比例只会是 100% 或者恰好等于逐位相等的比例。
    print("       注: 输出是 int8 且 |golden| <= 100，最小非零差值 1 对应相对误差 >= 1e-2，")
    print("           所以 %g 的判据在这里等价于逐位相等。" % rel_tol)

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
