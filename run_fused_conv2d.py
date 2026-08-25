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

设备上没装带 FusedConv2d 的算子包时，传第 4 个参数 om_dir：那是在有算子包的机器上
用 build_om.sh 编出来的单算子离线模型目录。脚本会先 aclopSetModelDir(om_dir)，
运行时就从那里找算子，不再要求本机的算子信息库里有它。

判据和 C++ 版完全一致，输出行可以逐字对比。
"""

import ctypes
import os
import struct
import sys

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
def as_i8(b):
    """.bin 和 memcpy 回来的都是无符号字节；判据全按 int8 解释。"""
    return b - 256 if b > 127 else b


def compare(got, want):
    """返回 (nonzero, unwritten, mismatches, first_bad, off_by_one)。

    三个判据各挡一类失败，缺一不可：
      unwritten  kernel 一个字节都没写（哨兵 127 还在）
      nonzero    kernel 只写了一部分（前一半对、后一半空白）
      mismatches 数值错
    """
    nonzero = len(got) - got.count(0)
    if got == want:
        return nonzero, 0, 0, -1, 0
    unwritten = mismatches = off_by_one = 0
    first_bad = -1
    for i, (g, w) in enumerate(zip(got, want)):
        if g == Y_SENTINEL and w != Y_SENTINEL:
            unwritten += 1
        if g != w:
            if first_bad < 0:
                first_bad = i
            mismatches += 1
            if abs(as_i8(g) - as_i8(w)) == 1:
                off_by_one += 1
    return nonzero, unwritten, mismatches, first_bad, off_by_one


# ---------------------------------------------------------------- main
def main():
    case_path = sys.argv[1] if len(sys.argv) > 1 else "fused_conv2d_case.bin"
    op_type = sys.argv[2] if len(sys.argv) > 2 else "FusedConv2d"
    device_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    om_dir = sys.argv[4] if len(sys.argv) > 4 else None

    print('FusedConv2d @ 5102 单算子验证 —— ctypes + aclopExecuteV2（目标机不需要编译器）')
    print('  case = "%s"   opType = "%s"   device = %d' % (case_path, op_type, device_id))
    print('  om_dir = %s\n' % (om_dir if om_dir else "(不用离线模型，走本机算子信息库)"))

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

    ret = acl.aclopExecuteV2(op_type.encode(), 7, in_desc, in_buf, 1, out_desc, out_buf, attr, stream)
    if ret != ACL_SUCCESS:
        die('aclopExecuteV2("%s") = %d\n'
            "        算子没找到(161001) -> 本机的算子信息库里没有它，或者类型名拼错\n"
            "                      查: grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/\n"
            "                      这台机器没装带 FusedConv2d 的算子包的话，先在有包的机器上\n"
            "                      跑 build_om.sh 编出 .om，把目录拷过来当第 4 个参数传进来\n"
            "        找到但选不出 kernel -> shape/dtype 和 binary.json 里登记的组合对不上"
            % (op_type, ret, op_type))
    check(acl.aclrtSynchronizeStream(stream),
          "aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志")

    out = ctypes.create_string_buffer(y_elems)
    check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), y_elems, dev_ptrs[7], y_elems,
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
    got = out.raw[:y_elems]

    # ------------------------------------------------------------ 比对
    nonzero, unwritten, mismatches, first_bad, off_by_one = compare(got, want)

    print("\n[fused-conv2d-5102-device] out_elems=%d nonzero=%d golden_nonzero=%d "
          "unwritten=%d mismatches=%d" % (y_elems, nonzero, gold_nonzero, unwritten, mismatches))

    if mismatches:
        cout2 = y_dims[1]
        print("first mismatch @ %d: got %d, want %d (row %d, cout %d)；差 ±1 的 %d / 共 %d"
              % (first_bad, as_i8(got[first_bad]), as_i8(want[first_bad]),
                 first_bad // cout2, first_bad % cout2, off_by_one, mismatches))

    acl.aclopDestroyAttr(attr)
    for i in range(8):
        acl.aclDestroyDataBuffer(bufs[i])
        acl.aclDestroyTensorDesc(descs[i])
        acl.aclrtFree(dev_ptrs[i])
    acl.aclrtDestroyStream(stream)
    acl.aclrtResetDevice(device_id)
    acl.aclFinalize()

    if unwritten:
        print("[FAIL] %d 个输出从来没被写过 —— kernel 没跑，或只跑了一部分核" % unwritten)
        return 1
    if nonzero != gold_nonzero:
        print("[FAIL] 非零个数 %d != golden 的 %d —— 写的分布不对" % (nonzero, gold_nonzero))
        return 1
    if mismatches:
        print("[FAIL] %d 个元素数值不一致" % mismatches)
        return 1
    print("[PASS] %d 个输出全部写过，且与 CPU golden 逐位相等" % y_elems)
    return 0


if __name__ == "__main__":
    sys.exit(main())
