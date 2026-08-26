#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""通用单算子上板执行器 —— 任意算子、任意输入 bin，目标机不需要编译器。

    python3 run_op.py <case.json> [device_id]

用 ctypes 直接调 libascendcl.so 的 aclopExecuteV2。输入从裸 .bin 读，输出写回 .bin。
golden 是**可选**的：不给就只跑不比，仍然会报"输出到底被写过没有"。

case.json:

    {
      "op": "Conv2D",
      "om_dir": "om_out",                     // 可选。有 .om 就传，否则走本机算子信息库
      "device": 0,                            // 可选，命令行第 2 个参数优先
      "inputs": [
        {"file": "x.bin",      "dtype": "float16", "shape": [1,3,224,224], "format": "NCHW"},
        {"file": "filter.bin", "dtype": "float16", "shape": [64,3,7,7],    "format": "NCHW"},
        {"file": "bias.bin",   "dtype": "float32", "shape": [64],          "format": "ND"}
      ],
      "outputs": [
        {"dtype": "float16", "shape": [1,64,112,112], "format": "NCHW",
         "file": "y_out.bin",                 // 可选，输出落盘到哪
         "golden": "y_golden.bin"}            // 可选，给了才比精度
      ],
      "attrs": [                              // 可选，但走 .om 时**必须和编 .om 时一致**
        {"name": "strides",     "type": "list_int", "value": [1,1,2,2]},
        {"name": "pads",        "type": "list_int", "value": [3,3,3,3]},
        {"name": "dilations",   "type": "list_int", "value": [1,1,1,1]},
        {"name": "groups",      "type": "int",      "value": 1},
        {"name": "data_format", "type": "string",   "value": "NCHW"}
      ]
    }

dtype / format 也可以直接写 ACL 的**枚举整数**（比如 "dtype": 27），万一下面的名字表
在你这个 CANN 版本上对不上，不至于卡住。

环境变量：
    REPEAT=1         计时的下发次数。想避开冷启动就调大，比如 REPEAT=10
    WARMUP=0         正式计时之前先空跑多少次（不计入统计）。总下发次数 = WARMUP + REPEAT
    REL_TOL=1e-3     有 golden 时的相对误差判据
    RATIO_MIN=1.0    达标比例下限，低于它算 FAIL
"""

import ctypes
import json
import os
import statistics
import struct
import sys
import time

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

# 输出缓冲的哨兵。kernel 一个字节都没写的话，读回来还是它 —— 这是"返回码 0 但什么
# 都没算"唯一能被抓住的地方，没有 golden 时更是唯一的信号。
SENTINEL = 0xA5

# aclDataType: (枚举值, 每元素字节数, struct 码)。struct 码为 None 表示不做数值解包，
# 只能按字节比。
DTYPES = {
    "float":     (0,  4, "f"),
    "float32":   (0,  4, "f"),
    "float16":   (1,  2, "e"),
    "half":      (1,  2, "e"),
    "int8":      (2,  1, "b"),
    "int32":     (3,  4, "i"),
    "uint8":     (4,  1, "B"),
    "int16":     (6,  2, "h"),
    "uint16":    (7,  2, "H"),
    "uint32":    (8,  4, "I"),
    "int64":     (9,  8, "q"),
    "uint64":    (10, 8, "Q"),
    "double":    (11, 8, "d"),
    "float64":   (11, 8, "d"),
    "bool":      (12, 1, "B"),
    "bfloat16":  (27, 2, "bf16"),
    "bf16":      (27, 2, "bf16"),
}

FORMATS = {
    "UNDEFINED": -1, "NCHW": 0, "NHWC": 1, "ND": 2, "NC1HWC0": 3, "FRACTAL_Z": 4,
    "NC1HWC0_C04": 12, "HWCN": 16, "NDHWC": 27, "FRACTAL_NZ": 29, "NCDHW": 30,
    "NDC1HWC0": 32, "FRACTAL_Z_3D": 33,
}


ATTR_TYPES = {
    "bool", "int", "int64", "float", "float32", "string", "str",
    "dtype", "data_type", "datatype",
    "list_bool", "listbool", "list_int", "listint", "list_int64",
    "list_float", "listfloat", "list_string", "liststring",
    "list_list_int", "listlistint",
}
ATTR_TYPES_HELP = ("支持: bool int float string dtype list_bool list_int "
                   "list_float list_string list_list_int")


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


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


def check_attrs(attrs):
    """在碰设备之前就把 attr 描述本身校验掉。"""
    for j, a in enumerate(attrs):
        where = "attrs[%d]" % j
        if not isinstance(a, dict):
            die("%s 不是对象" % where)
        for k in ("name", "type", "value"):
            if k not in a:
                die("%s 少了 %r" % (where, k))
        t = str(a["type"]).lower()
        if t not in ATTR_TYPES:
            die("%s: 不支持的 attr 类型 %r。%s" % (where, a["type"], ATTR_TYPES_HELP))
        if t.startswith("list") and not isinstance(a["value"], (list, tuple)):
            die("%s: type=%s 但 value 不是数组" % (where, a["type"]))


def dtype_of(spec, where):
    """返回 (enum, itemsize, structcode)。允许直接给枚举整数。"""
    d = spec.get("dtype")
    if d is None:
        die("%s 少了 dtype" % where)
    if isinstance(d, int):
        for _, (e, sz, sc) in DTYPES.items():
            if e == d:
                return e, sz, sc
        die("%s 的 dtype=%d 不在已知表里，请改用名字，或者告诉我它每元素几字节" % (where, d))
    key = str(d).lower()
    if key not in DTYPES:
        die("%s 的 dtype=%r 不认识。已知: %s（也可以直接写 ACL 枚举整数）"
            % (where, d, ", ".join(sorted(set(DTYPES)))))
    return DTYPES[key]


def format_of(spec, where):
    f = spec.get("format", "ND")
    if isinstance(f, int):
        return f
    key = str(f).upper()
    if key not in FORMATS:
        die("%s 的 format=%r 不认识。已知: %s（也可以直接写枚举整数）"
            % (where, f, ", ".join(sorted(FORMATS))))
    return FORMATS[key]


def nelems(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def load_case(path):
    if not os.path.isfile(path):
        die("找不到 %s" % path)
    try:
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
    except ValueError as e:
        die("%s 不是合法 JSON: %s" % (path, e))
    if not isinstance(case, dict):
        die("case.json 顶层要是一个对象 {...}")
    for k in ("op", "inputs", "outputs"):
        if k not in case:
            die("case.json 少了字段 %r" % k)
    if not case["inputs"]:
        die("inputs 是空的")
    if not case["outputs"]:
        die("outputs 是空的")
    return case


def read_input(spec, i, base):
    where = "inputs[%d]" % i
    _, isz, _ = dtype_of(spec, where)
    shape = spec.get("shape")
    if shape is None:
        die("%s 少了 shape" % where)
    want = nelems(shape) * isz
    fn = spec.get("file")
    if fn is None:
        die("%s 少了 file" % where)
    path = fn if os.path.isabs(fn) else os.path.join(base, fn)
    if not os.path.isfile(path):
        die("%s 的文件 %s 不存在" % (where, path))
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != want:
        die("%s: %s 是 %d 字节，但 shape=%s dtype=%s 需要 %d 字节。\n"
            "    shape 或 dtype 写错了 —— 这是最常见的一类错，先改这里。\n"
            "    (差 %+d 字节)"
            % (where, path, len(data), shape, spec.get("dtype"), want, len(data) - want))
    return data, path


# ---------------------------------------------------------------- 数值解包 / 比对
def unpack(data, code, n):
    if code == "bf16":
        # bfloat16 就是 fp32 的高 16 位。左移 16 位补零即可无损展开成 fp32。
        u = struct.unpack("<%dH" % n, data)
        return struct.unpack("<%df" % n, struct.pack("<%dI" % n, *[x << 16 for x in u]))
    return struct.unpack("<%d%s" % (n, code), data)


def compare(got, want, code, n, rel_tol):
    """返回 (prec_ok, max_rel, max_idx, exact, hist)。code 为 None 时退化成按字节比。"""
    buckets = ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]
    hist = dict.fromkeys(buckets, 0)
    if code is None:
        exact = sum(1 for a, b in zip(got, want) if a == b)
        hist["=0"] = exact
        hist[">1"] = len(got) - exact
        return exact, (0.0 if exact == len(got) else float("inf")), -1, exact, hist

    g = unpack(got, code, n)
    w = unpack(want, code, n)
    ok = exact = 0
    mx, mi = 0.0, -1
    for i in range(n):
        a, b = g[i], w[i]
        if a == b:
            exact += 1
            err = 0.0
        elif b == 0:
            err = float("inf")
        else:
            err = abs(a - b) / float(abs(b))
        if err <= rel_tol:
            ok += 1
        if err > mx:
            mx, mi = err, i
        if err == 0.0:
            hist["=0"] += 1
        elif err == float("inf"):
            hist["inf(golden=0)"] += 1
        elif err <= 1e-3:
            hist["(0,1e-3]"] += 1
        elif err <= 1e-2:
            hist["(1e-3,1e-2]"] += 1
        elif err <= 1e-1:
            hist["(1e-2,1e-1]"] += 1
        elif err <= 1.0:
            hist["(1e-1,1]"] += 1
        else:
            hist[">1"] += 1
    return ok, mx, mi, exact, hist


# ---------------------------------------------------------------- ACL
def load_acl():
    try:
        acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        die("加载 libascendcl.so 失败：%s\n    先 source <CANN安装路径>/set_env.sh" % e)

    vp, i32, i, sz, cp = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_int,
                          ctypes.c_size_t, ctypes.c_char_p)
    req = [
        ("aclInit", [cp], i), ("aclFinalize", [], i),
        ("aclrtSetDevice", [i32], i), ("aclrtResetDevice", [i32], i),
        ("aclrtCreateStream", [ctypes.POINTER(vp)], i), ("aclrtDestroyStream", [vp], i),
        ("aclrtSynchronizeStream", [vp], i),
        ("aclrtMalloc", [ctypes.POINTER(vp), sz, i], i), ("aclrtFree", [vp], i),
        ("aclrtMemcpy", [vp, sz, vp, sz, i], i),
        ("aclCreateTensorDesc", [i, i, ctypes.POINTER(ctypes.c_int64), i], vp),
        ("aclDestroyTensorDesc", [vp], None),
        ("aclCreateDataBuffer", [vp, sz], vp), ("aclDestroyDataBuffer", [vp], i),
        ("aclopCreateAttr", [], vp), ("aclopDestroyAttr", [vp], None),
        ("aclopExecuteV2", [cp, i, ctypes.POINTER(vp), ctypes.POINTER(vp),
                            i, ctypes.POINTER(vp), ctypes.POINTER(vp), vp, vp], i),
    ]
    opt = [
        ("aclopSetModelDir", [cp], i),
        ("aclopSetAttrBool", [vp, cp, ctypes.c_ubyte], i),
        ("aclopSetAttrInt", [vp, cp, ctypes.c_int64], i),
        ("aclopSetAttrFloat", [vp, cp, ctypes.c_float], i),
        ("aclopSetAttrString", [vp, cp, cp], i),
        ("aclopSetAttrDataType", [vp, cp, i], i),
        ("aclopSetAttrListBool", [vp, cp, i, ctypes.POINTER(ctypes.c_ubyte)], i),
        ("aclopSetAttrListInt", [vp, cp, i, ctypes.POINTER(ctypes.c_int64)], i),
        ("aclopSetAttrListFloat", [vp, cp, i, ctypes.POINTER(ctypes.c_float)], i),
        ("aclopSetAttrListString", [vp, cp, i, ctypes.POINTER(cp)], i),
        ("aclopSetAttrListListInt", [vp, cp, i, ctypes.POINTER(i),
                                     ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))], i),
    ]
    for name, argtypes, restype in req:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            die("libascendcl.so 里没有 %s —— CANN 版本太老？" % name)
        fn.argtypes, fn.restype = argtypes, restype
    for name, argtypes, restype in opt:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            continue
        fn.argtypes, fn.restype = argtypes, restype
    return acl


def set_attrs(acl, attr, attrs):
    """按 case.json 里的 attrs 填 aclopAttr。类型名沿用 ATC singleop json 的写法。"""
    keep_inner = []
    for j, a in enumerate(attrs):
        where = "attrs[%d]" % j
        for k in ("name", "type", "value"):
            if k not in a:
                die("%s 少了 %r" % (where, k))
        name, t, v = a["name"].encode(), str(a["type"]).lower(), a["value"]

        def need(fn):
            f = getattr(acl, fn, None)
            if f is None:
                die("%s: libascendcl.so 里没有 %s" % (where, fn))
            return f

        if t in ("bool",):
            rc = need("aclopSetAttrBool")(attr, name, 1 if v else 0)
        elif t in ("int", "int64"):
            rc = need("aclopSetAttrInt")(attr, name, int(v))
        elif t in ("float", "float32"):
            rc = need("aclopSetAttrFloat")(attr, name, float(v))
        elif t in ("string", "str"):
            rc = need("aclopSetAttrString")(attr, name, str(v).encode())
        elif t in ("dtype", "data_type", "datatype"):
            e = v if isinstance(v, int) else dtype_of({"dtype": v}, where)[0]
            rc = need("aclopSetAttrDataType")(attr, name, int(e))
        elif t in ("list_bool", "listbool"):
            arr = (ctypes.c_ubyte * len(v))(*[1 if x else 0 for x in v])
            rc = need("aclopSetAttrListBool")(attr, name, len(v), arr)
        elif t in ("list_int", "listint", "list_int64"):
            arr = (ctypes.c_int64 * len(v))(*[int(x) for x in v])
            rc = need("aclopSetAttrListInt")(attr, name, len(v), arr)
        elif t in ("list_float", "listfloat"):
            arr = (ctypes.c_float * len(v))(*[float(x) for x in v])
            rc = need("aclopSetAttrListFloat")(attr, name, len(v), arr)
        elif t in ("list_list_int", "listlistint"):
            # aclopSetAttrListListInt(attr, name, numLists, const int *numValues,
            #                         const int64_t *const values[])
            inner = [(ctypes.c_int64 * len(x))(*[int(y) for y in x]) for x in v]
            keep_inner.append(inner)  # 挡住 GC，指针数组还指着它们
            cnts = (ctypes.c_int * len(v))(*[len(x) for x in v])
            ptrs = (ctypes.POINTER(ctypes.c_int64) * len(v))(
                *[ctypes.cast(a, ctypes.POINTER(ctypes.c_int64)) for a in inner])
            rc = need("aclopSetAttrListListInt")(attr, name, len(v), cnts, ptrs)
        elif t in ("list_string", "liststring"):
            bs = [str(x).encode() for x in v]
            arr = (ctypes.c_char_p * len(bs))(*bs)
            rc = need("aclopSetAttrListString")(attr, name, len(bs), arr)
        else:
            die("%s: 不支持的 attr 类型 %r。%s" % (where, a["type"], ATTR_TYPES_HELP))
        if rc != ACL_SUCCESS:
            die("%s: 设置 attr %r 失败, ret=%d" % (where, a["name"], rc))
        print("    %-16s %-12s %s" % (a["name"], a["type"], v))


# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    case_path = sys.argv[1]
    case = load_case(case_path)
    base = os.path.dirname(os.path.abspath(case_path))

    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else int(case.get("device", 0))
    om_dir = case.get("om_dir")

    def env_num(name, default, cast):
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        try:
            return cast(v)
        except ValueError:
            die("环境变量 %s=%r 不是合法数字" % (name, v))

    repeat = env_num("REPEAT", 1, int)
    warmup = env_num("WARMUP", 0, int)
    if repeat < 1:
        die("REPEAT 至少是 1")
    if warmup < 0:
        die("WARMUP 不能为负")
    rel_tol = env_num("REL_TOL", 1e-3, float)
    ratio_min = env_num("RATIO_MIN", 1.0, float)

    op = case["op"]
    print("单算子上板执行 —— ctypes + aclopExecuteV2")
    print('  case = %s   op = "%s"   device = %d' % (case_path, op, device_id))
    print("  om_dir = %s" % (om_dir if om_dir else "(不用离线模型，走本机算子信息库)"))
    print("  repeat = %d   warmup = %d   REL_TOL = %g   RATIO_MIN = %g\n"
          % (repeat, warmup, rel_tol, ratio_min))

    if om_dir:
        omp = om_dir if os.path.isabs(om_dir) else os.path.join(base, om_dir)
        if not os.path.isdir(omp):
            die("om_dir %s 不是目录" % omp)
        oms = [f for f in os.listdir(omp) if f.endswith(".om")]
        if not oms:
            die("%s 下没有 .om" % omp)
        print("  离线模型: %s" % ", ".join(sorted(oms)))

    check_attrs(case.get("attrs") or [])

    # ---- 读输入 ----
    print("\n==== 输入 ====")
    ins = []
    for i, spec in enumerate(case["inputs"]):
        data, path = read_input(spec, i, base)
        e, isz, _ = dtype_of(spec, "inputs[%d]" % i)
        ins.append((spec, data, e, isz))
        print("  [%d] %-24s %-9s %-22s %10d 字节  OK"
              % (i, os.path.basename(path), spec.get("dtype"), spec["shape"], len(data)))

    print("\n==== 输出 ====")
    outs = []
    for i, spec in enumerate(case["outputs"]):
        where = "outputs[%d]" % i
        if "shape" not in spec:
            die("%s 少了 shape" % where)
        e, isz, code = dtype_of(spec, where)
        nb = nelems(spec["shape"]) * isz
        gold = None
        if spec.get("golden"):
            gp = spec["golden"] if os.path.isabs(spec["golden"]) else os.path.join(base, spec["golden"])
            if not os.path.isfile(gp):
                die("%s 的 golden %s 不存在" % (where, gp))
            with open(gp, "rb") as f:
                gold = f.read()
            if len(gold) != nb:
                die("%s 的 golden 是 %d 字节，按 shape/dtype 应为 %d" % (where, len(gold), nb))
        outs.append((spec, e, isz, code, nb, gold))
        print("  [%d] %-9s %-22s %10d 字节  golden=%s"
              % (i, spec.get("dtype"), spec["shape"], nb,
                 os.path.basename(spec["golden"]) if gold else "(无，只跑不比)"))

    acl = load_acl()

    def check(rc, msg):
        if rc != ACL_SUCCESS:
            die(msg % rc if "%d" in msg else "%s (ret=%d)" % (msg, rc))

    check(acl.aclInit(None), "aclInit = %d")
    if om_dir:
        f = getattr(acl, "aclopSetModelDir", None)
        if f is None:
            die("libascendcl.so 里没有 aclopSetModelDir，用不了离线模型")
        check(f(os.path.abspath(omp).encode()), "aclopSetModelDir = %d")
        print("\n  aclopSetModelDir OK")
    check(acl.aclrtSetDevice(device_id), "aclrtSetDevice(" + str(device_id) + ") = %d —— 芯片被占？")
    stream = ctypes.c_void_p()
    check(acl.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream = %d")

    keep, devs, descs, bufs = [], [], [], []

    def make(dt, fmt, dims, data, nb):
        dev = ctypes.c_void_p()
        check(acl.aclrtMalloc(ctypes.byref(dev), nb, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(%d) 失败, ret = " % nb + "%d")
        host = ctypes.create_string_buffer(data, nb)
        keep.append(host)
        check(acl.aclrtMemcpy(dev, nb, ctypes.cast(host, ctypes.c_void_p), nb,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D = %d")
        arr = (ctypes.c_int64 * len(dims))(*[int(d) for d in dims]) if dims else None
        desc = acl.aclCreateTensorDesc(dt, len(dims), arr, fmt)
        if not desc:
            die("aclCreateTensorDesc 返回 null")
        buf = acl.aclCreateDataBuffer(dev, nb)
        if not buf:
            die("aclCreateDataBuffer 返回 null")
        devs.append(dev)
        descs.append(desc)
        bufs.append(buf)

    for spec, data, e, _ in ins:
        make(e, format_of(spec, "input"), spec["shape"], data, len(data))
    n_in = len(ins)
    for spec, e, _, _, nb, _ in outs:
        make(e, format_of(spec, "output"), spec["shape"], bytes([SENTINEL]) * nb, nb)
    n_out = len(outs)

    in_desc = (ctypes.c_void_p * n_in)(*descs[:n_in])
    in_buf = (ctypes.c_void_p * n_in)(*bufs[:n_in])
    out_desc = (ctypes.c_void_p * n_out)(*descs[n_in:])
    out_buf = (ctypes.c_void_p * n_out)(*bufs[n_in:])

    attr = acl.aclopCreateAttr()
    if not attr:
        die("aclopCreateAttr 返回 null")
    attrs = case.get("attrs") or []
    if attrs:
        print("\n==== attrs ====")
        set_attrs(acl, attr, attrs)
    else:
        print("\n==== attrs ==== (无)")
        print("  注意: 走 .om 时 attrs 必须和编 .om 时**完全一致**，否则匹配不上，")
        print("        报的是算子没找到(161001)，很容易误判成算子没编进去。")

    def launch():
        return acl.aclopExecuteV2(op.encode(), n_in, in_desc, in_buf,
                                  n_out, out_desc, out_buf, attr, stream)

    print("\n==== 执行 ====")
    # 总共下发 warmup + repeat 次；只有后 repeat 次计入统计。
    # 第 1 次单独给一段详细报错 —— 161001 有三种成因，分不清会查很久。
    durs = []
    for k in range(warmup + repeat):
        t0 = time.perf_counter()
        rc = launch()
        if rc != ACL_SUCCESS:
            if k == 0:
                die('aclopExecuteV2("%s") = %d\n'
                    "        161001 = 没匹配上。三种可能，按可能性排：\n"
                    "          1) attrs 和编 .om 时不一致（走离线模型时最常见）\n"
                    "          2) shape/dtype/format 和编 .om 时不一致\n"
                    "          3) 算子类型名拼错，或本机算子信息库里确实没有\n"
                    "        查: grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/"
                    % (op, rc, op))
            die("第 %d 次 aclopExecuteV2 = %d" % (k + 1, rc))
        rc = acl.aclrtSynchronizeStream(stream)
        if rc != ACL_SUCCESS:
            die("第 %d 次 aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志"
                % (k + 1, rc))
        dt = (time.perf_counter() - t0) * 1e6
        if k >= warmup:
            durs.append(dt)
    print("  aclopExecuteV2 + 同步 x %d OK（其中 %d 次预热不计入统计）"
          % (warmup + repeat, warmup))

    if repeat > 1 or warmup:
        print("\n==== 耗时（host 侧墙钟，含下发和同步；设备侧以 msprof 为准）====")
        report_times(durs)

    # ---- 取回 ----
    rcode = 0
    for i, (spec, e, isz, code, nb, gold) in enumerate(outs):
        out = ctypes.create_string_buffer(nb)
        check(acl.aclrtMemcpy(ctypes.cast(out, ctypes.c_void_p), nb, devs[n_in + i], nb,
                              ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
        got = out.raw[:nb]

        n = nelems(spec["shape"])
        sent = got.count(SENTINEL)
        zero = got.count(0)
        print("\n==== 输出[%d] ====" % i)
        print("  元素 %d，字节 %d" % (n, nb))
        print("  还是哨兵(0x%02X)的字节: %d / %d   全 0 的字节: %d"
              % (SENTINEL, sent, nb, zero))
        if sent == nb:
            print("  [FAIL] 整个缓冲一个字节都没被改写 —— kernel 没跑")
            rcode = 1
        elif sent * 10 > nb:
            print("  [!] 超过 10%% 的字节还是哨兵 —— 可能只写了一部分")

        fn = spec.get("file")
        if fn:
            fp = fn if os.path.isabs(fn) else os.path.join(base, fn)
            with open(fp, "wb") as f:
                f.write(got)
            print("  已写出 %s" % fp)

        if gold is None:
            continue
        ok, mx, mi, exact, hist = compare(got, gold, code, n, rel_tol)
        ratio = ok / float(n)
        print("  [精度] 逐位相等 %d / %d" % (exact, n))
        print("         相对误差 <= %g 的比例: %d / %d = %.6f%%   (下限 %.6f%%)"
              % (rel_tol, ok, n, ratio * 100.0, ratio_min * 100.0))
        if mi >= 0:
            print("         最大相对误差 %s @ %d"
                  % ("inf" if mx == float("inf") else "%.6e" % mx, mi))
        for k in ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]:
            if hist[k]:
                print("           %-14s %9d  (%.6f%%)" % (k, hist[k], hist[k] * 100.0 / n))
        if ratio < ratio_min:
            print("  [FAIL] 达标比例 %.6f%% < 下限 %.6f%%" % (ratio * 100.0, ratio_min * 100.0))
            rcode = 1

    acl.aclopDestroyAttr(attr)
    for i in range(len(descs)):
        acl.aclDestroyDataBuffer(bufs[i])
        acl.aclDestroyTensorDesc(descs[i])
        acl.aclrtFree(devs[i])
    acl.aclrtDestroyStream(stream)
    acl.aclrtResetDevice(device_id)
    acl.aclFinalize()

    print()
    print("[FAIL] 见上" if rcode else "[PASS] 执行完成")
    return rcode


if __name__ == "__main__":
    sys.exit(main())
