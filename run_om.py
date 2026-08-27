#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""只有 .om 和输入 bin，什么描述文件都没有时，照样把它跑起来。

    python3 run_om.py <model.om>                        # 只看这个 om 要什么，不执行
    python3 run_om.py <model.om> in0.bin in1.bin ...    # 执行
    python3 run_om.py <model.om> in0.bin --golden0=g.bin

和 run_op.py 的区别:run_op.py 走 aclopExecuteV2，要你告诉它 shape/dtype/format/attr;
这个走 **aclmdlLoadFromFile + aclmdlExecute**，把 .om 当普通模型加载，**输入输出的
个数、字节数、dtype、format、dims 全都从模型自己身上问出来**。所以不需要 json，
也不需要知道 attr —— attr 在编 .om 的时候就固化进去了。

不带 bin 参数时只做一件事:把这个 om 要什么、给什么打印出来。先跑这个。

选项:
    --device=N        用哪颗芯片，默认 0
    --repeat=N        **总共**执行多少次，默认 1。想避开冷启动就调大，比如 --repeat=10
    --warmup=N        正式计时之前先空跑多少次（不计入统计），默认 0
    --out=前缀        输出落盘成 <前缀>N.bin，默认 om_out_
    --goldenN=文件    第 N 个输出的 golden，给了才比精度
    --no-write        不写输出文件

环境变量（和同名选项等价，选项优先）:
    REPEAT / WARMUP / REL_TOL / RATIO_MIN

关于冷启动:第 1 次执行含 kernel 加载和各种一次性开销，通常显著偏大。--repeat 会把
每次的耗时都记下来，单独报第 1 次、并给出"去掉第 1 次"之后的统计，冷启动多贵一眼可见。
真正的设备侧时间还是以 msprof 为准（./run_prof.sh）。
"""

import ctypes
import os
import statistics
import struct
import sys
import time

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
SENTINEL = 0xA5

# aclDataType -> (名字, 每元素字节数, struct 码)
DT = {
    0: ("float32", 4, "f"), 1: ("float16", 2, "e"), 2: ("int8", 1, "b"),
    3: ("int32", 4, "i"), 4: ("uint8", 1, "B"), 6: ("int16", 2, "h"),
    7: ("uint16", 2, "H"), 8: ("uint32", 4, "I"), 9: ("int64", 8, "q"),
    10: ("uint64", 8, "Q"), 11: ("double", 8, "d"), 12: ("bool", 1, "B"),
    13: ("string", 0, None), 16: ("complex64", 8, None), 17: ("complex128", 16, None),
    27: ("bfloat16", 2, "bf16"),
}
FMT = {
    -1: "UNDEFINED", 0: "NCHW", 1: "NHWC", 2: "ND", 3: "NC1HWC0", 4: "FRACTAL_Z",
    12: "NC1HWC0_C04", 16: "HWCN", 27: "NDHWC", 29: "FRACTAL_NZ", 30: "NCDHW",
    32: "NDC1HWC0", 33: "FRACTAL_Z_3D",
}


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


def unpack(data, code, n):
    if code == "bf16":
        u = struct.unpack("<%dH" % n, data)
        return struct.unpack("<%df" % n, struct.pack("<%dI" % n, *[x << 16 for x in u]))
    return struct.unpack("<%d%s" % (n, code), data)


def compare(got, want, code, n, rel_tol):
    buckets = ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]
    hist = dict.fromkeys(buckets, 0)
    if code is None or n == 0:
        exact = sum(1 for a, b in zip(got, want) if a == b)
        hist["=0"] = exact
        hist[">1"] = len(got) - exact
        return exact, (0.0 if exact == len(got) else float("inf")), -1, exact, hist, len(got)
    g, w = unpack(got, code, n), unpack(want, code, n)
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
    return ok, mx, mi, exact, hist, n


def load_acl():
    try:
        acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        die("加载 libascendcl.so 失败：%s\n    先 source <CANN安装路径>/set_env.sh" % e)
    vp, u32, i32, i, sz, cp = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32,
                               ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p)
    sig = [
        ("aclInit", [cp], i), ("aclFinalize", [], i),
        ("aclrtSetDevice", [i32], i), ("aclrtResetDevice", [i32], i),
        ("aclrtMalloc", [ctypes.POINTER(vp), sz, i], i), ("aclrtFree", [vp], i),
        ("aclrtMemcpy", [vp, sz, vp, sz, i], i),
        ("aclmdlLoadFromFile", [cp, ctypes.POINTER(u32)], i),
        ("aclmdlUnload", [u32], i),
        ("aclmdlCreateDesc", [], vp), ("aclmdlDestroyDesc", [vp], i),
        ("aclmdlGetDesc", [vp, u32], i),
        ("aclmdlGetNumInputs", [vp], sz), ("aclmdlGetNumOutputs", [vp], sz),
        ("aclmdlGetInputSizeByIndex", [vp, sz], sz),
        ("aclmdlGetOutputSizeByIndex", [vp, sz], sz),
        ("aclmdlCreateDataset", [], vp), ("aclmdlDestroyDataset", [vp], i),
        ("aclmdlAddDatasetBuffer", [vp, vp], i),
        ("aclCreateDataBuffer", [vp, sz], vp), ("aclDestroyDataBuffer", [vp], i),
        ("aclmdlExecute", [u32, vp, vp], i),
    ]
    # 这几个拿不到也不影响执行，只影响打印得好不好看
    opt = [
        ("aclGetRecentErrMsg", [], cp),
        ("aclmdlGetInputDataType", [vp, sz], i), ("aclmdlGetOutputDataType", [vp, sz], i),
        ("aclmdlGetInputFormat", [vp, sz], i), ("aclmdlGetOutputFormat", [vp, sz], i),
        ("aclmdlGetInputNameByIndex", [vp, sz], cp), ("aclmdlGetOutputNameByIndex", [vp, sz], cp),
        ("aclmdlGetInputDims", [vp, sz, vp], i), ("aclmdlGetOutputDims", [vp, sz, vp], i),
    ]
    for name, a, rt in sig:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            die("libascendcl.so 里没有 %s —— CANN 版本太老，或者这不是完整的 runtime" % name)
        fn.argtypes, fn.restype = a, rt
    for name, a, rt in opt:
        try:
            fn = getattr(acl, name)
        except AttributeError:
            continue
        fn.argtypes, fn.restype = a, rt
    global _ACL_HANDLE
    _ACL_HANDLE = acl
    return acl


def get_dims(acl, desc, idx, is_in, nbytes, itemsize):
    """best-effort 读 aclmdlIODims。

    这个结构体是 {char name[128]; size_t dimCount; int64_t dims[128];}。常量万一
    在某个版本上不一样，硬按偏移解就会读出垃圾 —— 所以给一个大缓冲，解出来之后用
    "各维乘积 x 每元素字节数 == 模型报的字节数" 去验。验不过就当没拿到，不影响执行:
    **执行只需要字节数**，dims 纯粹是打印给人看的。
    """
    fn = getattr(acl, "aclmdlGetInputDims" if is_in else "aclmdlGetOutputDims", None)
    if fn is None:
        return None
    buf = ctypes.create_string_buffer(8192)
    if fn(desc, idx, ctypes.cast(buf, ctypes.c_void_p)) != ACL_SUCCESS:
        return None
    raw = buf.raw
    for name_len in (128, 256, 64):
        off = name_len
        try:
            cnt = struct.unpack_from("<Q", raw, off)[0]
        except struct.error:
            continue
        if not (0 < cnt <= 64):
            continue
        try:
            dims = list(struct.unpack_from("<%dq" % cnt, raw, off + 8))
        except struct.error:
            continue
        if any(d <= 0 for d in dims):
            continue
        if itemsize and nbytes:
            p = 1
            for d in dims:
                p *= d
            if p * itemsize != nbytes:
                continue
        return dims
    return None


def describe(acl, desc, n, is_in):
    """返回 [(name, nbytes, dt, dtname, itemsize, code, dims)]"""
    out = []
    for i in range(n):
        nb = (acl.aclmdlGetInputSizeByIndex if is_in else acl.aclmdlGetOutputSizeByIndex)(desc, i)
        dt = -1
        f = getattr(acl, "aclmdlGetInputDataType" if is_in else "aclmdlGetOutputDataType", None)
        if f is not None:
            dt = f(desc, i)
        fmt = -1
        f = getattr(acl, "aclmdlGetInputFormat" if is_in else "aclmdlGetOutputFormat", None)
        if f is not None:
            fmt = f(desc, i)
        nm = b""
        f = getattr(acl, "aclmdlGetInputNameByIndex" if is_in else "aclmdlGetOutputNameByIndex", None)
        if f is not None:
            nm = f(desc, i) or b""
        dtname, isz, code = DT.get(dt, ("dtype=%d?" % dt, 0, None))
        dims = get_dims(acl, desc, i, is_in, nb, isz)
        out.append((nm.decode(errors="replace"), int(nb), dt, dtname, isz, code, fmt, dims))
    return out


def show(tag, items):
    print("  %s: %d 个" % (tag, len(items)))
    for i, (nm, nb, dt, dtname, isz, code, fmt, dims) in enumerate(items):
        elems = (nb // isz) if isz else 0
        print("    [%d] %-10s %-10s %-13s %10d 字节  %s  %s"
              % (i, dtname, FMT.get(fmt, "fmt=%d?" % fmt),
                 ("x".join(str(d) for d in dims) if dims else "dims<读不出>"),
                 nb, ("%d 元素" % elems) if elems else "", nm))


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    opts = {}
    pos = []
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v != "" else "1"
        else:
            pos.append(a)

    om = pos[0] if pos else None
    bins = pos[1:]
    if not om or not os.path.isfile(om):
        die("找不到 om 文件 %r" % om)
    device_id = int(opts.get("device", 0))
    out_prefix = opts.get("out", "om_out_")
    write_out = "no-write" not in opts

    def env_num(name, default, cast):
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        try:
            return cast(v)
        except ValueError:
            die("环境变量 %s=%r 不是合法数字" % (name, v))

    def opt_num(name, envname, default, cast):
        if name in opts:
            try:
                return cast(opts[name])
            except ValueError:
                die("--%s=%r 不是合法数字" % (name, opts[name]))
        return env_num(envname, default, cast)

    repeat = opt_num("repeat", "REPEAT", 1, int)
    warmup = opt_num("warmup", "WARMUP", 0, int)
    rel_tol = opt_num("rel-tol", "REL_TOL", 1e-3, float)
    ratio_min = opt_num("ratio-min", "RATIO_MIN", 1.0, float)
    if repeat < 1:
        die("--repeat 至少是 1（传的是 %d）" % repeat)
    if warmup < 0:
        die("--warmup 不能为负")

    print("直接跑 .om —— aclmdlLoadFromFile + aclmdlExecute（不需要任何描述文件）")
    print("  om     = %s  (%d 字节)" % (om, os.path.getsize(om)))
    print("  device = %d   repeat = %d   warmup = %d\n" % (device_id, repeat, warmup))

    acl = load_acl()

    def check(rc, msg):
        if rc != ACL_SUCCESS:
            die(msg % rc if "%d" in msg else "%s (ret=%d)" % (msg, rc))

    check(acl.aclInit(None), "aclInit = %d")
    check(acl.aclrtSetDevice(device_id),
          "aclrtSetDevice(" + str(device_id) + ") = %d —— 芯片被占？")

    model_id = ctypes.c_uint32()
    rc = acl.aclmdlLoadFromFile(os.path.abspath(om).encode(), ctypes.byref(model_id))
    if rc != ACL_SUCCESS:
        die("aclmdlLoadFromFile = %d\n"
            "        这个 .om 加载不了。可能是:\n"
            "          1) 它是 atc --singleop 产出的**单算子模型**，只能通过\n"
            "             aclopSetModelDir + aclopExecuteV2 用 -> 走 run_op.py，\n"
            "             但那条路需要知道 shape/dtype/format/attr。没有 json 的话，\n"
            "             在装了 CANN 的机器上把 om 反解成 json：\n"
            "                 atc --mode=1 --om=%s --json=%s.json\n"
            "             把那份 json 发我，我照着生成 case.json。\n"
            "          2) om 是给别的 soc 编的\n"
            "          3) om 的 CANN 版本和这台机器的运行时不兼容" % (rc, om, om))
    print("  aclmdlLoadFromFile OK, modelId = %d" % model_id.value)

    desc = acl.aclmdlCreateDesc()
    if not desc:
        die("aclmdlCreateDesc 返回 null")
    check(acl.aclmdlGetDesc(desc, model_id), "aclmdlGetDesc = %d")

    n_in = int(acl.aclmdlGetNumInputs(desc))
    n_out = int(acl.aclmdlGetNumOutputs(desc))
    ins = describe(acl, desc, n_in, True)
    outs = describe(acl, desc, n_out, False)

    print("\n==== 这个 om 的接口（从模型自己身上问出来的）====")
    show("输入", ins)
    show("输出", outs)

    if not bins:
        print("\n没给输入 bin，只做了描述。执行的话:")
        print("  python3 %s %s %s" % (os.path.basename(sys.argv[0]), om,
                                      " ".join("in%d.bin" % i for i in range(n_in))))
        print("  按上面的顺序和字节数准备文件即可。")
        acl.aclmdlDestroyDesc(desc)
        acl.aclmdlUnload(model_id)
        acl.aclrtResetDevice(device_id)
        acl.aclFinalize()
        return 0

    if len(bins) != n_in:
        die("这个模型要 %d 个输入，你给了 %d 个 bin。顺序要和上面一致。" % (n_in, len(bins)))

    # ---- 读输入并核对字节数 ----
    print("\n==== 输入 ====")
    data_in = []
    bad = False
    for i, (path, item) in enumerate(zip(bins, ins)):
        if not os.path.isfile(path):
            print("  [%d] %-28s ** 文件不存在 **" % (i, path))
            bad = True
            continue
        with open(path, "rb") as f:
            d = f.read()
        nb = item[1]
        okmark = "OK" if len(d) == nb else "** 应为 %d，差 %+d **" % (nb, len(d) - nb)
        if len(d) != nb:
            bad = True
        print("  [%d] %-28s %10d 字节  %s" % (i, os.path.basename(path), len(d), okmark))
        data_in.append(d)
    if bad:
        die("输入 bin 的字节数和模型要的对不上（或文件不存在）。\n"
            "    顺序错了也会这样 —— 上面那张表就是模型要的顺序。")

    keep, devs, bufs = [], [], []

    def mkbuf(data, nb):
        dev = ctypes.c_void_p()
        check(acl.aclrtMalloc(ctypes.byref(dev), nb, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc(%d) 失败, ret = " % nb + "%d")
        host = ctypes.create_string_buffer(data, nb)
        keep.append(host)
        check(acl.aclrtMemcpy(dev, nb, ctypes.cast(host, ctypes.c_void_p), nb,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D = %d")
        b = acl.aclCreateDataBuffer(dev, nb)
        if not b:
            die("aclCreateDataBuffer 返回 null")
        devs.append(dev)
        bufs.append(b)
        return b

    ds_in = acl.aclmdlCreateDataset()
    ds_out = acl.aclmdlCreateDataset()
    if not ds_in or not ds_out:
        die("aclmdlCreateDataset 返回 null")
    for i, d in enumerate(data_in):
        check(acl.aclmdlAddDatasetBuffer(ds_in, mkbuf(d, ins[i][1])),
              "aclmdlAddDatasetBuffer(in %d) = " % i + "%d")
    # 输出预填哨兵：kernel 一个字节都没写的话读回来还是它
    for i, item in enumerate(outs):
        check(acl.aclmdlAddDatasetBuffer(ds_out, mkbuf(bytes([SENTINEL]) * item[1], item[1])),
              "aclmdlAddDatasetBuffer(out %d) = " % i + "%d")
    n_in_bufs = len(data_in)

    print("\n==== 执行 ====")
    for k in range(warmup):
        rc = acl.aclmdlExecute(model_id, ds_in, ds_out)
        if rc != ACL_SUCCESS:
            die("warmup 第 %d 次 aclmdlExecute = %d" % (k + 1, rc))
    if warmup:
        print("  warmup %d 次完成（不计入统计）" % warmup)

    durs = []
    for k in range(repeat):
        t0 = time.perf_counter()
        rc = acl.aclmdlExecute(model_id, ds_in, ds_out)
        if rc != ACL_SUCCESS:
            die("第 %d 次 aclmdlExecute = %d" % (k + 1, rc))
        durs.append((time.perf_counter() - t0) * 1e6)
    print("  aclmdlExecute x %d OK" % repeat)

    # 判据看的是**最后一次**的结果。连跑多次时这也顺带验了稳态下依然写对。
    if repeat > 1 or warmup:
        print("\n==== 耗时（host 侧墙钟，含下发和同步；设备侧以 msprof 为准）====")
        report_times(durs)

    rcode = 0
    for i, (nm, nb, dt, dtname, isz, code, fmt, dims) in enumerate(outs):
        buf = ctypes.create_string_buffer(nb)
        check(acl.aclrtMemcpy(ctypes.cast(buf, ctypes.c_void_p), nb, devs[n_in_bufs + i], nb,
                              ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H = %d")
        got = buf.raw[:nb]
        sent = got.count(SENTINEL)
        print("\n==== 输出[%d] %s ====" % (i, dtname))
        print("  字节 %d   还是哨兵(0x%02X)的: %d   全 0 的: %d" % (nb, SENTINEL, sent, got.count(0)))
        if sent == nb:
            print("  [FAIL] 整个缓冲一个字节都没被改写 —— kernel 没跑")
            rcode = 1
        elif sent * 10 > nb:
            print("  [!] 超过 10%% 的字节还是哨兵 —— 可能只写了一部分")

        if write_out:
            fp = "%s%d.bin" % (out_prefix, i)
            with open(fp, "wb") as f:
                f.write(got)
            print("  已写出 %s" % fp)

        gp = opts.get("golden%d" % i)
        if not gp:
            continue
        if not os.path.isfile(gp):
            die("golden 文件 %s 不存在" % gp)
        with open(gp, "rb") as f:
            gold = f.read()
        if len(gold) != nb:
            die("golden %s 是 %d 字节，输出是 %d 字节" % (gp, len(gold), nb))
        nelem = (nb // isz) if isz else 0
        ok, mx, mi, exact, hist, total = compare(got, gold, code, nelem, rel_tol)
        ratio = ok / float(total)
        print("  [精度] 逐位相等 %d / %d" % (exact, total))
        print("         相对误差 <= %g 的比例: %d / %d = %.6f%%   (下限 %.6f%%)"
              % (rel_tol, ok, total, ratio * 100.0, ratio_min * 100.0))
        if mi >= 0:
            print("         最大相对误差 %s @ %d"
                  % ("inf" if mx == float("inf") else "%.6e" % mx, mi))
        for k in ["=0", "(0,1e-3]", "(1e-3,1e-2]", "(1e-2,1e-1]", "(1e-1,1]", ">1", "inf(golden=0)"]:
            if hist[k]:
                print("           %-14s %9d  (%.6f%%)" % (k, hist[k], hist[k] * 100.0 / total))
        if ratio < ratio_min:
            print("  [FAIL] 达标比例 %.6f%% < 下限 %.6f%%" % (ratio * 100.0, ratio_min * 100.0))
            rcode = 1

    acl.aclmdlDestroyDataset(ds_in)
    acl.aclmdlDestroyDataset(ds_out)
    for b in bufs:
        acl.aclDestroyDataBuffer(b)
    for dv in devs:
        acl.aclrtFree(dv)
    acl.aclmdlDestroyDesc(desc)
    acl.aclmdlUnload(model_id)
    acl.aclrtResetDevice(device_id)
    acl.aclFinalize()

    print()
    print("[FAIL] 见上" if rcode else "[PASS] 执行完成")
    return rcode


if __name__ == "__main__":
    sys.exit(main())
