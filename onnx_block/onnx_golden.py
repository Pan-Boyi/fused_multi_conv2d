#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 numpy 复算一个昇腾量化 ONNX 图，造出输入 bin 和 golden bin。

    python3 onnx_golden.py block.onnx --outdir=io
    python3 onnx_golden.py block.onnx --outdir=io --round=away
    python3 onnx_golden.py block.onnx --outdir=io --in0=my_x.bin   # 用现成的输入

产物（编号就是模型的输入/输出顺序）：
    io/in0.<name>.bin       喂给 run_om.py 的输入
    io/golden0.<name>.bin   对应的 golden 输出
    io/report.txt           每一级的取值范围，出了问题按级对

**所有参数都从 .onnx 里读**，脚本里没有一个硬编码的 shape / scale。所以同一份
代码既能算 build_block.py 重建出来的图，也能算你手上那个真图 —— 后者才是重点：
golden 和 atc 吃的是同一个文件，参数不可能对不上。

设计上的一条硬规矩：**遇到没验证过的属性组合就报错退出，不猜。**
sqrt_mode=1、dst_type 不是 int8、group != 1、AscendDequant 的 relu_flag=1 之类，
都会直接停下来告诉你缺什么。一个偷偷猜错的 golden 比没有 golden 更糟 —— 后面所有
"实测对不上"的结论都会跟着错。

唯一一处只能靠实测定的是量化的**舍入模式**：

  --round=rint （默认）  向最近偶数舍入。我在 5102 上验过 Cast(CAST_RINT) 走的是
                         这个，融合卷积那个算子的 golden 也是按它写的、逐位对上。
  --round=away           .5 向远离 0 的方向舍入。

两者只在恰好落在 .5 的点上不同。脚本会**数出**有多少个这样的点并打印，你一眼能
看出这个选择到底影响多大；如果 ties=0，两种模式产物完全一样，这个疑问就不存在。
"""
import sys, os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

GE_DT = {0: "float32", 1: "float16", 2: "int8", 3: "int32", 9: "int64"}
GE_TO_NP = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32}
ONNX_TO_NP = {
    TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16,
    TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8,
    TensorProto.INT32: np.int32, TensorProto.INT64: np.int64,
    TensorProto.UINT64: np.uint64,
}


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


def attrs(node):
    d = {}
    A = onnx.AttributeProto
    for a in node.attribute:
        if a.type == A.FLOAT:    d[a.name] = a.f
        elif a.type == A.INT:    d[a.name] = a.i
        elif a.type == A.STRING: d[a.name] = a.s.decode()
        elif a.type == A.INTS:   d[a.name] = list(a.ints)
        elif a.type == A.FLOATS: d[a.name] = list(a.floats)
        elif a.type == A.TENSOR: d[a.name] = numpy_helper.to_array(a.t)
    return d


# --------------------------------------------------------------------- 算子
def conv(x, w, b, at, name):
    """int8 x int8 -> int32。累加用 float64 做，然后断言它精确。

    K = Cin*kh*kw 最大 1152，|x|,|w| <= 127，所以 |acc| <= 1152*127*127 = 1.86e7，
    离 2^53 远得很，float64 里每一位都是精确的 —— 用 BLAS 换来的是几百倍速度，
    不是精度上的妥协。断言把这一点钉死。
    """
    g = at.get("group", 1)
    if g != 1:
        die("%s: group=%d，这个 golden 只实现了 group=1（深度可分离要另写）" % (name, g))
    kh, kw = at.get("kernel_shape", list(w.shape[2:]))
    sh, sw = at.get("strides", [1, 1])
    dh, dw = at.get("dilations", [1, 1])
    pads = at.get("pads", [0, 0, 0, 0])
    if len(pads) != 4:
        die("%s: pads=%s，只支持 4 个数的 2D 形式" % (name, pads))
    pt, pl, pb, pr = pads[0], pads[1], pads[2], pads[3]

    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    if w.shape[1] != Cin:
        die("%s: 权重的 Cin=%d 和输入的 %d 对不上" % (name, w.shape[1], Cin))
    Ho = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1

    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    wf = w.astype(np.float64)
    acc = np.zeros((N * Ho * Wo, Cout), np.float64)
    for i in range(kh):
        for j in range(kw):
            xs = xp[:, :, i * dh: i * dh + Ho * sh: sh, j * dw: j * dw + Wo * sw: sw]
            xs = np.ascontiguousarray(xs.transpose(0, 2, 3, 1)).reshape(-1, Cin)
            acc += xs @ wf[:, :, i, j].T
    if not np.all(acc == np.rint(acc)):
        die("%s: float64 累加出现了非整数，说明量级超了预期，改用 int64 重算" % name)
    if acc.min() < -2**31 or acc.max() > 2**31 - 1:
        die("%s: 累加结果 [%g,%g] 溢出 int32" % (name, acc.min(), acc.max()))
    y = acc.astype(np.int32).reshape(N, Ho, Wo, Cout).transpose(0, 3, 1, 2)
    if b is not None:
        y = (y.astype(np.int64) + b.astype(np.int64).reshape(1, -1, 1, 1))
        if y.min() < -2**31 or y.max() > 2**31 - 1:
            die("%s: 加完 bias 溢出 int32" % name)
        y = y.astype(np.int32)
    return np.ascontiguousarray(y)


def unpack_deq(s, name):
    """AscendDequant 的第二输入 -> float32 的缩放系数。

    uint64 是 GE 的打包形式：float32 的比特原样在低 32 位。直接把 uint64 当数值
    用会得到 1e9 量级的乘数，结果整个爆掉 —— 这是这条链上最容易静默错的一步，
    所以高位非零时直接停下来问。
    """
    if s.dtype == np.uint64 or s.dtype == np.int64:
        u = s.astype(np.uint64)
        hi = (u >> np.uint64(32))
        if np.any(hi != 0):
            die("%s 的 deq_scale 高 32 位非零（%d 个），里面带了 offset 或标志位，"
                "本 golden 只实现了纯 scale 的情形" % (name, int((hi != 0).sum())))
        return (u & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32)
    if s.dtype in (np.float32, np.float64):
        return s.astype(np.float32)
    die("%s 的 deq_scale 是 %s，认不出来（预期 uint64 或 float32）" % (name, s.dtype))


def dequant(x, s, at, name, st):
    if at.get("sqrt_mode", 0):
        die("%s: sqrt_mode=1，语义我没在板上验过，不猜" % name)
    if at.get("relu_flag", 0):
        die("%s: relu_flag=1，请确认 relu 是在乘 scale 之前还是之后；不猜" % name)
    dt = at.get("dtype", 1)
    if dt not in GE_TO_NP:
        die("%s: dtype=%s 不认识" % (name, dt))
    sc = unpack_deq(s, name)
    if sc.size not in (1, x.shape[1]):
        die("%s: deq_scale 有 %d 个，既不是 1 也不是通道数 %d" % (name, sc.size, x.shape[1]))
    st[name + ".scale"] = (float(sc.min()), float(sc.max()))
    y = x.astype(np.float32) * sc.reshape(1, -1, 1, 1)
    return y.astype(GE_TO_NP[dt])


def quant(x, at, name, mode, st):
    if at.get("sqrt_mode", 0):
        die("%s: sqrt_mode=1，语义我没在板上验过，不猜" % name)
    dst = at.get("dst_type", 2)
    if dst != 2:
        die("%s: dst_type=%s，只实现了 int8(2)" % (name, GE_DT.get(dst, dst)))
    rm = at.get("round_mode", "Round")
    sc = float(at.get("scale", 1.0))
    off = float(at.get("offset", 0.0))
    v = x.astype(np.float32) * np.float32(sc) + np.float32(off)
    v = v.astype(np.float64)
    if rm == "Round":
        ties = int(np.sum(np.abs(v - np.trunc(v)) == 0.5))
        st[name + ".ties"] = ties
        r = np.rint(v) if mode == "rint" else np.trunc(v + np.copysign(0.5, v))
    elif rm == "Floor": r = np.floor(v)
    elif rm == "Ceil":  r = np.ceil(v)
    elif rm == "Trunc": r = np.trunc(v)
    else:
        die("%s: round_mode=%r 不认识" % (name, rm))
    sat = int(np.sum((r < -128) | (r > 127)))
    st[name + ".sat"] = sat
    return np.clip(r, -128, 127).astype(np.int8)


# --------------------------------------------------------------------- 主流程
def main():
    o, p = {}, []
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v or "1"
        else:
            p.append(a)
    if not p:
        print(__doc__)
        return 1
    path = p[0]
    outdir = o.get("outdir", "io")
    mode = o.get("round", "rint")
    seed = int(o.get("seed", "0"))
    if mode not in ("rint", "away"):
        die("--round 只能是 rint 或 away")
    if not os.path.isfile(path):
        die("找不到 %s" % path)
    os.makedirs(outdir, exist_ok=True)

    m = onnx.load(path)
    g = m.graph
    env = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    real_in = [vi for vi in g.input if vi.name not in env]

    rng = np.random.default_rng(seed)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log("==== 模型输入（喂 run_om.py 时就按这个顺序）====")
    for i, vi in enumerate(real_in):
        tt = vi.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        if any(s <= 0 for s in shape):
            die("输入 %s 的 shape 有动态维 %s；先用 atc 的 --input_shape 定死" % (vi.name, shape))
        np_dt = ONNX_TO_NP.get(tt.elem_type)
        if np_dt is None:
            die("输入 %s 的 dtype %s 不支持" % (vi.name, tt.elem_type))
        given = o.get("in%d" % i)
        if given:
            a = np.fromfile(given, dtype=np_dt)
            if a.size != int(np.prod(shape)):
                die("%s 有 %d 个元素，%s 要 %d 个" % (given, a.size, vi.name, int(np.prod(shape))))
            a = a.reshape(shape)
            src = given
        elif np_dt == np.int8:
            a = rng.integers(-128, 128, size=shape).astype(np.int8)
            src = "随机 int8"
        else:
            a = (rng.standard_normal(shape) * 3.0).astype(np_dt)
            src = "随机高斯 x3"
        env[vi.name] = a
        fn = os.path.join(outdir, "in%d.%s.bin" % (i, vi.name.replace("/", "_")))
        a.tofile(fn)
        log("  [%d] %-24s %-9s %-18s %9d 字节  <- %s"
            % (i, vi.name, a.dtype, list(a.shape), a.nbytes, src))

    st = {}
    log("\n==== 逐级复算 ====")
    for n in g.node:
        at = attrs(n)
        xs = [env.get(i) for i in n.input]
        for i, v in zip(n.input, xs):
            if v is None:
                die("节点 %s 要的张量 %s 还没算出来 —— 图不是拓扑序？" % (n.name, i))
        nm = n.name or n.op_type
        if n.op_type == "Conv":
            y = conv(xs[0], xs[1], xs[2] if len(xs) > 2 else None, at, nm)
        elif n.op_type == "AscendDequant":
            y = dequant(xs[0], xs[1], at, nm, st)
        elif n.op_type == "AscendQuant":
            y = quant(xs[0], at, nm, mode, st)
        elif n.op_type == "Relu":
            y = np.maximum(xs[0], 0)
        elif n.op_type == "Add":
            y = (xs[0].astype(np.float32) + xs[1].astype(np.float32)).astype(xs[0].dtype)
        elif n.op_type == "Mul":
            y = (xs[0].astype(np.float32) * xs[1].astype(np.float32)).astype(xs[0].dtype)
        else:
            die("没实现的算子 %s（节点 %s）。图里有它就说明我这份 golden 覆盖不全，"
                "补上再用。" % (n.op_type, nm))
        env[n.output[0]] = y
        log("  %-14s %-14s -> %-9s %-18s 范围 [%.6g, %.6g]"
            % (nm, n.op_type, y.dtype, list(y.shape), float(y.min()), float(y.max())))

    log("\n==== golden 输出 ====")
    for i, vi in enumerate(g.output):
        y = env.get(vi.name)
        if y is None:
            die("输出 %s 没算出来" % vi.name)
        fn = os.path.join(outdir, "golden%d.%s.bin" % (i, vi.name.replace("/", "_")))
        y.tofile(fn)
        nz = int(np.count_nonzero(y))
        log("  [%d] %-24s %-9s %-18s %9d 字节  非零 %d/%d"
            % (i, vi.name, y.dtype, list(y.shape), y.nbytes, nz, y.size))

    if st:
        log("\n==== 数值告警 ====")
        for k in sorted(st):
            v = st[k]
            if k.endswith(".ties"):
                log("  %-24s 恰好落在 .5 的点 %d 个  %s" % (k, v,
                    "-> --round 选哪个都一样" if v == 0 else
                    "-> --round=rint / away 会在这些点上不同，占 %.4f%%"
                    % (v * 100.0 / env[g.output[0].name].size)))
            elif k.endswith(".sat"):
                log("  %-24s 饱和到 +-128 的点 %d 个%s" % (k, v,
                    "" if v == 0 else "  <- scale 偏大，真实模型里不该有这么多"))
            elif k.endswith(".scale"):
                log("  %-24s deq_scale 范围 [%.6g, %.6g]" % (k, v[0], v[1]))

    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ins = " ".join(sorted(os.path.join(outdir, x) for x in os.listdir(outdir) if x.startswith("in")))
    gold = " ".join("--golden%d=%s" % (i, os.path.join(outdir, x))
                    for i, x in enumerate(sorted(x for x in os.listdir(outdir) if x.startswith("golden"))))
    print("\n  上板跑:")
    print("    python3 ../run_om.py <model.om> %s %s" % (ins, gold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
