#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一个 .onnx 拆开摊平打印 —— 节点、属性、initializer 的 dtype/shape/取值范围。

    python3 inspect_onnx.py model.onnx
    python3 inspect_onnx.py model.onnx --node=Conv_153      # 只看某个节点
    python3 inspect_onnx.py model.onnx --sub=Conv_152       # 从某个张量往下截一段

**这是整条链里第一个该跑的脚本。**

Netron 的截图能看出拓扑和 shape，看不出三样东西，而这三样恰好决定 golden 算得
对不对：

  1. AscendQuant 的 scale / offset / round_mode / dst_type —— 图上完全不显示；
  2. AscendDequant 那个第二输入（截图里标着 "1 <128>"）到底是 uint64 还是
     float32。uint64 的话，真正的 float 缩放系数藏在低 32 位里，直接当整数用会
     错得离谱；
  3. Conv 的 pads / strides / dilations / group。本例里 3x3、32x48 -> 32x48 能
     反推出 stride=1 pad=1 dilation=1 group=1，但这是这一个 case 的运气。

所以：有原始 .onnx 就别去重建图，直接 atc 它，然后用 onnx_golden.py 从**同一个
文件**里读参数算 golden。重建只在你手上只有截图时才有意义。

不依赖 numpy，只要 onnx。
"""
import sys, os, struct

try:
    import onnx
    from onnx import numpy_helper, TensorProto
except ImportError:
    print("[X] 没装 onnx：pip3 install onnx")
    sys.exit(1)

DT = {v: k for k, v in TensorProto.DataType.items()}


def attr_str(a):
    """把 AttributeProto 打成一行。"""
    t = a.type
    A = onnx.AttributeProto
    if t == A.FLOAT:   return repr(a.f)
    if t == A.INT:     return repr(a.i)
    if t == A.STRING:  return repr(a.s.decode(errors="replace"))
    if t == A.FLOATS:  return repr(list(a.floats))
    if t == A.INTS:    return repr(list(a.ints))
    if t == A.STRINGS: return repr([s.decode(errors="replace") for s in a.strings])
    if t == A.TENSOR:  return "<tensor %s %s>" % (DT.get(a.t.data_type, a.t.data_type), list(a.t.dims))
    return "<%s>" % A.AttributeType.Name(t)


def type_str(vi):
    tt = vi.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        dims.append(d.dim_param if d.HasField("dim_param") else d.dim_value)
    return "%-8s %s" % (DT.get(tt.elem_type, tt.elem_type), dims)


def summarize(t):
    """initializer 的一行摘要。uint64 的 deq_scale 额外把低 32 位当 float 解出来。"""
    dt = DT.get(t.data_type, str(t.data_type))
    dims = list(t.dims)
    head = "%-8s %-18s" % (dt, dims)
    try:
        arr = numpy_helper.to_array(t)
    except Exception as e:
        return head + "  <解不开: %s>" % e
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return head + "  <空>"
    extra = ""
    if t.data_type == TensorProto.UINT64:
        # AscendDequant 的 deq_scale：GE 把 float32 的比特塞在低 32 位，高位是
        # offset / 标志位。这一步是判断"这个 128 长的东西到底是什么"的关键。
        lo = (flat.astype("uint64") & 0xFFFFFFFF).astype("uint32")
        fs = lo.view("float32")
        extra = ("\n%s低32位当 float32 解: [%.6g, %.6g]  首个 %.9g  (高位非零的有 %d 个)"
                 % (" " * 6, float(fs.min()), float(fs.max()), float(fs[0]),
                    int((flat >> 32 != 0).sum())))
    try:
        return head + "  范围 [%s, %s]  前几个 %s%s" % (
            flat.min(), flat.max(), list(flat[:4]), extra)
    except TypeError:
        return head + extra


def main():
    argv = [a for a in sys.argv[1:]]
    opts = {}
    pos = []
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v or "1"
        else:
            pos.append(a)
    if not pos:
        print(__doc__)
        return 1
    path = pos[0]
    if not os.path.isfile(path):
        print("[X] 找不到 %s" % path)
        return 1

    m = onnx.load(path)
    g = m.graph
    print("==== 文件 ====")
    print("  %s  (%d 字节)" % (path, os.path.getsize(path)))
    print("  ir_version=%d  producer=%r" % (m.ir_version, m.producer_name))
    for o in m.opset_import:
        print("  opset  domain=%r  version=%d" % (o.domain, o.version))

    print("\n==== 图输入（.om 的输入顺序就是这个顺序）====")
    inits = {t.name for t in g.initializer}
    real_in = [vi for vi in g.input if vi.name not in inits]
    for i, vi in enumerate(real_in):
        print("  [%d] %-32s %s" % (i, vi.name, type_str(vi)))
    if len(real_in) != len(g.input):
        print("  （另有 %d 个 input 其实是 initializer，不算模型输入）"
              % (len(g.input) - len(real_in)))

    print("\n==== 图输出 ====")
    for i, vi in enumerate(g.output):
        print("  [%d] %-32s %s" % (i, vi.name, type_str(vi)))

    only = opts.get("node")
    sub = opts.get("sub")
    live = None
    if sub:
        live = {sub}

    print("\n==== 节点 ====")
    initmap = {t.name: t for t in g.initializer}
    ops = {}
    shown = 0
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
        if only and n.name != only:
            continue
        if live is not None:
            if not any(i in live for i in n.input):
                continue
            live.update(n.output)
        shown += 1
        print("\n  %s  (%s)" % (n.name or "<匿名>", n.op_type)
              + ("  domain=%r" % n.domain if n.domain else ""))
        for i, x in enumerate(n.input):
            tag = "初始化" if x in initmap else "张量  "
            print("      in [%d] %s %s" % (i, tag, x))
            if x in initmap:
                print("            %s" % summarize(initmap[x]))
        for i, x in enumerate(n.output):
            print("      out[%d]        %s" % (i, x))
        for a in n.attribute:
            print("      attr  %-18s = %s" % (a.name, attr_str(a)))
    if not shown:
        print("  （过滤后没有节点。--node 要写节点名，--sub 要写张量名）")

    print("\n==== 算子统计 ====")
    for k in sorted(ops, key=lambda k: -ops[k]):
        mark = "   <- 昇腾自定义，标准 onnxruntime 跑不了" if k.startswith("Ascend") else ""
        print("  %-20s %4d%s" % (k, ops[k], mark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
