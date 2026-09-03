#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按截图里那张图重建一个 .onnx（量化残差块）。

    python3 build_block.py                      # 全图，写出 block.onnx
    python3 build_block.py --boxed              # 只要红框那四个节点
    python3 build_block.py --fp-in              # 前面补一个 AscendQuant，模型输入变 fp32
    python3 build_block.py --deq-dtype=float32  # deq_scale 存 float32 而不是 uint64

重建出来的拓扑（和截图逐节点对应）：

    x:int8 [4,128,32,48]                     <- Conv_152.quant.output0
      |
      +-- Conv    W int8 [128,128,3,3]  B int32 [128]   -> int32     ]
      +-- AscendDequant  deq_scale [128]                -> float16   ] 红框
      +-- Relu                                          -> float16   ]
      +-- AscendQuant                                   -> int8      ]
      |
      +-- Conv    W int8 [128,128,3,3]  B int32 [128]   -> int32
      +-- AscendDequant  deq_scale [128]                -> float16
      |
    Add( ^ , res:float16 [4,128,32,48] )                -> y:float16
                                                           ^ 截图里的 1661

pads/strides/dilations/group 截图上没写，但被 shape 反推死了：3x3 卷积、
32x48 -> 32x48，只能是 stride=1 pad=1 dilation=1；W 是 (128,128,3,3) 而输入
128 通道，所以 group=1。这几个不用猜。

**猜的是这些**，用之前请拿 inspect_onnx.py 对着真文件核一遍：

  * AscendQuant / AscendDequant 的 domain。这里用默认域 ""（AMCT 导出的模型就是
    这样），--domain= 可以改。
  * 两个自定义算子的属性名和取值：AscendQuant 的 scale/offset/sqrt_mode/
    round_mode/dst_type，AscendDequant 的 dtype/sqrt_mode/relu_flag。
  * deq_scale 的存储类型。默认 uint64（GE 的打包形式，float32 的比特在低 32
    位）；AMCT 某些版本直接存 float32，用 --deq-dtype=float32。
  * 权重/scale 的**数值**本身当然是随机造的，只保证量级合理不至于 fp16 溢出。

注意 onnx.checker 会拒绝这个模型：标准 ONNX 的 Conv 类型约束只允许浮点，而这里
Conv 吃 int8 出 int32。这是 AMCT 量化模型的常态，atc 的 onnx 解析器直接读
protobuf、不跑 ONNX 的类型推导，所以照样能编。脚本因此只把 checker 的抱怨当
警告打印。
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

# ---- 截图上量出来的形状 ----
N, C, H, W = 4, 128, 32, 48
KH = KW = 3
PAD, STRIDE, DIL, GROUP = 1, 1, 1, 1

# GE 的 DataType 枚举，AscendDequant 的 dtype 属性用它（不是 ONNX 的 TensorProto）
GE_DT_FLOAT, GE_DT_FLOAT16, GE_DT_INT8 = 0, 1, 2


def opt(argv):
    o, p = {}, []
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v or "1"
        else:
            p.append(a)
    return o, p


def main():
    o, p = opt(sys.argv[1:])
    out = p[0] if p else o.get("out", "block.onnx")
    boxed = "boxed" in o
    fp_in = "fp-in" in o
    dom = o.get("domain", "")
    deq_dt = o.get("deq-dtype", "uint64")
    seed = int(o.get("seed", "0"))
    if deq_dt not in ("uint64", "float32"):
        print("[X] --deq-dtype 只能是 uint64 或 float32")
        return 1

    rng = np.random.default_rng(seed)
    inits, nodes = [], []

    def add_init(name, arr):
        t = numpy_helper.from_array(np.ascontiguousarray(arr), name)
        inits.append(t)
        return name

    def deq_scale(name, scale_f32):
        """AscendDequant 的第二输入。

        uint64 形态：float32 的比特原样放低 32 位，高 32 位（offset / 标志）留 0。
        这和硬件 fixpipe 的 VDEQF16 通路吃的是同一个打包。
        """
        s = np.asarray(scale_f32, dtype=np.float32)
        if deq_dt == "float32":
            return add_init(name, s)
        return add_init(name, s.view(np.uint32).astype(np.uint64))

    def conv(tag, x, cout, cin, s_w=24):
        w = rng.integers(-s_w, s_w + 1, size=(cout, cin, KH, KW)).astype(np.int8)
        b = rng.integers(-4096, 4096, size=(cout,)).astype(np.int32)
        nodes.append(helper.make_node(
            "Conv", [x, add_init(tag + ".w", w), add_init(tag + ".b", b)], [tag + ".out"],
            name=tag,
            kernel_shape=[KH, KW], pads=[PAD, PAD, PAD, PAD],
            strides=[STRIDE, STRIDE], dilations=[DIL, DIL], group=GROUP))
        return tag + ".out"

    def dequant(tag, x, scale, out_ge_dt=GE_DT_FLOAT16):
        nodes.append(helper.make_node(
            "AscendDequant", [x, deq_scale(tag + ".deq", np.full(C, scale, np.float32))],
            [tag + ".out"], name=tag, domain=dom,
            dtype=out_ge_dt, sqrt_mode=0, relu_flag=0))
        return tag + ".out"

    def quant(tag, x, scale, offset=0.0):
        nodes.append(helper.make_node(
            "AscendQuant", [x], [tag + ".out"], name=tag, domain=dom,
            scale=float(scale), offset=float(offset),
            sqrt_mode=0, round_mode="Round", dst_type=GE_DT_INT8))
        return tag + ".out"

    # ---- 量级：让 relu 后大致落在 [0, 10]，量化回 int8 用 12.7，最后 Add 完仍在
    # fp16 的舒适区。K = 128*9 = 1152 项，x/w 都在 +-24 上下，累加 std ~ 1e4。
    # 这两个数是**量出来**定的，不是拍的：先跑一遍 onnx_golden.py 看 Quant 的饱和
    # 计数，让 relu 后的量级正好铺满 int8 而几乎不撞 +-127。撞得多说明 scale 偏大，
    # 那是个不像真实量化模型的图，拿它测出来的精度结论没有参考价值。
    DEQ = 5.0e-5
    QS = 14.0

    graph_in = []
    if fp_in:
        graph_in.append(helper.make_tensor_value_info("x_fp", TensorProto.FLOAT, [N, C, H, W]))
        cur = quant("Quant_in", "x_fp", QS)
    else:
        graph_in.append(helper.make_tensor_value_info("x", TensorProto.INT8, [N, C, H, W]))
        cur = "x"

    cur = conv("Conv_1", cur, C, C)
    cur = dequant("Dequant_1", cur, DEQ)
    nodes.append(helper.make_node("Relu", [cur], ["Relu_1.out"], name="Relu_1"))
    cur = "Relu_1.out"
    cur = quant("Quant_1", cur, QS)

    if boxed:
        outs = [helper.make_tensor_value_info(cur, TensorProto.INT8, [N, C, H, W])]
    else:
        cur = conv("Conv_2", cur, C, C)
        cur = dequant("Dequant_2", cur, DEQ)
        graph_in.append(helper.make_tensor_value_info("res", TensorProto.FLOAT16, [N, C, H, W]))
        nodes.append(helper.make_node("Add", [cur, "res"], ["y"], name="Add"))
        outs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [N, C, H, W])]

    g = helper.make_graph(nodes, "quant_residual_block", graph_in, outs, inits)
    m = helper.make_model(g, producer_name="build_block.py",
                          opset_imports=[helper.make_opsetid("", 11)]
                          + ([helper.make_opsetid(dom, 1)] if dom else []))
    m.ir_version = 7   # 和 CANN 自带的 onnx 解析器对得上的保守取值

    try:
        onnx.checker.check_model(m)
        print("  onnx.checker: 过")
    except Exception as e:
        first = str(e).strip().splitlines()[0]
        print("  onnx.checker: 不过（预期之内，见文件头注释）—— %s" % first)

    onnx.save(m, out)
    print("\n[OK] 写出 %s (%d 字节)" % (out, os.path.getsize(out)))
    print("  节点   : %s" % " -> ".join(n.op_type for n in nodes))
    print("  输入   : %s" % ", ".join(
        "%s %s" % (v.name, TensorProto.DataType.Name(v.type.tensor_type.elem_type))
        for v in graph_in))
    print("  输出   : %s" % ", ".join(
        "%s %s" % (v.name, TensorProto.DataType.Name(v.type.tensor_type.elem_type))
        for v in outs))
    print("  deq_scale 存成 %s" % deq_dt)
    print("\n  下一步:")
    print("    python3 onnx_golden.py %s --outdir=io   # 造输入 + 算 golden" % out)
    print("    SOC=<soc> ./build_om_onnx.sh %s          # 编 om" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
