# -*- coding: utf-8 -*-
"""给桩生成「模型输入表」和 golden。

输入表用**和运行器同一条规则**（model_input_slots），不是「case.bin 里有哪些张量」
—— 桩要是自己另算一套，就会把运行器里那段「模型要的字节数 vs case.bin 给的字节
数」核对测成假通过。

恒长之后这两者**不再等价**：全 fp16 通路的 om 有 7 个输入（三个 DYNAMIC list 各
两个实例），而 case.bin 里只有 5 个张量 —— gen_case 在那条通路上不产定标表，那两
个 quant 实例是纯占位。按 case.bin 算会得到 5，桩就会说「模型报了 5 个输入」。
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import run_fused_conv2d as R   # noqa: E402

if len(sys.argv) < 3:
    raise SystemExit("用法: make_conf.py <case.bin> <输出目录>")
case, outdir = sys.argv[1], sys.argv[2]
tensors, _nonzero, _probe, _sat, y_elems, _s1, _s2, info = R.load_case(case)
present = R.model_input_slots(info)
y_bytes = y_elems * (1 if info["outInt8"] else 2)


def slot_bytes(name):
    """这个槽在 om 里是多少字节。

    不在 case.bin 里的是恒长占位，按 fc2d_onnx.tensor_plan 给它们声明的形状算：
      * scale 的两个实例  uint64，一个通道一项 -> cout * 8
      * relu_weight 的两个实例  float32 -> cout * 4
    """
    if name in tensors:
        return len(tensors[name][2])
    if name == "quant_scale1":
        return int(info["cout1"]) * 8
    if name in ("quant_scale2", "dequant_scale2"):
        return int(info["cout2"]) * 8
    if name == "relu_weight1":
        return int(info["cout1"]) * 4
    if name == "relu_weight2":
        return int(info["cout2"]) * 4
    raise SystemExit("不认识的槽 %s，而且 case.bin 里也没有" % name)


os.makedirs(outdir, exist_ok=True)
with io.open(os.path.join(outdir, "stub.conf"), "w") as f:
    for n in present:
        f.write(u"%s %d\n" % (n, slot_bytes(n)))
    f.write(u"OUT %d\n" % y_bytes)
with open(os.path.join(outdir, "golden.bin"), "wb") as f:
    f.write(tensors["y_expect"][2])

print("模型输入表（%d 个 = x + 四个 DYNAMIC list x 每层一个实例）:" % len(present))
for n in present:
    tag = "" if n in tensors else "   <- 恒长占位，case.bin 里没有"
    print("  %-16s %9d 字节%s" % (n, slot_bytes(n), tag))
print("输出 %d 字节" % y_bytes)
