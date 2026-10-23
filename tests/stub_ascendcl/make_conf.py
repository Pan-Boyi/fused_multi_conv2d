# -*- coding: utf-8 -*-
"""给桩生成「模型输入表」和 golden。

输入表用**和运行器同一条规则**（model_input_slots），不是「case.bin 里有哪些张量」
—— 桩要是自己另算一套，就会把运行器里那段「模型要的字节数 vs case.bin 给的字节
数」核对测成假通过。

Graphs expose x plus consumed bias tensors (1/3 inputs).
Filters and consumed scale tables are constant initializers.
The production pass creates unused list instances as internal Const nodes.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import run_fused_multi_conv2d as R   # noqa: E402

if len(sys.argv) < 3:
    raise SystemExit("用法: make_conf.py <case.bin> <输出目录>")
case, outdir = sys.argv[1], sys.argv[2]
tensors, _nonzero, _probe, _sat, y_elems, _s1, _s2, info = R.load_case(case)
present = R.model_input_slots(info)
y_bytes = y_elems * (1 if info["outInt8"] else 2)


def slot_bytes(name):
    """Every exposed source input has actual data in the case."""
    if name not in tensors:
        raise SystemExit("case is missing runtime input " + name)
    return len(tensors[name][2])


os.makedirs(outdir, exist_ok=True)
with io.open(os.path.join(outdir, "stub.conf"), "w") as f:
    for n in present:
        f.write(u"%s %d\n" % (n, slot_bytes(n)))
    f.write(u"OUT %d\n" % y_bytes)
with open(os.path.join(outdir, "golden.bin"), "wb") as f:
    f.write(tensors["y_expect"][2])

print("模型输入表（%d 个，x 与实际使用的 bias）:" % len(present))
for n in present:
    print("  %-16s %9d 字节" % (n, slot_bytes(n)))
print("输出 %d 字节" % y_bytes)
