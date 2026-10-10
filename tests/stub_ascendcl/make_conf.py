# -*- coding: utf-8 -*-
"""给桩生成「模型输入表」和 golden。

输入表**从 case.bin 派生**，和运行器实际要喂的东西同源 —— 桩要是自己另算一套
字节数，就会把运行器里那段「模型要的字节数 vs case.bin 给的字节数」核对
测成假通过。
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
present = [n for n in R.INPUT_SLOTS if n in tensors]
y_bytes = y_elems * (1 if info["outInt8"] else 2)

os.makedirs(outdir, exist_ok=True)
with io.open(os.path.join(outdir, "stub.conf"), "w") as f:
    for n in present:
        f.write(u"%s %d\n" % (n, len(tensors[n][2])))
    f.write(u"OUT %d\n" % y_bytes)
with open(os.path.join(outdir, "golden.bin"), "wb") as f:
    f.write(tensors["y_expect"][2])

print("模型输入表（%d 个，缺席的 optional 在图 om 里不是模型输入）:" % len(present))
for n in present:
    print("  %-16s %9d 字节" % (n, len(tensors[n][2])))
print("输出 %d 字节" % y_bytes)
