#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""把编 .om 用的 ATC singleop json 转成 run_op.py 的 case.json。

    python3 singleop2case.py <singleop.json> [out.json] [--om-dir=om_out] [--index=0]

为什么要有这个:走离线模型时,ACL 是拿 **op 类型 + 每个 tensor 的 shape/dtype/format +
全部 attr** 去匹配 .om 的。任何一项和编 .om 时不一致都匹配不上,而报出来的是
"算子没找到(161001)" —— 极容易误判成算子没编进包。从编 .om 的那份 json 直接生成,
就没有手抄出错的余地。

输入文件名先填成占位的 input_0.bin / input_1.bin ...,按实际的改。
"""

import json
import os
import sys


def die(msg):
    print("[X] %s" % msg)
    sys.exit(1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a[2:])
    if not args:
        print(__doc__)
        return 1

    src = args[0]
    dst = args[1] if len(args) > 1 else "case.json"
    om_dir = opts.get("om-dir")
    index = int(opts.get("index", 0))

    if not os.path.isfile(src):
        die("找不到 %s" % src)
    with open(src, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            die("%s 不是合法 JSON: %s" % (src, e))

    if isinstance(doc, dict):
        doc = [doc]
    if not isinstance(doc, list) or not doc:
        die("singleop json 应该是一个数组(或单个对象)")
    if index >= len(doc):
        die("--index=%d 超出范围,这个文件里有 %d 个算子" % (index, len(doc)))
    if len(doc) > 1:
        print("[!] 这个 singleop json 里有 %d 个算子,取第 %d 个 (--index 可改)"
              % (len(doc), index))
    e = doc[index]

    op = e.get("op")
    if not op:
        die("第 %d 个条目没有 op 字段" % index)

    def conv(descs, kind):
        out = []
        for i, d in enumerate(descs or []):
            item = {
                "dtype": d.get("type", d.get("dtype")),
                "shape": d.get("shape"),
                "format": d.get("format", "ND"),
            }
            if item["dtype"] is None or item["shape"] is None:
                die("%s[%d] 缺 type 或 shape" % (kind, i))
            if kind == "inputs":
                item["file"] = "input_%d.bin" % i
            else:
                item["file"] = "output_%d.bin" % i
            out.append(item)
        return out

    case = {"op": op}
    if om_dir:
        case["om_dir"] = om_dir
    case["inputs"] = conv(e.get("input_desc"), "inputs")
    case["outputs"] = conv(e.get("output_desc"), "outputs")

    attrs = e.get("attr") or e.get("attrs") or []
    if attrs:
        case["attrs"] = [{"name": a["name"], "type": a["type"], "value": a["value"]}
                         for a in attrs]

    if not case["inputs"]:
        die("input_desc 是空的")
    if not case["outputs"]:
        die("output_desc 是空的")

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("[OK] 写出 %s" % dst)
    print("  op      = %s" % op)
    print("  inputs  = %d   outputs = %d   attrs = %d"
          % (len(case["inputs"]), len(case["outputs"]), len(case.get("attrs", []))))
    print()
    print("  接下来把 inputs[].file 改成你实际的 bin 文件名：")
    for i, d in enumerate(case["inputs"]):
        need = d["shape"]
        print("    input_%d.bin   dtype=%-9s shape=%s" % (i, d["dtype"], need))
    print()
    print("  输出默认落盘到 output_N.bin；要比精度就给 outputs[N] 加一个 \"golden\": \"xxx.bin\"。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
