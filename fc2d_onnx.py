#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 case.bin 的 spec 造一张 .onnx，给 `atc --framework=5` 编 om 用。

为什么是 onnx 而不是别的：
  * `atc --singleop` 给 DYNAMIC 输入只实例化**一个**，而 filters 需要两个
    （conv1 的核 + conv2 的核）。两轮实验、2 个算子、3 个 SoC、4 种 json 键写法
    都是一个结论：1 个实例能过 FE，>=2 个过不去。
  * atc 的另一条前端（`--framework`）走的是完整 GE 图流水线，能表达多实例。
    `--mode` 只有 1/5/6 三个**输出**方向的转换，没有 json->om 的入口，
    所以 onnx 是 atc 这条路上唯一能表达多实例 DYNAMIC 的输入形态。

**形状和 dtype 只从 case.bin 的 spec 派生**，和 singleop 那条路同一个源头
（见 tensor_plan 的注释）。不要在这里引入第二套形状规则。
"""
import os
import sys

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError as e:   # noqa: F401
    raise SystemExit(
        "[X] 缺 python 包: %s\n"
        "    编 om 要先造 onnx，所以**编译机**上需要 onnx 和 protobuf：\n"
        "        pip3 install --target <某个目录> onnx protobuf\n"
        "        export PYTHONPATH=<某个目录>:$PYTHONPATH\n"
        "    **别装在 /tmp 下。** 踩过一次：装在 /tmp 的 TBE 依赖被 tmp 清理抹掉之后，\n"
        "    GE 报的是「There is no valid so about OpsKernelInfoStore or GraphOptimizer」——\n"
        "    离真正的根因（缺 numpy）隔了五层，没有日志根本猜不到。" % e)

# dtype 字符串 -> ONNX TensorProto。uint64 是 scale 的打包形态（GE 的 VDEQF16
# 通路吃的就是这个），不是真的当整数用。
ONNX_DT = {
    "float16": TensorProto.FLOAT16,
    "int8": TensorProto.INT8,
    "int32": TensorProto.INT32,
    "uint64": TensorProto.UINT64,
}

# 和 CANN 自带的 onnx 解析器对得上的保守取值。被删掉的 onnx_block/build_block.py
# 记过同一个数（那份注释原话：「和 CANN 自带的 onnx 解析器对得上的保守取值」）。
IR_VERSION = 7
OPSET = 11

# 节点输入必须按**新的 IR 顺序**排：x, filters[0], filters[1], bias1, bias2,
# dequant_scale2, quant_scale1, quant_scale2。
#
# 注意这和老的单算子 ABI 不同 —— 老的是 x, filter1, bias1, filter2, bias2, ...
# （filter2 在 3 号位）。filters 改成 DYNAMIC list 之后两个核必须相邻，
# filter2 从 3 号位挪到了 2 号位，后面三个 scale 的下标也跟着 +0 不变但 bias 往后挪。
# 槽位名沿用 case.bin 里的张量名（filter1/filter2 而不是 filters_0/filters_1），
# 这样「case.bin 里的 blob 名」和「om 的模型输入名」是同一个名字，板侧按名字取
# 下标就不需要任何映射表。
IR_SLOT_NAMES = [
    "x",
    "filter1",
    "filter2",
    "bias1",
    "bias2",
    "dequant_scale2",
    "quant_scale1",
    "quant_scale2",
]

DTYPE_FP16, DTYPE_INT8, DTYPE_S8F16, DTYPE_A16W8 = 0, 1, 2, 3


def tensor_plan(spec):
    """派生每个输入槽的 dtype/形状/有没有，外加八个属性和输出。

    这里的算法和 fc2d.singleop_json() 逐行同源 —— 四条 dtype 通路、权重的
    FRACTAL_Z 四元组、cout 向上补齐到 16、两层各按自己的核。改了一边一定要改
    另一边；更好的做法是哪天把 singleop_json 也改成调这个函数。

    返回 {"slots": [(名字, dtype 或 None, 形状)], "y": (dtype, 形状), "attrs": {...}}
    dtype 为 None 表示这个可选输入这条通路上**不传**。
    """
    mode = spec.get("dtype_mode", DTYPE_FP16 if spec["elem_bytes"] == 2 else DTYPE_INT8)
    fp16 = mode == DTYPE_FP16
    a16w8 = mode == DTYPE_A16W8
    s8f16 = mode == DTYPE_S8F16
    x_fp16 = fp16 or a16w8
    filter_fp16 = fp16
    out_fp16 = fp16 or s8f16 or a16w8
    xt = "float16" if x_fp16 else "int8"
    wt = "float16" if filter_fp16 else "int8"
    bt = "float16" if fp16 else "int32"
    yt = "float16" if out_fp16 else "int8"
    weight_c0 = 16 if filter_fp16 else 32
    mid_c0 = 16 if fp16 else 32
    # 老的 .bin 在 kh2/kw2 上是 0 —— 那时两层必然同核，退回 kh/kw 就是对的。
    kh2 = spec.get("kh2") or spec["kh"]
    kw2 = spec.get("kw2") or spec["kw"]
    fz1k = (spec["ci"] // weight_c0) * spec["kh"] * spec["kw"]
    fz2k = (spec["cout1"] // mid_c0) * kh2 * kw2

    # FRACTAL_Z 的 N 向上补齐到 16 —— cout2 可以是 2，整除会算出 0。
    slots = {
        "x": (xt, [spec["n"], spec["ci"], spec["hi"], spec["wi"]]),
        "filter1": (wt, [fz1k, (spec["cout1"] + 15) // 16, 16, weight_c0]),
        "filter2": (wt, [fz2k, (spec["cout2"] + 15) // 16, 16, weight_c0]),
        "bias1": (bt, [spec["cout1"]]),
        "bias2": (bt, [spec["cout2"]]),
    }
    # 三个 scale 按通路取舍。缺席的槽在 onnx 里是**空串输入**，不是占位张量 ——
    # 这和 singleop 的 RESERVED/UNDEFINED 占位是两套不同的约定，别混。
    if s8f16 or a16w8:
        slots["dequant_scale2"] = ("uint64", [spec["cout2"]])
        slots["quant_scale1"] = ("uint64", [spec["cout1"]])
        slots["quant_scale2"] = (None, [])
    elif mode == DTYPE_INT8:
        slots["dequant_scale2"] = (None, [])
        slots["quant_scale1"] = ("uint64", [spec["cout1"]])
        slots["quant_scale2"] = ("uint64", [spec["cout2"]])
    else:
        slots["dequant_scale2"] = (None, [])
        slots["quant_scale1"] = (None, [])
        slots["quant_scale2"] = (None, [])

    attrs = {
        "fixed_shift1": int(spec["shift1"]),
        "fixed_shift2": int(spec["shift2"]),
        "relu1": bool(spec["relu1"]),
        "relu2": bool(spec["relu2"]),
        # 两层同核时发长度 2，不同核时发长度 4 —— tiling 的 ReadKernelSize/ReadPads
        # 两种长度都认（fused_conv2d_tiling.cpp 的 ATTR_KERNEL_SIZE 那段注释）。
        "kernel_size": ([spec["kh"], spec["kw"]] if (kh2, kw2) == (spec["kh"], spec["kw"])
                        else [spec["kh"], spec["kw"], kh2, kw2]),
        "strides": [int(spec["s1"]), int(spec["s2"])],
        "pads": [int(spec["ph1"]), int(spec["pw1"]), int(spec["ph2"]), int(spec["pw2"])],
        "a16w8_shift1": 29,
    }
    return {
        "slots": [(n,) + tuple(slots[n]) for n in IR_SLOT_NAMES],
        "y": (yt, [spec["n"], spec["cout2"], spec["ho2"], spec["wo2"]]),
        "attrs": attrs,
    }


def build_model(plan):
    """按 plan 建图。

    所有张量都是**图输入**，不是 initializer —— 板上的数据来自 case.bin，
    权重烘进 om 里就喂不进去了。
    """
    graph_inputs = []
    node_inputs = []
    for name, dt, shape in plan["slots"]:
        if dt is None:
            # ONNX 表达「缺席的可选输入」就是空串。
            node_inputs.append("")
            continue
        graph_inputs.append(helper.make_tensor_value_info(name, ONNX_DT[dt], shape))
        node_inputs.append(name)

    yt, yshape = plan["y"]
    graph_outputs = [helper.make_tensor_value_info("y", ONNX_DT[yt], yshape)]

    # domain 用默认域（空串，解析器归一成 ai.onnx）。插件注册的是
    # ai.onnx::<9..22>::FusedConv2d，所以 opset 换个数也还对得上。
    node = helper.make_node("FusedConv2d", node_inputs, ["y"],
                            name="fused_conv2d", domain="", **plan["attrs"])
    graph = helper.make_graph([node], "fused_conv2d", graph_inputs, graph_outputs)
    model = helper.make_model(graph, producer_name="fc2d_onnx.py",
                              opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def input_shape_arg(plan):
    """拼 atc 的 --input_shape。只含真正存在的输入。"""
    return ";".join("%s:%s" % (n, ",".join(str(d) for d in shape))
                    for n, dt, shape in plan["slots"] if dt is not None)


def write_onnx(spec, path):
    """造图并存盘，返回 plan。"""
    plan = tensor_plan(spec)
    model = build_model(plan)
    # **onnx.checker 预期不过**：FusedConv2d 不是标准 ONNX 算子，标准类型推导
    # 认不出它。atc 的解析器直接读 protobuf、不跑 ONNX 的类型推导，照样能编 ——
    # 这一条被删掉的 onnx_block/build_block.py 已经实测过（那边是 Conv 吃 int8
    # 出 int32，同样 checker 不过而 atc 能编）。所以只当警告。
    try:
        onnx.checker.check_model(model)
        checker = "过"
    except Exception as e:  # noqa: BLE001 - checker 什么都可能抛
        checker = "不过（预期之内）: %s" % str(e).split("\n")[0][:120]
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    onnx.save(model, path)
    return plan, checker


def main(argv):
    if len(argv) < 2:
        print("用法: fc2d_onnx.py <case.bin> [out.onnx]")
        return 2
    import fc2d
    spec = fc2d.read_spec(argv[0])
    out = argv[1] if len(argv) > 1 else "fused_conv2d.onnx"
    plan, checker = write_onnx(spec, out)
    print("写出 %s" % out)
    print("  onnx.checker: %s" % checker)
    for n, dt, shape in plan["slots"]:
        print("    %-16s %s" % (n, "（不传）" if dt is None else "%-8s %s" % (dt, shape)))
    print("    %-16s %-8s %s" % ("y (输出)", plan["y"][0], plan["y"][1]))
    print("  --input_shape=%s" % input_shape_arg(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
