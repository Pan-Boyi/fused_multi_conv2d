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

**形状和 dtype 只从 case.bin 的 spec 派生**，而且 tensor_plan() 是唯一一处
派生 —— 不要在别处引入第二套形状规则。
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
# 节点输入列表的顺序。IR 是 x | filters(DYNAMIC) | bias(DYNAMIC) | quant(DYNAMIC)，
# 三个 list 的实例都**按层**排，所以摊平之后就是下面这个顺序。
#
# 槽位名沿用 case.bin 里的张量名 —— onnx 的图输入名、om 的模型输入名、case.bin
# 的张量名刻意三者同名，这样板侧按名字取下标不需要映射表。**IR 里的名字
# （filters/bias/quant）和这些名字无关**，IR 的归属是靠位置定的。
#
# 和旧 IR 的差别只有一处：dequant_scale2 从 quant_scale1 **前面**挪到了后面。
# 新约定是 quant 的 instance 0 恒为 conv1 的表、instance 1 恒为 conv2 的表，
# 而 dq2 是 conv2 的那张。
IR_SLOT_NAMES = [
    "x",
    "filter1",
    "filter2",
    "bias1",
    "bias2",
    "quant_scale1",
    "quant_scale2",
    "dequant_scale2",
]
# ge::DataType 的取值（graph/types.h）。out_dtype 属性直接发这个数。
GE_DT_FLOAT16 = 1
GE_DT_INT8 = 2

DTYPE_FP16, DTYPE_INT8, DTYPE_S8F16, DTYPE_A16W8 = 0, 1, 2, 3


def tensor_plan(spec):
    """派生每个输入槽的 dtype/形状/有没有，外加八个属性和输出。

    四条 dtype 通路、权重的 FRACTAL_Z 四元组、cout 向上补齐到 16、两层各按
    自己的核 —— 这些规则**只在这里**。上面那条废掉的 singleop 路曾经另有一份
    逐行同源的副本，改形状得同时改两处；副本已经随那条路一起删了。

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
    # **三个 list 都恒长发满，哪些是真的由属性/dtype 说。**
    # 本来想让「这一层没 bias」= list 短一截，但 **DYNAMIC 输入的实例数不能是 0**
    # —— atc 报 EZ3002「the quantity of dynamic input[bias] should be larger than
    # zero」。0 元素张量占位也走不通（E10011「Shape values must be positive
    # integers」），list 中间留空串更走不通（死在 GE 的 Verify）。三条都是实测。
    #
    # 于是：bias 恒发两个张量，bias_mask 说哪些是真的；quant 恒发两个张量，
    # 哪张表会被读由 dtype 组合决定。全 fp16 通路那两个 quant 是纯占位 ——
    # case.bin 里没有它们，运行器按模型报的字节数现合成零缓冲。
    has_bias = bool(spec["bias"])
    # quant 的 instance 0 恒是 conv1 的表、instance 1 恒是 conv2 的表。
    # instance 1 的**名字**按这条通路上的用途取：出 fp16 是反量化表、出 int8 是
    # 重量化表。全 fp16 通路两张都不读，沿用 quant_scale1/quant_scale2 这两个名字。
    slots["quant_scale1"] = ("uint64", [spec["cout1"]])
    if s8f16 or a16w8:
        slots["dequant_scale2"] = ("uint64", [spec["cout2"]])
        slots["quant_scale2"] = (None, [])
    else:
        slots["dequant_scale2"] = (None, [])
        slots["quant_scale2"] = ("uint64", [spec["cout2"]])

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
        # ---- bias/quant 合并成 DYNAMIC list 之后新增的两个 ----
        # y 的 dtype。以前是算子从「dq2 传没传」推断的，三张定标表并进 quant 之后
        # 那个判别消失了（x int8 出 int8 和 x int8 出 fp16 的输入侧完全同形），
        # 所以改成显式声明。
        "out_dtype": GE_DT_FLOAT16 if out_fp16 else GE_DT_INT8,
        # 哪些层带 bias。bit i = conv_{i+1}。bias list **恒有一层一个实例**，
        # 没被 mask 选中的那个张量算子不读（恒长的理由见上面）。
        "bias_mask": 3 if has_bias else 0,
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
            # **新 IR 里没有 OPTIONAL 输入了** —— x 是 REQUIRED，其余三个都是
            # DYNAMIC list，「没有」是靠 list 短一截表达的，不是靠空串占位。
            # 所以这里直接跳过，节点输入列表里不留空洞。
            #
            # 留洞这条路是实测走不通的：DYNAMIC list 内部给空串会死在 GE 自己的
            # Verify 上（我们的 infershape 根本不会被调到）。0 元素张量也走不通
            # （atc 的 --input_shape 直接拒，E10011）。
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
