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
    "float32": TensorProto.FLOAT,
    "int8": TensorProto.INT8,
    "int32": TensorProto.INT32,
    "uint64": TensorProto.UINT64,
}

# 和 CANN 自带的 onnx 解析器对得上的保守取值。被删掉的 onnx_block/build_block.py
# 记过同一个数（那份注释原话：「和 CANN 自带的 onnx 解析器对得上的保守取值」）。
IR_VERSION = 7
OPSET = 11

# 节点输入表的顺序 == IR 端口顺序：
#   x | filters[0..1] | bias[0..1] | scale[0..1] | relu_weight[0..1]
# 四个 DYNAMIC list 的实例都**按层**排（instance i 属于 conv_{i+1}），摊平之后就是
# 下面 IR_SLOT_NAMES 的顺序。
#
# **这里的名字和 IR 里的输入名无关。** IR 叫 filters/bias/scale/relu_weight，归属是
# 靠位置定的；这里沿用 case.bin 里的张量名（filter1/filter2 而不是 filters_0/1），
# 为的是让「case.bin 的 blob 名」「onnx 图的输入名」「om 的模型输入名」三者同名 ——
# 板侧按名字取下标就不需要任何映射表。
#
# 两条历史包袱，留着是因为读老 case.bin 时还会碰到：
#   * 更早的单算子 ABI 是 x, filter1, bias1, filter2, bias2, ...（filter2 在 3 号位）。
#     filters 变成 DYNAMIC list 后两个核必须相邻，filter2 挪到了 2 号位。
#   * 更早的 scale 槽序是 dequant_scale2, quant_scale1, quant_scale2，dq2 在 q1
#     **前面**。现在 scale 的 instance 0 恒是 conv1 的表、instance 1 恒是 conv2 的，
#     而 dq2 是 conv2 那张，所以它排到了后面。
IR_SLOT_NAMES = [
    "x",
    "filter1",
    "filter2",
    "bias1",
    "bias2",
    "quant_scale1",
    "quant_scale2",
    "dequant_scale2",
    "relu_weight1",
    "relu_weight2",
]
# ge::DataType 的取值（graph/types.h）。dtype 属性直接发这个数。
GE_DT_FLOAT16 = 1
GE_DT_INT8 = 2

DTYPE_FP16, DTYPE_INT8, DTYPE_S8F16, DTYPE_A16W8 = 0, 1, 2, 3

# relu 属性的取值 = 硬件的 ReluMode 枚举（kernel_utils_mode.h:54）。
# 2 = SCALAR_RELU / 3 = VECTOR_RELU 要用 relu_weight 那张斜率表，算子侧还没落地。
RELU_NO, RELU_NORMAL, RELU_SCALAR, RELU_VECTOR = 0, 1, 2, 3


def tensor_plan(spec):
    """派生算子输入实例、输出、十二个公开属性和两个私有定标值。

    四条 dtype 通路、权重的逻辑 NCHW 四元组、真实 Cout、两层各按
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
    kh2 = spec.get("kh2") or spec["kh"]
    kw2 = spec.get("kw2") or spec["kw"]
    # Logical graph inputs retain exact Kh/Kw and unpadded Cout. GE transforms
    # them to the FRACTAL_Z storage required by the operator registration.
    slots = {
        "x": (xt, [spec["n"], spec["ci"], spec["hi"], spec["wi"]]),
        "filter1": (wt, [spec["cout1"], spec["ci"], spec["kh"], spec["kw"]]),
        "filter2": (wt, [spec["cout2"], spec["cout1"], kh2, kw2]),
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
    # relu_weight 是第四个 list（照 ExtendConv2d 的 relu_weight0/1 声明的）。
    # **这一版它恒不被读**：算子侧 tiling 把用到斜率的 relu 模式（2 = 标量斜率、
    # 3 = 逐通道斜率）一律拒了。
    #
    # 但 list 恒长的要求照样成立，所以这两个槽**必须发真张量**，不能发 None ——
    # dt 是 None 的槽会被 build_model 整个跳过（那是「两个名字抢 instance 1」那个
    # 机制用的），跳过就只剩 7 个输入，而插件要求正好 9 个。
    # case.bin 里没有这两个张量，运行器按模型报的字节数现合成零缓冲。
    slots["relu_weight1"] = ("float32", [spec["cout1"]])
    slots["relu_weight2"] = ("float32", [spec["cout2"]])

    # Required attribute first, followed by optional attrs and private calibration.
    # dtype=-1 follows x; only s8f16 needs an explicit output type override.
    attrs = {
        "bias_mask": [1 if has_bias else 0, 1 if has_bias else 0],
        "dtype": GE_DT_FLOAT16 if s8f16 else -1,
        "strides": [int(spec["s1"]), int(spec["s1"]), int(spec["s2"]), int(spec["s2"])],
        "pads": [int(spec[k]) for k in ("ph1","ph1","pw1","pw1","ph2","ph2","pw2","pw2")],
        "dilations": [1] * 8,
        "groups": [1,1],
        "data_format": [0,0],
        "offset_x": [0,0],
        "round_mode": [0,0],
        "pad_mode": [0,0],
        "enable_hf32": [0,0],
        "enable_relu": [RELU_NORMAL if spec["relu1"] else RELU_NO,
                        RELU_NORMAL if spec["relu2"] else RELU_NO],
        "fixed_shift_value": [int(spec["shift1"]), int(spec["shift2"])],
    }
    return {
        "slots": [(n,) + tuple(slots[n]) for n in IR_SLOT_NAMES],
        "y": (yt, [spec["n"], spec["cout2"], spec["ho2"], spec["wo2"]]),
        "attrs": attrs,
    }


def build_model(plan, filter_tensors):
    """Build a constant-weight graph; GE folds logical filters to FRACTAL_Z."""
    graph_inputs = []
    node_inputs = []
    initializers = []
    for name, dt, shape in plan["slots"]:
        if dt is None:
            # dt 是 None 的槽直接跳过，节点输入列表里不留空洞。
            #
            # **这不是「这一层没有」的表达方式** —— 四个 list 都是恒长的。它只服务
            # 一件事：`quant_scale2` 和 `dequant_scale2` 是**同一个 instance 1 的两个
            # 名字**（出 int8 时是重量化表、出 fp16 时是反量化表），两者恒有且只有
            # 一个非 None，于是 scale list 恒好是 2 个实例。名字分开只是为了 case.bin
            # 和日志里能看出这一条通路用的是哪张表。
            #
            # 留洞那条路是实测走不通的：DYNAMIC list 内部给空串会死在 GE 自己的
            # Verify 上（我们的 infershape 根本不会被调到）。0 元素张量也走不通
            # （atc 的 --input_shape 直接拒，E10011）。
            continue
        node_inputs.append(name)
        if name in ("filter1", "filter2"):
            if name not in filter_tensors:
                raise ValueError("missing logical filter tensor " + name)
            _, dims, raw = filter_tensors[name]
            width = 2 if dt == "float16" else 1
            count = 1
            for dim in shape:
                count *= dim
            if list(dims) != shape or len(raw) != count * width:
                raise ValueError("logical filter shape/bytes differ from plan: " + name)
            initializers.append(helper.make_tensor(name, ONNX_DT[dt], shape, raw, raw=True))
        else:
            graph_inputs.append(helper.make_tensor_value_info(name, ONNX_DT[dt], shape))

    yt, yshape = plan["y"]
    graph_outputs = [helper.make_tensor_value_info("y", ONNX_DT[yt], yshape)]

    # domain 用默认域（空串，解析器归一成 ai.onnx）。插件注册的是
    # ai.onnx::<9..22>::FusedConv2d，所以 opset 换个数也还对得上。
    node = helper.make_node("FusedConv2d", node_inputs, ["y"],
                            name="fused_conv2d", domain="", **plan["attrs"])
    graph = helper.make_graph([node], "fused_conv2d", graph_inputs, graph_outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name="fc2d_onnx.py",
                              opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


def input_shape_arg(plan):
    """拼 atc 的 --input_shape。只含真正存在的输入。"""
    return ";".join("%s:%s" % (n, ",".join(str(d) for d in shape))
                    for n, dt, shape in plan["slots"] if dt is not None and n not in ("filter1", "filter2"))


def write_onnx(case_path, path):
    """Read spec and weights from the same case, save its graph, return the plan."""
    import fc2d
    spec = fc2d.read_spec(case_path)
    plan = tensor_plan(spec)
    import run_fused_conv2d as runner
    tensors, *_ = runner.load_case(case_path)
    model = build_model(plan, tensors)
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
    out = argv[1] if len(argv) > 1 else "fused_conv2d.onnx"
    plan, checker = write_onnx(argv[0], out)
    print("写出 %s" % out)
    print("  onnx.checker: %s" % checker)
    for n, dt, shape in plan["slots"]:
        print("    %-16s %s" % (n, "（不传）" if dt is None else "%-8s %s" % (dt, shape)))
    print("    %-16s %-8s %s" % ("y (输出)", plan["y"][0], plan["y"][1]))
    print("  --input_shape=%s" % input_shape_arg(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
