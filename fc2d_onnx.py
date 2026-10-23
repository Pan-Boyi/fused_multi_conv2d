#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build two Conv layers and their quantization/ReLU nodes from one ABI-8 case.

ATC's installed passes form Conv2D -> ExtendConv2D -> FusedMultiConv2d.
The input graph never contains either fusion target. Only CANN's built-in
Conv/AscendRequant/AscendDequant/Relu ONNX parsers are used.
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

# Expected fused IR instances, including placeholders created by the pass.
# runtime_slots() selects activations/biases; filters/scales stay constant.
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
    自己的核；外部输入由 runtime_slots() 选择。

    返回 {"slots": [(名字, dtype 或 None, 形状)], "y": (dtype, 形状), "attrs": {...}}
    dtype=None 表示量化表别名未被此通路使用。外部输入由 runtime_slots 选择。
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
    # These entries describe the target IR, not external model inputs.
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
    # The production pass creates unused relu_weight/bias/scale as Const.
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
        "mode": mode,
        "mid": [spec["n"], spec["cout1"],
                (spec["hi"] + 2 * spec["ph1"] - spec["kh"]) // spec["s1"] + 1,
                (spec["wi"] + 2 * spec["pw1"] - spec["kw"]) // spec["s1"] + 1],
    }


def runtime_slots(plan):
    """Only activations and consumed biases are runtime inputs."""
    names = {"x"}
    for i, enabled in enumerate(plan["attrs"]["bias_mask"], 1):
        if enabled:
            names.add("bias%d" % i)
    return [(name, dt, shape) for name, dt, shape in plan["slots"] if name in names]


def constant_slots(plan):
    """Scale parameters must be Const so GE can fold SetM1Dequant on the host."""
    names = ["filter1", "filter2"]
    if plan["mode"] != DTYPE_FP16:
        names += ["quant_scale1", "dequant_scale2" if plan["mode"] in
                  (DTYPE_S8F16, DTYPE_A16W8) else "quant_scale2"]
    return names


def build_model(plan, case_tensors):
    """Build the source computation; GE alone creates both fusion targets."""
    slots = {n: (dt, shape) for n, dt, shape in plan["slots"]}
    initializers = []
    for name in constant_slots(plan):
        dt, shape = slots[name]
        if name not in case_tensors:
            raise ValueError("missing constant tensor " + name)
        actual_dt, dims, raw = case_tensors[name]
        count = 1
        for dim in shape:
            count *= dim
        width = {"float16": 2, "int8": 1, "uint64": 8}[dt]
        case_dtype = {"float16": 1, "int8": 2, "uint64": 10}[dt]
        if (list(dims) != shape or len(raw) != count * width or
                actual_dt != case_dtype):
            raise ValueError("constant dtype/shape/bytes differ from plan: " + name)
        initializers.append(helper.make_tensor(name, ONNX_DT[dt], shape, raw, raw=True))
    graph_inputs = [helper.make_tensor_value_info(n, ONNX_DT[dt], shape)
                    for n, dt, shape in runtime_slots(plan)]
    nodes, values = [], []
    current = "x"
    for layer in (1, 2):
        i = layer - 1
        shape = plan["mid"] if layer == 1 else plan["y"][1]
        out_dt = "float16" if plan["mode"] == DTYPE_FP16 or (
            layer == 2 and plan["mode"] in (DTYPE_S8F16, DTYPE_A16W8)) else "int8"
        inputs = [current, "filter%d" % layer]
        if plan["attrs"]["bias_mask"][i]:
            inputs.append("bias%d" % layer)
        p = plan["attrs"]["pads"][i * 4:i * 4 + 4]
        # Case/Fused order T,B,L,R; ONNX Conv order T,L,B,R.
        conv_out = "conv%d_acc" % layer
        nodes.append(helper.make_node("Conv", inputs, [conv_out], name="conv%d" % layer,
                    kernel_shape=slots["filter%d" % layer][1][2:],
                    strides=plan["attrs"]["strides"][i * 2:i * 2 + 2],
                    pads=[p[0], p[2], p[1], p[3]], dilations=[1, 1], group=1))
        values.append(helper.make_tensor_value_info(conv_out, ONNX_DT[
            "float16" if plan["mode"] == DTYPE_FP16 else "int32"], shape))
        current = conv_out
        if plan["mode"] != DTYPE_FP16:
            scale = "dequant_scale2" if layer == 2 and out_dt == "float16" else "quant_scale%d" % layer
            op = "AscendDequant" if out_dt == "float16" else "AscendRequant"
            attrs = {"relu_flag": 0}
            if op == "AscendDequant":
                attrs.update(dtype=GE_DT_FLOAT16, sqrt_mode=0)
            current = "conv%d_scaled" % layer
            nodes.append(helper.make_node(op, [conv_out, scale], [current],
                                          name="scale%d" % layer, **attrs))
            values.append(helper.make_tensor_value_info(current, ONNX_DT[out_dt], shape))
        if plan["attrs"]["enable_relu"][i]:
            output = "mid" if layer == 1 else "y"
            nodes.append(helper.make_node("Relu", [current], [output], name="relu%d" % layer))
            current = output
        else:
            # Rename the producer output; do not insert Identity between convolutions.
            output = "mid" if layer == 1 else "y"
            nodes[-1].output[0] = output
            values[-1].name = output
            current = output
        if layer == 1 and plan["attrs"]["enable_relu"][i]:
            values.append(helper.make_tensor_value_info(current, ONNX_DT[out_dt], shape))
    yt, yshape = plan["y"]
    values = [value for value in values if value.name != "y"]
    graph = helper.make_graph(nodes, "conv_pair", graph_inputs,
                [helper.make_tensor_value_info("y", ONNX_DT[yt], yshape)],
                initializer=initializers, value_info=values)
    model = helper.make_model(graph, producer_name="fc2d_onnx.py",
                             opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    from fc2d_fusion import check_input_model
    check_input_model(model)
    return model


def input_shape_arg(plan):
    return ";".join("%s:%s" % (n, ",".join(str(d) for d in shape))
                    for n, dt, shape in runtime_slots(plan))


def write_onnx(case_path, path):
    import fc2d
    import run_fused_multi_conv2d as runner
    plan = tensor_plan(fc2d.read_spec(case_path))
    tensors, *_ = runner.load_case(case_path)
    model = build_model(plan, tensors)
    # AscendRequant/Dequant use CANN's built-in ONNX extensions. Unknown-op
    # schemas are expected for those; standard fp16 graphs must pass the checker.
    if plan["mode"] == DTYPE_FP16:
        onnx.checker.check_model(model)
        checker = "passed"
    else:
        checker = "CANN quantized Conv/Ascend quantization extensions; validated structurally"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    onnx.save(model, path)
    return plan, checker


def main(argv):
    if len(argv) < 2:
        print("用法: fc2d_onnx.py <case.bin> [out.onnx]")
        return 2
    out = argv[1] if len(argv) > 1 else "fused_multi_conv2d.onnx"
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
