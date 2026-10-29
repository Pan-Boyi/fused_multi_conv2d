# fused_multi_conv2d —— FusedMultiConv2d @ 5102 的上板验证

**改一个 json（`cases.json`）就能换形状**，别的地方不用动。

`dtype` 有五个取值：`fp16` / `int8` / `s8f16` / `a16w8` / `both`。`both` 展开成 fp16 + int8
两条（名字加 `_fp16` / `_int8` 后缀）。**`s8f16` 是「int8 进 / fp16 出」那条**：
conv1 用 `quant_scale1[Cout1]` 的 per-channel VREQ8 出 int8 mid，conv2 用
`dequant_scale2[Cout2]` 按通道反量化成 fp16。两张表都是 uint64，所以它不在
`both` 里面，要显式写。int8→int8 则传 `quant_scale1[Cout1]` +
`quant_scale2[Cout2]`；纯 fp16 不传 scale 表。

`a16w8` 是混合通路：`x` 为 fp16，两张 filter 为 int8，conv1 通过
`quant_scale1[Cout1]` 得到 int8 mid，conv2 再通过 `dequant_scale2[Cout2]` 输出 fp16。
case generator 根据 `fixed_shift_value[0]` 缩放输入，并保持与精确整数参考一致，因此板侧
结果可与精确整数 golden 逐位比对。

两个 C++ 小工具（`gen_case` / `fc2d_geom`）**不进仓**，脚本用到时自己编 ——
缺了就编，源码或任何一个头比二进制新也重编。要 `g++`，不要 CANN，不要设备。

```bash
# 1) 看看清单里有哪些形状、算子能不能服务（不动任何文件）
python3 fc2d.py cases.json --list
python3 fc2d.py cases.json --steps check

# 2) 在有 CANN 的机器上：生成 case 数据 + 编离线模型
source <CANN>/set_env.sh
python3 fc2d.py cases.json --steps check,case,om --soc <soc_version>

# 3) 把整个目录拷到有 5102 的机器上，跑
python3 fc2d.py cases.json --steps run
```

产物默认在 `out/<json 名>/<case 名>/`：`case.bin`（输入 + golden）、`.onnx`、`om/*.om`、`om/*.om.fc2d.json`。

## Python golden

`fc2d.py` 和 `run_profile.py` 的 `case` 步骤默认使用
`fused_multi_conv2d_golden.py` 计算最终参考输出。生成端需要 NumPy；板侧仍只读取
`case.bin` 中的参考数据，无须安装 NumPy。

现有 C++ 生成器保留随机输入、量化表选择及定标工作点检查。随后 Python 从实际的
`x`、逻辑 NCHW filter、bias 和逐通道 scale 独立重算两层输出，替换 `y_expect`、
FP16 的 `y_exact` 及输出统计。已有参考数据不参与 Python 计算，shift 和 scale 不变。
`case.bin.golden.json` 记录 Python 脚本、完整用例和输出的 SHA256，以及与原 C++ 参考的比较结果。

| 通路 | Python 参考计算 |
| --- | --- |
| FP16 | double 卷积累加；卷积和 bias 分别按 `2^(58-S)` 最近偶数舍入，限幅 int32 后反定标为 FP16；中间结果为 FP16。 |
| INT8 | 精确整数卷积，按逐通道 scale 重定标、最近偶数舍入及 INT8 饱和；中间结果为 INT8。 |
| INT8→FP16 | 第一层同 INT8，第二层先将 int32 转为 float32，再乘逐通道反量化系数并转换为 FP16。 |
| A16W8 | 按 `x × 2^(29-S1)` 恢复现有用例的精确 INT8 输入网格，再执行整数卷积、INT8 中间结果及 FP16 出口。 |

量化和反量化系数均按原参考清除 float32 的低 13 位尾数，ReLU 保持原有负零语义。
A16W8 沿用现有 CPU golden 的输入范围：恢复后的值必须为 `[-128,127]` 内的整数。
FP16 的 `y_exact` 也沿用原实现，使用第一层定点 FP16 中间结果，仅在第二层出口省略定点落格。

上板验证命令保持不变，重新执行包含 `case` 的流程即可使用 Python golden：

```bash
python3 run_profile.py --cases fusion_cases.json --only fusion_a16w8 \
  --steps check,case,om,push,prof --no-msprof

# 同时比较两个 ExtendConv2D 与 Python golden
python3 run_profile.py --cases fusion_cases.json --only fusion_a16w8 \
  --steps check,case,om,push,prof --no-msprof \
  --compare-unfused --compare-reference-golden
```

也可从已有用例独立导出 NCHW 参考数据，不需要 CANN、设备或 C++ 编译器：

```bash
python3 fused_multi_conv2d_golden.py out/fusion_cases/fusion_a16w8/case.bin \
  --output y_python.bin --workspace-mb 64
```

独立脚本也可按输入、两张卷积核及输出的数据类型直接读取 JSON，示例见
`golden_tensor_cases.json`；只需复制脚本和 JSON 到任意目录，并安装 NumPy：

```bash
python3 fused_multi_conv2d_golden.py golden_tensor_cases.json \
  --only golden_a16w8 --output y_python.bin
```

```json
{
  "name": "golden_a16w8",
  "input": {"shape": [1, 32, 8, 9], "dtype": "fp16"},
  "kernels": [
    {"shape": [64, 32, 3, 3], "dtype": "int8"},
    {"shape": [2, 64, 3, 3], "dtype": "int8"}
  ],
  "output": {"dtype": "fp16"},
  "attrs": {
    "strides": [1, 1, 1, 1],
    "pads": [0, 0, 0, 0, 0, 0, 0, 0],
    "enable_relu": [true, true],
    "bias_mask": [true, true],
    "fixed_shift_value": [13, 0]
  }
}
```

`input.shape` 为 `[N,C,H,W]`；每张 `kernels[i].shape` 为 `[Cout,Cin,Kh,Kw]`，
第二层 `Cin` 必须等于第一层 `Cout`。dtype 支持 `fp16`（或 `float16`）和 `int8`，
忽略大小写，两张卷积核的 dtype 必须相同。输出形状自动推导；也可填写 `output.shape`
校验期望形状。根据 dtype 组合自动选择现有四种通路：

| input.dtype | 两张 kernel.dtype | output.dtype | 通路 |
| --- | --- | --- | --- |
| fp16 | fp16 | fp16 | FP16 |
| int8 | int8 | int8 | INT8 |
| int8 | int8 | fp16 | S8F16 |
| fp16 | int8 | fp16 | A16W8 |

`attrs` 可省略，各项默认值及含义如下：

| 属性 | 默认值 | 含义 |
| --- | --- | --- |
| strides | `[1,1,1,1]` | `[H1,W1,H2,W2]`，每层 H/W 相等。 |
| pads | `[0,0,0,0,0,0,0,0]` | `[上1,下1,左1,右1,上2,下2,左2,右2]`，每层上下、左右分别对称。 |
| enable_relu | `[true,true]` | 每层 ReLU 开关，也接受 0/1。 |
| bias_mask | `[false,false]` | 每层 bias 开关，支持独立使能。 |
| fixed_shift_value | FP16 `[42,42]`；A16W8 `[13,0]`；其余 `[0,0]` | 沿用现有 CPU 定点模型。 |

`dilations`、`groups`、`offset_x`、`enable_hf32` 仅支持默认值：分别为八个 1、
`[1,1]`、`[0,0]`、`[false,false]`。`data_format`、`round_mode`、`pad_mode` 仅支持
`[0,0]`，也接受对应的默认名称列表 `["NCHW","NCHW"]`、`["rint","rint"]`、
`["SPECIFIC","SPECIFIC"]`。其他配置会明确报错。

JSON 支持单条用例对象、用例数组及 `{"cases": [...]}`。单条可省略 `--only`，多条需要
指定名称。旧 flat JSON 格式仍可读取，其中 `dtype: "both"` 按原规则展开为
`<name>_fp16`、`<name>_int8`。

JSON 模式在脚本内部生成确定性输入、bias 和 scale，再计算参考，无需 `fc2d.py`、
其他本地 Python 文件、C++ 生成器、编译器或 CANN；生成数据和参考结果沿用原生成器的规则。
原 `case.bin` 输入方式、case/OM 产物及上板验证流程保持不变。
新张量描述格式仅用于独立 golden 入口；`fc2d.py` 和 `run_profile.py` 继续使用原 JSON 格式。

原有精度阈值、前 50 个差异表格及总 Precision 百分比保持不变。

## 两阶段融合验证

源 ONNX 只构造两个连续的 `Conv`，两张 filter 保存真实 NCHW 形状并作为 initializer。
实际使用的 uint64 scale 表也直接取自 case，保存为 initializer。
按用例加入 `AscendRequant`、`AscendDequant` 和 `Relu`，由已安装 CANN 中的原始 Pass 完成：

```text
Conv + [重定标/反量化] + [Relu] -> Conv + [重定标/反量化] + [Relu]
                         ↓ 一阶段
             ExtendConv2D -> ExtendConv2D
                         ↓ 二阶段
                  FusedMultiConv2d
```

harness 不创建 ExtendConv2D/FusedMultiConv2d 源节点，不依赖 FusedMultiConv2d 的 ONNX
解析插件，也不实现或修改融合 Pass。量化节点使用 CANN 内置的 ONNX 解析器。
一阶段显式启用 `Conv2DFixPipeToExtendConv2DFusionPass`；
二阶段使用 [canndev PR 82567](https://gitcode.com/cann/canndev/pull/82567/diffs)
的 `ExtendConv2DPairToFusedMultiConv2DFusionPass`，编译时显式打开对应开关。

`om` 步骤只编译一次模型，同时开启两个 Pass。在同一次 ATC 编译中依次检查：

1. 第一轮融合后的 GE 图快照中恰好有两个 ExtendConv2D，且 y0 直接连接下一节点的 x。
2. 同一进程、同一图的后续快照中只有一个 FusedMultiConv2d，不残留独立卷积、量化或 ReLU。
3. 最终 OM 通过 `atc --mode=1` 导出 JSON，独立确认融合结果和运行输入。此步骤仅导出，不编译模型。
4. 捕获本次编译的 native tiling，核对真实几何、bias、ReLU、blockDim，以及 FP16/A16W8 使用的定标值。
5. 全部通过后写 schema **6** 的绑定，记录常量 weight/scale 契约及 case/OM、源 ONNX、两个阶段快照、最终图和 tiling 的哈希。

不再关闭二阶段编译 `stage1.om`，因此不会为了观察中间图而编译 ExtendConv2D 内核。
每次编译在 case 目录创建独立的 `fusion_*` 目录，保存 `fusion_switch.json`、`atc.log`、
`graphs/`、`model_graph.json`、`tiling_capture/` 和 `fusion_verification.json`。
失败时保存 `failure_report.json`，保留原始日志和快照，不签发可运行绑定。
schema 1–5 的模型需要重编；case ABI 仍为 **8**，精度阈值及 profiling 步骤保持不变。

量化融合会生成 `SetM1Dequant`，把源 scale 转成 Fixpipe 所需位域。它有 host 实现，
由 GE 在编译期常量折叠；把 scale 作为运行时 Data 会使该辅助节点无法消除，并触发运行时实现缺失。
因此中间快照允许 `Const -> SetM1Dequant`，最终 OM 必须已经消除该节点。
这与 ops-nn 的 [正式 GE 示例](https://gitcode.com/cann/ops-nn/blob/e8642461f87897148023868b00bb528d34ca934a/conv/fused_multi_conv2d/examples/test_geir_fused_multi_conv2d.cpp)
使用 Const scale 的构图方式一致。harness 不自行生成或替代该辅助节点。

图快照使用官方 `DUMP_GE_GRAPH=2`、`DUMP_GRAPH_LEVEL=1`、`DUMP_GRAPH_FORMAT=ge_proto`、
`DUMP_GRAPH_PATH` 配置，递归读取 `pid_*` 目录下的 ModelDef protobuf text。
进程退出后恢复父环境；编译期间清除会覆盖 dump 目录的 `ASCEND_DUMP_PATH` 和 `NPU_COLLECT_PATH`。
参见 [CANN 图快照配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0012.html)。

本次核对的 Pass 源码为 canndev `4b47d70a0445542c92f16865d0b6c039e65a2a59`。
第一阶段为 FP16 设置私有 shift=42，为 FP16×INT8 设置 shift=13；第二阶段传递每层值。
用例中的 `fixed_shift_value` 是 golden 的预期工作点，不会作为源 ONNX 属性发送：

| 通路 | harness 默认预期 | 实际检查 |
| :-- | :-- | :-- |
| FP16 | `[42,42]` | 两项 |
| A16W8 | `[13,0]` | 第一项 |
| INT8 / S8F16 | `[0,0]` | 不使用 shift |

显式填写的预期值会保留；若与安装包实际产生的值不同，native tiling 校验失败。
A16W8 的 CANN 混合 Conv2D 使用定点语义：第一层将 `x × 2^(29−S)` 转为整数后计算，
S=13 时为 `x × 2^16`。golden 和 CPU 参考按这一契约验证，不把它解释为标准浮点 ONNX Conv。
新版二阶段支持上下、左右分别对称的 padding，权重重排由原始 CANN 流程完成。

先使用四条小用例检查服务器 A 上的新 CANN 包：

`run_profile.py` 会 source `profile.json.local_cann_env`，该路径必须指向包含上述 Pass 的 CANN。
编译环境中同时加载当前 ops-nn 算子包的 `set_env`。

```bash
python3 run_profile.py profile.json --cases fusion_cases.json \
  --only fusion_int8 --steps check,case,om
python3 run_profile.py profile.json --cases fusion_cases.json \
  --steps check,case,om,push,prof --no-msprof --keep-going
```

本地不具备新 CANN 时，仍可运行 `python3 -m unittest discover -s tests -v` 以及
`python3 fc2d.py fusion_cases.json --steps check,case`；真实两阶段 OM 和实板结果需要在 A/B 验证。
测试中的 synthetic OM 和 ACL 桩只验证框架链路，不能作为融合或设备精度证据。

编译失败时，`failure_report.json` 根据本次实际图和 tiling 区分：一阶段未命中、
二阶段未命中、已融合但尚无 native tiling、已完成 native tiling 的内核编译失败。
`SetM1Dequant` 仅作为日志诊断记录，不用于单独判断根因。
若 FP16 层没有 ReLU 等 Fixpipe 运算，是否命中一阶段仍以真实图为准。

## 以融合前设备输出验证精度

添加 `--compare-unfused --no-msprof`，用第二阶段融合前两个 ExtendConv2D 的实际设备输出作为 golden：

```bash
python3 run_profile.py --cases fusion_cases.json --only fusion_a16w8 \
  --compare-unfused --no-msprof --steps check,case,om,push,prof
```

同一份 ONNX 编译为两个模型：`om/` 中两个 Pass 均开启，得到一个 FusedMultiConv2d；
`reference/` 中第一阶段开启、第二阶段关闭，最终图必须保留两个直接连接的 ExtendConv2D。
参考编译前，harness 临时从当前 CANN 的 `support_fusion_pass.json` 架构名单中移除
`ExtendConv2DPairToFusedMultiConv2DFusionPass`；编译、导图和检查结束后，恢复原始字节、
所有者和权限，恢复成功才生成参考模型绑定。正常融合模型先在完整名单下编译。
该方法适用于忽略 `fusion_switch_file` 关闭请求、仍强制执行第二阶段的安装包。

配置默认按当前 `ASCEND_OPP_PATH`、`ASCEND_HOME_PATH` 和 ATC 路径定位，软链接按真实文件去重。
也可显式传 `--fusion-pass-config /path/to/support_fusion_pass.json`，
或设置 `FC2D_SUPPORT_FUSION_PASS_JSON`。不会跨多个 CANN 安装修改配置。
若 SDK 属于当前用户但安装目录只读，原子替换时临时增加目录写权限，随后恢复原权限；
没有相应权限时直接报错，不自动使用 sudo。

所有 harness 编译共享配置锁。参考编译失败、Ctrl+C 或 SIGTERM 时会先结束 ATC，再恢复名单；
SIGKILL 或机器掉电无法立即执行恢复，下一次 harness 编译会先恢复遗留事务。
原文件备份和事务保存在 `${XDG_STATE_HOME:-~/.local/state}/fused_multi_conv2d/fusion_registry/`。
如果文件在此期间被外部改动，保留备份并报错，后续编译不会在未恢复的配置下继续。
该锁只协调使用同一用户状态目录的 harness；参考编译期间应避免另行启动独立 ATC。

两个模型固定同一份权重、scale 和源图，绑定同一 case；基准模型不会退回 CPU golden。
此模式验证第二阶段融合等价性，不验证第一阶段把 Conv 改为 ExtendConv2D 的等价性。

板侧先运行参考模型，再运行融合模型，逐字节使用同一份 x 和 bias。
每个模型核对实际输出 dtype、NCHW 形状及字节数；执行失败或输出未写完整时不接受结果。
INT8 要求逐位相同；FP16 默认相对误差不超过 `1e-3`，参考值为零时要求实际值也为零，
全部元素均须通过。该模式不使用 `RATIO_MIN` 放宽通过比例，原有 CPU golden 保留不变。

与 CPU golden 比较和与未融合设备输出比较，两种模式都会打印全量数据中相对误差
`<= 0.001` 的数量、总数及百分比。前 50 个差异统一使用六列表格：
`index | position | Output | Reference | Absolute Error | Relative Error`，
其中 position 为 NCHW 坐标，表后打印总 `Precision` 百分比。
该固定精度统计独立于 `REL_TOL`；参考值为零时要求输出也为零。
前 50 个差异按展平顺序选取，包含在容差内但位模式不相同的元素。

如需同时检查两个 ExtendConv2D 与 CPU golden 函数生成的 `y_expect` 是否一致，添加
`--compare-reference-golden`：

```bash
python3 run_profile.py --cases fusion_cases.json --only fusion_a16w8 \
  --compare-unfused --compare-reference-golden --no-msprof --steps push,prof
```

此开关使用同一次参考模型执行的实际输出，无须重新编译 OM。
报告额外保存 `reference_vs_golden` 和 CPU golden 摘要，两项比较均通过时整体才通过；
未开启时，融合前后比较的通过判据仍只使用设备参考输出。
板侧 runner 同样支持 `--reference-om reference.om --compare-reference-golden`，
本机 `fc2d.py` 使用同名开关并配合 `--compare-unfused`。

结果自动拉回本地 case 目录的 `comparison_<本次运行标识>/comparison/`：

| 文件 | 内容 |
| :-- | :-- |
| `reference.bin` | 两个 ExtendConv2D 的实际输出，作为本次 golden |
| `fused.bin` | FusedMultiConv2d 的实际输出 |
| `comparison.json` | 位差数量、超差数量、0.001 达标统计、前 50 个差异、最大误差及首个超差坐标，以及输入、模型和输出哈希 |

数值失败仍拉回输出和报告，命令返回失败。单个模型执行失败时保存错误及已完成模型的输出。
上述精度对比模式不启用 profiling；性能对比使用下节的独立开关。

只重新编译两个模型可使用 `--steps om`；已有成对模型时可使用 `--steps push,prof`。
本机直接执行使用 `fc2d.py ... --compare-unfused --steps case,om,run`，或给板侧 runner 传入：

```bash
python3 run_fused_multi_conv2d.py case.bin FusedMultiConv2d 0 fused.om \
  --reference-om reference.om --comparison-output comparison
```

## 对比融合前后上板耗时

添加 `--compare-performance`，在同一块设备上执行两个 ExtendConv2D 和一个
FusedMultiConv2d，并使用 msprof 的算子耗时比较性能：

```bash
python3 run_profile.py --cases fusion_cases.json \
  --compare-performance --keep-going \
  --steps check,case,om,push,prof,pull,export,parse
```

此开关自动生成并运行融合、未融合两份 OM，同时检查输出精度，不需要额外添加
`--compare-unfused`。已有成对模型时可从 `push` 开始；本地已有本次 profiling 数据时，
可仅执行 `--compare-performance --steps parse`。性能模式要求启用 profiling，
不与 `--no-msprof` 一起使用。

两份模型在同一个 profiling 进程内依次执行。统计排除 `remote.warmup` 次预热和用于检查
输出完整性的额外执行，仅保留 `remote.repeat` 次正式执行。每次未融合执行的耗时为
两个 ExtendConv2D 算子耗时之和，最终分别取正式执行的中位数。

结果使用简洁表格展示：

| Case | Shape | Fused (us) | Extend total (us) | Saved (us) | Improvement (%) |
| :-- | :-- | --: | --: | --: | --: |
| 用例名 | 输入及两层卷积形状 | 融合耗时 | 两个 Extend 耗时之和 | 未融合 − 融合 | 节省耗时 / 未融合耗时 × 100 |

精度检查或上板执行失败的用例不进入性能结果。这里比较的是设备算子耗时，
包含算子内部搬运，不计模型加载、主机调用和算子之外的输入输出拷贝。
汇总同时保存为 `out/<配置名>/performance_comparison.json` 和
`performance_comparison.csv`；配置了 `outdir` 时保存在该目录。
表中形状取自本次模型绑定的 `case.bin`，JSON 保留完整 case 参数；恢复解析时使用
采集时记录的预热和正式执行次数。

添加 `--pipeline-details` 可查看同一用例中 Fused、第一层 Extend、第二层 Extend
及两层 Extend 合计的流水耗时。该开关自动启用 `--compare-performance`：

```bash
python3 run_profile.py --cases fusion_cases.json --only fusion_a16w8 \
  --pipeline-details --steps check,case,om,push,prof,pull,export,parse
```

每个用例打印 `Total、Cube、Scalar、MTE1、MTE2、MTE3、Fixpipe`，单位为微秒。
数据来自 `op_summary.csv`，Cube 对应 `mac_exe_time(us)`，同时支持 `aic_mac_time(us)`
等 AIC 耗时列；AIV 指标不会混入该表。统计使用相同的正式执行轮次，
排除预热和末次写出探针后取中位数；两层 Extend 的每项合计先逐轮相加再取中位数。
流水之间可以重叠，各项相加不等于 Total。未采集或不完整的流水数据显示 `N/A`。
流水指标及各轮样本保存在 `performance_comparison.json` 的 `pipeline_details` 中。
同时，`performance_comparison.csv` 保留原有列，并追加第一、第二层 Extend 总耗时
（`extend1_us`、`extend2_us`），以及 `fused`、`extend1`、`extend2`、`extend_total`
各自的 `cube_us`、`scalar_us`、`mte1_us`、`mte2_us`、`mte3_us`、`fixpipe_us` 中位数列。
列名以角色为前缀，例如 `fused_scalar_us`、`extend_total_mte2_us`。

同目录的 `pipeline_samples.csv` 保存每个用例、每个角色、每轮正式执行的 Total 和六条流水耗时，
包含 `case`、`shape`、`operator`、`op_name`、`sample_index`、`total_us` 及上述六个流水列。
`operator` 使用上述四种角色，`op_name` 记录图中节点名称，`sample_index` 从 1 开始，
不含预热和写出探针。CSV 中缺失指标留空，实际测得的 0 保留为 0；耗时单位均为微秒。
去掉 `--only` 即可批跑所有用例，加入 `--keep-going` 可在失败后继续。
每次解析覆盖当前批次的 CSV；未启用流水明细时，汇总仍为原有六列，逐轮文件仅保留表头。
已有成对模型的 profiling 数据时，可使用同一开关加 `--steps parse` 重新解析，无需上板重跑。

## 上板 + profiling：run_profile.py

`fc2d.py` 假设你在同一台机器上出 `.om` 又跑。真实情况常常是两台：一台有 CANN 和
`atc`，另一台插着 5102。`run_profile.py` 覆盖这种情况,并且把 profiling 一起做了。

```bash
# 远端登录。profile.json 里 remote.auth = "password"：那台机器只认密码，公钥禁掉了。
# 用 read 是为了不让密码进 shell 历史；还需要本机有 sshpass（apt/yum install sshpass）。
read -s -p '远端密码: ' FC2D_REMOTE_PASSWORD && export FC2D_REMOTE_PASSWORD
python3 run_profile.py profile.json
```

八个步骤,`--steps` 可以只挑其中几个:

| 步骤 | 在哪跑 | 做什么 |
| :-- | :-- | :-- |
| `check` | 本地 | 形状预检 |
| `case` | 本地 | 生成 `case.bin` |
| `om` | 本地 | 生成双 Conv ONNX，单次编译验证两阶段融合和 native tiling，写 schema-6 binding |
| `push` | 本地 -> 远端 | 每条 case 一个干净目录,拷 `case.bin`、`.om`、`.om.fc2d.json`、`run_fused_multi_conv2d.py` |
| `prof` | 远端 | `$MSPROF bash <runner wrapper>`，分别记录 runner 与 profiler 返回码，并找出**本次**新生成的 `PROF_*`。`$MSPROF` 是 `profile.msprof` 给死的绝对路径(缺省 `/var/msprof`),**不搜 PATH** —— 远端那个 ssh 是非交互 shell,读不到 `~/.bashrc` 里 source 的 `set_env.sh`,搜 PATH 会说「没有 msprof」而其实装着。指向 msprof 那个可执行文件本身。 |
| `pull` | 远端 -> 本地 | 把那个 `PROF_*` 拉回 `prof_out/<json 名>/<case 名>/` |
| `export` | 本地 | `TbeWorkTestSuit` 的 `msprof.py export summary` |
| `parse` | 本地 | 解析 `op_summary_*.csv`,每条打明细,最后打一张横向对比表 |

几个不显眼但要紧的点:

- **`run_fused_multi_conv2d.py` 每次都重新拷过去。** 它和 `case.bin` 的格式是配套的,
  远端留着一份旧的会报「case 文件版本 N,本脚本认 M」,而那时人往往已经在查算子了。
- **图模型有 1 或 3 个运行输入。** x 必有；启用 bias 时增加 bias1/bias2。
  两张 filter 和实际使用的 scale 固定在 OM 内，未使用的 list 实例由原始融合 Pass 创建为 Const。
  目标算子仍有 9 个输入实例，不需要外部零缓冲。
- **上传和执行前都校验完整 case/OM SHA256。** prof-only 也会核对远端产物是否
  等于本地选中的 binding；旧模型或修改过的 case 必须重新编译 OM。
- **每条 case 在远端使用独立目录。** 上传 case、唯一 OM、sidecar 和配套 runner。
- **新 `PROF_*` 靠执行前后的差集找,不靠"取最新"。** 目录名里带时间戳,并发或重跑
  时"最新"会拿错。一次冒出多个也会直接报错而不是随便挑一个。
- **执行失败仍可拉回、导出和解析 PROF，但命令返回失败。** runner RC 与 msprof RC
  独立保存；失败数据不进入性能对比。没有新的 PROF 或一次生成多个 PROF 也算失败。
- **`REPEAT` 次会得到多行,汇总表取最小值。** host 侧抖动和别的进程抢核只会让某几次
  变慢,不会让它变快,所以最小值是对"这个 kernel 本身有多快"掺杂噪声最少的估计。
  行数和均值也一并给出,好判断抖动有多大。
- **密码不写进 json。** 那份 json 是要进仓的。密码从 `remote.password_env` 指定的
  环境变量读,用 sshpass -e 传（不是 -p —— 那会把密码放进命令行，ps 一下就看见）。
  方式由 remote.auth 定：password / key / auto。只认密码的机器要写 password，
  auto 在那种机器上会不声不响地退回密钥然后失败。
  凭据只在真要连远端时才查,--steps check,case,om 不设密码也能跑。

`profile.json` 比 `cases.json` 多三段:

```json
{
  "soc_version": "MC62CM12AA",
  "local_cann_env": "/home/p84341448/Ascend-52/cann/set_env.sh",
  "remote": {"host": "...", "user": "...", "dir": "/home/p84341448/fc2d_run",
             "device": 0, "repeat": 10},
  "profile": {"tbe_work_test_suit": "/home/p84341448/TbeWorkTestSuit"},
  "cases": [ ... 和 cases.json 完全一样的写法 ... ]
}
```

`local_cann_env` 会被 source 进脚本自己的环境,所以调用前不必先 source。
只想验功能不想 profiling 就加 `--no-msprof`,它会自动跳过 `pull` / `export` / `parse`。
想看远端到底会执行什么,加 `--print-remote-script`。

产物按 json 的文件名分目录:`out/<json 名>/<case 名>/` 和 `prof_out/<json 名>/<case 名>/`。
两份 json 里同名但形状不同的 case 因此不会互相覆盖。

## 为什么是一个 json 而不是三处

case.bin 保存逻辑输入、golden 和属性 spec。ONNX 从它派生两个 Conv 的计算、
逻辑 NCHW 常量权重、常量 scale 和必要的 bias 输入。GE 负责布局转换和两阶段融合，
harness 在写绑定前核对最终图和真实 tiling。运行器在加载 ACL 前校验 schema-6
`<model.om>.fc2d.json` 中完整 case 与 OM 的 SHA256 以及融合证据摘要。

```text
case 配置 -> gen_case -> ABI 8 case.bin -> 双 Conv ONNX
  -> ATC 原始两阶段融合 -> 最终 OM + 验证证据 + binding -> 上传 -> runner
```

修改 case 的形状、dtype、weights、golden 或属性后都要重新生成 ONNX/OM/binding。
历史 cases.json 与 profile.json 的旧 stride/pad 已显式填写，保持原有板测几何；
新增 case 省略时使用 strides=[1,1,1,1]、pads=[0,0,0,0,0,0,0,0]。

## 恢复 profiling 阶段

每次 `prof` 在本地 case 目录保存 `execution_<uuid>.log` 和 `profile_state.json`，
包含合并的 stdout/stderr、runner RC、profiler RC、SSH RC、PROF 名和 case/OM binding。
`pull/export/parse` 可分别执行，会读取该状态并重新核对当前 case/OM：

```bash
python3 run_profile.py profile.json --only base_int8 --steps push,prof
python3 run_profile.py profile.json --only base_int8 --steps pull
python3 run_profile.py profile.json --only base_int8 --steps export
python3 run_profile.py profile.json --only base_int8 --steps parse
```

`pull` 使用已记录的远端目录，`export/parse` 使用已下载的本地 PROF；缺少前置状态、
PROF 或 binding 不匹配时明确失败。`export/parse` 不连接远端。重新执行 `prof` 会
创建新日志并替换状态，之前的日志仍保留。失败运行的分析也保持非零返回码。

## cases.json 怎么写

```json
{
  "device": 0,
  "cases": [
    {"name": "base", "dtype": "both", "ci": 32, "hi": 288, "wi": 112, "cout1": 64, "cout2": 96},
    {"name": "k5x5", "dtype": "int8", "ci": 32, "hi": 64, "wi": 64, "cout1": 32, "cout2": 32,
     "kernel": [5, 5], "pads": [2,2,2,2,2,2,2,2], "bias": true}
  ]
}
```

| 字段 | 缺省 | 说明 |
| :-- | :-- | :-- |
| `name` | `caseNN` | 产物落在 `out/<json 名>/<name>/` |
| `dtype` | `fp16` | `fp16` / `int8` / `s8f16` / `a16w8` / `both`（`both` 展开成 `<name>_fp16` 和 `<name>_int8`）|
| `n` `ci` `hi` `wi` | 1 / 32 / 288 / 112 | 输入形状 NCHW |
| `cout1` `cout2` | 64 / 96 | 两层的输出通道 |
| `kernel` | `[3, 3]` | `[kh, kw]`（两层同核）或 `[kh1, kw1, kh2, kw2]`（两层各一个核，比如 `[1,1,3,3]` 是 1x1 接 3x3） |
| `strides` | `[1,1,1,1]` | `[H1,W1,H2,W2]`，每层 H=W |
| `pads` | 八个 0 | `[T1,B1,L1,R1,T2,B2,L2,R2]`，每层上下和左右分别对称；高度与宽度可不同 |
| `bias` | `false` | 带不带 bias（两层一起）|
| `enable_relu` | `[1,1]` | 每层一个 ReluMode，目前仅支持 0/1 |
| `fixed_shift_value` | `[0,0]` | 统一私有定标；fp16 使用两项，A16W8 仅使用第一项 |
| `data_format` | `[0,0]` | 每层格式编号；当前只支持 0=NCHW |
| `round_mode` | `[0,0]` | 每层舍入编号；当前只支持 0=rint |
| `pad_mode` | `[0,0]` | 每层补边模式；当前只支持 0=SPECIFIC，5=NOTSET 为保留值 |

写错字段名会直接报错并列出认识的字段 —— 不会静默用缺省值跑一个你没打算跑的形状。

## 四条通路的判据不一样

**fp16 定点**：两个 golden。
- `y_expect` 定点模型，假设成立时应当**逐位相等**
- `y_exact` 纯 fp32 参考，判「这个算子有没有在算这个卷积」，不受定点假设影响

板上先看 `y_exact` 的相对误差过不过，再看 `y_expect` 能不能逐位对上。
只有前者过、后者不过，说明卷积算对了但定点模型猜错了 —— 是两件不同的事。

`fixed_shift_value` 为统一的私有 ListInt，默认 `[0,0]`；原型不公开该属性。
由 CANN 融合及编译阶段产生，harness 只验证 native tiling，不通过 ONNX 属性传入。全 fp16 两项范围 `[0,58]`，
A16W8 第一项范围 `[0,29]`、第二项不参与计算。旧 `fixed_shift`、`a16w8_shift` 名称不再接受。
该零默认属于算子原型；harness 的 golden 默认期望来自第一阶段 Pass：FP16 为 `[42,42]`，A16W8 为 `[13,0]`。

**int8 量化**：一个 golden，精确整数运算 + per-channel VREQ8
（round-half-to-even + 饱和 + relu），应当逐位相等。当前 case 为了与历史 golden
逐位对齐，把同一 scale 复制到所有通道；表格式仍是每通道一个 uint64。

**int8 出问题时先看饱和那一行**：设备侧大面积饱和而 golden 没有，通常是
per-channel `quant_scale` 表的位型或 bit46 不对，不是卷积本身算错。

**s8f16**：前两层乘累加都是精确整数，conv1 通过 `quant_scale1` 输出 int8 mid，
conv2 通过 `dequant_scale2` 输出 fp16；golden 应逐位相等。

**a16w8**：输入是 fp16、filter 是 int8。生成器按 `2^(S-29)` 缩放整数输入；
S 较小时将整数选在可精确表示的格点上。MMAD 恢复同一整数参考，支持 S=0..29；
历史用例显式使用 S=29，其余量化/反量化与 s8f16 相同。

## 单独跑一条

```bash
python3 fc2d.py cases.json --only base_fp16 --steps run
# 或者直接调 runner（参数顺序随意：.bin 结尾的是 case，目录是 om，数字是 device）
python3 run_fused_multi_conv2d.py out/base_fp16/case.bin FusedMultiConv2d 0 out/base_fp16/om
```

`run_fused_multi_conv2d.py` 的环境变量：`REL_TOL`（默认 1e-3）、`RATIO_MIN`（默认 1.0）、
`REPEAT` / `WARMUP`（计时）。加 `--run-arg --dry-run` 可以不碰硬件把主流程走一遍，
用来在没有设备的机器上验脚本本身。

## 文件

跑用例**需要且仅需要**这些：

| 文件 | 角色 |
| :-- | :-- |
| `profile.json` | **你要改的就是这个**：dtype、形状、可选属性、远端信息 |
| `run_profile.py` | 总驱动，八步 |
| `fc2d.py` | 被 `run_profile.py` import；也能单独跑单机流程 |
| `fc2d_onnx.py` | 从 case 构造双 Conv 源图、常量 filters 与必要输入 |
| `fc2d_fusion.py` | 核对源图及两个阶段的编译图，不实现融合 |
| `fusion_cases.json` | 两阶段融合的四 dtype 零 padding 小用例 |
| `gen_case.cpp` | 编成 `gen_case`，按形状生成 `case.bin`（输入 + golden）|
| `fused_multi_conv2d_golden.h` | 两条通路的 CPU golden，被 `gen_case.cpp` include |
| `fc2d_geom.cpp` | 编成 `fc2d_geom`，形状预检 |
| `fused_multi_conv2d_shape.h` | 算子几何规划的预检副本，被 `fc2d_geom.cpp` include |
| `run_fused_multi_conv2d.py` | 在板子上执行的那一端，由 `run_profile.py` 自动拷过去 |
| `cases.json` | 可选。历史功能算例清单，使用显式 stride/pad。它只有形状、没有 remote 段，所以要这样接进来：<br>`python3 run_profile.py profile.json --cases cases.json --no-msprof` |

两个 C++ 工具是**编出来的，不进仓**（`.gitignore` 里就是 `gen_case` / `fc2d_geom`）。
不用手动编：`fc2d.py` 在用到它们之前会检查 —— 缺了就编，源码或同目录任何一个 `.h`
比二进制新也重编，编译行会打在日志里。想自己编也行：

```bash
g++ -std=c++17 -O2 -ffp-contract=off gen_case.cpp  -o gen_case  -I.
g++ -std=c++17 -O2                   fc2d_geom.cpp -o fc2d_geom -I.
```

`gen_case` 的 `-ffp-contract=off` 不是可选的：它要拿 fp32 参考模型和定点模型对账，
开着 FMA 合并的话参考值本身会随编译器优化漂，自检的容差就没意义了。

`fused_multi_conv2d_shape.h` 是算子那份的副本。**算子那边改了几何就要重拷**：

```bash
python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_multi_conv2d/op_host/op_tiling/fused_multi_conv2d_tiling.cpp
```

同步工具从 `tiling.cpp` 提取几何规划代码，并合并其引用的 `arch35/fused_multi_conv2d_geometry.h`，生成可独立编译的预检副本。适配逻辑仅位于 harness。

重拷之后不用管重编 —— 头变新了，下次用到 `fc2d_geom` 时会自动重来。它是顾问性质的：
真正说了算的永远是算子的 tiling，两边不一致时预检的结论会和 `atc` 不同，那本身就是
「该重拷了」的信号。

产物 `out/` 和 `prof_out/` 都在 `.gitignore` 里,不进仓。

## 不上板也能验板侧下发：ACL 桩

板侧那段（模型加载 → 按名字取输入下标 → dataset → `aclmdlExecute` → 读回 → 比对）
在**没有 NPU 的机器上执行不到** —— 真机 `aclInit` 直接失败（`chipType=0`），
而 `--dry-run` 是刻意绕开 ACL 分支的。代价是真的：`y_bytes` 的
`UnboundLocalError` 只是个先用后赋，却要等到板上第一次跑才暴露。

`tests/stub_ascendcl/` 用一个假的 `libascendcl.so` 顶掉真的那个，把整条路径跑完：

```bash
bash tests/stub_ascendcl/run_stub_test.sh            # out/ 下所有有 om 的 case
bash tests/stub_ascendcl/run_stub_test.sh --out-root out/profile base_int8
# 同名 case 出现在多份配置时，用完整目录消除歧义
bash tests/stub_ascendcl/run_stub_test.sh --case-dir out/profile/base_int8
```

每条 case 跑**两种定位方式**：

| | 含义 |
|---|---|
| 名字 | `aclmdlGetInputIndexByName` 能用 —— 期望的情形 |
| 退路 | 它全部失败，运行器退回 IR 顺序 |

**两条都要绿。** 退路那条在真板上很可能被用到：ACL 把图 om 的模型输入名报成什么样
不是我们能控制的，而运行器只有「名字取不到就按 IR 顺序猜」这一条退路，
靠逐项字节数校验兜底。

桩的 `aclmdlExecute` 把 `case.bin` 里的 `y_expect` 拷进输出缓冲，所以比对和表格报告
也会真的跑一遍。**它不验数值正确性** —— 那只能靠板子；它验的是
「这段代码会不会崩、分支走得对不对、字节数核对管不管用」。

`--out-root` 支持 `out/<case>/` 和 `out/<config>/<case>/`。每个完整 case 目录
必须有 case.bin、唯一 om/*.om 和有效 sidecar；缺失、同名歧义或 runner 非零退出都
算失败，PASS 文本不能覆盖失败返回码。测试子进程单独设置桩库路径，产品加载逻辑不变。

本地回归入口：

```bash
python3 -m unittest discover -s tests -v
python3 fc2d.py contract_cases.json --steps check,case
# A 上使用当前 OPP 和真实 ATC 编译后再验下发链
python3 fc2d.py contract_cases.json --steps om --soc <soc_version>
bash tests/stub_ascendcl/run_stub_test.sh --out-root out/contract_cases
```

`fusion_cases.json` 用于两阶段融合入口检查；`contract_cases.json` 保留算子契约覆盖，包含四 dtype、缺省几何、对称 pads、两层不同非方核、真实 Cout2=2、
私有 shift 与独立 bias/relu。测试中的 synthetic ATC/OM 只验证框架控制流；ACL 桩复制
golden 到输出，不代表设备计算正确或性能结果。真实精度和 profiling 按
HARDWARE_VALIDATION_WORKFLOW.md 在服务器 A/B 执行。

## Filter contract and case ABI 8

Case ABI 8 retains true logical NCHW filter tensors:
filter1=[Cout1,Cin,Kh1,Kw1], filter2=[Cout2,Cout1,Kh2,Kw2].
The ONNX builder embeds both tensors as initializers. GE folds them to FRACTAL_Z
at compilation and preserves their origin shape/format; kernel dimensions and
true Cout2 are inferred from that metadata. kernel_size is removed.

The target operator retains nine input ports. The source graph exposes x and
only consumed biases (1/3 runtime inputs). Unused instances become internal Const
nodes created by the production pass. Both filters and consumed scale tables are
initializers; GE handles FRACTAL_Z conversion and host scale folding. The frontend
uses built-in Conv and Ascend quantization ONNX parsers, with input_format=NCHW.
It contains neither ExtendConv2D nor FusedMultiConv2d.

Regenerate ABI 6 cases with the matching gen_case (ABI 8) and rebuild ONNX/OM
with the CANN package containing both original fusion stages. Both case readers reject old versions. Cout2=2
stays two logical output channels while internal FRACTAL_Z weights pad to 16.
Measure performance again using freshly compiled models.

The OM step writes `<model.om>.fc2d.json` with SHA256 hashes of the OM and complete
case.bin. The runner checks both hashes before loading ACL and rejects old
direct-graph model bindings. Copy case.bin, the OM, binding file and runner together;
run_profile.py uploads all four. Changing weights, attributes, geometry or any
case data requires a new case/ONNX/OM/binding.


The public order is `bias_mask, dtype, strides, pads, dilations, groups,
data_format, offset_x, round_mode, pad_mode, enable_hf32, enable_relu`.
Only bias_mask is required. dtype defaults to -1 and follows x; s8f16 explicitly selects fp16.
Private fixed_shift_value follows the SDK workspace slot at index 13.
The three attributes now use two-entry ListInt values, defaulting to [0,0]:

| Attribute | Integer mapping |
| --- | --- |
| data_format | 0=NCHW, 1=NHWC |
| round_mode | 0=rint, 1=round |
| pad_mode | 0=SPECIFIC, 1=SAME, 2=VALID, 3=SAME_UPPER, 4=SAME_LOWER, 5=NOTSET |

Each entry belongs to one convolution. Only zero is implemented; other defined codes,
including NOTSET=5, are reserved and rejected. Unknown codes, scalar strings/integers,
and lists whose length is not two are rejected. JSON case configuration accepts these
three keys as [0,0]; they describe the expected fused IR. The source uses standard ONNX Conv geometry.

Case ABI remains 8 because geometry and tensor payloads are unchanged. Model binding
schema is now 4 and records `op_type: FusedMultiConv2d`, the actual input names, both fusion stages and native tiling evidence; older bindings and OMs must be regenerated.
The operator directory is `conv/fused_multi_conv2d`, the kernel entry is `fused_multi_conv2d`,
and the generated ACLNN prefix is `aclnnFusedMultiConv2d`.
Existing ABI-8 cases may be used as compilation input, but regenerate ONNX, OM and binding
after rebuilding and loading the new operator package.

ABI 8 records the unified effective shift, including A16W8. Regenerate ABI 7 or earlier cases,
ONNX, OM and binding; old bindings cannot be reused after this contract change.

## 检查真实 ATC tiling 中的定标值

AscendC 的静态编译在 12 个公开属性后插入内部 `ascendc_op_para_size`，
因此统一私有 `fixed_shift_value` 的索引是 13。
旧代码少算这个内部位置，会把 fp16 的 conv1 shift 读成 0、A16W8 读成 2；
runner 从 case 打印的 42 / 29 并不能证明 kernel 实际收到了这些值。

在 A 加载已修正的算子包环境后，可对现有 case 做独立的真实编译验证：

```bash
python3 tools/verify_fc2d_atc_tiling.py out/profile/base_fp16/case.bin \
  --soc <soc_version> --out-dir out/native_tiling/base_fp16_new
```

输出目录必须是新目录。工具复制 case、生成 fresh ONNX/OM，并在本次 ATC
进程内记录官方 `do_op_tiling` 的原始返回值，核对 312 字节 tiling 中实际使用的 shift、
bias/relu 和 launch blockDim；不修改 SDK 文件。原始捕获与
`tiling_verification.json` 一起保存。没有捕获、旧属性错位或不同 case 的数据会失败，
对应 OM 的 binding 会删除，阻止它通过现有上传/运行检查。

该工具验证编译参数，实板精度仍按上板流程执行。单元测试保留了两份真实修复前
ATC 数据，防止用只包含公开/私有属性的手造 fixture 漏掉 SDK 插入位置。
