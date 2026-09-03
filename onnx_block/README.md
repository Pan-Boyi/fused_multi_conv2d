# onnx_block —— 把一张昇腾量化 ONNX 图编成 .om，并给它一套 numpy golden

对应截图里那个量化残差块：

```
x:int8 [4,128,32,48]
  ├─ Conv   W int8 [128,128,3,3]  B int32 [128]  →int32  ┐
  ├─ AscendDequant  deq_scale[128]               →fp16   │ 截图红框
  ├─ Relu                                        →fp16   │
  ├─ AscendQuant                                 →int8   ┘
  ├─ Conv   W int8 [128,128,3,3]  B int32 [128]  →int32
  ├─ AscendDequant  deq_scale[128]               →fp16
  └─ Add( ↑ , res:fp16 [4,128,32,48] )           →fp16     ← res 就是图里的 1661
```

## 你手上有原始 .onnx 的话（推荐）

**不要重建图。** 直接用真文件走全程 —— golden 和 atc 吃的是同一个文件，参数不可能对不上：

```bash
python3 inspect_onnx.py 你的.onnx                    # 先看清楚属性和 dtype
python3 onnx_golden.py 你的.onnx --outdir=io         # 造输入 + 算 golden
SOC=<soc_version> ./build_om_onnx.sh 你的.onnx       # 编 om
python3 ../run_om.py om_out/*.om io/in*.bin --golden0=io/golden0.*.bin
```

整张网络太大只想要这一块的话，先用 `onnx.utils.extract_model` 按张量名裁一段出来
（起点 `Conv_152.quant.output0`，终点是 Add 的输出），再走上面四步。

## 只有截图的话

```bash
python3 build_block.py block.onnx      # 重建
```
拓扑和 shape 是照截图搭的；`pads/strides/dilations/group` 被 shape 反推死了
（3×3、32×48→32×48 ⇒ stride 1 pad 1 dilation 1；W 是 (128,128,3,3) 而输入 128 通道 ⇒ group 1），
不用猜。**猜的是** 两个自定义算子的 domain、属性名、`deq_scale` 存 uint64 还是 float32，
以及权重/scale 的数值。这些用 `inspect_onnx.py` 对着真文件核，`build_block.py` 有对应的开关。

## 四个脚本

| | 干什么 |
|---|---|
| `inspect_onnx.py` | 把 .onnx 摊平打印。**整条链第一个该跑的。** uint64 的 `deq_scale` 会额外把低 32 位当 float32 解出来 |
| `build_block.py` | 从截图重建这张图 |
| `onnx_golden.py` | numpy 复算。**所有参数从 .onnx 读**，没有硬编码 |
| `build_om_onnx.sh` | `atc --framework=5`，外加 ld.lld 和解析器的前置探测 |

跑模型用上一层现成的 `../run_om.py`（`aclmdlLoadFromFile` + `aclmdlExecute`，
自带 `--goldenN=` 比对），不需要新写。

## golden 的两条设计规矩

**遇到没验证过的属性组合就报错，不猜。** `sqrt_mode=1`、`dst_type` 不是 int8、
`group != 1`、`relu_flag=1` 都会直接停下来说缺什么。一个偷偷猜错的 golden 比没有
golden 更糟——后面所有"实测对不上"的结论都会跟着错。

**`deq_scale` 的 uint64 解包是这条链上最容易静默错的一步。** GE 把 float32 的比特
原样放在低 32 位，直接当整数用会得到 1e9 量级的乘数。`inspect_onnx.py` 会把两种
读法都打出来，高位非零时 golden 直接拒绝执行。

## 唯一需要实测定的一件事

量化的舍入模式。`--round=rint`（默认，向最近偶数）是我在 5102 上验过
`Cast(CAST_RINT)` 走的那个，融合卷积的 golden 按它写、逐位对上；`--round=away`
是 .5 向远离 0 舍入。两者只在恰好落在 .5 的点上不同，`onnx_golden.py` 会**数出**
有多少个这样的点。重建图上是 725 个（0.09%）；ties=0 的话这个疑问就不存在。

## 没验证的部分

这四个脚本在 macOS 上跑通了 numpy/onnx 那一半（重建→golden→逐级范围自洽，
Quant 饱和 0 个）。**`atc` 和 `run_om.py` 那一半我这里没有 CANN、没有板子，
一次都没跑过。** 第 1、2 步的探测就是为这个准备的：atc 挂了先看它们的输出，
能立刻区分"图写错了"和"这台机器的包不带量化算子映射"。
