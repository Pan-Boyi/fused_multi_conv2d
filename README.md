# fused_multi_conv2d —— MC62/5102 上融合卷积的上板验证脚本

配套算子:`ops-nn` 的 `conv/fused_conv2d`(分支 `feat/fused-conv2d`)。

```
x fp16 NCHW ──conv1 3x3 s1 p1 +bias1(fp16) +relu──► mid fp16
            ──conv2 3x3 s2 p1 +bias2(fp16) +relu──► y fp16 NCHW
```

全程定点(f162s32),没有量化、没有向量。权重 FRACTAL_Z fp16(C0 = 16)。
接口是 **5 输入 + 2 属性**:`x, filter1, bias1, filter2, bias2 → y`,
外加 `fixed_shift1` / `fixed_shift2`。

## 最短路径

```bash
# 1. 造 case（任何能编译的机器，不需要 CANN、不需要设备）
./build_case.sh                          # 产物 fused_conv2d_case.bin，5,309,240 B

# 2. 出 om（装了带 FusedConv2d 算子包的机器）
SOC=<soc_version> ./build_om.sh

# 3. 跑（有 5102 的机器，不需要编译器）
python3 run_fused_conv2d.py fused_conv2d_case.bin FusedConv2d 0
```

定点定标由 golden 按数据算出、打包进 `.bin` 的 header,脚本取出来当算子属性下发 ——
不用手填,主机和设备也不可能用成不同的值。

**详细步骤、判据、排障见 [`5102单算子验证步骤.md`](5102单算子验证步骤.md)。**

## 文件

| | 跑在哪 | 干什么 |
| :--- | :--- | :--- |
| `fused_conv2d_golden.h` | 任意 | CPU 参考。同时给**定点模型**和**纯 fp32 参考**两个 golden |
| `golden_selfcheck.cpp` | 任意 | fp16 位型穷举往返 + FRACTAL_Z 往返 + 算出该传的定标值 |
| `gen_case.cpp` | 任意 | 打包成 `fused_conv2d_case.bin` |
| `build_case.sh` | 任意 | 串起上面三个 |
| `build_om.sh` / `fused_conv2d_singleop.json` | 有算子包的机器 | 编单算子离线模型 |
| `run_fused_conv2d.py` | 有 5102 的机器 | 主执行器,ctypes 调 `aclopExecuteV2`,不需要编译器 |
| `test_aclop_fused_conv2d.cpp` | 有 CANN 的机器 | 同一件事的 C++ 版本 |
| `build_and_run_aclop.sh` / `build_cross.sh` | | 编上面那个(本地 / 交叉) |
| `run_om.py` / `run_op.py` | 有 5102 的机器 | **通用**执行器,不绑这个算子 |
| `singleop2case.py` | 任意 | 从 om 的 json 生成 case 描述,避免手抄 |
| `run_prof.sh` / `parse_prof.py` | 有 5102 的机器 | msprof 采集与解析 |
| `onnx_block/` | | 另一件事:把昇腾量化 ONNX 图编成 om,见该目录的 README |

## 两个 golden 怎么用

| 张量 | 是什么 | 怎么判 |
| :--- | :--- | :--- |
| `y_expect` | 定点模型 | 应**逐位相等** |
| `y_exact` | 纯 fp32 参考 | 用相对误差判 |

`fixed_shift` 的语义**已经从 CANN 源码确证**,不再是假设。记属性为 `S`:

```
acc_int32 = round( 真值 * 2^(58 - S) )
out_fp16  = fp16( acc_int32 * 2^-(58 - S) )
```

依据是 DEQF16 模式下 fixpipe 根本不看 `deqScalar`,只按 shift 现搭一个 float
(`dav_m510/kernel_operator_fixpipe_impl.h:77` `SetDeqScalarDepOnMode`):

```c
uint64_t newExponent = (127 - shiftVal) & 0xFF;
uint64_t newScalar   = (floatOne & ~mask) | (newExponent << shift);
```

指数为 `127-F` 的 float 就是 `2^-F`,而 `F` 正是传给 fixpipe 的 `58 - S`。

> **S 越大,累加器的定标越小 —— 和「shift 越大精度越高」的直觉正好相反。**
> 这个方向写反了不报任何错。本算子踩过一次:S 按 26/22 下发,硬件实际按
> `2^32`/`2^36` 定标,几乎每个非零点都撑爆 int32,输出的符号退化成掷硬币。
> 板上的表征很干脆 —— 非零元素一个都没对上,对上的全部是两边都为零的点。

所以判据是:**先看 `mismatches`(对 `y_expect`,应逐位相等)**。不等时脚本会报一行
以累加器 LSB 计的偏差 —— 个位数 LSB 说明只是 round 的级别猜得不同,卷积是对的;
成千上万个 LSB 才是真错了。`y_exact` 那一路不设门槛,只回答「有没有在算这个卷积」。
