# fused_multi_conv2d —— FusedConv2d @ 5102 的上板验证

**改一个 json（`cases.json`）就能换形状**，别的地方不用动。

```bash
# 0) 一次性：编两个小工具（只要 g++，不需要 CANN，不需要设备）
g++ -std=c++17 -O2 -ffp-contract=off gen_case.cpp  -o gen_case  -I.
g++ -std=c++17 -O2                   fc2d_geom.cpp -o fc2d_geom -I.

# 1) 看看清单里有哪些形状、算子能不能服务（不动任何文件）
python3 fc2d.py cases.json --list
python3 fc2d.py cases.json --steps check

# 2) 在有 CANN 的机器上：生成 case 数据 + 编离线模型
source <CANN>/set_env.sh
python3 fc2d.py cases.json --steps check,case,om --soc <soc_version>

# 3) 把整个目录拷到有 5102 的机器上，跑
python3 fc2d.py cases.json --steps run
```

产物都在 `out/<name>/`：`case.bin`（输入 + golden）、`singleop.json`、`om/*.om`。

## 为什么是一个 json 而不是三处

一个形状要三样东西严丝合缝：

| | 谁产生 |
| :-- | :-- |
| `case.bin` 里的输入和 golden | `gen_case` |
| `.om`（ACL 按 **op 类型 + 每个张量的 shape/dtype/format + 全部属性的值** 匹配） | `atc --singleop` |
| 执行时下发的属性 | `run_fused_conv2d.py` |

三者只要有一处不一致，板上报的是 `100024` / `MatchOpModel fail`，翻译过来是
**「算子没找到」** —— 那条信息完全不指向真正的原因。这条链上折过好几次：
改了 golden 忘了改 json、int8 分支之后又多设了一个属性、om 目录里留着上一个形状的
`.om`……

所以现在三者**都从同一处派生**：

```
cases.json 里的一条
   -> gen_case 的命令行
   -> case.bin 头部的 spec（32 个 int，形状 + 属性的唯一真相）
   -> singleop.json（fc2d.py 从 spec 读，不从 json 读）
   -> 执行时下发的属性（run_fused_conv2d.py 从 spec 读）
```

中间没有第二份形状。另外属性是**九个一个不少地全设**，两边都一样 —— 不再有
「这条通路设这几个、那条通路设那几个」的子集，那正是最容易出错的地方。

## cases.json 怎么写

```json
{
  "device": 0,
  "cases": [
    {"name": "base", "dtype": "both", "ci": 32, "hi": 288, "wi": 112, "cout1": 64, "cout2": 96},
    {"name": "k5x5", "dtype": "int8", "ci": 32, "hi": 64, "wi": 64, "cout1": 32, "cout2": 32,
     "kernel": [5, 5], "pads": [2, 2], "bias": true}
  ]
}
```

| 字段 | 缺省 | 说明 |
| :-- | :-- | :-- |
| `name` | `caseNN` | 产物落在 `out/<name>/` |
| `dtype` | `fp16` | `fp16` / `int8` / `both`（`both` 展开成 `<name>_fp16` 和 `<name>_int8`）|
| `n` `ci` `hi` `wi` | 1 / 32 / 288 / 112 | 输入形状 NCHW |
| `cout1` `cout2` | 64 / 96 | 两层的输出通道 |
| `kernel` | `[3, 3]` | `[kh, kw]`，两层共用 |
| `strides` | `[1, 2]` | `[stride1, stride2]`，两层各一个 |
| `pads` | `[1, 1]` | `[p1, p2]`（H/W 同值）或 `[padH1, padW1, padH2, padW2]`。非方核必须用长度 4 的那种 |
| `bias` | `false` | 带不带 bias（两层一起）|
| `relu` | `[true, true]` | `[relu1, relu2]` |
| `fixed_shift` | `[42, 42]` | 只有 fp16 通路用，见下 |

写错字段名会直接报错并列出认识的字段 —— 不会静默用缺省值跑一个你没打算跑的形状。

## 两条通路的判据不一样

**fp16 定点**：两个 golden。
- `y_expect` 定点模型，假设成立时应当**逐位相等**
- `y_exact` 纯 fp32 参考，判「这个算子有没有在算这个卷积」，不受定点假设影响

板上先看 `y_exact` 的相对误差过不过，再看 `y_expect` 能不能逐位对上。
只有前者过、后者不过，说明卷积算对了但定点模型猜错了 —— 是两件不同的事。

`fixed_shift` 的语义是 **累加器 = 真值 × 2^(58 − S)**，所以 **S 越大定标越小**，
和直觉相反。42（F = 16）是厂商工作点。`gen_case` 会打出「部分和不溢出允许到 F ≤ ?」，
超了会警告 —— 超了的话逐位比对不可能成立，因为累加器真的溢出了。

**int8 量化**：一个 golden，精确整数运算 + REQ8（round-half-to-even + 饱和 + relu），
应当逐位相等。scale 由 golden 按 `127 / 峰值` 自动挑并按硬件丢掉低 13 位尾数。

**int8 出问题时先看饱和那一行**：设备侧大面积饱和而 golden 没有，是 `quant_scale`
没传到，不是算子算错了。

## 单独跑一条

```bash
python3 fc2d.py cases.json --only base_fp16 --steps run
# 或者直接调 runner（参数顺序随意：.bin 结尾的是 case，目录是 om，数字是 device）
python3 run_fused_conv2d.py out/base_fp16/case.bin FusedConv2d 0 out/base_fp16/om
```

`run_fused_conv2d.py` 的环境变量：`REL_TOL`（默认 1e-3）、`RATIO_MIN`（默认 1.0）、
`REPEAT` / `WARMUP`（计时）。加 `--run-arg --dry-run` 可以不碰硬件把主流程走一遍，
用来在没有设备的机器上验脚本本身。

## 文件

| 文件 | 说明 |
| :-- | :-- |
| `cases.json` | **你要改的就是这个** |
| `fc2d.py` | 总驱动：check → case → om → run |
| `gen_case.cpp` | 按命令行给的形状生成 `case.bin`（输入 + golden） |
| `fused_conv2d_golden.h` | 两条通路的 CPU golden，按形状参数化 |
| `fc2d_geom.cpp` | 形状预检。用的是算子共用几何头的**副本**，**顾问性质** |
| `fused_conv2d_shape.h` | 上面那份副本。**算子那边改了就要重拷**：<br>`python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_conv2d/op_kernel/fused_conv2d_shape.h`<br>然后重编 `fc2d_geom` |
| `run_fused_conv2d.py` | ctypes 直调 `aclopExecuteV2`，目标机不需要编译器 |
