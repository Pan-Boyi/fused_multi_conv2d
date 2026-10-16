# fused_multi_conv2d —— FusedConv2d @ 5102 的上板验证

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
| `om` | 本地 | 从 ABI 8 case 生成含常量 filters 的 ONNX，跑 `atc --framework=5`，写 OM/case SHA256 binding |
| `push` | 本地 -> 远端 | 每条 case 一个干净目录,拷 `case.bin`、`.om`、`.om.fc2d.json`、`run_fused_conv2d.py` |
| `prof` | 远端 | `$MSPROF bash <runner wrapper>`，分别记录 runner 与 profiler 返回码，并找出**本次**新生成的 `PROF_*`。`$MSPROF` 是 `profile.msprof` 给死的绝对路径(缺省 `/var/msprof`),**不搜 PATH** —— 远端那个 ssh 是非交互 shell,读不到 `~/.bashrc` 里 source 的 `set_env.sh`,搜 PATH 会说「没有 msprof」而其实装着。指向 msprof 那个可执行文件本身。 |
| `pull` | 远端 -> 本地 | 把那个 `PROF_*` 拉回 `prof_out/<json 名>/<case 名>/` |
| `export` | 本地 | `TbeWorkTestSuit` 的 `msprof.py export summary` |
| `parse` | 本地 | 解析 `op_summary_*.csv`,每条打明细,最后打一张横向对比表 |

几个不显眼但要紧的点:

- **`run_fused_conv2d.py` 每次都重新拷过去。** 它和 `case.bin` 的格式是配套的,
  远端留着一份旧的会报「case 文件版本 N,本脚本认 M」,而那时人往往已经在查算子了。
- **图模型有 7 个运行输入。** 顺序为 x、bias1、bias2、scale1、scale2、
  relu_weight1、relu_weight2。scale2 按 dtype 使用 quant_scale2 或 dequant_scale2；
  未使用的辅助值补零。两个 filters 固定在 OM 内，算子节点仍有 9 个输入实例。
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

case.bin 保存逻辑输入、golden 和属性 spec。ONNX 从这份 spec 读取公开属性与
私有定标值，两张逻辑 NCHW filter 作为常量进入编译，GE 在编译期转成 FRACTAL_Z。
模型只接收 7 个运行输入，属性已经写入 OM。运行器在加载 ACL 前校验
`<model.om>.fc2d.json`，以完整 case 与 OM 的 SHA256 防止错配。

```text
case 配置 -> gen_case -> ABI 8 case.bin -> ONNX -> ATC -> OM + binding
                                                        -> 上传 -> runner
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

写错字段名会直接报错并列出认识的字段 —— 不会静默用缺省值跑一个你没打算跑的形状。

## 四条通路的判据不一样

**fp16 定点**：两个 golden。
- `y_expect` 定点模型，假设成立时应当**逐位相等**
- `y_exact` 纯 fp32 参考，判「这个算子有没有在算这个卷积」，不受定点假设影响

板上先看 `y_exact` 的相对误差过不过，再看 `y_expect` 能不能逐位对上。
只有前者过、后者不过，说明卷积算对了但定点模型猜错了 —— 是两件不同的事。

`fixed_shift_value` 为统一的私有 ListInt，默认 `[0,0]`；原型不公开该属性。
ONNX 插件或融合阶段将其写入 GE，tiling 校验后下发。全 fp16 两项范围 `[0,58]`，
A16W8 第一项范围 `[0,29]`、第二项不参与计算。旧 `fixed_shift`、`a16w8_shift` 名称不再接受。
历史数值用例显式保留原有工作点；原型默认零需要按实际输入进行标定，不能直接代替原工作点。

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
python3 run_fused_conv2d.py out/base_fp16/case.bin FusedConv2d 0 out/base_fp16/om
```

`run_fused_conv2d.py` 的环境变量：`REL_TOL`（默认 1e-3）、`RATIO_MIN`（默认 1.0）、
`REPEAT` / `WARMUP`（计时）。加 `--run-arg --dry-run` 可以不碰硬件把主流程走一遍，
用来在没有设备的机器上验脚本本身。

## 文件

跑用例**需要且仅需要**这些：

| 文件 | 角色 |
| :-- | :-- |
| `profile.json` | **你要改的就是这个**：dtype、形状、可选属性、远端信息 |
| `run_profile.py` | 总驱动，八步 |
| `fc2d.py` | 被 `run_profile.py` import；也能单独跑单机流程 |
| `gen_case.cpp` | 编成 `gen_case`，按形状生成 `case.bin`（输入 + golden）|
| `fused_conv2d_golden.h` | 两条通路的 CPU golden，被 `gen_case.cpp` include |
| `fc2d_geom.cpp` | 编成 `fc2d_geom`，形状预检 |
| `fused_conv2d_shape.h` | 算子共用几何头的**副本**，被 `fc2d_geom.cpp` include |
| `run_fused_conv2d.py` | 在板子上执行的那一端，由 `run_profile.py` 自动拷过去 |
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

`fused_conv2d_shape.h` 是算子那份的副本。**算子那边改了几何就要重拷**：

```bash
python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_conv2d/op_kernel/fused_conv2d_shape.h
```

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

桩的 `aclmdlExecute` 把 `case.bin` 里的 `y_expect` 拷进输出缓冲，所以比对、诊断、
抽样那几段也会真的跑一遍。**它不验数值正确性** —— 那只能靠板子；它验的是
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

`contract_cases.json` 覆盖四 dtype、缺省几何、对称 pads、两层不同非方核、真实 Cout2=2、
私有 shift 与独立 bias/relu。测试中的 synthetic ATC/OM 只验证框架控制流；ACL 桩复制
golden 到输出，不代表设备计算正确或性能结果。真实精度和 profiling 按
HARDWARE_VALIDATION_WORKFLOW.md 在服务器 A/B 执行。

## Filter contract and case ABI 8

Case ABI 8 retains true logical NCHW filter tensors:
filter1=[Cout1,Cin,Kh1,Kw1], filter2=[Cout2,Cout1,Kh2,Kw2].
The ONNX builder embeds both tensors as initializers. GE folds them to FRACTAL_Z
at compilation and preserves their origin shape/format; kernel dimensions and
true Cout2 are inferred from that metadata. kernel_size is removed.

The operator retains nine input ports. Its compiled graph has seven runtime
inputs: x and two instances each of bias, scale and relu_weight. Both filters
are fixed in the OM. This path avoids the unavailable runtime NCHW-to-FRACTAL_Z
TransData engine in the locally validated MC62 SDK. ATC uses input_format=NCHW;
bias/scale/relu_weight remain one-dimensional ND inputs.

Regenerate ABI 6 cases with the matching gen_case (ABI 8) and rebuild ONNX/OM
with the new operator package. Both case readers reject old versions. Cout2=2
stays two logical output channels while internal FRACTAL_Z weights pad to 16.
Measure performance again using freshly compiled models.

The OM step writes `<model.om>.fc2d.json` with SHA256 hashes of the OM and complete
case.bin. The runner checks both hashes before loading ACL and rejects old
nine-input models. Copy case.bin, the OM, binding file and runner together;
run_profile.py uploads all four. Changing weights, attributes, geometry or any
case data requires a new case/ONNX/OM/binding.


The public order is `bias_mask, dtype, strides, pads, dilations, groups,
data_format, offset_x, round_mode, pad_mode, enable_hf32, enable_relu`.
Only bias_mask is required. dtype defaults to -1 and follows x; s8f16 explicitly selects fp16.
Private fixed_shift_value follows the SDK workspace slot at index 13.
ListString migration remains pending: current CANN 9.2 lacks OpDef ListString;
data_format/round_mode/pad_mode currently retain String.

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
进程内记录官方 `do_op_tiling` 的原始返回值，核对 312 字节 tiling 中的实际 shift、
bias/relu 和 launch blockDim；不修改 SDK 文件。原始捕获与
`tiling_verification.json` 一起保存。没有捕获、旧属性错位或不同 case 的数据会失败，
对应 OM 的 binding 会删除，阻止它通过现有上传/运行检查。

该工具验证编译参数，实板精度仍按上板流程执行。单元测试保留了两份真实修复前
ATC 数据，防止用只包含公开/私有属性的手造 fixture 漏掉 SDK 插入位置。
