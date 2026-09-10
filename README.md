# fused_multi_conv2d —— FusedConv2d @ 5102 的上板验证

**改一个 json（`cases.json`）就能换形状**，别的地方不用动。

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

产物都在 `out/<name>/`：`case.bin`（输入 + golden）、`singleop.json`、`om/*.om`。

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
| `om` | 本地 | 写 `singleop.json`,跑 `atc --singleop` |
| `push` | 本地 -> 远端 | 每条 case 一个干净目录,拷 `case.bin`、`.om`、`run_fused_conv2d.py` |
| `prof` | 远端 | `$MSPROF python3 run_fused_conv2d.py ...`,并找出**本次**新生成的 `PROF_*`。`$MSPROF` 是 `profile.msprof` 给死的绝对路径(缺省 `/var/msprof`),**不搜 PATH** —— 远端那个 ssh 是非交互 shell,读不到 `~/.bashrc` 里 source 的 `set_env.sh`,搜 PATH 会说「没有 msprof」而其实装着。给目录也行,会自动接一层 `/msprof`。 |
| `pull` | 远端 -> 本地 | 把那个 `PROF_*` 拉回 `prof_out/<json 名>/<case 名>/` |
| `export` | 本地 | `TbeWorkTestSuit` 的 `msprof.py export summary` |
| `parse` | 本地 | 解析 `op_summary_*.csv`,每条打明细,最后打一张横向对比表 |

几个不显眼但要紧的点:

- **`run_fused_conv2d.py` 每次都重新拷过去。** 它和 `case.bin` 的格式是配套的,
  远端留着一份旧的会报「case 文件版本 N,本脚本认 M」,而那时人往往已经在查算子了。
- **每条 case 在远端有自己的干净目录,不复用。** `aclopSetModelDir` 会把目录下所有
  `.om` 都装进去,留着上一个形状的那个,它可能反而先匹配上,跑出来的是上个形状的
  结果,而且不报错。
- **新 `PROF_*` 靠执行前后的差集找,不靠"取最新"。** 目录名里带时间戳,并发或重跑
  时"最新"会拿错。一次冒出多个也会直接报错而不是随便挑一个。
- **kernel 跑挂了照样导 PROF。** 挂掉时的流水图往往正是要看的东西。
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
| `cases.json` | 可选。28 个形状 x 两条通路的功能清单。它只有形状、没有 remote 段，所以要这样接进来：<br>`python3 run_profile.py profile.json --cases cases.json --no-msprof` |

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
