#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fused_conv2d 上板验证的总驱动 —— **改一个 json 就能换形状**。

    python3 fc2d.py cases.json                  # 全流程：生成 case -> 出 om -> 上板跑
    python3 fc2d.py cases.json --only base_fp16 # 只做某几条（名字，可给多个）
    python3 fc2d.py cases.json --steps case,om  # 在能编译的机器上先把前两步做完
    python3 fc2d.py cases.json --steps run      # 在有 5102 的机器上只跑
    python3 fc2d.py cases.json --list           # 只列出会做哪些，什么也不动

三步各自的前提不一样，所以能分开做：
    case  只要 g++。不需要 CANN，不需要设备，架构无所谓。
    om    要 atc（CANN），不需要设备。soc_version 由 --soc 或 FC2D_SOC 给。
    run   要设备。

================================================================================
为什么要有这个脚本
================================================================================
形状变成运行期之后，一个形状要三样东西对齐：
    .bin        输入数据 + golden
    singleop.json / .om   ACL 按 **op 类型 + shape/dtype/format + attr 的值** 匹配
    执行时设的 attr
三者只要有一处不一致，板上报的是「算子没找到」（100024 / MatchOpModel fail），
那条错误信息完全不指向真正的原因，上一版在这上面折过好几次。

这个脚本让三者**都从同一处派生**：json 里的一条 case -> gen_case 的命令行 ->
.bin 头部的 spec -> singleop.json -> 执行时的 attr。中间没有第二份形状。
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# gen_case.cpp 里 SpecIdx 的顺序。**两边必须一致**，改那边就得改这边。
SPEC_KEYS = [
    "n", "ci", "hi", "wi", "cout1", "cout2", "kh", "kw", "s1", "s2",
    "ph1", "pw1", "ph2", "pw2", "elem_bytes", "bias", "relu1", "relu2",
    "shift1", "shift2", "ho2", "wo2", "safe_f1", "safe_f2",
]
SPEC_N = 32
HDR_FMT = "<8sIIQQQ" + "%di" % SPEC_N + "II"
HDR_LEN = struct.calcsize(HDR_FMT)
CASE_VERSION = 4


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


def step(msg):
    print("\n==== %s ====" % msg)
    sys.stdout.flush()


def run(argv):
    """跑一个子进程。先 flush —— 输出被重定向到文件时 Python 这边是块缓冲的，
    不 flush 的话子进程的输出会插到不相干的地方，日志读起来像另一条 case 的。"""
    sys.stdout.flush()
    return subprocess.call(argv)


# ---------------------------------------------------------------- json -> case
DEFAULTS = {
    "dtype": "fp16",
    "n": 1, "ci": 32, "hi": 288, "wi": 112,
    "cout1": 64, "cout2": 96,
    "kernel": [3, 3],
    "strides": [1, 2],
    "pads": [1, 1, 1, 1],
    "bias": False,
    "relu": [True, True],
    "fixed_shift": [42, 42],
}


def normalise(case, index):
    """把 json 里的一条补全成完整的一条，并做基本校验。"""
    out = dict(DEFAULTS)
    unknown = [k for k in case if k not in DEFAULTS and k != "name"]
    if unknown:
        die("case #%d 里有不认识的字段 %s。认识的是：%s"
            % (index, unknown, ", ".join(sorted(DEFAULTS) + ["name"])))
    out.update(case)
    out["name"] = case.get("name", "case%02d" % index)
    if out["dtype"] not in ("fp16", "int8", "both"):
        die("case %s 的 dtype 只能是 fp16 / int8 / both" % out["name"])
    k = out["kernel"]
    if not isinstance(k, list) or len(k) != 2:
        die("case %s 的 kernel 要写成 [kh, kw]" % out["name"])
    s = out["strides"]
    if not isinstance(s, list) or len(s) != 2:
        die("case %s 的 strides 要写成 [stride1, stride2]（两个卷积各一个）" % out["name"])
    p = out["pads"]
    if not isinstance(p, list) or len(p) not in (2, 4):
        die("case %s 的 pads 要写成 [p1, p2] 或 [padH1, padW1, padH2, padW2]" % out["name"])
    if len(p) == 2:
        # 长度 2 = H 和 W 同值。非方核必须用长度 4 的那种。
        out["pads"] = [p[0], p[0], p[1], p[1]]
    r = out["relu"]
    if not isinstance(r, list) or len(r) != 2:
        die("case %s 的 relu 要写成 [relu1, relu2]" % out["name"])
    fs = out["fixed_shift"]
    if not isinstance(fs, list) or len(fs) != 2:
        die("case %s 的 fixed_shift 要写成 [shift1, shift2]" % out["name"])
    return out


def gen_case_argv(c, out_path):
    ph1, pw1, ph2, pw2 = c["pads"]
    return [
        os.path.join(HERE, "gen_case"), out_path,
        "--dtype", c["dtype"],
        "--n", str(c["n"]), "--ci", str(c["ci"]), "--hi", str(c["hi"]), "--wi", str(c["wi"]),
        "--cout1", str(c["cout1"]), "--cout2", str(c["cout2"]),
        "--kh", str(c["kernel"][0]), "--kw", str(c["kernel"][1]),
        "--s1", str(c["strides"][0]), "--s2", str(c["strides"][1]),
        "--ph1", str(ph1), "--pw1", str(pw1), "--ph2", str(ph2), "--pw2", str(pw2),
        "--bias", "1" if c["bias"] else "0",
        "--relu1", "1" if c["relu"][0] else "0", "--relu2", "1" if c["relu"][1] else "0",
        "--shift1", str(c["fixed_shift"][0]), "--shift2", str(c["fixed_shift"][1]),
    ]


# ---------------------------------------------------------------- 读 .bin 的 spec
def read_spec(path):
    """把 .bin 头部的 spec 读出来。**singleop.json 只从这里派生**，不从 json 派生 ——
    这样 .bin 和 .om 不可能对不上形状。"""
    with open(path, "rb") as f:
        hdr = f.read(HDR_LEN)
    if len(hdr) < HDR_LEN:
        die("%s 不完整（只有 %d 字节）" % (path, len(hdr)))
    vals = struct.unpack(HDR_FMT, hdr)
    magic, version, ntensors, nonzero, sat, y_elems = vals[:6]
    spec = vals[6:6 + SPEC_N]
    qs1_bits, qs2_bits = vals[6 + SPEC_N], vals[7 + SPEC_N]
    if magic != b"FC2DCASE":
        die("%s 不是 case 文件（magic = %r）" % (path, magic))
    if version != CASE_VERSION:
        die("%s 是版本 %d，本脚本认版本 %d —— gen_case 和 fc2d.py 要配套"
            % (path, version, CASE_VERSION))
    d = dict(zip(SPEC_KEYS, spec))
    d["nonzero"], d["sat"], d["y_elems"], d["ntensors"] = nonzero, sat, y_elems, ntensors
    d["qs1"] = struct.unpack("<f", struct.pack("<I", qs1_bits))[0]
    d["qs2"] = struct.unpack("<f", struct.pack("<I", qs2_bits))[0]
    return d


def singleop_json(spec):
    """从 .bin 的 spec 造 singleop 描述。

    **九个属性一个不少地全写上。** ACL 是按属性的**值**匹配 .om 的，多一个少一个
    都会匹配不上，报出来是「算子没找到」。上一版 fp16 和 int8 各写一份 json、各带
    一个子集，于是执行时多设一个属性就炸 —— 这里统一成全集，那类问题不存在了。
    """
    fp16 = spec["elem_bytes"] == 2
    t = "float16" if fp16 else "int8"
    bt = "float16" if fp16 else "int32"
    c0 = 16 if fp16 else 32
    fz1k = (spec["ci"] // c0) * spec["kh"] * spec["kw"]
    fz2k = (spec["cout1"] // c0) * spec["kh"] * spec["kw"]
    inputs = [
        {"format": "ND", "shape": [spec["n"], spec["ci"], spec["hi"], spec["wi"]], "type": t},
        {"format": "ND", "shape": [fz1k, spec["cout1"] // 16, 16, c0], "type": t},
        {"format": "ND", "shape": [spec["cout1"]], "type": bt},
        {"format": "ND", "shape": [fz2k, spec["cout2"] // 16, 16, c0], "type": t},
        {"format": "ND", "shape": [spec["cout2"]], "type": bt},
    ]
    return [{
        "op": "FusedConv2d",
        "input_desc": inputs,
        "output_desc": [
            {"format": "ND",
             "shape": [spec["n"], spec["cout2"], spec["ho2"], spec["wo2"]],
             "type": t}
        ],
        "attr": [
            {"name": "fixed_shift1", "type": "int", "value": spec["shift1"]},
            {"name": "fixed_shift2", "type": "int", "value": spec["shift2"]},
            {"name": "relu1", "type": "bool", "value": bool(spec["relu1"])},
            {"name": "relu2", "type": "bool", "value": bool(spec["relu2"])},
            {"name": "quant_scale1", "type": "float", "value": spec["qs1"]},
            {"name": "quant_scale2", "type": "float", "value": spec["qs2"]},
            {"name": "kernel_size", "type": "list_int", "value": [spec["kh"], spec["kw"]]},
            {"name": "strides", "type": "list_int", "value": [spec["s1"], spec["s2"]]},
            {"name": "pads", "type": "list_int",
             "value": [spec["ph1"], spec["pw1"], spec["ph2"], spec["pw2"]]},
        ],
    }]


# ---------------------------------------------------------------- 两个 C++ 工具
# 源码进仓，二进制不进（.gitignore 里就写着 gen_case / fc2d_geom）。所以
# **每一次新克隆的第一次运行都会缺它们**，而且改过 fused_conv2d_shape.h 之后
# 旧二进制还在、还能跑，只是按过期的几何算 —— 那种错比编译失败难查得多。
#
# 与其在报错里写一行「先编：g++ ...」让人照抄，不如脚本自己编：缺了就编，
# 源码或任何一个头比二进制新也重编。两个工具都是单文件，几秒钟的事。
TOOLS = {
    # gen_case 要算 fp32 参考模型并和定点模型对账，**必须关掉 FMA 合并**，
    # 否则参考值本身会随编译器优化漂，自检的容差就没有意义了。
    "gen_case": ("gen_case.cpp", ["-ffp-contract=off"]),
    "fc2d_geom": ("fc2d_geom.cpp", []),
}


def ensure_tool(name):
    """返回可执行文件的路径，必要时先编出来。"""
    exe = os.path.join(HERE, name)
    srcname, extra = TOOLS[name]
    src = os.path.join(HERE, srcname)
    if not os.path.isfile(src):
        die("找不到 %s —— 这个文件是进仓的，克隆不完整？" % src)
    # 依赖就是同目录下所有头文件。两个工具各自只 include 其中一两个，但多编
    # 几次的代价是几秒钟，漏编一次的代价是一批按旧几何算出来的结论。
    deps = [src] + [os.path.join(HERE, h) for h in os.listdir(HERE) if h.endswith(".h")]
    newest = max(os.path.getmtime(d) for d in deps)
    if os.path.isfile(exe) and os.path.getmtime(exe) >= newest:
        return exe
    why = "还没编" if not os.path.isfile(exe) else "比源码旧"
    cxx = os.environ.get("CXX", "g++")
    argv = [cxx, "-std=c++17", "-O2"] + extra + [src, "-o", exe, "-I", HERE]
    print("  [build] %s %s: %s" % (name, why, " ".join(argv)))
    if run(argv) != 0:
        die("编 %s 失败。手动跑一遍看完整报错：\n    %s" % (name, " ".join(argv)))
    return exe


# ---------------------------------------------------------------- 形状预检
def do_check(c, l1):
    """在生成 case、编 .om 之前先问一句「这个形状算子能不能跑」。

    **顾问性质**：真正说了算的是算子的 tiling。但 fc2d_geom 用的是算子共用几何头
    的副本、调的是同一个 PickHb，所以只要副本没过期结论就一致；不一致本身就是
    「该重拷 fused_conv2d_shape.h 了」的信号。

    没有这一步的话，形状不合法要等到 atc 失败才知道，而 atc 的原因只在
    ~/ascend/log/ 里，找起来很费劲。
    """
    exe = ensure_tool("fc2d_geom")
    ph1, pw1, ph2, pw2 = c["pads"]
    argv = [exe, "--dtype", c["dtype"],
            "--n", str(c["n"]), "--ci", str(c["ci"]), "--hi", str(c["hi"]), "--wi", str(c["wi"]),
            "--cout1", str(c["cout1"]), "--cout2", str(c["cout2"]),
            "--kh", str(c["kernel"][0]), "--kw", str(c["kernel"][1]),
            "--s1", str(c["strides"][0]), "--s2", str(c["strides"][1]),
            "--ph1", str(ph1), "--pw1", str(pw1), "--ph2", str(ph2), "--pw2", str(pw2),
            "--l1", str(l1)]
    if run(argv) != 0:
        raise RuntimeError("形状预检没过 —— 这个形状算子服务不了（上面一行是原因）")


# ---------------------------------------------------------------- 三个步骤
def do_case(c, paths):
    ensure_tool("gen_case")
    argv = gen_case_argv(c, paths["bin"])
    print("  " + " ".join(argv[1:]))
    rc = run(argv)
    if rc != 0:
        die("gen_case 失败（%s）" % c["name"])


def do_om(c, paths, soc):
    if not shutil.which("atc"):
        die("找不到 atc。先 source <CANN>/set_env.sh")
    if not soc:
        die("没给 soc_version。用 --soc <名字> 或设 FC2D_SOC；\n"
            "    合法的名字就是 $ASCEND_HOME_PATH/compiler/data/platform_config 下每个 .ini 的文件名")
    if not os.path.isfile(paths["bin"]):
        die("%s 还没生成 —— 先跑 --steps case" % paths["bin"])
    spec = read_spec(paths["bin"])
    os.makedirs(paths["omdir"], exist_ok=True)
    with open(paths["singleop"], "w") as f:
        json.dump(singleop_json(spec), f, indent=2)
        f.write("\n")
    print("  singleop: %s" % paths["singleop"])
    # atc 会在 --output 目录下生成 <something>.om。先清空，免得目录里留着上一次
    # 形状的 .om —— aclopSetModelDir 会把目录里所有 .om 都装进去，旧的那个可能
    # 反而先匹配上，跑出来的是上一个形状的结果。
    for fn in os.listdir(paths["omdir"]):
        if fn.endswith(".om"):
            os.remove(os.path.join(paths["omdir"], fn))
    cmd = ["atc", "--singleop=" + paths["singleop"], "--soc_version=" + soc,
           "--output=" + paths["omdir"], "--log=error"]
    print("  " + " ".join(cmd))
    rc = run(cmd)
    if rc != 0:
        die("atc 返回 %d（%s）。常见原因：算子包没装、soc_version 不对、"
            "或者这个形状被 tiling 拒了（看 ~/ascend/log/ 里的 OP_LOGE）" % (rc, c["name"]))
    oms = [fn for fn in os.listdir(paths["omdir"]) if fn.endswith(".om")]
    if not oms:
        die("atc 说成功了但 %s 下没有 .om" % paths["omdir"])
    print("  产出: %s" % ", ".join(oms))


def do_run(c, paths, device, extra):
    runner = os.path.join(HERE, "run_fused_conv2d.py")
    if not os.path.isfile(runner):
        die("找不到 run_fused_conv2d.py")
    if not os.path.isfile(paths["bin"]):
        die("%s 还没生成" % paths["bin"])
    if not os.path.isdir(paths["omdir"]):
        die("%s 不存在 —— 先跑 --steps om" % paths["omdir"])
    cmd = [sys.executable, runner, paths["bin"], "FusedConv2d", str(device), paths["omdir"]] + extra
    print("  " + " ".join(cmd))
    return run(cmd)


def main():
    ap = argparse.ArgumentParser(description="fused_conv2d 上板验证的总驱动")
    ap.add_argument("cases", nargs="?", default=os.path.join(HERE, "cases.json"),
                    help="形状清单 json（默认 cases.json）")
    ap.add_argument("--only", action="append", default=[],
                    help="只做名字匹配的这些 case，可给多次")
    ap.add_argument("--steps", default="check,case,om,run",
                    help="做哪几步，逗号分隔：check / case / om / run")
    ap.add_argument("--l1", type=int, default=1024 * 1024,
                    help="形状预检时按多大的 L1 算（默认真机的 1 MB）")
    ap.add_argument("--soc", default=os.environ.get("FC2D_SOC", ""),
                    help="atc 的 soc_version（也可以用环境变量 FC2D_SOC）")
    ap.add_argument("--device", type=int, default=None, help="设备号，默认取 json 里的 device")
    ap.add_argument("--outdir", default=None, help="产物目录，默认 <json 所在目录>/out")
    ap.add_argument("--list", action="store_true", help="只列出会做哪些")
    ap.add_argument("--sync-shape-header", metavar="PATH", default=None,
                    help="从 ops-nn 重拷一份 fused_conv2d_shape.h（形状预检用的副本）。"
                         "算子那边改了几何就跑一次，否则预检的结论会和 atc 不一致。")
    ap.add_argument("--keep-going", action="store_true",
                    help="某条失败了继续做下一条（默认第一条失败就停）")
    # 不用 argparse.REMAINDER：位置参数 cases 之后的**所有**东西都会被它吞掉，
    # 连 --steps 都进不了 args.steps —— 踩过，表现是 --steps check 完全不起作用。
    ap.add_argument("--run-arg", action="append", default=[],
                    help="原样转给 run_fused_conv2d.py 的参数，可给多次（如 --run-arg --dry-run）")
    args = ap.parse_args()

    if args.sync_shape_header:
        src = args.sync_shape_header
        if not os.path.isfile(src):
            die("找不到 %s" % src)
        dst = os.path.join(HERE, "fused_conv2d_shape.h")
        with open(dst) as f:
            old = f.read()
        # 保留副本顶上那段说明（它讲的是「这是副本、怎么重拷」），只换正文。
        marker = "// ==========================================================================="
        end = old.find(marker, old.find(marker) + 1)
        banner = old[:end + len(marker) + 1] if end > 0 else ""
        with open(src) as f:
            body = f.read()
        with open(dst, "w") as f:
            f.write(banner + body)
        print("已从 %s 重拷 fused_conv2d_shape.h" % src)
        print("两个 C++ 工具会在下次用到时自动重编（头比二进制新）")
        return 0

    if not os.path.isfile(args.cases):
        die("找不到 %s" % args.cases)
    with open(args.cases) as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            die("%s 不是合法的 json：%s" % (args.cases, e))
    if isinstance(doc, list):
        doc = {"cases": doc}
    raw = doc.get("cases")
    if not isinstance(raw, list) or not raw:
        die("%s 里没有 cases 数组" % args.cases)
    cases = []
    for i, c in enumerate(raw):
        one = normalise(c, i)
        if one["dtype"] == "both":
            # dtype: "both" 展开成两条。同一个形状在两条通路上各跑一遍是这个
            # 任务的要求，写两遍 json 只会让它们慢慢长歪。
            for dt in ("fp16", "int8"):
                d = dict(one)
                d["dtype"] = dt
                d["name"] = "%s_%s" % (one["name"], dt)
                cases.append(d)
        else:
            cases.append(one)
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        die("case 的 name 有重复：%s" % [n for n in names if names.count(n) > 1])
    if args.only:
        wanted = set(args.only)
        missing = wanted - set(names)
        if missing:
            die("--only 里这些名字不在清单里：%s" % sorted(missing))
        cases = [c for c in cases if c["name"] in wanted]

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ("check", "case", "om", "run")]
    if bad:
        die("--steps 里不认识的步骤 %s，只能是 check / case / om / run" % bad)

    device = args.device if args.device is not None else int(doc.get("device", 0))
    # 产物按 json 的文件名分目录。两份 json 里可能有同名的 case（比如 cases.json
    # 和 profile.json 都有 base_fp16）而形状不同，共用一个 out/ 会互相覆盖，
    # 然后单跑 --steps om 时拿到的是另一份 json 的数据 —— 而且不报错。
    stem = os.path.splitext(os.path.basename(args.cases))[0]
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.cases)), "out", stem)
    extra = list(args.run_arg)

    print("清单: %s" % args.cases)
    print("步骤: %s   产物: %s   设备: %d" % (" -> ".join(steps), outdir, device))
    print("共 %d 条:" % len(cases))
    for c in cases:
        ph1, pw1, ph2, pw2 = c["pads"]
        print("  %-16s %-5s n%d %d->%d->%d  %dx%d  k%dx%d s%d/%d p%d,%d/%d,%d  bias=%s relu=%s/%s"
              % (c["name"], c["dtype"], c["n"], c["ci"], c["cout1"], c["cout2"], c["hi"], c["wi"],
                 c["kernel"][0], c["kernel"][1], c["strides"][0], c["strides"][1],
                 ph1, pw1, ph2, pw2, c["bias"], c["relu"][0], c["relu"][1]))
    if args.list:
        return 0

    failed = []
    for c in cases:
        paths = {
            "bin": os.path.join(outdir, c["name"], "case.bin"),
            "omdir": os.path.join(outdir, c["name"], "om"),
            "singleop": os.path.join(outdir, c["name"], "singleop.json"),
        }
        os.makedirs(os.path.join(outdir, c["name"]), exist_ok=True)
        step("%s  (%s)" % (c["name"], c["dtype"]))
        try:
            if "check" in steps:
                do_check(c, args.l1)
            if "case" in steps:
                do_case(c, paths)
            if "om" in steps:
                do_om(c, paths, args.soc)
            if "run" in steps:
                rc = do_run(c, paths, device, extra)
                if rc != 0:
                    raise RuntimeError("run_fused_conv2d.py 返回 %d" % rc)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - 驱动脚本，要把哪条失败了说清楚
            print("\n[X] %s 失败：%s" % (c["name"], e))
            failed.append(c["name"])
            if not args.keep_going:
                print("    （加 --keep-going 可以跳过失败的继续做下一条）")
                break

    print("\n==== 汇总 ====")
    print("  做了 %d 条，失败 %d 条" % (len(cases), len(failed)))
    if failed:
        print("  失败: %s" % ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
