#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fused_multi_conv2d 上板验证的总驱动 —— **改一个 json 就能换形状**。

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
形状变成运行期之后，一个形状要两样东西对齐：
    .bin        输入数据 + golden
    .onnx / .om 图的形状 + 属性（属性在编译期就烘进 om 了）
对不上的表现是数值不对或者 atc 直接拒，而不是某个好懂的报错。

这个脚本让两者**都从同一处派生**：json 里的一条 case -> gen_case 的命令行 ->
.bin 头部的 spec -> .onnx -> om。中间没有第二份形状。

（历史：原来编 om 走 `atc --singleop`，那条路上属性要在**执行期**逐个下发、
由 ACL 连 attr 一起匹配 .om，于是「三者有一处不一致 -> 报成算子没找到
100024 / MatchOpModel fail」。换成图 om 之后属性是编译期烘进去的，
这一整类失败不存在了。）
"""
import argparse
import json
import re
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# gen_case.cpp 里 SpecIdx 的顺序。**两边必须一致**，改那边就得改这边。
# 末尾的 kh2 / kw2 是 conv2 的核，**只能往后加**：老的 .bin 在这两个位置上是 0，
# 而那时两层必然同核，所以退回 kh / kw 就是对的 —— 相同核可按此回退；
# ABI 8 retains logical NCHW filters and records the unified effective shift.
# 注意这张表是被 _check_spec_keys_in_sync() 正则逐字比对的，**列表里不能写注释**。
SPEC_KEYS = [
    "n", "ci", "hi", "wi", "cout1", "cout2", "kh", "kw", "s1", "s2",
    "ph1", "pw1", "ph2", "pw2", "elem_bytes", "bias", "relu1", "relu2",
    "shift1", "shift2", "ho2", "wo2", "safe_f1", "safe_f2", "out_fp16",
    "dtype_mode", "kh2", "kw2",
]
SPEC_N = 32
HDR_FMT = "<8sIIQQQ" + "%di" % SPEC_N + "II"
HDR_LEN = struct.calcsize(HDR_FMT)
CASE_VERSION = 8

DTYPE_FP16 = 0
DTYPE_INT8 = 1
DTYPE_S8F16 = 2
DTYPE_A16W8 = 3


# run_fused_multi_conv2d.py 里有**同一份** SPEC_KEYS 的抄本。它必须能单独跑在板子机上
# （do_push 只拷 case.bin / .om / .om.fc2d.json / run_fused_multi_conv2d.py 四个文件过去），所以没法
# import 这边的定义。两份一旦长歪，读到的就是相邻字段的值 —— 不报错，只是全错。
# 加了 out_fp16 那次就是这么被绊了一下：这边加了、那边没加，表现是「golden 输出
# 1548288 字节，按 774144 个 fp16 元素应为 774144」，跟字段清单看不出关系。
def _check_spec_keys_in_sync():
    path = os.path.join(HERE, "run_fused_multi_conv2d.py")
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        m = re.search(r"SPEC_KEYS = \[(.*?)\]", fh.read(), re.S)
    if m is None:
        return
    other = [x.strip().strip('"') for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if other != SPEC_KEYS:
        die("SPEC_KEYS 和 run_fused_multi_conv2d.py 里那份对不上：\n"
            "    这边 %s\n    那边 %s\n"
            "  两边读的是同一个 .bin，对不上会静默读错字段。" % (SPEC_KEYS, other))


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


_check_spec_keys_in_sync()


def step(msg):
    print("\n==== %s ====" % msg)
    sys.stdout.flush()


def run(argv, log_path=None):
    """Stream subprocess output and optionally retain the same output on disk."""
    sys.stdout.flush()
    if log_path is None:
        return subprocess.call(argv)
    with open(log_path, "w") as log:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


# ---------------------------------------------------------------- json -> case
DEFAULTS = {
    "dtype": "fp16",
    "n": 1, "ci": 32, "hi": 288, "wi": 112,
    "cout1": 64, "cout2": 96,
    "kernel": [3, 3],
    "strides": [1, 1, 1, 1],
    "pads": [0, 0, 0, 0, 0, 0, 0, 0],
    "bias": False,
    "enable_relu": [True, True],
    "fixed_shift_value": [0, 0],
    "data_format": [0, 0],
    "round_mode": [0, 0],
    "pad_mode": [0, 0],
}


# Public enum codes, independent of the device instruction's enum values.
MAPPED_ENUM_VALUES = {
    "data_format": ("NCHW", "NHWC"),
    "round_mode": ("rint", "round"),
    "pad_mode": ("SPECIFIC", "SAME", "VALID", "SAME_UPPER", "SAME_LOWER", "NOTSET"),
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
    # s8f16 = int8 进 / fp16 出；a16w8 = fp16 fmap + int8 filters，mid 为 int8。
    # both 只展开 fp16 + int8 这两条（那是「同一形状两条通路都跑一遍」的意思），
    # s8f16 要显式写 —— 它多一个输入，不是前两条的变体。
    if out["dtype"] not in ("fp16", "int8", "s8f16", "a16w8", "both"):
        die("case %s 的 dtype 只能是 fp16 / int8 / s8f16 / a16w8 / both" % out["name"])
    # kernel 接受两种长度，和 pads 同一条约定：
    #   [kh, kw]                 两个卷积同一个核（老写法）
    #   [kh1, kw1, kh2, kw2]     两层分开给，比如 1x1 接 3x3
    k = out["kernel"]
    if not isinstance(k, list) or len(k) not in (2, 4):
        die("case %s 的 kernel 要写成 [kh, kw] 或 [kh1, kw1, kh2, kw2]" % out["name"])
    if len(k) == 2:
        out["kernel"] = [k[0], k[1], k[0], k[1]]
    s = out["strides"]
    if not isinstance(s, list) or len(s) != 4 or any(type(v) is not int or v < 1 or v > 2147483647 for v in s):
        die("case %s: strides requires four positive integers [H1,W1,H2,W2]" % out["name"])
    if s[0] != s[1] or s[2] != s[3]:
        die("case %s: each convolution requires equal H/W strides" % out["name"])
    p = out["pads"]
    if not isinstance(p, list) or len(p) != 8 or any(type(v) is not int or v < 0 or v > 2147483647 for v in p):
        die("case %s: pads requires eight nonnegative integers [T1,B1,L1,R1,T2,B2,L2,R2]" % out["name"])
    if any(p[i] != p[i+1] for i in range(0, 8, 2)):
        die("case %s: padding must be symmetric on each axis" % out["name"])
    r = out["enable_relu"]
    if not isinstance(r, list) or len(r) != 2 or any(v not in (0,1) for v in r):
        die("case %s: enable_relu supports two modes, each 0 or 1" % out["name"])
    fs = out["fixed_shift_value"]
    if not isinstance(fs, list) or len(fs) != 2 or any(type(v) is not int or not 0 <= v <= 58 for v in fs):
        die("case %s: fixed_shift_value requires two integers in [0,58]" % out["name"])
    if out["dtype"] == "a16w8" and fs[0] > 29:
        die("case %s: A16W8 fixed_shift_value[0] must be in [0,29]" % out["name"])
    for name, mapping in MAPPED_ENUM_VALUES.items():
        values = out[name]
        if not isinstance(values, list) or len(values) != 2 or any(type(v) is not int for v in values):
            die("case %s: %s requires two integer enum codes" % (out["name"], name))
        for layer, value in enumerate(values):
            if value < 0 or value >= len(mapping):
                die("case %s: %s[%d]=%d is an invalid enum code" % (out["name"], name, layer, value))
            if value != 0:
                die("case %s: %s[%d]=%d (%s) is not implemented; only 0 (%s) is supported" %
                    (out["name"], name, layer, value, mapping[value], mapping[0]))
    return out


def gen_case_argv(c, out_path):
    ph1, pw1, ph2, pw2 = c["pads"][::2]
    return [
        os.path.join(HERE, "gen_case"), out_path,
        "--dtype", c["dtype"],
        "--n", str(c["n"]), "--ci", str(c["ci"]), "--hi", str(c["hi"]), "--wi", str(c["wi"]),
        "--cout1", str(c["cout1"]), "--cout2", str(c["cout2"]),
        "--kh", str(c["kernel"][0]), "--kw", str(c["kernel"][1]),
        "--kh2", str(c["kernel"][2]), "--kw2", str(c["kernel"][3]),
        "--s1", str(c["strides"][0]), "--s2", str(c["strides"][2]),
        "--ph1", str(ph1), "--pw1", str(pw1), "--ph2", str(ph2), "--pw2", str(pw2),
        "--bias", "1" if c["bias"] else "0",
        "--relu1", "1" if c["enable_relu"][0] else "0", "--relu2", "1" if c["enable_relu"][1] else "0",
        "--shift1", str(c["fixed_shift_value"][0]), "--shift2", str(c["fixed_shift_value"][1]),
    ]


# ---------------------------------------------------------------- 读 .bin 的 spec
def read_spec(path):
    """把 .bin 头部的 spec 读出来。**.onnx 只从这里派生**，不从 json 派生 ——
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


# ---------------------------------------------------------------- 两个 C++ 工具
# 源码进仓，二进制不进（.gitignore 里就写着 gen_case / fc2d_geom）。所以
# **每一次新克隆的第一次运行都会缺它们**，而且改过 fused_multi_conv2d_shape.h 之后
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
    「该重拷 fused_multi_conv2d_shape.h 了」的信号。

    没有这一步的话，形状不合法要等到 atc 失败才知道，而 atc 的原因只在
    ~/ascend/log/ 里，找起来很费劲。
    """
    exe = ensure_tool("fc2d_geom")
    # dtype 原样传下去：fc2d_geom 也认 a16w8，会分别按 fp16 x、int8 filter/mid
    # 以及 q1/dq2 两张 per-channel 表计算 L1，预检不能漏掉这些真实占用。
    ph1, pw1, ph2, pw2 = c["pads"][::2]
    argv = [exe, "--dtype", c["dtype"],
            "--n", str(c["n"]), "--ci", str(c["ci"]), "--hi", str(c["hi"]), "--wi", str(c["wi"]),
            "--cout1", str(c["cout1"]), "--cout2", str(c["cout2"]),
            "--kh", str(c["kernel"][0]), "--kw", str(c["kernel"][1]),
            "--kh2", str(c["kernel"][2]), "--kw2", str(c["kernel"][3]),
            "--s1", str(c["strides"][0]), "--s2", str(c["strides"][2]),
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
    """Compile the unfused source and verify both installed production passes."""
    if not shutil.which("atc"):
        die("找不到 atc。先 source <CANN>/set_env.sh")
    if not soc:
        die("没给 soc_version。用 --soc 或 FC2D_SOC")
    if not os.path.isfile(paths["bin"]):
        die("%s 还没生成 —— 先跑 --steps case" % paths["bin"])
    import tempfile
    from pathlib import Path
    import fc2d_onnx
    import fc2d_fusion as fusion
    import run_fused_multi_conv2d as runner
    from tools.verify_fc2d_atc_tiling import compilation_environment, check_capture
    frozen = runner._file_sha256(paths["bin"])
    spec = read_spec(paths["bin"])
    source_sha = None
    omdir = Path(paths["omdir"])
    omdir.mkdir(parents=True, exist_ok=True)
    # Invalidate previous runnable artifacts even if source generation later fails.
    for pattern in ("*.om.fc2d.json", "*.om"):
        for old in omdir.glob(pattern):
            old.unlink()
    audit = Path(tempfile.mkdtemp(prefix="fusion_", dir=str(omdir.parent)))

    def unchanged():
        if runner._file_sha256(paths["bin"]) != frozen:
            die("case SHA256 changed during compilation; regenerate case and OM")
        if source_sha is not None and runner._file_sha256(paths["onnx"]) != source_sha:
            die("source ONNX SHA256 changed during compilation; regenerate OM")

    def invoke(command, log):
        # Keep run() as the orchestration seam used by workflow tests.
        print("  " + " ".join(command), flush=True)
        rc = run(command, log_path=str(log))
        unchanged()
        if rc != 0:
            die("ATC 返回 %d；检查已安装 CANN 的两阶段融合 Pass、SoC 及算子注册。产物: %s" % (rc, audit))

    def dump(model, destination):
        invoke(["atc", "--mode=1", "--om=" + str(model), "--json=" + str(destination)], destination.with_suffix(".log"))
        if not destination.is_file():
            raise ValueError("ATC did not export the compiled graph JSON: %s" % destination)
        return json.loads(destination.read_text())

    try:
        plan, checker = fc2d_onnx.write_onnx(paths["bin"], paths["onnx"])
        unchanged()
        source_sha = runner._file_sha256(paths["onnx"])
        names = [name for name, _, _ in fc2d_onnx.runtime_slots(plan)]
        if names != runner.model_input_slots(spec):
            raise ValueError("frontend and runner input contracts differ")
        print("  source: %s (%s); Conv -> ExtendConv2D -> FusedMultiConv2d" % (paths["onnx"], checker))
        base = ["atc", "--model=" + paths["onnx"], "--framework=5",
                "--soc_version=" + soc, "--input_format=NCHW",
                "--input_shape=" + fc2d_onnx.input_shape_arg(plan),
                "--op_compiler_cache_mode=disable", "--log=error"]
        # A diagnostic compilation with stage 2 disabled proves stage 1 produces
        # a real ExtendConv2D pair. It is never uploaded or executed.
        stage1_switch, final_switch = audit / "stage1_switch.json", audit / "fusion_switch.json"
        for path, state in ((stage1_switch, "off"), (final_switch, "on")):
            path.write_text(json.dumps({"Switch": {"GraphFusion": {fusion.FIRST_PASS_NAME: "on", fusion.PASS_NAME: state}}}, indent=2) + "\n")
        stage1 = audit / "stage1"
        invoke(base + ["--fusion_switch_file=" + str(stage1_switch), "--output=" + str(stage1)], audit / "stage1.log")
        stage1_graph = audit / "stage1_graph.json"
        first = fusion.check_stage1_graph(dump(stage1.with_suffix(".om"), stage1_graph), names)
        output = omdir / c["name"]
        capture = Path(os.environ.get("FC2D_ATC_TILING_CAPTURE", str(audit / "tiling_capture")))
        cache = audit / "cache"
        capture.mkdir(parents=True, exist_ok=True)
        cache.mkdir()
        with compilation_environment(capture, cache):
            # Recompile the same two-Conv source with the original second pass on.
            invoke(base + ["--fusion_switch_file=" + str(final_switch), "--output=" + str(output)], audit / "atc.log")
        models = list(omdir.glob("*.om"))
        if len(models) != 1:
            raise ValueError("ATC must generate exactly one fresh final OM")
        model_graph = audit / "model_graph.json"
        proof = fusion.check_compiled_graph(dump(models[0], model_graph), names)
        tiling = check_capture(spec, capture)
        proof.update(first, stage1_pass_name=fusion.FIRST_PASS_NAME, pass_name=fusion.PASS_NAME, pass_source=fusion.PR_URL,
                     source_conv_count=2, source_graph_sha256=source_sha,
                     stage1_graph_sha256=runner._file_sha256(stage1_graph),
                     model_graph_sha256=runner._file_sha256(model_graph), native_tiling=tiling)
        unchanged()
        (audit / "fusion_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
        binding = runner.write_model_binding(paths["bin"], str(models[0]),
                    expected_case_sha256=frozen, fusion_report=proof)
        print("  两阶段融合及真实 tiling 已验证；case/OM binding: %s" % binding)
        print("  融合证据: %s" % audit)
    except (OSError, ValueError) as exc:
        die("融合验证失败: %s。未生成可执行绑定；诊断目录: %s" % (exc, audit))


def do_run(c, paths, device, extra):
    runner = os.path.join(HERE, "run_fused_multi_conv2d.py")
    if not os.path.isfile(runner):
        die("找不到 run_fused_multi_conv2d.py")
    if not os.path.isfile(paths["bin"]):
        die("%s 还没生成" % paths["bin"])
    if not os.path.isdir(paths["omdir"]):
        die("%s 不存在 —— 先跑 --steps om" % paths["omdir"])
    cmd = [sys.executable, runner, paths["bin"], "FusedMultiConv2d", str(device), paths["omdir"]] + extra
    print("  " + " ".join(cmd))
    return run(cmd)


def main():
    ap = argparse.ArgumentParser(description="fused_multi_conv2d 上板验证的总驱动")
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
                    help="从 ops-nn 重拷一份 fused_multi_conv2d_shape.h（形状预检用的副本）。"
                         "算子那边改了几何就跑一次，否则预检的结论会和 atc 不一致。")
    ap.add_argument("--keep-going", action="store_true",
                    help="某条失败了继续做下一条（默认第一条失败就停）")
    # 不用 argparse.REMAINDER：位置参数 cases 之后的**所有**东西都会被它吞掉，
    # 连 --steps 都进不了 args.steps —— 踩过，表现是 --steps check 完全不起作用。
    ap.add_argument("--run-arg", action="append", default=[],
                    help="原样转给 run_fused_multi_conv2d.py 的参数，可给多次（如 --run-arg --dry-run）")
    args = ap.parse_args()

    if args.sync_shape_header:
        src = args.sync_shape_header
        if not os.path.isfile(src):
            die("找不到 %s" % src)
        dst = os.path.join(HERE, "fused_multi_conv2d_shape.h")
        with open(dst) as f:
            old = f.read()
        # 保留副本顶上那段说明（它讲的是「这是副本、怎么重拷」），只换正文。
        marker = "// ==========================================================================="
        end = old.find(marker, old.find(marker) + 1)
        banner = old[:end + len(marker) + 1] if end > 0 else ""
        with open(src) as f:
            body = f.read()
        common_header = "../../op_kernel/arch35/fused_multi_conv2d_geometry_common.h"
        common_include = '#include "%s"' % common_header
        if common_include in body:
            common_path = os.path.join(os.path.dirname(src), common_header)
            with open(common_path) as f:
                body = body.replace(common_include, f.read())
        with open(dst, "w") as f:
            f.write(banner + body)
        print("已从 %s 重拷 fused_multi_conv2d_shape.h" % src)
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
        ph1, pw1, ph2, pw2 = c["pads"][::2]
        k = c["kernel"]
        ktxt = ("%dx%d" % (k[0], k[1]) if k[:2] == k[2:] else "%dx%d>%dx%d" % tuple(k))
        print("  %-16s %-5s n%d %d->%d->%d  %dx%d  k%-9s s%d/%d p%d,%d/%d,%d  bias=%s relu=%s/%s"
              % (c["name"], c["dtype"], c["n"], c["ci"], c["cout1"], c["cout2"], c["hi"], c["wi"],
                 ktxt, c["strides"][0], c["strides"][2],
                 ph1, pw1, ph2, pw2, c["bias"], c["enable_relu"][0], c["enable_relu"][1]))
    if args.list:
        return 0

    failed = []
    for c in cases:
        paths = {
            "bin": os.path.join(outdir, c["name"], "case.bin"),
            "omdir": os.path.join(outdir, c["name"], "om"),
            # 编 om 的输入。
            "onnx": os.path.join(outdir, c["name"], c["name"] + ".onnx"),
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
                    raise RuntimeError("run_fused_multi_conv2d.py 返回 %d" % rc)
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
