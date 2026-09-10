#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusedConv2d 上板执行 + profiling 的总驱动 —— **全部配置来自一个 json**。

    python3 run_profile.py profile.json                    # 全流程
    python3 run_profile.py profile.json --only base_fp16   # 只做某几条
    python3 run_profile.py profile.json --steps check,case,om   # 只做本地那几步
    python3 run_profile.py profile.json --steps push,prof,pull,export,parse
    python3 run_profile.py profile.json --list             # 只列出会做哪些
    python3 run_profile.py profile.json --print-remote-script --only base_fp16
                                                           # 看远端到底会跑什么

================================================================================
这个脚本替代了原来的 run_profile.sh，多做了三件事
================================================================================
1. **把 build_case 也包进来。** 原来 case.bin 得靠人提前放到远端，形状一变就要
   自己记得重新生成、重新拷。现在形状变了，case.bin、singleop.json、.om、执行时
   下发的属性一起重新产生。

2. **形状 / dtype / 可选属性全部从 json 来。** 原来 dtype 是命令行参数，形状写死
   在 gen_case 里，属性写死在两份手写的 singleop json 里。现在一条 case 长这样：

       {"name": "base", "dtype": "both", "ci": 32, "hi": 288, "wi": 112,
        "cout1": 64, "cout2": 96, "kernel": [3,3], "strides": [1,2],
        "pads": [1,1], "bias": true, "relu": [true,false], "fixed_shift": [42,42]}

   dtype 写 "both" 会自动展开成 fp16 和 int8 两条。

3. **一次可以跑很多条。** 原来一次一个 dtype，PROF 目录靠"当前 dtype"分。现在每条
   case 有自己的名字，本地产物落在 out/<name>/ 和 prof_out/<name>/，互不干扰，
   最后打一张横向对比表。

================================================================================
为什么形状必须从**同一处**派生
================================================================================
一个形状要三样东西严丝合缝：
    case.bin 里的输入和 golden
    .om（ACL 按 op 类型 + 每个张量的 shape/dtype/format + **全部属性的值** 匹配）
    执行时下发的属性
任何一处不一致，板上报的都是 100024 / MatchOpModel fail，翻译过来是「算子没找到」
—— 那条信息完全不指向真正的原因。这条链上折过好几次。

所以这里的派生链是单向的、没有分叉的：

    json 里的一条 case
      -> gen_case 的命令行
      -> case.bin 头部的 spec（32 个 int，形状 + 属性的唯一真相）
      -> singleop.json（从 spec 读，**不从 json 读**）
      -> 执行时下发的属性（run_fused_conv2d.py 也从 spec 读）

中间没有第二份形状。前四步的实现直接复用 fc2d.py，不另写一份。

================================================================================
登录远端
================================================================================
**json 里不放密码。** 原来的 run_profile.sh 把密码写在脚本里，那份脚本是要进仓的。
这里密码只从环境变量取（默认 FC2D_REMOTE_PASSWORD），用 sshpass -e 传给 ssh ——
不是 -p，-p 会把密码放进**命令行参数**，同一台机器上任何人 ps 一下就看见了。

方式由 remote.auth 定，三个值：

    "password"  这台机器**只认密码**，公钥登录是禁掉的（板子机常见）。
                缺环境变量或缺 sshpass 时直接停下并说缺什么。
    "key"       免密登录。
    "auto"      有密码且有 sshpass 就用密码，否则走密钥。**有陷阱**：在只认密码
                的机器上，没装 sshpass 会不声不响地退回密钥，而那条路必然失败，
                报出来是一句和认证毫无关系的「远端建目录失败（rc=255）」。

    read -s -p '远端密码: ' FC2D_REMOTE_PASSWORD && export FC2D_REMOTE_PASSWORD
    python3 run_profile.py profile.json

用 read 是为了不让密码进 shell 历史。凭据只在真要连远端时才检查 ——
--steps check,case,om 那几步不碰远端，不设密码也能跑。
"""
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys

import fc2d  # 形状归一化、gen_case 命令行、spec 解析、singleop.json 全部复用它

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_STEPS = ["check", "case", "om", "push", "prof", "pull", "export", "parse"]

# op_summary 里要看的列。原来的 run_profile.sh 就是这七个，保持一致。
DEFAULT_COLUMNS = [
    "total_exe_time(us)",
    "mac_exe_time(us)",
    "scalar_exe_time(us)",
    "mte1_exe_time(us)",
    "mte2_exe_time(us)",
    "mte3_exe_time(us)",
    "fixpipe_time(us)",
]

REMOTE_DEFAULTS = {
    "host": "",
    "user": "",
    "password_env": "FC2D_REMOTE_PASSWORD",
    # 登录方式。"password" / "key" / "auto"。
    # **有些板子机禁掉了公钥登录，只认密码。** 那种环境下 "auto" 是个陷阱：本机
    # 没装 sshpass 时它会不声不响地退回密钥，而密钥在那儿根本不可能成功 ——
    # 表现是一句和认证毫无关系的「远端建目录失败（rc=255）」。写死 "password"
    # 就会在开工之前直接说清楚缺什么。
    "auth": "auto",
    "dir": "",                 # 远端工作目录，脚本会在下面建 fc2d_run/<name>/
    "cann_env": "",            # 远端要 source 的 set_env.sh，空 = 不 source
    "device": 0,
    "repeat": 10,              # run_fused_conv2d.py 的 REPEAT，计时用
    "warmup": 0,
    "ssh_opts": ["-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null",
                 "-o", "LogLevel=ERROR"],
}

PROFILE_DEFAULTS = {
    "enabled": True,
    # 远端 msprof 可执行文件的**绝对路径，写死，不搜 PATH**。
    # 搜 PATH 有两种错法，而且都不响：远端那个 ssh 是非交互 shell，读不到
    # ~/.bashrc 里 source 的 set_env.sh，于是明明装了也说「PATH 里没有」；
    # 或者机器上有好几套 CANN，搜到的不是这次要用的那一套，数据照出、来源不明。
    "msprof": "/var/msprof",
    "tbe_work_test_suit": "",  # 本地 TbeWorkTestSuit，用它的 msprof.py export summary
    "columns": DEFAULT_COLUMNS,
}


def die(msg):
    print("\n[X] %s" % msg)
    sys.exit(1)


def step(msg):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)
    sys.stdout.flush()


def run(argv, **kw):
    sys.stdout.flush()
    return subprocess.call(argv, **kw)


def check_output(argv):
    sys.stdout.flush()
    return subprocess.run(argv, stdout=subprocess.PIPE, check=False).stdout.decode(errors="replace")


# ---------------------------------------------------------------- 本地 CANN 环境
def source_env(script):
    """把一个 set_env.sh source 进当前进程的环境。

    Python 没法直接 source shell 脚本，所以起一个 bash 把它 source 完再把整个
    environ 用 \\0 分隔打印出来，读回来合并。这样后面调 atc 就不用要求调用方
    自己先 source 了 —— 原来的 run_profile.sh 第 1 步做的就是这件事。
    """
    if not script:
        return
    if not os.path.isfile(script):
        die("CANN 环境脚本不存在: %s" % script)
    out = subprocess.run(
        ["bash", "-c", "set -a; source %s >/dev/null 2>&1; env -0" % shlex.quote(script)],
        stdout=subprocess.PIPE, check=False).stdout
    n = 0
    for item in out.split(b"\0"):
        if not item:
            continue
        k, _, v = item.decode(errors="replace").partition("=")
        if k and os.environ.get(k) != v:
            os.environ[k] = v
            n += 1
    print("[OK] 已 source %s，更新了 %d 个环境变量" % (script, n))


# ---------------------------------------------------------------- ssh / scp
class Remote(object):
    def __init__(self, cfg):
        self.host = cfg["host"]
        self.user = cfg["user"]
        self.dir = cfg["dir"]
        self.cann_env = cfg["cann_env"]
        self.device = int(cfg["device"])
        self.repeat = int(cfg["repeat"])
        self.warmup = int(cfg["warmup"])
        self.ssh_opts = list(cfg["ssh_opts"])
        self.target = "%s@%s" % (self.user, self.host) if self.user else self.host
        # 密码只从环境变量取，json 里不放 —— 那份 json 是要进仓的。
        self.password_env = cfg["password_env"]
        self.password = os.environ.get(cfg["password_env"], "")
        self.auth = cfg["auth"]
        if self.auth not in ("auto", "password", "key"):
            die("remote.auth 只能是 auto / password / key，当前是 %r" % self.auth)
        has_sshpass = shutil.which("sshpass") is not None

        # **不在这里 die。** 只跑本地几步（check / case / om，「在有 CANN 的机器上
        # 出数据和 .om，再把目录整个搬到板子机上」正是 README 写的用法）时根本不
        # 需要凭据，构造函数里一刀切会把那条路也堵死。缺什么先记下来，等 preflight
        # 真要连远端时再报。
        self.auth_missing = []
        if self.auth == "key":
            self.use_sshpass = False
        elif self.auth == "password":
            # 声明了只能走密码，就没有「退回密钥」这一说 —— 那条路在这种机器上
            # 必然失败，退回去只是把失败推迟成一条看不出原因的报错。
            missing = []
            if not self.password:
                missing.append("环境变量 %s 没设（json 里不放密码，那份 json 要进仓）"
                               % self.password_env)
            if not has_sshpass:
                missing.append("本机没有 sshpass（Ubuntu/Debian: sudo apt install -y sshpass;"
                               " CentOS/openEuler: sudo yum install -y sshpass）")
            self.auth_missing = missing
            self.use_sshpass = not missing
        else:  # auto
            self.use_sshpass = bool(self.password) and has_sshpass
            if self.password and not has_sshpass:
                print("[!] 设了 %s 但本机没有 sshpass，只能退回密钥。\n"
                      "    那台机器要是禁了公钥登录，这一步注定失败 —— 把 profile.json 的\n"
                      "    remote.auth 改成 \"password\"，就会在这里直接停下并说缺什么。"
                      % self.password_env)

    def _prefix(self):
        # -e 从环境变量 SSHPASS 读，不是 -p。-p 会把密码放进**命令行参数**，
        # 同一台机器上任何人 ps 一下就看见了。
        return ["sshpass", "-e"] if self.use_sshpass else []

    def _env(self):
        """给子进程的环境。只有走密码时才多一个 SSHPASS。"""
        if not self.use_sshpass:
            return None
        e = dict(os.environ)
        e["SSHPASS"] = self.password
        return e

    def ssh(self, remote_cmd, stdin_script=None, capture=False):
        argv = self._prefix() + ["ssh"] + self.ssh_opts + [self.target, remote_cmd]
        sys.stdout.flush()
        if stdin_script is not None:
            p = subprocess.run(argv, input=stdin_script.encode(), env=self._env(),
                               stdout=subprocess.PIPE if capture else None, check=False)
            return p.returncode, (p.stdout.decode(errors="replace") if capture else "")
        p = subprocess.run(argv, env=self._env(),
                           stdout=subprocess.PIPE if capture else None, check=False)
        return p.returncode, (p.stdout.decode(errors="replace") if capture else "")

    def scp_to(self, local, remote_path, recursive=False):
        argv = self._prefix() + ["scp"] + (["-r"] if recursive else []) + self.ssh_opts + \
            [local, "%s:%s" % (self.target, remote_path)]
        return run(argv, env=self._env())

    def scp_from(self, remote_path, local, recursive=False):
        argv = self._prefix() + ["scp"] + (["-r"] if recursive else []) + self.ssh_opts + \
            ["%s:%s" % (self.target, remote_path), local]
        return run(argv, env=self._env())

    def describe(self):
        how = "密码（sshpass -e）" if self.use_sshpass else "密钥"
        if self.auth == "password" and self.auth_missing:
            how = "密码（**还不能用**，见下）"
        if self.auth != "auto":
            how += "，auth=%s" % self.auth
        return "%s:%s  device=%d  repeat=%d  登录方式=%s" % (
            self.target, self.dir, self.device, self.repeat, how)


def preflight(rt):
    """在做任何本地重活之前先确认远端进得去。

    没有这一步的话，一次 ssh 认证失败要等 check / case / om 全跑完（om 那步几十秒
    起，还要 atc）才在 push 里冒出来，而且报出来是「远端建目录失败（rc=255）」——
    255 是 ssh 自己没连上，和「建目录」一点关系都没有，看着像远端权限问题。
    """
    if rt.auth_missing:
        die("remote.auth = \"password\"（这台机器只认密码，公钥是禁掉的），但还缺：\n    - "
            + "\n    - ".join(rt.auth_missing) +
            "\n\n  设密码用 read，别让它进 shell 历史：\n"
            "    read -s -p '远端密码: ' %s && export %s\n\n"
            "  只想在本机出数据和 .om、之后手工搬过去的话，加 --steps check,case,om 就行，"
            "那几步不碰远端。" % (rt.password_env, rt.password_env))
    rc, _ = rt.ssh("true")
    if rc == 0:
        return
    print()
    print("[X] 连不上远端 %s（ssh 返回 %d）" % (rt.target, rc))
    if rc == 255:
        print("    255 是 **ssh 自己**没进去（认证失败 / 主机连不上），不是远端命令跑挂了。")
        print("    ssh 自己的报错在上面几行，说明了是哪一种。")
        if rt.use_sshpass:
            print("    这次走的是密码（sshpass -e），密码取自环境变量 %s。" % rt.password_env)
            print("    密码错了 ssh 会说 Permission denied；主机不通会说 Connection timed out。")
        elif rt.auth == "key":
            print("    这次走的是密钥（remote.auth = \"key\"）。")
            print("    手工 ssh %s 能免密进去吗？不能的话先 ssh-copy-id。" % rt.target)
        else:
            print("    这次走的是**密钥** —— 因为 %s 没设，或者本机没有 sshpass。" % rt.password_env)
            print("    那台机器要是禁了公钥登录（不少板子机是这样），这条路怎么试都不会成功：")
            print("      1) profile.json 里把 remote.auth 写成 \"password\"")
            print("      2) sudo apt install -y sshpass   （或 yum install -y sshpass）")
            print("      3) read -s -p '远端密码: ' %s && export %s"
                  % (rt.password_env, rt.password_env))
    else:
        print("    ssh 连上了，但远端连一条 `true` 都没跑通 —— 登录 shell 有问题？")
    raise SystemExit(1)


# ---------------------------------------------------------------- 远端脚本
def remote_script(rt, case_name, om_name, use_msprof, msprof):
    """生成在远端跑的那段 bash。

    它自己做三件本地做不了的事：
      1. 检查 msprof / case / om 在不在，不在就明确说哪个不在。msprof 用的是
         profile.msprof 给死的绝对路径，**不搜 PATH**
      2. 执行前后各记一次 PROF_* 目录，用差集找出**本次**新生成的那个 ——
         目录名里带时间戳，靠"最新"去猜在并发或重跑时会拿错
      3. 只往 stdout 打一行 PROF 目录名，别的全走 stderr，这样调用方能直接读
    """
    wd = "%s/%s" % (rt.dir.rstrip("/"), case_name)
    lines = [
        "set -o pipefail",
        "cd %s || { echo '[REMOTE ERROR] 进不去 %s' >&2; exit 1; }" % (shlex.quote(wd), wd),
    ]
    if rt.cann_env:
        lines += [
            "if [ -f %s ]; then source %s >/dev/null 2>&1; "
            "else echo '[REMOTE ERROR] 找不到 %s' >&2; exit 1; fi"
            % (shlex.quote(rt.cann_env), shlex.quote(rt.cann_env), rt.cann_env),
        ]
    lines += [
        "for f in case.bin run_fused_conv2d.py %s; do" % shlex.quote(om_name),
        "  [ -f \"$f\" ] || { echo \"[REMOTE ERROR] 缺文件 $f\" >&2; exit 1; }",
        "done",
    ]
    if use_msprof:
        q = shlex.quote(msprof)
        lines += [
            # 路径是配置给死的，不搜 PATH，也不猜。只确认它真的能执行 ——
            # 这一条不是搜索，是为了让「路径写错了」当场说清楚，而不是变成
            # 一句 command not found 混在 msprof 自己的输出里。
            "MSPROF=%s" % q,
            "[ -x \"$MSPROF\" ] || { echo \"[REMOTE ERROR] $MSPROF 不存在或不可执行\" >&2; "
            "echo '  这个路径来自 profile.json 的 profile.msprof，改那里' >&2; exit 127; }",
            "echo \"[REMOTE] msprof = $MSPROF\" >&2",
            "BEFORE=$(mktemp); AFTER=$(mktemp)",
            "trap 'rm -f \"$BEFORE\" \"$AFTER\"' EXIT",
            "find . -maxdepth 1 -mindepth 1 -type d -name 'PROF_*' -printf '%f\\n' | sort > \"$BEFORE\"",
        ]
    lines += [
        "export REPEAT=%d" % rt.repeat,
        "export WARMUP=%d" % rt.warmup,
        "echo '[REMOTE] 开始执行' >&2",
    ]
    exec_cmd = "python3 run_fused_conv2d.py case.bin FusedConv2d %d %s" % (
        rt.device, shlex.quote(om_name))
    if use_msprof:
        exec_cmd = "\"$MSPROF\" " + exec_cmd
    lines += [
        "%s >&2" % exec_cmd,
        "RC=$?",
        "echo \"[REMOTE] 执行返回码 $RC\" >&2",
    ]
    if use_msprof:
        lines += [
            "find . -maxdepth 1 -mindepth 1 -type d -name 'PROF_*' -printf '%f\\n' | sort > \"$AFTER\"",
            "NEW=$(comm -13 \"$BEFORE\" \"$AFTER\")",
            "CNT=$(printf '%s\\n' \"$NEW\" | sed '/^[[:space:]]*$/d' | wc -l)",
            "if [ \"$CNT\" -eq 0 ]; then echo '[REMOTE ERROR] 没有生成新的 PROF_* 目录' >&2; exit 2; fi",
            "if [ \"$CNT\" -gt 1 ]; then echo '[REMOTE ERROR] 一次生成了多个 PROF_*:' >&2; "
            "printf '%s\\n' \"$NEW\" >&2; exit 3; fi",
            # kernel 就算跑挂了，只要 PROF 出来了就继续导 —— 挂掉时的流水图往往
            # 正是要看的东西。
            "if [ \"$RC\" -ne 0 ]; then echo \"[REMOTE 警告] 执行返回 $RC，但 PROF 已生成，继续\" >&2; fi",
            "printf '%s\\n' \"$NEW\"",
        ]
    else:
        lines += ["exit $RC"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 各步骤
def do_push(rt, c, paths, om_file):
    # 下面要 rm -rf 这个目录。远端路径是 json 给的，先把明显危险的挡掉 ——
    # remote.dir 空着的话 wd 会变成 "/<case 名>"，那条 rm 就打到根目录下面去了。
    if not rt.dir.startswith("/") or rt.dir.rstrip("/").count("/") < 1:
        raise RuntimeError("remote.dir 必须是一个绝对路径且不能是根目录附近，当前是 %r" % rt.dir)
    wd = "%s/%s" % (rt.dir.rstrip("/"), c["name"])
    rc, _ = rt.ssh("rm -rf %s && mkdir -p %s" % (shlex.quote(wd), shlex.quote(wd)))
    if rc != 0:
        raise RuntimeError(
            "远端建目录失败（rc=%d）%s" % (rc,
            "—— 255 是 ssh 自己没连上（认证失败/主机不可达），不是 mkdir 失败" if rc == 255
            else "—— ssh 进去了，是远端 mkdir 没成功：%s 下有写权限吗？" % rt.dir))
    # 每条 case 一个干净目录。**不复用**：aclopSetModelDir 会把目录下所有 .om 都
    # 装进去，留着上一个形状的那个，它可能反而先匹配上，跑出来的是上个形状的结果，
    # 而且不报错。
    for f in (paths["bin"], om_file, os.path.join(HERE, "run_fused_conv2d.py")):
        if not os.path.isfile(f):
            raise RuntimeError("本地缺文件 %s" % f)
    if rt.scp_to(paths["bin"], "%s/case.bin" % wd) != 0:
        raise RuntimeError("拷 case.bin 失败")
    if rt.scp_to(om_file, "%s/%s" % (wd, os.path.basename(om_file))) != 0:
        raise RuntimeError("拷 .om 失败")
    # run_fused_conv2d.py 每次都拷。case.bin 的格式和它是配套的，远端留着一份旧的
    # 会报「case 文件版本 N，本脚本认 M」，而那时人往往已经在查算子了。
    if rt.scp_to(os.path.join(HERE, "run_fused_conv2d.py"), "%s/run_fused_conv2d.py" % wd) != 0:
        raise RuntimeError("拷 run_fused_conv2d.py 失败")
    print("[OK] 已拷到 %s:%s" % (rt.target, wd))


def do_prof(rt, c, om_file, use_msprof, msprof, show_script=False):
    script = remote_script(rt, c["name"], os.path.basename(om_file), use_msprof, msprof)
    if show_script:
        print("---- 远端将执行 ----")
        print(script)
        print("--------------------")
    rc, out = rt.ssh("bash -s", stdin_script=script, capture=True)
    if not use_msprof:
        if rc != 0:
            raise RuntimeError("远端执行返回 %d" % rc)
        return None
    if rc != 0:
        raise RuntimeError("远端 profiling 返回 %d" % rc)
    prof = out.strip().replace("\r", "")
    if not prof.startswith("PROF_"):
        raise RuntimeError("没能确定新生成的 PROF 目录，远端 stdout 是 %r" % out[:200])
    print("[OK] 新 PROF 目录: %s" % prof)
    return prof


def do_pull(rt, c, prof, prof_root):
    dst_dir = os.path.join(prof_root, c["name"])
    os.makedirs(dst_dir, exist_ok=True)
    local = os.path.join(dst_dir, prof)
    if os.path.exists(local):
        print("[!] 本地已有 %s，先删掉" % local)
        shutil.rmtree(local, ignore_errors=True)
    src = "%s/%s/%s" % (rt.dir.rstrip("/"), c["name"], prof)
    if rt.scp_from(src, dst_dir + "/", recursive=True) != 0:
        raise RuntimeError("拉回 PROF 失败")
    if not os.path.isdir(local):
        raise RuntimeError("拉回来了但 %s 不存在" % local)
    print("[OK] PROF 已拉到 %s" % local)
    return local


def do_export(local_prof, suite):
    if not suite:
        raise RuntimeError("没配 profile.tbe_work_test_suit，导不了 summary")
    if not os.path.isdir(suite):
        raise RuntimeError("TbeWorkTestSuit 目录不存在: %s" % suite)
    msprof_py = os.path.join(suite, "analysis", "msprof", "msprof.py")
    if not os.path.isfile(msprof_py):
        raise RuntimeError("找不到 %s" % msprof_py)
    rc = run([sys.executable, msprof_py, "export", "summary", "-dir", local_prof], cwd=suite)
    if rc != 0:
        raise RuntimeError("msprof.py export summary 返回 %d" % rc)


def find_op_summary(local_prof):
    d = os.path.join(local_prof, "mindstudio_profiler_output")
    if not os.path.isdir(d):
        raise RuntimeError("没有 mindstudio_profiler_output: %s" % d)
    cands = [os.path.join(d, f) for f in os.listdir(d)
             if f.startswith("op_summary_") and f.endswith(".csv")]
    if not cands:
        raise RuntimeError("%s 下没有 op_summary_*.csv" % d)
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def parse_summary(csv_path, columns):
    """读 op_summary，打一张表，并返回每一列的统计。

    REPEAT 次下发会得到多行。**报最小值**而不是平均：host 侧的抖动、别的进程抢核
    只会让某几次变慢，不会让它变快，所以最小值是对「这个 kernel 本身有多快」最少
    掺杂噪声的估计。平均值和行数也一并给出，好判断抖动有多大。
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV 没有表头: %s" % csv_path)
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise RuntimeError("op_summary 缺列 %s；实际有的列: %s"
                               % (missing, ", ".join(reader.fieldnames)))
        rows = list(reader)
    if not rows:
        raise RuntimeError("op_summary 一行数据都没有: %s" % csv_path)

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    rw = max(len("row"), len(str(len(rows))))
    header = ("%*s  " % (rw, "row")) + "  ".join("%*s" % (widths[c], c) for c in columns)
    print()
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(("%*d  " % (rw, i)) + "  ".join("%*s" % (widths[c], str(r.get(c, ""))) for c in columns))
    print("-" * len(header))

    stat = {"rows": len(rows)}
    for c in columns:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(c, "")))
            except (TypeError, ValueError):
                pass
        stat[c] = {"min": min(vals), "mean": sum(vals) / len(vals)} if vals else None
    return stat


# ---------------------------------------------------------------- 配置
def load_cases(raw):
    """把 json 里的 cases 数组归一化 + 展开 dtype: both。"""
    cases = []
    for i, one in enumerate(raw):
        c = fc2d.normalise(one, i)
        if c["dtype"] == "both":
            # dtype: "both" 展开成两条。同一个形状在两条通路上各跑一遍是常态，
            # 写两遍 json 只会让它们慢慢长歪。
            for dt in ("fp16", "int8"):
                d = dict(c)
                d["dtype"] = dt
                d["name"] = "%s_%s" % (c["name"], dt)
                cases.append(d)
        else:
            cases.append(c)
    names = [c["name"] for c in cases]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        die("case 的 name 有重复: %s" % dup)
    return cases


def load_config(path):
    if not os.path.isfile(path):
        die("找不到 %s" % path)
    with open(path) as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            die("%s 不是合法的 json: %s" % (path, e))
    if isinstance(doc, list):
        doc = {"cases": doc}

    remote = dict(REMOTE_DEFAULTS)
    remote.update(doc.get("remote", {}))
    unknown = [k for k in remote if k not in REMOTE_DEFAULTS]
    if unknown:
        die("remote 里有不认识的字段 %s；认识的是 %s" % (unknown, ", ".join(sorted(REMOTE_DEFAULTS))))

    prof = dict(PROFILE_DEFAULTS)
    prof.update(doc.get("profile", {}))
    unknown = [k for k in prof if k not in PROFILE_DEFAULTS]
    if unknown:
        die("profile 里有不认识的字段 %s；认识的是 %s" % (unknown, ", ".join(sorted(PROFILE_DEFAULTS))))

    raw = doc.get("cases")
    if not isinstance(raw, list) or not raw:
        die("%s 里没有 cases 数组" % path)
    return doc, remote, prof, load_cases(raw)


def main():
    ap = argparse.ArgumentParser(description="FusedConv2d 上板执行 + profiling，全部配置来自一个 json")
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "profile.json"),
                    help="配置 json（默认 profile.json）")
    ap.add_argument("--only", action="append", default=[], help="只做名字匹配的这些 case，可给多次")
    ap.add_argument("--cases", metavar="JSON", default=None,
                    help="从另一个 json 取 cases 数组，remote / profile / soc 仍用主配置里的。"
                         "比如拿 profile.json 的远端信息去跑 cases.json 的那批功能用例。")
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help="做哪几步，逗号分隔：" + " / ".join(ALL_STEPS))
    ap.add_argument("--soc", default=None, help="atc 的 soc_version，覆盖 json 里的 soc_version")
    ap.add_argument("--l1", type=int, default=1024 * 1024, help="形状预检按多大的 L1 算")
    ap.add_argument("--no-msprof", action="store_true",
                    help="远端只跑功能，不套 msprof；自动跳过 pull/export/parse")
    ap.add_argument("--print-remote-script", action="store_true", help="把远端要执行的 bash 打出来")
    ap.add_argument("--list", action="store_true", help="只列出会做哪些")
    ap.add_argument("--keep-going", action="store_true", help="某条失败了继续做下一条")
    args = ap.parse_args()

    doc, remote_cfg, prof_cfg, cases = load_config(args.config)
    if args.cases:
        # cases.json 那种只有形状清单、没有 remote 段的文件，用这条把它接进来。
        if not os.path.isfile(args.cases):
            die("找不到 %s" % args.cases)
        with open(args.cases) as f:
            other = json.load(f)
        raw = other.get("cases") if isinstance(other, dict) else other
        if not isinstance(raw, list) or not raw:
            die("%s 里没有 cases 数组" % args.cases)
        cases = load_cases(raw)
        print("case 清单来自 %s（remote / profile / soc 仍用 %s）" % (args.cases, args.config))
    if args.only:
        want = set(args.only)
        miss = want - {c["name"] for c in cases}
        if miss:
            die("--only 里这些名字不在清单里: %s" % sorted(miss))
        cases = [c for c in cases if c["name"] in want]

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        die("--steps 里不认识的步骤 %s，只能是 %s" % (bad, " / ".join(ALL_STEPS)))
    use_msprof = bool(prof_cfg["enabled"]) and not args.no_msprof
    if not use_msprof:
        steps = [s for s in steps if s not in ("pull", "export", "parse")]

    soc = args.soc or doc.get("soc_version", "") or os.environ.get("FC2D_SOC", "")
    # 产物按 json 的文件名分目录：两份 json 里同名但形状不同的 case 不会互相覆盖。
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    stem = os.path.splitext(os.path.basename(args.cases or args.config))[0]
    out_root = doc.get("outdir") or os.path.join(cfg_dir, "out", stem)
    prof_root = doc.get("profdir") or os.path.join(cfg_dir, "prof_out", stem)
    rt = Remote(remote_cfg)

    print("配置   : %s" % args.config)
    print("步骤   : %s" % " -> ".join(steps))
    print("产物   : %s" % out_root)
    if use_msprof:
        print("PROF   : %s" % prof_root)
    print("soc    : %s" % (soc or "(未给，出 om 时会报错)"))
    print("远端   : %s" % rt.describe())
    print("共 %d 条:" % len(cases))
    for c in cases:
        ph1, pw1, ph2, pw2 = c["pads"]
        print("  %-18s %-5s n%d %d->%d->%d %dx%d k%dx%d s%d/%d p%d,%d/%d,%d bias=%s relu=%s/%s shift=%d/%d"
              % (c["name"], c["dtype"], c["n"], c["ci"], c["cout1"], c["cout2"], c["hi"], c["wi"],
                 c["kernel"][0], c["kernel"][1], c["strides"][0], c["strides"][1],
                 ph1, pw1, ph2, pw2, c["bias"], c["relu"][0], c["relu"][1],
                 c["fixed_shift"][0], c["fixed_shift"][1]))
    if args.list:
        return 0

    # 远端先探一次。放在 om 之前 —— 进不去的话没必要先花几十秒编 .om。
    if {"push", "prof", "pull"} & set(steps):
        preflight(rt)

    # 本地 CANN。出 om 要 atc，所以在这里 source，调用方不必自己先 source。
    if "om" in steps:
        source_env(doc.get("local_cann_env", ""))

    failed, stats = [], {}
    for c in cases:
        paths = {
            "bin": os.path.join(out_root, c["name"], "case.bin"),
            "omdir": os.path.join(out_root, c["name"], "om"),
            "singleop": os.path.join(out_root, c["name"], "singleop.json"),
        }
        os.makedirs(os.path.join(out_root, c["name"]), exist_ok=True)
        step("%s   (%s)" % (c["name"], c["dtype"]))
        try:
            if "check" in steps:
                fc2d.do_check(c, args.l1)
            if "case" in steps:
                fc2d.do_case(c, paths)
            if "om" in steps:
                fc2d.do_om(c, paths, soc)

            om_file = None
            if {"push", "prof"} & set(steps):
                oms = [os.path.join(paths["omdir"], f)
                       for f in os.listdir(paths["omdir"])] if os.path.isdir(paths["omdir"]) else []
                oms = [f for f in oms if f.endswith(".om")]
                if not oms:
                    raise RuntimeError("%s 下没有 .om —— 先跑 --steps om" % paths["omdir"])
                if len(oms) > 1:
                    # do_om 每次都会先清空目录，所以这里出现多个只可能是人手动放的。
                    raise RuntimeError("%s 下有 %d 个 .om，说不清该用哪个: %s"
                                       % (paths["omdir"], len(oms), [os.path.basename(f) for f in oms]))
                om_file = oms[0]
                print("[OK] .om = %s" % os.path.basename(om_file))

            if "push" in steps:
                do_push(rt, c, paths, om_file)
            prof = None
            if "prof" in steps:
                prof = do_prof(rt, c, om_file, use_msprof, prof_cfg["msprof"],
                               args.print_remote_script)
            if "pull" in steps and prof:
                local_prof = do_pull(rt, c, prof, prof_root)
                if "export" in steps:
                    do_export(local_prof, prof_cfg["tbe_work_test_suit"])
                if "parse" in steps:
                    stats[c["name"]] = parse_summary(find_op_summary(local_prof), prof_cfg["columns"])
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - 驱动脚本，要把哪条失败了说清楚
            print("\n[X] %s 失败: %s" % (c["name"], e))
            failed.append(c["name"])
            if not args.keep_going:
                print("    （加 --keep-going 可以跳过失败的继续做下一条）")
                break

    if stats:
        cols = prof_cfg["columns"]
        step("横向对比（每列取 REPEAT 次里的最小值，单位 us）")
        name_w = max(len("case"), max(len(n) for n in stats))
        head = "%-*s  %5s  " % (name_w, "case", "rows") + "  ".join("%12s" % c[:12] for c in cols)
        print(head)
        print("-" * len(head))
        for n in stats:
            s = stats[n]
            cells = []
            for c in cols:
                v = s.get(c)
                cells.append("%12.3f" % v["min"] if v else "%12s" % "-")
            print("%-*s  %5d  " % (name_w, n, s["rows"]) + "  ".join(cells))
        print("-" * len(head))
        print("列名按 12 字符截断；完整列名见上面每条 case 的明细表。")

    print("\n" + "=" * 72)
    print("做了 %d 条，失败 %d 条" % (len(cases), len(failed)))
    if failed:
        print("失败: %s" % ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
