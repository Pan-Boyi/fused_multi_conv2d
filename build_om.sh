#!/bin/bash
# 在**装了带 FusedConv2d 的算子包**的 x86 机器上，把这个单算子离线编译成 .om。
#
#   SOC=<soc_version> ./build_om.sh
#   ./build_om.sh                      # 不传 SOC 时先帮你把候选列出来
#
# 为什么要这一步：aclopExecuteV2 是**运行时按算子类型字符串分发**的，算子的实现
# （kernel 二进制 + 算子信息库）必须在**执行的那台机器**上。编译验证程序并不会把
# 算子带过去 —— 那个可执行文件里一行 FusedConv2d 的代码都没有，只有一个字符串。
#
# 单算子离线模型是唯一能"在 A 机器上编译、拿到 B 机器上执行"的形态：atc 在这台
# 机器上把算子编成 .om（里面是给 5102 的 device 二进制），目标机只要有基础的 CANN
# 运行时就能加载执行，不需要装算子包。
#
# 不用 set -e，每步显式检查。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"
JSON="$HERE/fused_conv2d_singleop.json"
CASE="${CASE:-$HERE/fused_conv2d_case.bin}"
OUTDIR="${OUTDIR:-$HERE/om_out}"

die()  { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

step "0) 环境"
[ -n "$ASCEND_HOME_PATH" ] || die "ASCEND_HOME_PATH 没设。先 source <CANN安装路径>/set_env.sh"
command -v atc >/dev/null 2>&1 || die "找不到 atc。set_env.sh 里应当把 \$ASCEND_HOME_PATH/bin 加进 PATH"
echo "  atc  = $(command -v atc)"
echo "  本机 = $(uname -m)"
OPP="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
echo "  OPP  = $OPP"
[ -f "$JSON" ] || die "缺 $JSON"

# ---------------------------------------------------------------------------
# ld.lld 必须是**认识 aicorelinux 的那份**。
#
# 编 AI Core kernel 的最后一步是 `ld.lld -m aicorelinux`。这个 emulation 只有毕昇
# 编译器带的那份 lld 认识；系统的 /usr/bin/ld.lld、conda 里 llvm 带的、乃至 CANN
# 自己 tools/llvm/ 下那份通用 lld，都不认，撞上就是
#     ld.lld: error: unknown emulation: aicorelinux
#
# 各版本的目录布局不一样（ccec_compiler / bisheng_compiler / tools/llvm / <arch>-linux/bin
# 都见过），而且 ld.lld 通常是**符号链接**。所以这里不按目录名猜，直接把找得到的每一个
# 都探一遍：喂一个空输入让它走到 emulation 解析那一步，看它抱怨的是不是 aicorelinux。
# ---------------------------------------------------------------------------
step "0.5) 哪个 ld.lld 认识 aicorelinux"

# 返回 0 = 认识；1 = 不认识；2 = 这个二进制根本跑不起来
probe_lld() {
    _tmp=$(mktemp 2>/dev/null) || _tmp="/tmp/.lldprobe.$$"
    _out=$("$1" -m aicorelinux /dev/null -o "$_tmp" 2>&1)
    _rc=$?
    rm -f "$_tmp"
    case "$_out" in
        *"error while loading shared libraries"*|*"cannot open shared object"*) return 2 ;;
    esac
    if [ "$_rc" -eq 126 ] || [ "$_rc" -eq 127 ]; then
        return 2
    fi
    case "$_out" in
        *"nknown emulation"*|*"nsupported emulation"*|*"nvalid emulation"*) return 1 ;;
    esac
    return 0
}

# 候选：ASCEND_HOME_PATH 下所有的 ld.lld（不加 -type f，它们多半是符号链接）+ PATH 上的
CANDS=$(find "$ASCEND_HOME_PATH" -maxdepth 7 -name ld.lld 2>/dev/null)
OLDIFS="$IFS"; IFS=:
for d in $PATH; do
    [ -n "$d" ] || d="."
    [ -x "$d/ld.lld" ] && CANDS="$CANDS
$d/ld.lld"
done
IFS="$OLDIFS"

[ -n "$CANDS" ] || die "$ASCEND_HOME_PATH 下和 PATH 上都找不到 ld.lld。
    CANN 的编译器组件没装全，或者 ASCEND_HOME_PATH 指错了。"

GOOD_LLD=""
SEEN=""
# 注意：用 for 而不是 `... | while read`，管道会开子 shell，GOOD_LLD 出不来。
# 代价是路径里不能有空格 —— CANN 的安装路径不会有。
for f in $(echo "$CANDS" | sed '/^$/d' | sort -u); do
    [ -x "$f" ] || { printf '  %-58s %s\n' "$f" "(不可执行，跳过)"; continue; }
    real=$(readlink -f "$f" 2>/dev/null || echo "$f")
    case " $SEEN " in *" $real "*) dup=" (同 $real)" ;; *) dup=""; SEEN="$SEEN $real" ;; esac
    probe_lld "$f"
    case $? in
        0) printf '  %-58s [认识 aicorelinux]%s\n' "$f" "$dup"
           [ -n "$GOOD_LLD" ] || GOOD_LLD="$f" ;;
        1) printf '  %-58s  不认识%s\n' "$f" "$dup" ;;
        2) printf '  %-58s  跑不起来（缺依赖库）%s\n' "$f" "$dup" ;;
    esac
done

[ -n "$GOOD_LLD" ] || die "找到的 ld.lld 没有一个认识 aicorelinux。
    毕昇编译器组件(bisheng_compiler / ccec_compiler)没装，或者装的版本不对。
    上面列出的路径可以贴给我。"

GOOD_BIN=$(dirname "$GOOD_LLD")
CUR_LLD=$(command -v ld.lld 2>/dev/null)
echo
echo "  PATH 上第一个 ld.lld: ${CUR_LLD:-<没有>}"
if [ "$CUR_LLD" = "$GOOD_LLD" ]; then
    echo "  OK，正是认识 aicorelinux 的那个"
else
    echo "  [!] 不是认识 aicorelinux 的那个 —— 这就是报错的原因。"
    echo "      已在本次运行内把 $GOOD_BIN 提到 PATH 最前面。"
    echo "      要永久生效，在 source set_env.sh **之后**（conda activate / module load"
    echo "      之类会重新往前插路径，所以顺序很重要）加上："
    echo "          export PATH=$GOOD_BIN:\$PATH"
    PATH="$GOOD_BIN:$PATH"
    export PATH
    echo "      现在: $(command -v ld.lld)"
fi

step "1) 这台机器的包里到底有没有 FusedConv2d"
CFGDIR="$OPP/built-in/op_impl/ai_core/tbe/config"
HITS=$(grep -rhoE '"[A-Za-z0-9_]*[Ff]used[Cc]onv2[dD]"' "$CFGDIR" 2>/dev/null | tr -d '"' | sort -u)
if [ -n "$HITS" ]; then
    echo "$HITS" | sed 's/^/  算子信息库里: /'
else
    die "在 $CFGDIR 下没搜到 FusedConv2*。
    这台机器上的包也没有这个算子 —— 那 .om 也编不出来。先确认你装的是刚编出来的那个包。"
fi

step "2) 哪些 soc 编进了这个算子"
KDIR="$OPP/built-in/op_impl/ai_core/tbe/kernel"
CAND=""
if [ -d "$KDIR" ]; then
    for d in "$KDIR"/*/; do
        soc=$(basename "$d")
        if ls "$d" 2>/dev/null | grep -qi "fused_conv2d"; then
            echo "  $soc   <- 有 fused_conv2d 的 kernel"
            CAND="$CAND $soc"
        fi
    done
fi
[ -n "$CAND" ] || echo "  (在 $KDIR 下没按 soc 找到 fused_conv2d 的 kernel 目录)"

# platform_config 下每个 .ini 的**文件名**就是一个合法的 soc_version。
# 这是这台机器上唯一权威的清单 —— 别照着别处的文档猜。
PCDIR="$ASCEND_HOME_PATH/compiler/data/platform_config"
[ -d "$PCDIR" ] || PCDIR=$(find "$ASCEND_HOME_PATH" -maxdepth 5 -type d -name platform_config 2>/dev/null | head -1)

step "3) 这个 CANN 认哪些 soc_version"
if [ -d "$PCDIR" ]; then
    echo "  清单目录: $PCDIR"
    ls "$PCDIR" 2>/dev/null | sed 's/\.ini$//' | sed 's/^/    /'

    # 5102 / MC62 / dav-510 的那一项在哪。ini 里会带 soc 短名和 aicore 架构，
    # 直接按 510 / mc62 捞，比人肉在几十个里面认要稳。
    echo
    echo "  含 510 / mc62 字样的（5102 大概率就在这里）:"
    FOUND510=""
    for ini in "$PCDIR"/*.ini; do
        [ -f "$ini" ] || continue
        if grep -qiE '510|mc62' "$ini" 2>/dev/null; then
            sv=$(basename "$ini" .ini)
            FOUND510="$FOUND510 $sv"
            echo "    soc_version = $sv"
            grep -iE '510|mc62|short_soc|soc_version|ai_core_arch' "$ini" 2>/dev/null \
                | head -6 | sed 's/^/        /'
        fi
    done
    [ -n "$FOUND510" ] || echo "    (一个都没捞到 —— 这个 CANN 可能根本不支持 5102，那 .om 也编不出来)"
else
    echo "  找不到 platform_config 目录，没法列清单。"
    echo "  另一个办法: 故意传一个不存在的 soc_version，atc 的报错里通常会带上它认识的列表。"
fi

if [ -z "$SOC" ]; then
    echo
    echo "  没传 SOC。从上面挑出 5102 对应的那个，然后："
    echo "      SOC=<soc_version> ./build_om.sh"
    exit 0
fi

if [ -d "$PCDIR" ] && [ ! -f "$PCDIR/$SOC.ini" ]; then
    echo
    echo "  [!] $PCDIR 下没有 $SOC.ini —— 这个 soc_version 多半不合法，atc 会拒。"
    echo "      注意 soc_version 和算子编译时的 --soc=mc62 那个短名不是一回事。"
fi

step "4) atc 单算子编译  soc_version=$SOC"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# 定点定标（fixed_shift1/2）不能在这份 json 里手写死。
#
# ACL 匹配 .om 是拿 **op 类型 + 每个 tensor 的 shape/dtype/format + 全部 attr 的值**
# 一起匹配的。json 里写 26/22 而脚本按 .bin 的 header 下发 34/37，就匹配不上，而且
# 报出来的是 100024 / "算子没找到(161001)" —— 看着像算子没装，其实是属性对不上。
# 这个坑真踩过一次。
#
# 所以属性值只有 .bin 的 header 一个来源，这里现读现填，人不参与。
# ---------------------------------------------------------------------------
USED_JSON="$OUTDIR/singleop_used.json"
if [ -f "$CASE" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "$JSON" "$CASE" "$USED_JSON" <<'PYSYNC'
import json, struct, sys
tpl, case, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(case, "rb") as f:
    hdr = f.read(56)
if len(hdr) < 56 or hdr[:8] != b"FC2DCASE":
    sys.exit("case 文件头不对，读不出定标")
attrs = struct.unpack_from("<Q", hdr, 48)[0]     # header 里最后一个 uint64
shifts = {"fixed_shift1": attrs & 0xFF, "fixed_shift2": (attrs >> 8) & 0xFF}
desc = json.load(open(tpl, encoding="utf-8"))
hit = 0
for op in desc:
    for a in op.get("attr", []):
        if a.get("name") in shifts:
            a["value"] = shifts[a["name"]]
            hit += 1
if hit != 2:
    sys.exit("模板里没找齐 fixed_shift1/2（找到 %d 个）" % hit)
json.dump(desc, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("  定标取自 %s: fixed_shift1=%d fixed_shift2=%d"
      % (case, shifts["fixed_shift1"], shifts["fixed_shift2"]))
PYSYNC
    [ $? -eq 0 ] || die "从 $CASE 同步定标失败。
    要么 .bin 是旧版本的，要么模板 json 里没有 fixed_shift1/2。"
    JSON="$USED_JSON"
else
    [ -f "$CASE" ] || echo "  [!] 找不到 $CASE"
    command -v python3 >/dev/null 2>&1 || echo "  [!] 没有 python3"
    echo "  [!] 没法从 case 文件同步定标，直接用模板里写死的值。"
    echo "      模板和 .bin 的定标一旦不一致，执行时会报 100024 / 算子没找到 ——"
    echo "      那不是算子没装，是属性对不上。跑之前先自己核一遍："
    grep -E 'fixed_shift[12]' "$JSON" | sed 's/^/        /'
fi

echo "  json   = $JSON"
echo "  output = $OUTDIR"
echo
atc --singleop="$JSON" --soc_version="$SOC" --output="$OUTDIR" --log=error
RC=$?
echo
[ "$RC" -eq 0 ] || die "atc 返回 $RC。
    常见原因：
      unknown emulation: aicorelinux -> PATH 上的 ld.lld 不是昇腾那份，见第 0.5 步
      soc_version 填错          -> 回第 3 步的清单里挑
      算子不支持这个 shape/dtype -> tiling 只接受一种 shape，见步骤文档"

step "5) 产物"
FOUND=$(find "$OUTDIR" -name '*.om' 2>/dev/null)
[ -n "$FOUND" ] || die "atc 说成功了但没找到 .om？看看 $OUTDIR 下面有什么"
echo "$FOUND" | while read -r f; do
    printf '  %-60s %s 字节\n' "$f" "$(wc -c < "$f" | tr -d ' \t')"
done

cat <<TIP

  .om 里是给 $SOC 的 device 二进制，和 host 的 CPU 架构无关，
  所以在 x86 上编、拿到 aarch64 上跑是成立的。

  拷过去：

    scp -r $(basename "$OUTDIR") fused_conv2d_case.bin run_fused_conv2d.py <5102机器>:~/a5102_st/

  在 5102 机器上（第 4 个参数就是 .om 所在目录）：

    cd ~/a5102_st
    source <CANN安装路径>/set_env.sh
    python3 run_fused_conv2d.py fused_conv2d_case.bin FusedConv2d 0 $(basename "$OUTDIR")

  脚本会先 aclopSetModelDir($(basename "$OUTDIR")) 再调 aclopExecuteV2，
  运行时就从这个目录里找匹配的模型，不再要求设备上装算子包。
TIP
