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
# ld.lld 必须是**昇腾自带的那份**。
#
# 编 AI Core kernel 的最后一步是 `ld.lld -m aicorelinux`，aicorelinux 这个 emulation
# 只有 CANN 里 ccec_compiler/bishengir 带的那份 lld 认识。PATH 上要是先撞见系统的
# /usr/bin/ld.lld、或者 conda 环境里 llvm 带的那个，就会报
#     ld.lld: error: unknown emulation: aicorelinux
# 这不是算子的问题，是 PATH 顺序的问题。
# ---------------------------------------------------------------------------
step "0.5) ld.lld 是不是昇腾那份"
ASC_LLD=""
for d in "$ASCEND_HOME_PATH/compiler/ccec_compiler/bin" \
         "$ASCEND_HOME_PATH/compiler/bishengir/bin" \
         "$ASCEND_HOME_PATH/toolkit/toolchain/hcc/bin"; do
    if [ -x "$d/ld.lld" ]; then
        ASC_LLD="$d/ld.lld"
        echo "  昇腾的: $ASC_LLD"
        break
    fi
done
if [ -z "$ASC_LLD" ]; then
    ASC_LLD=$(find "$ASCEND_HOME_PATH" -maxdepth 6 -type f -name ld.lld -perm -u+x 2>/dev/null | head -1)
    [ -n "$ASC_LLD" ] && echo "  昇腾的: $ASC_LLD  (在非标准位置找到)"
fi
[ -n "$ASC_LLD" ] || die "在 $ASCEND_HOME_PATH 下一个 ld.lld 都找不到。
    CANN 的编译器组件(ccec_compiler / bishengir)没装全，或者 ASCEND_HOME_PATH 指错了。"

# 手动走一遍 PATH。`type -a` 在不同 shell 下行为不一致，这样最稳，
# 而且能把"谁排在谁前面"直接摆出来 —— 这正是要看的东西。
echo "  PATH 上的 ld.lld（按先后顺序）:"
NFOUND=0
OLDIFS="$IFS"; IFS=:
for d in $PATH; do
    [ -n "$d" ] || d="."
    if [ -x "$d/ld.lld" ]; then
        NFOUND=$((NFOUND + 1))
        if [ "$d/ld.lld" = "$ASC_LLD" ]; then
            echo "    $NFOUND. $d/ld.lld   <- 昇腾的"
        else
            echo "    $NFOUND. $d/ld.lld"
        fi
    fi
done
IFS="$OLDIFS"
[ "$NFOUND" -gt 0 ] || echo "    (PATH 上一个都没有)"

CUR_LLD=$(command -v ld.lld 2>/dev/null)
ASC_BIN=$(dirname "$ASC_LLD")
if [ "$CUR_LLD" != "$ASC_LLD" ]; then
    echo
    echo "  [!] PATH 上第一个 ld.lld 是 ${CUR_LLD:-<没有>}，不是昇腾那份。"
    echo "      这正是 \"unknown emulation: aicorelinux\" 的原因。已在本次运行内把"
    echo "      $ASC_BIN 提到 PATH 最前面。"
    echo "      要永久生效，在 source set_env.sh **之后**（注意顺序，conda activate"
    echo "      之类会重新往前插路径）加上："
    echo "          export PATH=\$ASCEND_HOME_PATH/$(echo "$ASC_BIN" | sed "s|^$ASCEND_HOME_PATH/||"):\$PATH"
    PATH="$ASC_BIN:$PATH"
    export PATH
    echo "      现在: $(command -v ld.lld)"
else
    echo "  OK，PATH 上第一个就是它"
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
