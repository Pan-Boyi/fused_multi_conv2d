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

PCDIR="$ASCEND_HOME_PATH/compiler/data/platform_config"
if [ -d "$PCDIR" ]; then
    echo "  这个 CANN 支持的 soc_version（atc --soc_version 要填的就是这些之一）:"
    ls "$PCDIR" 2>/dev/null | sed 's/\.ini$//' | sed 's/^/    /'
fi

if [ -z "$SOC" ]; then
    echo
    echo "  没传 SOC。从上面挑出 5102 对应的那个，然后："
    echo "      SOC=<soc_version> ./build_om.sh"
    exit 0
fi

step "3) atc 单算子编译  soc_version=$SOC"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"
echo "  json   = $JSON"
echo "  output = $OUTDIR"
echo
atc --singleop="$JSON" --soc_version="$SOC" --output="$OUTDIR" --log=error
RC=$?
echo
[ "$RC" -eq 0 ] || die "atc 返回 $RC。
    常见原因：
      soc_version 填错          -> 回第 2 步的清单里挑
      算子不支持这个 shape/dtype -> tiling 只接受一种 shape，见步骤文档"

step "4) 产物"
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
