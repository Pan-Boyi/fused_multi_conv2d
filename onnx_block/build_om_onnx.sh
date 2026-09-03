#!/bin/bash
# 把一个 ONNX 图整体编成 .om（--framework=5）。
#
#   SOC=<soc_version> ./build_om_onnx.sh block.onnx
#   ./build_om_onnx.sh block.onnx            # 不传 SOC 时先把候选列出来
#   SHAPE="x:4,128,32,48;res:4,128,32,48" SOC=... ./build_om_onnx.sh block.onnx
#
# 和同目录上一层的 build_om.sh 的区别：那个编的是**单算子** json（--singleop），
# 这个编的是**整张图**。两者产出的 .om 用法也不同 —— 单算子的要 aclopExecuteV2，
# 整图的直接 aclmdlLoadFromFile + aclmdlExecute，也就是 ../run_om.py。
#
# 不用 set -e，每步显式检查。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"
MODEL="${1:-$HERE/block.onnx}"
OUTDIR="${OUTDIR:-$HERE/om_out}"
BASE="$(basename "$MODEL" .onnx)"

die()  { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

step "0) 环境"
[ -f "$MODEL" ] || die "找不到 $MODEL"
[ -n "$ASCEND_HOME_PATH" ] || die "ASCEND_HOME_PATH 没设。先 source <CANN安装路径>/set_env.sh"
command -v atc >/dev/null 2>&1 || die "找不到 atc"
echo "  atc   = $(command -v atc)"
echo "  model = $MODEL ($(wc -c < "$MODEL") 字节)"
OPP="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"

# ---------------------------------------------------------------------------
# ld.lld 必须是认识 aicorelinux 的那份。完整解释见 ../build_om.sh 里同名的一段
# —— 编图同样要编 TBE kernel，最后一步同样是 `ld.lld -m aicorelinux`。
# ---------------------------------------------------------------------------
step "0.5) 哪个 ld.lld 认识 aicorelinux"
probe_lld() {
    _tmp=$(mktemp 2>/dev/null) || _tmp="/tmp/.lldprobe.$$"
    _out=$("$1" -m aicorelinux /dev/null -o "$_tmp" 2>&1); _rc=$?
    rm -f "$_tmp"
    case "$_out" in *"error while loading shared libraries"*|*"cannot open shared object"*) return 2 ;; esac
    [ "$_rc" -eq 126 ] || [ "$_rc" -eq 127 ] && return 2
    case "$_out" in *"nknown emulation"*|*"nsupported emulation"*|*"nvalid emulation"*) return 1 ;; esac
    return 0
}
GOOD_LLD=""
for f in $(find "$ASCEND_HOME_PATH" -maxdepth 7 -name ld.lld 2>/dev/null | sort -u); do
    [ -x "$f" ] || continue
    probe_lld "$f"
    case $? in
        0) printf '  %-58s [认识]\n' "$f"; [ -n "$GOOD_LLD" ] || GOOD_LLD="$f" ;;
        1) printf '  %-58s  不认识\n' "$f" ;;
        2) printf '  %-58s  跑不起来\n' "$f" ;;
    esac
done
[ -n "$GOOD_LLD" ] || die "没有一个 ld.lld 认识 aicorelinux（毕昇编译器组件没装全）"
if [ "$(command -v ld.lld 2>/dev/null)" != "$GOOD_LLD" ]; then
    PATH="$(dirname "$GOOD_LLD"):$PATH"; export PATH
    echo "  [!] 已把 $(dirname "$GOOD_LLD") 提到 PATH 最前（本次运行内）"
fi

# ---------------------------------------------------------------------------
# AscendQuant / AscendDequant 不是标准 ONNX 算子，是 AMCT 量化后才有的昇腾自定义
# 算子。atc 能不能吃这张图，取决于**这台机器的 onnx 解析器有没有注册它们**。
# 编译失败时这一步的输出能立刻区分两种情况：图写错了，还是这台机器的包不带量化。
# 探不到不代表一定不行（有的版本把映射编进 .so 里，grep 不出来），所以只是警告。
# ---------------------------------------------------------------------------
step "1) 这台机器的 onnx 解析器认不认识 AscendQuant / AscendDequant"
for op in AscendQuant AscendDequant; do
    n=$(grep -rl "$op" "$OPP/built-in/framework" 2>/dev/null | head -3)
    if [ -z "$n" ]; then
        n=$(find "$ASCEND_HOME_PATH" -maxdepth 6 -name 'libfmk_onnx_parser*' -o -maxdepth 6 -name '*onnx*parser*.so' 2>/dev/null \
            | while read -r so; do strings "$so" 2>/dev/null | grep -qx "$op" && echo "$so"; done | head -2)
    fi
    if [ -n "$n" ]; then
        echo "  $op  找到:"; echo "$n" | sed 's/^/      /'
    else
        echo "  [!] $op  没搜到 —— 也可能只是编进 .so 里 grep 不出来。"
        echo "      如果下面 atc 报 'op type AscendQuant is not supported'，就是这里的问题:"
        echo "      这台机器的 CANN 没带量化算子的 onnx 映射，需要装 AMCT / 完整的 opp。"
    fi
done

step "2) 算子信息库里有没有对应的实现"
CFGDIR="$OPP/built-in/op_impl/ai_core/tbe/config"
for op in AscendQuant AscendDequant Conv2D; do
    c=$(grep -rho "\"$op\"" "$CFGDIR" 2>/dev/null | sort -u | head -1)
    printf '  %-16s %s\n' "$op" "${c:-** 没搜到 **}"
done

step "3) soc_version"
if [ -z "$SOC" ]; then
    echo "  没传 SOC。这台机器上编进算子包的候选:"
    ls "$OPP/built-in/op_impl/ai_core/tbe/kernel" 2>/dev/null | sed 's/^/    /'
    die "选一个再来：SOC=<上面某一个> $0 $MODEL"
fi
echo "  SOC = $SOC"

step "4) atc"
mkdir -p "$OUTDIR"
set -- --model="$MODEL" --framework=5 --output="$OUTDIR/$BASE" \
       --soc_version="$SOC" --input_format=NCHW --log=error
[ -n "$SHAPE" ] && set -- "$@" --input_shape="$SHAPE"
[ -n "$ATC_EXTRA" ] && set -- "$@" $ATC_EXTRA
echo "  atc \\"; for a in "$@"; do echo "      $a \\"; done; echo

atc "$@"
RC=$?
echo
if [ $RC -ne 0 ]; then
    echo "[X] atc 返回 $RC。常见原因："
    echo "    * 'op type AscendQuant is not supported' -> 见上面第 1 步"
    echo "    * 'Conv' 的类型约束报错          -> 解析器把它当浮点 Conv 了，说明这张图"
    echo "                                        的量化标记没被识别；确认 .onnx 确实是"
    echo "                                        AMCT 导出的，而不是手工拼的"
    echo "    * shape 里有动态维              -> 传 SHAPE=\"名字:4,128,32,48;...\""
    echo "    详细日志: ~/ascend/log/  或者加 --log=info 重跑"
    exit 1
fi
OM="$OUTDIR/$BASE.om"
[ -f "$OM" ] || die "atc 说成功了但没有 $OM"
echo "[OK] $OM  ($(wc -c < "$OM") 字节)"
cat <<TIP

  下一步（在有 5102 的机器上）：

    python3 ../run_om.py $OM
        先不带输入，只把模型的输入输出接口打出来 —— 顺序/字节数以它为准，
        不要以 onnx_golden.py 打印的顺序为准（atc 理论上保序，但值得核一眼）

    python3 ../run_om.py $OM io/in0.*.bin io/in1.*.bin --golden0=io/golden0.*.bin
TIP
