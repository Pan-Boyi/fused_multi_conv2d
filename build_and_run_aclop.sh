#!/bin/bash
# FusedConv2d 单算子验证 —— aclopExecuteV2 版（不需要 aclnn 头）。
#
#   ./build_and_run_aclop.sh [op_type] [device_id]
#   默认: FusedConv2d 0
#
# 刻意**不用** `set -e`。前几版用了，结果任何一个返回非零的命令（哪怕只是一次
# 目录不存在的判断、或者一个 grep 没匹配上）都会让脚本一声不吭地退出，
# 现象就是"跑到某一行之后什么都不打印了"，极难定位。这里改成每一步显式检查、
# 显式报错，宁可啰嗦。
#
# 想看每条命令，跑: bash -x ./build_and_run_aclop.sh

OPTYPE="${1:-FusedConv2d}"
DEV="${2:-0}"

# 交叉编译用（在 x86 开发机上编、拿到 aarch64 板子上跑）：
#   CXX=aarch64-linux-gnu-g++ CANN_ARCH=aarch64 ./build_and_run_aclop.sh
# 不设就是本机编本机跑。
CXX="${CXX:-g++}"
CANN_ARCH="${CANN_ARCH:-$(uname -m)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"

die() { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

# ---------------------------------------------------------------- 0) 环境
step "0) 环境"
[ -n "$ASCEND_HOME_PATH" ] || die "ASCEND_HOME_PATH 没设。先 source <CANN安装路径>/set_env.sh"
ARCH="$ASCEND_HOME_PATH"
[ -d "$ARCH/include" ] || ARCH="$ASCEND_HOME_PATH/$CANN_ARCH-linux"
[ -d "$ARCH/include" ] || die "在 $ASCEND_HOME_PATH 下找不到 include/，ASCEND_HOME_PATH 指错了？"
OPP="${ASCEND_OPP_PATH:-$ARCH/opp}"
echo "  脚本目录 HERE = $HERE"
echo "  ARCH          = $ARCH"
echo "  OPP           = $OPP"
echo "  本机 uname -m  = $(uname -m)"
echo "  目标 CANN_ARCH = $CANN_ARCH"
echo "  CXX           = $CXX  ($($CXX --version 2>/dev/null | head -1))"
if ! command -v "$CXX" >/dev/null 2>&1; then
    die "找不到编译器 $CXX"
fi

# ---------------------------------------------------------------- 1) 文件齐不齐
step "1) 检查所需文件"
MISSING=0
for f in fused_conv2d_int8_golden.h golden_selfcheck.cpp test_aclop_fused_conv2d.cpp; do
    if [ -f "$HERE/$f" ]; then
        printf '  %-32s %s 行\n' "$f" "$(wc -l < "$HERE/$f" 2>/dev/null)"
    else
        echo "  $f  ** 缺失 **"
        MISSING=1
    fi
done
[ "$MISSING" = 0 ] || die "上面标 ** 缺失 ** 的文件要和脚本放在同一个目录"

# golden 头必须完整：截断的话后面会报一句和 golden 毫无关系的
# "undefined reference to \`main'"
if ! grep -q "^int main()" "$HERE/fused_conv2d_int8_golden.h"; then
    die "fused_conv2d_int8_golden.h 里找不到 main() —— 文件被截断了，重新拿一份"
fi
echo "  golden 头完整性 OK"

# ---------------------------------------------------------------- 2) 算子类型名
step "2) 算子信息库里登记的 op type"
HITS=$(grep -rhoE '[A-Za-z0-9_]*[Ff]used[Cc]onv2[dD]' "$OPP"/built-in/op_impl/ai_core/tbe/config/ 2>/dev/null | sort -u)
if [ -n "$HITS" ]; then
    echo "$HITS" | sed 's/^/  /'
    if ! echo "$HITS" | grep -qx "$OPTYPE"; then
        echo "  注意: 你传的是 \"$OPTYPE\"，上面没有完全一致的项。aclopExecuteV2 是精确匹配。"
    fi
else
    echo "  (在 $OPP/built-in/.../config/ 下没搜到 FusedConv2*)"
    echo "  如果第 5 步报算子未找到，就是类型名不对或者算子没编进包。"
fi

# ---------------------------------------------------------------- 3) golden 自检
step "3) golden 自检（纯 host，不碰设备）"
# 这一步固定用**本机** g++：它只是在开发机上验证 golden 自己算得对，
# 和目标板子的架构无关，别跟着 CXX 走。
rm -f "$HERE/golden_selfcheck"
g++ -std=c++17 -O2 -ffp-contract=off "$HERE/golden_selfcheck.cpp" -o "$HERE/golden_selfcheck" -I"$HERE"
[ $? -eq 0 ] || die "golden 自检编译失败（上面是编译器的话）"
[ -x "$HERE/golden_selfcheck" ] || die "编译报告成功但没产出可执行文件？"

"$HERE/golden_selfcheck"
GRC=$?
if [ "$GRC" -ne 0 ]; then
    die "golden 自检返回 $GRC（SELF-CHECK: FAIL 或 layout round-trip 失败）。
    golden 不可信的话，后面比对出来的任何结论都没有意义，先解决它。"
fi
echo "  golden 自检通过"

# ---------------------------------------------------------------- 4) 编译
step "4) 编译验证程序"
# -ffp-contract=off: aarch64 上 gcc 默认允许把 a*b+c 融成 fmadd,结果和分开算不一样。
# golden 里目前没有可融的表达式(唯一的浮点乘是 (float)acc*scale,后面没有加法),
# 所以这只是把"换个架构 golden 会不会变"这个疑问彻底钉死,不改变 x86 上的行为。
# libascendcl 会拉进 libascend_dump.so，后者依赖**驱动**库 libascend_hal.so。
# 有卡的机器上它在 /usr/local/Ascend/driver/lib64 下。少了这个路径，链接会吐一堆
# "undefined reference to drvXxx / halXxx" —— 那不是代码问题。
DRVFLAGS=()
for d in /usr/local/Ascend/driver/lib64 \
         /usr/local/Ascend/driver/lib64/driver \
         /usr/local/Ascend/driver/lib64/common \
         /usr/local/Ascend/driver/lib64/stub; do
    if [ -d "$d" ]; then
        DRVFLAGS+=(-L"$d" -Wl,-rpath-link,"$d" -Wl,-rpath,"$d")
        echo "  驱动库目录: $d"
    fi
done
[ ${#DRVFLAGS[@]} -gt 0 ] || echo "  (没找到 /usr/local/Ascend/driver/lib64，如果链接报 drvXxx 未定义就是缺它)"

rm -f "$HERE/test_aclop_fused_conv2d"
"$CXX" -std=c++17 -O2 -ffp-contract=off "$HERE/test_aclop_fused_conv2d.cpp" -o "$HERE/test_aclop_fused_conv2d" \
    -I"$HERE" -I"$ARCH/include" \
    -L"$ARCH/lib64" "${DRVFLAGS[@]}" -lascendcl \
    -Wl,-rpath,"$ARCH/lib64"
[ $? -eq 0 ] || die "编译/链接失败（上面是编译器的话）"
echo "  built $HERE/test_aclop_fused_conv2d"

# 产物到底是给哪个架构的？这一条能挡掉"编译过了，一跑 Exec format error"。
BINMACH=$(readelf -h "$HERE/test_aclop_fused_conv2d" 2>/dev/null | sed -n 's/^ *Machine: *//p')
BINSIZE=$(wc -c < "$HERE/test_aclop_fused_conv2d" 2>/dev/null)
echo "  ELF Machine   = ${BINMACH:-<readelf 不可用>}   size = ${BINSIZE:-?} 字节"
case "$(uname -m)" in
    x86_64)  HOSTMACH="X86-64" ;;
    aarch64) HOSTMACH="AArch64" ;;
    *)       HOSTMACH="" ;;
esac
RUNNABLE_HERE=1
if [ -n "$BINMACH" ] && [ -n "$HOSTMACH" ] && ! echo "$BINMACH" | grep -qi "$HOSTMACH"; then
    RUNNABLE_HERE=0
    echo
    echo "  [!] 这个可执行文件不是给本机（$(uname -m)）的，本机跑会报"
    echo "      \"cannot execute binary file: Exec format error\"。"
    echo "      把**源码**（4 个文件）拷到目标机器上，在目标机器上重跑本脚本；"
    echo "      拷可执行文件是没用的。"
fi

# ---------------------------------------------------------------- 5) 跑
step "5) 真机运行  opType=$OPTYPE  device=$DEV"
if [ "$RUNNABLE_HERE" = 0 ]; then
    die "产物架构和本机不一致，跳过运行（原因见上一步）。"
fi
timeout 300 "$HERE/test_aclop_fused_conv2d" "$OPTYPE" "$DEV"
RC=$?
echo
if [ "$RC" -eq 124 ]; then
    die "超时 300s —— 芯片多半被别人占着。看看 /dev/davinci$DEV 有没有别的进程打开着，或换一颗。"
fi
[ "$RC" -eq 0 ] || die "验证失败，退出码 $RC（上面有具体是哪一类）"
echo "[√] 全部通过"
