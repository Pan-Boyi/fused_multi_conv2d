#!/bin/bash
# FusedConv2d 单算子验证 —— aclopExecuteV2 版(不需要 aclnn 头)。
#
#   ./build_and_run_aclop.sh [op_type] [device_id]
#   默认: FusedConv2d 0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTYPE="${1:-FusedConv2d}"
DEV="${2:-0}"
: "${ASCEND_HOME_PATH:?先 source <CANN安装路径>/set_env.sh}"

# CANN 有两种目录布局：$ASCEND_HOME_PATH 下直接有 include/，或者在 <arch>-linux/ 下。
ARCH="$ASCEND_HOME_PATH"
[ -d "$ARCH/include" ] || ARCH="$ASCEND_HOME_PATH/$(uname -m)-linux"
OPP="${ASCEND_OPP_PATH:-$ARCH/opp}"
echo "[env] ARCH=$ARCH"

# ---- 1) 确认算子类型名拼写 ------------------------------------------------
echo "[1] 算子信息库里登记的 op type："
HITS=$(grep -rhoE '"[A-Za-z0-9_]*[Ff]used[Cc]onv2[dD]"' "$OPP"/built-in/op_impl/ai_core/tbe/config/ 2>/dev/null \
       | tr -d '"' | sort -u || true)
if [ -n "$HITS" ]; then
    echo "$HITS" | sed 's/^/    /'
    if ! echo "$HITS" | grep -qx "$OPTYPE"; then
        echo "    警告: 你传的是 \"$OPTYPE\"，上面没有完全一致的。用上面列出的名字重跑。"
    fi
else
    echo "    (没在 config/ 里搜到，继续，但如果 aclopExecuteV2 报算子未找到就是名字不对)"
fi

# ---- 2) golden 自检（纯 host，不碰设备）-----------------------------------
echo "[2] golden 自检："
g++ -std=c++17 -O2 -DFUSED_CONV2D_GOLDEN_MAIN -DFUSED_CONV2D_GOLDEN_INT8_OUT=1 \
    -x c++ "$HERE/fused_conv2d_int8_golden.h" -o "$HERE/golden_selfcheck"
"$HERE/golden_selfcheck" | sed 's/^/    /'

# ---- 3) 编译 --------------------------------------------------------------
# libascendcl 会拉进 libascend_dump.so，后者依赖**驱动**库 libascend_hal.so。
# 有卡的机器上它在 /usr/local/Ascend/driver/lib64 下，把这些目录喂给 -rpath-link，
# 否则链接会报一堆 "undefined reference to drvXxx / halXxx" —— 那不是代码问题。
DRVFLAGS=()
for d in /usr/local/Ascend/driver/lib64 \
         /usr/local/Ascend/driver/lib64/driver \
         /usr/local/Ascend/driver/lib64/common; do
    [ -d "$d" ] && DRVFLAGS+=(-L"$d" -Wl,-rpath-link,"$d" -Wl,-rpath,"$d")
done
echo "[3] 编译（驱动库目录: ${DRVFLAGS[*]:-<无，机器上没装驱动？>}）："
g++ -std=c++17 -O2 "$HERE/test_aclop_fused_conv2d.cpp" -o "$HERE/test_aclop_fused_conv2d" \
    -I"$HERE" -I"$ARCH/include" \
    -L"$ARCH/lib64" "${DRVFLAGS[@]}" -lascendcl \
    -Wl,-rpath,"$ARCH/lib64"
echo "    built $HERE/test_aclop_fused_conv2d"

# ---- 4) 跑 ----------------------------------------------------------------
echo "[4] 真机运行 (opType=$OPTYPE, device=$DEV)："
timeout 300 "$HERE/test_aclop_fused_conv2d" "$OPTYPE" "$DEV"
