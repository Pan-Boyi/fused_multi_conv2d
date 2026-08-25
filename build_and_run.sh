#!/bin/bash
# FusedConv2d @ 5102 单算子验证 —— 编译并运行。
#
#   ./build_and_run.sh [device_id]
#
# 前提：FusedConv2d 已经作为**内置算子**编进 CANN 包并安装好（不是自定义 vendor 包）。
# 所以这里链的是内置的 libopapi*.so，不是 libcust_opapi.so。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV="${1:-0}"
: "${ASCEND_HOME_PATH:?先 source <CANN安装路径>/set_env.sh}"

ARCH_DIR="$ASCEND_HOME_PATH"
[ -d "$ARCH_DIR/include" ] || ARCH_DIR="$ASCEND_HOME_PATH/$(uname -m)-linux"
INC="$ARCH_DIR/include"
LIB="$ARCH_DIR/lib64"

# ---- 0) 算子名大小写：FusedConv2d 还是 FusedConv2D -------------------------
EXTRA=""
if [ -f "$INC/aclnnop/aclnn_fused_conv2d.h" ]; then
    echo "[0] 找到 aclnn_fused_conv2d.h        -> OpType 是 FusedConv2d"
elif [ -f "$INC/aclnnop/aclnn_fused_conv2_d.h" ]; then
    echo "[0] 找到 aclnn_fused_conv2_d.h       -> OpType 是 FusedConv2D（大写 D）"
    EXTRA="-DFUSED_CONV2D_UPPER_D=1"
else
    echo "[0] FAIL: $INC/aclnnop/ 下没有 aclnn_fused_conv2*.h"
    echo "         包里没有这个算子的 aclnn 接口。检查 op_host/CMakeLists.txt 是不是 ACLNNTYPE aclnn。"
    exit 1
fi

# ---- 1) 符号在不在内置库里 ------------------------------------------------
echo "[1] 内置 opapi 库里的符号："
FOUND=0
for so in "$LIB"/libopapi*.so; do
    if nm -D --defined-only "$so" 2>/dev/null | grep -qi "aclnnFusedConv2"; then
        echo "    $(basename "$so"): $(nm -D --defined-only "$so" | grep -i aclnnFusedConv2 | awk '{print $3}' | tr '\n' ' ')"
        FOUND=1
    fi
done
[ "$FOUND" = 1 ] || { echo "    FAIL: 所有 libopapi*.so 里都没有 aclnnFusedConv2* —— 包没编进去"; exit 1; }

# ---- 2) 有没有自定义 vendor 把内置算子盖住 --------------------------------
echo "[2] 自定义 vendor 遮挡检查："
if [ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
    echo "    警告: ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    if find "$ASCEND_CUSTOM_OPP_PATH" -iname '*fused_conv2d*' -print -quit 2>/dev/null | grep -q .; then
        echo "    FAIL: 自定义包里也有 FusedConv2d，它会**盖过**内置的那个。"
        echo "          验证内置算子时请先 unset ASCEND_CUSTOM_OPP_PATH。"
        exit 1
    fi
    echo "    (自定义路径里没有同名算子，不影响)"
else
    echo "    ASCEND_CUSTOM_OPP_PATH 未设置 —— 走内置算子，正确"
fi
VCFG="${ASCEND_OPP_PATH:-$ARCH_DIR/opp}/vendors/config.ini"
[ -f "$VCFG" ] && { echo "    $VCFG:"; sed 's/^/      /' "$VCFG"; }

# ---- 3) golden 自检（不碰设备）--------------------------------------------
echo "[3] golden 自检（纯 host，不碰设备）："
# 先做个完整性检查：header 少了尾巴(比如传输被截断)会表现成一句和 golden
# 毫无关系的 "undefined reference to `main'"。
GLINES=$(grep -c "" "$HERE/fused_conv2d_int8_golden.h")
grep -q "^int main()" "$HERE/fused_conv2d_int8_golden.h" || {
    echo "    FAIL: fused_conv2d_int8_golden.h 只有 $GLINES 行且没有 main() —— 文件不完整，重新拿一份"
    exit 1
}
# 用真正的 .cpp 入口，不用 `-x c++ header.h`：后者依赖编译器把 .h 当源文件处理，
# gcc 11 和 gcc 13 的行为并不一致，不一致时就报上面那句 undefined reference to `main'。
g++ -std=c++17 -O2 "$HERE/golden_selfcheck.cpp" -o "$HERE/golden_selfcheck" -I"$HERE"
# 显式接住退出码。直接 `prog | sed` 在 set -e + pipefail 下会让脚本静默退出，
# 你只会看到 golden 的输出然后什么都没有。
set +e
"$HERE/golden_selfcheck" | sed 's/^/    /'
GRC=${PIPESTATUS[0]}
set -e
if [ "$GRC" != 0 ]; then
    echo "    golden 自检返回 $GRC（SELF-CHECK: FAIL 或 layout round-trip 失败）"
    echo "    先解决它 —— golden 不可信的话，后面比对出来的任何结论都没有意义。"
    exit 1
fi

# ---- 4) 编译验证程序 ------------------------------------------------------
echo "[4] 编译验证程序："
g++ -std=c++17 -O2 $EXTRA "$HERE/test_aclnn_fused_conv2d.cpp" -o "$HERE/test_aclnn_fused_conv2d" \
    -I"$HERE" -I"$INC" -I"$INC/aclnnop" \
    -L"$LIB" -lascendcl -lnnopbase -lopapi \
    -Wl,-rpath,"$LIB"
echo "    built $HERE/test_aclnn_fused_conv2d"

# ---- 5) 跑 ----------------------------------------------------------------
echo "[5] 真机运行 (device $DEV)："
"$HERE/test_aclnn_fused_conv2d" "$DEV"
