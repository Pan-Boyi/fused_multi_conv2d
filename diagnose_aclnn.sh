#!/bin/bash
# 回答"为什么 aclnnop 下没有 aclnn_fused_conv2*.h"，并告诉你该走哪条路。
# 只读，不改任何东西。
set -uo pipefail
: "${ASCEND_HOME_PATH:?先 source <CANN安装路径>/set_env.sh}"

ARCH="$ASCEND_HOME_PATH"
[ -d "$ARCH/include" ] || ARCH="$ASCEND_HOME_PATH/$(uname -m)-linux"
OPP="${ASCEND_OPP_PATH:-$ARCH/opp}"

echo "ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
echo "解析到的 arch 目录 = $ARCH"
echo "ASCEND_OPP_PATH  = $OPP"
echo

echo "=== 1. aclnn 头文件(整个安装里找，不只 aclnnop/) ==="
find "$ASCEND_HOME_PATH" -name 'aclnn_fused_conv2*.h' 2>/dev/null | head -5 || true
echo "  aclnnop/ 下的算子头总数: $(ls "$ARCH/include/aclnnop"/*.h 2>/dev/null | wc -l)"
echo "  (作为参照，看看 conv 系有哪些 —— 大多数卷积算子本来就没有自己的 aclnn 头)"
ls "$ARCH/include/aclnnop"/ 2>/dev/null | grep -iE '^aclnn_(conv|.*conv)' | head -8
echo

echo "=== 2. aclnn 符号(有头没库、或有库没头，都得知道) ==="
for so in "$ARCH"/lib64/libopapi*.so "$ARCH"/lib64/libcust_opapi.so; do
    [ -f "$so" ] || continue
    syms=$(nm -D --defined-only "$so" 2>/dev/null | grep -i 'aclnnFusedConv2' | awk '{print $NF}' | tr '\n' ' ')
    [ -n "$syms" ] && echo "  $(basename "$so"): $syms"
done
echo "  (上面没输出 = 包里没有 aclnn 接口)"
echo

echo "=== 3. 算子本身在不在(这才是能不能跑的关键) ==="
echo "  --- 算子信息库里的 op type ---"
grep -ril 'FusedConv2' "$OPP"/built-in/op_impl/ai_core/tbe/config/ 2>/dev/null | head -5
echo "  --- 编出来的 kernel 二进制 ---"
find "$OPP"/built-in/op_impl/ai_core/tbe/kernel -iname '*fused_conv2*' 2>/dev/null | head -5
echo "  --- op proto / tiling ---"
find "$OPP"/built-in -iname '*fused_conv2*' 2>/dev/null | grep -vE '/kernel/|/config/' | head -5
echo

echo "=== 4. 有没有自定义 vendor 抢在前面 ==="
echo "  ASCEND_CUSTOM_OPP_PATH = ${ASCEND_CUSTOM_OPP_PATH:-<未设置>}"
[ -f "$OPP/vendors/config.ini" ] && sed 's/^/  /' "$OPP/vendors/config.ini"
find "$OPP/vendors" -iname '*fused_conv2*' 2>/dev/null | head -3
echo

echo "==================== 结论 ===================="
HDR=$(find "$ASCEND_HOME_PATH" -name 'aclnn_fused_conv2*.h' 2>/dev/null | head -1)
OPI=$(grep -ril 'FusedConv2' "$OPP"/built-in/op_impl/ai_core/tbe/config/ 2>/dev/null | head -1)
if [ -n "$HDR" ]; then
    echo "有 aclnn 头: $HDR"
    echo "  -> 用 ./build_and_run.sh（aclnn 版）"
elif [ -n "$OPI" ]; then
    echo "没有 aclnn 头，但算子已经注册进算子信息库："
    echo "  $OPI"
    echo "  -> 这是正常的：仓里大多数算子(Conv2DV2 等)都没有自己的 aclnn 接口。"
    echo "  -> 用 aclopExecuteV2 那一版，按算子类型字符串调用，不需要头文件："
    echo "       g++ -std=c++17 -O2 test_aclop_fused_conv2d.cpp -o test_aclop_fused_conv2d \\"
    echo "           -I. -I$ARCH/include -L$ARCH/lib64 -lascendcl -Wl,-rpath,$ARCH/lib64"
    echo "       ./test_aclop_fused_conv2d FusedConv2d 0"
else
    echo "算子信息库里也找不到 FusedConv2 —— 包里根本没有这个算子。"
    echo "  -> 先确认 --pkg / 整仓构建时它真的被编进去了，再谈验证。"
fi
