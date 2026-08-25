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

# ---- 3) 编译 --------------------------------------------------------------
# libascendcl 会拉进 libascend_dump.so，后者依赖**驱动**库 libascend_hal.so。
# 有卡的机器上它在 /usr/local/Ascend/driver/lib64 下，把这些目录喂给 -rpath-link，
# 否则链接会报一堆 "undefined reference to drvXxx / halXxx" —— 那不是代码问题。
# 注意这里要用 if，不能写成 `[ -d "$d" ] && DRVFLAGS+=(...)`：
# 在 set -e 下，最后一次循环如果目录不存在，整个 for 的退出码就是 1，
# 脚本会**一声不吭地退出** —— 上一版就栽在这儿。
DRVFLAGS=()
for d in /usr/local/Ascend/driver/lib64 \
         /usr/local/Ascend/driver/lib64/driver \
         /usr/local/Ascend/driver/lib64/common; do
    if [ -d "$d" ]; then
        DRVFLAGS+=(-L"$d" -Wl,-rpath-link,"$d" -Wl,-rpath,"$d")
    fi
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
