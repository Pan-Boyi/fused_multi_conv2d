#!/bin/bash
# 用 msprof 采一次 FusedConv2d 的设备侧耗时，然后把结果解析出来。
#
#   ./run_prof.sh [case.bin] [op_type] [device_id] [om_dir]
#   默认: fused_conv2d_case.bin  FusedConv2d  0  (无 om_dir)
#
# 环境变量:
#   REPEAT=20        采集期间额外下发多少次（默认 20，越多统计越稳）
#   OUT=prof_out     msprof 的输出目录
#   MSPROF_ARGS=...  覆盖默认的采集开关（各版本开关名不一样时用）
#
# 不用 set -e，每步显式检查。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"

CASE="${1:-fused_conv2d_case.bin}"
OPTYPE="${2:-FusedConv2d}"
DEV="${3:-0}"
OMDIR="${4:-}"

REPEAT="${REPEAT:-20}"
OUT="${OUT:-$HERE/prof_out}"

die()  { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

step "0) 环境"
[ -n "$ASCEND_HOME_PATH" ] || die "ASCEND_HOME_PATH 没设。先 source <CANN安装路径>/set_env.sh"

MSPROF=$(command -v msprof 2>/dev/null)
if [ -z "$MSPROF" ]; then
    for d in "$ASCEND_HOME_PATH/tools/profiler/bin" \
             "$ASCEND_HOME_PATH/$(uname -m)-linux/tools/profiler/bin" \
             "$ASCEND_HOME_PATH/bin"; do
        [ -x "$d/msprof" ] && { MSPROF="$d/msprof"; break; }
    done
fi
[ -z "$MSPROF" ] && MSPROF=$(find "$ASCEND_HOME_PATH" -maxdepth 6 -name msprof -type f 2>/dev/null | head -1)
[ -n "$MSPROF" ] || die "找不到 msprof。
    设备上装的可能只是 nnrt 运行时（不带 profiler）。
    没有 msprof 也能拿个粗数：REPEAT=50 python3 run_fused_conv2d.py $CASE $OPTYPE $DEV $OMDIR
    那是 host 侧墙钟，含下发开销，只能当上界看。"
echo "  msprof = $MSPROF"

# 给下发套超时是这套流程最重要的护身符：芯片被别人占着时，msprof 不报错、不超时、
# 也没有任何输出，进程就那么挂着。没有 timeout 命令时降级为直跑，但要说清楚。
TMO=""
if command -v timeout >/dev/null 2>&1; then
    TMO="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TMO="gtimeout"
else
    echo "  [!] 这台机器没有 timeout 命令，采集将不带超时保护。"
    echo "      如果卡住不动，多半是芯片被别人占着，Ctrl-C 后换一颗试试。"
fi
echo "  REPEAT = $REPEAT   OUT = $OUT"

for f in run_fused_conv2d.py parse_prof.py; do
    [ -f "$HERE/$f" ] || die "缺 $HERE/$f"
done
[ -f "$HERE/$CASE" ] || [ -f "$CASE" ] || die "找不到 case 文件 $CASE"

step "1) 采集"
rm -rf "$OUT" && mkdir -p "$OUT" || die "建不了输出目录 $OUT"

APP="python3 $HERE/run_fused_conv2d.py $CASE $OPTYPE $DEV"
[ -n "$OMDIR" ] && APP="$APP $OMDIR"
echo "  application: $APP"

export REPEAT
DEFAULT_ARGS="--ai-core=on --task-time=on --aic-metrics=PipeUtilization"
ARGS="${MSPROF_ARGS:-$DEFAULT_ARGS}"
echo "  开关: $ARGS"
echo

${TMO:+$TMO 900} "$MSPROF" --application="$APP" --output="$OUT" $ARGS
RC=$?
if [ "$RC" -eq 124 ]; then
    die "msprof 超时 900s —— 芯片多半被别人占着"
fi
if [ "$RC" -ne 0 ]; then
    echo
    echo "  [!] 带开关跑失败(rc=$RC)。各版本的开关名不一样，改用最小开关重试一次。"
    rm -rf "$OUT" && mkdir -p "$OUT"
    ${TMO:+$TMO 900} "$MSPROF" --application="$APP" --output="$OUT"
    RC=$?
    [ "$RC" -eq 0 ] || die "msprof 还是失败(rc=$RC)。上面是它自己的话。
    想换开关: MSPROF_ARGS='--ai-core=on' ./run_prof.sh ..."
fi

step "2) 导出（有的版本要这一步才生成 csv，失败不影响后面读 db）"
${TMO:+$TMO 300} "$MSPROF" --export=on --output="$OUT" >/dev/null 2>&1
echo "  export rc=$?（非 0 也没关系）"

step "3) 解析"
python3 "$HERE/parse_prof.py" "$OUT" "$OPTYPE"
PRC=$?

echo
if [ "$PRC" -ne 0 ]; then
    echo "[!] 解析没拿到数，原始目录还在: $OUT"
    echo "    把上面列出的文件清单贴出来，可以据此补解析路径。"
    exit 1
fi
echo "[√] 采集 + 解析完成。原始数据在 $OUT"
echo "    注意第一次下发通常显著偏大（含 kernel 加载），看 min/p50 而不是 mean。"
