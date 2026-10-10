#!/bin/bash
# 在**没有 NPU 的机器上**把 run_fused_conv2d.py 的整条板侧下发路径跑一遍。
#
# 为什么需要这个：
#   * 真机上 aclInit 直接失败（chipType=0），到不了下发那一段；
#   * --dry-run 是刻意绕开 ACL 分支的（它的目的是在没有 CANN 的机器上验代码路径）。
# 于是「模型加载 -> 按名字取下标 -> dataset -> execute -> 读回 -> 比对 -> 诊断」
# 这一整段在本地**一行都执行不到**。实际代价：y_bytes 的 UnboundLocalError 是在
# 板上第一次跑才暴露的，而它只是个先用后赋。
#
# 桩实现了 libascendcl.so 里这条路径用到的全部符号，用 LD_LIBRARY_PATH 顶掉真的那个。
# aclmdlExecute 把 case.bin 里的 y_expect 拷进输出缓冲，所以比对、诊断、抽样那几段
# 也都会真的跑一遍（结论当然不验数值正确性 —— 那要靠板子）。
#
# 两种定位方式都要跑：
#   名字   aclmdlGetInputIndexByName 能用（期望的情形）
#   退路   它全部失败，运行器退回 IR 顺序 —— **这条在真板上很可能被用到**，
#          因为 ACL 把模型输入名报成什么样是我们控制不了的
#
# 用法（在 harness 根目录）：
#     bash tests/stub_ascendcl/run_stub_test.sh                 # 跑 out/ 下所有有 om 的 case
#     bash tests/stub_ascendcl/run_stub_test.sh base_int8 ...   # 只跑指定的
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
W="$HERE/.work"
mkdir -p "$W"

command -v gcc >/dev/null || { echo "[X] 需要 gcc 来编桩"; exit 1; }
gcc -shared -fPIC -O1 -o "$W/libascendcl.so" "$HERE/stub.c" || { echo "[X] 桩编译失败"; exit 1; }

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$W:${LD_LIBRARY_PATH:-}"

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
  mapfile -t cases < <(ls -d out/*/*/om/*.om 2>/dev/null | awk -F/ '{print $(NF-2)}' | sort -u)
fi
[ ${#cases[@]} -eq 0 ] && { echo "[X] out/ 下找不到任何 .om —— 先跑 --steps case,om"; exit 1; }

fail=0
for c in "${cases[@]}"; do
  bin=$(ls out/*/"$c"/case.bin 2>/dev/null | head -1)
  om=$(ls out/*/"$c"/om/*.om 2>/dev/null | head -1)
  if [ -z "$bin" ] || [ -z "$om" ]; then
    printf "  %-16s ** 缺 case.bin 或 .om，跳过 **\n" "$c"; continue
  fi
  cw="$W/$c"; mkdir -p "$cw"
  python3 "$HERE/make_conf.py" "$bin" "$cw" > "$cw/plan.txt" 2>&1 || {
    printf "  %-16s ** 配置生成失败，看 %s **\n" "$c" "$cw/plan.txt"; fail=1; continue; }
  for mode in name fallback; do
    extra=""; label="名字"
    [ "$mode" = fallback ] && { extra="FC2D_STUB_NONAME=1"; label="退路"; }
    out=$(env FC2D_STUB_CONF="$cw/stub.conf" FC2D_STUB_GOLDEN="$cw/golden.bin" $extra \
          python3 run_fused_conv2d.py "$bin" FusedConv2d 0 "$om" 2>&1)
    echo "$out" > "$cw/run_$mode.log"
    if echo "$out" | grep -q "^\[PASS\]"; then
      printf "  %-16s %-5s PASS\n" "$c" "$label"
    else
      printf "  %-16s %-5s ** 不通过 ** %s\n" "$c" "$label" \
        "$(echo "$out" | grep -oE '\[X\].{0,70}|\[FAIL\].{0,70}|[A-Za-z]*Error.{0,50}' | head -1)"
      echo "      完整输出: $cw/run_$mode.log"
      fail=1
    fi
  done
done
exit $fail
