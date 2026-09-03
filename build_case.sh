#!/bin/bash
# 在**能编译的机器**上生成 case 文件（不需要 CANN，不需要设备，架构无所谓）。
#
#   ./build_case.sh
#
# 产物: fused_conv2d_case.bin —— 和 run_fused_conv2d.py 一起拷到有 5102 的机器上。
#
# 不用 set -e，每步显式检查。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"
CXX_HOST="${CXX_HOST:-g++}"

die()  { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

step "0) 环境"
command -v "$CXX_HOST" >/dev/null 2>&1 || die "找不到 $CXX_HOST"
echo "  $CXX_HOST = $($CXX_HOST --version 2>/dev/null | head -1)"
echo "  本机 uname -m = $(uname -m)   (无所谓，golden 与架构无关)"
echo "  脚本目录 = $HERE"

step "1) 检查所需文件"
MISSING=0
for f in fused_conv2d_golden.h golden_selfcheck.cpp gen_case.cpp run_fused_conv2d.py; do
    if [ -f "$HERE/$f" ]; then
        printf '  %-32s %s 行\n' "$f" "$(wc -l < "$HERE/$f" 2>/dev/null)"
    else
        echo "  $f  ** 缺失 **"
        MISSING=1
    fi
done
[ "$MISSING" = 0 ] || die "上面标 ** 缺失 ** 的文件要和脚本放在同一个目录"

grep -q "^} // namespace fc2d_golden" "$HERE/fused_conv2d_golden.h" \
    || die "fused_conv2d_golden.h 没有闭合的命名空间 —— 文件被截断了"
echo "  golden 头完整性 OK"

# -ffp-contract=off: 有的架构上 gcc 默认允许把 a*b+c 融成 fmadd，结果和分开算不一样。
# golden 里目前没有可融的表达式，加上只是把这个疑问钉死。
CXXFLAGS="-std=c++17 -O2 -ffp-contract=off"

step "2) golden 自检"
rm -f "$HERE/golden_selfcheck"
$CXX_HOST $CXXFLAGS "$HERE/golden_selfcheck.cpp" -o "$HERE/golden_selfcheck" -I"$HERE"
[ $? -eq 0 ] || die "golden 自检编译失败（上面是编译器的话）"
"$HERE/golden_selfcheck"
[ $? -eq 0 ] || die "golden 自检没过。golden 不可信的话，后面比对出来的结论都没意义"
echo "  golden 自检通过"

step "3) 生成 case 文件"
rm -f "$HERE/gen_case" "$HERE/fused_conv2d_case.bin"
$CXX_HOST $CXXFLAGS "$HERE/gen_case.cpp" -o "$HERE/gen_case" -I"$HERE"
[ $? -eq 0 ] || die "gen_case 编译失败（上面是编译器的话）"
( cd "$HERE" && ./gen_case fused_conv2d_case.bin )
[ $? -eq 0 ] || die "gen_case 运行失败"
[ -f "$HERE/fused_conv2d_case.bin" ] || die "gen_case 说成功了但没产出 .bin？"

SZ=$(wc -c < "$HERE/fused_conv2d_case.bin" | tr -d " \t")   # macOS 的 wc 会补前导空格
[ "$SZ" = 5309240 ] || echo "  [!] .bin 是 $SZ 字节，预期 5309240 —— 形状变了？"

step "3.5) 顺带生成两个探针 case"
# 探针 = 把某一层权重换成中心抽头恒等，把两层拆开单独看。权重是运行时输入不是
# 属性，所以这些 .bin 和正常 case 用**同一个 .om**，不用重编。
for pm in passthrough mid chanid colid; do
    "$HERE/gen_case" "$HERE/fused_conv2d_probe_$pm.bin" "--probe=$pm" >/dev/null 2>&1
    if [ -f "$HERE/fused_conv2d_probe_$pm.bin" ]; then
        printf '  %-38s %s 字节\n' "fused_conv2d_probe_$pm.bin" \
               "$(wc -c < "$HERE/fused_conv2d_probe_$pm.bin" | tr -d ' \t')"
    else
        echo "  [!] 探针 $pm 没生成出来"
    fi
done

step "4) 拷过去"
echo "  文件: $HERE/fused_conv2d_case.bin  ($SZ 字节)"
command -v md5sum >/dev/null 2>&1 && md5sum "$HERE/fused_conv2d_case.bin" | sed 's/^/  md5: /'
cat <<TIP

  scp fused_conv2d_case.bin run_fused_conv2d.py <5102机器>:~/a5102_st/

  然后在 5102 机器上（不需要编译器）：

  cd ~/a5102_st
  source <CANN安装路径>/set_env.sh
  python3 run_fused_conv2d.py fused_conv2d_case.bin FusedConv2d 0

  定点定标（fixed_shift1 / fixed_shift2）由 golden 按数据算出，打包在 .bin 的
  header 里，脚本自己取来当算子属性下发 —— 不用手填，也不会主机设备两边不一致。

  传完先在两边各跑一次 md5sum 对一下，.bin 传坏了脚本会报"文件被截断"，
  但传成另一个完整文件它是看不出来的。
TIP
