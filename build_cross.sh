#!/bin/bash
# 备选路线：在 x86 上交叉编译出 aarch64 的验证程序，再拷到 5102 机器上跑。
#
# 优先用 build_case.sh + run_fused_conv2d.py（目标机不需要编译器，也没有 glibc
# 版本风险）。只有在目标机连 python3 都没有时才用这条。
#
#   ./build_cross.sh                       # 用默认的 aarch64-linux-gnu-g++
#   CROSS_CXX=<你的交叉g++> ./build_cross.sh
#
# 前置：
#   1) 交叉工具链   apt install g++-aarch64-linux-gnu   /   ARM 官方 toolchain 解压即用
#   2) 目标机的 libascendcl.so（aarch64 的那份），拷到本机某个目录，用 AARCH64_LIB 指过去：
#        mkdir -p ~/cann-aarch64/lib64
#        scp <5102机>:$ASCEND_HOME_PATH/lib64/libascendcl.so ~/cann-aarch64/lib64/
#        AARCH64_LIB=~/cann-aarch64/lib64 ./build_cross.sh
#      acl 的头文件是架构无关的，直接用本机 x86 CANN 里的那份就行。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "$HERE" ] || HERE="$PWD"
CROSS_CXX="${CROSS_CXX:-aarch64-linux-gnu-g++}"
OUT="$HERE/test_aclop_fused_conv2d.aarch64"

die()  { echo; echo "[X] $*"; exit 1; }
step() { echo; echo "==== $* ===="; }

step "0) 前置检查"
command -v "$CROSS_CXX" >/dev/null 2>&1 \
    || die "找不到交叉编译器 $CROSS_CXX
    Ubuntu/Debian:  sudo apt install g++-aarch64-linux-gnu
    没有 root 的话到 developer.arm.com 下 aarch64 的 GNU toolchain，解压后
    CROSS_CXX=<解压目录>/bin/aarch64-none-linux-gnu-g++ ./build_cross.sh"
echo "  CROSS_CXX = $CROSS_CXX  ($($CROSS_CXX --version 2>/dev/null | head -1))"

[ -n "$ASCEND_HOME_PATH" ] || die "ASCEND_HOME_PATH 没设（要用它下面的 acl 头文件，架构无关）"
INC="$ASCEND_HOME_PATH"
[ -d "$INC/include/acl" ] || INC="$ASCEND_HOME_PATH/$(uname -m)-linux"
[ -d "$INC/include/acl" ] || die "在 $ASCEND_HOME_PATH 下找不到 include/acl/"
echo "  acl 头   = $INC/include"

[ -n "$AARCH64_LIB" ] || die "AARCH64_LIB 没设。需要目标机上那份 aarch64 的 libascendcl.so：
    mkdir -p ~/cann-aarch64/lib64
    scp <5102机>:\$ASCEND_HOME_PATH/lib64/libascendcl.so ~/cann-aarch64/lib64/
    AARCH64_LIB=~/cann-aarch64/lib64 ./build_cross.sh"
[ -f "$AARCH64_LIB/libascendcl.so" ] || die "$AARCH64_LIB 下没有 libascendcl.so"

LIBMACH=$(readelf -h "$AARCH64_LIB/libascendcl.so" 2>/dev/null | sed -n 's/^ *Machine: *//p')
echo "  libascendcl.so 的 Machine = ${LIBMACH:-<未知>}"
echo "$LIBMACH" | grep -qi "AArch64" \
    || die "这份 libascendcl.so 不是 aarch64 的（是 $LIBMACH）—— 你拷的是本机 x86 那份"

step "1) 检查源文件"
for f in fused_conv2d_int8_golden.h test_aclop_fused_conv2d.cpp; do
    [ -f "$HERE/$f" ] || die "缺 $f"
    printf '  %-32s %s 行\n' "$f" "$(wc -l < "$HERE/$f" | tr -d " \t")"
done

step "2) 交叉编译"
# --allow-shlib-undefined: libascendcl 还会 DT_NEEDED 一串别的库（最终连到驱动的
# libascend_hal.so）。那些在本机上没有，但**目标机上全都有**，所以让链接器别管
# 共享库里的未定义符号，只把 libascendcl 自己导出的符号解析掉就行。
# 不设 -rpath：目标机上 source set_env.sh 会把 LD_LIBRARY_PATH 设好。
rm -f "$OUT"
"$CROSS_CXX" -std=c++17 -O2 -ffp-contract=off \
    "$HERE/test_aclop_fused_conv2d.cpp" -o "$OUT" \
    -I"$HERE" -I"$INC/include" \
    -L"$AARCH64_LIB" -lascendcl \
    -Wl,--allow-shlib-undefined
[ $? -eq 0 ] || die "交叉编译失败（上面是编译器的话）"
echo "  built $OUT"
readelf -h "$OUT" | sed -n 's/^ *Machine: */  ELF Machine   = /p'

step "3) glibc 版本 —— 这条最容易翻车"
NEED=$(readelf -V "$OUT" 2>/dev/null | grep -o 'GLIBC_[0-9][0-9.]*' | sort -uV | tail -1)
echo "  这个二进制要求目标机的 glibc >= ${NEED:-<读不出来>}"
cat <<TIP

  到 5102 机器上核一下：

    ldd --version | head -1

  目标机的 glibc 比 ${NEED:-上面那个} 低的话，一跑就是
  "version \`GLIBC_2.xx' not found"。那就说明这条交叉路线走不通，
  回去用 build_case.sh + run_fused_conv2d.py（那条完全不受 glibc 影响）。

  没问题的话：

    scp $(basename "$OUT") <5102机器>:~/a5102_st/
    ssh <5102机器>
    cd ~/a5102_st && source <CANN安装路径>/set_env.sh
    ./$(basename "$OUT") FusedConv2d 0
TIP
