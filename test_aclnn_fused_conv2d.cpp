/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_fused_conv2d.cpp
 * \brief FusedConv2d 在 5102 真机上的单算子功能/精度验证。
 *
 *   ./test_aclnn_fused_conv2d [device_id]        默认 0
 *
 * 退出码 0 = 逐元素精确相等；非 0 = 失败，且会打印是哪一类失败。
 *
 * golden 直接复用 kernel UT 那一份（fused_conv2d_int8_golden.h），所以真机跑出来的
 * 判据行和 CPU 仿真那条可以逐字对比 —— 两边不一致，差异一定在"仿真 vs 真机"，
 * 而不是"两个 golden 实现不一样"。
 *
 * 算子名大小写：本文件默认按 OP_ADD(FusedConv2d) 生成的接口。如果你的包里是
 * 老的 FusedConv2D（大写 D），编译时加 -DFUSED_CONV2D_UPPER_D=1。
 * 怎么确认见同目录的 5102单算子验证步骤.md 第 1 节。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"

#define FUSED_CONV2D_GOLDEN_INT8_OUT 1
#include "fused_conv2d_int8_golden.h"

#if FUSED_CONV2D_UPPER_D
#include "aclnn_fused_conv2_d.h"
#define ACLNN_GET_WS aclnnFusedConv2DGetWorkspaceSize
#define ACLNN_RUN aclnnFusedConv2D
#else
#include "aclnn_fused_conv2d.h"
#define ACLNN_GET_WS aclnnFusedConv2dGetWorkspaceSize
#define ACLNN_RUN aclnnFusedConv2d
#endif

namespace G = FusedConv2dGolden;

#define CHECK(cond, msg, ...)                          \
    do {                                               \
        if (!(cond)) {                                 \
            printf("[FAIL] " msg "\n", ##__VA_ARGS__); \
            return 1;                                  \
        }                                              \
    } while (0)

namespace {

// 输出缓冲的哨兵。任何真实结果都可能等于 127（饱和上界），所以单独用哨兵判"写没写过"
// 是不够的 —— 见下面的 nonzero/golden_nonzero 双重判据。这里取 127 是为了和 kernel UT
// 保持一致，两边的 unwritten 数字可以直接对比。
constexpr int8_t Y_SENTINEL = 127;

// 打包一条 fixpipe VREQ8 deq 表项。scale1/scale2 是**算子输入**，kernel 只负责把它们
// 搬进 L1，打包是调用方的事。位域（实测）：
//   [31:13] scale 的 fp32 位模式，低 13 位尾数丢弃；符号位有效
//   [45:37] 有符号 9 位补码 offset，加在**取整之后**的整数上
//   [46]    1 = 有符号饱和到 [-128,127]；0 = 无符号饱和到 [0,255]
//
// bit[46] 不会从输出 dtype 推断。忘了置位算子照样跑完，给出一串看起来完全合理的
// 无符号字节 —— 这是这类测试里最容易犯又最难发现的错。
//
// 另外：ops-nn 其它地方出现的 0x3F80000000000000 写法（fp32 放在高半部）对这条通路
// 是错的，实测得 0。
uint64_t PackScaleEntry(float scale, int offset = 0)
{
    uint32_t u;
    std::memcpy(&u, &scale, sizeof(u));
    uint64_t e = static_cast<uint64_t>(u & 0xFFFFE000u);
    e |= (static_cast<uint64_t>(static_cast<uint32_t>(offset) & 0x1FFu)) << 37;
    e |= (1ULL << 46);
    return e;
}

int CreateTensor(const void* host, size_t bytes, const std::vector<int64_t>& shape, aclDataType dt, void** devAddr,
                 aclTensor** out)
{
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK(ret == ACL_SUCCESS, "aclrtMalloc(%zu) = %d", bytes, ret);
    ret = aclrtMemcpy(*devAddr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK(ret == ACL_SUCCESS, "aclrtMemcpy H2D = %d", ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *out = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                           shape.size(), *devAddr);
    CHECK(*out != nullptr, "aclCreateTensor returned null");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const int32_t deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;

    printf("FusedConv2d @ 5102 单算子验证\n");
    printf("  x[%zu,%d]int8 (NC1HWC0, ci=%d hi=%d wi=%d)\n", (size_t)G::C1 * G::HI * G::WI, G::C0, G::CI, G::HI, G::WI);
    printf("  -> conv1 3x3 s%d p%d -> mid[%d,%d,%d]int8\n", G::STRIDE1, G::PAD, G::COUT1, G::HO1, G::WO1);
    printf("  -> conv2 3x3 s%d p%d -> y[%zu,%d]int8\n\n", G::STRIDE2, G::PAD, G::M_ROWS, G::COUT2);

    // ---- host: 构造输入 + 算 golden -------------------------------------
    G::Inputs in = G::GenerateInputs();
    G::Golden gold = G::BuildGolden(in);
    const std::vector<int8_t>& want = gold.yI8;

    // golden 自检：全 0 或者全饱和的 golden 会让任何比对都毫无意义
    long long goldNonZero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        goldNonZero += (want[i] != 0) ? 1 : 0;
    }
    printf("golden: acc1_range=[%d,%d] mid_range=[%d,%d] acc2_range=[%d,%d] y_range=[%d,%d]\n", gold.acc1Min,
           gold.acc1Max, gold.midMin, gold.midMax, gold.acc2Min, gold.acc2Max, gold.yI8Min, gold.yI8Max);
    printf("golden: nonzero=%lld/%zu ties=%lld sat=%lld\n\n", goldNonZero, want.size(), gold.ties1 + gold.ties2,
           gold.sat1 + gold.sat2);
    CHECK(goldNonZero > 0, "golden 全是 0 —— 输入生成或 golden 本身坏了，先别看设备");
    CHECK(gold.sat1 + gold.sat2 < (long long)want.size() / 10, "golden 大面积饱和，scale 选得不对");

    // scale 表：float -> VREQ8 uint64
    std::vector<uint64_t> s1(G::COUT1), s2(G::COUT2);
    for (int i = 0; i < G::COUT1; ++i) {
        s1[i] = PackScaleEntry(gold.scale1[i]);
    }
    for (int i = 0; i < G::COUT2; ++i) {
        s2[i] = PackScaleEntry(gold.scale2[i]);
    }

    // ---- device ---------------------------------------------------------
    auto ret = aclInit(nullptr);
    CHECK(ret == ACL_SUCCESS, "aclInit = %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK(ret == ACL_SUCCESS, "aclrtSetDevice(%d) = %d —— 芯片被占？", deviceId, ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK(ret == ACL_SUCCESS, "aclrtCreateStream = %d", ret);

    std::vector<int8_t> ySentinel(G::Y_ELEMS, Y_SENTINEL);

    aclTensor *tx = nullptr, *tf1 = nullptr, *tb1 = nullptr, *ts1 = nullptr;
    aclTensor *tf2 = nullptr, *tb2 = nullptr, *ts2 = nullptr, *ty = nullptr;
    void *dx = nullptr, *df1 = nullptr, *db1 = nullptr, *ds1 = nullptr;
    void *df2 = nullptr, *db2 = nullptr, *ds2 = nullptr, *dy = nullptr;

    // ABI 顺序是钉死的：x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    if (CreateTensor(in.xDev.data(), in.xDev.size(), {(int64_t)G::C1 * G::HI * G::WI, G::C0}, ACL_INT8, &dx, &tx))
        return 1;
    if (CreateTensor(in.w1Dev.data(), in.w1Dev.size(), {G::COUT1, G::K1}, ACL_INT8, &df1, &tf1)) return 1;
    if (CreateTensor(in.b1.data(), in.b1.size() * sizeof(int32_t), {G::COUT1}, ACL_INT32, &db1, &tb1)) return 1;
    if (CreateTensor(s1.data(), s1.size() * sizeof(uint64_t), {G::COUT1}, ACL_UINT64, &ds1, &ts1)) return 1;
    if (CreateTensor(in.w2Dev.data(), in.w2Dev.size(), {G::COUT2, G::K2}, ACL_INT8, &df2, &tf2)) return 1;
    if (CreateTensor(in.b2.data(), in.b2.size() * sizeof(int32_t), {G::COUT2}, ACL_INT32, &db2, &tb2)) return 1;
    if (CreateTensor(s2.data(), s2.size() * sizeof(uint64_t), {G::COUT2}, ACL_UINT64, &ds2, &ts2)) return 1;
    if (CreateTensor(ySentinel.data(), ySentinel.size(), {(int64_t)G::M_ROWS, G::COUT2}, ACL_INT8, &dy, &ty)) return 1;

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = ACLNN_GET_WS(tx, tf1, tb1, ts1, tf2, tb2, ts2, ty, &wsSize, &executor);
    CHECK(ret == ACL_SUCCESS,
          "GetWorkspaceSize = %d\n"
          "        161001 = 算子没找到：包里没有 FusedConv2d，或者被别的 vendor 盖住了\n"
          "        其它    = tiling 拒绝了，多半是 shape/dtype 和算子约定对不上",
          ret);
    printf("workspace = %llu bytes\n", (unsigned long long)wsSize);

    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK(ret == ACL_SUCCESS, "aclrtMalloc(workspace %llu) = %d", (unsigned long long)wsSize, ret);
    }

    ret = ACLNN_RUN(ws, wsSize, executor, stream);
    CHECK(ret == ACL_SUCCESS, "aclnn 第二段 = %d", ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志", ret);

    std::vector<int8_t> got(G::Y_ELEMS, 0);
    ret = aclrtMemcpy(got.data(), got.size(), dy, got.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK(ret == ACL_SUCCESS, "aclrtMemcpy D2H = %d", ret);

    // ---- 判据 -----------------------------------------------------------
    // 三条一起看，少一条都会放过一种"看起来通过"的失败：
    //   unwritten  : kernel 什么都没写（缓冲里还是哨兵）
    //   nonZero    : kernel 只写了一部分（分布对不上 golden）
    //   mismatches : 数值错
    long long unwritten = 0, nonZero = 0, mismatches = 0;
    long long firstBad = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == Y_SENTINEL && want[i] != Y_SENTINEL) {
            unwritten++;
        }
        if (got[i] != 0) {
            nonZero++;
        }
        if (got[i] != want[i]) {
            if (firstBad < 0) {
                firstBad = (long long)i;
            }
            mismatches++;
        }
    }

    printf("\n[fused-conv2d-5102-device] out_elems=%zu nonzero=%lld golden_nonzero=%lld "
           "unwritten=%lld mismatches=%lld\n",
           got.size(), nonZero, goldNonZero, unwritten, mismatches);

    if (mismatches != 0) {
        printf("first mismatch @ %lld: got %d, want %d  (row %lld, cout %lld)\n", firstBad, (int)got[firstBad],
               (int)want[firstBad], firstBad / G::COUT2, firstBad % G::COUT2);
        // 差 ±1 且集中在 tie 上 -> 舍入模型不一致；差得离谱 -> 布局或 scale 错了
        long long offByOne = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const int d = (int)got[i] - (int)want[i];
            if (d == 1 || d == -1) {
                offByOne++;
            }
        }
        printf("其中差 ±1 的有 %lld 个 / 共 %lld 个不一致\n", offByOne, mismatches);
        printf("  全部是 ±1  -> 舍入模型对不上（乘积应先舍到 fp32，再 round-half-to-even，offset 加在取整之后）\n");
        printf("  差得离谱   -> 多半是布局（NC1HWC0 / K 序）或 scale 表打包错了\n");
    }

    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclDestroyTensor(tx); aclDestroyTensor(tf1); aclDestroyTensor(tb1); aclDestroyTensor(ts1);
    aclDestroyTensor(tf2); aclDestroyTensor(tb2); aclDestroyTensor(ts2); aclDestroyTensor(ty);
    aclrtFree(dx); aclrtFree(df1); aclrtFree(db1); aclrtFree(ds1);
    aclrtFree(df2); aclrtFree(db2); aclrtFree(ds2); aclrtFree(dy);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (unwritten != 0) {
        printf("[FAIL] %lld 个输出从来没被写过 —— kernel 没跑，或者只跑了一部分核\n", unwritten);
        return 1;
    }
    if (nonZero != goldNonZero) {
        printf("[FAIL] 非零元素个数 %lld != golden 的 %lld —— 写的分布不对\n", nonZero, goldNonZero);
        return 1;
    }
    if (mismatches != 0) {
        printf("[FAIL] %lld 个元素数值不一致\n", mismatches);
        return 1;
    }
    printf("[PASS] 516096 个输出全部写过，且与 CPU golden 逐位相等\n");
    return 0;
}
