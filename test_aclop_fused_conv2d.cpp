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
 * \file test_aclop_fused_conv2d.cpp
 * \brief FusedConv2d 单算子验证 —— **不需要 aclnn 头文件**的版本。
 *
 * 走 ACL 的单算子执行接口 aclopExecuteV2(opType, ...)：按**算子类型字符串**分发，
 * 不依赖任何生成的 aclnn 头/库。仓里绝大多数算子(Conv2DV2、QuantConv2D 等等)本来
 * 就没有自己的 aclnn 接口，验它们用的就是这条路。
 *
 *   ./test_aclop_fused_conv2d [op_type] [device_id]
 *   默认 FusedConv2d 0
 *
 * 如果你的包里注册的是大写 D 的 FusedConv2D，第一个参数传它。
 *
 * 判据和 golden 与 aclnn 版完全一致，两个版本的输出行可以直接对比。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op.h"

#define FUSED_CONV2D_GOLDEN_INT8_OUT 1
#include "fused_conv2d_int8_golden.h"

namespace G = FusedConv2dGolden;

#define CHECK(cond, msg, ...)                          \
    do {                                               \
        if (!(cond)) {                                 \
            printf("[FAIL] " msg "\n", ##__VA_ARGS__); \
            return 1;                                  \
        }                                              \
    } while (0)

namespace {

constexpr int8_t Y_SENTINEL = 127;

// VREQ8 deq 表项打包。位域说明见 test_aclnn_fused_conv2d.cpp / 步骤文档。
// bit[46] 不会从输出 dtype 推断 —— 忘了置 1 算子照样跑完，给出一串合理的无符号字节。
uint64_t PackScaleEntry(float scale, int offset = 0)
{
    uint32_t u;
    std::memcpy(&u, &scale, sizeof(u));
    uint64_t e = static_cast<uint64_t>(u & 0xFFFFE000u);
    e |= (static_cast<uint64_t>(static_cast<uint32_t>(offset) & 0x1FFu)) << 37;
    e |= (1ULL << 46);
    return e;
}

struct Operand {
    aclTensorDesc* desc = nullptr;
    aclDataBuffer* buf = nullptr;
    void* dev = nullptr;
};

int MakeOperand(const void* host, size_t bytes, const std::vector<int64_t>& dims, aclDataType dt, Operand* out)
{
    auto ret = aclrtMalloc(&out->dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK(ret == ACL_SUCCESS, "aclrtMalloc(%zu) = %d", bytes, ret);
    ret = aclrtMemcpy(out->dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK(ret == ACL_SUCCESS, "aclrtMemcpy H2D = %d", ret);
    out->desc = aclCreateTensorDesc(dt, dims.size(), dims.data(), ACL_FORMAT_ND);
    CHECK(out->desc != nullptr, "aclCreateTensorDesc returned null");
    out->buf = aclCreateDataBuffer(out->dev, bytes);
    CHECK(out->buf != nullptr, "aclCreateDataBuffer returned null");
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string opType = (argc > 1) ? argv[1] : "FusedConv2d";
    const int32_t deviceId = (argc > 2) ? std::atoi(argv[2]) : 0;

    printf("FusedConv2d @ 5102 单算子验证 —— 走 aclopExecuteV2(不需要 aclnn 头)\n");
    printf("  opType = \"%s\"   device = %d\n\n", opType.c_str(), deviceId);

    G::Inputs in = G::GenerateInputs();
    G::Golden gold = G::BuildGolden(in);
    const std::vector<int8_t>& want = gold.yI8;

    long long goldNonZero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        goldNonZero += (want[i] != 0) ? 1 : 0;
    }
    printf("golden: acc1_range=[%d,%d] mid_range=[%d,%d] acc2_range=[%d,%d] y_range=[%d,%d]\n", gold.acc1Min,
           gold.acc1Max, gold.midMin, gold.midMax, gold.acc2Min, gold.acc2Max, gold.yI8Min, gold.yI8Max);
    printf("golden: nonzero=%lld/%zu ties=%lld sat=%lld\n\n", goldNonZero, want.size(), gold.ties1 + gold.ties2,
           gold.sat1 + gold.sat2);
    CHECK(goldNonZero > 0, "golden 全是 0 —— 先别看设备");

    std::vector<uint64_t> s1(G::COUT1), s2(G::COUT2);
    for (int i = 0; i < G::COUT1; ++i) {
        s1[i] = PackScaleEntry(gold.scale1[i]);
    }
    for (int i = 0; i < G::COUT2; ++i) {
        s2[i] = PackScaleEntry(gold.scale2[i]);
    }

    auto ret = aclInit(nullptr);
    CHECK(ret == ACL_SUCCESS, "aclInit = %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK(ret == ACL_SUCCESS, "aclrtSetDevice(%d) = %d —— 芯片被占？", deviceId, ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK(ret == ACL_SUCCESS, "aclrtCreateStream = %d", ret);

    std::vector<int8_t> ySentinel(G::Y_ELEMS, Y_SENTINEL);

    // ABI 顺序钉死：x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    Operand op[8];
    const int64_t xRows = (int64_t)G::C1 * G::HI * G::WI;
    if (MakeOperand(in.xDev.data(), in.xDev.size(), {xRows, G::C0}, ACL_INT8, &op[0])) return 1;
    if (MakeOperand(in.w1Dev.data(), in.w1Dev.size(), {G::COUT1, G::K1}, ACL_INT8, &op[1])) return 1;
    if (MakeOperand(in.b1.data(), in.b1.size() * 4, {G::COUT1}, ACL_INT32, &op[2])) return 1;
    if (MakeOperand(s1.data(), s1.size() * 8, {G::COUT1}, ACL_UINT64, &op[3])) return 1;
    if (MakeOperand(in.w2Dev.data(), in.w2Dev.size(), {G::COUT2, G::K2}, ACL_INT8, &op[4])) return 1;
    if (MakeOperand(in.b2.data(), in.b2.size() * 4, {G::COUT2}, ACL_INT32, &op[5])) return 1;
    if (MakeOperand(s2.data(), s2.size() * 8, {G::COUT2}, ACL_UINT64, &op[6])) return 1;
    if (MakeOperand(ySentinel.data(), ySentinel.size(), {(int64_t)G::M_ROWS, G::COUT2}, ACL_INT8, &op[7])) return 1;

    aclTensorDesc* inDesc[7] = {op[0].desc, op[1].desc, op[2].desc, op[3].desc, op[4].desc, op[5].desc, op[6].desc};
    aclDataBuffer* inBuf[7] = {op[0].buf, op[1].buf, op[2].buf, op[3].buf, op[4].buf, op[5].buf, op[6].buf};
    aclTensorDesc* outDesc[1] = {op[7].desc};
    aclDataBuffer* outBuf[1] = {op[7].buf};

    // 本算子没有属性。文档说 attr 可以传 nullptr，但传一个空的 attr 更保险：
    // 有些版本按 attr 指针参与 kernel 选择的哈希。
    aclopAttr* attr = aclopCreateAttr();

    ret = aclopExecuteV2(opType.c_str(), 7, inDesc, inBuf, 1, outDesc, outBuf, attr, stream);
    CHECK(ret == ACL_SUCCESS,
          "aclopExecuteV2(\"%s\") = %d\n"
          "        算子没找到 -> 类型名拼错，或者包里根本没这个 op\n"
          "                      查: grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/\n"
          "        找到但选不出 kernel -> shape/dtype 和 binary.json 里登记的组合对不上",
          opType.c_str(), ret, opType.c_str());

    ret = aclrtSynchronizeStream(stream);
    CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志", ret);

    std::vector<int8_t> got(G::Y_ELEMS, 0);
    ret = aclrtMemcpy(got.data(), got.size(), op[7].dev, got.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK(ret == ACL_SUCCESS, "aclrtMemcpy D2H = %d", ret);

    long long unwritten = 0, nonZero = 0, mismatches = 0, firstBad = -1;
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
        long long offByOne = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const int d = (int)got[i] - (int)want[i];
            if (d == 1 || d == -1) {
                offByOne++;
            }
        }
        printf("first mismatch @ %lld: got %d, want %d (row %lld, cout %lld)；差 ±1 的 %lld / 共 %lld\n", firstBad,
               (int)got[firstBad], (int)want[firstBad], firstBad / G::COUT2, firstBad % G::COUT2, offByOne, mismatches);
    }

    aclopDestroyAttr(attr);
    for (int i = 0; i < 8; ++i) {
        aclDestroyDataBuffer(op[i].buf);
        aclDestroyTensorDesc(op[i].desc);
        aclrtFree(op[i].dev);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (unwritten != 0) {
        printf("[FAIL] %lld 个输出从来没被写过 —— kernel 没跑，或只跑了一部分核\n", unwritten);
        return 1;
    }
    if (nonZero != goldNonZero) {
        printf("[FAIL] 非零个数 %lld != golden 的 %lld —— 写的分布不对\n", nonZero, goldNonZero);
        return 1;
    }
    if (mismatches != 0) {
        printf("[FAIL] %lld 个元素数值不一致\n", mismatches);
        return 1;
    }
    printf("[PASS] 516096 个输出全部写过，且与 CPU golden 逐位相等\n");
    return 0;
}
