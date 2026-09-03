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

#include "fused_conv2d_golden.h"

namespace G = fc2d_golden;

#define CHECK(cond, msg, ...)                          \
    do {                                               \
        if (!(cond)) {                                 \
            printf("[FAIL] " msg "\n", ##__VA_ARGS__); \
            return 1;                                  \
        }                                              \
    } while (0)

namespace {

// y 是 fp16，哨兵取 0x7F7F（一个 fp16 NaN 位模式，golden 里不可能出现）。
constexpr uint16_t Y_SENTINEL = 0x7F7Fu;

// 定点版没有量化表。上一版这里有 PackScaleEntry / PackDeqF16Entry 两个函数，
// 把 float 打包成 VREQ8 / VDEQF16 的表项 —— 现在 fixpipe 走标量 DEQF16，
// deqScalar 恒为 FLOAT_ONE_FIXED_POINT，缩放全由 fixShiftVal 完成，表没了。

// fp16 位模式 -> float，只为打印和 ULP 分析用；不参与判据（判据是位模式逐位相等）。
float F16ToFloat(uint16_t h)
{
    const uint32_t s = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    if (e == 0) {
        // 次正规（含 ±0）：值就是 m * 2^-24。m <= 1023，2^-24 是 2 的幂，所以
        // 这个乘法在 float 上是精确的 —— 比手写规格化循环短，也少一个能写错的地方。
        // （手写那版第一稿就把指数算差了 1，65536 个位模式里错了 2046 个。）
        const float v = (float)m * 5.9604644775390625e-08f; // 2^-24
        return s ? -v : v;
    }
    uint32_t u;
    if (e == 31) {
        u = (s << 31) | 0x7F800000u | (m << 13); // inf / NaN
    } else {
        u = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
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
    // 两个 golden：yFixed 是定点模型（假设成立时应逐位相等），yExact 是纯 fp32
    // 参考（判「有没有在算这个卷积」，不受定点假设影响）。见 golden 头部那段。
    const std::vector<uint16_t>& want = gold.yFixed;
    const std::vector<uint16_t>& wantExact = gold.yExact;

    // 忽略符号位:fixpipe 的 relu 对负数输出 -0（0x8000），数值上就是 0。
    long long goldNonZero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        goldNonZero += ((want[i] & 0x7FFFu) != 0) ? 1 : 0;
    }
    printf("golden: conv1 峰值 %.4f -> shift %d   conv2 峰值 %.4f -> shift %d   int32 饱和 %ld\n", gold.peak1,
           gold.shift1, gold.peak2, gold.shift2, gold.sat);
    // 哨兵必须不可能是真结果，否则 unwritten 那一条判据是假的。
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(want[i] != Y_SENTINEL, "golden 里出现了哨兵位模式 0x%04X @ %zu —— 换一个哨兵", Y_SENTINEL, i);
    }
    printf("golden: nonzero=%lld/%zu\n\n", goldNonZero, want.size());
    CHECK(goldNonZero > 0, "golden 全是 0 —— 先别看设备");

    auto ret = aclInit(nullptr);
    CHECK(ret == ACL_SUCCESS, "aclInit = %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK(ret == ACL_SUCCESS, "aclrtSetDevice(%d) = %d —— 芯片被占？", deviceId, ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK(ret == ACL_SUCCESS, "aclrtCreateStream = %d", ret);

    std::vector<uint16_t> ySentinel(G::Y_ELEMS, Y_SENTINEL);

    // ABI 顺序钉死（fused_conv2d_def.cpp）：x, filter1, bias1, filter2, bias2 -> y
    // 外加两个属性 fixed_shift1 / fixed_shift2。上一版的三张量化表已删。
    // x / y 是 4 维 NCHW，两个 filter 是 FRACTAL_Z（fp16，C0 = 16）的 4 个维度摊开。
    Operand op[6];
    const int64_t fz1k = (G::CI / G::C0) * G::KH * G::KW;    // 18
    const int64_t fz1n = (G::COUT1 + 15) / 16;               // 4
    const int64_t fz2k = (G::COUT1 / G::C0) * G::KH * G::KW; // 36
    const int64_t fz2n = (G::COUT2 + 15) / 16;               // 6
    if (MakeOperand(in.xNchw.data(), in.xNchw.size() * 2, {1, G::CI, G::HI, G::WI}, ACL_FLOAT16, &op[0])) return 1;
    if (MakeOperand(in.w1Dev.data(), in.w1Dev.size() * 2, {fz1k, fz1n, 16, G::C0}, ACL_FLOAT16, &op[1])) return 1;
    if (MakeOperand(in.b1.data(), in.b1.size() * 2, {G::COUT1}, ACL_FLOAT16, &op[2])) return 1;
    if (MakeOperand(in.w2Dev.data(), in.w2Dev.size() * 2, {fz2k, fz2n, 16, G::C0}, ACL_FLOAT16, &op[3])) return 1;
    if (MakeOperand(in.b2.data(), in.b2.size() * 2, {G::COUT2}, ACL_FLOAT16, &op[4])) return 1;
    if (MakeOperand(ySentinel.data(), ySentinel.size() * sizeof(uint16_t), {1, G::COUT2, G::HO2, G::WO2},
                    ACL_FLOAT16, &op[5]))
        return 1;

    aclTensorDesc* inDesc[5] = {op[0].desc, op[1].desc, op[2].desc, op[3].desc, op[4].desc};
    aclDataBuffer* inBuf[5] = {op[0].buf, op[1].buf, op[2].buf, op[3].buf, op[4].buf};
    aclTensorDesc* outDesc[1] = {op[5].desc};
    aclDataBuffer* outBuf[1] = {op[5].buf};

    // 两个定点定标是**必需属性**，顺序和 def.cpp 里 Attr() 的调用顺序一致
    // （fixed_shift1 在前），tiling 侧按 GetInt(0)/GetInt(1) 取。值由 golden 按
    // 数据算出来，主机和设备用的是同一个数。
    aclopAttr* attr = aclopCreateAttr();
    CHECK(attr != nullptr, "aclopCreateAttr 返回 null");
    ret = aclopSetAttrInt(attr, "fixed_shift1", gold.shift1);
    CHECK(ret == ACL_SUCCESS, "aclopSetAttrInt(fixed_shift1=%d) = %d", gold.shift1, ret);
    ret = aclopSetAttrInt(attr, "fixed_shift2", gold.shift2);
    CHECK(ret == ACL_SUCCESS, "aclopSetAttrInt(fixed_shift2=%d) = %d", gold.shift2, ret);

    ret = aclopExecuteV2(opType.c_str(), 5, inDesc, inBuf, 1, outDesc, outBuf, attr, stream);
    CHECK(ret == ACL_SUCCESS,
          "aclopExecuteV2(\"%s\") = %d\n"
          "        算子没找到 -> 类型名拼错，或者包里根本没这个 op\n"
          "                      查: grep -ri '\"%s\"' $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/\n"
          "        找到但选不出 kernel -> shape/dtype 和 binary.json 里登记的组合对不上",
          opType.c_str(), ret, opType.c_str());

    ret = aclrtSynchronizeStream(stream);
    CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream = %d —— kernel 可能 abort 了，查 device 日志", ret);

    std::vector<uint16_t> got(G::Y_ELEMS, 0);
    const size_t yBytes = got.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(got.data(), yBytes, op[5].dev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK(ret == ACL_SUCCESS, "aclrtMemcpy D2H = %d", ret);

    long long unwritten = 0, nonZero = 0, mismatches = 0, firstBad = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == Y_SENTINEL) {
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

    // 和纯 fp32 参考比一次相对误差。定点模型（上面的 mismatches）不过、但这里过，
    // 就说明卷积算对了、只是 fixShiftVal 的语义我猜错了 —— 这两条要分开看。
    double maxRel = 0.0;
    long long relBad = 0, zeroRef = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = F16ToFloat(got[i]);
        const double b = F16ToFloat(wantExact[i]);
        if (b == 0.0) {
            ++zeroRef;
            if (a != 0.0) ++relBad;
            continue;
        }
        const double r = std::fabs(a - b) / std::fabs(b);
        if (r > maxRel) maxRel = r;
        if (r > 1e-2) ++relBad;
    }
    printf("[fused-conv2d-5102-device] vs 纯 fp32 参考: 最大相对误差 %.4e，超过 1e-2 的 %lld 个"
           "（参考为 0 的 %lld 个不参与）\n", maxRel, relBad, zeroRef);

    if (mismatches != 0) {
        long long offByOne = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const int d = (int)got[i] - (int)want[i];
            if (d == 1 || d == -1) {
                offByOne++;
            }
        }
        printf("first mismatch @ %lld: got %.6g (0x%04X), want %.6g (0x%04X) (row %lld, cout %lld)；"
               "位模式差 ±1 的 %lld / 共 %lld\n",
               firstBad, (double)F16ToFloat(got[firstBad]), got[firstBad], (double)F16ToFloat(want[firstBad]),
               want[firstBad], firstBad / G::COUT2, firstBad % G::COUT2, offByOne, mismatches);
    }

    aclopDestroyAttr(attr);
    for (int i = 0; i < 6; ++i) {
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
    printf("[PASS] %zu 个输出全部写过，且与 CPU golden 逐位相等\n", got.size());
    return 0;
}
