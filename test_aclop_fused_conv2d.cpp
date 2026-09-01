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

// y 走 VDEQF16 出 fp16。这个宏只选 GoldenOutput()/OutElem 的类型，跟
// gold.yI8 / gold.yF16 的直接取用无关，但留成 int8 会误导后来人。
#define FUSED_CONV2D_GOLDEN_INT8_OUT 0
#define FUSED_CONV2D_GOLDEN_FP16_OUT 1
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

// y 是 fp16，哨兵取 0x7F7F（一个 fp16 NaN 位模式，golden 里不可能出现）。
constexpr uint16_t Y_SENTINEL = 0x7F7Fu;

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

// VDEQF16 反量化表项打包 —— 这是 scale2（conv2 -> fp16 y）。编码和上面**不同**：
// 只用 [31:0] 的 fp32 位模式，没有 offset 也没有饱和位，高 32 位必须是 0。
// 和 tests/ut/op_kernel/test_fused_conv2d.cpp、gen_case.cpp 里的那份必须一模一样。
uint64_t PackDeqF16Entry(float scale)
{
    uint32_t u;
    std::memcpy(&u, &scale, sizeof(u));
    return static_cast<uint64_t>(u & 0xFFFFE000u);
}

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
    const std::vector<uint16_t>& want = gold.yF16;

    // 忽略符号位:fixpipe 的 relu 对负数输出 -0（0x8000），数值上就是 0。
    long long goldNonZero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        goldNonZero += ((want[i] & 0x7FFFu) != 0) ? 1 : 0;
    }
    printf("golden: acc1_range=[%d,%d] mid_range=[%d,%d] acc2_range=[%d,%d] y_range=[%.4g,%.4g]\n", gold.acc1Min,
           gold.acc1Max, gold.midMin, gold.midMax, gold.acc2Min, gold.acc2Max, (double)gold.yMin,
           (double)gold.yMax);
    // 哨兵必须不可能是真结果，否则 unwritten 那一条判据是假的。
    for (size_t i = 0; i < want.size(); ++i) {
        CHECK(want[i] != Y_SENTINEL, "golden 里出现了哨兵位模式 0x%04X @ %zu —— 换一个哨兵", Y_SENTINEL, i);
    }
    printf("golden: nonzero=%lld/%zu ties=%lld sat=%lld\n\n", goldNonZero, want.size(), gold.tiesQ + gold.ties1,
           gold.satQ + gold.sat1);
    CHECK(goldNonZero > 0, "golden 全是 0 —— 先别看设备");

    std::vector<uint64_t> s1(G::COUT1), s2(G::COUT2);
    for (int i = 0; i < G::COUT1; ++i) {
        s1[i] = PackScaleEntry(gold.scale1[i]);
    }
    for (int i = 0; i < G::COUT2; ++i) {
        s2[i] = PackDeqF16Entry(gold.scale2[i]);
    }

    auto ret = aclInit(nullptr);
    CHECK(ret == ACL_SUCCESS, "aclInit = %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK(ret == ACL_SUCCESS, "aclrtSetDevice(%d) = %d —— 芯片被占？", deviceId, ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK(ret == ACL_SUCCESS, "aclrtCreateStream = %d", ret);

    std::vector<uint16_t> ySentinel(G::Y_ELEMS, Y_SENTINEL);

    // ABI 顺序钉死：x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    Operand op[9];
    // ABI 顺序:x, scale_x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    // x / y 是 4 维 NCHW，两个 filter 是 FRACTAL_Z 的 4 个维度摊开。
    const int64_t fz1k = (G::CI / G::C0) * G::KH * G::KW;    // 9
    const int64_t fz1n = G::COUT1 / 16;                      // 4
    const int64_t fz2k = (G::COUT1 / G::C0) * G::KH * G::KW; // 18
    const int64_t fz2n = G::COUT2 / 16;                      // 6
    if (MakeOperand(in.xNchw.data(), in.xNchw.size() * 2, {1, G::CI, G::HI, G::WI}, ACL_FLOAT16, &op[0])) return 1;
    if (MakeOperand(&in.qScaleF32, sizeof(float), {1}, ACL_FLOAT, &op[1])) return 1;
    if (MakeOperand(in.w1Dev.data(), in.w1Dev.size(), {fz1k, fz1n, 16, G::C0}, ACL_INT8, &op[2])) return 1;
    if (MakeOperand(in.b1.data(), in.b1.size() * 4, {G::COUT1}, ACL_INT32, &op[3])) return 1;
    if (MakeOperand(s1.data(), s1.size() * 8, {G::COUT1}, ACL_UINT64, &op[4])) return 1;
    if (MakeOperand(in.w2Dev.data(), in.w2Dev.size(), {fz2k, fz2n, 16, G::C0}, ACL_INT8, &op[5])) return 1;
    if (MakeOperand(in.b2.data(), in.b2.size() * 4, {G::COUT2}, ACL_INT32, &op[6])) return 1;
    if (MakeOperand(s2.data(), s2.size() * 8, {G::COUT2}, ACL_UINT64, &op[7])) return 1;
    if (MakeOperand(ySentinel.data(), ySentinel.size() * sizeof(uint16_t), {1, G::COUT2, G::HO2, G::WO2},
                    ACL_FLOAT16, &op[8]))
        return 1;

    aclTensorDesc* inDesc[8] = {op[0].desc, op[1].desc, op[2].desc, op[3].desc,
                                op[4].desc, op[5].desc, op[6].desc, op[7].desc};
    aclDataBuffer* inBuf[8] = {op[0].buf, op[1].buf, op[2].buf, op[3].buf,
                               op[4].buf, op[5].buf, op[6].buf, op[7].buf};
    aclTensorDesc* outDesc[1] = {op[8].desc};
    aclDataBuffer* outBuf[1] = {op[8].buf};

    // 本算子没有属性。文档说 attr 可以传 nullptr，但传一个空的 attr 更保险：
    // 有些版本按 attr 指针参与 kernel 选择的哈希。
    aclopAttr* attr = aclopCreateAttr();

    ret = aclopExecuteV2(opType.c_str(), 8, inDesc, inBuf, 1, outDesc, outBuf, attr, stream);
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
    ret = aclrtMemcpy(got.data(), yBytes, op[7].dev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
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
    for (int i = 0; i < 9; ++i) {
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
