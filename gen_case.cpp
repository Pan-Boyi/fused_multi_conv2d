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
 * \file gen_case.cpp
 * \brief 把 FusedConv2d 的 7 个输入 + golden 输出打成一个 .bin，供无编译器的机器使用。
 *
 * 这个程序**不碰 CANN、不碰设备**，纯 host C++。所以它可以在任何一台能编译的机器上
 * 跑（x86 开发机就行），产物 .bin 拷到有 5102 的机器上，由 run_fused_conv2d.py
 * 通过 ctypes 直接调 libascendcl 执行、比对。整条路上目标机器不需要编译器。
 *
 *   g++ -std=c++17 -O2 -ffp-contract=off gen_case.cpp -o gen_case && ./gen_case
 *
 * golden 与 kernel UT 是同一份原件，且不含任何可被 FMA 融合的浮点表达式，
 * 所以在 x86 上算出来的 golden 和在 aarch64 上算出来的逐位相同。
 *
 * .bin 布局（全部小端；x86 和 aarch64 都是小端）：
 *
 *   头 56 字节
 *     0   8   magic "FC2DCASE"
 *     8   4   version (u32) = 1
 *     12  4   ntensors (u32) = 8
 *     16  8   gold_nonzero (u64)
 *     24  8   ties (u64)
 *     32  8   sat (u64)
 *     40  8   y_elems (u64)
 *     48  8   reserved = 0
 *   之后 ntensors 条记录，每条 64 字节头 + 数据
 *     0   16  name，NUL 补齐
 *     16  4   dtype (u32)，ACL 的 aclDataType 枚举值
 *     20  4   ndim (u32)
 *     24  32  dims[4] (i64)
 *     56  8   nbytes (u64)
 *     64  ..  数据
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// y 走 VDEQF16 出 fp16。这个宏只选 GoldenOutput()/OutElem 的类型，跟
// gold.yI8 / gold.yF16 的直接取用无关，但留成 int8 会误导后来人。
#define FUSED_CONV2D_GOLDEN_INT8_OUT 0
#define FUSED_CONV2D_GOLDEN_FP16_OUT 1
#include "fused_conv2d_int8_golden.h"

namespace G = FusedConv2dGolden;

namespace {

// ACL 的 aclDataType 枚举。这里写死数值，免得为了三个常量去 include acl.h ——
// 这个程序刻意不依赖 CANN。
constexpr uint32_t ACL_DT_FLOAT = 0;
constexpr uint32_t ACL_DT_FLOAT16 = 1;
constexpr uint32_t ACL_DT_INT8 = 2;
constexpr uint32_t ACL_DT_INT32 = 3;
constexpr uint32_t ACL_DT_UINT64 = 10;

// VREQ8 重量化表项打包 —— 这是 scale1（conv1 -> int8 mid）。
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
// 和 tests/ut/op_kernel/test_fused_conv2d.cpp 里的那份必须一模一样。
uint64_t PackDeqF16Entry(float scale)
{
    uint32_t u;
    std::memcpy(&u, &scale, sizeof(u));
    return static_cast<uint64_t>(u & 0xFFFFE000u);
}

bool WriteTensor(std::FILE* f, const char* name, uint32_t dtype, const std::vector<int64_t>& dims, const void* data,
                 uint64_t nbytes)
{
    char nameBuf[16];
    std::memset(nameBuf, 0, sizeof(nameBuf));
    std::strncpy(nameBuf, name, sizeof(nameBuf) - 1);

    int64_t dimBuf[4] = {0, 0, 0, 0};
    const uint32_t ndim = static_cast<uint32_t>(dims.size());
    if (ndim > 4) {
        return false;
    }
    for (uint32_t i = 0; i < ndim; ++i) {
        dimBuf[i] = dims[i];
    }

    if (std::fwrite(nameBuf, 1, sizeof(nameBuf), f) != sizeof(nameBuf)) return false;
    if (std::fwrite(&dtype, 1, 4, f) != 4) return false;
    if (std::fwrite(&ndim, 1, 4, f) != 4) return false;
    if (std::fwrite(dimBuf, 1, sizeof(dimBuf), f) != sizeof(dimBuf)) return false;
    if (std::fwrite(&nbytes, 1, 8, f) != 8) return false;
    if (nbytes != 0 && std::fwrite(data, 1, nbytes, f) != nbytes) return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string out = (argc > 1) ? argv[1] : "fused_conv2d_case.bin";

    // 小端 + 类型宽度自检。生成机和运行机必须在这几件事上一致，否则 .bin 没法跨机器用。
    const uint16_t probe = 0x0102;
    if (*reinterpret_cast<const uint8_t*>(&probe) != 0x02) {
        std::printf("[FAIL] 这台机器是大端，.bin 格式是小端\n");
        return 1;
    }
    static_assert(sizeof(float) == 4, "float 必须是 4 字节");

    G::Inputs in = G::GenerateInputs();
    G::Golden gold = G::BuildGolden(in);
    // CHANGED：y 走 VDEQF16，输出是 fp16 的位模式。
    const std::vector<uint16_t>& want = gold.yF16;

    // 忽略符号位:fixpipe 的 relu 对负数输出 -0（0x8000），数值上就是 0。
    long long goldNonZero = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        goldNonZero += ((want[i] & 0x7FFFu) != 0) ? 1 : 0;
    }

    std::printf("golden: q_range=[%d,%d] acc1_range=[%d,%d] mid_range=[%d,%d] acc2_range=[%d,%d] "
                "y_range=[%.4g,%.4g]\n",
                gold.qMin, gold.qMax, gold.acc1Min, gold.acc1Max, gold.midMin, gold.midMax, gold.acc2Min,
                gold.acc2Max, (double)gold.yMin, (double)gold.yMax);
    std::printf("golden: nonzero=%lld/%zu ties=%lld sat=%lld  scale_x=%.9g\n", goldNonZero, want.size(),
                gold.tiesQ + gold.ties1, gold.satQ + gold.sat1, (double)in.qScaleF32);
    if (goldNonZero <= 0) {
        std::printf("[FAIL] golden 全是 0 —— 先别管设备\n");
        return 1;
    }
    if (want.size() != G::Y_ELEMS) {
        std::printf("[FAIL] golden 输出 %zu 个元素，应为 %zu\n", want.size(), G::Y_ELEMS);
        return 1;
    }

    std::vector<uint64_t> s1(G::COUT1), s2(G::COUT2);
    for (int i = 0; i < G::COUT1; ++i) {
        s1[i] = PackScaleEntry(gold.scale1[i]);
    }
    for (int i = 0; i < G::COUT2; ++i) {
        s2[i] = PackDeqF16Entry(gold.scale2[i]);
    }

    std::FILE* f = std::fopen(out.c_str(), "wb");
    if (f == nullptr) {
        std::printf("[FAIL] 打不开 %s\n", out.c_str());
        return 1;
    }

    const uint32_t version = 1;
    const uint32_t ntensors = 9;
    const uint64_t hNonZero = static_cast<uint64_t>(goldNonZero);
    const uint64_t hTies = static_cast<uint64_t>(gold.tiesQ + gold.ties1);
    const uint64_t hSat = static_cast<uint64_t>(gold.satQ + gold.sat1);
    const uint64_t hYElems = static_cast<uint64_t>(G::Y_ELEMS);
    const uint64_t reserved = 0;

    bool ok = std::fwrite("FC2DCASE", 1, 8, f) == 8;
    ok = ok && std::fwrite(&version, 1, 4, f) == 4;
    ok = ok && std::fwrite(&ntensors, 1, 4, f) == 4;
    ok = ok && std::fwrite(&hNonZero, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hTies, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hSat, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hYElems, 1, 8, f) == 8;
    ok = ok && std::fwrite(&reserved, 1, 8, f) == 8;

    // ABI 顺序钉死：x, scale_x, filter1, bias1, scale1, filter2, bias2, scale2 -> y
    // x / y 是 4 维 NCHW，权重是 FRACTAL_Z 的 4 个维度摊开。
    const int64_t fz1k = (G::CI / G::C0) * G::KH * G::KW;    // 9
    const int64_t fz1n = G::COUT1 / 16;                      // 4
    const int64_t fz2k = (G::COUT1 / G::C0) * G::KH * G::KW; // 18
    const int64_t fz2n = G::COUT2 / 16;                      // 6
    ok = ok && WriteTensor(f, "x", ACL_DT_FLOAT16, {1, G::CI, G::HI, G::WI}, in.xNchw.data(), in.xNchw.size() * 2);
    ok = ok && WriteTensor(f, "scale_x", ACL_DT_FLOAT, {1}, &in.qScaleF32, sizeof(float));
    ok = ok && WriteTensor(f, "filter1", ACL_DT_INT8, {fz1k, fz1n, 16, G::C0}, in.w1Dev.data(), in.w1Dev.size());
    ok = ok && WriteTensor(f, "bias1", ACL_DT_INT32, {G::COUT1}, in.b1.data(), in.b1.size() * 4);
    ok = ok && WriteTensor(f, "scale1", ACL_DT_UINT64, {G::COUT1}, s1.data(), s1.size() * 8);
    ok = ok && WriteTensor(f, "filter2", ACL_DT_INT8, {fz2k, fz2n, 16, G::C0}, in.w2Dev.data(), in.w2Dev.size());
    ok = ok && WriteTensor(f, "bias2", ACL_DT_INT32, {G::COUT2}, in.b2.data(), in.b2.size() * 4);
    ok = ok && WriteTensor(f, "scale2", ACL_DT_UINT64, {G::COUT2}, s2.data(), s2.size() * 8);
    ok = ok && WriteTensor(f, "y_expect", ACL_DT_FLOAT16, {1, G::COUT2, G::HO2, G::WO2}, want.data(),
                           want.size() * 2);

    const bool closed = (std::fclose(f) == 0);
    if (!ok || !closed) {
        std::printf("[FAIL] 写 %s 失败（磁盘满？）\n", out.c_str());
        return 1;
    }

    std::printf("\n[OK] 写出 %s\n", out.c_str());
    std::printf("     8 个输入 + golden 输出，共 9 个张量\n");
    std::printf("     把它和 run_fused_conv2d.py 一起拷到有 5102 的机器上\n");
    return 0;
}
