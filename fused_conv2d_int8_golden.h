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
 * \file fused_conv2d_int8_golden.h
 * \brief CPU golden for FusedConv2d on MC62 (5102) -- fp16 in / int8 compute / fp16 out
 *
 * Header-only, no dependencies beyond the C++17 standard library. Include it in
 * tests/ut/op_kernel/test_fused_conv2d.cpp, or compile it standalone with
 *     g++ -std=c++17 -O2 -DFUSED_CONV2D_GOLDEN_MAIN -x c++ int8_golden.h -o golden && ./golden
 *
 * ===========================================================================
 * THE PIPELINE, EXACTLY
 * ===========================================================================
 *   x        fp16, NCHW [1, 32, 288, 112]
 *   scale_x  fp32 scalar. The KERNEL narrows it to half once and then issues a
 *            Muls<half>, so the golden narrows it the same way -- see qScaleF16.
 *
 *   quant    prod = half(float(x) * float(qScaleF16))      Muls<half>
 *            q    = sat_i8(round_half_even(prod))          Cast<int8_t>(CAST_RINT)
 *
 *   conv1    acc1 = bias1 + filter1 (*) q                  int8 x int8 -> int32
 *            EXACT: 288 * 127 * 127 = 4.65e6 << 2^31, no rounding inside.
 *   requant  mid  = relu(sat_i8(round_half_even(acc1 * scale1[co])))
 *            fixpipe VREQ8 with reluEn. relu and the requant commute here
 *            because scale1 > 0 and the offset field is 0.
 *
 *   conv2    acc2 = bias2 + filter2 (*) mid                int8 x int8 -> int32
 *            bias2 是 int32，和 bias1 一样经 BT（C2）表进 L0C 当累加初值。
 *   dequant  y    = relu(half(float(acc2) * scale2[co]))   fixpipe VDEQF16 + reluEn
 *            |acc2| < 2^24 so the int32 -> fp32 step is exact. relu 和反量化
 *            可交换（scale2 > 0），硬件做在反量化之后。
 *
 *   y        fp16, NCHW [1, 96, 144, 56]
 *
 * Every rounding step above is modelled explicitly. There are exactly three:
 * the Muls, the quantise, and the two fixpipes' conversions. Everything else is
 * exact integer arithmetic.
 *
 * ===========================================================================
 * LAYOUTS -- what this file is the authority on
 * ===========================================================================
 * x and y are NCHW, which is also their GM byte order, so neither needs a
 * transform here: the vectors this file builds ARE the device buffers. Two
 * things still do need one:
 *
 *   weights  [Cout, Cin, KH, KW] -> FRACTAL_Z [Cin1*KH*KW][Cout1][16][C0].
 *            Flattened, that is kf*align16(Cout)*C0 + cout*C0 + cin0, which is
 *            byte-for-byte the NZ block the kernel loads into L0B. Getting it
 *            wrong produces plausible-looking wrong numbers, not an error.
 *   mid      the requantised intermediate, in the NC1HWC0 the fixpipe leaves in
 *            L1 and img2col reads back. Only used to check milestone M3.
 *
 * The NCHW <-> NC1HWC0 transposes that the KERNEL does on the way in and out
 * are deliberately NOT modelled here: this golden works in NCHW throughout, so
 * if the kernel's transposes are wrong the comparison catches it. Sharing the
 * transpose code would hide exactly the bug it is there to find.
 */

#ifndef FUSED_CONV2D_INT8_GOLDEN_H
#define FUSED_CONV2D_INT8_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// 输出恒为 fp16 位模式；这两个宏留着只是为了让老的调用方还能编过。
#ifndef FUSED_CONV2D_GOLDEN_INT8_OUT
#define FUSED_CONV2D_GOLDEN_INT8_OUT 0
#endif
#ifndef FUSED_CONV2D_GOLDEN_FP16_OUT
#define FUSED_CONV2D_GOLDEN_FP16_OUT 1
#endif

// 1 = acc*scale 先舍到 fp32 再取整（硬件就是这么干的）；0 = 全程 double。
#ifndef FUSED_CONV2D_GOLDEN_REQUANT_FP32
#define FUSED_CONV2D_GOLDEN_REQUANT_FP32 1
#endif
// 0 = round-half-to-even（硬件），1 = round-half-away-from-zero。
#ifndef FUSED_CONV2D_GOLDEN_ROUND_MODE
#define FUSED_CONV2D_GOLDEN_ROUND_MODE 0
#endif

namespace FusedConv2dGolden {

// ---------------------------------------------------------------------------
// Geometry. Mirror of op_kernel/arch35/fused_conv2d_geometry.h, which is the
// authority; test_fused_conv2d.cpp static_asserts the two against each other,
// so drift is a compile error rather than a numeric mystery.
// ---------------------------------------------------------------------------
constexpr int C0 = 32; // int8: 32 bytes / 1 byte
constexpr int CI = 32;
constexpr int HI = 288;
constexpr int WI = 112;
constexpr int C1 = CI / C0; // 1 -- the quantised feature map is one NC1HWC0 strip
constexpr int KH = 3;
constexpr int KW = 3;
constexpr int PAD = 1;

constexpr int COUT1 = 64;
constexpr int STRIDE1 = 1;
constexpr int HO1 = 288;
constexpr int WO1 = 112;

constexpr int COUT2 = 96;
constexpr int STRIDE2 = 2;
constexpr int HO2 = 144;
constexpr int WO2 = 56;

constexpr int MID_C1 = COUT1 / C0; // 2
constexpr int K1 = C1 * KH * KW * C0;     // 288
constexpr int K2 = MID_C1 * KH * KW * C0; // 576

constexpr size_t X_ELEMS = (size_t)CI * HI * WI;              // 1,032,192 half  (NCHW)
constexpr size_t W1_ELEMS = (size_t)COUT1 * K1;               //    18,432 int8  (FRACTAL_Z)
constexpr size_t W2_ELEMS = (size_t)COUT2 * K2;               //    55,296 int8  (FRACTAL_Z)
constexpr size_t Y_ELEMS = (size_t)COUT2 * HO2 * WO2;         //   774,144 half  (NCHW)
constexpr size_t MID_ELEMS = (size_t)MID_C1 * HO1 * WO1 * C0; // 2,064,384 int8

// 量化之后 |q| 的目标上界，以及 mid / y 的目标幅度。127 之下留出余量给 scale 的
// float19 舍入，同时把 int8 的量程用满。
constexpr int QUANT_TARGET_ABS = 100;
constexpr int REQUANT_TARGET_ABS = 100;
constexpr float DEQUANT_TARGET_ABS = 100.0f;

inline uint64_t Mix64(uint64_t i, uint64_t salt)
{
    uint64_t x = i * 6364136223846793005ULL + salt * 1442695040888963407ULL + 1ULL;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
inline int RandInt(uint64_t i, uint64_t salt, int lo, int hi)
{
    const uint64_t span = (uint64_t)(hi - lo + 1);
    return lo + (int)(Mix64(i, salt) % span);
}
inline uint16_t F32ToF16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t man = u & 0x007FFFFFu;

    if (exp == 128) {                       // Inf / NaN
        return (uint16_t)(sign | 0x7C00u | (man ? 0x0200u : 0u));
    }
    if (exp > 15) {                         // 上溢 -> Inf
        return (uint16_t)(sign | 0x7C00u);
    }
    // exp == -25 这一档不能砍：它正好落在最小次正规 2^-24 的一半上，尾数非零就要
    // 进位到 1。只有 exp <= -26 才恒为 0（此时 shift 也会超过 31，是真正的 UB 边界）。
    if (exp < -25) {                        // 太小 -> 0
        return (uint16_t)sign;
    }
    uint32_t sig;                           // 24 位有效数（含隐含位）
    int32_t shift;
    if (exp < -14) {                        // 次正规
        // 次正规结果：value = sig * 2^(exp-23)，要表示成 m * 2^-24，
        // 于是 m = sig >> (-exp-1)。（写成 14-exp 是错的，扫描对拍抓到的。）
        sig = man | 0x00800000u;
        shift = -exp - 1;
        exp = -15;
    } else {
        sig = man;
        shift = 13;
    }
    const uint32_t lsb = 1u << shift;
    const uint32_t half_ = lsb >> 1;
    const uint32_t rem = sig & (lsb - 1);
    uint32_t out = sig >> shift;
    if (rem > half_ || (rem == half_ && (out & 1u))) {   // round-half-to-even
        ++out;
    }
    uint32_t e16 = (uint32_t)(exp + 15);
    if (exp == -15) {                       // 次正规：进位可能把它抬成正规数
        return (uint16_t)(sign | out);
    }
    if (out & 0x0400u) {                    // 尾数进位溢出，阶码加一
        out = 0;
        ++e16;
        if (e16 >= 31u) {
            return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | (e16 << 10) | (out & 0x03FFu));
}
inline float ChooseScale(float raw)
{
    if (!(raw > 0.0f) || !std::isfinite(raw)) {
        return 1.0f;
    }
    uint32_t u;
    std::memcpy(&u, &raw, sizeof(u));
    u &= 0xFFFFE000u; // keep sign + exponent + top 10 mantissa bits  == the table's float19
    u |= 0x00002000u; // force the lowest retained mantissa bit -> odd significand
    float out;
    std::memcpy(&out, &u, sizeof(out));
    return out;
}
inline double RoundNearest(double v, bool* isTie)
{
    const double f = std::floor(v);
    const double d = v - f;
    if (d > 0.5) {
        if (isTie != nullptr) {
            *isTie = false;
        }
        return f + 1.0;
    }
    if (d < 0.5) {
        if (isTie != nullptr) {
            *isTie = false;
        }
        return f;
    }
    if (isTie != nullptr) {
        *isTie = true; // v is exactly x.5 -- the one place the two modes differ
    }
#if FUSED_CONV2D_GOLDEN_ROUND_MODE == 0
    // half to even
    return (std::fmod(f, 2.0) == 0.0) ? f : f + 1.0;
#else
    // half away from zero
    return (v >= 0.0) ? f + 1.0 : f;
#endif
}
inline int8_t RequantToInt8(int32_t acc, float scale, bool* isTie, bool* isSat)
{
#if FUSED_CONV2D_GOLDEN_REQUANT_FP32
    const double v = (double)((float)acc * scale);
#else
    const double v = (double)acc * (double)scale;
#endif
    const double r = RoundNearest(v, isTie);
    if (isSat != nullptr) {
        *isSat = (r > 127.0 || r < -128.0);
    }
    const double c = std::min(127.0, std::max(-128.0, r));
    return (int8_t)(int)c;
}

// ---------------------------------------------------------------------------
// fp16 位模式 -> float。F32ToF16Bits 的逆。次正规那一支写成 m * 2^-24 的直算法:
// 手写规格化循环的第一稿把指数算差了 1，65536 个位模式里错了 2046 个。
// ---------------------------------------------------------------------------
inline float F16BitsToF32(uint16_t h)
{
    const uint32_t s = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-08f; // 2^-24
        return s ? -v : v;
    }
    uint32_t u;
    if (e == 31) {
        u = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        u = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// 输入量化。一步一步照着 kernel 的两条指令来:
//   Muls<half>          prod = half(x * s)      —— 两个 fp16 相乘，积在 fp32 上
//                                                 是精确的，再舍一次到 fp16
//   Cast<int8_t>(RINT)  q    = sat(rne(prod))   —— round-half-to-even + 饱和
// ---------------------------------------------------------------------------
inline int8_t QuantizeOne(uint16_t xBits, uint16_t sBits, bool* isTie, bool* isSat)
{
    const float prod = F16BitsToF32(xBits) * F16BitsToF32(sBits);
    const double v = (double)F16BitsToF32(F32ToF16Bits(prod));
    const double r = RoundNearest(v, isTie);
    if (isSat != nullptr) {
        *isSat = (r > 127.0 || r < -128.0);
    }
    return (int8_t)(int)std::min(127.0, std::max(-128.0, r));
}

// ---------------------------------------------------------------------------
// [COUT, CIN, KH, KW] -> FRACTAL_Z [Cin1*KH*KW][Cout1][16][C0]，摊平成一维。
// 元素 (co, ci, kh, kw)，记 c1 = ci/C0、c0 = ci%C0、n1 = co/16、n0 = co%16，落在
//     ((c1*KH + kh)*KW + kw) * align16(COUT) * C0 + (n1*16 + n0) * C0 + c0
// 也就是 kf*align16(Cout)*C0 + co*C0 + c0 —— 和 kernel 装进 L0B 的 NZ 块同形。
// ---------------------------------------------------------------------------
inline void WeightToFractalZ(const int8_t* nchw, int cin, int cout, int8_t* dev)
{
    const int c1n = cin / C0;
    const int coutAl = (cout + 15) / 16 * 16;
    for (int co = 0; co < cout; ++co) {
        for (int c1 = 0; c1 < c1n; ++c1) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t kf = ((size_t)c1 * KH + kh) * KW + kw;
                    const size_t dst = kf * coutAl * C0 + (size_t)co * C0;
                    for (int c0 = 0; c0 < C0; ++c0) {
                        const int ci = c1 * C0 + c0;
                        dev[dst + c0] = nchw[(((size_t)co * cin + ci) * KH + kh) * KW + kw];
                    }
                }
            }
        }
    }
}

// FRACTAL_Z -> [COUT, CIN, KH, KW]。WeightToFractalZ 的严格逆，用来做布局往返自检。
inline void WeightFromFractalZ(const int8_t* dev, int cin, int cout, std::vector<int8_t>& nchw)
{
    const int c1n = cin / C0;
    const int coutAl = (cout + 15) / 16 * 16;
    nchw.assign((size_t)cout * cin * KH * KW, 0);
    for (int co = 0; co < cout; ++co) {
        for (int c1 = 0; c1 < c1n; ++c1) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t kf = ((size_t)c1 * KH + kh) * KW + kw;
                    const size_t src = kf * coutAl * C0 + (size_t)co * C0;
                    for (int c0 = 0; c0 < C0; ++c0) {
                        const int ci = c1 * C0 + c0;
                        nchw[(((size_t)co * cin + ci) * KH + kh) * KW + kw] = dev[src + c0];
                    }
                }
            }
        }
    }
}

// 重量化之后的中间量，按 conv1 的 fixpipe 留在 L1、conv2 的 img2col 读回的那个
// 排布来:整张 conv1 输出的 NC1HWC0，即 [MID_C1*HO1*WO1, C0]。kernel 一次只持有
// 其中 MID_ROWS 行，所以要验「conv1 之后 dump 中间量」时，从每个 strip 里切
// [a, a+MID_ROWS) 行出来比。fixpipe 的 dstStride 写错就表现为 strip 0 对、
// 后面的 strip 整体错位。
inline void MidToNc1hwc0(const int8_t* midNchw, std::vector<int8_t>& dev)
{
    dev.assign(MID_ELEMS, 0);
    for (int c1 = 0; c1 < MID_C1; ++c1) {
        for (int h = 0; h < HO1; ++h) {
            for (int w = 0; w < WO1; ++w) {
                const size_t dst = (((size_t)c1 * HO1 + h) * WO1 + w) * C0;
                for (int c0 = 0; c0 < C0; ++c0) {
                    const int ci = c1 * C0 + c0;
                    dev[dst + c0] = midNchw[((size_t)ci * HO1 + h) * WO1 + w];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 卷积。刻意和 kernel 的形状不同 —— 权重驻留的散射，而不是 img2col + gemm ——
// 这样两边不会共享同一个 bug。累加是 int32 且精确。
// bias 可以为 nullptr（conv2 在 L0C 里没有 bias）。
// ---------------------------------------------------------------------------
inline void ConvFwdNchwI32(const int8_t* in, int cin, int hi, int wi, const int8_t* wt, const int32_t* bias, int cout,
                           int stride, int ho, int wo, std::vector<int32_t>& out)
{
    out.assign((size_t)cout * ho * wo, 0);
    for (int co = 0; co < cout; ++co) {
        int32_t* o = &out[(size_t)co * ho * wo];
        const int32_t b = (bias != nullptr) ? bias[co] : 0;
        for (int i = 0; i < ho * wo; ++i) {
            o[i] = b;
        }
        for (int ci = 0; ci < cin; ++ci) {
            const int8_t* ip = &in[(size_t)ci * hi * wi];
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int32_t w = (int32_t)wt[(((size_t)co * cin + ci) * KH + kh) * KW + kw];
                    if (w == 0) {
                        continue;
                    }
                    // 把 pad 的边界从最内层循环里提出来，纯粹是为了让 -g / -O0 的
                    // UT 构建还跑得动。
                    int hLo = 0;
                    while (hLo < ho && hLo * stride + kh - PAD < 0) {
                        ++hLo;
                    }
                    int hHiEx = ho;
                    while (hHiEx > hLo && (hHiEx - 1) * stride + kh - PAD >= hi) {
                        --hHiEx;
                    }
                    int wLo = 0;
                    while (wLo < wo && wLo * stride + kw - PAD < 0) {
                        ++wLo;
                    }
                    int wHiEx = wo;
                    while (wHiEx > wLo && (wHiEx - 1) * stride + kw - PAD >= wi) {
                        --wHiEx;
                    }
                    for (int h = hLo; h < hHiEx; ++h) {
                        const int ih = h * stride + kh - PAD;
                        const int8_t* irow = &ip[(size_t)ih * wi];
                        int32_t* orow = &o[(size_t)h * wo];
                        for (int ww = wLo; ww < wHiEx; ++ww) {
                            orow[ww] += w * (int32_t)irow[ww * stride + kw - PAD];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs / outputs
// ---------------------------------------------------------------------------
constexpr int W_ABS = 31;    // |filter|
constexpr int B_ABS = 8191; // |bias|，int32，加在各自卷积的累加器上
constexpr int X_STEPS = 1000; // x 取 k/1024，k 在 [-1000, 1000]

struct Inputs {
    // x 是 NCHW fp16，而 NCHW 就是它在 GM 上的字节序 —— 这个 vector 直接就是设备
    // 缓冲，没有第二份布局。
    std::vector<uint16_t> xNchw; // [CI, HI, WI] 的 fp16 位模式
    float qScaleF32 = 1.0f;      // 算子输入 scale_x
    uint16_t qScaleF16 = 0;      // kernel 实际乘的那个 half（scale_x 窄化一次）
    std::vector<int8_t> w1Nchw;  // [COUT1, CI, KH, KW]
    std::vector<int8_t> w2Nchw;  // [COUT2, COUT1, KH, KW]
    std::vector<int32_t> b1;     // [COUT1] int32
    std::vector<int32_t> b2;     // [COUT2] int32

    std::vector<int8_t> w1Dev; // FRACTAL_Z
    std::vector<int8_t> w2Dev; // FRACTAL_Z
};

struct Golden {
    std::vector<int8_t> qNchw;   // [CI, HI, WI] 量化之后的输入
    std::vector<float> scale1;   // [COUT1] conv1 -> mid 重量化，float19 精确
    std::vector<float> scale2;   // [COUT2] conv2 -> fp16 反量化，float19 精确
    std::vector<int32_t> acc1;   // [COUT1, HO1, WO1] conv1 的精确 int32 累加器
    std::vector<int8_t> midNchw; // [COUT1, HO1, WO1] 重量化 + relu 之后的中间量
    std::vector<int8_t> midDev;  // 同上，kernel 的 L1 排布
    std::vector<int32_t> acc2;   // [COUT2, HO2, WO2]
    std::vector<uint16_t> yF16;  // [COUT2, HO2, WO2] NCHW，fp16 位模式 == 设备缓冲

    // 诊断量 —— UT 该对这些断言，而不只是打印出来
    int qMin = 0, qMax = 0;
    long long qNonZero = 0, tiesQ = 0, satQ = 0;
    int32_t acc1Min = 0, acc1Max = 0;
    int midMin = 0, midMax = 0;
    long long midNonZero = 0, ties1 = 0, sat1 = 0;
    int32_t acc2Min = 0, acc2Max = 0;
    float yMin = 0.0f, yMax = 0.0f;
    long long yNonZero = 0;
    float scale1Min = 0.0f, scale1Max = 0.0f;
    float scale2Min = 0.0f, scale2Max = 0.0f;
};

// 确定性输入。x 取 k/1024（k 整数），在 fp16 里是精确可表示的，所以「主机写下的
// x」和「设备读到的 x」之间不存在第二次舍入。
inline Inputs GenerateInputs()
{
    Inputs in;
    in.xNchw.resize(X_ELEMS);
    float maxAbsX = 0.0f;
    for (size_t i = 0; i < X_ELEMS; ++i) {
        const float v = (float)RandInt(i, 1, -X_STEPS, X_STEPS) * (1.0f / 1024.0f);
        in.xNchw[i] = F32ToF16Bits(v);
        maxAbsX = std::max(maxAbsX, std::fabs(v));
    }
    // scale_x 选成「最大的 |x| 正好量到 QUANT_TARGET_ABS」，再吸附到一个 half 上。
    // 吸附这一下很重要:kernel 拿到的是 fp32，但它会窄化成 half 再乘，吸附之后
    // 主机和设备对「乘数是多少」的理解就不可能有分歧。
    const float rawQ = (maxAbsX > 0.0f) ? ((float)QUANT_TARGET_ABS / maxAbsX) : 1.0f;
    in.qScaleF16 = F32ToF16Bits(rawQ);
    in.qScaleF32 = F16BitsToF32(in.qScaleF16);

    in.w1Nchw.resize((size_t)COUT1 * CI * KH * KW);
    for (size_t i = 0; i < in.w1Nchw.size(); ++i) {
        in.w1Nchw[i] = (int8_t)RandInt(i, 2, -W_ABS, W_ABS);
    }
    in.w2Nchw.resize((size_t)COUT2 * COUT1 * KH * KW);
    for (size_t i = 0; i < in.w2Nchw.size(); ++i) {
        in.w2Nchw[i] = (int8_t)RandInt(i, 3, -W_ABS, W_ABS);
    }
    in.b1.resize(COUT1);
    for (int i = 0; i < COUT1; ++i) {
        in.b1[i] = RandInt((uint64_t)i, 4, -B_ABS, B_ABS);
    }
    in.b2.resize(COUT2);
    for (int i = 0; i < COUT2; ++i) {
        in.b2[i] = RandInt((uint64_t)i, 5, -B_ABS, B_ABS);
    }

    in.w1Dev.assign(W1_ELEMS, 0);
    in.w2Dev.assign(W2_ELEMS, 0);
    WeightToFractalZ(in.w1Nchw.data(), CI, COUT1, in.w1Dev.data());
    WeightToFractalZ(in.w2Nchw.data(), COUT1, COUT2, in.w2Dev.data());
    return in;
}

// 逐通道的 scale：把该通道最大的 |累加器| 映到 target，再吸附到 float19 精确的
// 奇尾数值上。这就是真实量化器做的标定，且是确定性的，所以 UT 能算出来并把同一
// 张表喂给 kernel。
inline void DeriveScales(const std::vector<int32_t>& acc, int cout, int hw, float target, std::vector<float>& scale)
{
    scale.assign((size_t)cout, 1.0f);
    for (int co = 0; co < cout; ++co) {
        int32_t m = 0;
        const int32_t* p = &acc[(size_t)co * hw];
        for (int i = 0; i < hw; ++i) {
            const int32_t a = p[i] < 0 ? -p[i] : p[i];
            if (a > m) {
                m = a;
            }
        }
        scale[co] = (m == 0) ? 1.0f : ChooseScale(target / (float)m);
    }
}

inline Golden BuildGolden(const Inputs& in)
{
    Golden g;

    // ---- 量化：fp16 -> int8 ----
    g.qNchw.assign(X_ELEMS, 0);
    g.qMin = 127;
    g.qMax = -128;
    for (size_t i = 0; i < X_ELEMS; ++i) {
        bool tie = false;
        bool sat = false;
        const int8_t q = QuantizeOne(in.xNchw[i], in.qScaleF16, &tie, &sat);
        g.qNchw[i] = q;
        g.tiesQ += tie ? 1 : 0;
        g.satQ += sat ? 1 : 0;
        g.qNonZero += (q != 0) ? 1 : 0;
        g.qMin = std::min(g.qMin, (int)q);
        g.qMax = std::max(g.qMax, (int)q);
    }

    // ---- conv1：精确 int32，bias1 加在累加器上 ----
    ConvFwdNchwI32(g.qNchw.data(), CI, HI, WI, in.w1Nchw.data(), in.b1.data(), COUT1, STRIDE1, HO1, WO1, g.acc1);
    g.acc1Min = g.acc1[0];
    g.acc1Max = g.acc1[0];
    for (size_t i = 0; i < g.acc1.size(); ++i) {
        g.acc1Min = std::min(g.acc1Min, g.acc1[i]);
        g.acc1Max = std::max(g.acc1Max, g.acc1[i]);
    }

    // ---- 融合点：int32 L0C -> 逐通道重量化 + relu -> int8 L1 ----
    // relu 和重量化在这里可交换（scale1 > 0、offset 为 0），所以先量化后 relu 与
    // 先 relu 后量化逐位相同；fixpipe 的 reluEn 走的是前者。
    DeriveScales(g.acc1, COUT1, HO1 * WO1, (float)REQUANT_TARGET_ABS, g.scale1);
    g.scale1Min = g.scale1[0];
    g.scale1Max = g.scale1[0];
    for (int co = 0; co < COUT1; ++co) {
        g.scale1Min = std::min(g.scale1Min, g.scale1[co]);
        g.scale1Max = std::max(g.scale1Max, g.scale1[co]);
    }
    g.midNchw.assign((size_t)COUT1 * HO1 * WO1, 0);
    g.midMin = 127;
    g.midMax = -128;
    for (int co = 0; co < COUT1; ++co) {
        const float s = g.scale1[co];
        for (int i = 0; i < HO1 * WO1; ++i) {
            const size_t idx = (size_t)co * HO1 * WO1 + i;
            bool tie = false;
            bool sat = false;
            int8_t v = RequantToInt8(g.acc1[idx], s, &tie, &sat);
            if (v < 0) { // ReLU
                v = 0;
            }
            g.midNchw[idx] = v;
            g.ties1 += tie ? 1 : 0;
            g.sat1 += sat ? 1 : 0;
            g.midNonZero += (v != 0) ? 1 : 0;
            g.midMin = std::min(g.midMin, (int)v);
            g.midMax = std::max(g.midMax, (int)v);
        }
    }
    MidToNc1hwc0(g.midNchw.data(), g.midDev);

    // ---- conv2：精确 int32，bias2 经 BT 表加在累加器上 ----
    ConvFwdNchwI32(g.midNchw.data(), COUT1, HO1, WO1, in.w2Nchw.data(), in.b2.data(), COUT2, STRIDE2, HO2, WO2,
                   g.acc2);
    g.acc2Min = g.acc2[0];
    g.acc2Max = g.acc2[0];
    for (size_t i = 0; i < g.acc2.size(); ++i) {
        g.acc2Min = std::min(g.acc2Min, g.acc2[i]);
        g.acc2Max = std::max(g.acc2Max, g.acc2[i]);
    }

    // ---- 收尾：反量化 -> relu -> NCHW fp16，全部由 fixpipe 随路完成 ----
    DeriveScales(g.acc2, COUT2, HO2 * WO2, DEQUANT_TARGET_ABS, g.scale2);
    g.scale2Min = g.scale2[0];
    g.scale2Max = g.scale2[0];
    for (int co = 0; co < COUT2; ++co) {
        g.scale2Min = std::min(g.scale2Min, g.scale2[co]);
        g.scale2Max = std::max(g.scale2Max, g.scale2[co]);
    }
    g.yF16.assign(Y_ELEMS, 0);
    bool first = true;
    for (int co = 0; co < COUT2; ++co) {
        const float s = g.scale2[co];
        const int32_t* src = &g.acc2[(size_t)co * HO2 * WO2];
        uint16_t* dst = &g.yF16[(size_t)co * HO2 * WO2];
        for (int i = 0; i < HO2 * WO2; ++i) {
            // VDEQF16：acc(int32) -> fp32（|acc| < 2^24，精确）-> 乘 float19 的
            // scale -> 舍到 fp16。scale 已经被 ChooseScale 掩成 float19，和硬件表
            // 里的位一致，所以这一步逐位可对。
            const uint16_t dq = F32ToF16Bits((float)src[i] * s);
            // fixpipe 的 reluEn：做在反量化之后，而且是**保符号的置零** —— 负数
            // 出来的是 -0（位模式 0x8000），不是 +0。数学上 -0 == +0，下游谁也
            // 不受影响，但逐位比较会全灭，所以这里照它的行为建模。
            // 这是整份 golden 里唯一一处「照实测行为写」而不是从定义推出来的:
            // 硬件把 relu 实现成清掉阶码和尾数、留下符号位。
            const float dv = F16BitsToF32(dq);
            const uint16_t y = (dv > 0.0f) ? dq : (uint16_t)(dq & 0x8000u);
            dst[i] = y;
            const float yv = F16BitsToF32(y);
            g.yNonZero += ((y & 0x7FFFu) != 0) ? 1 : 0;
            if (first) {
                g.yMin = g.yMax = yv;
                first = false;
            } else {
                g.yMin = std::min(g.yMin, yv);
                g.yMax = std::max(g.yMax, yv);
            }
        }
    }
    return g;
}

// UT 该拿哪个输出去比。
using OutElem = uint16_t;
inline const std::vector<uint16_t>& GoldenOutput(const Golden& g)
{
    return g.yF16;
}

} // namespace FusedConv2dGolden

// ---------------------------------------------------------------------------
// Standalone self-check.
//   g++ -std=c++17 -O2 -DFUSED_CONV2D_GOLDEN_MAIN -x c++ int8_golden.h -o golden && ./golden
// 它回答的是「什么样的 golden 不可信」:全 0、全饱和、布局变换写反了。
// 注意 relu 之后 mid 和 y 大约一半是 0 —— 那是对的，不是退化。
// ---------------------------------------------------------------------------
#ifdef FUSED_CONV2D_GOLDEN_MAIN
#include <cstdio>

int main()
{
    using namespace FusedConv2dGolden;

    // 布局往返。FRACTAL_Z 写反了的话，后面再怎么比数值都发现不了。
    Inputs in = GenerateInputs();
    std::vector<int8_t> w1Back;
    std::vector<int8_t> w2Back;
    WeightFromFractalZ(in.w1Dev.data(), CI, COUT1, w1Back);
    WeightFromFractalZ(in.w2Dev.data(), COUT1, COUT2, w2Back);
    const bool w1Rt = (w1Back == in.w1Nchw);
    const bool w2Rt = (w2Back == in.w2Nchw);
    std::printf("FRACTAL_Z round-trip:  filter1 %s   filter2 %s\n", w1Rt ? "OK" : "FAIL", w2Rt ? "OK" : "FAIL");
    if (!w1Rt || !w2Rt) {
        return 1;
    }

    Golden g = BuildGolden(in);

    std::printf("scale_x            %.9g  (窄化成 half 之后 0x%04X，主机和设备乘的是同一个数)\n",
                (double)in.qScaleF32, in.qScaleF16);
    std::printf("quant  int8        min %5d   max %5d   nonzero %lld / %zu   saturated %lld   ties %lld\n", g.qMin,
                g.qMax, g.qNonZero, X_ELEMS, g.satQ, g.tiesQ);
    std::printf("conv1 acc  int32   min %11d   max %11d   (int32 headroom %.0fx)\n", g.acc1Min, g.acc1Max,
                2147483647.0 / (double)std::max(1, std::max(-g.acc1Min, g.acc1Max)));
    std::printf("scale1             min %.9g  max %.9g   (float19-exact, odd significand)\n", (double)g.scale1Min,
                (double)g.scale1Max);
    std::printf("mid   int8 (relu)  min %5d   max %5d   nonzero %lld / %zu   saturated %lld   ties %lld\n", g.midMin,
                g.midMax, g.midNonZero, g.midNchw.size(), g.sat1, g.ties1);
    std::printf("conv2 acc  int32   min %11d   max %11d   (fp32 精确上限 2^24 = 16777216)\n", g.acc2Min, g.acc2Max);
    std::printf("scale2             min %.9g  max %.9g\n", (double)g.scale2Min, (double)g.scale2Max);
    std::printf("y  fp16 (+bias,relu) min %.6g   max %.6g   nonzero %lld / %zu\n", (double)g.yMin, (double)g.yMax,
                g.yNonZero, Y_ELEMS);

    // |mid| 的分布，证明 int8 的量程是被用满的，而不只是碰到了两端。
    long long hist[8] = {0};
    for (size_t i = 0; i < g.midNchw.size(); ++i) {
        const int a = g.midNchw[i] < 0 ? -(int)g.midNchw[i] : (int)g.midNchw[i];
        hist[std::min(7, a / 16)]++;
    }
    std::printf("|mid| histogram (buckets of 16): ");
    for (int i = 0; i < 8; ++i) {
        std::printf("%lld ", hist[i]);
    }
    std::printf("\n");

    const double midNzFrac = (double)g.midNonZero / (double)g.midNchw.size();
    const double yNzFrac = (double)g.yNonZero / (double)Y_ELEMS;

    const bool ok =
        // 量化用满了 int8 的两端，且没有饱和
        (g.qMax > 64) && (g.qMin < -64) && (g.satQ == 0) &&
        // conv1 的累加器远没有溢出
        (g.acc1Max > 1000) && (g.acc1Min < -1000) &&
        // relu 之后 mid 非负、用到了量程上端、没有饱和
        (g.midMin == 0) && (g.midMax > 64) && (g.sat1 == 0) &&
        // relu 砍掉的应当是「一部分」，不是全部也不是没有
        (midNzFrac > 0.2 && midNzFrac < 0.8) &&
        // conv2 的累加器留在 fp32 可精确表示的范围里
        (std::max(-(long long)g.acc2Min, (long long)g.acc2Max) < (1LL << 24)) &&
        // 输出非负、有量级、relu 同样只砍掉一部分
        (g.yMin >= 0.0f) && (g.yMax > 1.0f) && (yNzFrac > 0.2 && yNzFrac < 0.9);
    std::printf("SELF-CHECK: %s   (mid 非零 %.1f%%，y 非零 %.1f%% —— relu 之后本来就该有一半左右是 0)\n",
                ok ? "PASS" : "FAIL", midNzFrac * 100.0, yNzFrac * 100.0);
    if (g.tiesQ != 0 || g.ties1 != 0) {
        std::printf("NOTE: %lld + %lld 个恰好 x.5 的样本 —— 舍入模式在这组数据上是可观测的，\n"
                    "      设备在每一个上都和 half-to-even 一致。\n",
                    g.tiesQ, g.ties1);
    } else {
        std::printf("NOTE: 一个 x.5 都没有 —— 两种舍入模式给出逐位相同的 golden，\n"
                    "      这组向量证明不了舍入模式。\n");
    }
    return ok ? 0 : 1;
}
#endif // FUSED_CONV2D_GOLDEN_MAIN

#endif // FUSED_CONV2D_INT8_GOLDEN_H
