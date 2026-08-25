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
 * \file int8_golden.h
 * \brief self-contained CPU golden + device-layout helpers for the int8 FusedConv2d UT (MC62 / 5102)
 *
 * Header-only, no dependencies beyond the C++17 standard library. Include it in
 * tests/ut/op_kernel/test_fused_conv2d.cpp, or compile it standalone with
 *     g++ -std=c++17 -O2 -DFUSED_CONV2D_GOLDEN_MAIN -x c++ int8_golden.h -o golden && ./golden
 * to get the self-check described at the bottom.
 *
 * ===========================================================================
 * WHY THIS FILE EXISTS, AND WHAT IT IS THE AUTHORITY ON
 * ===========================================================================
 * The kernel img2cols straight out of L1, so it needs the feature map already
 * in NC1HWC0 and the weights already in (c1, kh, kw, c0) K-order -- the order
 * load3d walks. Getting either wrong produces plausible-looking wrong numbers
 * rather than an error, so both transforms live here, once, next to the golden
 * that depends on them. This mirrors harness/fused_data.py, from which the
 * structure (to_nc1hwc0 / weight_to_kmajor / a deliberately naive conv) is
 * taken verbatim; only the dtypes and the requant are new.
 *
 * ===========================================================================
 * THE INT8 NUMERIC PIPELINE, EXACTLY
 * ===========================================================================
 *   x        int8     feature map
 *   w1, w2   int8     weights
 *   b1, b2   int32    biases, delivered through the BT (C2) bias table
 *   L0C      int32    EXACT accumulation. The 5102 cube has no floating-point
 *                     accumulator; mad() on dav-510r2 only exists with an int32
 *                     destination. int8 x int8 -> int32 over K = 576 terms
 *                     cannot overflow (576 * 127 * 127 = 9.29e6 << 2^31), so
 *                     conv1's and conv2's accumulators are EXACT INTEGERS.
 *                     There is no rounding anywhere inside a convolution.
 *
 *   THE ONLY ROUNDING IN THE OPERATOR is the fusion point:
 *
 *   mid_i8 = saturate_int8( round( acc1_i32 * scale1[co] ) )
 *
 *   scale1 is a PER-OUTPUT-CHANNEL float. On the device this is the fixpipe's
 *   VREQ8 quant mode with a per-channel scale table (QuantMode_t::VREQ8 = 8;
 *   the scalar twin REQ8 = 9 is the same arithmetic with one shared scale).
 *   In real quantised inference scale1 is the folded s_x1 * s_w1 / s_x2,
 *   computed on the host; the kernel never derives it.
 *
 *   conv2's int32 accumulator is the output. Two output paths, selected at
 *   compile time by FUSED_CONV2D_GOLDEN_INT8_OUT:
 *       0 (default)  y is int32, straight out of L0C, no rounding at all.
 *                    START HERE: it makes the whole operator exactly integral,
 *                    so any mismatch is a real bug and not a rounding argument.
 *       1            y = saturate_int8(round(acc2 * scale2[co])), the same
 *                    requant again.
 *
 * ===========================================================================
 * ROUNDING MODE -- SETTLED: ROUND-HALF-TO-EVEN
 * ===========================================================================
 * The fixpipe rounds half to EVEN (banker's rounding), i.e. 2.5 -> 2, 3.5 -> 4,
 * -2.5 -> -2. That is FUSED_CONV2D_GOLDEN_ROUND_MODE = 0 and this file's
 * default. Define it to 1 for round-half-AWAY-from-zero (2.5 -> 3, -2.5 -> -3);
 * that mode is kept only so the difference stays demonstrable.
 *
 * HOW IT WAS SETTLED -- it began as an assumption, and this is the measurement
 * that closed it. The fixpipe's quant arithmetic is not visible in any header
 * or in the CPU-debug library's disassembly: libcpudebug.so's
 * copy_matrix_cc_to_cbuf(signed char*, int*, ...) -- which DOES exist, so the
 * int8 fusion point is simulatable -- funnels the conversion through
 * ModelFactory::RunWithPipeId into the closed functional model behind
 * pv_step/pv_mem_read, and the library exports both AscendC::__roundf (which
 * dlopens libm's roundf: half AWAY from zero) and AscendC::__rintf (a bare
 * `frintx`: half to EVEN) with no in-library caller of either, so reading the
 * library settles nothing. What settled it was the fp32-product fix in
 * RequantToInt8: rounding the acc*scale product to fp32 before rounding to
 * integer CREATES three genuine exact ties on this vector (ties1 = 2,
 * ties2 = 1) where the infinite-precision product had none -- and the device
 * agrees with half-to-even on all three, in a run that matches
 * 516,096 / 516,096. See the comment on RequantToInt8.
 *
 * WHY THE TIE COUNT IS SO SMALL: the scales chosen below are odd multiples of a
 * negative power of two (see ChooseScale), so an exact tie needs
 * acc * significand == 2^(e-1) (mod 2^e) with e ~= 21 -- one residue class in
 * 2^21. The harness COUNTS the ties it actually hits and prints them. With the
 * fp32 product it finds 3; with FUSED_CONV2D_GOLDEN_REQUANT_FP32 = 0 it finds
 * 0, and the two rounding modes then give bit-identical goldens on this vector.
 *
 * ===========================================================================
 * A SECOND FIXPIPE DETAIL WORTH KNOWING: float19
 * ===========================================================================
 * A deq-table entry keeps only the top 19 bits of the fp32 scale in [31:13]
 * -- 1 sign + 8 exponent + 10 mantissa bits. A scale the
 * host computes in fp32 is therefore TRUNCATED before the hardware sees it.
 * ChooseScale below emits only float19-exact values, so the golden and the
 * device use bit-identical scales and this cannot become a mystery residual.
 * (If the per-channel VREQ8 table turns out to carry more mantissa bits than
 * the scalar deqScalar, nothing breaks -- a float19 value is exact in any
 * wider format.)
 */

/*
 * PROVENANCE. This file is the design-phase int8 golden (~/scratch5102/
 * int8_golden.h) moved into the test tree so the kernel UT has no dependency on
 * a scratch directory. Exactly one thing changed on the way in: RequantToInt8
 * now rounds its product to fp32 before rounding to integer, because the
 * fixpipe does. See the comment on that function for the measurement.
 */

#ifndef FUSED_CONV2D_INT8_GOLDEN_H
#define FUSED_CONV2D_INT8_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// 0 = int32 output (default, and where the bring-up should start)
// 1 = requantised int8 output
#ifndef FUSED_CONV2D_GOLDEN_INT8_OUT
#define FUSED_CONV2D_GOLDEN_INT8_OUT 0
#endif

// 0 = round half to even (what the fixpipe does)  1 = round half away from zero
// Model the fixpipe's fp32 product precision (see RequantToInt8). ON by
// default: with it off the golden is exact-in-double and disagrees with the
// device on 22 of 516,096 outputs.
#ifndef FUSED_CONV2D_GOLDEN_REQUANT_FP32
#define FUSED_CONV2D_GOLDEN_REQUANT_FP32 1
#endif

#ifndef FUSED_CONV2D_GOLDEN_ROUND_MODE
#define FUSED_CONV2D_GOLDEN_ROUND_MODE 0
#endif

namespace FusedConv2dGolden {

// ---------------------------------------------------------------------------
// Geometry. Mirror of the kernel's own op_kernel/arch35/fused_conv2d_geometry.h,
// which is the authority; test_fused_conv2d.cpp static_asserts the two against
// each other, so drift is a compile error rather than a numeric mystery.
// ---------------------------------------------------------------------------
constexpr int C0 = 32; // int8: 32 bytes / 1 byte
constexpr int CI = 64;
constexpr int HI = 288;
constexpr int WI = 112;
constexpr int C1 = CI / C0; // 2   (was 4 for fp16)
constexpr int KH = 3;
constexpr int KW = 3;
constexpr int PAD = 1;

constexpr int COUT1 = 64;
constexpr int STRIDE1 = 2;
constexpr int HO1 = 144;
constexpr int WO1 = 56;

constexpr int COUT2 = 64;
constexpr int STRIDE2 = 1;
constexpr int HO2 = 144;
constexpr int WO2 = 56;

constexpr int MID_C1 = COUT1 / C0; // 2
constexpr int K1 = C1 * KH * KW * C0;     // 576
constexpr int K2 = MID_C1 * KH * KW * C0; // 576

constexpr size_t X_ELEMS = (size_t)C1 * HI * WI * C0;   // 2,064,384 int8
constexpr size_t W1_ELEMS = (size_t)COUT1 * K1;         //    36,864 int8
constexpr size_t W2_ELEMS = (size_t)COUT2 * K2;         //    36,864 int8
constexpr size_t M_ROWS = (size_t)HO2 * WO2;            //     8,064
constexpr size_t Y_ELEMS = M_ROWS * COUT2;              //   516,096
constexpr size_t MID_ELEMS = (size_t)MID_C1 * HO1 * WO1 * C0; // 516,096 int8

// Target |mid| for the busiest element of each output channel. 100 of 127
// leaves headroom for the float19 rounding of the scale (worst case +2^-10
// relative) while still exercising most of the int8 range.
constexpr int REQUANT_TARGET_ABS = 100;

// ---------------------------------------------------------------------------
// Deterministic pseudo-random source. Same mixer as the fp16 kernel UT, with a
// final avalanche step so that the low bits (which is all an int8 keeps) are
// well distributed. Small, not symmetric, so a wrong index mapping shows up as
// a mismatch instead of cancelling.
// ---------------------------------------------------------------------------
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

// Uniform integer in [lo, hi]. Ranges below are symmetric (odd span) so the
// mean is exactly 0 and a systematic drift in the accumulator is impossible.
inline int RandInt(uint64_t i, uint64_t salt, int lo, int hi)
{
    const uint64_t span = (uint64_t)(hi - lo + 1);
    return lo + (int)(Mix64(i, salt) % span);
}

// Input ranges, and why:
//   x  in [-63, 63]  span 127, sigma ~ 36.7
//   w  in [-31, 31]  span  63, sigma ~ 18.2
// so sigma(acc1) = sqrt(576) * 36.7 * 18.2 ~ 16,000 and the largest of the
// 516,096 conv1 outputs sits near 4.7 sigma ~ 75,000 -- three orders of
// magnitude clear of int32, and wide enough that a per-channel scale genuinely
// has to do work. The absolute worst case 576*63*31 + 8191 = 1.13e6 is still
// 1900x inside int32, so the accumulator is provably exact.
//   b  in [-8191, 8191]: ~0.5 sigma of the accumulator. Big enough that a
// dropped or mis-strided bias is obvious, small enough not to dominate.
constexpr int X_ABS = 63;
constexpr int W_ABS = 31;
constexpr int B_ABS = 8191;

// ---------------------------------------------------------------------------
// Scale handling.
// ---------------------------------------------------------------------------

// Snap a scale to a value that is (a) exactly representable in the fixpipe's
// float19 deqScalar ([31:13] of the fp32 bits) and (b) has an ODD significand.
// (a) removes any host/device disagreement about the scale itself.
// (b) makes exact ties in acc*scale astronomically rare: with scale = m*2^-e
// and m odd, acc*m == 2^(e-1) (mod 2^e) has exactly one solution for acc mod
// 2^e, so ties occur with probability 2^-e.
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

// Round to nearest, with the tie rule selected at compile time. Written out
// rather than calling rint/nearbyint so that it does not depend on the
// process's floating-point rounding mode.
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

// acc_i32 * scale -> int8, with saturation.
//
// THE PRODUCT IS COMPUTED IN FP32, NOT DOUBLE, AND THAT IS LOAD-BEARING.
// `scale` came from ChooseScale, so acc*scale is a dyadic rational that IS
// exact in double. The fixpipe is not: acc needs up to 17 significand bits
// (|acc1| <= 83,979) and the float19 scale up to 11, so the exact product needs
// up to 28 -- four more than fp32's 24. The hardware (and the CPU model)
// therefore round the product to fp32 BEFORE rounding to integer, and a
// double-precision golden disagrees with the device wherever that first
// rounding pushes the value across the .5 boundary.
//
// MEASURED, not assumed. With the double product, the full 24-chunk /
// blockDim-8 run mismatched 22 of 516,096 outputs, every one of them by
// exactly +/-1 and clustered in a few output rows. With the fp32 product the
// same run matches 516,096 / 516,096 EXACTLY. Note also that the fp32 product
// creates 3 genuine exact ties (ties1 = 2, ties2 = 1) where the double product
// had none -- and the device agrees on all three, which independently confirms
// the fixpipe's round-half-to-even (established fact 6).
//
// Set FUSED_CONV2D_GOLDEN_REQUANT_FP32 = 0 to get the old infinite-precision
// behaviour; it is kept only so the difference stays demonstrable.
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
// Layout transforms -- device buffer <-> plain dense NCHW.
//
// These four are the whole reason this file is the authority. Each pair is an
// exact inverse; the UT should GENERATE with the forward direction and DECODE
// with the reverse, so that a layout mistake shows up as a numeric failure
// rather than the golden quietly agreeing with the kernel.
// ---------------------------------------------------------------------------

// [1, CI, HI, WI] -> [C1*HI*WI, C0]. Element (c1, h, w, c0) lands at
// ((c1*HI + h)*WI + w)*C0 + c0, with the true channel ci = c1*C0 + c0.
// This is fused_data.py's to_nc1hwc0 with C0 = 32, C1 = 2.
// GM2L1_ND2NZ in the kernel reads strip c1 as a contiguous [XROWS*WI, C0]
// block at offset (WI*b + c1*HI*WI)*C0 -- the same statement, read the other
// way round. With C0 = 32 that offset is 32x the strip index, not 16x.
inline void ToNc1hwc0(const int8_t* nchw, int8_t* dev)
{
    for (int c1 = 0; c1 < C1; ++c1) {
        for (int h = 0; h < HI; ++h) {
            for (int w = 0; w < WI; ++w) {
                const size_t dst = (((size_t)c1 * HI + h) * WI + w) * C0;
                for (int c0 = 0; c0 < C0; ++c0) {
                    const int ci = c1 * C0 + c0;
                    dev[dst + c0] = nchw[((size_t)ci * HI + h) * WI + w];
                }
            }
        }
    }
}

// [C1*HI*WI, C0] -> [1, CI, HI, WI]. Exact inverse of ToNc1hwc0.
inline void FromNc1hwc0(const int8_t* dev, std::vector<int8_t>& nchw)
{
    nchw.assign((size_t)CI * HI * WI, 0);
    for (int c1 = 0; c1 < C1; ++c1) {
        for (int h = 0; h < HI; ++h) {
            for (int w = 0; w < WI; ++w) {
                const size_t src = (((size_t)c1 * HI + h) * WI + w) * C0;
                for (int c0 = 0; c0 < C0; ++c0) {
                    const int ci = c1 * C0 + c0;
                    nchw[((size_t)ci * HI + h) * WI + w] = dev[src + c0];
                }
            }
        }
    }
}

// [COUT, CIN, KH, KW] -> [COUT, K] with K walked as (c1, kh, kw, c0), which is
// load3d's own order. Element (co, c1, kh, kw, c0) lands at
// co*K + ((c1*KH + kh)*KW + kw)*C0 + c0. fused_data.py's weight_to_kmajor.
// NOTE the int8 shift: K = 576 is unchanged, but it is now 2 strips of 288
// instead of 4 of 144, so a fp16-era weight blob is NOT reinterpretable.
inline void WeightToKMajor(const int8_t* nchw, int cin, int cout, int8_t* dev)
{
    const int c1n = cin / C0;
    const int kdim = c1n * KH * KW * C0;
    for (int co = 0; co < cout; ++co) {
        for (int c1 = 0; c1 < c1n; ++c1) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t dst = (size_t)co * kdim + (((size_t)c1 * KH + kh) * KW + kw) * C0;
                    for (int c0 = 0; c0 < C0; ++c0) {
                        const int ci = c1 * C0 + c0;
                        dev[dst + c0] = nchw[(((size_t)co * cin + ci) * KH + kh) * KW + kw];
                    }
                }
            }
        }
    }
}

// [COUT, K] -> [COUT, CIN, KH, KW]. Exact inverse of WeightToKMajor.
inline void WeightFromKMajor(const int8_t* dev, int cin, int cout, std::vector<int8_t>& nchw)
{
    const int c1n = cin / C0;
    const int kdim = c1n * KH * KW * C0;
    nchw.assign((size_t)cout * cin * KH * KW, 0);
    for (int co = 0; co < cout; ++co) {
        for (int c1 = 0; c1 < c1n; ++c1) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t src = (size_t)co * kdim + (((size_t)c1 * KH + kh) * KW + kw) * C0;
                    for (int c0 = 0; c0 < C0; ++c0) {
                        const int ci = c1 * C0 + c0;
                        nchw[(((size_t)co * cin + ci) * KH + kh) * KW + kw] = dev[src + c0];
                    }
                }
            }
        }
    }
}

// The requantised intermediate, laid out the way conv1's fixpipe leaves it in
// L1 and conv2's img2col reads it: NC1HWC0 over the FULL conv1 image, i.e.
// [MID_C1*HO1*WO1, C0]. The kernel only ever holds MID_ROWS rows of this at a
// time (its `mid` buffer is one chunk's window), so to check milestone M3 --
// "dump the intermediate after conv1's fixpipe" -- slice rows [a, a+MID_ROWS)
// out of each of the MID_C1 strips. This is the layout in which a wrong
// fixpipe dstStride shows up as "strip 0 right, strip 1 displaced".
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
// The convolution. Deliberately a different SHAPE of computation from the
// kernel -- weight-stationary scatter rather than img2col + gemm -- so the two
// cannot share a bug. Accumulation is int32 and therefore EXACT: unlike the
// fp16 golden, there is no summation-order argument to make, and the kernel
// must match this BIT FOR BIT.
//
// The padding bounds are hoisted out of the innermost loop purely so that
// ~300M MACs per conv stay tolerable in a -g / no-optimisation UT build.
// ---------------------------------------------------------------------------
inline void ConvFwdNchwI32(const int8_t* in, int cin, int hi, int wi, const int8_t* wt, const int32_t* bias, int cout,
                           int stride, int ho, int wo, std::vector<int32_t>& out)
{
    out.assign((size_t)cout * ho * wo, 0);
    for (int co = 0; co < cout; ++co) {
        int32_t* op = &out[(size_t)co * ho * wo];
        for (int i = 0; i < ho * wo; ++i) {
            op[i] = bias[co]; // the device seeds L0C with the int32 bias via the BT table
        }
        for (int ci = 0; ci < cin; ++ci) {
            const int8_t* ip = in + (size_t)ci * hi * wi;
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int32_t wv = (int32_t)wt[(((size_t)co * cin + ci) * KH + kh) * KW + kw];
                    if (wv == 0) {
                        continue;
                    }
                    // Column range for which wIdx = w*stride + kw - PAD lands inside [0, wi).
                    int wLo = 0;
                    while (wLo < wo && wLo * stride + kw - PAD < 0) {
                        ++wLo;
                    }
                    int wHiEx = wo;
                    while (wHiEx > wLo && (wHiEx - 1) * stride + kw - PAD >= wi) {
                        --wHiEx;
                    }
                    for (int h = 0; h < ho; ++h) {
                        const int hIdx = h * stride + kh - PAD;
                        if (hIdx < 0 || hIdx >= hi) {
                            continue;
                        }
                        const int8_t* ir = ip + (size_t)hIdx * wi;
                        int32_t* orow = op + (size_t)h * wo;
                        for (int w = wLo; w < wHiEx; ++w) {
                            orow[w] += wv * (int32_t)ir[w * stride + kw - PAD];
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
struct Inputs {
    std::vector<int8_t> xNchw;  // [CI, HI, WI]
    std::vector<int8_t> w1Nchw; // [COUT1, CI, KH, KW]
    std::vector<int8_t> w2Nchw; // [COUT2, COUT1, KH, KW]
    std::vector<int32_t> b1;    // [COUT1]
    std::vector<int32_t> b2;    // [COUT2]
    // device-layout copies, ready to memcpy into the GmAlloc'd buffers
    std::vector<int8_t> xDev;  // [C1*HI*WI, C0]
    std::vector<int8_t> w1Dev; // [COUT1, K1]
    std::vector<int8_t> w2Dev; // [COUT2, K2]
};

struct Golden {
    std::vector<float> scale1;    // [COUT1] conv1 -> mid requant, float19-exact
    std::vector<float> scale2;    // [COUT2] conv2 -> y requant (int8-out path only)
    std::vector<int32_t> acc1;    // [COUT1, HO1, WO1] conv1's exact int32 accumulator
    std::vector<int8_t> midNchw;  // [COUT1, HO1, WO1] the requantised intermediate
    std::vector<int8_t> midDev;   // [MID_C1*HO1*WO1, C0] the same, in the kernel's L1 layout
    std::vector<int32_t> acc2;    // [COUT2, HO2, WO2] conv2's exact int32 accumulator
    std::vector<int32_t> yI32;    // [M_ROWS, COUT2] device layout, int32 output path
    std::vector<int8_t> yI8;      // [M_ROWS, COUT2] device layout, int8 output path

    // diagnostics -- the UT should assert on these, not just print them
    int32_t acc1Min = 0, acc1Max = 0;
    int midMin = 0, midMax = 0;
    int32_t acc2Min = 0, acc2Max = 0;
    int yI8Min = 0, yI8Max = 0;
    long long ties1 = 0, ties2 = 0; // exact x.5 cases: where the rounding mode matters
    long long sat1 = 0, sat2 = 0;   // clamped to [-128, 127]
    long long midNonZero = 0;
    float scale1Min = 0.0f, scale1Max = 0.0f;
};

// Deterministic inputs, in both NCHW and device layout.
inline Inputs GenerateInputs()
{
    Inputs in;
    in.xNchw.resize((size_t)CI * HI * WI);
    for (size_t i = 0; i < in.xNchw.size(); ++i) {
        in.xNchw[i] = (int8_t)RandInt(i, 1, -X_ABS, X_ABS);
    }
    in.w1Nchw.resize((size_t)COUT1 * CI * KH * KW);
    for (size_t i = 0; i < in.w1Nchw.size(); ++i) {
        in.w1Nchw[i] = (int8_t)RandInt(i, 2, -W_ABS, W_ABS);
    }
    in.w2Nchw.resize((size_t)COUT2 * COUT1 * KH * KW);
    for (size_t i = 0; i < in.w2Nchw.size(); ++i) {
        in.w2Nchw[i] = (int8_t)RandInt(i, 3, -W_ABS, W_ABS);
    }
    in.b1.resize(COUT1);
    in.b2.resize(COUT2);
    for (int i = 0; i < COUT1; ++i) {
        in.b1[i] = RandInt((uint64_t)i, 4, -B_ABS, B_ABS);
        in.b2[i] = RandInt((uint64_t)i, 5, -B_ABS, B_ABS);
    }

    in.xDev.assign(X_ELEMS, 0);
    in.w1Dev.assign(W1_ELEMS, 0);
    in.w2Dev.assign(W2_ELEMS, 0);
    ToNc1hwc0(in.xNchw.data(), in.xDev.data());
    WeightToKMajor(in.w1Nchw.data(), CI, COUT1, in.w1Dev.data());
    WeightToKMajor(in.w2Nchw.data(), COUT1, COUT2, in.w2Dev.data());
    return in;
}

// Per-output-channel scale that maps that channel's largest |accumulator| onto
// REQUANT_TARGET_ABS, then snapped to a float19-exact odd-significand value.
// This is a calibration pass, exactly as a real quantiser would do it: it is
// deterministic given the deterministic inputs, so the UT can compute it and
// hand the identical table to the kernel.
inline void DeriveScales(const std::vector<int32_t>& acc, int cout, int hw, std::vector<float>& scale)
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
        scale[co] = (m == 0) ? 1.0f : ChooseScale((float)REQUANT_TARGET_ABS / (float)m);
    }
}

// conv1 (exact int32) -> requant to int8 -> conv2 (exact int32) -> device layout.
inline Golden BuildGolden(const Inputs& in)
{
    Golden g;

    // ---- conv1: exact int32 ----
    ConvFwdNchwI32(in.xNchw.data(), CI, HI, WI, in.w1Nchw.data(), in.b1.data(), COUT1, STRIDE1, HO1, WO1, g.acc1);
    g.acc1Min = g.acc1[0];
    g.acc1Max = g.acc1[0];
    for (size_t i = 0; i < g.acc1.size(); ++i) {
        g.acc1Min = std::min(g.acc1Min, g.acc1[i]);
        g.acc1Max = std::max(g.acc1Max, g.acc1[i]);
    }

    // ---- THE FUSION POINT: int32 L0C -> per-channel requant -> int8 L1 ----
    DeriveScales(g.acc1, COUT1, HO1 * WO1, g.scale1);
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
            const int8_t v = RequantToInt8(g.acc1[idx], s, &tie, &sat);
            g.midNchw[idx] = v;
            g.ties1 += tie ? 1 : 0;
            g.sat1 += sat ? 1 : 0;
            g.midNonZero += (v != 0) ? 1 : 0;
            g.midMin = std::min(g.midMin, (int)v);
            g.midMax = std::max(g.midMax, (int)v);
        }
    }
    MidToNc1hwc0(g.midNchw.data(), g.midDev);

    // ---- conv2: exact int32 ----
    ConvFwdNchwI32(g.midNchw.data(), COUT1, HO1, WO1, in.w2Nchw.data(), in.b2.data(), COUT2, STRIDE2, HO2, WO2, g.acc2);
    g.acc2Min = g.acc2[0];
    g.acc2Max = g.acc2[0];
    for (size_t i = 0; i < g.acc2.size(); ++i) {
        g.acc2Min = std::min(g.acc2Min, g.acc2[i]);
        g.acc2Max = std::max(g.acc2Max, g.acc2[i]);
    }

    // ---- output, in the device's M-major [Ho*Wo, Cout] layout ----
    g.yI32.assign(Y_ELEMS, 0);
    for (int co = 0; co < COUT2; ++co) {
        const int32_t* src = &g.acc2[(size_t)co * HO2 * WO2];
        for (size_t m = 0; m < M_ROWS; ++m) {
            g.yI32[m * COUT2 + co] = src[m];
        }
    }

    DeriveScales(g.acc2, COUT2, HO2 * WO2, g.scale2);
    g.yI8.assign(Y_ELEMS, 0);
    g.yI8Min = 127;
    g.yI8Max = -128;
    for (int co = 0; co < COUT2; ++co) {
        const float s = g.scale2[co];
        const int32_t* src = &g.acc2[(size_t)co * HO2 * WO2];
        for (size_t m = 0; m < M_ROWS; ++m) {
            bool tie = false;
            bool sat = false;
            const int8_t v = RequantToInt8(src[m], s, &tie, &sat);
            g.yI8[m * COUT2 + co] = v;
            g.ties2 += tie ? 1 : 0;
            g.sat2 += sat ? 1 : 0;
            g.yI8Min = std::min(g.yI8Min, (int)v);
            g.yI8Max = std::max(g.yI8Max, (int)v);
        }
    }
    return g;
}

// Which output the UT should compare against, per the compile-time flag.
#if FUSED_CONV2D_GOLDEN_INT8_OUT
using OutElem = int8_t;
inline const std::vector<int8_t>& GoldenOutput(const Golden& g)
{
    return g.yI8;
}
#else
using OutElem = int32_t;
inline const std::vector<int32_t>& GoldenOutput(const Golden& g)
{
    return g.yI32;
}
#endif

} // namespace FusedConv2dGolden

// ---------------------------------------------------------------------------
// Standalone self-check.
//   g++ -std=c++17 -O2 -DFUSED_CONV2D_GOLDEN_MAIN -x c++ int8_golden.h -o golden && ./golden
// It answers the two questions that make a golden untrustworthy: is it all
// zeros, and is it saturating everything?
// ---------------------------------------------------------------------------
#ifdef FUSED_CONV2D_GOLDEN_MAIN
#include <cstdio>

int main()
{
    using namespace FusedConv2dGolden;

    // Layout round-trip. If either transform is wrong the golden is wrong in a
    // way no amount of numeric comparison would reveal, so check it first.
    Inputs in = GenerateInputs();
    std::vector<int8_t> xBack;
    std::vector<int8_t> w1Back;
    FromNc1hwc0(in.xDev.data(), xBack);
    WeightFromKMajor(in.w1Dev.data(), CI, COUT1, w1Back);
    const bool xRt = (xBack == in.xNchw);
    const bool wRt = (w1Back == in.w1Nchw);
    std::printf("layout round-trip:   x %s   w1 %s\n", xRt ? "OK" : "FAIL", wRt ? "OK" : "FAIL");
    if (!xRt || !wRt) {
        return 1;
    }

    Golden g = BuildGolden(in);

    std::printf("conv1 acc  int32   min %11d   max %11d   (int32 headroom %.0fx)\n", g.acc1Min, g.acc1Max,
                2147483647.0 / (double)std::max(1, std::max(-g.acc1Min, g.acc1Max)));
    std::printf("scale1             min %.9g  max %.9g   (float19-exact, odd significand)\n", (double)g.scale1Min,
                (double)g.scale1Max);
    std::printf("mid   int8         min %5d   max %5d   nonzero %lld / %zu   saturated %lld   ties %lld\n", g.midMin,
                g.midMax, g.midNonZero, g.midNchw.size(), g.sat1, g.ties1);
    std::printf("conv2 acc  int32   min %11d   max %11d\n", g.acc2Min, g.acc2Max);
    std::printf("y int8             min %5d   max %5d   saturated %lld   ties %lld\n", g.yI8Min, g.yI8Max, g.sat2,
                g.ties2);

    // Distribution of |mid|, to prove the int8 range is used and not merely
    // touched at the extremes.
    long long hist[8] = {0};
    for (size_t i = 0; i < g.midNchw.size(); ++i) {
        int a = g.midNchw[i] < 0 ? -(int)g.midNchw[i] : (int)g.midNchw[i];
        hist[std::min(7, a / 16)]++;
    }
    std::printf("|mid| histogram (buckets of 16): ");
    for (int i = 0; i < 8; ++i) {
        std::printf("%lld ", hist[i]);
    }
    std::printf("\n");

    long long yNonZero = 0;
    for (size_t i = 0; i < g.yI32.size(); ++i) {
        yNonZero += (g.yI32[i] != 0) ? 1 : 0;
    }
    std::printf("y int32            nonzero %lld / %zu\n", yNonZero, g.yI32.size());

    const bool ok = (g.midMax > 32) && (g.midMin < -32) && (g.sat1 == 0) && (g.sat2 == 0) &&
                    (g.midNonZero > (long long)g.midNchw.size() * 9 / 10) &&
                    (yNonZero > (long long)g.yI32.size() * 9 / 10) && (g.acc2Max > 1000) && (g.acc2Min < -1000);
    std::printf("SELF-CHECK: %s\n", ok ? "PASS" : "FAIL");
    if (g.ties1 != 0 || g.ties2 != 0) {
        std::printf("NOTE: %lld + %lld exact ties -- the rounding mode IS observable on this vector, "
                    "and the device agrees with half-to-even on every one of them.\n",
                    g.ties1, g.ties2);
    } else {
        std::printf("NOTE: zero exact ties -- half-to-even and half-away-from-zero give bit-identical "
                    "goldens on this vector, so the rounding mode cannot affect the test either way.\n");
    }
    return ok ? 0 : 1;
}
#endif // FUSED_CONV2D_GOLDEN_MAIN

#endif // FUSED_CONV2D_INT8_GOLDEN_H
