/*
 * fused_conv2d 的 CPU 参考实现 —— 全 fp16 定点版。
 *
 *   x fp16 NCHW ─conv1 3x3 s1 p1 +bias1(fp16) +relu─► mid fp16
 *               ─conv2 3x3 s2 p1 +bias2(fp16) +relu─► y fp16 NCHW
 *
 * 全程定点（f162s32），没有量化、没有向量。权重 FRACTAL_Z fp16（C0 = 16）。
 *
 * ===========================================================================
 * 这份 golden 里哪些是确证的、哪些是假设 —— 先说清楚。
 * ===========================================================================
 *
 * 确证（读 CANN / ops-nn 源码得到）：
 *   * cube 走 f162s32：half x half -> int32，带 fixShiftVal 操作数
 *     (dav_m510/kernel_operator_mm_impl.h:355)
 *   * fixedShiftValue 是算子属性，范围 [0, 58]
 *     (conv2d_v2_base_tiling_check_attrs.cpp:449 CheckFixedShiftValueLegal)
 *   * 三个消费者的方向：mmad 用原值，L1->BT 和 fixpipe 用 58 - 原值
 *     (conv2d_small_kernel.h:354 / :937 / :1111)
 *   * fixpipe 的 deqScalar 恒为 float 1.0 的位型，缩放全由 shift 完成
 *     (conv2d_small_kernel.h:27 FLOAT_ONE_FIXED_POINT / :1160)
 *
 * **假设**（没有文档，也没有硬件实测）：
 *   fixedShiftValue = S 表示定点表示的二进制小数点位置，即
 *       acc_int32 = round( sum(a*b) * 2^S ) + round(bias * 2^S)
 *       out_fp16  = fp16( acc_int32 * 2^-S )
 *   那个 58 是硬件移位域的宽度，所以出口方向写成 58 - S。
 *
 * 这个假设**必须先在板上验**再依赖：拿现成的 conv2d_v2 跑一个小 shape，同一组
 * 输入把 fixedShiftValue 从 0 扫到 58，看输出是不是每加 1 就整体乘/除 2。在那之
 * 前，下面 GoldenFixed() 算出来的数只能当参考，不能当判据。
 *
 * 所以这份 golden 同时给两个答案：
 *   GoldenExact()  纯 fp32 参考 —— 「这个算子到底有没有在算这个卷积」，用相对
 *                  误差判，不受上面那个假设影响；
 *   GoldenFixed(S) 定点模型 —— 假设成立时应当逐位相等。
 * 板上先看 GoldenExact 的相对误差；过了再看 GoldenFixed 能不能逐位对上。
 */
#ifndef FUSED_CONV2D_GOLDEN_H
#define FUSED_CONV2D_GOLDEN_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdio>

namespace fc2d_golden {

// ---------------------------------------------------------------------------
// 形状（和 op_kernel/arch35/fused_conv2d_geometry.h 必须一致）
// ---------------------------------------------------------------------------
constexpr int CI = 32, HI = 288, WI = 112;
constexpr int COUT1 = 64, HO1 = 288, WO1 = 112, STRIDE1 = 1;
constexpr int COUT2 = 96, HO2 = 144, WO2 = 56, STRIDE2 = 2;
constexpr int KH = 3, KW = 3, PAD = 1;
constexpr int C0 = 16;                 // fp16 -> 32B / 2
constexpr int FIX_SHIFT_LEN_A16W16 = 58;

// ---------------------------------------------------------------------------
// fp16 <-> fp32。自己写而不是用 _Float16，是为了在任何编译器上位型都一样 ——
// 板上和主机上比对的是**位**，不是近似值。
//
// 上一版这里的 F16ToFloat 有个次正规数的指数 bug（2046/65536 个值不对），是靠
// 对 _Float16 做穷举自检抓出来的。下面这两个函数同样有穷举自检，见 main()。
// ---------------------------------------------------------------------------
inline uint16_t F32ToF16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) {                       // Inf / NaN
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {                                       // 上溢 -> Inf
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {                                          // 次正规 / 下溢
        if (exp < -10) {
            return (uint16_t)sign;
        }
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t q = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) {        // 向最近偶数
            ++q;
        }
        return (uint16_t)(sign | q);
    }
    uint32_t q = man >> 13;
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1))) {
        ++q;
        if (q == 0x400u) { q = 0; ++exp; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | q);
}

inline float F16BitsToF32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) { u = sign; }
        else {
            // 次正规：值 = man * 2^-24。直接用这个式子，不要去凑指数域 —— 上一
            // 版就是在这里错的。
            float f = (float)man * 1.0f / 16777216.0f;
            uint32_t fu; std::memcpy(&fu, &f, 4);
            u = sign | fu;
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; std::memcpy(&out, &u, 4);
    return out;
}

inline float F16(float v) { return F16BitsToF32(F32ToF16Bits(v)); }

// ---------------------------------------------------------------------------
// 确定性伪随机（和上一版同一套，换 shape 不换数）
// ---------------------------------------------------------------------------
inline uint64_t Mix64(uint64_t i, uint64_t salt)
{
    uint64_t z = i * 0x9E3779B97F4A7C15ull + salt;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// 返回 [-1, 1] 里一个 fp16 可精确表示的值
inline uint16_t RandF16Unit(uint64_t i, uint64_t salt)
{
    const int q = (int)(Mix64(i, salt) % 2001) - 1000;   // -1000..1000
    return F32ToF16Bits((float)q / 1000.0f);
}

// ---------------------------------------------------------------------------
// FRACTAL_Z（fp16，C0 = 16）
//
// 排布 [Cin1*KH*KW][ceil(Cout/16)][16][C0]，展平后就是 L0B 要的 NZ 块，所以
// kernel 的 GM->L1 是一次连续拷贝，没有格式转换。
//   下标 = ((cin/C0)*KH*KW + kh*KW + kw) * ceil(Cout/16)*16*C0
//          + (cout/16)*16*C0 + (cout%16)*C0 + (cin%C0)
// ---------------------------------------------------------------------------
inline void WeightToFractalZ(const uint16_t* nchw, int cin, int cout, std::vector<uint16_t>& dev)
{
    const int cin1 = cin / C0;
    const int cout1 = (cout + 15) / 16;
    dev.assign((size_t)cin1 * KH * KW * cout1 * 16 * C0, 0);
    for (int co = 0; co < cout; ++co) {
        for (int ci = 0; ci < cin; ++ci) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t src = (((size_t)co * cin + ci) * KH + kh) * KW + kw;
                    const size_t kf = ((size_t)(ci / C0) * KH + kh) * KW + kw;
                    const size_t dst = kf * cout1 * 16 * C0 + (size_t)(co / 16) * 16 * C0 + (size_t)(co % 16) * C0 +
                                       (size_t)(ci % C0);
                    dev[dst] = nchw[src];
                }
            }
        }
    }
}

inline void WeightFromFractalZ(const uint16_t* dev, int cin, int cout, std::vector<uint16_t>& nchw)
{
    const int cout1 = (cout + 15) / 16;
    nchw.assign((size_t)cout * cin * KH * KW, 0);
    for (int co = 0; co < cout; ++co) {
        for (int ci = 0; ci < cin; ++ci) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const size_t kf = ((size_t)(ci / C0) * KH + kh) * KW + kw;
                    const size_t src = kf * cout1 * 16 * C0 + (size_t)(co / 16) * 16 * C0 + (size_t)(co % 16) * C0 +
                                       (size_t)(ci % C0);
                    nchw[(((size_t)co * cin + ci) * KH + kh) * KW + kw] = dev[src];
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// 卷积。两条路：
//   ConvFwdExact  纯 fp32 累加 —— 「这个算子有没有在算这个卷积」的判据，不受
//                 定点模型的假设影响。
//   ConvFwdFixed  定点模型 —— 假设 acc_int32 = round(sum(a*b) * 2^S)。
//
// 两条都在 fp32 里做乘加（fp16 输入本来就能精确转 fp32），差别只在结果怎么落格。
// 输入输出都是 NCHW 的 fp16 位型。
// ---------------------------------------------------------------------------
struct ConvSpec {
    int cin, hi, wi, cout, ho, wo, stride;
};

// 返回每个输出点的精确 sum(a*b)（不含 bias），供上层落格用。
inline void ConvFwdRaw(const uint16_t* in, const uint16_t* wt, const ConvSpec& sp, std::vector<double>& acc)
{
    acc.assign((size_t)sp.cout * sp.ho * sp.wo, 0.0);
    for (int co = 0; co < sp.cout; ++co) {
        for (int oh = 0; oh < sp.ho; ++oh) {
            for (int ow = 0; ow < sp.wo; ++ow) {
                double sum = 0.0;
                for (int ci = 0; ci < sp.cin; ++ci) {
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sp.stride + kh - PAD;
                        if (ih < 0 || ih >= sp.hi) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sp.stride + kw - PAD;
                            if (iw < 0 || iw >= sp.wi) continue;
                            const float a = F16BitsToF32(in[((size_t)ci * sp.hi + ih) * sp.wi + iw]);
                            const float b = F16BitsToF32(wt[(((size_t)co * sp.cin + ci) * KH + kh) * KW + kw]);
                            sum += (double)a * (double)b;
                        }
                    }
                }
                acc[((size_t)co * sp.ho + oh) * sp.wo + ow] = sum;
            }
        }
    }
}

// 给定这一层的原始累加值，报告 int32 不溢出所允许的最大 S。
// 这就是标定 fixed_shift1 / fixed_shift2 该用的数：取它，或比它小一两位留余量。
inline int MaxSafeShift(const std::vector<double>& acc, const uint16_t* bias, int cout, int hw, double* peak)
{
    double m = 0.0;
    for (int co = 0; co < cout; ++co) {
        const double b = (double)F16BitsToF32(bias[co]);
        for (int i = 0; i < hw; ++i) {
            const double v = std::fabs(acc[(size_t)co * hw + i] + b);
            if (v > m) m = v;
        }
    }
    *peak = m;
    if (m <= 0.0) return FIX_SHIFT_LEN_A16W16;
    int S = 0;
    while (S < FIX_SHIFT_LEN_A16W16 && m * std::ldexp(1.0, S + 1) < 2147483000.0) ++S;
    return S;
}

// 定点落格 + bias + relu -> fp16 位型。
//   acc_i32 = round(sum * 2^S) + round(bias * 2^S)
//   out     = fp16(acc_i32 * 2^-S)，再 relu
// bias 的那一次 round 对应 kernel 里 L1->BT 带 fixShiftVal 的那条搬运。
inline void FixedEpilogue(const std::vector<double>& acc, const uint16_t* bias, int cout, int hw, int S, bool relu,
                          std::vector<uint16_t>& out, long* satCount)
{
    out.assign((size_t)cout * hw, 0);
    const double scale = std::ldexp(1.0, S);
    for (int co = 0; co < cout; ++co) {
        const double bq = std::nearbyint((double)F16BitsToF32(bias[co]) * scale);
        for (int i = 0; i < hw; ++i) {
            double q = std::nearbyint(acc[(size_t)co * hw + i] * scale) + bq;
            if (q > 2147483647.0) { q = 2147483647.0; ++*satCount; }
            if (q < -2147483648.0) { q = -2147483648.0; ++*satCount; }
            float v = (float)(q / scale);
            if (relu && !(v > 0.0f)) {
                // fixpipe 的 relu 是保号钳位：负数出来是 -0（0x8000），不是 +0。
                // 这一条是上一版在板上实测出来的，不是从定义推的 —— 383,242 个
                // 「不一致」全是零的符号。数值上 -0 == +0，但逐位比对必须照做。
                out[(size_t)co * hw + i] = (uint16_t)(F32ToF16Bits(v) & 0x8000u);
            } else {
                out[(size_t)co * hw + i] = F32ToF16Bits(v);
            }
        }
    }
}

// 纯 fp32 参考：不落定点格，只在每层末尾窄化到 fp16（硬件的输出确实是 fp16）。
inline void ExactEpilogue(const std::vector<double>& acc, const uint16_t* bias, int cout, int hw, bool relu,
                          std::vector<uint16_t>& out)
{
    out.assign((size_t)cout * hw, 0);
    for (int co = 0; co < cout; ++co) {
        const double b = (double)F16BitsToF32(bias[co]);
        for (int i = 0; i < hw; ++i) {
            float v = (float)(acc[(size_t)co * hw + i] + b);
            if (relu && !(v > 0.0f)) v = 0.0f;
            out[(size_t)co * hw + i] = F32ToF16Bits(v);
        }
    }
}

// ---------------------------------------------------------------------------
// 输入与 golden
// ---------------------------------------------------------------------------
struct Inputs {
    std::vector<uint16_t> xNchw;   // [1, CI, HI, WI]
    std::vector<uint16_t> w1Nchw;  // [COUT1, CI, 3, 3]
    std::vector<uint16_t> w2Nchw;  // [COUT2, COUT1, 3, 3]
    std::vector<uint16_t> b1;      // [COUT1]
    std::vector<uint16_t> b2;      // [COUT2]
    std::vector<uint16_t> w1Dev;   // FRACTAL_Z
    std::vector<uint16_t> w2Dev;
};

struct Golden {
    std::vector<uint16_t> midFixed, yFixed;   // 定点模型
    std::vector<uint16_t> midExact, yExact;   // 纯 fp32 参考
    int shift1 = 0, shift2 = 0;
    double peak1 = 0, peak2 = 0;
    long sat = 0;
    long yNonZero = 0;
};

inline Inputs GenerateInputs()
{
    Inputs in;
    in.xNchw.resize((size_t)CI * HI * WI);
    for (size_t i = 0; i < in.xNchw.size(); ++i) in.xNchw[i] = RandF16Unit(i, 0x11);
    in.w1Nchw.resize((size_t)COUT1 * CI * KH * KW);
    for (size_t i = 0; i < in.w1Nchw.size(); ++i) in.w1Nchw[i] = RandF16Unit(i, 0x22);
    in.w2Nchw.resize((size_t)COUT2 * COUT1 * KH * KW);
    for (size_t i = 0; i < in.w2Nchw.size(); ++i) in.w2Nchw[i] = RandF16Unit(i, 0x33);
    in.b1.resize(COUT1);
    for (int i = 0; i < COUT1; ++i) in.b1[i] = RandF16Unit((uint64_t)i, 0x44);
    in.b2.resize(COUT2);
    for (int i = 0; i < COUT2; ++i) in.b2[i] = RandF16Unit((uint64_t)i, 0x55);
    WeightToFractalZ(in.w1Nchw.data(), CI, COUT1, in.w1Dev);
    WeightToFractalZ(in.w2Nchw.data(), COUT1, COUT2, in.w2Dev);
    return in;
}

// shiftFor* 传 -1 表示「按数据自动挑一个安全的 S」，否则用给定值。
inline Golden BuildGolden(const Inputs& in, int shiftFor1 = -1, int shiftFor2 = -1)
{
    Golden g;
    const ConvSpec sp1{CI, HI, WI, COUT1, HO1, WO1, STRIDE1};
    std::vector<double> acc1;
    ConvFwdRaw(in.xNchw.data(), in.w1Nchw.data(), sp1, acc1);
    const int hw1 = HO1 * WO1;
    const int auto1 = MaxSafeShift(acc1, in.b1.data(), COUT1, hw1, &g.peak1);
    g.shift1 = shiftFor1 >= 0 ? shiftFor1 : auto1;
    FixedEpilogue(acc1, in.b1.data(), COUT1, hw1, g.shift1, true, g.midFixed, &g.sat);
    ExactEpilogue(acc1, in.b1.data(), COUT1, hw1, true, g.midExact);

    const ConvSpec sp2{COUT1, HO1, WO1, COUT2, HO2, WO2, STRIDE2};
    std::vector<double> acc2;
    // conv2 吃的是 conv1 的**定点**结果 —— 板上就是这条链，不能拿 exact 的中间值。
    ConvFwdRaw(g.midFixed.data(), in.w2Nchw.data(), sp2, acc2);
    const int hw2 = HO2 * WO2;
    const int auto2 = MaxSafeShift(acc2, in.b2.data(), COUT2, hw2, &g.peak2);
    g.shift2 = shiftFor2 >= 0 ? shiftFor2 : auto2;
    FixedEpilogue(acc2, in.b2.data(), COUT2, hw2, g.shift2, true, g.yFixed, &g.sat);

    std::vector<double> acc2e;
    ConvFwdRaw(g.midExact.data(), in.w2Nchw.data(), sp2, acc2e);
    ExactEpilogue(acc2e, in.b2.data(), COUT2, hw2, true, g.yExact);

    for (size_t i = 0; i < g.yFixed.size(); ++i) {
        // -0 也算「写过但为零」，所以掩掉符号位再判非零。
        g.yNonZero += ((g.yFixed[i] & 0x7FFFu) != 0);
    }
    return g;
}

} // namespace fc2d_golden
#endif
