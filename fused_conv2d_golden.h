/*
 * fused_conv2d 的上板 golden —— **按形状参数化**，两条通路各一套。
 *
 * ===========================================================================
 * 这个文件和 ops-nn 仓里那份同名文件的分工
 * ===========================================================================
 * ops-nn 的 tests/ut/op_kernel/fused_conv2d_golden.h 是 CPU 仿真用的：int8 那条
 * 链能在仿真里整条跑完并逐位比对，fp16 那条在仿真里根本没有 mmad 模型。
 *
 * 这一份是**上板**用的，所以两条通路都要有数值 golden：
 *
 *   fp16 定点   y_expect  定点模型：acc_i32 = round(sum(a*b) * 2^F)，F = 58 - S
 *               y_exact   纯 fp32 参考，只在每层末尾窄化到 fp16
 *   int8 量化   y         精确整数卷积 + REQ8（round-half-even + 饱和 + relu）
 *
 * fp16 给两个 golden 是因为定点模型带假设（累加器的定标语义），而 y_exact 不带。
 * 板上先看 y_exact 的相对误差过不过 —— 那判的是「这个算子有没有在算这个卷积」；
 * 再看 y_expect 能不能逐位对上 —— 那判的是「定点模型对不对」。只有前者过、后者
 * 不过，说明卷积算对了但定点模型猜错了，是两件不同的事。
 *
 * int8 只有一个 golden：那条链上的运算全是精确整数，没有需要第二个参照的假设。
 */
#ifndef FUSED_CONV2D_GOLDEN_H
#define FUSED_CONV2D_GOLDEN_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fc2d_golden {

constexpr int FIX_SHIFT_LEN_A16W16 = 58;

// 厂商的默认工作点。CANN 自己的 matmul 在 5102 上就用这个值：
//   matmul/common/cmct/block/block_mmad_pingpong_without_que.h:144
//       #if __NPU_ARCH__ == 5102
//           uint8_t shiftValue_{42};
//   58 - 42 = 16
// 权重 int8 的那条路（weight_quant_batch_matmul_v2_tiling.cpp:33）默认 13，
// FIX_SHIFT_LEN_A16W8 = 29，29 - 13 也是 16。旧版 API 里 DEQF16 的系数干脆写死成
// 0x37800000 = 2^-16，注释就是 "fix point 1/2^16"。三条独立证据都落在 F = 16。
constexpr int DEFAULT_ATTR_SHIFT = 42;

// ---------------------------------------------------------------------------
// 一个测试形状。和算子的 Params 一一对应。
// ---------------------------------------------------------------------------
struct Case {
    int n = 1, ci = 32, hi = 288, wi = 112;
    int cout1 = 64, cout2 = 96;
    int kh = 3, kw = 3;
    int stride1 = 1, stride2 = 2;
    int padH1 = 1, padW1 = 1, padH2 = 1, padW2 = 1;
    bool bias = false;
    bool relu1 = true, relu2 = true;
    int elemBytes = 2; // 2 = fp16 定点，1 = int8 量化
    int shift1 = DEFAULT_ATTR_SHIFT, shift2 = DEFAULT_ATTR_SHIFT;

    int C0() const { return 32 / elemBytes; }
    int Ho1() const { return (hi + 2 * padH1 - kh) / stride1 + 1; }
    int Wo1() const { return (wi + 2 * padW1 - kw) / stride1 + 1; }
    int Ho2() const { return (Ho1() + 2 * padH2 - kh) / stride2 + 1; }
    int Wo2() const { return (Wo1() + 2 * padW2 - kw) / stride2 + 1; }
    long XElems() const { return (long)n * ci * hi * wi; }
    long YElems() const { return (long)n * cout2 * Ho2() * Wo2(); }
    long W1Elems() const { return (long)cout1 * ci * kh * kw; }
    long W2Elems() const { return (long)cout2 * cout1 * kh * kw; }
    std::string Text() const
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "n%d %d->%d->%d %dx%d k%dx%d s%d/%d p%d,%d/%d,%d %s%s%s -> %dx%d %s", n,
                 ci, cout1, cout2, hi, wi, kh, kw, stride1, stride2, padH1, padW1, padH2, padW2,
                 bias ? "bias " : "", relu1 ? "relu1 " : "", relu2 ? "relu2" : "", Ho2(), Wo2(),
                 elemBytes == 2 ? "fp16" : "int8");
        return std::string(buf);
    }
};

// ---------------------------------------------------------------------------
// fp16 的位型转换。不用编译器的 _Float16：同一份 golden 要在开发机和板子上给出
// **逐位相同**的结果，自己实现才能保证舍入模式一致。
// ---------------------------------------------------------------------------
inline uint16_t F32ToF16Bits(float v)
{
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int32_t exp = (int32_t)((b >> 23) & 0xFF);
    uint32_t man = b & 0x7FFFFFu;
    if (exp == 0xFF) {
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u)); // inf / nan
    }
    exp = exp - 127 + 15;
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u); // 溢出到 inf
    }
    if (exp <= 0) {
        // 次正规：把隐含的 1 补回去再右移
        if (exp < -10) {
            return (uint16_t)sign;
        }
        man |= 0x800000u;
        const int shift = 14 - exp; // exp<=0 时 shift >= 14
        const uint32_t keep = man >> shift;
        const uint32_t rest = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t out = keep;
        if (rest > half || (rest == half && (keep & 1u))) {
            ++out;
        }
        return (uint16_t)(sign | out);
    }
    // round-half-to-even
    const uint32_t keep = man >> 13;
    const uint32_t rest = man & 0x1FFFu;
    uint32_t out = keep;
    if (rest > 0x1000u || (rest == 0x1000u && (keep & 1u))) {
        ++out;
        if (out == 0x400u) {
            out = 0;
            ++exp;
            if (exp >= 31) {
                return (uint16_t)(sign | 0x7C00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | out);
}

inline float F16BitsToF32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t b;
    if (exp == 0) {
        if (man == 0) {
            b = sign;
        } else {
            // 次正规：规格化
            int e = -1;
            do {
                ++e;
                man <<= 1;
            } while ((man & 0x400u) == 0);
            man &= 0x3FFu;
            b = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        b = sign | 0x7F800000u | (man << 13);
    } else {
        b = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline float F16(float v) { return F16BitsToF32(F32ToF16Bits(v)); }

// ---------------------------------------------------------------------------
// 确定性随机。不用 <random>：同一个 seed 在不同 libstdc++ 上可能给不同的序列，
// 而这份 golden 要在开发机和板子上给出**同一批数据**。
// ---------------------------------------------------------------------------
inline uint64_t Mix64(uint64_t i, uint64_t salt)
{
    uint64_t z = i + salt + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// [-1, 1) 上的 1/256 网格：fp16 能精确表示，定点累加也不容易溢出。
inline uint16_t RandF16Unit(uint64_t i, uint64_t salt)
{
    const int q = (int)(Mix64(i, salt) % 512u) - 256;
    return F32ToF16Bits((float)q / 256.0f);
}

// ---------------------------------------------------------------------------
// FRACTAL_Z 打包。[Cin/C0 * kh * kw][ceil(Cout/16)][16][C0]，摊平之后就是 kernel
// 要的 L0B NZ 块 —— GM->L1 是一条连续拷贝，中间没有任何格式转换。
// C0 由元素宽度定（fp16 16 / int8 32），所以以 C0 为参数。
// ---------------------------------------------------------------------------
template <typename T>
inline void WeightToFractalZ(const T* nchw, int cin, int cout, int kh, int kw, int c0, std::vector<T>& dev)
{
    const int cin1 = cin / c0;
    const int coutBlk = (cout + 15) / 16;
    dev.assign((size_t)cin1 * kh * kw * coutBlk * 16 * c0, (T)0);
    for (int co = 0; co < cout; ++co) {
        for (int ci = 0; ci < cin; ++ci) {
            for (int i = 0; i < kh; ++i) {
                for (int j = 0; j < kw; ++j) {
                    const size_t src = (((size_t)co * cin + ci) * kh + i) * kw + j;
                    const size_t kf = ((size_t)(ci / c0) * kh + i) * kw + j;
                    const size_t dst = kf * (size_t)coutBlk * 16 * c0 + (size_t)(co / 16) * 16 * c0 +
                                       (size_t)(co % 16) * c0 + (size_t)(ci % c0);
                    dev[dst] = nchw[src];
                }
            }
        }
    }
}

// ===========================================================================
// fp16 定点通路
// ===========================================================================
namespace f16path {

// 一层卷积的精确 sum(a*b)（不含 bias），在 double 里累加。
// absAcc 非空时同时返回 sum|a*b| —— 累加器里跑的是**部分和**，最终值不溢出不代表
// 中途不溢出。sum|a*b| 是任何 mmad 次序下部分和的保守上界，定标校验按它做。
inline void ConvRaw(const uint16_t* in, const uint16_t* wt, int cin, int hi, int wi, int cout, int ho, int wo,
                    int kh, int kw, int stride, int padH, int padW, std::vector<double>& acc,
                    std::vector<double>* absAcc)
{
    acc.assign((size_t)cout * ho * wo, 0.0);
    if (absAcc != nullptr) {
        absAcc->assign((size_t)cout * ho * wo, 0.0);
    }
    for (int co = 0; co < cout; ++co) {
        for (int oh = 0; oh < ho; ++oh) {
            for (int ow = 0; ow < wo; ++ow) {
                double sum = 0.0, asum = 0.0;
                for (int ci = 0; ci < cin; ++ci) {
                    for (int i = 0; i < kh; ++i) {
                        const int ih = oh * stride + i - padH;
                        if (ih < 0 || ih >= hi) {
                            continue;
                        }
                        for (int j = 0; j < kw; ++j) {
                            const int iw = ow * stride + j - padW;
                            if (iw < 0 || iw >= wi) {
                                continue;
                            }
                            const double a = (double)F16BitsToF32(in[((size_t)ci * hi + ih) * wi + iw]);
                            const double b =
                                (double)F16BitsToF32(wt[(((size_t)co * cin + ci) * kh + i) * kw + j]);
                            sum += a * b;
                            asum += std::fabs(a * b);
                        }
                    }
                }
                acc[((size_t)co * ho + oh) * wo + ow] = sum;
                if (absAcc != nullptr) {
                    (*absAcc)[((size_t)co * ho + oh) * wo + ow] = asum;
                }
            }
        }
    }
}

// int32 不溢出所允许的最大**反缩放指数 F**。约束取在部分和上，不是最终值上：
// L0C 一路累加，中途能到 sum|a*b| + |bias|，而 L0C 溢出后是回绕还是饱和无从确证，
// 所以直接按上界挑，把这个问题消掉。
//
// 注意 F 不是算子属性：属性是 S = 58 - F。要属性值请过 AttrFromDeqExp()。
inline int MaxSafeDeqExp(const std::vector<double>& acc, const std::vector<double>& absAcc,
                         const uint16_t* bias, int cout, int hw, double* peak, double* peakPartial)
{
    double m = 0.0, mp = 0.0;
    for (int co = 0; co < cout; ++co) {
        const double b = bias == nullptr ? 0.0 : (double)F16BitsToF32(bias[co]);
        const double ab = std::fabs(b);
        for (int i = 0; i < hw; ++i) {
            const double v = std::fabs(acc[(size_t)co * hw + i] + b);
            if (v > m) m = v;
            const double p = absAcc[(size_t)co * hw + i] + ab;
            if (p > mp) mp = p;
        }
    }
    *peak = m;
    *peakPartial = mp;
    if (mp <= 0.0) {
        return FIX_SHIFT_LEN_A16W16;
    }
    int F = 0;
    while (F < FIX_SHIFT_LEN_A16W16 && mp * std::ldexp(1.0, F + 1) < 2147483000.0) {
        ++F;
    }
    return F;
}

inline int AttrFromDeqExp(int deqExp)
{
    const int S = FIX_SHIFT_LEN_A16W16 - deqExp;
    return S < 0 ? 0 : (S > FIX_SHIFT_LEN_A16W16 ? FIX_SHIFT_LEN_A16W16 : S);
}

// 定点落格 + bias + relu -> fp16 位型。
// attrShift 是**算子属性 S**；实际用的反缩放指数是 F = 58 - S。
//   acc_i32 = round(sum * 2^F) + round(bias * 2^F)
//   out     = fp16(acc_i32 * 2^-F)，再 relu
// bias 的那一次 round 对应 kernel 里 L1->BT 带 fixShiftVal = 58 - S 的那条搬运。
inline void FixedEpilogue(const std::vector<double>& acc, const uint16_t* bias, int cout, int hw,
                          int attrShift, bool relu, std::vector<uint16_t>& out, long* satCount)
{
    out.assign((size_t)cout * hw, 0);
    const int deqExp = FIX_SHIFT_LEN_A16W16 - attrShift;
    const double scale = std::ldexp(1.0, deqExp);
    for (int co = 0; co < cout; ++co) {
        const double bq = bias == nullptr ? 0.0 : std::nearbyint((double)F16BitsToF32(bias[co]) * scale);
        for (int i = 0; i < hw; ++i) {
            double q = std::nearbyint(acc[(size_t)co * hw + i] * scale) + bq;
            if (q > 2147483647.0) {
                q = 2147483647.0;
                ++*satCount;
            }
            if (q < -2147483648.0) {
                q = -2147483648.0;
                ++*satCount;
            }
            const float v = (float)(q / scale);
            if (relu && !(v > 0.0f)) {
                // fixpipe 的 relu 是保号钳位：负数出来是 **-0**（0x8000），不是 +0。
                // 这条是上板实测出来的，不是从定义推的 —— 当时 383,242 个「不一致」
                // 全是零的符号。数值上 -0 == +0，但逐位比对必须照做。
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
        const double b = bias == nullptr ? 0.0 : (double)F16BitsToF32(bias[co]);
        for (int i = 0; i < hw; ++i) {
            float v = (float)(acc[(size_t)co * hw + i] + b);
            if (relu && !(v > 0.0f)) {
                out[(size_t)co * hw + i] = (uint16_t)(F32ToF16Bits(v) & 0x8000u);
            } else {
                out[(size_t)co * hw + i] = F32ToF16Bits(v);
            }
        }
    }
}

struct Result {
    std::vector<uint16_t> xNchw, w1Nchw, w2Nchw, w1Dev, w2Dev, b1, b2;
    std::vector<uint16_t> yExpect, yExact;
    long yNonZero = 0;
    long sat = 0;
    int safeF1 = 0, safeF2 = 0; // 部分和不溢出所允许的最大 F，用来判 shift 有没有余量
    double peak1 = 0, peak2 = 0, peakPartial1 = 0, peakPartial2 = 0;
};

inline Result Build(const Case& c)
{
    Result r;
    const int ho1 = c.Ho1(), wo1 = c.Wo1(), ho2 = c.Ho2(), wo2 = c.Wo2();
    r.xNchw.resize((size_t)c.XElems());
    for (size_t i = 0; i < r.xNchw.size(); ++i) {
        r.xNchw[i] = RandF16Unit(i, 0x41);
    }
    r.w1Nchw.resize((size_t)c.W1Elems());
    for (size_t i = 0; i < r.w1Nchw.size(); ++i) {
        r.w1Nchw[i] = RandF16Unit(i, 0x42);
    }
    r.w2Nchw.resize((size_t)c.W2Elems());
    for (size_t i = 0; i < r.w2Nchw.size(); ++i) {
        r.w2Nchw[i] = RandF16Unit(i, 0x43);
    }
    r.b1.assign(c.cout1, F32ToF16Bits(0.0f));
    r.b2.assign(c.cout2, F32ToF16Bits(0.0f));
    if (c.bias) {
        for (int i = 0; i < c.cout1; ++i) {
            r.b1[i] = F32ToF16Bits((float)(i - c.cout1 / 2) / 64.0f);
        }
        for (int i = 0; i < c.cout2; ++i) {
            r.b2[i] = F32ToF16Bits((float)(i - c.cout2 / 2) / 64.0f);
        }
    }
    WeightToFractalZ<uint16_t>(r.w1Nchw.data(), c.ci, c.cout1, c.kh, c.kw, 16, r.w1Dev);
    WeightToFractalZ<uint16_t>(r.w2Nchw.data(), c.cout1, c.cout2, c.kh, c.kw, 16, r.w2Dev);

    r.yExpect.assign((size_t)c.YElems(), 0);
    r.yExact.assign((size_t)c.YElems(), 0);
    // 不带 bias 时 b1/b2 已经全是 0，FixedEpilogue 的 bq 也就是 0 —— 数值上和
    // 「不加 bias」完全一样，所以下面不用再分一支。
    for (int b = 0; b < c.n; ++b) {
        std::vector<double> acc1, abs1;
        ConvRaw(r.xNchw.data() + (size_t)b * c.ci * c.hi * c.wi, r.w1Nchw.data(), c.ci, c.hi, c.wi, c.cout1,
                ho1, wo1, c.kh, c.kw, c.stride1, c.padH1, c.padW1, acc1, &abs1);
        double p, pp;
        const int f1 = MaxSafeDeqExp(acc1, abs1, r.b1.data(), c.cout1, ho1 * wo1, &p, &pp);
        if (b == 0 || f1 < r.safeF1) {
            r.safeF1 = f1;
        }
        if (p > r.peak1) r.peak1 = p;
        if (pp > r.peakPartial1) r.peakPartial1 = pp;

        // conv2 吃的是 conv1 的 **fp16** 输出 —— 和板上这条链一致（mid 是 fp16）。
        std::vector<uint16_t> mid;
        FixedEpilogue(acc1, r.b1.data(), c.cout1, ho1 * wo1, c.shift1, c.relu1, mid, &r.sat);
        std::vector<double> acc2, abs2;
        ConvRaw(mid.data(), r.w2Nchw.data(), c.cout1, ho1, wo1, c.cout2, ho2, wo2, c.kh, c.kw, c.stride2,
                c.padH2, c.padW2, acc2, &abs2);
        const int f2 = MaxSafeDeqExp(acc2, abs2, r.b2.data(), c.cout2, ho2 * wo2, &p, &pp);
        if (b == 0 || f2 < r.safeF2) {
            r.safeF2 = f2;
        }
        if (p > r.peak2) r.peak2 = p;
        if (pp > r.peakPartial2) r.peakPartial2 = pp;

        std::vector<uint16_t> ye, yx;
        FixedEpilogue(acc2, r.b2.data(), c.cout2, ho2 * wo2, c.shift2, c.relu2, ye, &r.sat);
        ExactEpilogue(acc2, r.b2.data(), c.cout2, ho2 * wo2, c.relu2, yx);
        const size_t base = (size_t)b * c.cout2 * ho2 * wo2;
        for (size_t i = 0; i < ye.size(); ++i) {
            r.yExpect[base + i] = ye[i];
            r.yExact[base + i] = yx[i];
            if ((ye[i] & 0x7FFFu) != 0) {
                ++r.yNonZero;
            }
        }
    }
    return r;
}

} // namespace f16path

// ===========================================================================
// int8 量化通路
// ===========================================================================
namespace int8path {

// scale 的低 13 位尾数会被硬件丢掉（REQ8 的 deqScalar 只有 [31:13] 这 19 位），
// golden 必须照做，否则会出现「只差最后一位」的不一致，而那种不一致最难判定谁错。
inline float EffectiveScale(float s)
{
    uint32_t b;
    std::memcpy(&b, &s, 4);
    b &= 0xFFFFE000u;
    float out;
    std::memcpy(&out, &b, 4);
    return out;
}

inline int8_t Req8(int32_t acc, float scale, bool relu)
{
    const double v = (double)acc * (double)scale;
    double r = std::nearbyint(v); // 默认舍入模式就是 round-half-to-even
    if (r > 127.0) r = 127.0;
    if (r < -128.0) r = -128.0;
    int8_t q = (int8_t)r;
    if (relu && q < 0) {
        q = 0;
    }
    return q;
}

// 精确的整数卷积。int8 x int8 累到 int32：|a*b| <= 127*128 = 16,256，K 就算到
// 4,608 也才 7.5e7，离 int32 的 2.1e9 远得很。
inline void ConvInt8(const int8_t* in, const int8_t* wt, const int32_t* bias, int cin, int hi, int wi,
                     int cout, int ho, int wo, int kh, int kw, int stride, int padH, int padW,
                     std::vector<int32_t>& acc)
{
    acc.assign((size_t)cout * ho * wo, 0);
    for (int co = 0; co < cout; ++co) {
        for (int oh = 0; oh < ho; ++oh) {
            for (int ow = 0; ow < wo; ++ow) {
                int32_t sum = (bias == nullptr) ? 0 : bias[co];
                for (int ci = 0; ci < cin; ++ci) {
                    for (int i = 0; i < kh; ++i) {
                        const int ih = oh * stride + i - padH;
                        if (ih < 0 || ih >= hi) {
                            continue;
                        }
                        for (int j = 0; j < kw; ++j) {
                            const int iw = ow * stride + j - padW;
                            if (iw < 0 || iw >= wi) {
                                continue;
                            }
                            sum += (int32_t)in[((size_t)ci * hi + ih) * wi + iw] *
                                   (int32_t)wt[(((size_t)co * cin + ci) * kh + i) * kw + j];
                        }
                    }
                }
                acc[((size_t)co * ho + oh) * wo + ow] = sum;
            }
        }
    }
}

// scale 挑法：让重量化之后的动态范围铺满 int8。**不能不挑** —— 累加器量级在
// 1e6~1e7，scale = 1.0 会让每个点都饱和到 ±127，那样比对全 127 = 全对，什么也
// 验不出来。板上漏传 quant_scale 的表征就是输出大面积贴在 ±127 上。
inline float PickScale(const std::vector<int32_t>& acc)
{
    int32_t m = 0;
    for (size_t i = 0; i < acc.size(); ++i) {
        const int32_t a = acc[i] < 0 ? -acc[i] : acc[i];
        if (a > m) {
            m = a;
        }
    }
    if (m == 0) {
        return EffectiveScale(1.0f);
    }
    return EffectiveScale(127.0f / (float)m);
}

struct Result {
    std::vector<int8_t> xNchw, w1Nchw, w2Nchw, w1Dev, w2Dev, y;
    std::vector<int32_t> b1, b2;
    float scale1 = 0, scale2 = 0;
    long yNonZero = 0;
    long sat = 0;
};

inline Result Build(const Case& c)
{
    Result r;
    const int ho1 = c.Ho1(), wo1 = c.Wo1(), ho2 = c.Ho2(), wo2 = c.Wo2();
    auto fill = [](std::vector<int8_t>& v, uint64_t salt) {
        for (size_t i = 0; i < v.size(); ++i) {
            // [-100, 100]：给累加留余量，也保证有足够多的负数去验 REQ8 的 bit 46
            // （有符号饱和）—— 忘了置那一位的话负数会被静默清零。
            v[i] = (int8_t)((int)(Mix64(i, salt) % 201u) - 100);
        }
    };
    r.xNchw.resize((size_t)c.XElems());
    fill(r.xNchw, 0x11);
    r.w1Nchw.resize((size_t)c.W1Elems());
    fill(r.w1Nchw, 0x22);
    r.w2Nchw.resize((size_t)c.W2Elems());
    fill(r.w2Nchw, 0x33);
    r.b1.assign(c.cout1, 0);
    r.b2.assign(c.cout2, 0);
    if (c.bias) {
        for (int i = 0; i < c.cout1; ++i) {
            r.b1[i] = (i - c.cout1 / 2) * 137;
        }
        for (int i = 0; i < c.cout2; ++i) {
            r.b2[i] = (i - c.cout2 / 2) * 91;
        }
    }
    WeightToFractalZ<int8_t>(r.w1Nchw.data(), c.ci, c.cout1, c.kh, c.kw, 32, r.w1Dev);
    WeightToFractalZ<int8_t>(r.w2Nchw.data(), c.cout1, c.cout2, c.kh, c.kw, 32, r.w2Dev);

    const int32_t* pb1 = c.bias ? r.b1.data() : nullptr;
    const int32_t* pb2 = c.bias ? r.b2.data() : nullptr;

    // scale 是 per-tensor 的，整个 batch 共用，所以先把所有图的累加器算出来。
    std::vector<std::vector<int32_t>> acc1(c.n), acc2(c.n);
    for (int b = 0; b < c.n; ++b) {
        ConvInt8(r.xNchw.data() + (size_t)b * c.ci * c.hi * c.wi, r.w1Nchw.data(), pb1, c.ci, c.hi, c.wi,
                 c.cout1, ho1, wo1, c.kh, c.kw, c.stride1, c.padH1, c.padW1, acc1[b]);
    }
    {
        std::vector<int32_t> all;
        for (int b = 0; b < c.n; ++b) {
            all.insert(all.end(), acc1[b].begin(), acc1[b].end());
        }
        r.scale1 = PickScale(all);
    }
    for (int b = 0; b < c.n; ++b) {
        std::vector<int8_t> mid(acc1[b].size());
        for (size_t i = 0; i < mid.size(); ++i) {
            mid[i] = Req8(acc1[b][i], r.scale1, c.relu1);
        }
        ConvInt8(mid.data(), r.w2Nchw.data(), pb2, c.cout1, ho1, wo1, c.cout2, ho2, wo2, c.kh, c.kw,
                 c.stride2, c.padH2, c.padW2, acc2[b]);
    }
    {
        std::vector<int32_t> all;
        for (int b = 0; b < c.n; ++b) {
            all.insert(all.end(), acc2[b].begin(), acc2[b].end());
        }
        r.scale2 = PickScale(all);
    }
    r.y.assign((size_t)c.YElems(), 0);
    for (int b = 0; b < c.n; ++b) {
        const size_t base = (size_t)b * c.cout2 * ho2 * wo2;
        for (size_t i = 0; i < acc2[b].size(); ++i) {
            const int8_t q = Req8(acc2[b][i], r.scale2, c.relu2);
            r.y[base + i] = q;
            if (q != 0) {
                ++r.yNonZero;
            }
            if (q == 127 || q == -128) {
                ++r.sat;
            }
        }
    }
    return r;
}

} // namespace int8path

} // namespace fc2d_golden

#endif // FUSED_CONV2D_GOLDEN_H
