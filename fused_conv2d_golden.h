/*
 * fused_conv2d 的 CPU 参考实现 —— 全 fp16 定点版。
 *
 *   x fp16 NCHW ─conv1 3x3 s1 p1 +bias1(fp16) +relu─► mid fp16
 *               ─conv2 3x3 s2 p1 +bias2(fp16) +relu─► y fp16 NCHW
 *
 * 全程定点（f162s32），没有量化、没有向量。权重 FRACTAL_Z fp16（C0 = 16）。
 *
 * ===========================================================================
 * 定标语义 —— 已从 CANN 源码确证，不再是假设。
 * ===========================================================================
 *
 *   * cube 走 f162s32：half x half -> int32，带 fixShiftVal 操作数
 *     (dav_m510/kernel_operator_mm_impl.h:355)
 *   * fixedShiftValue 是算子属性，范围 [0, 58]
 *     (conv2d_v2_base_tiling_check_attrs.cpp:449 CheckFixedShiftValueLegal)
 *   * 三个消费者的方向：mmad 用原值 S，L1->BT 和 fixpipe 用 58 - S
 *     (conv2d_small_kernel.h:354 / :937 / :1111)
 *   * **出口的反缩放系数就是 2^-(58-S)**。DEQF16 模式下 fixpipe 根本不看
 *     deqScalar，只按 shift 现搭一个 float 出来：
 *       (dav_m510/kernel_operator_fixpipe_impl.h:77 SetDeqScalarDepOnMode)
 *         uint64_t newExponent = (127 - shiftVal) & 0xFF;
 *         uint64_t newScalar   = (floatOne & ~mask) | (newExponent << shift);
 *     指数为 127-F 的 float 就是 2^-F，而 F 正是传给 fixpipe 的那个数 58 - S。
 *
 * 所以定点模型是（记 F = 58 - S）：
 *       acc_int32 = round( sum(a*b) * 2^F ) + round( bias * 2^F )
 *       out_fp16  = fp16( acc_int32 * 2^-F )
 *
 * **S 越大，累加器的定标越小。** 上一版把这个方向写反了（拿 S 直接当 2 的指
 * 数），于是 S=26 在板上实际跑成 2^32：几乎每个非零点都溢出 int32 并回绕，符
 * 号退化成掷硬币。板上的表征很干脆 —— 非零元素一个都没对上，对上的 216604 个
 * **全部**是两边都为零的点。那不是缩放错了，是回绕了。
 *
 * 还没实测的只剩一件事：那次 round 发生在哪一级 —— 每个乘积各 round 一次、每
 * 条 mmad 指令一次、还是整条 K 累完再 round。下面按「累完再 round」建模（误差
 * 最小的那种）。三者的差别至多是累加器的几十个 LSB，对靠近峰值的输出远在 fp16
 * 的 1 个 ULP 之下，只有接近零的输出才可能差最后几位。所以比对脚本会额外报一
 * 行「以累加器 LSB 计的偏差」：偏差全在个位数 LSB 以内，就说明只是 round 的级
 * 别猜得不同，卷积本身是对的。
 *
 * 两个 golden 都仍然给：
 *   GoldenExact()  纯 fp32 参考 —— 「这个算子到底有没有在算这个卷积」，用相对
 *                  误差判，完全不依赖定点模型；
 *   GoldenFixed(S) 定点模型 —— 上面那条链成立时应当逐位相等。
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

// 元素数。放在这里而不是让每个消费者自己乘，是因为它们要和 kernel 的几何一致，
// 只该有一个出处。
constexpr int X_ELEMS = CI * HI * WI;          // 1,032,192
constexpr int Y_ELEMS = COUT2 * HO2 * WO2;     //   774,144
constexpr int MID_ELEMS = COUT1 * HO1 * WO1;   // 2,064,384

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
//   ConvFwdFixed  定点模型 —— acc_int32 = round(sum(a*b) * 2^F)，F = 58 - S。
//
// 两条都在 fp32 里做乘加（fp16 输入本来就能精确转 fp32），差别只在结果怎么落格。
// 输入输出都是 NCHW 的 fp16 位型。
// ---------------------------------------------------------------------------
struct ConvSpec {
    int cin, hi, wi, cout, ho, wo, stride;
};

// 返回每个输出点的精确 sum(a*b)（不含 bias），供上层落格用。
// absAcc 非空时同时返回 sum|a*b| —— 累加器里跑的是**部分和**，最终值不溢出不代表
// 中途不溢出。sum|a*b| 是任何 mmad 次序下部分和的保守上界，定标就按它挑。
inline void ConvFwdRaw(const uint16_t* in, const uint16_t* wt, const ConvSpec& sp, std::vector<double>& acc,
                       std::vector<double>* absAcc = nullptr)
{
    acc.assign((size_t)sp.cout * sp.ho * sp.wo, 0.0);
    if (absAcc != nullptr) absAcc->assign((size_t)sp.cout * sp.ho * sp.wo, 0.0);
    for (int co = 0; co < sp.cout; ++co) {
        for (int oh = 0; oh < sp.ho; ++oh) {
            for (int ow = 0; ow < sp.wo; ++ow) {
                double sum = 0.0;
                double asum = 0.0;
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
                            asum += std::fabs((double)a * (double)b);
                        }
                    }
                }
                acc[((size_t)co * sp.ho + oh) * sp.wo + ow] = sum;
                if (absAcc != nullptr) (*absAcc)[((size_t)co * sp.ho + oh) * sp.wo + ow] = asum;
            }
        }
    }
}

// 报告 int32 不溢出所允许的最大**反缩放指数 F**。
//
// 约束取在**部分和**上，不是最终值上：L0C 一路累加，中途的部分和最大能到
// sum|a*b| + |bias|，这个 shape 下它比 |最终值| 大 3 倍还多。最终值不溢出而部分
// 和溢出时，结果对不对取决于 L0C 的加法器是回绕还是饱和 —— 我没有依据断定是哪
// 一种，所以直接按上界挑，把这个问题消掉。代价是 conv1 少 2 位、conv2 少 1 位
// 精度，而定点格本来就比 fp16 的 ULP 细上千倍，这点损失看不见。
//
// 注意 F 不是算子属性：属性是 S = 58 - F（见文件头）。要属性值请过一道
// AttrFromDeqExp()，别把这个返回值直接当 fixed_shift 下发 —— 上一版就是这么错的。
inline int MaxSafeDeqExp(const std::vector<double>& acc, const std::vector<double>& absAcc, const uint16_t* bias,
                         int cout, int hw, double* peak, double* peakPartial)
{
    double m = 0.0;
    double mp = 0.0;
    for (int co = 0; co < cout; ++co) {
        const double b = (double)F16BitsToF32(bias[co]);
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
    if (mp <= 0.0) return FIX_SHIFT_LEN_A16W16;
    int F = 0;
    while (F < FIX_SHIFT_LEN_A16W16 && mp * std::ldexp(1.0, F + 1) < 2147483000.0) ++F;
    return F;
}

// 厂商的默认工作点。CANN 自己的 matmul 在 5102 上就用这个值：
//   matmul/common/cmct/block/block_mmad_pingpong_without_que.h:144
//   matmul/common/cmct/block/block_mmad_iterbatch.h:85
//       #if __NPU_ARCH__ == 5102
//           uint8_t shiftValue_{42};
//   58 - 42 = 16
// 权重 int8 的那条路（weight_quant_batch_matmul_v2_tiling.cpp:33）默认 13，
// FIX_SHIFT_LEN_A16W8 = 29，29 - 13 也是 16。旧版 API 里 DEQF16 的系数干脆写死成
// 0x37800000 = 2^-16，注释就是 "fix point 1/2^16"。三条独立证据都落在 F=16。
//
// 上一版自作主张挑「最大安全 F」（24/21），偏离了这个工作点，上板结果不成立：
// 按部分和算本不该溢出的定标却出现了 int32 饱和，而且饱和与否和 |真值| 无关。
// 所以定标不再自己挑，跟厂商走。MaxSafeDeqExp 保留，只用来校验 F=16 有没有余量。
constexpr int DEFAULT_ATTR_SHIFT = 42;

// 反缩放指数 F -> 算子属性 S。两者之和恒为 58。
inline int AttrFromDeqExp(int deqExp)
{
    const int S = FIX_SHIFT_LEN_A16W16 - deqExp;
    return S < 0 ? 0 : (S > FIX_SHIFT_LEN_A16W16 ? FIX_SHIFT_LEN_A16W16 : S);
}

// 定点落格 + bias + relu -> fp16 位型。
// attrShift 是**算子属性 S**；实际用的反缩放指数是 F = 58 - S（推导见文件头）。
//   acc_i32 = round(sum * 2^F) + round(bias * 2^F)
//   out     = fp16(acc_i32 * 2^-F)，再 relu
// bias 的那一次 round 对应 kernel 里 L1->BT 带 fixShiftVal = 58 - S 的那条搬运。
inline void FixedEpilogue(const std::vector<double>& acc, const uint16_t* bias, int cout, int hw, int attrShift,
                          bool relu, std::vector<uint16_t>& out, long* satCount)
{
    out.assign((size_t)cout * hw, 0);
    const int deqExp = FIX_SHIFT_LEN_A16W16 - attrShift;
    const double scale = std::ldexp(1.0, deqExp);
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
    int shift1 = 0, shift2 = 0;      // 算子属性 S（下发给 fixed_shift1/2）
    int deqExp1 = 0, deqExp2 = 0;    // 实际定标指数 F = 58 - S
    double peak1 = 0, peak2 = 0;              // |最终值 + bias| 的峰值
    double peakPartial1 = 0, peakPartial2 = 0; // sum|a*b| + |bias| 的峰值
    int safeExp1 = 0, safeExp2 = 0;            // 部分和不溢出所允许的最大 F（只作校验）
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

// shiftFor* 是**算子属性 S**。传 -1 表示按数据自动挑（先算安全的 F，再取 58-F）。
inline Golden BuildGolden(const Inputs& in, int shiftFor1 = -1, int shiftFor2 = -1)
{
    Golden g;
    const ConvSpec sp1{CI, HI, WI, COUT1, HO1, WO1, STRIDE1};
    std::vector<double> acc1, abs1;
    ConvFwdRaw(in.xNchw.data(), in.w1Nchw.data(), sp1, acc1, &abs1);
    const int hw1 = HO1 * WO1;
    const int autoExp1 = MaxSafeDeqExp(acc1, abs1, in.b1.data(), COUT1, hw1, &g.peak1, &g.peakPartial1);
    g.safeExp1 = autoExp1;
    // 跟厂商的工作点走；只有当 F=16 都装不下时才退到自己算的安全值。
    g.shift1 = shiftFor1 >= 0 ? shiftFor1
             : (autoExp1 >= FIX_SHIFT_LEN_A16W16 - DEFAULT_ATTR_SHIFT ? DEFAULT_ATTR_SHIFT
                                                                      : AttrFromDeqExp(autoExp1));
    g.deqExp1 = FIX_SHIFT_LEN_A16W16 - g.shift1;
    FixedEpilogue(acc1, in.b1.data(), COUT1, hw1, g.shift1, true, g.midFixed, &g.sat);
    ExactEpilogue(acc1, in.b1.data(), COUT1, hw1, true, g.midExact);

    const ConvSpec sp2{COUT1, HO1, WO1, COUT2, HO2, WO2, STRIDE2};
    std::vector<double> acc2, abs2;
    // conv2 吃的是 conv1 的**定点**结果 —— 板上就是这条链，不能拿 exact 的中间值。
    ConvFwdRaw(g.midFixed.data(), in.w2Nchw.data(), sp2, acc2, &abs2);
    const int hw2 = HO2 * WO2;
    const int autoExp2 = MaxSafeDeqExp(acc2, abs2, in.b2.data(), COUT2, hw2, &g.peak2, &g.peakPartial2);
    g.safeExp2 = autoExp2;
    g.shift2 = shiftFor2 >= 0 ? shiftFor2
             : (autoExp2 >= FIX_SHIFT_LEN_A16W16 - DEFAULT_ATTR_SHIFT ? DEFAULT_ATTR_SHIFT
                                                                      : AttrFromDeqExp(autoExp2));
    g.deqExp2 = FIX_SHIFT_LEN_A16W16 - g.shift2;
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
