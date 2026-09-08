/*
 * 生成上板验证用的 case 文件。在**任何能编译的机器**上跑，不需要 CANN、不需要设备。
 *
 *   ./gen_case out.bin --dtype fp16 --n 1 --ci 32 --hi 288 --wi 112 \
 *              --cout1 64 --cout2 96 --kh 3 --kw 3 --s1 1 --s2 2 \
 *              --ph1 1 --pw1 1 --ph2 1 --pw2 1 --bias 0 --relu1 1 --relu2 1 \
 *              --shift1 42 --shift2 42
 *
 * 形状全部从命令行来 —— 上一版是写死的，一个形状一个二进制。现在
 * fc2d.py 从 cases.json 里读一条就展开成上面这样一行，所以**改 json 就能换形状**。
 *
 * ---------------------------------------------------------------------------
 * 文件格式（version = 4，和 run_fused_conv2d.py 配套）
 * ---------------------------------------------------------------------------
 *   magic "FC2DCASE" (8B)
 *   version u32 = 4
 *   ntensors u32
 *   nonzero  u64   golden 里非零输出的个数
 *   sat      u64   落格时撞到边界的个数（fp16 是 int32 饱和，int8 是 ±127）
 *   y_elems  u64
 *   spec     32 x i32  —— 完整的形状 + 属性，见下面 SPEC_* 的定义
 *   qscale   2 x u32   —— int8 通路的两个 quant_scale 的 float 位型
 *   然后每个张量：name[16], dtype u32, ndim u32, dims[4] i64, nbytes u64, data
 *
 * **spec 是这个文件里唯一的形状真相。** run_fused_conv2d.py 和 fc2d.py 生成
 * singleop.json 时都从它读，所以 .bin 和 .om 不可能对不上形状 —— 上一版靠脚本
 * 里另写一份 shape，改了 golden 忘了改 json，板上报的是「算子没找到」。
 *
 * 张量顺序就是算子的 ABI 顺序：x, filter1, bias1, filter2, bias2 -> y
 * 外加 golden：
 *   fp16   y_expect（定点模型，假设成立时应逐位相等）+ y_exact（纯 fp32 参考）
 *   int8   y_expect（精确整数运算 + REQ8，应逐位相等）
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fused_conv2d_golden.h"

using namespace fc2d_golden;

constexpr uint32_t ACL_DT_FLOAT16 = 1;
constexpr uint32_t ACL_DT_INT8 = 2;
constexpr uint32_t ACL_DT_INT32 = 3;

constexpr uint32_t CASE_VERSION = 4;
constexpr int SPEC_N = 32; // spec 数组的长度，留了余量

// spec 数组的下标。**只能往后加，不能插**：run_fused_conv2d.py 按同样的下标读。
enum SpecIdx {
    SPEC_N_BATCH = 0,
    SPEC_CI,
    SPEC_HI,
    SPEC_WI,
    SPEC_COUT1,
    SPEC_COUT2,
    SPEC_KH,
    SPEC_KW,
    SPEC_S1,
    SPEC_S2,
    SPEC_PH1,
    SPEC_PW1,
    SPEC_PH2,
    SPEC_PW2,
    SPEC_ELEM_BYTES, // 2 = fp16, 1 = int8
    SPEC_BIAS,
    SPEC_RELU1,
    SPEC_RELU2,
    SPEC_SHIFT1,
    SPEC_SHIFT2,
    SPEC_HO2,
    SPEC_WO2,
    SPEC_SAFE_F1, // 部分和不溢出所允许的最大 F（只有 fp16 有意义）
    SPEC_SAFE_F2,
    SPEC_COUNT
};
static_assert(SPEC_COUNT <= SPEC_N, "spec 数组放不下");

struct Writer {
    FILE* f;
    void U32(uint32_t v) { fwrite(&v, 4, 1, f); }
    void U64(uint64_t v) { fwrite(&v, 8, 1, f); }
    void I32(int32_t v) { fwrite(&v, 4, 1, f); }
    void Raw(const void* p, size_t n) { fwrite(p, 1, n, f); }
    void Tensor(const char* name, uint32_t dtype, const std::vector<int64_t>& dims, const void* data,
                size_t nbytes)
    {
        char nm[16];
        std::memset(nm, 0, sizeof(nm));
        std::snprintf(nm, sizeof(nm), "%s", name);
        Raw(nm, 16);
        U32(dtype);
        U32((uint32_t)dims.size());
        for (int i = 0; i < 4; ++i) {
            const int64_t d = i < (int)dims.size() ? dims[i] : 0;
            fwrite(&d, 8, 1, f);
        }
        U64((uint64_t)nbytes);
        Raw(data, nbytes);
    }
};

static int ArgInt(int argc, char** argv, const char* key, int dflt, bool* found = nullptr)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            if (found != nullptr) {
                *found = true;
            }
            return std::atoi(argv[i + 1]);
        }
    }
    return dflt;
}

static const char* ArgStr(int argc, char** argv, const char* key, const char* dflt)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return dflt;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr,
                     "用法: %s <out.bin> [--dtype fp16|int8] [--n N] [--ci N] [--hi N] [--wi N]\n"
                     "        [--cout1 N] [--cout2 N] [--kh N] [--kw N] [--s1 N] [--s2 N]\n"
                     "        [--ph1 N] [--pw1 N] [--ph2 N] [--pw2 N]\n"
                     "        [--bias 0|1] [--relu1 0|1] [--relu2 0|1] [--shift1 N] [--shift2 N]\n"
                     "缺省就是这个算子最早那组固定形状（32->64->96, 288x112, 3x3, s1/2, pad1）。\n",
                     argv[0]);
        return 2;
    }
    const char* out = argv[1];

    Case c;
    const std::string dtype = ArgStr(argc, argv, "--dtype", "fp16");
    if (dtype == "int8") {
        c.elemBytes = 1;
    } else if (dtype == "fp16") {
        c.elemBytes = 2;
    } else {
        std::fprintf(stderr, "[X] --dtype 只能是 fp16 或 int8，给的是 %s\n", dtype.c_str());
        return 2;
    }
    c.n = ArgInt(argc, argv, "--n", c.n);
    c.ci = ArgInt(argc, argv, "--ci", c.ci);
    c.hi = ArgInt(argc, argv, "--hi", c.hi);
    c.wi = ArgInt(argc, argv, "--wi", c.wi);
    c.cout1 = ArgInt(argc, argv, "--cout1", c.cout1);
    c.cout2 = ArgInt(argc, argv, "--cout2", c.cout2);
    c.kh = ArgInt(argc, argv, "--kh", c.kh);
    c.kw = ArgInt(argc, argv, "--kw", c.kw);
    c.stride1 = ArgInt(argc, argv, "--s1", c.stride1);
    c.stride2 = ArgInt(argc, argv, "--s2", c.stride2);
    c.padH1 = ArgInt(argc, argv, "--ph1", c.padH1);
    c.padW1 = ArgInt(argc, argv, "--pw1", c.padW1);
    c.padH2 = ArgInt(argc, argv, "--ph2", c.padH2);
    c.padW2 = ArgInt(argc, argv, "--pw2", c.padW2);
    c.bias = ArgInt(argc, argv, "--bias", c.bias ? 1 : 0) != 0;
    c.relu1 = ArgInt(argc, argv, "--relu1", c.relu1 ? 1 : 0) != 0;
    c.relu2 = ArgInt(argc, argv, "--relu2", c.relu2 ? 1 : 0) != 0;
    c.shift1 = ArgInt(argc, argv, "--shift1", c.shift1);
    c.shift2 = ArgInt(argc, argv, "--shift2", c.shift2);

    // 形状的基本自洽性。算子的 tiling 还会再查一遍（而且更严），这里只挡住那些
    // 会让 golden 本身算不出来的。
    const int c0 = c.C0();
    if (c.ci % c0 != 0 || c.cout1 % c0 != 0) {
        std::fprintf(stderr, "[X] ci(%d) 和 cout1(%d) 必须是 C0=%d 的整数倍\n", c.ci, c.cout1, c0);
        return 2;
    }
    if (c.cout1 % 16 != 0 || c.cout2 % 16 != 0) {
        std::fprintf(stderr, "[X] cout1(%d) / cout2(%d) 必须是 16 的整数倍\n", c.cout1, c.cout2);
        return 2;
    }
    if (c.Ho1() <= 0 || c.Wo1() <= 0 || c.Ho2() <= 0 || c.Wo2() <= 0) {
        std::fprintf(stderr, "[X] 卷出来是空的：conv1 -> %dx%d, conv2 -> %dx%d\n", c.Ho1(), c.Wo1(),
                     c.Ho2(), c.Wo2());
        return 2;
    }

    std::printf("形状: %s\n", c.Text().c_str());
    std::printf("  conv1 -> %dx%d，conv2 -> %dx%d，y 共 %ld 个元素\n", c.Ho1(), c.Wo1(), c.Ho2(), c.Wo2(),
                c.YElems());

    int32_t spec[SPEC_N];
    std::memset(spec, 0, sizeof(spec));
    spec[SPEC_N_BATCH] = c.n;
    spec[SPEC_CI] = c.ci;
    spec[SPEC_HI] = c.hi;
    spec[SPEC_WI] = c.wi;
    spec[SPEC_COUT1] = c.cout1;
    spec[SPEC_COUT2] = c.cout2;
    spec[SPEC_KH] = c.kh;
    spec[SPEC_KW] = c.kw;
    spec[SPEC_S1] = c.stride1;
    spec[SPEC_S2] = c.stride2;
    spec[SPEC_PH1] = c.padH1;
    spec[SPEC_PW1] = c.padW1;
    spec[SPEC_PH2] = c.padH2;
    spec[SPEC_PW2] = c.padW2;
    spec[SPEC_ELEM_BYTES] = c.elemBytes;
    spec[SPEC_BIAS] = c.bias ? 1 : 0;
    spec[SPEC_RELU1] = c.relu1 ? 1 : 0;
    spec[SPEC_RELU2] = c.relu2 ? 1 : 0;
    spec[SPEC_SHIFT1] = c.shift1;
    spec[SPEC_SHIFT2] = c.shift2;
    spec[SPEC_HO2] = c.Ho2();
    spec[SPEC_WO2] = c.Wo2();

    FILE* f = std::fopen(out, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "[X] 打不开 %s\n", out);
        return 1;
    }
    Writer w{f};

    const int64_t fz1k = (int64_t)(c.ci / c0) * c.kh * c.kw;
    const int64_t fz1n = c.cout1 / 16;
    const int64_t fz2k = (int64_t)(c.cout1 / c0) * c.kh * c.kw;
    const int64_t fz2n = c.cout2 / 16;
    const std::vector<int64_t> xDims = {c.n, c.ci, c.hi, c.wi};
    const std::vector<int64_t> yDims = {c.n, c.cout2, c.Ho2(), c.Wo2()};
    const std::vector<int64_t> f1Dims = {fz1k, fz1n, 16, (int64_t)c0};
    const std::vector<int64_t> f2Dims = {fz2k, fz2n, 16, (int64_t)c0};

    if (c.elemBytes == 2) {
        f16path::Result g = f16path::Build(c);
        spec[SPEC_SAFE_F1] = g.safeF1;
        spec[SPEC_SAFE_F2] = g.safeF2;
        const int F1 = FIX_SHIFT_LEN_A16W16 - c.shift1;
        const int F2 = FIX_SHIFT_LEN_A16W16 - c.shift2;
        std::printf("  定点: S=%d/%d 即 F=%d/%d；部分和不溢出允许到 F<=%d/%d\n", c.shift1, c.shift2, F1, F2,
                    g.safeF1, g.safeF2);
        std::printf("  峰值 |最终值| %.3g / %.3g，|部分和| %.3g / %.3g\n", g.peak1, g.peak2, g.peakPartial1,
                    g.peakPartial2);
        if (F1 > g.safeF1 || F2 > g.safeF2) {
            std::printf("  [!] 定标超出安全区间 —— 累加器会溢出，逐位比对不会成立。\n"
                        "      把 --shift1/--shift2 调大（S 越大定标越小）。\n");
        }
        std::printf("  golden: 非零 %ld / %ld，落格饱和 %ld 处\n", g.yNonZero, c.YElems(), g.sat);

        // golden 自检：定点模型和纯 fp32 参考必须彼此接近。
        //
        // 这不是重复计算 —— 两条路是**独立**的：y_exact 完全不经过定点格，
        // y_expect 经过 round(sum * 2^F) 再乘回去。F = 16 时定点格比 fp16 的 ULP
        // 细几千倍，所以两者应当几乎处处相同，个别点差一个 fp16 ULP。
        // 差得多说明定点模型本身写错了（比如把 F 和 S 搞反），而那种错误在板上
        // 表现成「算子算错了」，会把人带到完全错误的方向去查。
        {
            long bad = 0;
            double worst = 0.0;
            for (size_t i = 0; i < g.yExpect.size(); ++i) {
                const double a = (double)F16BitsToF32(g.yExpect[i]);
                const double b = (double)F16BitsToF32(g.yExact[i]);
                const double den = std::fabs(b) > 1e-6 ? std::fabs(b) : 1.0;
                const double rel = std::fabs(a - b) / den;
                if (rel > worst) worst = rel;
                if (rel > 1e-2) ++bad;
            }
            // relu 的边界上会有个别点：定点那边正好舍到 0（relu 出 -0），fp32 那边
            // 是个极小的正数，相对差 = 1。零星几个是正常的，所以判据是「超 1% 的点
            // 不到总数的 0.1%」，不是「一个都没有」。
            std::printf("  自检: 定点模型 vs fp32 参考，最大相对差 %.3g，超 1%% 的 %ld 处"
                        "（relu 边界上零星几个是正常的）\n", worst, bad);
            if (bad * 1000 > (long)g.yExpect.size()) {
                std::printf("  [X] 两个 golden 分歧太大 —— 定点模型多半写错了（F 和 S 搞反？）。\n"
                            "      板上比对之前先把这个查清楚，否则会把算子的问题和 golden 的问题混在一起。\n");
                // 把半成品删掉：留一个 0 字节的 .bin 在那里，下一步会拿它去跑，
                // 报出来的错和真正的原因就隔了一层。
                std::fclose(f);
                std::remove(out);
                return 3;
            }
        }

        w.Raw("FC2DCASE", 8);
        w.U32(CASE_VERSION);
        w.U32(8); // x, f1, b1, f2, b2, y_expect, y_exact + 占位（见下面的 ntensors）
        w.U64((uint64_t)g.yNonZero);
        w.U64((uint64_t)g.sat);
        w.U64((uint64_t)c.YElems());
        for (int i = 0; i < SPEC_N; ++i) {
            w.I32(spec[i]);
        }
        w.U32(0);
        w.U32(0); // fp16 通路不用 quant_scale

        w.Tensor("x", ACL_DT_FLOAT16, xDims, g.xNchw.data(), g.xNchw.size() * 2);
        w.Tensor("filter1", ACL_DT_FLOAT16, f1Dims, g.w1Dev.data(), g.w1Dev.size() * 2);
        w.Tensor("bias1", ACL_DT_FLOAT16, {(int64_t)c.cout1}, g.b1.data(), g.b1.size() * 2);
        w.Tensor("filter2", ACL_DT_FLOAT16, f2Dims, g.w2Dev.data(), g.w2Dev.size() * 2);
        w.Tensor("bias2", ACL_DT_FLOAT16, {(int64_t)c.cout2}, g.b2.data(), g.b2.size() * 2);
        w.Tensor("y_expect", ACL_DT_FLOAT16, yDims, g.yExpect.data(), g.yExpect.size() * 2);
        w.Tensor("y_exact", ACL_DT_FLOAT16, yDims, g.yExact.data(), g.yExact.size() * 2);
        // 第 8 个张量：把 conv1 的 fp16 权重按 NCHW 也存一份，出问题时能在主机上
        // 重算任何中间量，不用回头再生成一次。
        w.Tensor("w1_nchw", ACL_DT_FLOAT16, {(int64_t)c.cout1, (int64_t)c.ci, (int64_t)c.kh, (int64_t)c.kw},
                 g.w1Nchw.data(), g.w1Nchw.size() * 2);
    } else {
        int8path::Result g = int8path::Build(c);
        uint32_t qs1 = 0, qs2 = 0;
        std::memcpy(&qs1, &g.scale1, 4);
        std::memcpy(&qs2, &g.scale2, 4);
        std::printf("  量化: scale1 = %.9g，scale2 = %.9g（低 13 位尾数已按硬件丢掉）\n", g.scale1,
                    g.scale2);
        std::printf("  golden: 非零 %ld / %ld，饱和到 ±127 的 %ld 处\n", g.yNonZero, c.YElems(), g.sat);
        if (g.sat * 20 > c.YElems()) {
            std::printf("  [!] 饱和比例超过 5%% —— scale 挑得偏大，比对的判别力会下降\n");
        }

        w.Raw("FC2DCASE", 8);
        w.U32(CASE_VERSION);
        w.U32(7); // x, f1, b1, f2, b2, y_expect, w1_nchw
        w.U64((uint64_t)g.yNonZero);
        w.U64((uint64_t)g.sat);
        w.U64((uint64_t)c.YElems());
        for (int i = 0; i < SPEC_N; ++i) {
            w.I32(spec[i]);
        }
        w.U32(qs1);
        w.U32(qs2);

        w.Tensor("x", ACL_DT_INT8, xDims, g.xNchw.data(), g.xNchw.size());
        w.Tensor("filter1", ACL_DT_INT8, f1Dims, g.w1Dev.data(), g.w1Dev.size());
        w.Tensor("bias1", ACL_DT_INT32, {(int64_t)c.cout1}, g.b1.data(), g.b1.size() * 4);
        w.Tensor("filter2", ACL_DT_INT8, f2Dims, g.w2Dev.data(), g.w2Dev.size());
        w.Tensor("bias2", ACL_DT_INT32, {(int64_t)c.cout2}, g.b2.data(), g.b2.size() * 4);
        w.Tensor("y_expect", ACL_DT_INT8, yDims, g.y.data(), g.y.size());
        w.Tensor("w1_nchw", ACL_DT_INT8, {(int64_t)c.cout1, (int64_t)c.ci, (int64_t)c.kh, (int64_t)c.kw},
                 g.w1Nchw.data(), g.w1Nchw.size());
    }

    const long sz = std::ftell(f);
    std::fclose(f);
    std::printf("  写出 %s，%ld 字节\n", out, sz);
    return 0;
}
