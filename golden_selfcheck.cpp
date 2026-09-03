// golden 的自检。build_case.sh 在生成 .bin 之前先跑它 —— golden 不可信的话，
// 后面比对出来的所有结论都没有意义。
#include "fused_conv2d_golden.h"
#include <cstdio>
using namespace fc2d_golden;

int main()
{
    // 1) fp16 位型转换：对全部 65536 个位型做往返自检。上一版的 F16ToFloat 有个
    //    次正规数的指数 bug（2046 个值不对），就是被这一条抓出来的。
    int bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint16_t b = (uint16_t)h;
        const uint32_t exp = (b >> 10) & 0x1Fu;
        if (exp == 0x1F) continue;                    // Inf/NaN 不参与往返
        const float f = F16BitsToF32(b);
        const uint16_t rt = F32ToF16Bits(f);
        const uint16_t norm = ((b & 0x7FFFu) == 0) ? 0 : b;   // +0 / -0 都归 +0
        const uint16_t rtn = ((rt & 0x7FFFu) == 0) ? 0 : rt;
        if (norm != rtn) { if (bad < 5) std::printf("  往返不符 %04x -> %g -> %04x\n", b, f, rt); ++bad; }
    }
    std::printf("fp16 往返自检: %d/65536 不符\n", bad);
    if (bad) return 1;

    // 2) FRACTAL_Z 往返
    Inputs in = GenerateInputs();
    std::vector<uint16_t> back;
    WeightFromFractalZ(in.w1Dev.data(), CI, COUT1, back);
    if (back != in.w1Nchw) { std::printf("[X] w1 的 FRACTAL_Z 往返不符\n"); return 1; }
    WeightFromFractalZ(in.w2Dev.data(), COUT1, COUT2, back);
    if (back != in.w2Nchw) { std::printf("[X] w2 的 FRACTAL_Z 往返不符\n"); return 1; }
    std::printf("FRACTAL_Z 往返自检: 两套权重都对上\n");

    // 3) 建 golden，报告标定用的量
    Golden g = BuildGolden(in);
    std::printf("\n定点定标（这两个数就是算子属性 fixed_shift1 / fixed_shift2 该传的值）:\n");
    std::printf("  conv1  峰值 |sum+bias| = %.4f   ->  最大安全 S = %d\n", g.peak1, g.shift1);
    std::printf("  conv2  峰值 |sum+bias| = %.4f   ->  最大安全 S = %d\n", g.peak2, g.shift2);
    std::printf("  int32 饱和点 %ld 个（应为 0）\n", g.sat);

    // 4) 定点模型 vs 纯 fp32 参考差多少 —— 这个差就是定点落格引入的误差，
    //    上板时 GoldenExact 的判据要按它放宽。
    double maxRel = 0.0, sumAbs = 0.0;
    long n = 0, exactZero = 0;
    for (size_t i = 0; i < g.yFixed.size(); ++i) {
        const double a = F16BitsToF32(g.yFixed[i]);
        const double b = F16BitsToF32(g.yExact[i]);
        sumAbs += std::fabs(a - b);
        ++n;
        if (b == 0.0) { ++exactZero; continue; }
        const double r = std::fabs(a - b) / std::fabs(b);
        if (r > maxRel) maxRel = r;
    }
    std::printf("\n定点模型 vs fp32 参考:  最大相对误差 %.3e   平均绝对差 %.3e\n", maxRel, sumAbs / (double)n);
    std::printf("  （fp32 参考里为 0 的点 %ld 个，不参与相对误差）\n", exactZero);
    std::printf("  y 非零 %ld / %zu\n", g.yNonZero, g.yFixed.size());
    if (g.yNonZero == 0) { std::printf("[X] golden 全是 0，先别管设备\n"); return 1; }

    std::printf("\n[OK] golden 自检通过\n");
    return 0;
}
