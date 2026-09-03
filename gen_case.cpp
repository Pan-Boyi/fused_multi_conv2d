/*
 * 生成上板验证用的 case 文件。在**任何能编译的机器**上跑，不需要 CANN、不需要设备。
 *
 *   ./gen_case fused_conv2d_case.bin
 *
 * 文件格式（和 run_fused_conv2d.py 配套，version = 2）：
 *   header  magic "FC2DCASE", version, ntensors, nonzero, ties, sat, y_elems,
 *           attrs —— 低 8 位 fixed_shift1，次 8 位 fixed_shift2
 *   然后每个张量：name[16], dtype(u32), ndim(u32), dims[4](i64), nbytes(u64), data
 *
 * 张量顺序就是算子的 ABI 顺序：
 *   x, filter1, bias1, filter2, bias2  ->  y
 * 外加两个 golden：
 *   y_expect  定点模型算出来的，假设成立时应当**逐位相等**
 *   y_exact   纯 fp32 参考，判「这个算子有没有在算这个卷积」，不受定点假设影响
 *
 * 两个 golden 都给，是因为定点模型里那个 fixShiftVal 的语义我没有硬件实测过
 * （见 golden.h 顶部）。板上先看 y_exact 的相对误差过不过，再看 y_expect 能不能
 * 逐位对上；只有前者过、后者不过，说明卷积算对了但定点模型猜错了。
 */
#include <cmath>
#include "fused_conv2d_golden.h"
#include <cstdio>
#include <cstdlib>

using namespace fc2d_golden;

constexpr uint32_t ACL_DT_FLOAT16 = 1;

static bool WriteTensor(std::FILE* f, const char* name, uint32_t dtype, const std::vector<int64_t>& dims,
                        const void* data, uint64_t nbytes)
{
    char nameBuf[16] = {0};
    std::snprintf(nameBuf, sizeof(nameBuf), "%s", name);
    uint32_t ndim = (uint32_t)dims.size();
    int64_t dimBuf[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < dims.size() && i < 4; ++i) dimBuf[i] = dims[i];
    if (std::fwrite(nameBuf, 1, sizeof(nameBuf), f) != sizeof(nameBuf)) return false;
    if (std::fwrite(&dtype, 1, 4, f) != 4) return false;
    if (std::fwrite(&ndim, 1, 4, f) != 4) return false;
    if (std::fwrite(dimBuf, 1, sizeof(dimBuf), f) != sizeof(dimBuf)) return false;
    if (std::fwrite(&nbytes, 1, 8, f) != 8) return false;
    if (nbytes != 0 && std::fwrite(data, 1, nbytes, f) != nbytes) return false;
    return true;
}

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "fused_conv2d_case.bin";

    Inputs in = GenerateInputs();
    Golden g = BuildGolden(in);

    // S 是下发的属性；真正决定定标的是 F = 58 - S，两个都打出来，免得又搞反。
    std::printf("golden: conv1 峰值 %.4f -> 定标 2^%d (属性 S=%d)   "
                "conv2 峰值 %.4f -> 定标 2^%d (属性 S=%d)   饱和 %ld\n",
                g.peak1, g.deqExp1, g.shift1, g.peak2, g.deqExp2, g.shift2, g.sat);
    std::printf("golden: y 非零 %ld / %zu\n", g.yNonZero, g.yFixed.size());
    if (g.yNonZero == 0) { std::printf("[X] golden 全是 0\n"); return 1; }
    if (g.sat != 0) { std::printf("[X] 定点累加溢出了 int32 %ld 次 —— 定标 F 挑大了（即 S 挑小了）\n", g.sat); return 1; }

    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) { std::printf("[X] 打不开 %s\n", path); return 1; }

    const uint32_t version = 2;
    const uint32_t ntensors = 7;
    const uint64_t hNonZero = (uint64_t)g.yNonZero;
    const uint64_t hTies = 0;
    const uint64_t hSat = (uint64_t)g.sat;
    const uint64_t hYElems = (uint64_t)g.yFixed.size();
    // 两个 shift 打包进 header —— run_fused_conv2d.py 从这里取，当算子属性传下去。
    // 放在文件里而不是脚本里写死，是因为它由 golden 按数据算出来，两边必须是同一个。
    const uint64_t attrs = (uint64_t)(uint8_t)g.shift1 | ((uint64_t)(uint8_t)g.shift2 << 8);

    bool ok = std::fwrite("FC2DCASE", 1, 8, f) == 8;
    ok = ok && std::fwrite(&version, 1, 4, f) == 4;
    ok = ok && std::fwrite(&ntensors, 1, 4, f) == 4;
    ok = ok && std::fwrite(&hNonZero, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hTies, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hSat, 1, 8, f) == 8;
    ok = ok && std::fwrite(&hYElems, 1, 8, f) == 8;
    ok = ok && std::fwrite(&attrs, 1, 8, f) == 8;

    const int64_t fz1k = (CI / C0) * KH * KW, fz1n = (COUT1 + 15) / 16;
    const int64_t fz2k = (COUT1 / C0) * KH * KW, fz2n = (COUT2 + 15) / 16;

    ok = ok && WriteTensor(f, "x", ACL_DT_FLOAT16, {1, CI, HI, WI}, in.xNchw.data(), in.xNchw.size() * 2);
    ok = ok && WriteTensor(f, "filter1", ACL_DT_FLOAT16, {fz1k, fz1n, 16, C0}, in.w1Dev.data(), in.w1Dev.size() * 2);
    ok = ok && WriteTensor(f, "bias1", ACL_DT_FLOAT16, {COUT1}, in.b1.data(), in.b1.size() * 2);
    ok = ok && WriteTensor(f, "filter2", ACL_DT_FLOAT16, {fz2k, fz2n, 16, C0}, in.w2Dev.data(), in.w2Dev.size() * 2);
    ok = ok && WriteTensor(f, "bias2", ACL_DT_FLOAT16, {COUT2}, in.b2.data(), in.b2.size() * 2);
    ok = ok && WriteTensor(f, "y_expect", ACL_DT_FLOAT16, {1, COUT2, HO2, WO2}, g.yFixed.data(), g.yFixed.size() * 2);
    ok = ok && WriteTensor(f, "y_exact", ACL_DT_FLOAT16, {1, COUT2, HO2, WO2}, g.yExact.data(), g.yExact.size() * 2);

    std::fclose(f);
    if (!ok) { std::printf("[X] 写 %s 失败\n", path); return 1; }
    std::printf("\n[OK] 写出 %s\n     5 个输入 + 2 个 golden，共 %u 个张量\n", path, ntensors);
    std::printf("     算子属性 fixed_shift1 = %d, fixed_shift2 = %d（已打包进 header）\n", g.shift1, g.shift2);
    std::printf("     对应的累加器定标 2^%d / 2^%d，LSB = %.3g / %.3g\n",
                g.deqExp1, g.deqExp2, std::ldexp(1.0, -g.deqExp1), std::ldexp(1.0, -g.deqExp2));
    return 0;
}
