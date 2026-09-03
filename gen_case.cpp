/*
 * 生成上板验证用的 case 文件。在**任何能编译的机器**上跑，不需要 CANN、不需要设备。
 *
 *   ./gen_case fused_conv2d_case.bin
 *
 * 文件格式（和 run_fused_conv2d.py 配套，version = 2）：
 *   header  magic "FC2DCASE", version, ntensors, nonzero, probe, sat, y_elems,
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
#include <cstring>
#include <algorithm>

using namespace fc2d_golden;

constexpr uint32_t ACL_DT_FLOAT16 = 1;

// ---------------------------------------------------------------------------
// 探针模式。把某一层的权重换成「中心抽头恒等」，用来把两层拆开单独看。
//
// 为什么值得做：随机权重下输出错了，分不清是**定点运算**错了还是**数据搬运**
// （im2col / FRACTAL_Z / L0B 分块 / NZ 排布）错了。恒等权重让卷积退化成一次
// 取值，搬运一旦错位，结果立刻能看出来；反过来如果恒等下是对的，那搬运没问题，
// 问题就锁死在定点运算上。
//
// 关键：权重是**运行时输入**，不是算子属性 —— 换权重不影响 .om 的匹配
// （op 类型 + shape/dtype/format + attr 才参与匹配）。所以这些探针只要重生成
// .bin，**不用重编 .om**。属性仍然固定 S=42，不能动，动了 .om 就对不上。
//
//   passthrough  conv1 conv2 都恒等、bias 全 0
//                y[c][h][w] = relu(x[c][2h][2w])，c < 32；c >= 32 全 0
//                纯粹的数据搬运测试。这个都不过，说明和 fixShiftVal 无关。
//   mid          只有 conv2 恒等，conv1 保持随机
//                y 直接暴露 conv1 的输出（下采样后），把 conv1 单独拎出来看。
// ---------------------------------------------------------------------------
enum class Probe { None, Passthrough, Mid };

// NCHW [cout][cin][KH][KW]：co == ci 的中心抽头置 1.0，其余全 0。
// cout > cin 时多出来的输出通道全 0 —— golden 会照样算出来，不用特殊处理。
static void MakeIdentityWeight(std::vector<uint16_t>& w, int cout, int cin)
{
    std::fill(w.begin(), w.end(), (uint16_t)0);
    const int n = cout < cin ? cout : cin;
    for (int c = 0; c < n; ++c) {
        w[(((size_t)c * cin + c) * KH + (KH / 2)) * KW + (KW / 2)] = 0x3C00u;  // fp16 1.0
    }
}

static void ApplyProbe(Inputs& in, Probe mode)
{
    if (mode == Probe::None) return;
    if (mode == Probe::Passthrough) {
        MakeIdentityWeight(in.w1Nchw, COUT1, CI);
        std::fill(in.b1.begin(), in.b1.end(), (uint16_t)0);
    }
    MakeIdentityWeight(in.w2Nchw, COUT2, COUT1);
    std::fill(in.b2.begin(), in.b2.end(), (uint16_t)0);
    // 权重改了，设备用的 FRACTAL_Z 副本必须重算 —— 否则 golden 和下发的权重是两份。
    WeightToFractalZ(in.w1Nchw.data(), CI, COUT1, in.w1Dev);
    WeightToFractalZ(in.w2Nchw.data(), COUT1, COUT2, in.w2Dev);
}


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
    const char* path = "fused_conv2d_case.bin";
    Probe probe = Probe::None;
    const char* probeName = "none";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--probe=", 8) == 0) {
            const char* v = a + 8;
            if (std::strcmp(v, "passthrough") == 0)      { probe = Probe::Passthrough; probeName = v; }
            else if (std::strcmp(v, "mid") == 0)          { probe = Probe::Mid;         probeName = v; }
            else if (std::strcmp(v, "none") == 0)         { probe = Probe::None;        probeName = v; }
            else { std::printf("[X] --probe 只认 none / passthrough / mid，收到 %s\n", v); return 1; }
        } else if (a[0] == '-') {
            std::printf("[X] 不认识的参数 %s\n", a); return 1;
        } else {
            path = a;
        }
    }

    Inputs in = GenerateInputs();
    ApplyProbe(in, probe);
    if (probe != Probe::None) {
        std::printf("探针模式 = %s（权重换成恒等；属性仍是 S=42，.om 不用重编）\n", probeName);
    }
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
    // 这个字段原本恒为 0（int8 时代的 ties 计数），现在用来放探针 id：
    // 0=none 1=passthrough 2=mid。文件自己说明自己是哪一份，免得搞混。
    const uint64_t hTies = (uint64_t)probe;
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
