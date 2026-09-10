/*
 * 形状预检 —— 在生成 case、编 .om 之前就回答「这个形状算子能不能跑」。
 *
 *   ./fc2d_geom --dtype fp16 --ci 32 --hi 288 --wi 112 --cout1 64 --cout2 96 [--l1 1048576]
 *
 * 用的是算子共用几何头的一份副本（fused_conv2d_shape.h）。**顾问性质**：真正说了
 * 算的是算子的 tiling。不过两边调的是同一个 DeriveWithHb / PickHb，只要副本没过期
 * 结论就一致。
 *
 * 退出码 0 = 这个形状成立，非 0 = 不成立（原因打在 stderr）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fused_conv2d_shape.h"

using namespace FusedConv2dShape;

static int ArgInt(int argc, char** argv, const char* key, int dflt)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    }
    return dflt;
}
static const char* ArgStr(int argc, char** argv, const char* key, const char* dflt)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return dflt;
}

int main(int argc, char** argv)
{
    Params p;
    p.n = ArgInt(argc, argv, "--n", 1);
    p.ci = ArgInt(argc, argv, "--ci", 32);
    p.hi = ArgInt(argc, argv, "--hi", 288);
    p.wi = ArgInt(argc, argv, "--wi", 112);
    p.cout1 = ArgInt(argc, argv, "--cout1", 64);
    p.cout2 = ArgInt(argc, argv, "--cout2", 96);
    p.kh = ArgInt(argc, argv, "--kh", 3);
    p.kw = ArgInt(argc, argv, "--kw", 3);
    p.stride1 = ArgInt(argc, argv, "--s1", 1);
    p.stride2 = ArgInt(argc, argv, "--s2", 2);
    p.padH1 = ArgInt(argc, argv, "--ph1", 1);
    p.padW1 = ArgInt(argc, argv, "--pw1", 1);
    p.padH2 = ArgInt(argc, argv, "--ph2", 1);
    p.padW2 = ArgInt(argc, argv, "--pw2", 1);
    // s8f16 = int8 进 / fp16 出：入口和 int8 一样（elemBytes = 1），但 L1 末尾
    // 多一段 per-channel 反量化表，所以几何要按带表的算。
    const char* dt = ArgStr(argc, argv, "--dtype", "fp16");
    const bool s8f16 = std::strcmp(dt, "s8f16") == 0;
    p.elemBytes = (std::strcmp(dt, "int8") == 0 || s8f16) ? 1 : 2;
    p.hasDeqScale2 = s8f16 ? 1 : 0;
    const int l1 = ArgInt(argc, argv, "--l1", 1024 * 1024);
    const int aic = ArgInt(argc, argv, "--cores", 8);

    Geometry g;
    const int rc = PickHb(p, aic, l1, g);
    if (rc != RJ_OK) {
        std::fprintf(stderr, "不支持: %s\n", RejectText(rc));
        return 1;
    }
    std::printf("    conv1 -> %dx%d, conv2 -> %dx%d | band=%d 行 x %d 块(共 %d) | "
                "L1 %d/%d 字节 (%.0f%%) | tileK %d/%d, M 子块 <=%d/%d 行, L0B 折 %d/%d 段\n",
                g.ho1, g.wo1, g.ho2, g.wo2, g.hb, g.nchunk, g.chunkTotal, g.l1Used, l1,
                100.0 * g.l1Used / l1, g.tileK1, g.tileK2, g.rowsMax1, g.rowsMax2, g.l0bChunks1,
                g.l0bChunks2);
    if (g.dq2Bytes > 0) {
        std::printf("    反量化表: %d 字节常驻 L1（%d 个通道 x uint64）\n", g.dq2Bytes, p.cout2);
    }
    return 0;
}
