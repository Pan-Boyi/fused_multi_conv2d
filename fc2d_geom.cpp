/*
 * 形状预检 —— 在生成 case、编 .om 之前就回答「这个形状算子能不能跑」。
 *
 *   ./fc2d_geom --dtype fp16|int8|s8f16|a16w8 --ci 32 --hi 288 --wi 112
 *                --cout1 64 --cout2 96 [--l1 1048576]
 *
 * 用的是从算子 tiling.cpp 提取的几何副本（fused_multi_conv2d_shape.h）。**顾问性质**：真正说了
 * 算的是算子的 tiling。不过两边调的是同一个 DeriveWithHb / PickHb，只要副本没过期
 * 结论就一致。
 *
 * 退出码 0 = 这个形状成立，非 0 = 不成立（原因打在 stderr）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fused_multi_conv2d_shape.h"

using namespace FusedMultiConv2dShape;

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
    ConvTilingInputs inputs;
    inputs.batchSize = ArgInt(argc, argv, "--n", 1);
    inputs.inputChannels = ArgInt(argc, argv, "--ci", 32);
    inputs.inputHeight = ArgInt(argc, argv, "--hi", 288);
    inputs.inputWidth = ArgInt(argc, argv, "--wi", 112);
    inputs.outputChannels1 = ArgInt(argc, argv, "--cout1", 64);
    inputs.outputChannels2 = ArgInt(argc, argv, "--cout2", 96);
    inputs.kernelHeight1 = ArgInt(argc, argv, "--kh", 3);
    inputs.kernelWidth1 = ArgInt(argc, argv, "--kw", 3);
    // conv2 的核。不给就是 0 = 沿用 conv1（S::Kh2 / S::Kw2 负责翻译）。
    inputs.kernelHeight2 = ArgInt(argc, argv, "--kh2", 0);
    inputs.kernelWidth2 = ArgInt(argc, argv, "--kw2", 0);
    inputs.stride1 = ArgInt(argc, argv, "--s1", 1);
    inputs.stride2 = ArgInt(argc, argv, "--s2", 1);
    inputs.paddingHeight1 = ArgInt(argc, argv, "--ph1", 0);
    inputs.paddingWidth1 = ArgInt(argc, argv, "--pw1", 0);
    inputs.paddingHeight2 = ArgInt(argc, argv, "--ph2", 0);
    inputs.paddingWidth2 = ArgInt(argc, argv, "--pw2", 0);
    // s8f16 = int8 进 / fp16 出：入口和 int8 一样（elemBytes = 1），但 L1 末尾
    // 多一段 per-channel 反量化表，所以几何要按带表的算。
    const char* dt = ArgStr(argc, argv, "--dtype", "fp16");
    const bool int8 = std::strcmp(dt, "int8") == 0;
    const bool s8f16 = std::strcmp(dt, "s8f16") == 0;
    const bool a16w8 = std::strcmp(dt, "a16w8") == 0;
    inputs.inputElementBytes = (int8 || s8f16) ? 1 : 2;
    inputs.hasDequantScale2 = (s8f16 || a16w8) ? 1 : 0;
    inputs.hasQuantScale1 = (int8 || s8f16 || a16w8) ? 1 : 0;
    inputs.hasQuantScale2 = int8 ? 1 : 0;
    if (a16w8) {
        inputs.weightElementBytes = 1;
        inputs.intermediateElementBytes = 1;
        inputs.biasElementBytes = 4;
    }
    const int l1 = ArgInt(argc, argv, "--l1", 1024 * 1024);
    const int aic = ArgInt(argc, argv, "--cores", 8);

    ConvTilingGeometry geometry;
    const int rc = PickHb(inputs, aic, l1, geometry);
    if (rc != RJ_OK) {
        // 带上共用头的修订号。这份头是**算子那份的副本**，过期了预检的结论就和
        // 算子的 tiling 不一致 —— 而不带版本号的话，两边报的是一模一样的一句话，
        // 从输出上完全分辨不出来「是形状真不行」还是「这份副本旧了」。
        // 真实踩过：算子那边连推了三笔纯 tiling 改动（tileK 受 L0B 限、权重按段
        // 滚动、几何修订号），副本没跟着重拷，于是预检一直在用旧判据拒，而算子包
        // 其实已经能跑了 —— 白花了一整轮来回。重拷见 README 的 --sync-shape-header。
        std::fprintf(stderr, "不支持[%s]: %s\n", FC2D_GEOM_REV, RejectText(rc));
        std::fprintf(stderr, "    （这份 fused_multi_conv2d_shape.h 是算子那份的副本。"
                             "如果算子那边比 %s 更新，先跑 "
                             "python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_multi_conv2d/"
                             "op_host/op_tiling/fused_multi_conv2d_tiling.cpp）\n", FC2D_GEOM_REV);
        return 1;
    }
    std::printf("    [%s] ", FC2D_GEOM_REV);
    std::printf("    conv1 -> %dx%d, conv2 -> %dx%d | 分核 %d 行带 x %d 列段 = %d 块"
                "（每带 %d 行、每段 %d 列）| L1 %d/%d 字节 (%.0f%%) | "
                "tileK %d/%d, M 子块 <=%d/%d 个位置, "
                "L0B 折 %d/%d 段\n",
                geometry.outputHeight1, geometry.outputWidth1, geometry.outputHeight2, geometry.outputWidth2, geometry.heightBandCount, geometry.widthSegmentCount, geometry.totalBandTasks, geometry.bandHeight, geometry.segmentWidth,
                geometry.l1UsedBytes, l1, 100.0 * geometry.l1UsedBytes / l1, geometry.tileK1, geometry.tileK2, geometry.tileMMax1, geometry.tileMMax2,
                geometry.weightSegmentCount1, geometry.weightSegmentCount2);
    if (geometry.widthSegmentCount > 1) {
        std::printf("    按 W 分核：每段算 mid 的 %d 列 / x 的 %d 列（整幅是 %d / %d）\n",
                    geometry.intermediateWidthMax, geometry.inputWindowWidthMax, geometry.outputWidth1, inputs.inputWidth);
    }
    if (geometry.dequantBufferBytes2 > 0) {
        std::printf("    反量化表: %d 字节常驻 L1（%d 个通道 x uint64）\n", geometry.dequantBufferBytes2, inputs.outputChannels2);
    }
    if (geometry.quantBufferBytes1 > 0) {
        std::printf("    conv1 重量化表: %d 字节常驻 L1（%d 个通道 x uint64）\n",
                    geometry.quantBufferBytes1, inputs.outputChannels1);
    }
    if (geometry.quantBufferBytes2 > 0) {
        std::printf("    conv2 重量化表: %d 字节常驻 L1（%d 个通道 x uint64）\n",
                    geometry.quantBufferBytes2, inputs.outputChannels2);
    }
    return 0;
}
