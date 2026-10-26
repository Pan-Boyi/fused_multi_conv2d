// ===========================================================================
// 这是从 ops-nn 仓里 conv/fused_multi_conv2d/op_host/op_tiling/fused_multi_conv2d_tiling.cpp 提取的几何预检副本。
//
// 放一份在 harness 里，只为了 fc2d.py 能在生成 case 之前就回答「这个形状能不能
// 跑」—— 否则要等到 atc 失败，而那条原因只在 ~/ascend/log/ 里。
//
// **算子那边改了就要重拷。** 一条命令：
//     python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_multi_conv2d/op_host/op_tiling/fused_multi_conv2d_tiling.cpp
//
// 它是顾问性质的：真正说了算的永远是算子的 tiling。两边不一致时 fc2d.py 的预检会
// 和 atc 的结论不同，那本身就是「该重拷了」的信号。
// ===========================================================================
#ifndef FUSED_MULTI_CONV2D_SHAPE_H
#define FUSED_MULTI_CONV2D_SHAPE_H

/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_GEOMETRY_H
#define FUSED_MULTI_CONV2D_GEOMETRY_H

#if defined(__NPU_ARCH__)
#define FC2D_GEOM_FN __aicore__ inline
#define FC2D_GEOM_CE __aicore__ constexpr
#else
#define FC2D_GEOM_FN inline
#define FC2D_GEOM_CE constexpr
#endif

namespace FusedMultiConv2dShape {

constexpr int L0A_BYTES = 64 * 1024;
constexpr int L0A_SLOT_BYTES = L0A_BYTES / 2;
constexpr int BT_BYTES = 4 * 1024;
constexpr int BT_SLOT_ELEMS = 512;
constexpr int BT_SLOT1_BYTE_OFF = 2048;
constexpr int MMAD_M0 = 16;
constexpr int MAX_M_TILES = 64;
constexpr int FIX_SHIFT_LEN_A16W16 = 58;

// 返回较小值；left/right 为比较值。
FC2D_GEOM_CE int MinI(int left, int right) { return left < right ? left : right; }
// 返回较大值；left/right 为比较值。
FC2D_GEOM_CE int MaxI(int left, int right) { return left > right ? left : right; }
// 向上取整除法；value 为被除数，divisor 为除数。
FC2D_GEOM_CE int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }
// 向上对齐；value 为原值，alignment 为对齐粒度。
FC2D_GEOM_CE int Align(int value, int alignment) { return (value + alignment - 1) / alignment * alignment; }

struct TileSpec {
    int n;     // M 块总数。
    int base;  // 每块基础粒度数。
    int extra; // 额外分配一个粒度的前部块数。
    int gran;  // 每个粒度包含的空间位置数。
};

// 均衡划分 M 轴；total 为位置总数，maxRows 为单块位置上限，gran 为块长粒度。
FC2D_GEOM_CE TileSpec MakeTileSpec(int total, int maxRows, int gran)
{
    TileSpec tiles = {0, 0, 0, gran};
    if (total <= 0 || gran <= 0 || maxRows < gran || (total % gran) != 0) {
        return tiles;
    }
    const int totalUnits = total / gran;
    const int maxTileUnits = maxRows / gran;
    const int tileCount = CeilDiv(totalUnits, maxTileUnits);
    if (tileCount > MAX_M_TILES) {
        return tiles;
    }
    tiles.n = tileCount;
    tiles.base = totalUnits / tileCount;
    tiles.extra = totalUnits % tileCount;
    return tiles;
}

// 返回指定块的长度；tiles 为均衡划分，index 为块索引。
FC2D_GEOM_CE int TileRows(const TileSpec& tiles, int index)
{
    return (tiles.base + (index < tiles.extra ? 1 : 0)) * tiles.gran;
}
// 返回指定块的起点；tiles 为均衡划分，index 为块索引。
FC2D_GEOM_CE int TileRow0(const TileSpec& tiles, int index)
{
    return (tiles.base * index + MinI(index, tiles.extra)) * tiles.gran;
}

struct FmLoadSpec {
    int n;      // 输入通道搬入批次数。
    int firstC; // 首批搬入的通道数。
};

// 选择输入分批搬入；inputChannels/channelBlock 为通道数/分形，firstTileK 为首个 K 块长度，
// firstWeightSegments 为权重段数，columnSegmentCount 为输出列段数，firstMTileCount 为第一层 M 块数。
FC2D_GEOM_CE FmLoadSpec MakeFmLoadSpec(int inputChannels, int channelBlock, int firstTileK, int firstWeightSegments,
                                       int columnSegmentCount, int firstMTileCount)
{
    FmLoadSpec loadSpec = {1, inputChannels};
    if (columnSegmentCount > 1 && firstWeightSegments == 1 && firstMTileCount == 1 && firstTileK >= channelBlock &&
        firstTileK < inputChannels && (firstTileK % channelBlock) == 0) {
        loadSpec.n = 2;
        loadSpec.firstC = firstTileK;
    }
    return loadSpec;
}

struct FlatGeom {
    int n;               // 批次数。
    int ci;              // 输入通道数。
    int hi;              // 输入高度。
    int wi;              // 输入宽度。
    int cout1;           // 第一层输出通道数。
    int cout2;           // 第二层输出通道数。
    int kh1;             // 第一层卷积核高度。
    int kw1;             // 第一层卷积核宽度。
    int kh2;             // 第二层卷积核高度。
    int kw2;             // 第二层卷积核宽度。
    int stride1;         // 第一层空间步长。
    int stride2;         // 第二层空间步长。
    int padH1;           // 第一层上下对称补边。
    int padW1;           // 第一层左右对称补边。
    int padH2;           // 第二层上下对称补边。
    int padW2;           // 第二层左右对称补边。
    int elemBytes;       // 输入元素字节数。
    int weightElemBytes; // 权重元素字节数。
    int midElemBytes;    // 中间结果元素字节数。
    int biasElemBytes;   // 偏置元素字节数。
    int c0;              // 输入通道分形大小。
    int weightC0;        // 权重通道分形大小。
    int midC0;           // 中间结果通道分形大小。
    int ho1;             // 第一层输出高度。
    int wo1;             // 第一层输出宽度。
    int ho2;             // 第二层输出高度。
    int wo2;             // 第二层输出宽度。
    int k1;              // 第一层完整归约 K 长度。
    int k2;              // 第二层完整归约 K 长度。
    int hb;              // 每个输出行带的行数。
    int nchunk;          // 每个批次的行带数。
    int chunkTotal;      // 全部输出块数。
    int chunksPerCore;   // 每核分配的连续块数。
    int wseg;            // 每个输出列段的宽度。
    int nwseg;           // 每个行带的列段数。
    int midRows;         // 最大中间窗口行数（含补边）。
    int mMax1;           // 第一层 M 块位置数上限。
    int mMax2;           // 第二层 M 块位置数上限。
    int tileK1;          // 第一层单次矩阵乘 K 长度。
    int tileK2;          // 第二层单次矩阵乘 K 长度。
    int l0bChunks1;      // 第一层 L0B 权重段数。
    int l0bChunkK1;      // 第一层每个 L0B 段 K 长度。
    int tilesPerChunk1;  // 第一层每段矩阵乘次数。
    int l0bChunks2;      // 第二层 L0B 权重段数。
    int l0bChunkK2;      // 第二层每个 L0B 段 K 长度。
    int tilesPerChunk2;  // 第二层每段矩阵乘次数。
    int l0bPP1;          // 第一层是否交替使用 L0B 槽。
    int l0bPP2;          // 第二层是否交替使用 L0B 槽。
    int l0bSlot1Elems;   // 第一层 L0B 单槽元素数。
    int l0bSlot2Elems;   // 第二层 L0B 单槽元素数。
    int l0bElems;        // L0B 总元素容量。
    int l0cElems;        // L0C 总累加元素容量。
    int l0cStride;       // 第一层 L0C 交替槽元素跨距（0 为单槽）。
    int l0cStride2;      // 第二层 L0C 交替槽元素跨距（0 为单槽）。
    int w1Stage;         // 第一层权重 L1 策略（0 完整区，1 环形区）。
    int w2Stage;         // 第二层权重 L1 策略（0 完整区，1 环形区）。
    int fmAddr;          // 输入 L1 字节地址。
    int fmElems;         // 输入 L1 元素容量。
    int midAddr;         // 中间结果 L1 字节地址。
    int midElems;        // 中间结果 L1 元素容量。
    int w1Addr;          // 第一层权重 L1 字节地址。
    int w1Elems;         // 第一层权重 L1 元素容量。
    int w2Addr;          // 第二层权重 L1 字节地址。
    int w2Elems;         // 第二层权重 L1 元素容量。
    int b1Addr;          // 第一层偏置 L1 字节地址。
    int b2Addr;          // 第二层偏置 L1 字节地址。
    int dq2Addr;         // 第二层反量化表 L1 字节地址。
    int q1Addr;          // 第一层重量化表 L1 字节地址。
    int q2Addr;          // 第二层重量化表 L1 字节地址。
    int xPlane;          // 输入单个批次元素跨度。
    int yPlane;          // 输出单个批次元素跨度。
    int yPlaneM;         // 输出单通道空间元素跨度。
};

#define FC2D_FLAT_FIELDS(X)             \
    X(n, batch)                         \
    X(ci, cin)                          \
    X(hi, hin)                          \
    X(wi, win)                          \
    X(cout1, cout1)                     \
    X(cout2, cout2)                     \
    X(kh1, kh1)                         \
    X(kw1, kw1)                         \
    X(kh2, kh2)                         \
    X(kw2, kw2)                         \
    X(stride1, stride1)                 \
    X(stride2, stride2)                 \
    X(padH1, padH1)                     \
    X(padW1, padW1)                     \
    X(padH2, padH2)                     \
    X(padW2, padW2)                     \
    X(elemBytes, elemBytes)             \
    X(weightElemBytes, weightElemBytes) \
    X(midElemBytes, midElemBytes)       \
    X(biasElemBytes, biasElemBytes)     \
    X(c0, c0)                           \
    X(weightC0, weightC0)               \
    X(midC0, midC0)                     \
    X(ho1, ho1)                         \
    X(wo1, wo1)                         \
    X(ho2, ho2)                         \
    X(wo2, wo2)                         \
    X(k1, k1)                           \
    X(k2, k2)                           \
    X(hb, hb)                           \
    X(nchunk, nchunk)                   \
    X(wseg, wseg)                       \
    X(nwseg, nwseg)                     \
    X(chunkTotal, chunkTotal)           \
    X(chunksPerCore, chunksPerCore)     \
    X(midRows, midRows)                 \
    X(mMax1, mMax1)                     \
    X(mMax2, mMax2)                     \
    X(tileK1, tileK1)                   \
    X(tileK2, tileK2)                   \
    X(l0bChunks1, l0bChunks1)           \
    X(l0bChunkK1, l0bChunkK1)           \
    X(tilesPerChunk1, tilesPerChunk1)   \
    X(l0bChunks2, l0bChunks2)           \
    X(l0bChunkK2, l0bChunkK2)           \
    X(tilesPerChunk2, tilesPerChunk2)   \
    X(l0bPP1, l0bPP1)                   \
    X(l0bPP2, l0bPP2)                   \
    X(l0bSlot1Elems, l0bSlot1Elems)     \
    X(l0bSlot2Elems, l0bSlot2Elems)     \
    X(l0bElems, l0bElems)               \
    X(l0cElems, l0cElems)               \
    X(l0cStride, l0cStride)             \
    X(l0cStride2, l0cStride2)           \
    X(w1Stage, w1Stage)                 \
    X(w2Stage, w2Stage)                 \
    X(fmAddr, fmAddr)                   \
    X(fmElems, fmElems)                 \
    X(midAddr, midAddr)                 \
    X(midElems, midElems)               \
    X(w1Addr, w1Addr)                   \
    X(w1Elems, w1Elems)                 \
    X(w2Addr, w2Addr)                   \
    X(w2Elems, w2Elems)                 \
    X(b1Addr, b1Addr)                   \
    X(b2Addr, b2Addr)                   \
    X(dq2Addr, dq2Addr)                 \
    X(q1Addr, q1Addr)                   \
    X(q2Addr, q2Addr)                   \
    X(xPlane, xPlane)                   \
    X(yPlane, yPlane)                   \
    X(yPlaneM, yPlaneM)

struct BandLite {
    int aMid;           // 有效中间窗口起始行。
    int midReal;        // 有效中间窗口行数。
    int padT2;          // 第二层窗口上补边。
    int padB2;          // 第二层窗口下补边。
    int xRow0;          // 原始输入窗口起始行。
    int xRows;          // 原始输入有效行数。
    int padT1;          // 第一层窗口上补边。
    int padB1;          // 第一层窗口下补边。
    int outRow0;        // 输出行带起始行。
    int outCol0;        // 输出列段起始列。
    int outCols;        // 输出列段有效宽度。
    int midCol0;        // 有效中间窗口起始列。
    int wSub;           // 有效中间窗口宽度。
    int padL2;          // 第二层窗口左补边。
    int padR2;          // 第二层窗口右补边。
    int xCol0;          // 原始输入窗口起始列。
    int wIn;            // 原始输入有效宽度。
    int padL1;          // 第一层窗口左补边。
    int padR1;          // 第一层窗口右补边。
    int conv1MPerBand;  // 当前第一层展开矩阵 M 长度。
    int rowsPerRun;     // 第二层单次连续行组行数。
    int mRun;           // 第二层单次行组 M 长度。
    int nRun;           // 第二层连续行组数。
    TileSpec conv1Tile; // 第一层 M 块均衡划分。
    TileSpec conv2Tile; // 第二层 M 块均衡划分。
};

// 反推当前输入窗口及补边；geometry 为整体几何，rowBand/columnSegment 为输出块坐标，window 为计算结果。
FC2D_GEOM_FN void MakeBandLite(const FlatGeom& geometry, int rowBand, int columnSegment, BandLite& window)
{
    window.outRow0 = rowBand * geometry.hb;

    // 卷积反向窗口为 stride×首位置-pad，到 stride×末位置-pad+kernel；越界部分交给 Load3D 补零。
    const int rawMidRow = geometry.stride2 * geometry.hb * rowBand - geometry.padH2;
    window.aMid = MaxI(0, rawMidRow);
    window.padT2 = window.aMid - rawMidRow;
    const int requiredMidRows = geometry.midRows - window.padT2;
    window.midReal = MinI(requiredMidRows, geometry.ho1 - window.aMid);
    window.padB2 = geometry.midRows - window.padT2 - window.midReal;

    const int rawInputRow = geometry.stride1 * window.aMid - geometry.padH1;
    const int requiredInputRows = geometry.stride1 * (window.midReal - 1) + geometry.kh1;
    window.xRow0 = MaxI(0, rawInputRow);
    window.padT1 = window.xRow0 - rawInputRow;
    const int inputRowEnd = MinI(geometry.hi, rawInputRow + requiredInputRows);
    window.xRows = inputRowEnd - window.xRow0;
    window.padB1 = rawInputRow + requiredInputRows - inputRowEnd;

    window.outCol0 = columnSegment * geometry.wseg;
    window.outCols = MinI(geometry.wseg, geometry.wo2 - window.outCol0);

    const int rawMidColumnBegin = window.outCol0 * geometry.stride2 - geometry.padW2;
    const int rawMidColumnEnd = (window.outCol0 + window.outCols - 1) * geometry.stride2 - geometry.padW2 +
                                geometry.kw2;
    window.midCol0 = MaxI(0, rawMidColumnBegin);
    const int midColumnEnd = MinI(geometry.wo1, rawMidColumnEnd);
    window.padL2 = window.midCol0 - rawMidColumnBegin;
    window.padR2 = rawMidColumnEnd - midColumnEnd;
    window.wSub = midColumnEnd - window.midCol0;

    const int rawInputColumnBegin = window.midCol0 * geometry.stride1 - geometry.padW1;
    const int rawInputColumnEnd = (midColumnEnd - 1) * geometry.stride1 - geometry.padW1 + geometry.kw1;
    window.xCol0 = MaxI(0, rawInputColumnBegin);
    const int inputColumnEnd = MinI(geometry.wi, rawInputColumnEnd);
    window.padL1 = window.xCol0 - rawInputColumnBegin;
    window.padR1 = rawInputColumnEnd - inputColumnEnd;
    window.wIn = inputColumnEnd - window.xCol0;

    window.conv1MPerBand = window.midReal * window.wSub;
    window.conv1Tile = MakeTileSpec(window.conv1MPerBand, geometry.mMax1, 1);

    // 完整输出宽度可连续跨行；切列后按单行执行，避免把行间空隙误当作连续 M。
    window.rowsPerRun = (window.outCols == geometry.wo2) ? geometry.hb : 1;
    window.mRun = window.rowsPerRun * window.outCols;
    window.nRun = geometry.hb / window.rowsPerRun;
    window.conv2Tile = MakeTileSpec(window.mRun, geometry.mMax2, 1);
}

} // namespace FusedMultiConv2dShape

#endif


namespace FusedMultiConv2dShape {

#define FC2D_GEOM_REV "geom14"

constexpr int L0B_BYTES = 64 * 1024;
constexpr int L0B_SLOT_BYTES = L0B_BYTES / 2;
constexpr int L0C_BYTES = 256 * 1024;
constexpr int L0C_ELEMS = L0C_BYTES / 4;
constexpr int L1_SEG_ALIGN = 512;
constexpr int MTE1_COST_FACTOR = 190;
constexpr int MTE2_COST_FACTOR = 120;
constexpr int MTE2_ROW_COST = 8;
constexpr int FIX_L1_COST_FACTOR = 290;
constexpr int FIX_GM_COST_FACTOR = 105;
constexpr int FIX_GM_REQUEST_COST_X2 = 19;
constexpr int SCALAR_TILE_COST = 420;
constexpr int SCALAR_K_COST = 256;
constexpr int SCALAR_WEIGHT_SEGMENT_COST = 64;
constexpr int SCALAR_BASE_COST = 4000;

// 根据输入宽度选择默认偏置宽度；elemBytes 为输入元素字节数。
constexpr int BiasElemBytes(int elemBytes) { return elemBytes == 2 ? 2 : 4; }

struct ConvTilingInputs {
    int batchSize;                    // 批次数
    int inputChannels;                // 输入通道数
    int inputHeight;                  // 输入高度
    int inputWidth;                   // 输入宽度
    int outputChannels1;              // 第一层输出通道数
    int outputChannels2;              // 第二层输出通道数
    int kernelHeight1;                // 第一层卷积核高
    int kernelWidth1;                 // 第一层卷积核宽
    int stride1;                      // 第一层高宽公共步长
    int stride2;                      // 第二层高宽公共步长
    int paddingHeight1;               // 第一层上下对称补边
    int paddingWidth1;                // 第一层左右对称补边
    int paddingHeight2;               // 第二层上下对称补边
    int paddingWidth2;                // 第二层左右对称补边
    int inputElementBytes;            // 输入元素字节数
    int hasDequantScale2 = 0;         // 第二层是否使用反量化表
    int weightElementBytes = 0;       // 权重元素字节数，零时同输入
    int intermediateElementBytes = 0; // 中间元素字节数，零时同输入
    int biasElementBytes = 0;         // 偏置元素字节数，零时按输入推导
    int hasQuantScale1 = 0;           // 第一层是否使用量化表
    int hasQuantScale2 = 0;           // 第二层是否使用量化表
    int kernelHeight2 = 0;            // 第二层卷积核高，零时同第一层
    int kernelWidth2 = 0;             // 第二层卷积核宽，零时同第一层
};

// 读取权重元素宽度；inputs 为卷积配置，未指定时沿用输入宽度。
constexpr int WeightElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.weightElementBytes == 0 ? inputs.inputElementBytes : inputs.weightElementBytes;
}
// 读取中间元素宽度；inputs 为卷积配置，未指定时沿用输入宽度。
constexpr int MidElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.intermediateElementBytes == 0 ? inputs.inputElementBytes : inputs.intermediateElementBytes;
}
// 读取偏置元素宽度；inputs 为卷积配置，未指定时按输入类型推导。
constexpr int ParamBiasElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.biasElementBytes == 0 ? BiasElemBytes(inputs.inputElementBytes) : inputs.biasElementBytes;
}

// 读取第二层卷积核高；inputs 为卷积配置，零值时沿用第一层。
constexpr int Kh2(const ConvTilingInputs& inputs)
{
    return inputs.kernelHeight2 == 0 ? inputs.kernelHeight1 : inputs.kernelHeight2;
}
// 读取第二层卷积核宽；inputs 为卷积配置，零值时沿用第一层。
constexpr int Kw2(const ConvTilingInputs& inputs)
{
    return inputs.kernelWidth2 == 0 ? inputs.kernelWidth1 : inputs.kernelWidth2;
}

enum Reject {
    RJ_OK = 0,
    RJ_ELEM_BYTES,
    RJ_SHAPE_POSITIVE,
    RJ_CHANNEL_C0,
    RJ_OUT_EMPTY,
    RJ_HB_DIVIDE,
    RJ_MIDROWS,
    RJ_TILE_K,
    RJ_M_TILES,
    RJ_L0C,
    RJ_L0B,
    RJ_BT,
    RJ_L1,
    RJ_U16,
    RJ_NO_HB,
    RJ_WSEG,
};

// 返回分块失败原因；reason 为几何校验返回码。
inline const char* RejectText(int reason)
{
    switch (reason) {
        case RJ_OK:
            return "ok";
        case RJ_ELEM_BYTES:
            return "element width must be 1 (int8) or 2 (fp16)";
        case RJ_SHAPE_POSITIVE:
            return "dimensions must be positive";
        case RJ_CHANNEL_C0:
            return "input and intermediate channels must satisfy C0 alignment";
        case RJ_OUT_EMPTY:
            return "convolution output is empty";
        case RJ_HB_DIVIDE:
            return "band height must divide output height";
        case RJ_MIDROWS:
            return "band has no valid intermediate rows";
        case RJ_TILE_K:
            return "no valid K tile fits L0A";
        case RJ_M_TILES:
            return "M tile count exceeds MAX_M_TILES";
        case RJ_L0C:
            return "M tile exceeds L0C capacity";
        case RJ_L0B:
            return "weight segments exceed L0B capacity";
        case RJ_BT:
            return "bias channels exceed BT slot capacity";
        case RJ_L1:
            return "L1 allocation exceeds available capacity";
        case RJ_U16:
            return "instruction extent exceeds the 16-bit limit";
        case RJ_NO_HB:
            return "no band height satisfies all constraints";
        case RJ_WSEG:
            return "column segment count must be within output width";
        default:
            return "unknown rejection";
    }
}

struct ConvTilingGeometry {
    ConvTilingInputs inputs;        // 两层卷积输入配置
    int inputC0;                    // 输入通道打包粒度
    int weightC0;                   // 权重通道打包粒度
    int intermediateC0;             // 中间通道打包粒度
    int inputWeightGroups;          // 输入通道的权重分组数
    int intermediateWeightGroups;   // 中间通道的权重分组数
    int outputHeight1;              // 第一层输出高
    int outputWidth1;               // 第一层输出宽
    int outputHeight2;              // 第二层输出高
    int outputWidth2;               // 第二层输出宽
    int reductionLength1;           // 第一层完整归约长度
    int reductionLength2;           // 第二层完整归约长度
    int bandHeight;                 // 输出行带高
    int heightBandCount;            // 每图高度方向行带数
    int segmentWidth;               // 输出列段宽度上限
    int widthSegmentCount;          // 每行带列段数
    int intermediateWidthMax;       // 中间窗口宽度上限
    int inputWindowWidthMax;        // 输入窗口宽度上限
    int totalBandTasks;             // 全部批次的行带列段任务数
    int intermediateRows;           // 第二层所需中间行跨度
    int inputWindowRows;            // 输入窗口有效行数上限
    int intermediatePositionsMax;   // 中间窗口位置数上限
    int tileK1;                     // 第一层单次归约长度
    int tileK2;                     // 第二层单次归约长度
    int tileMMax1;                  // 第一层单次 M 上限
    int tileMMax2;                  // 第二层单次 M 上限
    int weightSegmentCount1;        // 第一层权重段数
    int weightSegmentK1;            // 第一层单段归约长度
    int kTilesPerWeightSegment1;    // 第一层每权重段 K 分块数
    int weightSegmentCount2;        // 第二层权重段数
    int weightSegmentK2;            // 第二层单段归约长度
    int kTilesPerWeightSegment2;    // 第二层每权重段 K 分块数
    int l0bPingPong1;               // 第一层 L0B ping/pong 开关
    int l0bPingPong2;               // 第二层 L0B ping/pong 开关
    int l0bSlotElements1;           // 第一层 L0B 槽跨度，单位元素
    int l0bSlotElements2;           // 第二层 L0B 槽跨度，单位元素
    int l0bElements;                // 共享 L0B 元素容量
    int l0cElements;                // 共享 L0C 累加元素容量
    int l0cSlotStride;              // 第一层 L0C 槽跨度，单位元素
    int l0cSlotStride2;             // 第二层 L0C 槽跨度，单位元素
    int weightRing1;                // 第一层权重循环搬运开关
    int weightRing2;                // 第二层权重循环搬运开关
    int inputBufferElements;        // L1 输入缓冲元素数
    int intermediateBufferElements; // L1 中间缓冲元素数
    int weightBufferElements1;      // L1 第一层权重元素数
    int weightBufferElements2;      // L1 第二层权重元素数
    int inputBufferAddress;         // L1 输入缓冲字节地址
    int intermediateBufferAddress;  // L1 中间缓冲字节地址
    int weightBufferAddress1;       // L1 第一层权重字节地址
    int weightBufferAddress2;       // L1 第二层权重字节地址
    int biasBufferAddress1;         // L1 第一层偏置字节地址
    int biasBufferAddress2;         // L1 第二层偏置字节地址
    int biasBufferBytes1;           // 第一层偏置分配字节数
    int biasBufferBytes2;           // 第二层偏置分配字节数
    int dequantBufferAddress2;      // L1 第二层反量化表字节地址
    int dequantBufferBytes2;        // 第二层反量化表分配字节数
    int quantBufferAddress1;        // L1 第一层量化表字节地址
    int quantBufferBytes1;          // 第一层量化表分配字节数
    int quantBufferAddress2;        // L1 第二层量化表字节地址
    int quantBufferBytes2;          // 第二层量化表分配字节数
    int l1UsedBytes;                // L1 分配总字节数
    int inputImageElements;         // 每图输入元素数
    int outputImageElements;        // 每图输出元素数
    int filterStorageK1;            // 第一层 FRACTAL_Z 外层 K 维
    int filterStorageN1;            // 第一层 FRACTAL_Z 外层 N 维
    int filterStorageK2;            // 第二层 FRACTAL_Z 外层 K 维
    int filterStorageN2;            // 第二层 FRACTAL_Z 外层 N 维
};

struct ComputeTileChoice {
    int tileK;        // 单次归约长度
    int maxPositions; // 单次 M 上限
    int tileCount;    // M 分块数
    bool valid;       // 是否找到合法分块
};

// 选择 K 步长及 M 上限；reductionLength/kAlignment 为归约长度/粒度，inputBytes/weightBytes 为元素宽度，
// outputChannels 为输出通道数，positions 为待划分的位置数。
inline ComputeTileChoice PickCompactMTile(int reductionLength, int kAlignment, int inputBytes, int outputChannels,
                                          int positions, int weightBytes)
{
    ComputeTileChoice best;
    best.tileK = 0;
    best.maxPositions = 0;
    best.tileCount = 0;
    best.valid = false;
    bool bestPingPong = false;
    if (reductionLength <= 0 || kAlignment <= 0 || outputChannels <= 0 || positions <= 0 || weightBytes <= 0) {
        return best;
    }

    const int accumulatorLimit = L0C_ELEMS / Align(outputChannels, MMAD_M0);

    const int weightKLimit = L0B_BYTES / (Align(outputChannels, MMAD_M0) * weightBytes);
    const int pingPongKLimit = L0B_SLOT_BYTES / (Align(outputChannels, MMAD_M0) * weightBytes);
    for (int tileK = kAlignment; tileK <= reductionLength; tileK += kAlignment) {
        if ((reductionLength % tileK) != 0) {
            continue;
        }
        if (tileK > weightKLimit) {
            continue;
        }

        const int operandLimit = L0A_SLOT_BYTES / (tileK * inputBytes);

        int maxPositions = MinI(operandLimit, accumulatorLimit) / MMAD_M0 * MMAD_M0;
        if (maxPositions <= 0) {
            continue;
        }
        if (maxPositions > positions) {
            maxPositions = positions;
        }
        const int tileCount = CeilDiv(positions, maxPositions);
        if (tileCount > MAX_M_TILES) {
            continue;
        }

        // M 分块数相同时，非整套常驻权重优先保留 L0B ping/pong，再选择较大的 K。
        const bool pingPong = reductionLength > weightKLimit && tileK <= pingPongKLimit;
        if (!best.valid || tileCount < best.tileCount ||
            (tileCount == best.tileCount &&
             ((pingPong && !bestPingPong) || (pingPong == bestPingPong && tileK > best.tileK)))) {
            best.tileK = tileK;
            best.maxPositions = maxPositions;
            best.tileCount = tileCount;
            best.valid = true;
            bestPingPong = pingPong;
        }
    }
    return best;
}

// 选择整分 K 的权重段；outputChannels/reductionLength 为通道/归约长度，weightBytes 为元素宽度，
// tileK 为归约步长，budgetBytes 为容量，segmentCount/segmentK 接收段数/段长。
inline bool PickL0bChunksIn(int outputChannels, int reductionLength, int weightBytes, int tileK, int budgetBytes,
                            int& segmentCount, int& segmentK)
{
    const int alignedChannels = Align(outputChannels, MMAD_M0);
    for (int count = 1; count <= reductionLength / tileK; ++count) {
        if ((reductionLength % count) != 0) {
            continue;
        }
        const int candidateK = reductionLength / count;
        if ((candidateK % tileK) != 0) {
            continue;
        }
        if (alignedChannels * candidateK * weightBytes <= budgetBytes) {
            segmentCount = count;
            segmentK = candidateK;
            return true;
        }
    }
    return false;
}

// 优先选择 L0B ping/pong；outputChannels/reductionLength 为通道/归约长度，weightBytes/tileK 为元素宽度/步长，
// segmentCount/segmentK/pingPong 接收段数/段长/缓冲开关。
inline bool PickL0bChunks(int outputChannels, int reductionLength, int weightBytes, int tileK, int& segmentCount,
                          int& segmentK, int& pingPong)
{
    // 整套权重能放入 L0B 时只搬入一次，无需为 ping/pong 拆段。
    if ((long long)Align(outputChannels, MMAD_M0) * reductionLength * weightBytes <= L0B_BYTES) {
        segmentCount = 1;
        segmentK = reductionLength;
        pingPong = 0;
        return true;
    }
    if (PickL0bChunksIn(outputChannels, reductionLength, weightBytes, tileK, L0B_SLOT_BYTES, segmentCount, segmentK)) {
        pingPong = 1;
        return true;
    }
    pingPong = 0;
    return PickL0bChunksIn(outputChannels, reductionLength, weightBytes, tileK, L0B_BYTES, segmentCount, segmentK);
}

// 估算一层分块的计算、搬运和调度代价；tiles 为实际 M 划分，tileK 为 K 步长，
// reductionLength/inputBytes/outputChannels/weightBytes 为归约及数据宽度，segmentCount 为权重段数，
// pingPong/weightRing 为缓冲策略，outputToGm 指示最终写出。
inline long long EstimateComputeTileCost(const TileSpec& tiles, int tileK, int reductionLength, int inputBytes,
                                         int outputChannels, int weightBytes, int segmentCount, bool pingPong,
                                         bool outputToGm, bool weightRing)
{
    long long alignedPositions = 0;
    for (int tileIndex = 0; tileIndex < tiles.n; ++tileIndex) {
        alignedPositions += Align(TileRows(tiles, tileIndex), MMAD_M0);
    }
    const int alignedChannels = Align(outputChannels, MMAD_M0);
    const int cubeKFactor = inputBytes == 2 ? 16 : 32;
    const long long dispatches = (long long)tiles.n * (reductionLength / tileK);
    const long long weightLoads = segmentCount == 1 ? 1 : tiles.n;
    const long long featureCost = alignedPositions * reductionLength * inputBytes / MTE1_COST_FACTOR;
    const long long weightCost = weightLoads * alignedChannels * reductionLength * weightBytes / MTE1_COST_FACTOR;
    const long long cubeCost = alignedPositions / MMAD_M0 * (alignedChannels / MMAD_M0) *
                               (reductionLength / cubeKFactor);
    const long long fixCost = outputToGm ? (long long)tiles.n * alignedChannels * FIX_GM_REQUEST_COST_X2 / 2 :
                                           alignedPositions * alignedChannels * 4 / FIX_L1_COST_FACTOR;
    // L0B ping/pong 允许后续权重搬运与计算重叠；单槽则单独计入权重搬运。
    const long long operandCost = featureCost + (pingPong ? weightCost : 0);
    const long long pipelineCost = (cubeCost > operandCost ? cubeCost : operandCost) + (pingPong ? 0 : weightCost);
    const long long ringCost = weightRing ?
                                   weightLoads * alignedChannels * reductionLength * weightBytes / MTE2_COST_FACTOR :
                                   0;
    return pipelineCost + ringCost + fixCost + (long long)tiles.n * SCALAR_TILE_COST + dispatches * SCALAR_K_COST +
           weightLoads * segmentCount * SCALAR_WEIGHT_SEGMENT_COST;
}

// 联合选择 M/K 分块；归约、元素宽度及通道决定容量，positions 为位置数，
// outputToGm 指示最终输出，weightRing 指示权重需从 GM 循环搬运。
inline ComputeTileChoice PickComputeTile(int reductionLength, int kAlignment, int inputBytes, int outputChannels,
                                         int positions, int weightBytes, bool outputToGm, bool weightRing)
{
    ComputeTileChoice best{};
    long long bestCost = 0;
    if (reductionLength <= 0 || kAlignment <= 0 || outputChannels <= 0 || positions <= 0 || weightBytes <= 0) {
        return best;
    }
    const int accumulatorLimit = L0C_ELEMS / Align(outputChannels, MMAD_M0);
    const int weightKLimit = L0B_BYTES / (Align(outputChannels, MMAD_M0) * weightBytes);
    for (int tileK = kAlignment; tileK <= MinI(reductionLength, weightKLimit); tileK += kAlignment) {
        if ((reductionLength % tileK) != 0) {
            continue;
        }
        const int operandLimit = L0A_SLOT_BYTES / (tileK * inputBytes);
        const int maxPositions = MinI(MinI(operandLimit, accumulatorLimit) / MMAD_M0 * MMAD_M0, positions);
        const TileSpec tiles = MakeTileSpec(positions, maxPositions, 1);
        int segmentCount = 0, segmentK = 0, pingPong = 0;
        if (tiles.n == 0 ||
            !PickL0bChunks(outputChannels, reductionLength, weightBytes, tileK, segmentCount, segmentK, pingPong)) {
            continue;
        }
        // 逐块计算 16 对齐开销，并计入 K 循环、权重重搬及输出请求，避免只追求较少 M 块。
        const long long cost = EstimateComputeTileCost(tiles, tileK, reductionLength, inputBytes, outputChannels,
                                                       weightBytes, segmentCount, pingPong != 0, outputToGm,
                                                       weightRing);
        if (!best.valid || cost < bestCost || (cost == bestCost && tileK > best.tileK)) {
            best = {tileK, maxPositions, tiles.n, true};
            bestCost = cost;
        }
    }
    return best;
}

// 写入 kernel 共用参数；geometry 为推导结果，chunksPerCore 为每核任务上限，flat 接收扁平字段。
inline void Flatten(const ConvTilingGeometry& geometry, int chunksPerCore, FlatGeom& flat)
{
    flat.n = geometry.inputs.batchSize;
    flat.ci = geometry.inputs.inputChannels;
    flat.hi = geometry.inputs.inputHeight;
    flat.wi = geometry.inputs.inputWidth;
    flat.cout1 = geometry.inputs.outputChannels1;
    flat.cout2 = geometry.inputs.outputChannels2;
    flat.kh1 = geometry.inputs.kernelHeight1;
    flat.kw1 = geometry.inputs.kernelWidth1;
    flat.kh2 = Kh2(geometry.inputs);
    flat.kw2 = Kw2(geometry.inputs);
    flat.stride1 = geometry.inputs.stride1;
    flat.stride2 = geometry.inputs.stride2;
    flat.padH1 = geometry.inputs.paddingHeight1;
    flat.padW1 = geometry.inputs.paddingWidth1;
    flat.padH2 = geometry.inputs.paddingHeight2;
    flat.padW2 = geometry.inputs.paddingWidth2;
    flat.elemBytes = geometry.inputs.inputElementBytes;
    flat.weightElemBytes = WeightElemBytes(geometry.inputs);
    flat.midElemBytes = MidElemBytes(geometry.inputs);
    flat.biasElemBytes = ParamBiasElemBytes(geometry.inputs);

    flat.c0 = geometry.inputC0;
    flat.weightC0 = geometry.weightC0;
    flat.midC0 = geometry.intermediateC0;
    flat.ho1 = geometry.outputHeight1;
    flat.wo1 = geometry.outputWidth1;
    flat.ho2 = geometry.outputHeight2;
    flat.wo2 = geometry.outputWidth2;
    flat.k1 = geometry.reductionLength1;
    flat.k2 = geometry.reductionLength2;

    flat.hb = geometry.bandHeight;
    flat.nchunk = geometry.heightBandCount;
    flat.wseg = geometry.segmentWidth;
    flat.nwseg = geometry.widthSegmentCount;
    flat.chunkTotal = geometry.totalBandTasks;
    flat.chunksPerCore = chunksPerCore;

    flat.midRows = geometry.intermediateRows;
    flat.mMax1 = geometry.tileMMax1;
    flat.mMax2 = geometry.tileMMax2;
    flat.tileK1 = geometry.tileK1;
    flat.tileK2 = geometry.tileK2;
    flat.l0bChunks1 = geometry.weightSegmentCount1;
    flat.l0bChunkK1 = geometry.weightSegmentK1;
    flat.tilesPerChunk1 = geometry.kTilesPerWeightSegment1;
    flat.l0bChunks2 = geometry.weightSegmentCount2;
    flat.l0bChunkK2 = geometry.weightSegmentK2;
    flat.tilesPerChunk2 = geometry.kTilesPerWeightSegment2;
    flat.l0bPP1 = geometry.l0bPingPong1;
    flat.l0bPP2 = geometry.l0bPingPong2;
    flat.l0bSlot1Elems = geometry.l0bSlotElements1;
    flat.l0bSlot2Elems = geometry.l0bSlotElements2;
    flat.l0bElems = geometry.l0bElements;
    flat.l0cElems = geometry.l0cElements;
    flat.l0cStride = geometry.l0cSlotStride;
    flat.l0cStride2 = geometry.l0cSlotStride2;
    flat.w1Stage = geometry.weightRing1;
    flat.w2Stage = geometry.weightRing2;

    flat.fmAddr = geometry.inputBufferAddress;
    flat.fmElems = geometry.inputBufferElements;
    flat.midAddr = geometry.intermediateBufferAddress;
    flat.midElems = geometry.intermediateBufferElements;
    flat.w1Addr = geometry.weightBufferAddress1;
    flat.w1Elems = geometry.weightBufferElements1;
    flat.w2Addr = geometry.weightBufferAddress2;
    flat.w2Elems = geometry.weightBufferElements2;
    flat.b1Addr = geometry.biasBufferAddress1;
    flat.b2Addr = geometry.biasBufferAddress2;
    flat.dq2Addr = geometry.dequantBufferAddress2;
    flat.q1Addr = geometry.quantBufferAddress1;
    flat.q2Addr = geometry.quantBufferAddress2;

    flat.xPlane = geometry.inputImageElements;
    flat.yPlane = geometry.outputImageElements;
    flat.yPlaneM = geometry.outputHeight2 * geometry.outputWidth2;
}

// 分配 L1 字节地址；inputs 为通道配置，geometry 为缓冲大小及输出地址，
// xBytes/midBytes/weightBytes/biasBytes 分别为输入/中间/权重/偏置元素宽度。
inline void LayOutL1(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry, int xBytes, int midBytes,
                     int weightBytes, int biasBytes)
{
    geometry.inputBufferAddress = 0;
    geometry.intermediateBufferAddress = Align(geometry.inputBufferAddress + geometry.inputBufferElements * xBytes,
                                               L1_SEG_ALIGN);
    geometry.weightBufferAddress1 = Align(
        geometry.intermediateBufferAddress + geometry.intermediateBufferElements * midBytes, L1_SEG_ALIGN);
    geometry.weightBufferAddress2 = Align(geometry.weightBufferAddress1 + geometry.weightBufferElements1 * weightBytes,
                                          L1_SEG_ALIGN);
    geometry.biasBufferAddress1 = Align(geometry.weightBufferAddress2 + geometry.weightBufferElements2 * weightBytes,
                                        L1_SEG_ALIGN);
    geometry.biasBufferBytes1 = Align(inputs.outputChannels1 * biasBytes, L1_SEG_ALIGN);
    geometry.biasBufferAddress2 = geometry.biasBufferAddress1 + geometry.biasBufferBytes1;
    geometry.biasBufferBytes2 = Align(inputs.outputChannels2 * biasBytes, L1_SEG_ALIGN);

    geometry.dequantBufferAddress2 = geometry.biasBufferAddress2 + geometry.biasBufferBytes2;
    geometry.dequantBufferBytes2 = (inputs.hasDequantScale2 != 0) ?
                                       Align(inputs.outputChannels2 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                       0;
    geometry.quantBufferAddress1 = geometry.dequantBufferAddress2 + geometry.dequantBufferBytes2;
    geometry.quantBufferBytes1 = (inputs.hasQuantScale1 != 0) ?
                                     Align(inputs.outputChannels1 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                     0;
    geometry.quantBufferAddress2 = geometry.quantBufferAddress1 + geometry.quantBufferBytes1;
    geometry.quantBufferBytes2 = (inputs.hasQuantScale2 != 0) ?
                                     Align(inputs.outputChannels2 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                     0;
    geometry.l1UsedBytes = geometry.quantBufferAddress2 + geometry.quantBufferBytes2;
}

// 校验分块推导的基础输入；inputs 为两层卷积配置，hb 为候选行带高。
inline int ValidateDeriveInputs(const ConvTilingInputs& inputs, int hb)
{
    const int xBytes = inputs.inputElementBytes;
    const int weightBytes = WeightElemBytes(inputs);
    const int midBytes = MidElemBytes(inputs);
    const int biasBytes = ParamBiasElemBytes(inputs);
    if ((xBytes != 1 && xBytes != 2) || (weightBytes != 1 && weightBytes != 2) || (midBytes != 1 && midBytes != 2) ||
        (biasBytes != 2 && biasBytes != 4)) {
        return RJ_ELEM_BYTES;
    }
    if (inputs.batchSize <= 0 || inputs.inputChannels <= 0 || inputs.inputHeight <= 0 || inputs.inputWidth <= 0 ||
        inputs.outputChannels1 <= 0 || inputs.outputChannels2 <= 0 || inputs.kernelHeight1 <= 0 ||
        inputs.kernelWidth1 <= 0 || Kh2(inputs) <= 0 || Kw2(inputs) <= 0 || inputs.stride1 <= 0 ||
        inputs.stride2 <= 0 || inputs.paddingHeight1 < 0 || inputs.paddingWidth1 < 0 || inputs.paddingHeight2 < 0 ||
        inputs.paddingWidth2 < 0 || hb <= 0) {
        return RJ_SHAPE_POSITIVE;
    }
    return RJ_OK;
}

// 推导通道分组、两层输出和归约长度；inputs 为输入配置，geometry 接收几何结果。
inline int DeriveConvolutionGeometry(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    geometry.inputs = inputs;
    geometry.inputC0 = 32 / inputs.inputElementBytes;
    geometry.weightC0 = 32 / WeightElemBytes(inputs);
    geometry.intermediateC0 = 32 / MidElemBytes(inputs);
    if ((inputs.inputChannels % geometry.inputC0) != 0 || (inputs.inputChannels % geometry.weightC0) != 0 ||
        (inputs.outputChannels1 % geometry.intermediateC0) != 0 || (inputs.outputChannels1 % geometry.weightC0) != 0) {
        return RJ_CHANNEL_C0;
    }
    geometry.inputWeightGroups = inputs.inputChannels / geometry.weightC0;
    geometry.intermediateWeightGroups = inputs.outputChannels1 / geometry.weightC0;
    geometry.outputHeight1 = (inputs.inputHeight + 2 * inputs.paddingHeight1 - inputs.kernelHeight1) / inputs.stride1 +
                             1;
    geometry.outputWidth1 = (inputs.inputWidth + 2 * inputs.paddingWidth1 - inputs.kernelWidth1) / inputs.stride1 + 1;
    if (geometry.outputHeight1 <= 0 || geometry.outputWidth1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    geometry.outputHeight2 = (geometry.outputHeight1 + 2 * inputs.paddingHeight2 - Kh2(inputs)) / inputs.stride2 + 1;
    geometry.outputWidth2 = (geometry.outputWidth1 + 2 * inputs.paddingWidth2 - Kw2(inputs)) / inputs.stride2 + 1;
    if (geometry.outputHeight2 <= 0 || geometry.outputWidth2 <= 0) {
        return RJ_OUT_EMPTY;
    }
    geometry.reductionLength1 = inputs.inputChannels * inputs.kernelHeight1 * inputs.kernelWidth1;
    geometry.reductionLength2 = inputs.outputChannels1 * Kh2(inputs) * Kw2(inputs);
    geometry.filterStorageK1 = geometry.inputWeightGroups * inputs.kernelHeight1 * inputs.kernelWidth1;
    geometry.filterStorageN1 = CeilDiv(inputs.outputChannels1, MMAD_M0);
    geometry.filterStorageK2 = geometry.intermediateWeightGroups * Kh2(inputs) * Kw2(inputs);
    geometry.filterStorageN2 = CeilDiv(inputs.outputChannels2, MMAD_M0);
    return RJ_OK;
}

// 反推输出行带所需的中间及输入窗口；hb/nwseg 为候选划分，geometry 为已知卷积几何。
inline int DeriveWindowGeometry(const ConvTilingInputs& inputs, int hb, int nwseg, ConvTilingGeometry& geometry)
{
    if ((geometry.outputHeight2 % hb) != 0) {
        return RJ_HB_DIVIDE;
    }
    if (nwseg <= 0 || nwseg > geometry.outputWidth2) {
        return RJ_WSEG;
    }
    geometry.bandHeight = hb;
    geometry.heightBandCount = geometry.outputHeight2 / hb;
    geometry.segmentWidth = CeilDiv(geometry.outputWidth2, nwseg);
    geometry.widthSegmentCount = CeilDiv(geometry.outputWidth2, geometry.segmentWidth);
    geometry.totalBandTasks = inputs.batchSize * geometry.heightBandCount * geometry.widthSegmentCount;
    // 连续输出的覆盖范围为步长乘间隔数，再加本层卷积核跨度。
    geometry.intermediateRows = inputs.stride2 * (hb - 1) + Kh2(inputs);
    const int midRowsCap = MinI(geometry.intermediateRows, geometry.outputHeight1);
    geometry.intermediateWidthMax = MinI(geometry.outputWidth1,
                                         (geometry.segmentWidth - 1) * inputs.stride2 + Kw2(inputs));
    geometry.inputWindowWidthMax = MinI(inputs.inputWidth,
                                        (geometry.intermediateWidthMax - 1) * inputs.stride1 + inputs.kernelWidth1);
    geometry.intermediatePositionsMax = midRowsCap * geometry.intermediateWidthMax;
    geometry.inputWindowRows = MinI(inputs.stride1 * (midRowsCap - 1) + inputs.kernelHeight1, inputs.inputHeight);
    if (geometry.intermediatePositionsMax > 65535 ||
        (long long)geometry.inputWindowRows * geometry.inputWindowWidthMax > 65535) {
        return RJ_U16;
    }
    return RJ_OK;
}

// 选择两层的 K 步长和 M 上限；inputs/geometry 为输入及窗口几何，mRunMax 为第二层单次 M 范围。
inline int SelectComputeTiles(const ConvTilingInputs& inputs, int mRunMax, ConvTilingGeometry& geometry)
{
    const int weightBytes = WeightElemBytes(inputs);
    const int kBlock1 = MaxI(geometry.inputC0, geometry.weightC0);
    const int kBlock2 = MaxI(geometry.intermediateC0, geometry.weightC0);
    if (Align(inputs.outputChannels1, MMAD_M0) * kBlock1 * weightBytes > L0B_BYTES ||
        Align(inputs.outputChannels2, MMAD_M0) * kBlock2 * weightBytes > L0B_BYTES) {
        return RJ_L0B;
    }
    const ComputeTileChoice tileChoice1 = PickCompactMTile(geometry.reductionLength1, kBlock1, inputs.inputElementBytes,
                                                           inputs.outputChannels1, geometry.intermediatePositionsMax,
                                                           weightBytes);
    const ComputeTileChoice tileChoice2 = PickCompactMTile(geometry.reductionLength2, kBlock2, MidElemBytes(inputs),
                                                           inputs.outputChannels2, mRunMax, weightBytes);
    if (!tileChoice1.valid || !tileChoice2.valid) {
        return RJ_TILE_K;
    }
    geometry.tileK1 = tileChoice1.tileK;
    geometry.tileMMax1 = tileChoice1.maxPositions;
    geometry.tileK2 = tileChoice2.tileK;
    geometry.tileMMax2 = tileChoice2.maxPositions;
    return RJ_OK;
}

// 配置权重段和 L0B 复用；inputs/geometry 为卷积配置及分块结果。
inline int SelectWeightSegments(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int weightBytes = WeightElemBytes(inputs);
    if (!PickL0bChunks(inputs.outputChannels1, geometry.reductionLength1, weightBytes, geometry.tileK1,
                       geometry.weightSegmentCount1, geometry.weightSegmentK1, geometry.l0bPingPong1) ||
        !PickL0bChunks(inputs.outputChannels2, geometry.reductionLength2, weightBytes, geometry.tileK2,
                       geometry.weightSegmentCount2, geometry.weightSegmentK2, geometry.l0bPingPong2)) {
        return RJ_L0B;
    }
    if (geometry.weightSegmentCount1 == 1) {
        geometry.l0bPingPong1 = 0;
    }
    if (geometry.weightSegmentCount2 == 1) {
        geometry.l0bPingPong2 = 0;
    }
    geometry.kTilesPerWeightSegment1 = geometry.weightSegmentK1 / geometry.tileK1;
    geometry.kTilesPerWeightSegment2 = geometry.weightSegmentK2 / geometry.tileK2;
    return RJ_OK;
}

// 检查边界行带的有效 M 划分；inputs/geometry 为卷积配置及候选几何，mRunMax 为第二层 M 范围。
inline int ValidateBandTiles(const ConvTilingInputs& inputs, int mRunMax, const ConvTilingGeometry& geometry)
{
    for (int bandIndex = 0; bandIndex < geometry.heightBandCount; ++bandIndex) {
        const int rawMidRow = inputs.stride2 * geometry.bandHeight * bandIndex - inputs.paddingHeight2;
        const int midRowBegin = MaxI(0, rawMidRow);
        const int topPadding2 = midRowBegin - rawMidRow;
        const int validMidRows = MinI(geometry.intermediateRows - topPadding2, geometry.outputHeight1 - midRowBegin);
        if (validMidRows <= 0) {
            return RJ_MIDROWS;
        }
        if (MakeTileSpec(validMidRows * geometry.intermediateWidthMax, geometry.tileMMax1, 1).n == 0) {
            return RJ_M_TILES;
        }
    }
    return MakeTileSpec(mRunMax, geometry.tileMMax2, 1).n == 0 ? RJ_M_TILES : RJ_OK;
}

// 检查 L0A/L0B 占用并记录缓冲跨度；inputs 为元素宽度配置，geometry 接收容量结果。
inline int ConfigureOperandBuffers(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int l0a1 = Align(geometry.tileMMax1, MMAD_M0) * geometry.tileK1 * inputs.inputElementBytes;
    const int l0a2 = Align(geometry.tileMMax2, MMAD_M0) * geometry.tileK2 * MidElemBytes(inputs);
    if (l0a1 > L0A_SLOT_BYTES || l0a2 > L0A_SLOT_BYTES) {
        return RJ_TILE_K;
    }
    const int weightBytes = WeightElemBytes(inputs);
    const int l0b1 = Align(inputs.outputChannels1, MMAD_M0) * geometry.weightSegmentK1 * weightBytes;
    const int l0b2 = Align(inputs.outputChannels2, MMAD_M0) * geometry.weightSegmentK2 * weightBytes;
    const int usedBytes1 = l0b1 * (geometry.l0bPingPong1 != 0 ? 2 : 1);
    const int usedBytes2 = l0b2 * (geometry.l0bPingPong2 != 0 ? 2 : 1);
    const int l0bMax = MaxI(usedBytes1, usedBytes2);
    if (l0bMax > L0B_BYTES) {
        return RJ_L0B;
    }
    geometry.l0bElements = l0bMax / weightBytes;
    geometry.l0bSlotElements1 = (geometry.l0bPingPong1 != 0) ? l0b1 / weightBytes : 0;
    geometry.l0bSlotElements2 = (geometry.l0bPingPong2 != 0) ? l0b2 / weightBytes : 0;
    return RJ_OK;
}

// 配置两层共用的 L0C 和固定分层 BT；inputs/geometry 为通道及 M 分块配置。
inline int ConfigureAccumulatorBuffers(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int l0c1 = Align(geometry.tileMMax1, MMAD_M0) * Align(inputs.outputChannels1, MMAD_M0);
    const int l0c2 = Align(geometry.tileMMax2, MMAD_M0) * Align(inputs.outputChannels2, MMAD_M0);
    if (MaxI(l0c1, l0c2) > L0C_ELEMS) {
        return RJ_L0C;
    }
    // 各层独立判断能否容纳两份累加结果，两层执行阶段复用同一片 L0C。
    geometry.l0cSlotStride = (2 * l0c1 <= L0C_ELEMS) ? l0c1 : 0;
    geometry.l0cSlotStride2 = (2 * l0c2 <= L0C_ELEMS) ? l0c2 : 0;
    geometry.l0cElements = MaxI(geometry.l0cSlotStride + l0c1, geometry.l0cSlotStride2 + l0c2);
    return inputs.outputChannels1 > BT_SLOT_ELEMS || inputs.outputChannels2 > BT_SLOT_ELEMS ? RJ_BT : RJ_OK;
}

// 按驻留优先级选择 L1 权重布局；inputs/geometry 为配置及容量，l1Budget 为可用字节数。
inline int SelectL1WeightLayout(const ConvTilingInputs& inputs, int l1Budget, ConvTilingGeometry& geometry)
{
    geometry.inputBufferElements = geometry.inputWindowRows * geometry.inputWindowWidthMax * inputs.inputChannels;
    geometry.intermediateBufferElements = geometry.intermediatePositionsMax * inputs.outputChannels1;
    const int w1Full = Align(inputs.outputChannels1, MMAD_M0) * geometry.reductionLength1;
    const int w2Full = Align(inputs.outputChannels2, MMAD_M0) * geometry.reductionLength2;
    const int w1Ring = 2 * Align(inputs.outputChannels1, MMAD_M0) * geometry.weightSegmentK1;
    const int w2Ring = 2 * Align(inputs.outputChannels2, MMAD_M0) * geometry.weightSegmentK2;
    const bool w1CanStage = (geometry.weightSegmentCount1 > 1) && (w1Ring < w1Full);
    const bool w2CanStage = (geometry.weightSegmentCount2 > 1) && (w2Ring < w2Full);
    const bool w2Bigger = (w2Full - w2Ring) >= (w1Full - w1Ring);
    const int modes[4] = {0, w2Bigger ? 2 : 1, w2Bigger ? 1 : 2, 3};
    const int budget = (l1Budget > 0) ? l1Budget : 0x7fffffff;
    // 优先两套权重常驻；容量不足时先循环搬运更能节省空间的一层。
    for (int modeIndex = 0; modeIndex < 4; ++modeIndex) {
        const int mode = modes[modeIndex];
        const int stage1 = (mode & 1) != 0 ? 1 : 0;
        const int stage2 = (mode & 2) != 0 ? 1 : 0;
        if ((stage1 != 0 && !w1CanStage) || (stage2 != 0 && !w2CanStage)) {
            continue;
        }
        geometry.weightRing1 = stage1;
        geometry.weightRing2 = stage2;
        geometry.weightBufferElements1 = (stage1 != 0) ? w1Ring : w1Full;
        geometry.weightBufferElements2 = (stage2 != 0) ? w2Ring : w2Full;
        LayOutL1(inputs, geometry, inputs.inputElementBytes, MidElemBytes(inputs), WeightElemBytes(inputs),
                 ParamBiasElemBytes(inputs));
        if (geometry.l1UsedBytes <= budget) {
            return RJ_OK;
        }
    }
    return RJ_L1;
}

// 推导一组完整候选；inputs 为卷积配置，hb/nwseg 为划分，l1Budget 为预算，geometry 接收结果。
inline int DeriveWith(const ConvTilingInputs& inputs, int hb, int nwseg, int l1Budget, ConvTilingGeometry& geometry)
{
    int status = ValidateDeriveInputs(inputs, hb);
    if (status != RJ_OK) {
        return status;
    }
    status = DeriveConvolutionGeometry(inputs, geometry);
    if (status != RJ_OK) {
        return status;
    }
    status = DeriveWindowGeometry(inputs, hb, nwseg, geometry);
    if (status != RJ_OK) {
        return status;
    }
    const int mRunMax = (geometry.widthSegmentCount == 1) ? (hb * geometry.outputWidth2) : geometry.segmentWidth;
    if ((status = SelectComputeTiles(inputs, mRunMax, geometry)) != RJ_OK ||
        (status = SelectWeightSegments(inputs, geometry)) != RJ_OK ||
        (status = ValidateBandTiles(inputs, mRunMax, geometry)) != RJ_OK ||
        (status = ConfigureOperandBuffers(inputs, geometry)) != RJ_OK ||
        (status = ConfigureAccumulatorBuffers(inputs, geometry)) != RJ_OK ||
        (status = SelectL1WeightLayout(inputs, l1Budget, geometry)) != RJ_OK) {
        return status;
    }
    geometry.inputImageElements = inputs.inputChannels * inputs.inputHeight * inputs.inputWidth;
    geometry.outputImageElements = inputs.outputChannels2 * geometry.outputHeight2 * geometry.outputWidth2;
    return RJ_OK;
}

// 在已选行带内优化 M/K 分块；inputs/l1Budget 为配置及容量，失败时保留 geometry 的原可行方案。
inline void RefineComputeTiles(const ConvTilingInputs& inputs, int l1Budget, ConvTilingGeometry& geometry)
{
    ConvTilingGeometry refined = geometry;
    const int outputPositions = geometry.widthSegmentCount == 1 ? geometry.bandHeight * geometry.outputWidth2 :
                                                                  geometry.segmentWidth;
    const auto first = PickComputeTile(geometry.reductionLength1, MaxI(geometry.inputC0, geometry.weightC0),
                                       inputs.inputElementBytes, inputs.outputChannels1,
                                       geometry.intermediatePositionsMax, WeightElemBytes(inputs), false,
                                       geometry.weightRing1 != 0);
    const auto second = PickComputeTile(geometry.reductionLength2, MaxI(geometry.intermediateC0, geometry.weightC0),
                                        MidElemBytes(inputs), inputs.outputChannels2, outputPositions,
                                        WeightElemBytes(inputs), true, geometry.weightRing2 != 0);
    if (!first.valid || !second.valid) {
        return;
    }
    refined.tileK1 = first.tileK;
    refined.tileMMax1 = first.maxPositions;
    refined.tileK2 = second.tileK;
    refined.tileMMax2 = second.maxPositions;
    if (SelectWeightSegments(inputs, refined) != RJ_OK ||
        ValidateBandTiles(inputs, outputPositions, refined) != RJ_OK ||
        ConfigureOperandBuffers(inputs, refined) != RJ_OK || ConfigureAccumulatorBuffers(inputs, refined) != RJ_OK ||
        SelectL1WeightLayout(inputs, l1Budget, refined) != RJ_OK || refined.weightRing1 > geometry.weightRing1 ||
        refined.weightRing2 > geometry.weightRing2) {
        return;
    }
    geometry = refined;
}

struct BandCost {
    long long cube = 0;   // 矩阵计算代价
    long long mte1 = 0;   // L1 到 L0 搬运代价
    long long mte2 = 0;   // GM 到 L1 搬运代价
    long long fix = 0;    // 累加结果输出代价
    long long scalar = 0; // 分块调度代价
};

// 估算一个行带的矩阵计算代价；geometry 为候选几何，band 为裁剪后的执行窗口。
inline long long EstimateBandCubeCost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int cubeKFactor1 = (geometry.inputs.inputElementBytes == 2) ? 16 : 32;
    const int cubeKFactor2 = (MidElemBytes(geometry.inputs) == 2) ? 16 : 32;
    long long cost = 0;
    for (int tileIndex = 0; tileIndex < band.conv1Tile.n; ++tileIndex) {
        cost += (long long)(Align(TileRows(band.conv1Tile, tileIndex), MMAD_M0) / MMAD_M0) * (cout1A / MMAD_M0) *
                (geometry.reductionLength1 / cubeKFactor1);
    }
    for (int runIndex = 0; runIndex < band.nRun; ++runIndex) {
        for (int tileIndex = 0; tileIndex < band.conv2Tile.n; ++tileIndex) {
            cost += (long long)(Align(TileRows(band.conv2Tile, tileIndex), MMAD_M0) / MMAD_M0) * (cout2A / MMAD_M0) *
                    (geometry.reductionLength2 / cubeKFactor2);
        }
    }
    return cost;
}

// 估算操作数搬运代价；geometry 为候选几何，band 为本行带的有效 M 范围。
inline long long EstimateBandMte1Cost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int weightBytes = WeightElemBytes(geometry.inputs);
    long long operandBytes = (long long)band.conv1MPerBand * geometry.reductionLength1 *
                                 geometry.inputs.inputElementBytes +
                             (long long)band.nRun * band.mRun * geometry.reductionLength2 *
                                 MidElemBytes(geometry.inputs);
    operandBytes += (geometry.weightSegmentCount1 == 1) ?
                        (long long)cout1A * geometry.reductionLength1 * weightBytes :
                        (long long)band.conv1Tile.n * cout1A * geometry.reductionLength1 * weightBytes;
    operandBytes += (geometry.weightSegmentCount2 == 1) ?
                        (long long)cout2A * geometry.reductionLength2 * weightBytes :
                        (long long)band.nRun * band.conv2Tile.n * cout2A * geometry.reductionLength2 * weightBytes;
    return operandBytes / MTE1_COST_FACTOR;
}

// 估算输入及循环权重搬运代价；geometry 为驻留配置，band 为有效输入窗口。
inline long long EstimateBandMte2Cost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const long long inputBytes = (long long)band.xRows * band.wIn * geometry.inputs.inputChannels *
                                 geometry.inputs.inputElementBytes;
    long long cost = inputBytes / MTE2_COST_FACTOR;
    if (band.wIn != geometry.inputs.inputWidth) {
        cost += (long long)band.xRows * geometry.inputs.inputChannels * MTE2_ROW_COST;
    }
    long long ringWeightBytes = 0;
    if (geometry.weightRing1 != 0) {
        ringWeightBytes += (long long)band.conv1Tile.n * Align(geometry.inputs.outputChannels1, MMAD_M0) *
                           geometry.reductionLength1 * WeightElemBytes(geometry.inputs);
    }
    if (geometry.weightRing2 != 0) {
        ringWeightBytes += (long long)band.nRun * band.conv2Tile.n * Align(geometry.inputs.outputChannels2, MMAD_M0) *
                           geometry.reductionLength2 * WeightElemBytes(geometry.inputs);
    }
    return cost + ringWeightBytes / MTE2_COST_FACTOR;
}

// 估算中间与最终结果写回代价；geometry 为数据通路配置，band 为本行带分块。
inline long long EstimateBandFixCost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int outputBytes = geometry.inputs.hasDequantScale2 ? 2 : MidElemBytes(geometry.inputs);
    long long cost = (long long)band.conv1MPerBand * cout1A * 4 / FIX_L1_COST_FACTOR;
    const long long outputRequests = (long long)band.nRun * band.conv2Tile.n * cout2A;
    const long long outputSize = (long long)band.nRun * band.mRun * cout2A * outputBytes;
    cost += outputRequests * FIX_GM_REQUEST_COST_X2 / 2 + outputSize / FIX_GM_COST_FACTOR;
    return cost;
}

// 估算最忙核心的候选代价；geometry 为候选几何，chunksPerCore 为每核最多处理的行带数。
inline long long CoreCostCycles(const ConvTilingGeometry& geometry, int chunksPerCore)
{
    FlatGeom flat;
    Flatten(geometry, chunksPerCore, flat);
    const int bands[3] = {0, geometry.heightBandCount / 2, geometry.heightBandCount - 1};
    const int segments[3] = {0, geometry.widthSegmentCount / 2, geometry.widthSegmentCount - 1};
    BandCost worst;
    // 对首、中、尾窗口分别取各流水代价上界，避免边界裁剪低估最忙核心。
    for (int bandIndex : bands) {
        for (int segmentIndex : segments) {
            if (bandIndex < 0 || segmentIndex < 0) {
                continue;
            }
            BandLite band;
            MakeBandLite(flat, bandIndex, segmentIndex, band);
            const long long cube = EstimateBandCubeCost(geometry, band);
            const long long mte1 = EstimateBandMte1Cost(geometry, band);
            const long long mte2 = EstimateBandMte2Cost(geometry, band);
            const long long fix = EstimateBandFixCost(geometry, band);
            const long long scalar = (long long)(band.conv1Tile.n + band.nRun * band.conv2Tile.n) * SCALAR_TILE_COST;
            worst.cube = cube > worst.cube ? cube : worst.cube;
            worst.mte1 = mte1 > worst.mte1 ? mte1 : worst.mte1;
            worst.mte2 = mte2 > worst.mte2 ? mte2 : worst.mte2;
            worst.fix = fix > worst.fix ? fix : worst.fix;
            worst.scalar = scalar > worst.scalar ? scalar : worst.scalar;
        }
    }
    long long residentWeightBytes = 0;
    if (geometry.weightRing1 == 0) {
        residentWeightBytes += (long long)Align(geometry.inputs.outputChannels1, MMAD_M0) * geometry.reductionLength1 *
                               WeightElemBytes(geometry.inputs);
    }
    if (geometry.weightRing2 == 0) {
        residentWeightBytes += (long long)Align(geometry.inputs.outputChannels2, MMAD_M0) * geometry.reductionLength2 *
                               WeightElemBytes(geometry.inputs);
    }
    return (long long)chunksPerCore * (worst.cube + worst.mte1 + worst.mte2 + worst.fix + worst.scalar) +
           residentWeightBytes / MTE2_COST_FACTOR + SCALAR_BASE_COST;
}

struct RowBandChoice {
    int height = 0;             // 最佳完整行带高
    long long cost = 0;         // 行覆盖代价
    int stagedLayers = 3;       // 循环搬运权重的层数
    int firstReject = RJ_NO_HB; // 首个有效候选失败原因
};

struct SearchCostBounds {
    long long fullWidth[4] = {-1, -1, -1, -1};  // 各驻留等级的完整行带最小代价
    long long splitWidth[4] = {-1, -1, -1, -1}; // 各驻留等级的切列最小代价
};

struct SearchPreference {
    int stagedLayers = -1;   // 允许的权重循环层数
    bool allowSplit = false; // 是否允许切列
    long long cutoff = 0;    // 候选代价上界
};

struct SplitBandChoice {
    int height = 0;        // 最佳行带高
    int segments = 1;      // 最佳列段数
    int chunksPerCore = 0; // 每核最多行带数
};

// 校验搜索输入并得到最终输出高宽；inputs 为卷积配置，outputHeight/outputWidth 接收尺寸。
inline int GetSearchOutputShape(const ConvTilingInputs& inputs, int& outputHeight, int& outputWidth)
{
    if (inputs.inputElementBytes != 1 && inputs.inputElementBytes != 2) {
        return RJ_ELEM_BYTES;
    }
    if (inputs.batchSize <= 0 || inputs.inputChannels <= 0 || inputs.inputHeight <= 0 || inputs.inputWidth <= 0 ||
        inputs.outputChannels1 <= 0 || inputs.outputChannels2 <= 0 || inputs.kernelHeight1 <= 0 ||
        inputs.kernelWidth1 <= 0 || Kh2(inputs) <= 0 || Kw2(inputs) <= 0 || inputs.stride1 <= 0 ||
        inputs.stride2 <= 0 || inputs.paddingHeight1 < 0 || inputs.paddingWidth1 < 0 || inputs.paddingHeight2 < 0 ||
        inputs.paddingWidth2 < 0) {
        return RJ_SHAPE_POSITIVE;
    }
    const int midHeight = (inputs.inputHeight + 2 * inputs.paddingHeight1 - inputs.kernelHeight1) / inputs.stride1 + 1;
    if (midHeight <= 0) {
        return RJ_OUT_EMPTY;
    }
    outputHeight = (midHeight + 2 * inputs.paddingHeight2 - Kh2(inputs)) / inputs.stride2 + 1;
    if (outputHeight <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int midWidth = (inputs.inputWidth + 2 * inputs.paddingWidth1 - inputs.kernelWidth1) / inputs.stride1 + 1;
    if (midWidth <= 0) {
        return RJ_OUT_EMPTY;
    }
    outputWidth = (midWidth + 2 * inputs.paddingWidth2 - Kw2(inputs)) / inputs.stride2 + 1;
    return outputWidth <= 0 ? RJ_OUT_EMPTY : RJ_OK;
}

// 搜索完整行带；inputs 为输入，outputHeight/aicNum/l1Budget 为搜索约束，返回最优行带和失败原因。
inline RowBandChoice SearchFullWidthBands(const ConvTilingInputs& inputs, int outputHeight, int aicNum, int l1Budget)
{
    RowBandChoice best;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        const int status = DeriveWith(inputs, hb, 1, l1Budget, probe);
        if (status != RJ_OK) {
            if (best.firstReject == RJ_NO_HB && status != RJ_HB_DIVIDE) {
                best.firstReject = status;
            }
            continue;
        }
        const long long rounds = CeilDiv(probe.totalBandTasks, aicNum);
        const long long cost = rounds * (long long)probe.intermediateRows;
        const int stagedLayers = probe.weightRing1 + probe.weightRing2;
        if (best.height == 0 || stagedLayers < best.stagedLayers ||
            (stagedLayers == best.stagedLayers && (cost < best.cost || (cost == best.cost && hb > best.height)))) {
            best.height = hb;
            best.cost = cost;
            best.stagedLayers = stagedLayers;
        }
    }
    return best;
}

// 汇总两种列布局的最小代价；inputs 为卷积配置，outputHeight/maxSegments 为搜索范围，
// aicNum/l1Budget 为资源，firstReject 记录首个失败。
inline SearchCostBounds CollectSearchCosts(const ConvTilingInputs& inputs, int outputHeight, int maxSegments,
                                           int aicNum, int l1Budget, int& firstReject)
{
    SearchCostBounds bounds;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= maxSegments; ++nw) {
            const int status = DeriveWith(inputs, hb, nw, l1Budget, probe);
            if (status != RJ_OK) {
                if (firstReject == RJ_NO_HB && status != RJ_HB_DIVIDE && status != RJ_WSEG) {
                    firstReject = status;
                }
                continue;
            }
            const int activeCores = MinI(aicNum, probe.totalBandTasks);
            const int chunksPerCore = CeilDiv(probe.totalBandTasks, activeCores);
            const long long cost = CoreCostCycles(probe, chunksPerCore);
            const int stagedLayers = probe.weightRing1 + probe.weightRing2;
            long long& minimum = nw == 1 ? bounds.fullWidth[stagedLayers] : bounds.splitWidth[stagedLayers];
            if (minimum < 0 || cost < minimum) {
                minimum = cost;
            }
        }
    }
    return bounds;
}

// 确定可接受的代价区间；bounds 为各驻留等级的最优代价，返回切列许可及阈值。
inline SearchPreference ChooseSearchPreference(const SearchCostBounds& bounds)
{
    SearchPreference preference;
    for (int stage = 0; stage < 4; ++stage) {
        if (bounds.fullWidth[stage] >= 0 || bounds.splitWidth[stage] >= 0) {
            preference.stagedLayers = stage;
            break;
        }
    }
    if (preference.stagedLayers < 0) {
        return preference;
    }
    const long long fullBest = bounds.fullWidth[preference.stagedLayers];
    const long long splitBest = bounds.splitWidth[preference.stagedLayers];
    // 切列至少降低 20% 代价才启用；随后允许 2% 区间内按任务均衡和较少切列择优。
    preference.allowSplit = splitBest >= 0 && (fullBest < 0 || splitBest * 5 <= fullBest * 4);
    long long minimumCost = fullBest >= 0 ? fullBest : -1;
    if (preference.allowSplit && (minimumCost < 0 || splitBest < minimumCost)) {
        minimumCost = splitBest;
    }
    preference.cutoff = minimumCost + minimumCost / 50;
    return preference;
}

// 在代价区间内择优；inputs 为卷积配置，outputHeight/maxSegments 为搜索范围，
// aicNum/l1Budget 为资源，preference 为代价及驻留筛选规则。
inline SplitBandChoice SearchSplitBands(const ConvTilingInputs& inputs, int outputHeight, int maxSegments, int aicNum,
                                        int l1Budget, const SearchPreference& preference)
{
    SplitBandChoice best;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= maxSegments; ++nw) {
            if (DeriveWith(inputs, hb, nw, l1Budget, probe) != RJ_OK ||
                probe.weightRing1 + probe.weightRing2 != preference.stagedLayers ||
                (nw > 1 && !preference.allowSplit)) {
                continue;
            }
            const int chunksPerCore = CeilDiv(probe.totalBandTasks, MinI(aicNum, probe.totalBandTasks));
            if (CoreCostCycles(probe, chunksPerCore) > preference.cutoff) {
                continue;
            }
            if (best.height == 0 || chunksPerCore < best.chunksPerCore ||
                (chunksPerCore == best.chunksPerCore &&
                 (nw < best.segments || (nw == best.segments && hb > best.height)))) {
                best.height = hb;
                best.segments = nw;
                best.chunksPerCore = chunksPerCore;
            }
        }
    }
    return best;
}

// 搜索行带高和列段数；inputs 为输入，aicNum/l1Budget 为资源，geometry 接收最终可执行几何。
inline int PickBandGeometry(const ConvTilingInputs& inputs, int aicNum, int l1Budget, ConvTilingGeometry& geometry)
{
    if (aicNum <= 0) {
        aicNum = 1;
    }
    int outputHeight = 0, outputWidth = 0;
    const int status = GetSearchOutputShape(inputs, outputHeight, outputWidth);
    if (status != RJ_OK) {
        return status;
    }
    RowBandChoice rowChoice = SearchFullWidthBands(inputs, outputHeight, aicNum, l1Budget);
    if (rowChoice.height != 0) {
        const long long fullWidthTasks = (long long)inputs.batchSize * (outputHeight / rowChoice.height);
        if (fullWidthTasks >= (long long)aicNum && fullWidthTasks % (long long)aicNum == 0) {
            return DeriveWith(inputs, rowChoice.height, 1, l1Budget, geometry);
        }
    }
    const int maxSegments = MinI(aicNum, outputWidth);
    const auto bounds = CollectSearchCosts(inputs, outputHeight, maxSegments, aicNum, l1Budget, rowChoice.firstReject);
    const auto preference = ChooseSearchPreference(bounds);
    const auto splitChoice = preference.stagedLayers >= 0 ?
                                 SearchSplitBands(inputs, outputHeight, maxSegments, aicNum, l1Budget, preference) :
                                 SplitBandChoice{};
    if (splitChoice.height != 0) {
        return DeriveWith(inputs, splitChoice.height, splitChoice.segments, l1Budget, geometry);
    }
    return rowChoice.height == 0 ? rowChoice.firstReject : DeriveWith(inputs, rowChoice.height, 1, l1Budget, geometry);
}

// 先确定行带和列段，再优化内部计算分块；inputs/aicNum/l1Budget 为输入及资源，geometry 接收结果。
inline int PickHb(const ConvTilingInputs& inputs, int aicNum, int l1Budget, ConvTilingGeometry& geometry)
{
    const int status = PickBandGeometry(inputs, aicNum, l1Budget, geometry);
    if (status == RJ_OK) {
        RefineComputeTiles(inputs, l1Budget, geometry);
    }
    return status;
}

} // namespace FusedMultiConv2dShape

#endif
