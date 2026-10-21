// ===========================================================================
// 这是 ops-nn 仓里 conv/fused_multi_conv2d/op_host/op_tiling/fused_multi_conv2d_tiling_geometry.h 的**副本**。
//
// 放一份在 harness 里，只为了 fc2d.py 能在生成 case 之前就回答「这个形状能不能
// 跑」—— 否则要等到 atc 失败，而那条原因只在 ~/ascend/log/ 里。
//
// **算子那边改了就要重拷。** 一条命令：
//     python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_multi_conv2d/op_host/op_tiling/fused_multi_conv2d_tiling_geometry.h
//
// 它是顾问性质的：真正说了算的永远是算子的 tiling。两边不一致时 fc2d.py 的预检会
// 和 atc 的结论不同，那本身就是「该重拷了」的信号。
// ===========================================================================
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_TILING_GEOMETRY_H
#define FUSED_MULTI_CONV2D_TILING_GEOMETRY_H

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

FC2D_GEOM_CE int MinI(int a, int b) { return a < b ? a : b; }
FC2D_GEOM_CE int MaxI(int a, int b) { return a > b ? a : b; }
FC2D_GEOM_CE int CeilDiv(int a, int b) { return (a + b - 1) / b; }
FC2D_GEOM_CE int Align(int a, int m) { return (a + m - 1) / m * m; }

struct TileSpec {
    int n;
    int base;
    int extra;
    int gran;
};

// 将 total 个位置均匀划分，单块不超过 maxRows，块长按 gran 对齐。
FC2D_GEOM_CE TileSpec MakeTileSpec(int total, int maxRows, int gran)
{
    TileSpec s = {0, 0, 0, gran};
    if (total <= 0 || gran <= 0 || maxRows < gran || (total % gran) != 0) {
        return s;
    }
    const int units = total / gran;
    const int maxUnits = maxRows / gran;
    const int n = CeilDiv(units, maxUnits);
    if (n > MAX_M_TILES) {
        return s;
    }
    s.n = n;
    s.base = units / n;
    s.extra = units % n;
    return s;
}

FC2D_GEOM_CE int TileRows(const TileSpec& s, int i) { return (s.base + (i < s.extra ? 1 : 0)) * s.gran; }
FC2D_GEOM_CE int TileRow0(const TileSpec& s, int i) { return (s.base * i + MinI(i, s.extra)) * s.gran; }

struct FmLoadSpec {
    int n;
    int firstC;
};

// 选择输入通道分段搬运方式及首段通道数。
FC2D_GEOM_CE FmLoadSpec MakeFmLoadSpec(int ci, int c0, int tileK1, int l0bChunks1, int nwseg, int conv1MTiles)
{
    FmLoadSpec s = {1, ci};
    if (nwseg > 1 && l0bChunks1 == 1 && conv1MTiles == 1 && tileK1 >= c0 && tileK1 < ci && (tileK1 % c0) == 0) {
        s.n = 2;
        s.firstC = tileK1;
    }
    return s;
}

struct FlatGeom {
    int n, ci, hi, wi, cout1, cout2, kh1, kw1, kh2, kw2, stride1, stride2;
    int padH1, padW1, padH2, padW2, elemBytes;
    int weightElemBytes, midElemBytes, biasElemBytes;

    int c0, weightC0, midC0, ho1, wo1, ho2, wo2, k1, k2;
    int hb, nchunk, chunkTotal, chunksPerCore;
    int wseg, nwseg;
    int midRows;
    int mMax1, mMax2;
    int tileK1, tileK2;
    int l0bChunks1, l0bChunkK1, tilesPerChunk1;
    int l0bChunks2, l0bChunkK2, tilesPerChunk2;
    int l0bPP1, l0bPP2;
    int l0bSlot1Elems, l0bSlot2Elems;
    int l0bElems, l0cElems, l0cStride;
    int w1Stage, w2Stage;

    int fmAddr, fmElems, midAddr, midElems, w1Addr, w1Elems, w2Addr, w2Elems, b1Addr, b2Addr;
    int dq2Addr;
    int q1Addr;
    int q2Addr;

    int xPlane, yPlane, yPlaneM;
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
    int aMid;
    int midReal;
    int padT2, padB2;
    int xRow0, xRows, padT1, padB1;
    int outRow0;

    int outCol0, outCols;
    int midCol0, wSub;
    int padL2, padR2;
    int xCol0, wIn;
    int padL1, padR1;
    int conv1MPerBand;

    int rowsPerRun, mRun, nRun;
    TileSpec conv1Tile;
    TileSpec conv2Tile;
};

// 根据行带和列段索引计算当前输入窗口及补边。
FC2D_GEOM_FN void MakeBandLite(const FlatGeom& f, int band, int woSegIdx, BandLite& b)
{
    b.outRow0 = band * f.hb;

    const int raw = f.stride2 * f.hb * band - f.padH2;
    b.aMid = MaxI(0, raw);
    b.padT2 = b.aMid - raw;
    const int wantRows = f.midRows - b.padT2;
    b.midReal = MinI(wantRows, f.ho1 - b.aMid);
    b.padB2 = f.midRows - b.padT2 - b.midReal;

    const int xRaw = f.stride1 * b.aMid - f.padH1;
    const int xSpan = f.stride1 * (b.midReal - 1) + f.kh1;
    b.xRow0 = MaxI(0, xRaw);
    b.padT1 = b.xRow0 - xRaw;
    const int xEnd = MinI(f.hi, xRaw + xSpan);
    b.xRows = xEnd - b.xRow0;
    b.padB1 = xRaw + xSpan - xEnd;

    b.outCol0 = woSegIdx * f.wseg;
    b.outCols = MinI(f.wseg, f.wo2 - b.outCol0);

    const int mcRaw0 = b.outCol0 * f.stride2 - f.padW2;
    const int mcRaw1 = (b.outCol0 + b.outCols - 1) * f.stride2 - f.padW2 + f.kw2;
    b.midCol0 = MaxI(0, mcRaw0);
    const int mcEnd = MinI(f.wo1, mcRaw1);
    b.padL2 = b.midCol0 - mcRaw0;
    b.padR2 = mcRaw1 - mcEnd;
    b.wSub = mcEnd - b.midCol0;

    const int icRaw0 = b.midCol0 * f.stride1 - f.padW1;
    const int icRaw1 = (mcEnd - 1) * f.stride1 - f.padW1 + f.kw1;
    b.xCol0 = MaxI(0, icRaw0);
    const int icEnd = MinI(f.wi, icRaw1);
    b.padL1 = b.xCol0 - icRaw0;
    b.padR1 = icRaw1 - icEnd;
    b.wIn = icEnd - b.xCol0;

    b.conv1MPerBand = b.midReal * b.wSub;
    b.conv1Tile = MakeTileSpec(b.conv1MPerBand, f.mMax1, 1);

    b.rowsPerRun = (b.outCols == f.wo2) ? f.hb : 1;
    b.mRun = b.rowsPerRun * b.outCols;
    b.nRun = f.hb / b.rowsPerRun;
    b.conv2Tile = MakeTileSpec(b.mRun, f.mMax2, 1);
}

} // namespace FusedMultiConv2dShape

#endif


namespace FusedMultiConv2dShape {

#define FC2D_GEOM_REV "geom13"

constexpr int L0B_BYTES = 64 * 1024;
constexpr int L0B_SLOT_BYTES = L0B_BYTES / 2;
constexpr int L0C_BYTES = 256 * 1024;
constexpr int L0C_ELEMS = L0C_BYTES / 4;
constexpr int L1_SEG_ALIGN = 512;
constexpr int MTE1_BYTES_PER_CYC = 190;
constexpr int MTE2_BYTES_PER_CYC = 120;
constexpr int MTE2_ROW_REQ_CYC = 8;
constexpr int FIX_L1_BYTES_PER_CYC = 290;
constexpr int FIX_GM_BYTES_PER_CYC = 105;
constexpr int FIX_GM_REQ_CYC_X2 = 19;
constexpr int SCALAR_TILE_CYC = 420;
constexpr int SCALAR_BASE_CYC = 4000;

constexpr int BiasElemBytes(int elemBytes) { return elemBytes == 2 ? 2 : 4; }

struct Params {
    int n;
    int ci;
    int hi;
    int wi;
    int cout1;
    int cout2;

    int kh1;
    int kw1;
    int stride1;
    int stride2;

    int padH1, padW1;
    int padH2, padW2;
    int elemBytes;

    int hasDeqScale2 = 0;

    int weightElemBytes = 0;
    int midElemBytes = 0;
    int biasElemBytes = 0;
    int hasQuantScale1 = 0;
    int hasQuantScale2 = 0;

    int kh2 = 0;
    int kw2 = 0;
};

constexpr int WeightElemBytes(const Params& p) { return p.weightElemBytes == 0 ? p.elemBytes : p.weightElemBytes; }
constexpr int MidElemBytes(const Params& p) { return p.midElemBytes == 0 ? p.elemBytes : p.midElemBytes; }
constexpr int ParamBiasElemBytes(const Params& p)
{
    return p.biasElemBytes == 0 ? BiasElemBytes(p.elemBytes) : p.biasElemBytes;
}

constexpr int Kh2(const Params& p) { return p.kh2 == 0 ? p.kh1 : p.kh2; }
constexpr int Kw2(const Params& p) { return p.kw2 == 0 ? p.kw1 : p.kw2; }

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

// 将几何校验返回码转换为错误说明。
inline const char* RejectText(int r)
{
    switch (r) {
        case RJ_OK:
            return "ok";
        case RJ_ELEM_BYTES:
            return "elemBytes 只能是 1(int8) 或 2(fp16)";
        case RJ_SHAPE_POSITIVE:
            return "形状里有非正数";
        case RJ_CHANNEL_C0:
            return "ci 和 cout1 必须是 C0 的整数倍（C0 = 32/elemBytes）";
        case RJ_OUT_EMPTY:
            return "卷积输出为空（kernel 比补零后的输入还大）";
        case RJ_HB_DIVIDE:
            return "band 高度不整除 ho2";
        case RJ_MIDROWS:
            return "一个 band 需要的 conv1 行数超过了 ho1";
        case RJ_TILE_K:
            return "找不到既整除 K 又装得进 L0A 的 K 切分";
        case RJ_M_TILES:
            return "M 方向的块数超过 MAX_M_TILES";
        case RJ_L0C:
            return "一个 M 子块装不进 L0C";
        case RJ_L0B:
            return "权重按 K 折之后仍然装不进 L0B";
        case RJ_BT:
            return "bias 的通道数超过一个 BT 槽（512）";
        case RJ_L1:
            return "L1 装不下 fm + mid + w1 + w2";
        case RJ_U16:
            return "指令字段只有 16 位：m1Max（min(midRows,ho1)*wSubMax）或 staging 的位置数（xRows*wInMax）超过 65535";
        case RJ_NO_HB:
            return "没有任何 band 高度能同时满足全部约束";
        case RJ_WSEG:
            return "列分段数不合法（必须在 1..wo2 之间）";
        default:
            return "未知";
    }
}

struct Geometry {
    Params p;

    int c0, weightC0, midC0, c1, midC1;
    int ho1, wo1, ho2, wo2;
    int k1, k2;

    int hb, nchunk;

    int wseg, nwseg;
    int wSubMax, wInMax;
    int chunkTotal;

    int midRows;
    int xRows;
    int m1Max;

    int tileK1, tileK2;
    int mMax1, mMax2;
    int l0bChunks1, l0bChunkK1, tilesPerChunk1;
    int l0bChunks2, l0bChunkK2, tilesPerChunk2;
    int l0bPP1, l0bPP2;
    int l0bSlot1Elems, l0bSlot2Elems;

    int l0bElems;
    int l0cElems;

    int l0cStride;

    int w1Stage, w2Stage;

    int fmElems, midElems, w1Elems, w2Elems;
    int fmAddr, midAddr, w1Addr, w2Addr, b1Addr, b2Addr;
    int b1Bytes, b2Bytes;
    int dq2Addr, dq2Bytes;
    int q1Addr, q1Bytes;
    int q2Addr, q2Bytes;
    int l1Used;

    int xPlane, yPlane;

    int fz1K, fz1N, fz2K, fz2N;
};

struct KChoice {
    int tileK;
    int mMax;
    int nTiles;
    bool ok;
};

// 在片上容量约束内选择 K 分块和 M 上限。
inline KChoice PickTileK(int k, int c0, int elemBytes, int cout, int totalM, int weightBytes)
{
    KChoice best;
    best.tileK = 0;
    best.mMax = 0;
    best.nTiles = 0;
    best.ok = false;
    if (k <= 0 || c0 <= 0 || cout <= 0 || totalM <= 0 || weightBytes <= 0) {
        return best;
    }

    const int mCapC = L0C_ELEMS / Align(cout, MMAD_M0);

    const int ckMax = L0B_BYTES / (Align(cout, MMAD_M0) * weightBytes);
    for (int tk = c0; tk <= k; tk += c0) {
        if ((k % tk) != 0) {
            continue;
        }
        if (tk > ckMax) {
            continue;
        }

        const int mCapA = L0A_SLOT_BYTES / (tk * elemBytes);

        int mMax = MinI(mCapA, mCapC) / MMAD_M0 * MMAD_M0;
        if (mMax <= 0) {
            continue;
        }
        if (mMax > totalM) {
            mMax = totalM;
        }
        const int nT = CeilDiv(totalM, mMax);
        if (nT > MAX_M_TILES) {
            continue;
        }

        if (!best.ok || nT < best.nTiles || (nT == best.nTiles && tk > best.tileK)) {
            best.tileK = tk;
            best.mMax = mMax;
            best.nTiles = nT;
            best.ok = true;
        }
    }
    return best;
}

// 在 budget 字节内选择整分 K 的权重段数及段长。
inline bool PickL0bChunksIn(int cout, int k, int elemBytes, int tileK, int budget, int& chunks, int& chunkK)
{
    const int coutA = Align(cout, MMAD_M0);
    for (int c = 1; c <= k / tileK; ++c) {
        if ((k % c) != 0) {
            continue;
        }
        const int ck = k / c;
        if ((ck % tileK) != 0) {
            continue;
        }
        if (coutA * ck * elemBytes <= budget) {
            chunks = c;
            chunkK = ck;
            return true;
        }
    }
    return false;
}

// 选择 L0B 权重分段，优先使用 ping/pong 缓冲。
inline bool PickL0bChunks(int cout, int k, int elemBytes, int tileK, int& chunks, int& chunkK, int& pingpong)
{
    if (PickL0bChunksIn(cout, k, elemBytes, tileK, L0B_SLOT_BYTES, chunks, chunkK)) {
        pingpong = 1;
        return true;
    }
    pingpong = 0;
    return PickL0bChunksIn(cout, k, elemBytes, tileK, L0B_BYTES, chunks, chunkK);
}

// 将几何和每核分块数写入 tiling 共用的扁平结构。
inline void Flatten(const Geometry& g, int chunksPerCore, FlatGeom& f)
{
    f.n = g.p.n;
    f.ci = g.p.ci;
    f.hi = g.p.hi;
    f.wi = g.p.wi;
    f.cout1 = g.p.cout1;
    f.cout2 = g.p.cout2;
    f.kh1 = g.p.kh1;
    f.kw1 = g.p.kw1;
    f.kh2 = Kh2(g.p);
    f.kw2 = Kw2(g.p);
    f.stride1 = g.p.stride1;
    f.stride2 = g.p.stride2;
    f.padH1 = g.p.padH1;
    f.padW1 = g.p.padW1;
    f.padH2 = g.p.padH2;
    f.padW2 = g.p.padW2;
    f.elemBytes = g.p.elemBytes;
    f.weightElemBytes = WeightElemBytes(g.p);
    f.midElemBytes = MidElemBytes(g.p);
    f.biasElemBytes = ParamBiasElemBytes(g.p);

    f.c0 = g.c0;
    f.weightC0 = g.weightC0;
    f.midC0 = g.midC0;
    f.ho1 = g.ho1;
    f.wo1 = g.wo1;
    f.ho2 = g.ho2;
    f.wo2 = g.wo2;
    f.k1 = g.k1;
    f.k2 = g.k2;

    f.hb = g.hb;
    f.nchunk = g.nchunk;
    f.wseg = g.wseg;
    f.nwseg = g.nwseg;
    f.chunkTotal = g.chunkTotal;
    f.chunksPerCore = chunksPerCore;

    f.midRows = g.midRows;
    f.mMax1 = g.mMax1;
    f.mMax2 = g.mMax2;
    f.tileK1 = g.tileK1;
    f.tileK2 = g.tileK2;
    f.l0bChunks1 = g.l0bChunks1;
    f.l0bChunkK1 = g.l0bChunkK1;
    f.tilesPerChunk1 = g.tilesPerChunk1;
    f.l0bChunks2 = g.l0bChunks2;
    f.l0bChunkK2 = g.l0bChunkK2;
    f.tilesPerChunk2 = g.tilesPerChunk2;
    f.l0bPP1 = g.l0bPP1;
    f.l0bPP2 = g.l0bPP2;
    f.l0bSlot1Elems = g.l0bSlot1Elems;
    f.l0bSlot2Elems = g.l0bSlot2Elems;
    f.l0bElems = g.l0bElems;
    f.l0cElems = g.l0cElems;
    f.l0cStride = g.l0cStride;
    f.w1Stage = g.w1Stage;
    f.w2Stage = g.w2Stage;

    f.fmAddr = g.fmAddr;
    f.fmElems = g.fmElems;
    f.midAddr = g.midAddr;
    f.midElems = g.midElems;
    f.w1Addr = g.w1Addr;
    f.w1Elems = g.w1Elems;
    f.w2Addr = g.w2Addr;
    f.w2Elems = g.w2Elems;
    f.b1Addr = g.b1Addr;
    f.b2Addr = g.b2Addr;
    f.dq2Addr = g.dq2Addr;
    f.q1Addr = g.q1Addr;
    f.q2Addr = g.q2Addr;

    f.xPlane = g.xPlane;
    f.yPlane = g.yPlane;
    f.yPlaneM = g.ho2 * g.wo2;
}

// 按元素宽度和权重驻留方式分配 L1 缓冲地址。
inline void LayOutL1(const Params& p, Geometry& g, int xBytes, int midBytes, int weightBytes, int biasBytes)
{
    g.fmAddr = 0;
    g.midAddr = Align(g.fmAddr + g.fmElems * xBytes, L1_SEG_ALIGN);
    g.w1Addr = Align(g.midAddr + g.midElems * midBytes, L1_SEG_ALIGN);
    g.w2Addr = Align(g.w1Addr + g.w1Elems * weightBytes, L1_SEG_ALIGN);
    g.b1Addr = Align(g.w2Addr + g.w2Elems * weightBytes, L1_SEG_ALIGN);
    g.b1Bytes = Align(p.cout1 * biasBytes, L1_SEG_ALIGN);
    g.b2Addr = g.b1Addr + g.b1Bytes;
    g.b2Bytes = Align(p.cout2 * biasBytes, L1_SEG_ALIGN);

    g.dq2Addr = g.b2Addr + g.b2Bytes;
    g.dq2Bytes = (p.hasDeqScale2 != 0) ? Align(p.cout2 * (int)sizeof(long long), L1_SEG_ALIGN) : 0;
    g.q1Addr = g.dq2Addr + g.dq2Bytes;
    g.q1Bytes = (p.hasQuantScale1 != 0) ? Align(p.cout1 * (int)sizeof(long long), L1_SEG_ALIGN) : 0;
    g.q2Addr = g.q1Addr + g.q1Bytes;
    g.q2Bytes = (p.hasQuantScale2 != 0) ? Align(p.cout2 * (int)sizeof(long long), L1_SEG_ALIGN) : 0;
    g.l1Used = g.q2Addr + g.q2Bytes;
}

// 由行带高 hb、列段数 nwseg 和 L1 字节预算计算几何。
inline int DeriveWith(const Params& p, int hb, int nwseg, int l1Budget, Geometry& g)
{
    const int xBytes = p.elemBytes;
    const int weightBytes = WeightElemBytes(p);
    const int midBytes = MidElemBytes(p);
    const int biasBytes = ParamBiasElemBytes(p);
    if ((xBytes != 1 && xBytes != 2) || (weightBytes != 1 && weightBytes != 2) || (midBytes != 1 && midBytes != 2) ||
        (biasBytes != 2 && biasBytes != 4)) {
        return RJ_ELEM_BYTES;
    }
    if (p.n <= 0 || p.ci <= 0 || p.hi <= 0 || p.wi <= 0 || p.cout1 <= 0 || p.cout2 <= 0 || p.kh1 <= 0 || p.kw1 <= 0 ||
        Kh2(p) <= 0 || Kw2(p) <= 0 || p.stride1 <= 0 || p.stride2 <= 0 || p.padH1 < 0 || p.padW1 < 0 || p.padH2 < 0 ||
        p.padW2 < 0 || hb <= 0) {
        return RJ_SHAPE_POSITIVE;
    }

    g.p = p;
    g.c0 = 32 / xBytes;
    g.weightC0 = 32 / weightBytes;
    g.midC0 = 32 / midBytes;

    if ((p.ci % g.c0) != 0 || (p.ci % g.weightC0) != 0 || (p.cout1 % g.midC0) != 0 || (p.cout1 % g.weightC0) != 0) {
        return RJ_CHANNEL_C0;
    }

    g.c1 = p.ci / g.weightC0;
    g.midC1 = p.cout1 / g.weightC0;

    const int kh1 = p.kh1;
    const int kw1 = p.kw1;
    const int kh2 = Kh2(p);
    const int kw2 = Kw2(p);

    g.ho1 = (p.hi + 2 * p.padH1 - kh1) / p.stride1 + 1;
    g.wo1 = (p.wi + 2 * p.padW1 - kw1) / p.stride1 + 1;
    if (g.ho1 <= 0 || g.wo1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    g.ho2 = (g.ho1 + 2 * p.padH2 - kh2) / p.stride2 + 1;
    g.wo2 = (g.wo1 + 2 * p.padW2 - kw2) / p.stride2 + 1;
    if (g.ho2 <= 0 || g.wo2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    g.k1 = p.ci * kh1 * kw1;
    g.k2 = p.cout1 * kh2 * kw2;
    g.fz1K = g.c1 * kh1 * kw1;
    g.fz1N = CeilDiv(p.cout1, MMAD_M0);
    g.fz2K = g.midC1 * kh2 * kw2;
    g.fz2N = CeilDiv(p.cout2, MMAD_M0);

    if ((g.ho2 % hb) != 0) {
        return RJ_HB_DIVIDE;
    }
    if (nwseg <= 0 || nwseg > g.wo2) {
        return RJ_WSEG;
    }
    g.hb = hb;
    g.nchunk = g.ho2 / hb;

    g.wseg = CeilDiv(g.wo2, nwseg);
    g.nwseg = CeilDiv(g.wo2, g.wseg);
    g.chunkTotal = p.n * g.nchunk * g.nwseg;

    g.midRows = p.stride2 * (hb - 1) + kh2;

    const int midRowsCap = MinI(g.midRows, g.ho1);

    g.wSubMax = MinI(g.wo1, (g.wseg - 1) * p.stride2 + kw2);
    g.wInMax = MinI(p.wi, (g.wSubMax - 1) * p.stride1 + kw1);
    g.m1Max = midRowsCap * g.wSubMax;
    g.xRows = MinI(p.stride1 * (midRowsCap - 1) + kh1, p.hi);

    if (g.m1Max > 65535 || (long long)g.xRows * g.wInMax > 65535) {
        return RJ_U16;
    }

    const int mRunMax = (g.nwseg == 1) ? (hb * g.wo2) : g.wseg;
    const int kBlock1 = MaxI(g.c0, g.weightC0);
    const int kBlock2 = MaxI(g.midC0, g.weightC0);

    if (Align(p.cout1, MMAD_M0) * kBlock1 * weightBytes > L0B_BYTES ||
        Align(p.cout2, MMAD_M0) * kBlock2 * weightBytes > L0B_BYTES) {
        return RJ_L0B;
    }
    const KChoice kc1 = PickTileK(g.k1, kBlock1, xBytes, p.cout1, g.m1Max, weightBytes);
    const KChoice kc2 = PickTileK(g.k2, kBlock2, midBytes, p.cout2, mRunMax, weightBytes);
    if (!kc1.ok || !kc2.ok) {
        return RJ_TILE_K;
    }
    g.tileK1 = kc1.tileK;
    g.mMax1 = kc1.mMax;
    g.tileK2 = kc2.tileK;
    g.mMax2 = kc2.mMax;

    if (!PickL0bChunks(p.cout1, g.k1, weightBytes, g.tileK1, g.l0bChunks1, g.l0bChunkK1, g.l0bPP1) ||
        !PickL0bChunks(p.cout2, g.k2, weightBytes, g.tileK2, g.l0bChunks2, g.l0bChunkK2, g.l0bPP2)) {
        return RJ_L0B;
    }

    if (nwseg > 1 && g.l0bPP1 != 0) {
        int c1 = 0, ck1 = 0;
        if (PickL0bChunksIn(p.cout1, g.k1, weightBytes, g.tileK1, L0B_BYTES, c1, ck1) && c1 == 1) {
            g.l0bChunks1 = c1;
            g.l0bChunkK1 = ck1;
        }
    }

    if (g.l0bChunks1 == 1) {
        g.l0bPP1 = 0;
    }
    if (g.l0bChunks2 == 1) {
        g.l0bPP2 = 0;
    }
    g.tilesPerChunk1 = g.l0bChunkK1 / g.tileK1;
    g.tilesPerChunk2 = g.l0bChunkK2 / g.tileK2;

    for (int c = 0; c < g.nchunk; ++c) {
        const int raw = p.stride2 * hb * c - p.padH2;
        const int a = MaxI(0, raw);
        const int padT2 = a - raw;
        const int midReal = MinI(g.midRows - padT2, g.ho1 - a);
        if (midReal <= 0) {
            return RJ_MIDROWS;
        }
        if (MakeTileSpec(midReal * g.wSubMax, g.mMax1, 1).n == 0) {
            return RJ_M_TILES;
        }
    }
    if (MakeTileSpec(mRunMax, g.mMax2, 1).n == 0) {
        return RJ_M_TILES;
    }

    const int l0a1 = Align(g.mMax1, MMAD_M0) * g.tileK1 * xBytes;
    const int l0a2 = Align(g.mMax2, MMAD_M0) * g.tileK2 * midBytes;
    if (l0a1 > L0A_SLOT_BYTES || l0a2 > L0A_SLOT_BYTES) {
        return RJ_TILE_K;
    }

    const int l0b1 = Align(p.cout1, MMAD_M0) * g.l0bChunkK1 * weightBytes;
    const int l0b2 = Align(p.cout2, MMAD_M0) * g.l0bChunkK2 * weightBytes;
    const int use1 = l0b1 * (g.l0bPP1 != 0 ? 2 : 1);
    const int use2 = l0b2 * (g.l0bPP2 != 0 ? 2 : 1);
    const int l0bMax = MaxI(use1, use2);
    if (l0bMax > L0B_BYTES) {
        return RJ_L0B;
    }
    g.l0bElems = l0bMax / weightBytes;

    g.l0bSlot1Elems = (g.l0bPP1 != 0) ? l0b1 / weightBytes : 0;
    g.l0bSlot2Elems = (g.l0bPP2 != 0) ? l0b2 / weightBytes : 0;

    const int l0c1 = Align(g.mMax1, MMAD_M0) * Align(p.cout1, MMAD_M0);
    const int l0c2 = Align(g.mMax2, MMAD_M0) * Align(p.cout2, MMAD_M0);

    if (MaxI(l0c1, l0c2) > L0C_ELEMS) {
        return RJ_L0C;
    }

    g.l0cStride = (2 * l0c1 <= L0C_ELEMS) ? l0c1 : 0;
    g.l0cElems = MaxI(g.l0cStride + l0c1, l0c2);

    if (p.cout1 > BT_SLOT_ELEMS || p.cout2 > BT_SLOT_ELEMS) {
        return RJ_BT;
    }

    g.fmElems = g.xRows * g.wInMax * p.ci;
    g.midElems = g.m1Max * p.cout1;
    const int w1Full = Align(p.cout1, MMAD_M0) * g.k1;
    const int w2Full = Align(p.cout2, MMAD_M0) * g.k2;

    const int w1Ring = 2 * Align(p.cout1, MMAD_M0) * g.l0bChunkK1;
    const int w2Ring = 2 * Align(p.cout2, MMAD_M0) * g.l0bChunkK2;
    const bool w1CanStage = (g.l0bChunks1 > 1) && (w1Ring < w1Full);
    const bool w2CanStage = (g.l0bChunks2 > 1) && (w2Ring < w2Full);

    const bool w2Bigger = (w2Full - w2Ring) >= (w1Full - w1Ring);
    int modes[4];
    modes[0] = 0;
    modes[1] = w2Bigger ? 2 : 1;
    modes[2] = w2Bigger ? 1 : 2;
    modes[3] = 3;

    const int budget = (l1Budget > 0) ? l1Budget : 0x7fffffff;
    int used = -1;
    for (int mi = 0; mi < 4; ++mi) {
        const int mode = modes[mi];
        const int s1 = (mode & 1) != 0 ? 1 : 0;
        const int s2 = (mode & 2) != 0 ? 1 : 0;
        if ((s1 != 0 && !w1CanStage) || (s2 != 0 && !w2CanStage)) {
            continue;
        }
        g.w1Stage = s1;
        g.w2Stage = s2;
        g.w1Elems = (s1 != 0) ? w1Ring : w1Full;
        g.w2Elems = (s2 != 0) ? w2Ring : w2Full;
        LayOutL1(p, g, xBytes, midBytes, weightBytes, biasBytes);
        if (g.l1Used <= budget) {
            used = mode;
            break;
        }
    }
    if (used < 0) {
        return RJ_L1;
    }

    g.xPlane = p.ci * p.hi * p.wi;
    g.yPlane = p.cout2 * g.ho2 * g.wo2;
    return RJ_OK;
}

// 估算最忙核心的计算和搬运周期。
inline long long CoreCostCycles(const Geometry& g, int chunksPerCore)
{
    FlatGeom f;
    Flatten(g, chunksPerCore, f);
    const int cout1A = Align(g.p.cout1, MMAD_M0);
    const int cout2A = Align(g.p.cout2, MMAD_M0);
    const int xBytes = g.p.elemBytes;
    const int midBytes = MidElemBytes(g.p);
    const int wBytes = WeightElemBytes(g.p);

    const int yBytes = g.p.hasDeqScale2 ? 2 : midBytes;
    const int kPerMmad1 = (xBytes == 2) ? 16 : 32;
    const int kPerMmad2 = (midBytes == 2) ? 16 : 32;

    int bands[3] = {0, g.nchunk / 2, g.nchunk - 1};
    int wsegs[3] = {0, g.nwseg / 2, g.nwseg - 1};
    long long wCube = 0, wMte1 = 0, wMte2 = 0, wFix = 0, wScalar = 0;
    for (int bi = 0; bi < 3; ++bi) {
        for (int wj = 0; wj < 3; ++wj) {
            const int band = bands[bi];
            const int woSegIdx = wsegs[wj];
            if (band < 0 || woSegIdx < 0) {
                continue;
            }
            BandLite b;
            MakeBandLite(f, band, woSegIdx, b);
            long long cube = 0;
            for (int i = 0; i < b.conv1Tile.n; ++i) {
                cube += (long long)(Align(TileRows(b.conv1Tile, i), MMAD_M0) / MMAD_M0) * (cout1A / MMAD_M0) *
                        (g.k1 / kPerMmad1);
            }
            for (int r = 0; r < b.nRun; ++r) {
                for (int q = 0; q < b.conv2Tile.n; ++q) {
                    cube += (long long)(Align(TileRows(b.conv2Tile, q), MMAD_M0) / MMAD_M0) * (cout2A / MMAD_M0) *
                            (g.k2 / kPerMmad2);
                }
            }
            long long mte1Bytes = (long long)b.conv1MPerBand * g.k1 * xBytes +
                                  (long long)b.nRun * b.mRun * g.k2 * midBytes;
            mte1Bytes += (g.l0bChunks1 == 1) ? (long long)cout1A * g.k1 * wBytes :
                                               (long long)b.conv1Tile.n * cout1A * g.k1 * wBytes;
            mte1Bytes += (g.l0bChunks2 == 1) ? (long long)cout2A * g.k2 * wBytes :
                                               (long long)b.nRun * b.conv2Tile.n * cout2A * g.k2 * wBytes;
            const long long mte1 = mte1Bytes / MTE1_BYTES_PER_CYC;

            const long long fmBytes = (long long)b.xRows * b.wIn * g.p.ci * xBytes;
            long long mte2 = fmBytes / MTE2_BYTES_PER_CYC;
            if (b.wIn != g.p.wi) {
                mte2 += (long long)b.xRows * g.p.ci * MTE2_ROW_REQ_CYC;
            }

            long long fix = (long long)b.conv1MPerBand * cout1A * 4 / FIX_L1_BYTES_PER_CYC;
            const long long gmReq = (long long)b.nRun * b.conv2Tile.n * cout2A;
            const long long gmBytes = (long long)b.nRun * b.mRun * cout2A * yBytes;
            fix += gmReq * FIX_GM_REQ_CYC_X2 / 2 + gmBytes / FIX_GM_BYTES_PER_CYC;

            long long wRing = 0;
            if (g.w1Stage != 0) {
                wRing += (long long)b.conv1Tile.n * cout1A * g.k1 * wBytes;
            }
            if (g.w2Stage != 0) {
                wRing += (long long)b.nRun * b.conv2Tile.n * cout2A * g.k2 * wBytes;
            }
            mte2 += wRing / MTE2_BYTES_PER_CYC;
            const long long scalar = (long long)(b.conv1Tile.n + b.nRun * b.conv2Tile.n) * SCALAR_TILE_CYC;
            if (cube > wCube) {
                wCube = cube;
            }
            if (mte1 > wMte1) {
                wMte1 = mte1;
            }
            if (mte2 > wMte2) {
                wMte2 = mte2;
            }
            if (fix > wFix) {
                wFix = fix;
            }
            if (scalar > wScalar) {
                wScalar = scalar;
            }
        }
    }

    long long wtBytes = 0;
    if (g.w1Stage == 0) {
        wtBytes += (long long)cout1A * g.k1 * wBytes;
    }
    if (g.w2Stage == 0) {
        wtBytes += (long long)cout2A * g.k2 * wBytes;
    }
    return (long long)chunksPerCore * (wCube + wMte1 + wMte2 + wFix + wScalar) + wtBytes / MTE2_BYTES_PER_CYC +
           SCALAR_BASE_CYC;
}

// 搜索行带高和列段数，选择估算周期最小的有效几何。
inline int PickHb(const Params& p, int aicNum, int l1Budget, Geometry& g)
{
    if (aicNum <= 0) {
        aicNum = 1;
    }
    Geometry probe;

    if (p.elemBytes != 1 && p.elemBytes != 2) {
        return RJ_ELEM_BYTES;
    }
    if (p.n <= 0 || p.ci <= 0 || p.hi <= 0 || p.wi <= 0 || p.cout1 <= 0 || p.cout2 <= 0 || p.kh1 <= 0 || p.kw1 <= 0 ||
        Kh2(p) <= 0 || Kw2(p) <= 0 || p.stride1 <= 0 || p.stride2 <= 0 || p.padH1 < 0 || p.padW1 < 0 || p.padH2 < 0 ||
        p.padW2 < 0) {
        return RJ_SHAPE_POSITIVE;
    }
    const int ho1 = (p.hi + 2 * p.padH1 - p.kh1) / p.stride1 + 1;
    if (ho1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int ho2 = (ho1 + 2 * p.padH2 - Kh2(p)) / p.stride2 + 1;
    if (ho2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    const int wo1 = (p.wi + 2 * p.padW1 - p.kw1) / p.stride1 + 1;
    if (wo1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int wo2 = (wo1 + 2 * p.padW2 - Kw2(p)) / p.stride2 + 1;
    if (wo2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    int bestHb = 0;
    long long bestCost = 0;
    int bestStage = 3;
    int firstReal = RJ_NO_HB;
    for (int hb = 1; hb <= ho2; ++hb) {
        if ((ho2 % hb) != 0) {
            continue;
        }
        const int rc = DeriveWith(p, hb, 1, l1Budget, probe);
        if (rc != RJ_OK) {
            if (firstReal == RJ_NO_HB && rc != RJ_HB_DIVIDE) {
                firstReal = rc;
            }
            continue;
        }
        const long long rounds = CeilDiv(probe.chunkTotal, aicNum);
        const long long cost = rounds * (long long)probe.midRows;

        const int stg = probe.w1Stage + probe.w2Stage;
        if (bestHb == 0 || stg < bestStage ||
            (stg == bestStage && (cost < bestCost || (cost == bestCost && hb > bestHb)))) {
            bestHb = hb;
            bestCost = cost;
            bestStage = stg;
        }
    }

    if (bestHb != 0) {
        const long long ct1 = (long long)p.n * (ho2 / bestHb);
        if (ct1 >= (long long)aicNum && (ct1 % (long long)aicNum) == 0) {
            return DeriveWith(p, bestHb, 1, l1Budget, g);
        }
    }

    int bestHb2 = 0;
    int bestNw = 1;
    const int nwMax = MinI(aicNum, wo2);

    long long bn1[4] = {-1, -1, -1, -1};
    long long bng[4] = {-1, -1, -1, -1};
    for (int hb = 1; hb <= ho2; ++hb) {
        if ((ho2 % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= nwMax; ++nw) {
            const int rc = DeriveWith(p, hb, nw, l1Budget, probe);
            if (rc != RJ_OK) {
                if (firstReal == RJ_NO_HB && rc != RJ_HB_DIVIDE && rc != RJ_WSEG) {
                    firstReal = rc;
                }
                continue;
            }
            const int bdCap = (aicNum < probe.chunkTotal) ? aicNum : probe.chunkTotal;
            const int chunksPerCore = CeilDiv(probe.chunkTotal, bdCap);
            const long long cost = CoreCostCycles(probe, chunksPerCore);
            const int stg = probe.w1Stage + probe.w2Stage;
            if (nw == 1) {
                if (bn1[stg] < 0 || cost < bn1[stg]) {
                    bn1[stg] = cost;
                }
            } else {
                if (bng[stg] < 0 || cost < bng[stg]) {
                    bng[stg] = cost;
                }
            }
        }
    }

    int useStage = -1;
    for (int st = 0; st < 4; ++st) {
        if (bn1[st] >= 0 || bng[st] >= 0) {
            useStage = st;
            break;
        }
    }
    if (useStage >= 0) {
        const long long nw1Best = bn1[useStage];
        const long long nwgBest = bng[useStage];

        const bool nwgOk = (nwgBest >= 0) && (nw1Best < 0 || nwgBest * 5 <= nw1Best * 4);
        long long adminMin = -1;
        if (nw1Best >= 0) {
            adminMin = nw1Best;
        }
        if (nwgOk && (adminMin < 0 || nwgBest < adminMin)) {
            adminMin = nwgBest;
        }

        const long long cutoff = adminMin + adminMin / 50;

        int keyChunksPerCore = 0, keyNw = 0, keyHb = 0;
        for (int hb = 1; hb <= ho2; ++hb) {
            if ((ho2 % hb) != 0) {
                continue;
            }
            for (int nw = 1; nw <= nwMax; ++nw) {
                if (DeriveWith(p, hb, nw, l1Budget, probe) != RJ_OK) {
                    continue;
                }
                if (probe.w1Stage + probe.w2Stage != useStage) {
                    continue;
                }
                if (nw > 1 && !nwgOk) {
                    continue;
                }
                const int bdCap = (aicNum < probe.chunkTotal) ? aicNum : probe.chunkTotal;
                const int chunksPerCore = CeilDiv(probe.chunkTotal, bdCap);
                if (CoreCostCycles(probe, chunksPerCore) > cutoff) {
                    continue;
                }
                if (bestHb2 == 0 || chunksPerCore < keyChunksPerCore ||
                    (chunksPerCore == keyChunksPerCore && (nw < keyNw || (nw == keyNw && hb > keyHb)))) {
                    bestHb2 = hb;
                    bestNw = nw;
                    keyChunksPerCore = chunksPerCore;
                    keyNw = nw;
                    keyHb = hb;
                }
            }
        }
    }
    if (bestHb2 == 0) {
        if (bestHb == 0) {
            return firstReal;
        }
        return DeriveWith(p, bestHb, 1, l1Budget, g);
    }
    return DeriveWith(p, bestHb2, bestNw, l1Budget, g);
}

} // namespace FusedMultiConv2dShape

#endif
