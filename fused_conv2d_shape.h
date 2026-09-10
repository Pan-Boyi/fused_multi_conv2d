// ===========================================================================
// 这是 ops-nn 仓里 conv/fused_conv2d/op_kernel/fused_conv2d_shape.h 的**副本**。
//
// 放一份在 harness 里，只为了 fc2d.py 能在生成 case 之前就回答「这个形状能不能
// 跑」—— 否则要等到 atc 失败，而那条原因只在 ~/ascend/log/ 里。
//
// **算子那边改了就要重拷。** 一条命令：
//     python3 fc2d.py --sync-shape-header <ops-nn>/conv/fused_conv2d/op_kernel/fused_conv2d_shape.h
//
// 它是顾问性质的：真正说了算的永远是算子的 tiling。两边不一致时 fc2d.py 的预检会
// 和 atc 的结论不同，那本身就是「该重拷了」的信号。
// ===========================================================================
/*
 * fused_conv2d 的**运行期几何** —— host 和 device 共用的唯一出处。
 *
 * 这个文件存在的理由是一次真实的事故：改 kernel 几何时把 FC2D_HB 从 18 改成 6，
 * 却漏了 op_host/op_tiling/fused_conv2d_tiling.cpp 里那份同名的 HB = 18。
 * tiling 于是告诉 kernel「一共 8 个 chunk、每核 1 个」，而 kernel 按自己的
 * NCHUNK = 24 分带 —— 每个核只算了 1 个 chunk，输出正好只写了 1/3，上板才发现。
 * 更糟的是 tiling UT 里断言的是 chunkTotal == 8，把错误的期望一起编码进去了。
 *
 * 所以：**凡是 host 和 device 都要知道的数，只能从这里算出来。**
 * 现在这条纪律比以前更要紧 —— 形状不再是编译期常量，两边算的是同一个
 * DeriveWithHb()，谁也不许在自己那边另推一份。
 *
 * ---------------------------------------------------------------------------
 * 这一版做了什么
 * ---------------------------------------------------------------------------
 * 上一版所有形状（CI/HI/WI/COUT1/COUT2/HB/...）都是 constexpr，一个二进制只服务
 * 一个 shape。这一版把它们全变成运行期参数：
 *
 *     Params      调用方给的形状 + 卷积超参 + 元素宽度
 *     Geometry    由 Params 和一个 hb 推出来的全部派生量（L1 地址表、分块表、
 *                 K 切分、L0B 折叠、chunk 数……）
 *     Band        某一个 chunk 的运行期窗口（pad、行数、两张分块表）
 *
 * host 的 tiling 调 PickHb() 选带高，再把 Params + hb 下发；kernel 拿到之后调
 * 同一个 DeriveWithHb() 复原整套几何。两边不可能分家 —— 这正是上面那次事故的
 * 结构性修复。
 *
 * 约束：这个头会被 host 编译器编（op_host/op_tiling），所以里面不能出现
 * __aicore__、half、或任何 AscendC 的东西 —— 只有纯 C++。
 * 先例：conv3d_transpose_v2 的 tiling.cpp 也是这样 include op_kernel 下的头的。
 * 也不能用 <algorithm> / <cmath>：device 侧那边没有。
 */
#ifndef FUSED_CONV2D_SHAPE_H
#define FUSED_CONV2D_SHAPE_H

// ---------------------------------------------------------------------------
// host 和 device 共用一份函数体，但两边对函数的要求不一样。
//
// device 侧（ccec）要求被 kernel 调用的函数带 [aicore] 属性 —— 不带属性的函数被
// 当成 **host 函数**，根本不生成 device 代码，于是链接期报
//     ld.lld: error: undefined symbol: FusedConv2dShape::DeriveWithHb(...)
// 这条错误完全不提「少了个属性」，只说符号不见了。
//
// host 侧（普通 g++ 编 op_tiling）当然没有这个属性，写上去编不过。
//
// __NPU_ARCH__ 只有 kernel 那条编译链会定义（geometry.h 顶上就是靠它做 5102 门禁），
// host 那条不会 —— 所以用它分。kernel 那条里 CANN 的头总是先于本头被包含，
// __aicore__ 一定已经可用。
// ---------------------------------------------------------------------------
#if defined(__NPU_ARCH__)
#define FC2D_GEOM_FN __aicore__ inline
#define FC2D_GEOM_CE __aicore__ constexpr
#else
#define FC2D_GEOM_FN inline
#define FC2D_GEOM_CE constexpr
#endif

namespace FusedConv2dShape {

// ---------------------------------------------------------------------------
// 5102 的片上容量。**只有这一组仍然是编译期常量** —— 它们是芯片属性，不是形状。
// L1 是例外：真机 1 MB，而 CPU 仿真把它截在 512 KB
// （kernel_utils_mode_cpu.h:283），所以 L1 预算是 PickHb() 的入参，由调用方给。
// ---------------------------------------------------------------------------
constexpr int L0A_BYTES = 64 * 1024;
constexpr int L0A_SLOT_BYTES = L0A_BYTES / 2; // 双缓冲，一槽 32,768
constexpr int L0B_BYTES = 64 * 1024;
constexpr int L0C_BYTES = 256 * 1024;
constexpr int L0C_ELEMS = L0C_BYTES / 4; // int32 累加器
constexpr int BT_BYTES = 4 * 1024;
constexpr int BT_SLOT_ELEMS = 512;    // 一个 bias 槽最多几个通道
constexpr int BT_SLOT1_BYTE_OFF = 2048;
constexpr int L1_SEG_ALIGN = 512;     // fixpipe 的 L1->FB 取数要求每段 512 对齐
constexpr int MMAD_M0 = 16;           // cube 的 M 粒度；N 粒度也是 16

// M 方向最多切几块。分块表是定长数组，超了就拒 —— 与其在 device 上越界，不如在
// tiling 阶段说「这个形状我服务不了」。
constexpr int MAX_M_TILES = 64;

// 定点定标 S 的合法区间（A16W16）。见 conv/common/op_kernel/arch35/conv_util.h:153。
constexpr int FIX_SHIFT_LEN_A16W16 = 58;

// ---------------------------------------------------------------------------
// 小工具。<algorithm> 在 device 侧不能用，所以自己写。
// ---------------------------------------------------------------------------
// 全部 constexpr：geometry.h 里有几条 static_assert 要在编译期调它们。
FC2D_GEOM_CE int MinI(int a, int b) { return a < b ? a : b; }
FC2D_GEOM_CE int MaxI(int a, int b) { return a > b ? a : b; }
FC2D_GEOM_CE int CeilDiv(int a, int b) { return (a + b - 1) / b; }
FC2D_GEOM_CE int Align(int a, int m) { return (a + m - 1) / m * m; }
FC2D_GEOM_CE int GcdI(int a, int b)
{
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// 两个 bias 段在 L1 里各占一段，长度按**真实的 bias 宽度**补齐到 512 的整数倍。
// 写死 512 是错的：int8 通路的 bias 是 int32，cout=256 就要 1024 字节，写死之后
// bias2 会踩到 bias1 后面 512 字节之外的地方 —— 而那里正好是 L1 的末尾，越界。
// fp16 通路的 bias 是 half，512 字节够 256 个通道，所以这个坑只有 int8 大通道
// 才够得着。
FC2D_GEOM_CE int BiasElemBytes(int elemBytes) { return elemBytes == 2 ? 2 : 4; }
FC2D_GEOM_CE int BiasSegBytes(int cout, int elemBytes)
{
    return Align(cout * BiasElemBytes(elemBytes), L1_SEG_ALIGN);
}

// ---------------------------------------------------------------------------
// 调用方给的东西。elemBytes 是特征图 / 权重的元素宽度：2 = fp16 定点，1 = int8
// 量化。C0 由它决定（一个 C0 恒等于 32 字节），于是 C1、FRACTAL_Z 的维度、L1 占用
// 全跟着变 —— 两条通路的差别在这个头里只有这一个入口。
// ---------------------------------------------------------------------------
struct Params {
    int n;         // batch
    int ci;        // 输入通道
    int hi;
    int wi;
    int cout1;     // conv1 输出通道 == conv2 输入通道
    int cout2;     // conv2 输出通道 == y 的通道
    int kh;        // 两个卷积共用同一个 kernel 大小
    int kw;
    int stride1;
    int stride2;
    // pad 分 H / W 两个方向，每个方向上下（左右）对称。非方核必须能分开给 ——
    // 比如 1x3 的 separable conv 要 padH = 0、padW = 1，一个标量表达不了。
    int padH1, padW1;
    int padH2, padW2;
    int elemBytes; // 2 = fp16, 1 = int8
    // conv2 出口带不带 per-channel 反量化表（int8 进 / fp16 出那条通路）。
    // 它只影响 L1 —— 表要常驻 L1 给 fixpipe 取，所以必须在这里，不能只当成
    // 一个运行期开关。0 / 1。
    int hasDeqScale2;
};

// M 方向的分块。**整个算子只有这一条分块公式**：把 total 行按 gran 切成 units 个
// 单位，均分成 n 块，余数摊给靠前的块。
//   maxRows  一块最多几行 —— 由 L0A 一槽和 L0C 共同定
//   gran     行数粒度 —— rows*Wo 必须 16 对齐，所以 gran = 16 / gcd(Wo, 16)
//
// 上一版把它展开成一张 129 个 int 的定长表（TileTable）在 host 和 device 之间传，
// 而且**每个 band 现搭两张**。上板量出来这条路很贵：MakeTiles 每次要清 128 个 int、
// 按值返回又是 516 字节的拷贝，几何推导整体把 scalar 从 3.8us 抬到 18.9us。
// 现在 device 侧只拿三个标量（n / base / extra），第 i 块的行数和起点用两次乘法
// 现算 —— 表只在 host 和 UT 里展开，那边多花几百条指令无所谓。
struct TileSpec {
    int n;     // 分几块。**0 表示这个组合不成立**，调用方必须判
    int base;  // 每块至少几个单位
    int extra; // 前 extra 块各多一个单位
    int gran;  // 一个单位几行
};

FC2D_GEOM_CE TileSpec MakeTileSpec(int total, int maxRows, int gran)
{
    // 聚合初始化，不是先声明再赋值：constexpr 函数里默认初始化的局部量
    // 「永远产生不了常量表达式」，clang 直接把整个函数判成非法的 constexpr。
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

// 第 i 块有几行 / 从第几行开始。device 侧就用这两个，不建表。
FC2D_GEOM_CE int TileRows(const TileSpec& s, int i) { return (s.base + (i < s.extra ? 1 : 0)) * s.gran; }
FC2D_GEOM_CE int TileRow0(const TileSpec& s, int i) { return (s.base * i + MinI(i, s.extra)) * s.gran; }

// 展开的表。**只有 host 和 UT 用**（tiling 的逐 band 校验、golden 对账）。
// 它必须和 TileRows/TileRow0 逐项相等 —— 这一条由 UT 钉住，不靠人看。
struct TileTable {
    int n;
    int rows[MAX_M_TILES];
    int row0[MAX_M_TILES];
};

FC2D_GEOM_FN TileTable MakeTiles(int total, int maxRows, int gran)
{
    const TileSpec s = MakeTileSpec(total, maxRows, gran);
    TileTable t;
    t.n = s.n;
    for (int i = 0; i < MAX_M_TILES; ++i) {
        t.rows[i] = (i < s.n) ? TileRows(s, i) : 0;
        t.row0[i] = (i < s.n) ? TileRow0(s, i) : 0;
    }
    return t;
}

// ---------------------------------------------------------------------------
// 拒绝的原因。tiling 把它打进日志 —— 「GRAPH_FAILED」本身什么也不说明。
//
// 前缀是 RJ_ 而不是更自然的 R_：POSIX 的 <unistd.h> 把 R_OK / W_OK / X_OK / F_OK
// 定义成**宏**（访问权限位），host 侧的 tiling 一路 include 下去必然带进来，
// 于是 `R_OK = 0` 展开成 `4 = 0`，报的是「expected identifier before numeric
// constant」，和枚举本身看不出关系。
// ---------------------------------------------------------------------------
enum Reject {
    RJ_OK = 0,
    RJ_ELEM_BYTES,     // elemBytes 只能是 1 或 2
    RJ_SHAPE_POSITIVE, // 有维度 <= 0
    RJ_CHANNEL_C0,     // ci 或 cout1 不是 C0 的整数倍
    RJ_CHANNEL_16,     // cout1 不是 16 的整数倍（mid 要按 NC1HWC0 读）
    RJ_OUT_EMPTY,      // 卷出来是空的
    RJ_HB_DIVIDE,      // hb 不整除 ho2
    RJ_HB_GRAN,        // 【已不再产生】M 切到行内之后没有行粒度这回事了
    RJ_MIDROWS,        // 一个 band 的 conv1 行数超过了 ho1
    RJ_MID_GRAN,       // 【已不再产生】同上
    RJ_TILE_K,         // 找不到合法的 K 切分
    RJ_M_TILES,        // M 方向切出来的块数超过 MAX_M_TILES
    RJ_L0C,            // 一个 M 子块装不进 L0C
    RJ_L0B,            // 权重按 K 折之后仍然装不进 L0B
    RJ_BT,             // bias 放不进一个 BT 槽
    RJ_L1,             // L1 装不下
    RJ_U16,            // 某个指令字段是 16 位的，这个形状超了
    RJ_NO_HB,          // 所有 hb 都不成立
    RJ_WSEG,           // 列分段数不合法（<= 0 或 > wo2）
};

// **只在 host 和 CPU 仿真上存在。**
// ccec 把字符串字面量放在 __gm__ 地址空间里，`const __gm__ char[N]` 转不成
// `const char*`，所以这个函数在真机编译链上根本编不过。它也没必要存在于那里：
// 唯一的消费者是 host 的 tiling 日志和 kernel 里那条 ASCENDC_CPU_DEBUG 的 printf。
#if !defined(__NPU_ARCH__) || (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
FC2D_GEOM_FN const char* RejectText(int r)
{
    switch (r) {
        case RJ_OK: return "ok";
        case RJ_ELEM_BYTES: return "elemBytes 只能是 1(int8) 或 2(fp16)";
        case RJ_SHAPE_POSITIVE: return "形状里有非正数";
        case RJ_CHANNEL_C0: return "ci 和 cout1 必须是 C0 的整数倍（C0 = 32/elemBytes）";
        case RJ_CHANNEL_16: return "cout1 必须是 16 的整数倍（mid 要被 conv2 当 NC1HWC0 读）";
        case RJ_OUT_EMPTY: return "卷积输出为空（kernel 比补零后的输入还大）";
        case RJ_HB_DIVIDE: return "band 高度不整除 ho2";
        case RJ_HB_GRAN: return "band 高度不是 conv2 行粒度的整数倍（hb*wo2 要 16 对齐）";
        case RJ_MIDROWS: return "一个 band 需要的 conv1 行数超过了 ho1";
        case RJ_MID_GRAN: return "某个 band 的 conv1 行数 * wo1 不是 16 的整数倍";
        case RJ_TILE_K: return "找不到既整除 K 又装得进 L0A 的 K 切分";
        case RJ_M_TILES: return "M 方向的块数超过 MAX_M_TILES";
        case RJ_L0C: return "一个 M 子块装不进 L0C";
        case RJ_L0B: return "权重按 K 折之后仍然装不进 L0B";
        case RJ_BT: return "bias 的通道数超过一个 BT 槽（512）";
        case RJ_L1: return "L1 装不下 fm + mid + w1 + w2";
        case RJ_U16: return "指令字段只有 16 位：mid 的 M（midRows*wo1）或 staging 的位置数（xRows*wi）超过 65535";
        case RJ_NO_HB: return "没有任何 band 高度能同时满足全部约束";
        case RJ_WSEG: return "列分段数不合法（必须在 1..wo2 之间）";
        default: return "未知";
    }
}
#endif // RejectText 只在 host / CPU 仿真上存在

// ---------------------------------------------------------------------------
// 由 Params + 一个 band 高度推出来的全部派生量。
// ---------------------------------------------------------------------------
struct Geometry {
    Params p;

    int c0, c1, midC1;
    int ho1, wo1, ho2, wo2;
    int k1, k2;          // 两个卷积的矩阵乘 K = C1*kh*kw*C0

    int hb, nchunk;      // 每个 band 出多少行 conv2 输出 / 一张图几个 band
    // 列方向的分核。一个 chunk = 一个 band x 一个列段，所以
    //   chunkTotal = n * nchunk * nwseg
    // Ho2 只有一行的形状（比如 5x1790 那个）在行方向一点并行度都没有，
    // 不切列就只有一个核在干活。
    int wseg, nwseg;     // 一个列段出多少列 conv2 输出 / 一张图几个列段
    int wSubMax, wInMax; // 一个列段最多要算多少列 mid / 要灌多少列 x（L1 按它排）
    int chunkTotal;      // n * nchunk * nwseg

    int midRows;         // 一个 band 的 conv1 输出行数上限（= 窗口跨度）
    int xRows;           // 一个 band 要灌进 L1 的 x 行数上限
    int m1Max;           // midRows * wSubMax，mid 的 NZ M 上限

    int tileK1, tileK2;              // 每条 img2col 灌多少 K
    int mMax1, mMax2;                // 一个 M 子块最多几个位置（不是行）
    int l0bChunks1, l0bChunkK1, tilesPerChunk1;  // 权重按 K 折几段
    int l0bChunks2, l0bChunkK2, tilesPerChunk2;

    int l0aSlotElems;    // 一个 L0A 槽的元素数（按 elemBytes）
    int l0bElems;        // L0B 要多少元素（两个卷积取大）
    int l0cElems;        // 共享 L0C 的元素数（int32）

    // L1 常驻段，字节地址（每段 512 对齐）
    int fmElems, midElems, w1Elems, w2Elems;
    int fmAddr, midAddr, w1Addr, w2Addr, b1Addr, b2Addr;
    int b1Bytes, b2Bytes;
    int dq2Addr, dq2Bytes;   // conv2 的 per-channel 反量化表（不带时长度 0）
    int l1Used;

    // GM 侧一张图的元素数，供 batch 偏移和 UT 对账用
    int xPlane, yPlane;
    // FRACTAL_Z 的四个维度，tiling 拿它校验 filter 的 shape
    int fz1K, fz1N, fz2K, fz2N;
};

// K 切分和 M 大小是互相牵制的：tileK 越大，一个 L0A 槽装得下的 M 越小。
// 这里对每个候选 tileK（C0 的整数倍且整除 K）算出对应的 mMax，取「M 块数最少、
// 并列时 tileK 最大」的那个 —— M 块数直接等于 fixpipe 的轮数。
//
// **M 的单位是「位置」，不是「行」。** 上一版按整行切，于是一行放不进 L0C 的形状
// （比如 wo1 = 1788、cout1 = 64：align16(1788)*64 = 114,688 > 65,536）根本无解，
// 而且行数还得是 16/gcd(wo,16) 的整数倍。切到行内之后这两条限制都不存在了 ——
// 依据是 tests/ut/op_kernel 里那两条探针：img2col 的 mStartPt/mExtension 和
// fixpipe 的 mSize/strip 跨距都**不要求** 16 对齐（CPU 仿真结论，真机待确认）。
struct KChoice {
    int tileK;
    int mMax;    // 一个 M 子块最多几个**位置**（不是行）
    int nTiles;
    bool ok;
};

FC2D_GEOM_FN KChoice PickTileK(int k, int c0, int elemBytes, int cout, int totalM)
{
    KChoice best;
    best.tileK = 0;
    best.mMax = 0;
    best.nTiles = 0;
    best.ok = false;
    if (k <= 0 || c0 <= 0 || cout <= 0 || totalM <= 0) {
        return best;
    }
    // L0C 的上限和 tileK 无关：align16(m) * align16(cout) <= 65,536 个 int32。
    const int mCapC = L0C_ELEMS / Align(cout, MMAD_M0);
    for (int tk = c0; tk <= k; tk += c0) {
        if ((k % tk) != 0) {
            continue;
        }
        // L0A 一槽：align16(m) * tileK * elemBytes <= 32,768
        const int mCapA = L0A_SLOT_BYTES / (tk * elemBytes);
        // 往下取到 16 的整数倍：容量算的是 align16(m)，m 本身不对齐时会往上补，
        // 取整之后「m <= mMax」就等价于「align16(m) <= 容量」，不用再判一次。
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
        // 并列时取更大的 tileK：一条 mmad 干更多活，指令数更少。
        if (!best.ok || nT < best.nTiles || (nT == best.nTiles && tk > best.tileK)) {
            best.tileK = tk;
            best.mMax = mMax;
            best.nTiles = nT;
            best.ok = true;
        }
    }
    return best;
}

// 整套权重能不能一次装进 L0B；装不下就按 K 折。折出来的每段仍要是 tileK 的整数倍，
// 否则 img2col 的 K 偏移和 L0B 里的偏移对不上。
FC2D_GEOM_FN bool PickL0bChunks(int cout, int k, int elemBytes, int tileK, int& chunks, int& chunkK)
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
        if (coutA * ck * elemBytes <= L0B_BYTES) {
            chunks = c;
            chunkK = ck;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// tiling 下发的**扁平几何**。
//
// 上一版下发的是 Params + hb，kernel 自己调 DeriveWithHb() 复原整套几何。那在
// 「一个二进制服务所有形状」这件事上是对的，但代价上板才看得见：DeriveWithHb 里
// 有两轮 PickTileK（各几十次试探，每次四五个整数除法）、两轮 PickL0bChunks、
// 一遍逐 band 校验，加起来五百多个标量除法 —— scalar 耗时从 3.8us 涨到 18.9us，
// 而这些数**对一次 launch 而言全是常量**。
//
// 所以现在：host 把 Geometry 里 kernel 用得到的字段全抄进这个结构体下发，kernel
// 逐字段读成 const 局部量，一次除法都不做。
//
// 「host 和 device 只有一份几何」这条纪律没有松动，只是搬了个位置：算它的仍然只有
// DeriveWithHb() 一处，kernel 不再重算，也就更不可能算出和 host 不同的答案。
// 逐 band 那点窗口算术（四个 pad、行数、conv1 的分块）留在 device 上现算 ——
// 它随 chunk 变，本来就不是一个常量，而且只有加减乘和三次除法。
//
// 下面的 FC2D_FLAT_FIELDS 是这些字段的**唯一清单**：tiling 的下发、kernel 的解包、
// kernel UT 那份手抄的 tiling 结构体，三处都由它展开。漏一个字段或写错一个名字都
// 编不过，不会变成「读到相邻字段的值、不报错只算错」。
// ---------------------------------------------------------------------------
struct FlatGeom {
    // ---- 形状（img2col / Dn2Nz 直接要）----
    int n, ci, hi, wi, cout1, cout2, kh, kw, stride1, stride2;
    int padH1, padW1, padH2, padW2, elemBytes;
    // ---- 由形状推出来的 ----
    int c0, ho1, wo1, ho2, wo2, k1, k2;
    int hb, nchunk, chunkTotal, chunksPerCore;
    int wseg, nwseg;             // 列方向的分核，见 Geometry
    int midRows;                 // 一个 band 的窗口跨度（含补零）
    int mMax1, mMax2;            // 两个卷积的 M 子块上限（位置数），逐 chunk 现算用
    int tileK1, tileK2;
    int l0bChunks1, l0bChunkK1, tilesPerChunk1;
    int l0bChunks2, l0bChunkK2, tilesPerChunk2;
    int l0bElems, l0cElems;
    // ---- L1 地址表（字节地址）----
    int fmAddr, fmElems, midAddr, midElems, w1Addr, w1Elems, w2Addr, w2Elems, b1Addr, b2Addr;
    int dq2Addr;   // conv2 的 per-channel 反量化表（不带时这一段长度为 0）
    // ---- GM 侧 ----
    int xPlane, yPlane, yPlaneM;
};

// FlatGeom 的字段清单：X(FlatGeom 里的名字, tiling 里的名字)。
// 顺序就是 tiling 结构体的声明顺序 —— kernel UT 那份手抄的 packed 结构体靠它对齐
// 布局（那边是按字节 memcpy 的，顺序错了就全错）。host tiling 和 kernel 走的是
// 具名的 set_/成员访问，顺序对它们无所谓，但一处清单总比三处手抄可靠。
#define FC2D_FLAT_FIELDS(X)   \
    X(n, batch)               \
    X(ci, cin)                \
    X(hi, hin)                \
    X(wi, win)                \
    X(cout1, cout1)           \
    X(cout2, cout2)           \
    X(kh, kh)                 \
    X(kw, kw)                 \
    X(stride1, stride1)       \
    X(stride2, stride2)       \
    X(padH1, padH1)           \
    X(padW1, padW1)           \
    X(padH2, padH2)           \
    X(padW2, padW2)           \
    X(elemBytes, elemBytes)   \
    X(c0, c0)                 \
    X(ho1, ho1)               \
    X(wo1, wo1)               \
    X(ho2, ho2)               \
    X(wo2, wo2)               \
    X(k1, k1)                 \
    X(k2, k2)                 \
    X(hb, hb)                 \
    X(nchunk, nchunk)         \
    X(wseg, wseg)             \
    X(nwseg, nwseg)           \
    X(chunkTotal, chunkTotal) \
    X(chunksPerCore, chunksPerCore) \
    X(midRows, midRows)       \
    X(mMax1, mMax1)           \
    X(mMax2, mMax2)           \
    X(tileK1, tileK1)         \
    X(tileK2, tileK2)         \
    X(l0bChunks1, l0bChunks1) \
    X(l0bChunkK1, l0bChunkK1) \
    X(tilesPerChunk1, tilesPerChunk1) \
    X(l0bChunks2, l0bChunks2) \
    X(l0bChunkK2, l0bChunkK2) \
    X(tilesPerChunk2, tilesPerChunk2) \
    X(l0bElems, l0bElems)     \
    X(l0cElems, l0cElems)     \
    X(fmAddr, fmAddr)         \
    X(fmElems, fmElems)       \
    X(midAddr, midAddr)       \
    X(midElems, midElems)     \
    X(w1Addr, w1Addr)         \
    X(w1Elems, w1Elems)       \
    X(w2Addr, w2Addr)         \
    X(w2Elems, w2Elems)       \
    X(b1Addr, b1Addr)         \
    X(b2Addr, b2Addr)         \
    X(dq2Addr, dq2Addr)       \
    X(xPlane, xPlane)         \
    X(yPlane, yPlane)         \
    X(yPlaneM, yPlaneM)

FC2D_GEOM_FN void Flatten(const Geometry& g, int chunksPerCore, FlatGeom& f)
{
    f.n = g.p.n;
    f.ci = g.p.ci;
    f.hi = g.p.hi;
    f.wi = g.p.wi;
    f.cout1 = g.p.cout1;
    f.cout2 = g.p.cout2;
    f.kh = g.p.kh;
    f.kw = g.p.kw;
    f.stride1 = g.p.stride1;
    f.stride2 = g.p.stride2;
    f.padH1 = g.p.padH1;
    f.padW1 = g.p.padW1;
    f.padH2 = g.p.padH2;
    f.padW2 = g.p.padW2;
    f.elemBytes = g.p.elemBytes;

    f.c0 = g.c0;
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
    f.l0bElems = g.l0bElems;
    f.l0cElems = g.l0cElems;

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

    f.xPlane = g.xPlane;
    f.yPlane = g.yPlane;
    f.yPlaneM = g.ho2 * g.wo2;
}

// ---------------------------------------------------------------------------
// 某一个 band 的运行期窗口。**device 侧每个 chunk 算一次的就是这个**，全部是
// 加减乘和三次除法（MakeTileSpec 里的），没有数组、没有大结构体拷贝。
//
// 分带算术：
//   band b 出 conv2 的输出行 [b*hb, (b+1)*hb)
//   这些行的窗口覆盖 conv1 的输出行 [aMid, aMid + midRows)，raw = stride2*hb*b - pad2
//   其中真实存在的是和 [0, ho1) 的交集 —— 头尾两端各有可能被钳
//
// mid 在 L1 里只放这个 band 真实存在的那些行（midReal 行），所以
//   conv1 的 fixpipe   dst_M = midReal * wo1
//   conv2 的 img2col   l1H   = midReal，上下用 padT2 / padB2 补回 midRows
// 两者必须同时改 —— 这是「融合点的布局恒等式」，写偏一个 strip 就全错位。
// ---------------------------------------------------------------------------
struct BandLite {
    int aMid;     // 这个 band 的第一行 conv1 输出（全局行号）
    int midReal;  // 这个 band 真正算出来的 conv1 行数
    int padT2, padB2;
    int xRow0, xRows, padT1, padB1;
    int outRow0;  // 这个 chunk 的第一行 conv2 输出（图内行号）
    // ---- 列方向的窗口。和上面的行方向是同一套钳位算术，只是换成 W ----
    int outCol0, outCols;   // 这个 chunk 出哪些列 conv2 输出
    int midCol0, wSub;      // 要算的 mid 列区间（conv1 在这一段里的输出宽度）
    int padL2, padR2;       // conv2 在这一段两侧的补零（内部段是 0，边缘段才有）
    int xCol0, wIn;         // 要灌进 L1 的 x 列区间
    int padL1, padR1;
    int m1;       // midReal * wSub
    // conv2 的输出在 GM(NCHW) 里按行连续。整幅宽度时一个 run 覆盖整个 band（和
    // 不切列时完全一样）；切了列之后一个 run 只能是一行 —— fixpipe 的一段连续
    // 写要是跨了行，就会写到下一行的开头去。
    int rowsPerRun, mRun, nRun;
    TileSpec t1;  // conv1 的 M 分块（对 midReal x wSub 的全部位置）
    TileSpec t2;  // conv2 的 M 分块（对一个 run 的全部位置）
};

FC2D_GEOM_FN void MakeBandLite(const FlatGeom& f, int band, int wg, BandLite& b)
{
    b.outRow0 = band * f.hb;

    const int raw = f.stride2 * f.hb * band - f.padH2;
    b.aMid = MaxI(0, raw);
    b.padT2 = b.aMid - raw;
    const int wantRows = f.midRows - b.padT2;          // 窗口里还需要几行真实的 conv1 输出
    b.midReal = MinI(wantRows, f.ho1 - b.aMid);
    b.padB2 = f.midRows - b.padT2 - b.midReal;

    // conv1 侧：产出 mid 行 [aMid, aMid+midReal) 需要的 x 行区间，两头都可能被钳。
    const int xRaw = f.stride1 * b.aMid - f.padH1;
    const int xSpan = f.stride1 * (b.midReal - 1) + f.kh;
    b.xRow0 = MaxI(0, xRaw);
    b.padT1 = b.xRow0 - xRaw;
    const int xEnd = MinI(f.hi, xRaw + xSpan);
    b.xRows = xEnd - b.xRow0;
    b.padB1 = xRaw + xSpan - xEnd;

    // ---- 列方向 ----------------------------------------------------------
    // 出 conv2 的列 [outCol0, outCol0+outCols) 需要 mid 的列
    //   [outCol0*s2 - padW2, (outCol0+outCols-1)*s2 - padW2 + kw)
    // 和 [0, wo1) 求交，交不到的部分就是这一段自己的左右补零。恒等式：
    //   (wSub + padL2 + padR2 - kw)/s2 + 1 == outCols
    // conv1 那一层同理，出 wSub 列需要 x 的哪一段。
    b.outCol0 = wg * f.wseg;
    b.outCols = MinI(f.wseg, f.wo2 - b.outCol0);

    const int mcRaw0 = b.outCol0 * f.stride2 - f.padW2;
    const int mcRaw1 = (b.outCol0 + b.outCols - 1) * f.stride2 - f.padW2 + f.kw;
    b.midCol0 = MaxI(0, mcRaw0);
    const int mcEnd = MinI(f.wo1, mcRaw1);
    b.padL2 = b.midCol0 - mcRaw0;
    b.padR2 = mcRaw1 - mcEnd;
    b.wSub = mcEnd - b.midCol0;

    const int icRaw0 = b.midCol0 * f.stride1 - f.padW1;
    const int icRaw1 = (mcEnd - 1) * f.stride1 - f.padW1 + f.kw;
    b.xCol0 = MaxI(0, icRaw0);
    const int icEnd = MinI(f.wi, icRaw1);
    b.padL1 = b.xCol0 - icRaw0;
    b.padR1 = icRaw1 - icEnd;
    b.wIn = icEnd - b.xCol0;

    // conv1 的 M 空间：midReal 行 x wSub 列的**全部位置**，可以切到行内 —— mid 在
    // L1 里就是这么连续排的（行距 wSub），所以扁平区间直接对应一块矩形。
    b.m1 = b.midReal * b.wSub;
    b.t1 = MakeTileSpec(b.m1, f.mMax1, 1);

    b.rowsPerRun = (b.outCols == f.wo2) ? f.hb : 1;
    b.mRun = b.rowsPerRun * b.outCols;
    b.nRun = f.hb / b.rowsPerRun;
    b.t2 = MakeTileSpec(b.mRun, f.mMax2, 1);
}

// ---------------------------------------------------------------------------
// 某一个 chunk 的运行期窗口。**host 和 device 都调这个**，band 的 pad / 行数 /
// 分块表只有这一份定义。
//
// 分带算术：
//   band b 出 conv2 的输出行 [b*hb, (b+1)*hb)
//   这些行的窗口覆盖 conv1 的输出行 [aRaw, aRaw + midRows)，aRaw = stride2*hb*b - pad2
//   其中真实存在的是和 [0, ho1) 的交集 —— 头尾两端各有可能被钳
//
// mid 在 L1 里只放这个 band 真实存在的那些行（midReal 行），所以
//   conv1 的 fixpipe   dst_M = midReal * wo1
//   conv2 的 img2col   l1H   = midReal，上下用 padT2 / padB2 补回 midRows
// 两者必须同时改 —— 这是「融合点的布局恒等式」，写偏一个 strip 就全错位。
// ---------------------------------------------------------------------------
struct Band {
    int img;      // batch 里的第几张图
    int band;     // 图内第几个 band
    int wg;       // 图内第几个列段
    int aMid;     // 这个 band 的第一行 conv1 输出（全局行号）
    int midReal;  // 这个 band 真正算出来的 conv1 行数
    int padT2, padB2;
    int xRow0, xRows, padT1, padB1;
    int outRow0;  // 这个 chunk 的第一行 conv2 输出（图内行号）
    int outCol0, outCols, midCol0, wSub, padL2, padR2, xCol0, wIn, padL1, padR1;
    int rowsPerRun, mRun, nRun;
    int m1;       // midReal * wSub
    TileTable t1; // conv1 的 M 分块
    TileTable t2; // conv2 的 M 分块（对一个 run）
};

FC2D_GEOM_FN Band MakeBand(const Geometry& g, int chunk)
{
    FlatGeom f;
    Flatten(g, 1, f);
    // chunk 的拆法：列段跑得最快，其次是 band，最后是图。kernel 侧那个自增
    // （wg -> band -> img）必须和这里一致。
    const int perImg = g.nchunk * g.nwseg;
    const int img = chunk / perImg;
    const int rem = chunk - img * perImg;
    const int band = rem / g.nwseg;
    const int wg = rem - band * g.nwseg;
    BandLite lb;
    MakeBandLite(f, band, wg, lb);

    Band b;
    b.img = img;
    b.band = band;
    b.wg = wg;
    b.aMid = lb.aMid;
    b.midReal = lb.midReal;
    b.padT2 = lb.padT2;
    b.padB2 = lb.padB2;
    b.xRow0 = lb.xRow0;
    b.xRows = lb.xRows;
    b.padT1 = lb.padT1;
    b.padB1 = lb.padB1;
    b.outRow0 = lb.outRow0;
    b.outCol0 = lb.outCol0;
    b.outCols = lb.outCols;
    b.midCol0 = lb.midCol0;
    b.wSub = lb.wSub;
    b.padL2 = lb.padL2;
    b.padR2 = lb.padR2;
    b.xCol0 = lb.xCol0;
    b.wIn = lb.wIn;
    b.padL1 = lb.padL1;
    b.padR1 = lb.padR1;
    b.rowsPerRun = lb.rowsPerRun;
    b.mRun = lb.mRun;
    b.nRun = lb.nRun;
    b.m1 = lb.m1;
    b.t1 = MakeTiles(lb.m1, g.mMax1, 1);
    b.t2 = MakeTiles(lb.mRun, g.mMax2, 1);
    return b;
}

// ---------------------------------------------------------------------------
// 核心：Params + hb + 列段数 -> Geometry。返回 0 表示成立，否则是 Reject 码。
// l1Budget 是这次允许用多少 L1（真机 1 MB，CPU 仿真 512 KB）。
// ---------------------------------------------------------------------------
FC2D_GEOM_FN int DeriveWith(const Params& p, int hb, int nwseg, int l1Budget, Geometry& g)
{
    if (p.elemBytes != 1 && p.elemBytes != 2) {
        return RJ_ELEM_BYTES;
    }
    if (p.n <= 0 || p.ci <= 0 || p.hi <= 0 || p.wi <= 0 || p.cout1 <= 0 || p.cout2 <= 0 || p.kh <= 0 ||
        p.kw <= 0 || p.stride1 <= 0 || p.stride2 <= 0 || p.padH1 < 0 || p.padW1 < 0 || p.padH2 < 0 ||
        p.padW2 < 0 || hb <= 0) {
        return RJ_SHAPE_POSITIVE;
    }

    g.p = p;
    g.c0 = 32 / p.elemBytes;
    if ((p.ci % g.c0) != 0 || (p.cout1 % g.c0) != 0) {
        return RJ_CHANNEL_C0;
    }
    // cout1 要 16 对齐（其实更严：下面 ci/cout1 还要是 C0 的整数倍）—— mid 要被
    // conv2 当 NC1HWC0 读，通道必须整分成 C0 一组。
    //
    // **cout2 不需要。** 它只出现在 conv2 的出口，fixpipe 按 COLUMN_MAJOR 直接写
    // 到 GM，nSize 给多少写多少；L0B / L0C 那边用的是 align16(cout2)。上一版把它
    // 和 cout1 一起卡了 16，于是 cout2 = 2 这种（FRACTAL_Z 里补齐成 [.., 1, 16, C0]）
    // 直接被拒 —— 而 filter 的 shape 根本分不出 cout2 是 1 还是 16，得从 y 去取。
    if ((p.cout1 % MMAD_M0) != 0) {
        return RJ_CHANNEL_16;
    }
    g.c1 = p.ci / g.c0;
    g.midC1 = p.cout1 / g.c0;

    g.ho1 = (p.hi + 2 * p.padH1 - p.kh) / p.stride1 + 1;
    g.wo1 = (p.wi + 2 * p.padW1 - p.kw) / p.stride1 + 1;
    if (g.ho1 <= 0 || g.wo1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    g.ho2 = (g.ho1 + 2 * p.padH2 - p.kh) / p.stride2 + 1;
    g.wo2 = (g.wo1 + 2 * p.padW2 - p.kw) / p.stride2 + 1;
    if (g.ho2 <= 0 || g.wo2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    g.k1 = g.c1 * p.kh * p.kw * g.c0;
    g.k2 = g.midC1 * p.kh * p.kw * g.c0;
    g.fz1K = g.c1 * p.kh * p.kw;
    g.fz1N = CeilDiv(p.cout1, MMAD_M0);
    g.fz2K = g.midC1 * p.kh * p.kw;
    g.fz2N = CeilDiv(p.cout2, MMAD_M0);

    if ((g.ho2 % hb) != 0) {
        return RJ_HB_DIVIDE;
    }
    if (nwseg <= 0 || nwseg > g.wo2) {
        return RJ_WSEG;
    }
    g.hb = hb;
    g.nchunk = g.ho2 / hb;
    // 段宽先定，段数再由它反推 —— wo2 不被 nwseg 整除时最后一段短一点，
    // 段数也可能比请求的少（比如 wo2=10 要 6 段，wseg=2 只能分出 5 段）。
    g.wseg = CeilDiv(g.wo2, nwseg);
    g.nwseg = CeilDiv(g.wo2, g.wseg);
    g.chunkTotal = p.n * g.nchunk * g.nwseg;

    // 一个 band 的 conv2 输出行覆盖 midRows 行 conv1 输出（含上下补零）。
    g.midRows = p.stride2 * (hb - 1) + p.kh;
    // L1 按**实际可能出现的最大值**排，不按窗口跨度排：任何一个 band 真正算出来的
    // conv1 行数都不会超过 ho1（超出的部分是 conv2 的补零，不是数据），staging 的
    // x 行数同理不会超过 hi。按跨度排会白占 L1，还会把本来放得下的形状拒掉。
    const int midRowsCap = MinI(g.midRows, g.ho1);
    // 列方向同理，按**最宽的那个段**排 L1。内部段两侧都没有补零，所以它最宽；
    // 边缘段的一部分窗口落在补零里，真实列数只会更少。
    // 顺带一个副作用：不切列（nwseg = 1）时 wSubMax 也可能小于 wo1 —— stride2
    // 大于 kw 时 conv2 根本读不到 conv1 最后那几列，以前是白算的。
    g.wSubMax = MinI(g.wo1, (g.wseg - 1) * p.stride2 + p.kw);
    g.wInMax = MinI(p.wi, (g.wSubMax - 1) * p.stride1 + p.kw);
    g.m1Max = midRowsCap * g.wSubMax;
    g.xRows = MinI(p.stride1 * (midRowsCap - 1) + p.kh, p.hi);
    // 两个 16 位字段：img2col 的 mStartPt（最大到 m1Max）和 Dn2Nz 的 nValue /
    // dstNzC0Stride（都等于 xRows*wi）。超了会静默截断成完全不相干的地址。
    if (g.m1Max > 65535 || (long long)g.xRows * g.wInMax > 65535) {
        return RJ_U16;
    }

    // K 切分 + M 分块。conv1 对一个 chunk 的 midReal x wSub，conv2 对一个 run。
    // 不切列时一个 run 就是整个 band（hb 行 x wo2 列），和以前一模一样。
    const int mRunMax = (g.nwseg == 1) ? (hb * g.wo2) : g.wseg;
    const KChoice kc1 = PickTileK(g.k1, g.c0, p.elemBytes, p.cout1, g.m1Max);
    const KChoice kc2 = PickTileK(g.k2, g.c0, p.elemBytes, p.cout2, mRunMax);
    if (!kc1.ok || !kc2.ok) {
        return RJ_TILE_K;
    }
    g.tileK1 = kc1.tileK;
    g.mMax1 = kc1.mMax;
    g.tileK2 = kc2.tileK;
    g.mMax2 = kc2.mMax;

    if (!PickL0bChunks(p.cout1, g.k1, p.elemBytes, g.tileK1, g.l0bChunks1, g.l0bChunkK1) ||
        !PickL0bChunks(p.cout2, g.k2, p.elemBytes, g.tileK2, g.l0bChunks2, g.l0bChunkK2)) {
        return RJ_L0B;
    }
    g.tilesPerChunk1 = g.l0bChunkK1 / g.tileK1;
    g.tilesPerChunk2 = g.l0bChunkK2 / g.tileK2;

    // 每个 band 的 conv1 行数都必须让 rows*wo1 保持 16 对齐，且分得出块。
    // 只有三种取值（首、中、尾），但直接全扫一遍最省心 —— nchunk 至多几百。
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

    // 片上缓冲。L0A 按最大的那个子块分，两个卷积取大。
    const int l0a1 = Align(g.mMax1, MMAD_M0) * g.tileK1 * p.elemBytes;
    const int l0a2 = Align(g.mMax2, MMAD_M0) * g.tileK2 * p.elemBytes;
    if (l0a1 > L0A_SLOT_BYTES || l0a2 > L0A_SLOT_BYTES) {
        return RJ_TILE_K;
    }
    g.l0aSlotElems = L0A_SLOT_BYTES / p.elemBytes;

    const int l0b1 = Align(p.cout1, MMAD_M0) * g.l0bChunkK1 * p.elemBytes;
    const int l0b2 = Align(p.cout2, MMAD_M0) * g.l0bChunkK2 * p.elemBytes;
    const int l0bMax = MaxI(l0b1, l0b2);
    if (l0bMax > L0B_BYTES) {
        return RJ_L0B;
    }
    g.l0bElems = l0bMax / p.elemBytes;

    const int l0c1 = Align(g.mMax1, MMAD_M0) * Align(p.cout1, MMAD_M0);
    const int l0c2 = Align(g.mMax2, MMAD_M0) * Align(p.cout2, MMAD_M0);
    g.l0cElems = MaxI(l0c1, l0c2);
    if (g.l0cElems > L0C_ELEMS) {
        return RJ_L0C;
    }

    if (p.cout1 > BT_SLOT_ELEMS || p.cout2 > BT_SLOT_ELEMS) {
        return RJ_BT;
    }

    // L1 地址表。每段 512 对齐 —— fixpipe 的 L1->FB 取数要求。
    g.fmElems = g.xRows * g.wInMax * p.ci;
    g.midElems = g.m1Max * p.cout1;
    g.w1Elems = Align(p.cout1, MMAD_M0) * g.k1;
    g.w2Elems = Align(p.cout2, MMAD_M0) * g.k2;

    g.fmAddr = 0;
    g.midAddr = Align(g.fmAddr + g.fmElems * p.elemBytes, L1_SEG_ALIGN);
    g.w1Addr = Align(g.midAddr + g.midElems * p.elemBytes, L1_SEG_ALIGN);
    g.w2Addr = Align(g.w1Addr + g.w1Elems * p.elemBytes, L1_SEG_ALIGN);
    g.b1Addr = Align(g.w2Addr + g.w2Elems * p.elemBytes, L1_SEG_ALIGN);
    g.b1Bytes = BiasSegBytes(p.cout1, p.elemBytes);
    g.b2Addr = g.b1Addr + g.b1Bytes;
    g.b2Bytes = BiasSegBytes(p.cout2, p.elemBytes);
    // per-channel 反量化表：一个通道一个 uint64。fixpipe 取的是 L1 地址，所以它
    // 得常驻。不带这条通路时长度 0，一个字节都不占 —— 否则每个形状都要为一个
    // 用不上的段让出 4 KB，本来放得下的会被拒掉。
    g.dq2Addr = g.b2Addr + g.b2Bytes;
    g.dq2Bytes = (p.hasDeqScale2 != 0) ? Align(p.cout2 * (int)sizeof(long long), L1_SEG_ALIGN) : 0;
    g.l1Used = g.dq2Addr + g.dq2Bytes;
    if (l1Budget > 0 && g.l1Used > l1Budget) {
        return RJ_L1;
    }

    g.xPlane = p.ci * p.hi * p.wi;
    g.yPlane = p.cout2 * g.ho2 * g.wo2;
    return RJ_OK;
}

// 老签名的壳：不切列。UT 和离线扫盘还按 hb 单独取几何。
FC2D_GEOM_FN int DeriveWithHb(const Params& p, int hb, int l1Budget, Geometry& g)
{
    return DeriveWith(p, hb, 1, l1Budget, g);
}

// ---------------------------------------------------------------------------
// band 高度的选择。**只有 host 走这条**，选完把 hb 下发，kernel 只调 DeriveWithHb。
//
// 目标函数：总耗时正比于「轮数 x 每个 band 的 conv1 行数」。
//   轮数        = ceil(nchunk / aicNum)   —— 核不够时要跑几轮
//   每 band 行数 = midRows                —— band 之间重叠的那几行是重复计算
// hb 越大重复越少，但 nchunk 越小、核越闲。这个乘积把两者一起权衡。
// ---------------------------------------------------------------------------
FC2D_GEOM_FN int PickHb(const Params& p, int aicNum, int l1Budget, Geometry& g)
{
    if (aicNum <= 0) {
        aicNum = 1;
    }
    Geometry probe;
    // 先算一次 ho2，才知道候选集。用 hb = 1 探一次；它可能因为别的原因被拒，
    // 但 ho2 在被拒之前就已经算出来了，所以单独把这两行摊开算。
    if (p.elemBytes != 1 && p.elemBytes != 2) {
        return RJ_ELEM_BYTES;
    }
    if (p.n <= 0 || p.ci <= 0 || p.hi <= 0 || p.wi <= 0 || p.cout1 <= 0 || p.cout2 <= 0 || p.kh <= 0 ||
        p.kw <= 0 || p.stride1 <= 0 || p.stride2 <= 0 || p.padH1 < 0 || p.padW1 < 0 || p.padH2 < 0 ||
        p.padW2 < 0) {
        return RJ_SHAPE_POSITIVE;
    }
    const int ho1 = (p.hi + 2 * p.padH1 - p.kh) / p.stride1 + 1;
    if (ho1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int ho2 = (ho1 + 2 * p.padH2 - p.kh) / p.stride2 + 1;
    if (ho2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    const int wo1 = (p.wi + 2 * p.padW1 - p.kw) / p.stride1 + 1;
    if (wo1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int wo2 = (wo1 + 2 * p.padW2 - p.kw) / p.stride2 + 1;
    if (wo2 <= 0) {
        return RJ_OUT_EMPTY;
    }

    // ---- 第一趟：不切列。这一趟和加 W 分核之前**逐字相同** ------------------
    // 报哪一条拒绝原因是有讲究的：整除性（RJ_HB_DIVIDE / RJ_HB_GRAN）对绝大多数候选
    // hb 都成立，报它等于什么也没说。留住第一条**别的**原因 —— 那才是这个形状真正
    // 卡在哪里。一条都没有就说「没有任何 band 高度成立」。
    int bestHb = 0;
    long long bestCost = 0;
    int firstReal = RJ_NO_HB;
    for (int hb = 1; hb <= ho2; ++hb) {
        if ((ho2 % hb) != 0) {
            continue;
        }
        const int rc = DeriveWith(p, hb, 1, l1Budget, probe);
        if (rc != RJ_OK) {
            if (firstReal == RJ_NO_HB && rc != RJ_HB_DIVIDE && rc != RJ_HB_GRAN) {
                firstReal = rc;
            }
            continue;
        }
        const long long rounds = CeilDiv(probe.chunkTotal, aicNum);
        const long long cost = rounds * (long long)probe.midRows;
        if (bestHb == 0 || cost < bestCost || (cost == bestCost && hb > bestHb)) {
            bestHb = hb;
            bestCost = cost;
        }
    }
    // bestHb == 0 不在这里返回 —— 切列还能把 L1 救回来（每段的 fm / mid 都窄一截），
    // 所以一条 hb 都不成立时也要往下走第二趟。
    // **行方向已经喂得满核就到此为止。** 切列不是免费的：每段两侧各多算 kw-1 列的
    // halo，每段还要各重灌一次两套权重，而且 conv2 的一个 run 缩成一行，mmad 的 M
    // 变小、fixpipe 的次数变多。所以它是「行方向不够分」时的补救，不是优化。
    // 这一条同时保证了老形状的分核方式一个字都不变。
    if (bestHb != 0 && (long long)p.n * (ho2 / bestHb) >= (long long)aicNum) {
        return DeriveWith(p, bestHb, 1, l1Budget, g);
    }

    // ---- 第二趟：行方向填不满核，才把 W 也拿出来切 --------------------------
    int bestHb2 = 0;
    int bestNw = 1;
    long long bestCost2 = 0;
    const int nwMax = MinI(aicNum, wo2);
    for (int hb = 1; hb <= ho2; ++hb) {
        if ((ho2 % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= nwMax; ++nw) {
            const int rc = DeriveWith(p, hb, nw, l1Budget, probe);
            if (rc != RJ_OK) {
                if (firstReal == RJ_NO_HB && rc != RJ_HB_DIVIDE && rc != RJ_HB_GRAN &&
                    rc != RJ_WSEG) {
                    firstReal = rc;
                }
                continue;
            }
            // 一个 chunk 的代价 = conv1 要算的位置数（band 重叠和列 halo 都在里面）
            //                   + 每个 chunk 都要重灌一次的两套权重。
            // 后面那一项是防止「越切越好」的：只看第一项的话切得越碎每块越小，代价
            // 单调下降，最后会切成一堆只搬权重的碎片。
            const long long rounds = CeilDiv(probe.chunkTotal, aicNum);
            const long long perChunk =
                (long long)probe.midRows * probe.wSubMax + probe.k1 + probe.k2;
            const long long cost = rounds * perChunk;
            if (bestHb2 == 0 || cost < bestCost2 ||
                (cost == bestCost2 && (nw < bestNw || (nw == bestNw && hb > bestHb2)))) {
                bestHb2 = hb;
                bestNw = nw;
                bestCost2 = cost;
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

} // namespace FusedConv2dShape

#endif // FUSED_CONV2D_SHAPE_H
