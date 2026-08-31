/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file golden_selfcheck.cpp
 * \brief golden 自检的入口。纯 host，不碰设备，不依赖 CANN。
 *
 *   g++ -std=c++17 -O2 golden_selfcheck.cpp -o golden_selfcheck && ./golden_selfcheck
 *
 * 为什么要单独一个 .cpp，而不是直接 `g++ -x c++ fused_conv2d_int8_golden.h`：
 * 后者依赖编译器把 .h 当作源文件来处理，不同 gcc 版本/发行版对这一点的行为并不一致，
 * 表现出来就是一句和 golden 毫无关系的
 *     undefined reference to `main'
 * 用一个真正的 .cpp 把宏定义在 include 之前，就没有这个歧义了。
 */

// y 走 VDEQF16 出 fp16。这个宏只选 GoldenOutput()/OutElem 的类型，跟
// gold.yI8 / gold.yF16 的直接取用无关，但留成 int8 会误导后来人。
#define FUSED_CONV2D_GOLDEN_INT8_OUT 0
#define FUSED_CONV2D_GOLDEN_FP16_OUT 1
#define FUSED_CONV2D_GOLDEN_MAIN 1

#include "fused_conv2d_int8_golden.h"
