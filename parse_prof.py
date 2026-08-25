#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""把一次 msprof 采集里的设备侧算子耗时打出来。

    python3 parse_prof.py <msprof 的 --output 目录> [op_type]

不同 CANN 版本落盘的位置不一样，这里按下面的顺序找，用上第一个有数据的：

    1. */mindstudio_profiler_output/op_summary_*.csv     （--export=on 之后）
    2. */device_*/summary/op_summary_*.csv
    3. */device_*/sqlite/ai_core_op_summary.db 的 task_time 表
    4. PROF 目录下任何带 task_time 表的 .db

只依赖标准库，目标机上不需要 pandas / numpy。
"""

import csv
import glob
import os
import sqlite3
import statistics
import sys


def stats(name, unit, vals, extra=""):
    v = sorted(vals)
    print("  %-22s n=%-5d min=%9.3f  p50=%9.3f  mean=%9.3f  max=%9.3f  %s%s"
          % (name, len(v), v[0], statistics.median(v), sum(v) / len(v), v[-1], unit, extra))


def find_prof_dir(root):
    cands = sorted(glob.glob(os.path.join(root, "PROF_*")))
    return cands[-1] if cands else root


def pick_col(fields, *needles):
    """按关键字模糊找列名。不同版本的表头大小写和括号都不一样。"""
    for f in fields:
        low = f.lower()
        if all(n in low for n in needles):
            return f
    return None


def from_csv(prof, op_type):
    pats = [
        os.path.join(prof, "**", "mindstudio_profiler_output", "op_summary_*.csv"),
        os.path.join(prof, "**", "device_*", "summary", "op_summary_*.csv"),
        os.path.join(prof, "**", "op_summary_*.csv"),
    ]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    if not files:
        return False

    for path in files:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        fields = list(rows[0].keys())
        c_dur = pick_col(fields, "task", "duration") or pick_col(fields, "duration")
        c_type = pick_col(fields, "op", "type")
        c_name = pick_col(fields, "op", "name")
        c_tt = pick_col(fields, "task", "type")
        if c_dur is None:
            continue

        def match(r):
            if op_type is None:
                return True
            for c in (c_type, c_name):
                if c and op_type.lower() in str(r.get(c, "")).lower():
                    return True
            return False

        sel = [r for r in rows if match(r)]
        if not sel and op_type is not None:
            print("  [!] %s 里没有 %s 的行，退回统计全部算子" % (os.path.basename(path), op_type))
            sel = rows
        durs = []
        for r in sel:
            try:
                durs.append(float(r[c_dur]))
            except (TypeError, ValueError):
                pass
        if not durs:
            continue

        print("来源: %s" % path)
        print("  列: 时长=%r  op类型=%r  task类型=%r" % (c_dur, c_type, c_tt))
        unit = "us" if "us" in c_dur.lower() else ("ns" if "ns" in c_dur.lower() else "?")
        stats("全部匹配行", unit, durs)
        if c_tt:
            by = {}
            for r in sel:
                try:
                    by.setdefault(str(r.get(c_tt, "?")), []).append(float(r[c_dur]))
                except (TypeError, ValueError):
                    pass
            for k, v in sorted(by.items()):
                stats("  task_type=%s" % k, unit, v)
        # aicore 指标：列名各版本不一，凡是带 ratio 的都打个均值。
        # 注意 "Duration" 里本身就含 "ratio"（Du-ratio-n），得排掉，否则会把时长列
        # 当成一个利用率指标又打一遍。
        for f in fields:
            low = f.lower()
            if "ratio" in low and "duration" not in low:
                vv = []
                for r in sel:
                    try:
                        vv.append(float(r[f]))
                    except (TypeError, ValueError):
                        pass
                if vv:
                    print("  %-22s mean=%.4f" % (f, sum(vv) / len(vv)))
        return True
    return False


def from_sqlite(prof, op_type):
    dbs = sorted(set(
        glob.glob(os.path.join(prof, "**", "ai_core_op_summary.db"), recursive=True)
        + glob.glob(os.path.join(prof, "**", "*.db"), recursive=True)))
    for db in dbs:
        try:
            con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
            tabs = [t[0] for t in con.execute(
                "select name from sqlite_master where type='table'")]
            if "task_time" not in tabs:
                con.close()
                continue
            cols = [c[1] for c in con.execute("pragma table_info(task_time)")]
            c_dur = "duration_time" if "duration_time" in cols else None
            c_tt = "task_type" if "task_type" in cols else None
            if c_dur is None:
                con.close()
                continue
            sel = "select %s, %s from task_time" % (c_tt or "'?'", c_dur)
            rows = list(con.execute(sel))
            con.close()
        except sqlite3.Error as e:
            print("  [!] 读 %s 失败: %s" % (db, e))
            continue
        if not rows:
            continue

        print("来源: %s  (task_time，duration_time 单位是纳秒)" % db)
        by = {}
        for t, d in rows:
            try:
                by.setdefault(str(t), []).append(float(d) / 1000.0)
            except (TypeError, ValueError):
                pass
        for t, v in sorted(by.items()):
            stats("task_type=%s" % t, "us", v)
        # *_SQE 才是真正跑 kernel 的那条；PLACE_HOLDER_SQE 是占位，不算。
        kern = [d for t, v in by.items() if t.endswith("_SQE") and not t.startswith("PLACE")
                for d in v]
        if kern:
            print()
            stats("kernel 合计", "us", kern, extra=" 总设备时间 %.3f us" % sum(kern))
        return True
    return False


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    op_type = sys.argv[2] if len(sys.argv) > 2 else "FusedConv2d"
    if not os.path.isdir(root):
        print("[X] %s 不是目录" % root)
        return 1
    prof = find_prof_dir(root)
    print("PROF 目录: %s" % prof)
    print("算子过滤 : %s\n" % op_type)

    if from_csv(prof, op_type):
        return 0
    if from_sqlite(prof, op_type):
        return 0

    print("[X] 在 %s 下既没找到 op_summary*.csv，也没找到带 task_time 表的 .db。" % prof)
    print("    看看目录里到底有什么：")
    n = 0
    for dirpath, _, files in os.walk(prof):
        for f in files:
            print("      %s" % os.path.join(dirpath, f))
            n += 1
            if n >= 40:
                print("      ...（只列前 40 个）")
                return 1
    if n == 0:
        print("      （空的 —— 采集根本没落盘，看 msprof 自己的报错）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
