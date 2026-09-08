#!/bin/bash
# 这个脚本已经被 fc2d.py 取代 —— 形状不再写死在 gen_case 里，而是来自 cases.json。
echo "build_case.sh 已废弃。用："
echo "    g++ -std=c++17 -O2 -ffp-contract=off gen_case.cpp -o gen_case -I."
echo "    python3 fc2d.py cases.json --steps check,case"
echo "见 README.md。"
exit 1
