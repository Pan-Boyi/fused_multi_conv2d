#!/bin/bash
# 这个脚本已经被 fc2d.py 取代 —— singleop.json 现在从 case.bin 的 spec 现生成，
# 不再手写，也就不会和 .bin 的形状对不上了。
echo "build_om.sh 已废弃。用："
echo "    source <CANN>/set_env.sh"
echo "    python3 fc2d.py cases.json --steps om --soc <soc_version>"
echo "soc_version 的合法取值 = \$ASCEND_HOME_PATH/compiler/data/platform_config 下每个 .ini 的文件名。"
echo "见 README.md。"
exit 1
