#!/usr/bin/env bash
# 使用 AutoDL 镜像的 CUDA PyTorch 和无卡准备时安装的组件依赖。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /root/miniconda3/bin/python -u -c '
import runpy
import site
import sys

site.addsitedir(".venv-cpu/lib/python3.12/site-packages")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' scripts/autodl_artbench_cache.py "$@"
