#!/usr/bin/env bash
# 沿用已跑通 ArtBench 的 AutoDL 镜像 PyTorch 和持久盘上的 MiniGen 依赖。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /root/miniconda3/bin/python -u -c '
import runpy
import site
import sys

site.addsitedir(".venv-cpu/lib/python3.12/site-packages")
sys.path.insert(0, "scripts")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' scripts/run_anime_b1.py "$@"
