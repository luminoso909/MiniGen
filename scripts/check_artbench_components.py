"""以 CPU FP32 为参考，验收本机 MPS 上的冻结 SDXL VAE 与 CLIP。"""

import argparse
import json
from pathlib import Path

import torch

from minigen.data.representation_cache import cache_source
from minigen.evaluation.local_components import run_local_acceptance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """用固定 20 张真实训练图片和 Caption 执行双组件验收。

    Raises:
        SystemExit: 命令行参数无效。
        OSError/ValueError/RuntimeError: 本机 CPU 基准、固定资产或回执写入失败。

    副作用:
        设置当前脚本进程的 CPU 线程数，运行本机推理，创建通过回执并打印路径。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=5)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.output.exists():
        parser.error("--cpu-threads must be positive; --output must not exist")
    torch.set_num_threads(args.cpu_threads)
    vae_lock = ROOT / "configs/models/sdxl_vae.json"
    clip_lock = ROOT / "configs/models/clip_text.json"
    source = cache_source(
        ROOT / "data/processed/artbench_pd_256/v1",
        ROOT / "reports/v2/data/artbench_freeze.json",
        vae_lock,
        clip_lock,
        None,
    )
    receipt = run_local_acceptance(
        ROOT,
        source,
        ROOT / "reports/v2/components/vae_reconstruction_200_summary.json",
        ROOT / "artifacts/evaluations/p2_vae_artbench_200",
        vae_lock,
        clip_lock,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Local ArtBench-20 acceptance passed: {receipt['devices']} -> {args.output}")


if __name__ == "__main__":
    main()
