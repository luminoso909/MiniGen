"""构建或验证 P2 的 ArtBench 表示缓存，全程使用固定本地资产。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from minigen.autoencoders.sdxl import FrozenVAE
from minigen.conditioning.clip import FrozenCLIPText
from minigen.data.representation_cache import (
    SPLITS,
    RepresentationCache,
    build_cache,
    cache_source,
    verify_review,
)

ROOT = Path(__file__).resolve().parents[1]
VAE = ROOT / "configs/models/sdxl_vae.json"
CLIP = ROOT / "configs/models/clip_text.json"


def main() -> None:
    """用正式或试点模式打开固定 256px 数据，输出可核验的本地摘要。

    Raises:
        SystemExit: 参数错误，或请求的设备不可用。
        ValueError: 身份、审核或缓存校验失败。
        OSError: 本地数据、权重或目标目录不可读写。

    副作用:
        设置本进程 CPU 线程数；写新缓存时创建目录并输出 JSON 摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/processed/artbench_pd_256/v1"
    )
    parser.add_argument(
        "--freeze", type=Path, default=ROOT / "reports/v2/data/artbench_freeze.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot-rows", type=int, help="每个划分最多缓存的样本数")
    mode.add_argument("--approval", type=Path, help="正式缓存要求用户审核 JSON")
    parser.add_argument("--evidence", type=Path, help="与审核回执匹配的 200 张重建 JSON")
    parser.add_argument("--verify-only", action="store_true", help="只检查已有缓存，不加载编码器")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1:
        parser.error("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    start = perf_counter()
    source = cache_source(args.data_root, args.freeze, VAE, CLIP, args.pilot_rows)
    receipt = verify_review(source, args.approval, args.evidence)
    if not args.verify_only:
        if args.device == "mps" and not torch.backends.mps.is_available():
            parser.error("MPS is unavailable in this process")
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this process")
        build_cache(
            source,
            FrozenVAE.load(ROOT, VAE, device=args.device),
            FrozenCLIPText.from_lock(CLIP, root=ROOT, device=args.device),
            args.output,
            batch_size=args.batch_size,
            approval_path=args.approval,
            evidence_path=args.evidence,
        )
    cache = RepresentationCache(args.output, source, approval_sha256=receipt)
    blank, mask = cache.blank()
    print(
        json.dumps(
            {
                "mode": source.identity["kind"],
                "verified_only": args.verify_only,
                "manifest_sha256": source.identity["manifest_sha256"],
                "approval_sha256": receipt,
                "output": str(args.output),
                "counts": {split: len(cache.split(split)) for split in SPLITS},
                "blank_text_shape": list(blank.shape),
                "blank_valid_tokens": int(mask.sum()),
                "elapsed_seconds": round(perf_counter() - start, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
