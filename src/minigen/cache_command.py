"""构建或验证 P2 的 ArtBench 表示缓存，全程使用固定本地资产。"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.conditioning.clip import FrozenCLIPText
from minigen.data.representation_cache import (
    SPLITS,
    RepresentationCache,
    build_cache,
    cache_source,
    verify_cache_acceptance,
)


def main(
    argv: list[str] | None = None,
    *,
    root: Path | None = None,
) -> None:
    """用正式或试点模式打开固定 256px 数据，输出可核验的本地摘要。

    Args:
        argv: 命令行参数，None 使用当前进程参数。
        root: 默认项目根，可由 --root 覆盖；未指定则使用工作目录。

    Raises:
        SystemExit: 参数错误，或请求的设备不可用。
        ValueError: 身份、审核或缓存校验失败。
        OSError: 本地数据、权重或目标目录不可读写。

    副作用:
        设置本进程 CPU 线程数；写新缓存时创建目录并输出 JSON 摘要。
    """
    root = (root or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/artbench_pd_256/v1"))
    parser.add_argument("--freeze", type=Path, default=Path("reports/v2/data/artbench_freeze.json"))
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot-rows", type=int, help="每个划分最多缓存的样本数")
    mode.add_argument("--approval", type=Path, help="正式缓存要求用户审核 JSON")
    parser.add_argument("--evidence", type=Path, help="与审核回执匹配的 200 张重建 JSON")
    parser.add_argument("--local-acceptance", type=Path, help="固定 20 张的本机 FP32 回执")
    parser.add_argument("--verify-only", action="store_true", help="只检查已有缓存，不加载编码器")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "mps"), help="仅供试点指定编码设备")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    for name in ("data_root", "freeze", "output", "approval", "evidence", "local_acceptance"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, root / value)
    vae_lock = root / "configs/models/sdxl_vae.json"
    clip_lock = root / "configs/models/clip_text.json"
    if args.cpu_threads < 1 or args.batch_size < 1:
        parser.error("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    start = perf_counter()
    source = cache_source(args.data_root, args.freeze, vae_lock, clip_lock, args.pilot_rows)
    try:
        acceptance = verify_cache_acceptance(
            source, args.approval, args.evidence, args.local_acceptance
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.approval is not None and args.device is not None:
        parser.error("Formal cache uses the per-component devices from --local-acceptance")
    pilot_device = args.device or "cpu"
    vae_device = acceptance.vae_device or pilot_device
    clip_device = acceptance.clip_device or pilot_device
    if not args.verify_only:
        if "mps" in (vae_device, clip_device) and not torch.backends.mps.is_available():
            parser.error("MPS is unavailable in this process")
        if "mps" in (vae_device, clip_device) and any(
            os.environ.get(name, "0") != "0"
            for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")
        ):
            parser.error("MPS cache requires fallback and fast math disabled")
        build_cache(
            source,
            FrozenVAE.load(root, vae_lock, device=vae_device),
            FrozenCLIPText.from_lock(clip_lock, root=root, device=clip_device),
            args.output,
            batch_size=args.batch_size,
            approval_path=args.approval,
            evidence_path=args.evidence,
            local_acceptance_path=args.local_acceptance,
            report_progress=True,
        )
    cache = RepresentationCache(
        args.output,
        source,
        approval_sha256=acceptance.approval_sha256,
        local_acceptance_sha256=acceptance.local_acceptance_sha256,
    )
    blank, mask = cache.blank()
    index = args.output / "index.json"
    metadata = json.loads(index.read_text())
    payload = metadata["payload_sha256"]
    print(
        json.dumps(
            {
                "mode": source.identity["kind"],
                "verified_only": args.verify_only,
                "manifest_sha256": source.identity["manifest_sha256"],
                **acceptance._asdict(),
                "cache_index_sha256": hash_file(index),
                "output": str(args.output),
                "execution": metadata["execution"],
                "counts": {split: len(cache.split(split)) for split in SPLITS},
                "verified_rows": cache.verify_rows(source),
                "storage": source.identity["storage"],
                "payload_sha256": payload,
                "payload_bytes": sum((args.output / name).stat().st_size for name in payload),
                "blank_text_shape": list(blank.shape),
                "blank_valid_tokens": int(mask.sum()),
                "elapsed_seconds": round(perf_counter() - start, 3),
            },
            indent=2,
        )
    )
