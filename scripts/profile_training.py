"""P3-06 单卡 CUDA Pilot；显式运行后才产生资源与成本测量。"""

import argparse
import json
import math
import resource
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from torch.utils.data import Subset

from minigen.commands import open_cache, read_experiment, selected_indices
from minigen.training.checkpoint import load_checkpoint
from minigen.training.engine import train_flow
from minigen.training.runtime import resolve_device


def tensor_bytes(
    value: Any,
) -> int:
    """统计异构检查点里张量的逻辑字节数，不当作分配器峰值。

    Args:
        value: 优化器、EMA 或其他嵌套检查点对象。

    Results:
        int: 嵌套张量的元素数量乘元素字节数之和。
    """
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def main() -> int:
    """执行有步数上限的 CUDA 前向反向并分离预热与测量区间。

    Raises:
        SystemExit: 参数无效或 CUDA 不可用。
        OSError/ValueError/RuntimeError: 配置、数据、训练或结果写入失败。

    Results:
        int: 完成实测为 0；CUDA 不可用由 argparse 返回 2，不生成通过回执。

    副作用:
        只在 CUDA 可用时执行有限训练，保存检查点、计时与显存结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--hourly-cny", type=float)
    parser.add_argument("--planned-steps", type=int)
    args = parser.parse_args()
    if args.warmup < 1 or args.measure < 1:
        parser.error("warmup and measure must be positive")
    if args.hourly_cny is not None and (not math.isfinite(args.hourly_cny) or args.hourly_cny < 0):
        parser.error("hourly-cny must be finite and nonnegative")
    if args.planned_steps is not None and args.planned_steps < 1:
        parser.error("planned-steps must be positive")
    try:
        device = resolve_device("cuda", args.precision)
    except ValueError as error:
        parser.error(str(error))
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        parser.error("output must be a new directory")
    config, data = read_experiment(root, args.config)
    source, cache, blank, identity = open_cache(root, data)
    steps = args.warmup + args.measure
    config = replace(
        config,
        device="cuda",
        precision=args.precision,
        output_dir=str(output),
        max_steps=steps,
        max_seconds=3600,
        checkpoint_every=steps,
        checkpoint_steps=(),
        validation_every=steps + 1,
    )
    result = train_flow(
        config,
        Subset(cache.split("train"), selected_indices(source, data, "train")),
        blank,
        identity,
    )
    state = load_checkpoint(Path(result["checkpoint"]))
    if state["step"] != steps:
        raise RuntimeError("Pilot did not complete every requested warmup and measured step")
    measured = state["history"][args.warmup :]
    seconds = [row["step_seconds"] for row in measured]
    total = sum(seconds)
    planned_seconds = mean(seconds) * args.planned_steps if args.planned_steps else None
    # ru_maxrss 在 Linux 为 KiB，在 macOS 为字节；CUDA 主机通常是 Linux。
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    report = {
        "status": "measured",
        "cuda_device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "precision": args.precision,
        "warmup_steps": args.warmup,
        "measured_steps": len(measured),
        "step_seconds_mean": mean(seconds),
        "step_seconds_median": median(seconds),
        "samples_per_second": sum(row["samples"] for row in measured) / total,
        "data_wait_fraction": sum(row["data_seconds"] for row in measured) / total,
        "cuda_peak_allocated_bytes": result["memory"]["cuda_peak_allocated"],
        "cuda_peak_reserved_bytes": result["memory"]["cuda_peak_reserved"],
        "cpu_peak_rss_bytes": ram,
        "optimizer_tensor_bytes": tensor_bytes(state["optimizer"]),
        "ema_tensor_bytes": tensor_bytes(state["ema"]),
        "memory_scope": "CUDA 峰值覆盖预热和测量；optimizer/EMA 为逻辑张量字节数",
        "training_result": result,
        "planned_steps": args.planned_steps,
        "estimated_training_seconds": planned_seconds,
        "hourly_cny": args.hourly_cny,
        "estimated_training_cny": (planned_seconds * args.hourly_cny / 3600)
        if planned_seconds is not None and args.hourly_cny is not None
        else None,
        "estimate_scope": "只含优化步；Checkpoint、采样、验证、存储和传输费用另计，不是实际消费",
        "distributed": "single-process single-device; DDP not implemented",
    }
    (output / "pilot.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
