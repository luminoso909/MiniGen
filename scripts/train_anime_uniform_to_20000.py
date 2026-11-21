"""恢复动漫头像 B1 Uniform 训练并保留累计 15k 和 20k 的检查点。"""

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from run_anime_b1 import open_anime_cache
from torch.utils.data import Subset
from train_anime_time_ablation import INITIALIZATION, ROOT, RUNS, paired_configs

from minigen.autoencoders.sdxl import hash_file
from minigen.commands import selected_indices
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, FLOW_OBJECTIVE, run_identity
from minigen.training.engine import train_flow
from minigen.training.runtime import resolve_device

SOURCE = RUNS / "b1_time_ablation_v1/uniform/last.pt"
SOURCE_SHA256 = "84d4f05ac018f0f62d086cb05ae3db34be57ba584cda4fb23015dc2424ec2f03"
UNIFORM_RUN_ID = "c1a26066163c9360d6c1ce36c1299ef3e342fcdde58135d36f9e84bfe1c22b43"
OUTPUT = RUNS / "b1_time_ablation_v1/uniform_to_20000"
MILESTONES = {5000: 3000, 10000: 5000}


def checkpoint_step(
    path: Path,
) -> int:
    """只读核对 Uniform 恢复身份和检查点更新编号。

    Args:
        path: 13k 已验收快照或续训目录内的检查点。

    Raises:
        OSError/RuntimeError: 检查点无法读取或反序列化。
        ValueError: Flow 检查点身份、历史长度或日程编号不一致。

    Results:
        int: 当前消融分支的更新次数，累计训练步数需要另加 10,000。
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = state["step"]
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["objective"] != FLOW_OBJECTIVE
        or run_identity(state["contract"]) != UNIFORM_RUN_ID
        or len(state["history"]) != step
        or state["scheduler"]["step"] != step
    ):
        raise ValueError(f"Unexpected Uniform checkpoint identity: {path}")
    return step


def main() -> None:
    """执行显式选中的 Uniform 续训阶段并固定该阶段产物。

    Raises:
        OSError/RuntimeError: 检查点、缓存、设备、训练或产物复制失败。
        ValueError: 恢复检查点哈希、更新编号或训练契约不一致。
        SystemExit: 本次训练未到达指定的分支更新编号。

    副作用:
        训练循环恢复在线模型、EMA、优化器与数据及 RNG 状态，写入本阶段产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("step", type=int, choices=MILESTONES)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / OUTPUT
    snapshot = directory / f"step_{args.step:06d}.pt"
    if snapshot.exists():
        if checkpoint_step(snapshot) != args.step:
            raise ValueError(f"Wrong milestone step: {snapshot}")
        print(f"Uniform lineage {10000 + args.step}: {snapshot}")
        return

    start = MILESTONES[args.step]
    source = root / SOURCE if start == 3000 else directory / f"step_{start:06d}.pt"
    if start == 3000 and hash_file(source) != SOURCE_SHA256:
        raise ValueError("13k Uniform checkpoint differs from the benchmark source")
    if checkpoint_step(source) != start:
        raise ValueError(f"Wrong starting update: {source}")
    last = directory / "last.pt"
    resume = last if last.exists() else source
    if not start <= checkpoint_step(resume) <= args.step:
        raise ValueError(f"Resume checkpoint falls outside this stage: {resume}")

    config, data = paired_configs(root)["uniform"]
    config = replace(
        config, output_dir=str(directory), max_steps=args.step, max_seconds=24 * 60 * 60
    )
    resolve_device(config.device, config.precision)
    source_data, cache, blank, identity = open_anime_cache(root, data)
    result = train_flow(
        config,
        Subset(cache.split("train"), selected_indices(source_data, data, "train")),
        blank,
        identity,
        validation=Subset(
            cache.split("validation"), selected_indices(source_data, data, "validation")
        ),
        initialization=INITIALIZATION,
        resume=resume,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "completed" or result["step"] != args.step:
        raise SystemExit(1)
    if checkpoint_step(last) != args.step:
        raise ValueError(f"Wrong completed update: {last}")

    for name in ("run", "result"):
        shutil.copyfile(directory / f"{name}.json", snapshot.with_suffix(f".{name}.json"))
    # 引擎原子替换 last.pt，保留当前 inode 可固定快照且无需再写权重。
    os.link(last, snapshot)
    print(f"Uniform lineage {10000 + args.step}: {snapshot}")


if __name__ == "__main__":
    main()
