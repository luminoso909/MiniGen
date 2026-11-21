# ruff: noqa: F722

"""从同一 B2 50M 20k EMA 比较原 Flow 和五官区域加权 Flow。"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from run_anime_b1 import open_anime_cache
from torch import Tensor, nn
from torch.utils.data import Subset
from train_anime_b2_repa import PARENT_CONFIG, ROOT, RUNS, SOURCE, SOURCE_SHA

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.config import build_model
from minigen.training.engine import train_flow
from minigen.training.runtime import resolve_device

RUN = RUNS / "b2_face_weights_v1"
WEIGHTS = Path("artifacts/cache/anime_faces_256/b2_face_weights_v1")
STEPS = 5000


class FaceWeightedFlow:
    """按样本 ID 对齐冻结权重图和均匀 Flow 的速度误差。"""

    def __init__(
        self,
        root: Path,
        sample_ids: list[str],
        manifest_sha256: str,
    ) -> None:
        """读取按原训练顺序冻结的 32×32 权重图。

        Args:
            root: 项目根目录。
            sample_ids: 有序训练表示 ID。
            manifest_sha256: 冻结划分清单摘要。

        Raises:
            OSError/ValueError: 缓存不可读或与训练输入身份不符。
        """
        cache = root / WEIGHTS
        array_path = cache / "train.npy"
        receipt = json.loads((cache / "index.json").read_text())
        weights = np.load(array_path, mmap_mode="r", allow_pickle=False)
        ids_sha = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
        shape = (len(sample_ids), 32, 32)
        if (
            receipt["train_ids_sha256"] != ids_sha
            or receipt["source_manifest_sha256"] != manifest_sha256
            or receipt["array_sha256"] != hash_file(array_path)
            or tuple(receipt["shape"]) != shape
            or weights.shape != shape
            or weights.dtype != np.float16
        ):
            raise ValueError("Face weights do not match the frozen training cache")
        self.weights = weights
        self.rows = {sample_id: row for row, sample_id in enumerate(sample_ids)}
        self.identity = receipt

    def __call__(
        self,
        model: nn.Module,
        noisy: Float[Tensor, "b c_z h_z w_z"],
        times: Float[Tensor, " b"],
        condition: Condition,
        target: Float[Tensor, "b c_z h_z w_z"],
        sample_ids: Sequence[str],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """保留未加权损失供比较，反向传播只用区域加权损失。

        Args:
            model: 两组共享的 B2 速度预测主干。
            noisy: 带噪潜变量。
            times: 均匀采样的 Flow 时刻。
            condition: 当前微批的文本特征和掩码。
            target: 原配方的目标速度。
            sample_ids: 决定空间权重图行的训练图像 ID。

        Raises:
            KeyError: 批内样本不在冻结权重缓存中。

        Results:
            tuple: 原 Flow MSE 的只读统计值，以及加权目标相对于该值的差。
        """
        error = (model(noisy, times, condition).float() - target.float()).square()
        rows = [self.rows[sample_id] for sample_id in sample_ids]
        weights = torch.from_numpy(self.weights[rows]).to(device=error.device, dtype=torch.float32)
        # 重新归一化消除 FP16 缓存的舍入误差，等量分配每张图的总权重。
        weights = weights / weights.mean(dim=(-2, -1), keepdim=True)
        uniform = error.detach().mean()
        weighted = (error * weights[:, None]).mean()
        return uniform, weighted - uniform


def main() -> None:
    """回读共同 EMA 或在独立运行目录内训练一个 5,000 步分支。

    Raises:
        OSError/ValueError/RuntimeError: 缓存、检查点身份或设备验证失败。
        SystemExit: 命令无效或训练未达到目标步数。

    副作用:
        设置 CPU 线程数，输出 JSON。train 写入分支检查点与训练日志。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "train"))
    parser.add_argument("--arm", choices=("control", "weighted"), default="control")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    parent, data = read_experiment(root, PARENT_CONFIG)
    config = replace(
        parent,
        output_dir=str(root / RUN / args.arm),
        learning_rate=1e-5,
        warmup_steps=0,
        min_lr_ratio=1.0,
        schedule_steps=STEPS,
        max_steps=STEPS,
        max_seconds=14400.0,
        checkpoint_steps=(1000, 2000, 3000, 4000, 5000),
    )
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, identity = open_anime_cache(root, data)
    checkpoint = root / SOURCE
    if hash_file(checkpoint) != SOURCE_SHA:
        raise ValueError("Starting B2 EMA checksum changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        state["step"] != 20000
        or state["objective"] != "flow_velocity"
        or state["data_identity"] != identity
        or state["config"]["model"] != config.model
    ):
        raise ValueError("Starting checkpoint does not match B2 50M 20k and its data")
    model = build_model(config.model)
    model.load_state_dict(state["ema"], strict=True)
    del state
    initialization: JsonDict = {
        "checkpoint_sha256": SOURCE_SHA,
        "checkpoint_step": 20000,
        "weights": "ema",
        "runner_sha256": hash_file(Path(__file__)),
    }
    train = Subset(cache.split("train"), selected_indices(source, data, "train"))
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    objective = None
    if args.arm == "weighted":
        objective = FaceWeightedFlow(
            root,
            [row["sample_id"] for row in source.records["train"]],
            source.identity["manifest_sha256"],
        )
        initialization["face_weights"] = objective.identity
    if args.operation == "train":
        resolve_device(config.device, config.precision)
        result = train_flow(
            config,
            train,
            blank,
            identity,
            model=model,
            validation=validation,
            initialization=initialization,
            resume=(root / args.resume) if args.resume else None,
            microbatch_loss=objective,
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "completed" or result["step"] != STEPS:
            raise SystemExit(1)
        return
    row = train[0]
    latent = row.latent[None].float()
    condition = Condition(row.text[None].float(), row.mask[None])
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(1234))
    times = torch.tensor([0.5])
    noisy, target = flow_interpolate(latent, noise, times)
    with torch.no_grad():
        if objective is None:
            uniform, delta = flow_loss(model(noisy, times, condition), target), torch.zeros(())
        else:
            uniform, delta = objective(model, noisy, times, condition, target, [row.sample_id])
    if not torch.isfinite(uniform + delta):
        raise ValueError("Starting EMA produced a nonfinite check loss")
    print(
        json.dumps(
            {
                "status": "passed",
                "optimizer_updates": 0,
                "sample_id": row.sample_id,
                "ordinary_flow_loss": float(uniform),
                "objective_loss": float(uniform + delta),
                "initialization": initialization,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
