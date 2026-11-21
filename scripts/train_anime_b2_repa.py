"""从 B2 50M 20k EMA 暖启动原目标与 REPA 两组间的等步数对照。"""

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
from torch.nn import functional as F
from torch.utils.data import Subset

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.models.hybrid_dit import HybridDiT
from minigen.objectives.flow import flow_loss
from minigen.training.config import build_model
from minigen.training.engine import train_flow
from minigen.training.runtime import autocast_context, resolve_device

ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG = Path("configs/training/anime_b2_50m_20000.json")
RUNS = Path("artifacts/runs/anime_faces_256")
SOURCE = RUNS / "b2_50m_20000/step_020000.pt"
SOURCE_SHA = "289db105636596e42731e8be7d013fdb384a66d0b572489e0151dd7e53baff12"
RUN = RUNS / "b2_repa_v1"
TEACHER = Path("artifacts/cache/anime_faces_256/b2_repa_clip_v1")
PROBE = Path("artifacts/exports/anime-faces-b1/repa_teacher_probe/receipt.json")
PROBE_CONFIG = Path("configs/evaluation/anime_repa_teacher_probe.json")
STEPS = 2000
ALIGNMENT_WEIGHT = 0.1


class PatchAlignment:
    """按训练样本 ID 读取冻结教师特征，计算唯一新增的余弦损失。"""

    def __init__(
        self,
        root: Path,
        sample_ids: list[str],
        manifest_sha256: str,
    ) -> None:
        """打开与训练潜变量逐样本一致的教师特征数组。

        Args:
            root: 项目根目录。
            sample_ids: 冻结训练缓存的有序 ID。
            manifest_sha256: 冻结划分清单的 SHA-256。

        Raises:
            OSError/ValueError: 教师索引、输入清单、权重或数组完整性不符。
        """
        cache = root / TEACHER
        receipt = json.loads((cache / "index.json").read_text())
        teacher = np.load(cache / "train.npy", mmap_mode="r", allow_pickle=False)
        weights = json.loads((root / "configs/models/evaluation.json").read_text())
        clip_sha = weights["models"]["clip"]["files"]["pytorch_model.bin"]["sha256"]
        expected_ids = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
        expected_shape = (len(sample_ids), 256, 1024)
        if (
            receipt["train_ids_sha256"] != expected_ids
            or receipt["source_manifest_sha256"] != manifest_sha256
            or receipt["weights_sha256"] != clip_sha
            or receipt["array_sha256"] != hash_file(cache / "train.npy")
            or teacher.shape != expected_shape
            or tuple(receipt["shape"]) != expected_shape
            or teacher.dtype != np.float16
        ):
            raise ValueError("Teacher cache does not match the frozen training images")
        self.teacher = teacher
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
        """对齐同一输入图像的干净 patch 特征与带噪中间图像 token。

        Args:
            model: 带训练专用投影头的 B2。
            noisy: Flow 插值得到的带噪潜变量。
            times: 当前微批的均匀采样时间。
            condition: 已应用条件丢弃的文本及掩码。
            target: 原配方的目标速度。
            sample_ids: 对应潜变量的冻结图像 ID，决定教师行。

        Raises:
            KeyError: 没有对应的教师行。
            TypeError: 主干缺少 B2 图像 token 接口。

        Results:
            tuple: 原始 Flow MSE 和乘过系数的 REPA 余弦距离。
        """
        if not isinstance(model, HybridDiT):
            raise TypeError("REPA image-token alignment requires HybridDiT")
        prediction, tokens = model.forward_with_image_tokens(noisy, times, condition)
        projected = model.get_submodule("repa_projector")(tokens).float()
        rows = [self.rows[sample_id] for sample_id in sample_ids]
        features = torch.from_numpy(self.teacher[rows]).to(
            device=projected.device, dtype=torch.float32
        )
        distance = 1 - F.cosine_similarity(projected, features, dim=-1).mean()
        return flow_loss(prediction, target), ALIGNMENT_WEIGHT * distance


def main() -> None:
    """核查 EMA 起点和实际运行条件，或执行一组 2,000 步实验。

    Raises:
        OSError/ValueError/RuntimeError: 输入不一致、设备不可用或训练失败。
        SystemExit: 命令行错误、检查点状态与完成条件不一致。

    副作用:
        设置当前进程的执行线程及随机状态，输出检查回执。train 写入独立运行目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check-parent", "check-repa", "train"))
    parser.add_argument("--arm", choices=("control", "repa"), default="control")
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
        max_seconds=10800.0,
        checkpoint_steps=(1000, 2000),
    )
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, identity = open_anime_cache(root, data)
    checkpoint = root / SOURCE
    if hash_file(checkpoint) != SOURCE_SHA:
        raise ValueError("Starting B2 EMA checkpoint SHA-256 changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        state["step"] != 20000
        or state["objective"] != "flow_velocity"
        or state["data_identity"] != identity
        or state["config"]["model"] != config.model
    ):
        raise ValueError("Starting checkpoint does not match the accepted B2 50M run")
    model = build_model(config.model)
    model.load_state_dict(state["ema"], strict=True)
    del state
    initialization: JsonDict = {
        "checkpoint_sha256": SOURCE_SHA,
        "checkpoint_step": 20000,
        "weights": "ema",
        "runner_sha256": hash_file(Path(__file__)),
    }
    train_indices = selected_indices(source, data, "train")
    train = Subset(cache.split("train"), train_indices)
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    alignment = None
    if args.operation != "check-parent":
        probe = json.loads((root / PROBE).read_text())
        if probe["structure_passed"] is not True or probe["probe_config_sha256"] != hash_file(
            root / PROBE_CONFIG
        ):
            raise ValueError("The frozen CLIP teacher probe has not passed")
        initialization["probe_sha256"] = hash_file(root / PROBE)
        if args.arm == "repa" or args.operation == "check-repa":
            alignment = PatchAlignment(
                root,
                [row["sample_id"] for row in source.records["train"]],
                source.identity["manifest_sha256"],
            )
            hidden = config.model["hidden_size"]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                model.add_module(
                    "repa_projector",
                    nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1024)),
                )
            initialization["repa"] = {
                "weight": ALIGNMENT_WEIGHT,
                "teacher": alignment.identity,
                "layer": config.model["double_depth"],
            }
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
            microbatch_loss=alignment,
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "completed" or result["step"] != STEPS:
            raise SystemExit(1)
        return
    device = torch.device("cpu")
    row = train[0]
    latent = row.latent[None].float()
    condition = Condition(row.text[None].float(), row.mask[None])
    times = torch.tensor([0.5])
    target = torch.zeros_like(latent)
    with torch.no_grad(), autocast_context(device, "fp32"):
        if alignment is None:
            flow, auxiliary = flow_loss(model(latent, times, condition), target), None
        else:
            flow, auxiliary = alignment(model, latent, times, condition, target, [row.sample_id])
    if not torch.isfinite(flow) or (auxiliary is not None and not torch.isfinite(auxiliary)):
        raise ValueError("The starting EMA returned a nonfinite preflight loss")
    print(
        json.dumps(
            {
                "status": "passed",
                "sample_id": row.sample_id,
                "optimizer_updates": 0,
                "flow_loss": float(flow),
                "aux_loss": None if auxiliary is None else float(auxiliary),
                "initialization": initialization,
                "train_rows": len(train),
                "validation_rows": len(validation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
