"""从同一 Uniform 训练分支的固定快照生成 15k 开发图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from evaluate_anime_time_ablation import (
    ARMS,
    COUNT,
    INITIALIZATION,
    SEED,
    STAGE_STEPS,
    sample_variant,
    tensor_digest,
)
from run_anime_b1 import CONFIG, ROOT, open_anime_cache

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, FLOW_OBJECTIVE, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device

if TYPE_CHECKING:
    from torch import nn

UNIFORM_RUN_ID = "c1a26066163c9360d6c1ce36c1299ef3e342fcdde58135d36f9e84bfe1c22b43"
UNIFORM_13K_SHA256 = "84d4f05ac018f0f62d086cb05ae3db34be57ba584cda4fb23015dc2424ec2f03"
UNIFORM_13K_METRICS_SHA256 = "70ee6110e4a9c178f2c908d4707add6b4f2552a43719fa1330f21d914f969aa3"
DEV_STEP = 5000
FINAL_STEP = 10000


def read_uniform_milestone(
    path: Path,
    expected_step: int,
    model_config: JsonDict,
    data_identity: JsonDict,
) -> tuple[nn.Module, JsonDict]:
    """读取累计 15k 或 20k 的固定 EMA 快照并核对 Uniform 续训身份。

    Args:
        path: 续训编号快照；训练根目录下的 uniform 兄弟目录提供 13k 回执历史。
        expected_step: Uniform 分支步数，5,000 或 10,000。
        model_config: 同一动漫 B1 的 30M 结构配置。
        data_identity: 已回读的训练缓存和固定验证子集身份。

    Raises:
        OSError/RuntimeError: 训练记录或快照不可读，或 EMA 无法恢复。
        ValueError: 快照、原 Uniform 历史、训练合同或最终完成回执不符。

    Results:
        tuple: 仅用于推理的 EMA 模型和模型、数据、步数回执。
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract, config, history = state["contract"], state["config"], state["history"]
    baseline = path.parent.parent / "uniform"
    baseline_metrics = baseline / "metrics.jsonl"
    if (
        expected_step not in {DEV_STEP, FINAL_STEP}
        or state["schema"] != CHECKPOINT_SCHEMA
        or state["step"] != expected_step
        or state["scheduler"]["step"] != expected_step
        or state["objective"] != FLOW_OBJECTIVE
        or run_identity(contract) != UNIFORM_RUN_ID
        or config["model"] != model_config
        or config["time_mid_probability"] != ARMS["uniform"]
        or state["data_identity"] != data_identity
        or contract["data"] != data_identity
        or contract["initialization"] != INITIALIZATION
        or len(history) != expected_step
        or any(row["step"] != number for number, row in enumerate(history, start=1))
        or hash_file(baseline / "last.pt") != UNIFORM_13K_SHA256
        or hash_file(baseline_metrics) != UNIFORM_13K_METRICS_SHA256
        or [json.loads(row) for row in baseline_metrics.read_bytes().splitlines()]
        != history[:STAGE_STEPS]
    ):
        raise ValueError(f"Checkpoint does not match frozen Uniform lineage: {path}")

    samples_seen = sum(row["samples"] for row in history)
    middle_count = sum(row["mid_time_samples"] for row in history)
    middle_fraction = middle_count / samples_seen
    weights = state["ema"]
    if (
        abs(middle_fraction - ARMS["uniform"]) > 0.01
        or {key: [list(t.shape), str(t.dtype)] for key, t in weights.items()} != contract["tensors"]
    ):
        raise ValueError(f"Unexpected Uniform EMA layout or sampled times: {path}")
    checkpoint_sha256 = hash_file(path)
    receipt: JsonDict = {
        "arm": f"uniform{(INITIALIZATION['checkpoint_step'] + expected_step) // 1000}k",
        "objective": FLOW_OBJECTIVE,
        "objective_source": "explicit",
        "architecture": "b1",
        "run_id": UNIFORM_RUN_ID,
        "source_sha256": contract["source_sha256"],
        "model_config": model_config,
        "data_identity": data_identity,
        "checkpoint_step": expected_step,
        "cumulative_training_updates": INITIALIZATION["checkpoint_step"] + expected_step,
        "checkpoint_sha256": checkpoint_sha256,
        "baseline_checkpoint_sha256": UNIFORM_13K_SHA256,
        "initialization": INITIALIZATION,
        "time_mid_probability": ARMS["uniform"],
        "observed_time_mid_fraction": middle_fraction,
        "observed_mid_time_samples": middle_count,
        "observed_total_time_samples": samples_seen,
    }
    if expected_step == FINAL_STEP:
        result_path = path.parent / "result.json"
        metrics_path = path.parent / "metrics.jsonl"
        run = json.loads((path.parent / "run.json").read_text())
        result = json.loads(result_path.read_text())
        if (
            result["status"] != "completed"
            or result["step"] != expected_step
            or run["run_id"] != UNIFORM_RUN_ID
            or result["run_id"] != UNIFORM_RUN_ID
            or run["attempt_id"] != result["attempt_id"]
            or run["contract"] != contract
            or result["performance"]["optimizer_steps"] != expected_step
            or result["performance"]["samples_seen"] != samples_seen
            or hash_file(path.parent / "last.pt") != checkpoint_sha256
            or [json.loads(row) for row in metrics_path.read_bytes().splitlines()] != history
        ):
            raise ValueError("Completed Uniform 20k checkpoint and training receipts disagree")
        receipt.update(
            training_result_sha256=hash_file(result_path),
            training_metrics_sha256=hash_file(metrics_path),
        )

    with torch.device("meta"):
        model = build_model(config["model"])
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    receipt["trainable_parameters"] = sum(param.numel() for param in model.parameters())
    return model, receipt


@torch.inference_mode()
def evaluate_development(
    args: argparse.Namespace,
) -> None:
    """用固定 16 条开发条件和种子 1234 生成累计 15k 对照图。

    Args:
        args: 项目根路径、15k EMA 编号快照和输出根路径。

    Raises:
        OSError: 输出已存在或样本图、回执无法写入。
        ValueError/RuntimeError: 缓存、检查点或 CUDA 采样校验失败。

    副作用:
        设置推理精度、在线程数固定的条件下写入样本、四列网格与回执。
    """
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    device = resolve_device("cuda", "fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, data = read_experiment(root, CONFIG)
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, identity = open_anime_cache(root, data)
    indices = selected_indices(source, data, "validation")[:COUNT]
    if len(indices) != COUNT:
        raise ValueError("Expected sixteen locked development samples")
    dataset = cache.split("validation")
    entries = [dataset[i] for i in indices]
    latent_size = config.model["latent_size"]
    noise = torch.randn(
        (COUNT, config.model["in_channels"], latent_size, latent_size),
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        dtype=torch.float32,
    )
    records = []
    for position, index in enumerate(indices):
        row = source.records["validation"][index]
        records.append(
            {
                "index": position,
                "sample_id": row["sample_id"],
                "condition_sample_id": row["sample_id"],
                "source_image_path": row["image_path"],
                "condition_image_path": row["image_path"],
                "caption": row["caption"],
                "noise_sha256": tensor_digest(noise[position : position + 1]),
            }
        )
    model, receipt = read_uniform_milestone(
        root / args.checkpoint, DEV_STEP, config.model, identity
    )
    condition = Condition(
        torch.stack([entry.text for entry in entries]).float(),
        torch.stack([entry.mask for entry in entries]),
    ).to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    sample_variant(
        output / "correct",
        model.to(device),
        vae,
        noise.to(device),
        condition,
        blank.to(device),
        records,
        receipt
        | {
            "noise_sha256": tensor_digest(noise),
            "noise_shape": list(noise.shape),
            "noise_device": "cpu",
            "source_image_root": data["data_root"],
            "source": source.identity,
            "evaluation_torch": str(torch.__version__),
            "tf32": False,
        },
    )


def main() -> None:
    """解析命令参数并生成累计 15k 的固定开发样图。

    副作用:
        读取终端参数，输出开发图和采样回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate_development(parser.parse_args())


if __name__ == "__main__":
    main()
