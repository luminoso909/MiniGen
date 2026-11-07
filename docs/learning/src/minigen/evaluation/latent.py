"""有时间上限的冻结 VAE 表示诊断和显式教学模型优化检查。"""

import time
from pathlib import Path

import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor, nn

from minigen._typing import JsonDict
from minigen.autoencoders.frozen import FrozenVAE, image_batch, to_image
from minigen.evaluation.icons import aggregate_scores, score_image


def reconstruct(
    adapter: FrozenVAE,
    records: list[JsonDict],
    data_root: Path,
    deadline: float,
    batch_size: int = 4,
) -> tuple[Float[Tensor, "b c h w"], list[Image.Image], JsonDict]:
    """在绝对时间上限内测量真实冻结 VAE 的重建与属性。

    Args:
        adapter: 显式冻结组件，负责像素/潜变量缩放。
        records: 待诊断的有序清单，至少应能完成一个 batch。
        data_root: 图片根目录 Path。
        deadline: time.monotonic 的绝对截止时间，不是时长。
        batch_size: 编码与解码分块大小，调用者提供正值。

    Raises:
        TimeoutError: 截止前连一个 batch 都未完成；数据、编码、解码与评分异常原样透传。

    Results:
        tuple[Float[Tensor, "b c h w"], list[Image.Image], JsonDict]: CPU 潜变量、重建 PIL
            图像列表、真实完成数量/属性/MSE/通道统计；超时保留已完成部分。

    副作用:
        打印已完成批次的进度。
    """
    latents, decoded, original_scores, reconstructed_scores, pixel_errors = [], [], [], [], []
    begin = time.monotonic()
    for start in range(0, len(records), batch_size):
        if time.monotonic() >= deadline:
            break
        rows = records[start : start + batch_size]
        images = image_batch(rows, data_root)
        encoded = adapter.encode(images)
        reconstruction = adapter.decode(encoded).cpu()
        latents.append(encoded.cpu())
        for row, output, source in zip(rows, reconstruction, images, strict=True):
            with Image.open(data_root / row["image_path"]) as image:
                original_scores.append(score_image(image, row))
            reconstructed_scores.append(score_image(to_image(output), row))
            pixel_errors.append(float((output - source).square().mean()))
            decoded.append(to_image(output))
        if (start + len(rows)) % 16 == 0:
            print(f"VAE reconstruction {start + len(rows)}/{len(records)}", flush=True)
    if not latents:
        raise TimeoutError("Budget exhausted before VAE evaluation")
    values = torch.cat(latents)
    metrics = {
        "requested": len(records),
        "completed": len(decoded),
        "stop_reason": "complete" if len(decoded) == len(records) else "time_limit",
        "seconds": time.monotonic() - begin,
        "original": aggregate_scores(original_scores),
        "reconstruction": aggregate_scores(reconstructed_scores),
        "pixel_mse_raw_minus1_to1": sum(pixel_errors) / len(pixel_errors),
        "pixel_mse_clipping": False,
        "latent_shape": list(values.shape),
        "latent_channel_mean": values.mean((0, 2, 3)).tolist(),
        "latent_channel_std": values.std((0, 2, 3)).tolist(),
        "latent_abs_max": values.abs().max().item(),
        "per_image": reconstructed_scores,
        "sample_ids": [r["sample_id"] for r in records[: len(decoded)]],
    }
    return values, decoded, metrics


def encode_pilot(
    adapter: FrozenVAE,
    records: list[JsonDict],
    data_root: Path,
    deadline: float,
    batch_size: int = 4,
) -> tuple[Float[Tensor, "b c h w"], JsonDict]:
    """在预算内编码训练子集，不把部分结果记为完整试点。

    Args:
        adapter: 冻结 VAE 组件。
        records: 计划缓存的有序训练样本。
        data_root: 图像根目录。
        deadline: monotonic 绝对截止时间。
        batch_size: 每次编码张数，调用者保证为正。

    Raises:
        TimeoutError: 未完成任何一批；数据读取和模型异常透传。

    Results:
        tuple[Float[Tensor, "b c h w"], JsonDict]: CPU FP32 潜变量与请求数、完成数、
            停止原因和耗时。

    副作用:
        打印试点编码进度。
    """
    chunks = []
    begin = time.monotonic()
    for start in range(0, len(records), batch_size):
        if time.monotonic() >= deadline:
            break
        chunks.append(
            adapter.encode(image_batch(records[start : start + batch_size], data_root)).cpu()
        )
        if (start + len(chunks[-1])) % 64 == 0:
            print(f"Pilot encoding {start + len(chunks[-1])}/{len(records)}", flush=True)
    if not chunks:
        raise TimeoutError("Budget exhausted before pilot encoding")
    values = torch.cat(chunks)
    return values, {
        "requested": len(records),
        "completed": len(values),
        "stop_reason": "complete" if len(values) == len(records) else "time_limit",
        "seconds": time.monotonic() - begin,
    }


def fit_fixed_target(
    model: nn.Module,
    inputs: Float[Tensor, "b c h w"],
    times: Int[Tensor, " b"],
    target: Float[Tensor, "b c h w"],
    deadline: float,
    max_steps: int = 100,
) -> JsonDict:
    """对调用方显式传入的教学模型做固定监督优化检查。

    Args:
        model: forward(inputs,times) 返回同形噪声预测的模型，不在此创建核心。
        inputs: [b,c,h,w] 固定带噪输入。
        times: [b] 固定整数时间。
        target: 与 inputs 同形的固定监督噪声。
        deadline: monotonic 绝对截止时间。
        max_steps: 本次最多更新次数，默认 100。

    Raises:
        FloatingPointError: 损失出现非有限值；模型、优化器与设备异常原样透传。

    Results:
        JsonDict: JSON 记录: 初末损失、实际步数、梯度范数、耗时及停止原因；只证明固定目标优化。

    副作用:
        更新传入模型参数并将其保留在 eval 模式；不自动保存权重。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    model.train()
    with torch.no_grad():
        initial = (model(inputs, times) - target).square().mean().item()
    history = []
    start = time.monotonic()
    for step in range(max_steps):
        if time.monotonic() >= deadline:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs, times) - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DiT diagnostic loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step + 1, "loss": loss.item(), "gradient_norm": norm.item()})
    model.eval()
    with torch.no_grad():
        final = (model(inputs, times) - target).square().mean().item()
    return {
        "core_source": "notebook_reference",
        "initial_loss": initial,
        "final_loss": final,
        "steps": len(history),
        "requested_steps": max_steps,
        "stop_reason": "max_steps" if len(history) == max_steps else "time_limit",
        "seconds": time.monotonic() - start,
        "history": history,
        "interpretation": "Fixed targets only; no random-time training or generation",
    }
