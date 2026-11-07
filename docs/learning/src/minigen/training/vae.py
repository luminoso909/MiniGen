"""显式注入分布函数的 VAE 短训、验证与恢复工程。"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from minigen._typing import JsonDict, KLFn, ReparameterizeFn
from minigen.data.stateful import StatefulBatchStream
from minigen.models.tiny_vae import TinyVAE
from minigen.training.denoising import _restore_rng, _rng_state, save_checkpoint, seed_all


@dataclass(frozen=True)
class VAEConfig:
    """保存简化 VAE 短实验和恢复条件的不可变配置。

    Args:
        run_dir: 本地日志和 Checkpoint 目录。
        beta: KL 权重，训练要求非负。
        max_steps/max_seconds: 总更新步数与本次耗时上限。
        learning_rate/seed/device: 优化器学习率、种子和设备。
        checkpoint_every: 保存与验证间隔。
        core_source: 显式分布核心来源。

    Results:
        VAEConfig: VAEConfig 实例，训练入口负责检查范围。
    """

    run_dir: str
    beta: float = 1.0
    max_steps: int = 1000
    max_seconds: float = 300
    learning_rate: float = 0.001
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 100
    core_source: str = "caller_supplied"


def reconstruction_loss(
    logits: Float[Tensor, " b *p"],
    images: Float[Tensor, " b *p"],
) -> Float[Tensor, ""]:
    """按单图像素求和、batch 求均值计算 soft-target BCE。

    Args:
        logits: [b,*p] 解码器原始输出，无需预先 sigmoid；MNIST 为 [b,1,28,28]。
        images: 同形 [0,1] 目标图像；每张图的非 batch 维会展平求和。

    Raises:
        ValueError: 形状不一致或目标范围超出 [0,1]；底层 BCE 异常透传。

    Results:
        Float[Tensor, ""]: 零维重建损失；目标像素可为连续值，这不是精确连续似然。
    """
    if logits.shape != images.shape or images.min() < 0 or images.max() > 1:
        raise ValueError("Equal image shapes and targets in [0,1] required")
    return (
        F.binary_cross_entropy_with_logits(logits, images, reduction="none")
        .flatten(1)
        .sum(1)
        .mean()
    )


@torch.no_grad()
def evaluate_vae(
    model: TinyVAE,
    images: Float[Tensor, "b 1 28 28"],
    *,
    reparameterize_fn: ReparameterizeFn,
    kl_fn: KLFn,
    seed: int = 42,
    batch_size: int = 128,
) -> JsonDict:
    """使用独立固定噪声测量重建、KL 与潜空间统计。

    Args:
        model: 需具备 latent_dim、encode/decode 与显式分布函数前向的 TinyVAE。
        images: [b,1,28,28] 验证图像，像素 [0,1]。
        reparameterize_fn: 显式潜变量采样函数，不自动选择参考。
        kl_fn: 显式高斯 KL 函数，支持按坐标切片计算。
        seed: 独立 CPU 噪声种子，不消耗训练 RNG。
        batch_size: 验证分块大小，调用者应提供正值。

    Raises:
        ValueError: 底层重建范围/形状错误；RuntimeError/ZeroDivisionError: 空验证集等无效输入导致
            下层运算失败。
        分布核心或设备异常原样透传，不新增输入检查。

    Results:
        JsonDict: JSON 统计: 重建、KL、均值解码误差、逐维 KL、后验及聚合方差；finally
            恢复模型模式。

    副作用:
        临时设为 eval，并在 finally 中恢复原模式。
        仅使用独立噪声发生器，不推进训练的全局 RNG。
    """
    previous = model.training
    model.eval()
    device = next(model.parameters()).device
    noise = torch.randn(
        len(images), model.latent_dim, generator=torch.Generator().manual_seed(seed)
    )
    totals = {"reconstruction": 0.0, "kl": 0.0, "mean_reconstruction": 0.0}
    means, variances, per_dimension = [], [], []
    try:
        for start in range(0, len(images), batch_size):
            x = images[start : start + batch_size].to(device)
            eps = noise[start : start + len(x)].to(device)
            logits, mean, logvar = model(x, eps, reparameterize_fn=reparameterize_fn)
            values = {
                "reconstruction": reconstruction_loss(logits, x),
                "kl": kl_fn(mean, logvar),
                "mean_reconstruction": reconstruction_loss(model.decode(mean), x),
            }
            for key, value in values.items():
                totals[key] += value.item() * len(x)
            means.append(mean.cpu())
            variances.append(logvar.exp().cpu())
            # 按坐标调用显式传入的 KL，不在训练工程中维护另一份核心公式。
            per_dimension.append(
                torch.stack(
                    [
                        kl_fn(mean[:, j : j + 1], logvar[:, j : j + 1]).cpu()
                        for j in range(model.latent_dim)
                    ]
                )
                * len(x)
            )
    finally:
        model.train(previous)
    mean = torch.cat(means)
    variance = torch.cat(variances)
    return {
        **{k: v / len(images) for k, v in totals.items()},
        "per_dimension_kl": (torch.stack(per_dimension).sum(0) / len(images)).tolist(),
        "posterior_mean_average": mean.mean(0).tolist(),
        "posterior_mean_variance": mean.var(0, unbiased=False).tolist(),
        "posterior_variance_average": variance.mean(0).tolist(),
        "aggregate_variance": (mean.var(0, unbiased=False) + variance.mean(0)).tolist(),
        "count": len(images),
    }


def train_vae(
    model: TinyVAE,
    stream: StatefulBatchStream,
    config: VAEConfig,
    *,
    reparameterize_fn: ReparameterizeFn,
    kl_fn: KLFn,
    validation: Float[Tensor, "b_v 1 28 28"],
    fixed_noise: Float[Tensor, "b z"] | None = None,
    resume_path: str | None = None,
) -> JsonDict:
    """执行有界 VAE 训练，显式注入分布核心并保存可恢复的数据顺序。

    Args:
        model: 待训练的 TinyVAE 编解码网络。
        stream: 单进程 StatefulBatchStream，状态用于精确顺序恢复。
        config: 损失权重、运行上限、学习率与核心来源。
        reparameterize_fn: 采样核心，接收 mean、log_variance、noise。
        kl_fn: KL 核心，返回零维损失。
        validation: 固定 [b_v,1,28,28] 验证图像，像素 [0,1]。
        fixed_noise: 可选 [b,z] 固定噪声；调用者保证与每批图像对齐。
        resume_path: 可选 Checkpoint 路径，恢复配置和潜变量维度必须一致。

    Raises:
        ValueError: 训练限制、beta、学习率无效或恢复配置/维度不符；
            FloatingPointError: 非有限损失。
        数据流、重建、梯度检查与文件读写的异常原样透传。

    Results:
        JsonDict: VAE 训练、验证和恢复记录。
            config/latent_dim: 配置与潜变量宽度。
            start_step/final_step/stop_reason: 更新范围与停止原因。
            initial/final: 固定验证指标。
            metrics/probes: 训练记录和各保存时刻的验证结果。
            checkpoint/elapsed_seconds: 最终文件位置与耗时。

    副作用:
        更新传入模型、优化器和数据流，消耗/恢复随机状态。
        写日志、摘要、Checkpoint，并打印验证进度。
    """
    if config.max_steps <= 0 or config.max_seconds <= 0 or config.checkpoint_every <= 0:
        raise ValueError("Positive training limits required")
    if config.beta < 0 or config.learning_rate <= 0:
        raise ValueError("Nonnegative beta and positive learning rate required")
    directory = Path(config.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    model.to(config.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    step = 0
    if resume_path:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("beta", "learning_rate", "seed", "device", "core_source"):
            if saved["config"][key] != getattr(config, key):
                raise ValueError(f"Resume configuration changed: {key}")
        if saved["latent_dim"] != model.latent_dim:
            raise ValueError("Latent dimension changed")
        stream.load_state_dict(saved["data_state"])
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step = saved["step"]
        _restore_rng(saved["rng"])
    start_step = step
    initial = evaluate_vae(model, validation, reparameterize_fn=reparameterize_fn, kl_fn=kl_fn)
    start = time.monotonic()
    rows, probes = [], []
    stop = "max_steps"

    def checkpoint(
        path: Path,
    ) -> None:
        """保存当前闭包中的模型、优化器、数据流与随机状态。

        Args:
            path: 目标 Checkpoint，状态取自本轮 train_vae 的当前 step 等变量。

        Raises:
            OSError/RuntimeError: 文件写入或序列化失败时透传。

        副作用:
            将闭包中当前模型、步数、数据流和随机状态写入目标文件。
        """
        save_checkpoint(
            path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": asdict(config),
                "latent_dim": model.latent_dim,
                "data_state": stream.state_dict(),
                "rng": _rng_state(config.device),
                "input_range": "[0,1]",
                "reduction": "sum dimensions, mean batch",
            },
        )

    with (directory / "metrics.jsonl").open("a" if resume_path else "w") as log:
        while step < config.max_steps:
            if time.monotonic() - start >= config.max_seconds:
                stop = "time_limit"
                break
            batch = next(stream)
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(config.device)
            noise = (
                fixed_noise.to(config.device)
                if fixed_noise is not None
                else torch.randn(len(x), model.latent_dim, device=config.device)
            )
            optimizer.zero_grad(set_to_none=True)
            logits, mean, logvar = model(x, noise, reparameterize_fn=reparameterize_fn)
            reconstruction = reconstruction_loss(logits, x)
            kl = kl_fn(mean, logvar)
            loss = reconstruction + config.beta * kl
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite VAE objective")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {
                "step": step,
                "loss": loss.item(),
                "reconstruction": reconstruction.item(),
                "kl": kl.item(),
                "gradient_norm": float(norm),
                "elapsed_seconds": time.monotonic() - start,
            }
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % config.checkpoint_every == 0:
                checkpoint(directory / f"step_{step:04d}.pt")
                probe = evaluate_vae(
                    model, validation, reparameterize_fn=reparameterize_fn, kl_fn=kl_fn
                )
                probes.append({"step": step, **probe})
                print(
                    f"{directory.name}: step={step} recon={probe['reconstruction']:.3f} "
                    f"kl={probe['kl']:.3f}",
                    flush=True,
                )
    final = evaluate_vae(model, validation, reparameterize_fn=reparameterize_fn, kl_fn=kl_fn)
    checkpoint(directory / "latest.pt")
    result = {
        "config": asdict(config),
        "latent_dim": model.latent_dim,
        "start_step": start_step,
        "final_step": step,
        "stop_reason": stop,
        "elapsed_seconds": time.monotonic() - start,
        "initial": initial,
        "final": final,
        "metrics": rows,
        "probes": probes,
        "checkpoint": str(directory / "latest.pt"),
    }
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
