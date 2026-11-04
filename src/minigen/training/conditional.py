"""显式条件网络的有界训练；保存文本、丢弃、数据顺序及日程的恢复契约。"""

import copy
import hashlib
import json
import marshal
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from minigen._typing import JsonDict
from minigen.data.latent import identity_digest
from minigen.data.stateful import StatefulBatchStream
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.training.denoising import (
    TrainConfig,
    _restore_rng,
    _rng_state,
    save_checkpoint,
    seed_all,
)
from minigen.training.latent import _model_identity, tensor_digest

# 回调显式接收真值 token/mask、空条件 token/mask 与 p，返回逐样本替换后的条件和 mask。
DropCondition = Callable[
    [
        Float[Tensor, "b n d"],
        Bool[Tensor, "b n"],
        Float[Tensor, "1 n d"],
        Bool[Tensor, "1 n"],
        float,
    ],
    tuple[Float[Tensor, "b n d"], Bool[Tensor, "b n"]],
]


def train_conditional(
    model: nn.Module,
    scheduler: DDPMScheduler,
    latents: Float[Tensor, "n_s c h w"],
    text_ids: Int[Tensor, " n_s"],
    features: Float[Tensor, "n_c n d"],
    masks: Bool[Tensor, "n_c n"],
    empty_id: int,
    drop_condition: DropCondition,
    config: TrainConfig,
    *,
    identity: JsonDict,
    model_identity: JsonDict,
    batch_size: int = 16,
    drop_probability: float = 0.1,
    resume_path: str | None = None,
) -> JsonDict:
    """联合训练条件与空条件预测器，学习核心由调用者显式传入。

    Args:
        model: forward(z,t,text,mask) 噪声预测网络。
        scheduler: 已验收的个人 DDPM。
        latents: CPU FP32 [n_s,c,h,w] 训练缓存。
        text_ids: [n_s] 每张图对应的去重描述行号。
        features: CPU FP16/FP32 [n_c,n,d] 文本字典；n_c 为描述条数，n 为 token 槽数。
        masks: [n_c,n] 有效 token 位。
        empty_id: 字典中空文本行号。
        drop_condition: 显式教学或个人条件丢弃回调，不自动提供算法实现。
        config: 步数、时间、设备、优化器与 EMA 配置。
        identity: 数据、文本、分段与回调源码身份，必须包含 drop_source_sha256。
        model_identity: 显式网络来源、源码摘要与架构。
        batch_size: 正批量大小。
        drop_probability: [0,1] 的逐样本丢弃概率。
        resume_path: 可选 Checkpoint，契约不兼容即拒绝。

    Raises:
        ValueError: 输入形状、索引、有限值、资源上限、来源或恢复契约不符。
        FloatingPointError: 非有限训练损失。
        OSError/RuntimeError: 文件、模型或梯度异常透传。

    Results:
        JsonDict: 实际步数、耗时、停止原因、损失轨迹、Checkpoint 与契约摘要。

    副作用:
        更新模型、EMA 和优化器，消耗随机状态，推进确定性数据流并写日志/权重。
    """
    if (
        latents.ndim != 4
        or len(latents) == 0
        or latents.device.type != "cpu"
        or latents.dtype != torch.float32
        or not torch.isfinite(latents).all()
    ):
        raise ValueError("Finite CPU FP32 latent data required")
    if (
        features.ndim != 3
        or masks.shape != features.shape[:2]
        or masks.dtype != torch.bool
        or not masks.any(-1).all()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Finite text features with nonempty Bool masks required")
    if (
        text_ids.shape != (len(latents),)
        or text_ids.dtype != torch.long
        or text_ids.min() < 0
        or text_ids.max() >= len(features)
        or not 0 <= empty_id < len(features)
    ):
        raise ValueError("Aligned valid text indices required")
    if (
        batch_size < 1
        or not 0 <= drop_probability <= 1
        or config.max_steps < 1
        or config.max_seconds <= 0
        or config.checkpoint_every < 1
        or not 0 <= config.ema_decay < 1
        or config.learning_rate <= 0
    ):
        raise ValueError("Invalid training resource or dropout configuration")
    if (
        config.scheduler_source != "project_core"
        or len(identity.get("drop_source_sha256", "")) != 64
    ):
        raise ValueError("Explicit personal DDPM and dropout source required")
    contract = {
        "schema_version": 1,
        "identity": identity,
        "model": _model_identity(model, model_identity),
        "latents": tensor_digest(latents),
        "text_ids": tensor_digest(text_ids),
        "features": tensor_digest(features),
        "masks": tensor_digest(masks),
        "empty_id": empty_id,
        "drop_probability": drop_probability,
        "drop_executed_code": hashlib.sha256(marshal.dumps(drop_condition.__code__)).hexdigest(),
        "schedule": {
            "executed_methods": {
                name: hashlib.sha256(marshal.dumps(getattr(scheduler, name).__code__)).hexdigest()
                for name in ("__init__", "add_noise", "loss", "_extract", "_posterior", "step")
            },
            "buffers": {k: tensor_digest(v) for k, v in scheduler.state_dict().items()},
            "num_steps": scheduler.num_steps,
            "clip_denoised": scheduler.clip_denoised,
        },
        "batch_size": batch_size,
        "seed": config.seed,
        "device": config.device,
        "learning_rate": config.learning_rate,
        "ema_decay": config.ema_decay,
        "torch": str(torch.__version__),
    }
    contract_hash = identity_digest(contract)
    directory = Path(config.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    contract_path = directory / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing conditional training directory has changed identity")
    saved = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    if saved and saved["contract"] != contract:
        raise ValueError("Conditional resume contract changed")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    seed_all(config.seed)
    model.to(config.device).train()
    scheduler.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    stream = StatefulBatchStream(
        TensorDataset(latents, text_ids), batch_size, data_id=contract_hash, seed=config.seed + 1
    )
    step = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        ema.load_state_dict(saved["ema"])
        stream.load_state_dict(saved["data_state"])
        step = saved["step"]
        _restore_rng(saved["rng"])
    start_step = step
    start = time.monotonic()
    metrics = []
    reason = "max_steps"
    empty = features[empty_id : empty_id + 1].float().to(config.device)
    empty_mask = masks[empty_id : empty_id + 1].to(config.device)
    checkpoint = directory / "latest.pt"
    with (directory / "metrics.jsonl").open("a" if saved else "w") as log:
        while step < config.max_steps:
            if time.monotonic() - start >= config.max_seconds:
                reason = "time_limit"
                break
            clean, ids = next(stream)
            clean = clean.to(config.device)
            condition = features[ids].float().to(config.device)
            mask = masks[ids].to(config.device)
            condition, mask = drop_condition(condition, mask, empty, empty_mask, drop_probability)
            noise = torch.randn_like(clean)
            times = torch.randint(scheduler.num_steps, (len(clean),), device=config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = scheduler.loss(
                model(scheduler.add_noise(clean, noise, times), times, condition, mask), noise
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite conditional loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters(), strict=True):
                    target.lerp_(source, 1 - config.ema_decay)
            step += 1
            row = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": float(norm),
                "elapsed_seconds": time.monotonic() - start,
            }
            metrics.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % config.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step,
                        "config": asdict(config),
                        "contract": contract,
                        "rng": _rng_state(config.device),
                        "data_state": stream.state_dict(),
                    },
                )
                print(f"conditional step={step} loss={loss.item():.5f}", flush=True)
    save_checkpoint(
        checkpoint,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "step": step,
            "config": asdict(config),
            "contract": contract,
            "rng": _rng_state(config.device),
            "data_state": stream.state_dict(),
        },
    )
    result = {
        "start_step": start_step,
        "final_step": step,
        "stop_reason": reason,
        "elapsed_seconds": time.monotonic() - start,
        "config": asdict(config),
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "contract_sha256": contract_hash,
        "resume_mode": "single_process_same_environment_exact",
    }
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
