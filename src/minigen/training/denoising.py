"""显式注入目标的有界去噪训练、EMA、日志与恢复工程。"""

import copy
import json
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from minigen._typing import Batch, Checkpoint, DenoisingObjective, JsonDict


@dataclass(frozen=True)
class TrainConfig:
    """保存单卡去噪短训和恢复所需的不可变配置。

    Args:
        run_dir: 本地输出目录。
        max_steps/max_seconds: 优化总步数和本次运行时间上限。
        learning_rate/ema_decay: 优化器学习率和 EMA 衰减。
        seed/device: 随机种子与执行设备。
        checkpoint_every: 保存间隔。
        scheduler_source: 显式调度器来源标记。
        require_exact_resume: 是否拒绝缺少数据顺序状态的恢复。

    Results:
        TrainConfig: TrainConfig 实例，交给 train_denoiser；合法性由训练入口检查。
    """

    run_dir: str
    max_steps: int = 1000
    max_seconds: float = 900
    learning_rate: float = 2e-4
    ema_decay: float = 0.99
    seed: int = 20261026
    device: str = "cpu"
    checkpoint_every: int = 100
    scheduler_source: str = "caller_supplied"
    require_exact_resume: bool = False


def seed_all(
    seed: int,
) -> None:
    """同步 Python、NumPy、Torch 与可用加速设备的随机种子。

    Args:
        seed: 本次实验种子；跨设备设置不保证逐位相同。

    Raises:
        ValueError: NumPy 不接受该种子范围时透传；设备初始化异常透传。

    副作用:
        覆盖 Python、NumPy、Torch 及可用 MPS/CUDA 的全局随机状态。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(
    device: str,
) -> Checkpoint:
    """采集恢复训练需要的进程和指定设备随机状态。

    Args:
        device: 设备名，mps/cuda 时额外保存相应 RNG 状态。

    Raises:
        RuntimeError: 请求的设备 RNG 不可用时由 Torch 透传。

    Results:
        Checkpoint: 异构字典，含 Python 元组、NumPy 状态列表和 Torch 字节张量。
    """
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    name, values, position, has_gauss, cached_gauss = np.random.get_state()
    # 用普通列表保存 NumPy 状态，使 weights_only 加载无需恢复任意数组对象。
    state["numpy"] = [name, values.tolist(), position, has_gauss, cached_gauss]
    if device.startswith("mps"):
        state["mps"] = torch.mps.get_rng_state().cpu()
    if device.startswith("cuda"):
        state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(
    state: Checkpoint,
) -> None:
    """恢复 Checkpoint 中的 Python、NumPy 与 Torch 随机状态。

    Args:
        state: 由 _rng_state 产生的异构字典；包含可选 MPS/CUDA 状态。

    Raises:
        KeyError: 缺少必要状态键；ValueError/RuntimeError: 保存状态格式或设备不兼容时透传。

    副作用:
        覆盖当前进程和保存状态所涉及设备的随机状态。
    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, values, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state(
        (name, np.asarray(values, dtype=np.uint32), position, has_gauss, cached_gauss)
    )
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path: Path,
    payload: Checkpoint,
) -> None:
    """先写临时文件，再替换目标 Checkpoint，避免留下半文件。

    Args:
        path: 最终本地文件路径，父目录由调用者准备。
        payload: 模型、优化器、EMA、步数、配置与随机状态等异构内容。

    Raises:
        OSError/RuntimeError: 路径不可写或 Torch 序列化失败时透传。

    副作用:
        写临时文件后替换目标 Checkpoint。
    """
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def probe_loss(
    model: nn.Module,
    scheduler: DenoisingObjective,
    probe: tuple[
        Float[Tensor, "b c h w"],
        Float[Tensor, "b c h w"],
        Int[Tensor, " b"],
    ],
    device: str,
) -> float:
    """在固定图像、噪声和时间上测量无梯度去噪损失。

    Args:
        model: forward 接收带噪图像和时间的网络，调用前的训练模式会恢复。
        scheduler: 调用方显式提供的加噪与损失对象。
        probe: 依次为 clean、noise、timesteps，形状 [b,c,h,w]、同形和 [b]。
        device: 测量设备，三个固定输入会迁移到该设备。

    Results:
        float: Python float 损失，用于同一监督下的训练进展对照。

    副作用:
        临时切换模型到 eval，正常返回时恢复原模式；异常路径不保证恢复。
    """
    was_training = model.training
    model.eval()
    clean, noise, times = (x.to(device) for x in probe)
    prediction = model(scheduler.add_noise(clean, noise, times), times)
    loss = scheduler.loss(prediction, noise).item()
    model.train(was_training)
    return loss


def train_denoiser(
    model: nn.Module,
    scheduler: DenoisingObjective,
    loader: Iterable[Batch],
    config: TrainConfig,
    *,
    probe: tuple[
        Float[Tensor, "b c h w"],
        Float[Tensor, "b c h w"],
        Int[Tensor, " b"],
    ]
    | None = None,
    fixed_batch: tuple[
        Float[Tensor, "b c h w"],
        Float[Tensor, "b c h w"],
        Int[Tensor, " b"],
    ]
    | None = None,
    resume_path: str | None = None,
) -> JsonDict:
    """执行有步数和时间上限的噪声预测训练，记录 EMA 与恢复证据。

    Args:
        model: 待更新网络，forward([b,c,h,w],[b]) 返回同形噪声。
        scheduler: 显式目标对象，提供 num_steps、add_noise、loss；不自动使用参考实现。
        loader: 可迭代 batch；batch 可为图像张量或首项为图像的列表/元组。
        config: 训练资源、日志、种子、设备和恢复约束。
        probe: 可选固定 (clean,noise,times)，用于前后与 EMA 测量。
        fixed_batch: 可选固定监督，提供后不随机抽样噪声和时间。
        resume_path: 可选恢复文件；精确数据顺序需要 stateful loader。

    Raises:
        ValueError: 训练限制非正、恢复配置或数据状态不匹配；FloatingPointError: 非有限损失。
        RuntimeError: 非有限梯度或权重/设备不兼容时透传；
            OSError/KeyError: 文件或旧状态缺项时透传。

    Results:
        JsonDict: 去噪训练摘要与复现信息。
            config: 实际运行配置。
            start_step/final_step: 恢复起点和实际结束步数。
            stop_reason: max_steps 或 time_limit。
            metrics: 每步损失、梯度范数与耗时。
            initial_probe_loss/final_probe_loss/ema_probe_loss: 固定探针指标，未提供时为 None。
            checkpoint: 最终文件位置。
            resume_mode/resume_limitation: 恢复方式及真实保证范围。

    副作用:
        更新传入模型的参数和训练模式，消耗/恢复随机状态，推进数据流。
        写日志、摘要和 Checkpoint，维护 EMA 并打印训练进度。
    """
    if config.max_steps <= 0 or config.max_seconds <= 0 or config.checkpoint_every <= 0:
        raise ValueError("步数、时间和保存间隔必须为正")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    model.to(config.device).train()
    if hasattr(scheduler, "to"):
        scheduler.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    iterator = iter(loader)
    start_step = 0
    if resume_path:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("learning_rate", "ema_decay", "seed", "scheduler_source", "device"):
            if saved["config"][key] != getattr(config, key):
                raise ValueError(f"Resume configuration changed: {key}")
        if saved.get("data_state") is not None:
            if not hasattr(loader, "load_state_dict"):
                raise ValueError("Saved data state requires a stateful loader")
            loader.load_state_dict(saved["data_state"])
        elif config.require_exact_resume:
            raise ValueError("Checkpoint has no exact data-order state")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        ema.load_state_dict(saved["ema"])
        start_step = saved["step"]
        _restore_rng(saved["rng"])
    if fixed_batch is not None:
        fixed_batch = tuple(x.to(config.device) for x in fixed_batch)
    initial_probe = (
        probe_loss(model, scheduler, probe, config.device) if probe is not None else None
    )
    start = time.monotonic()
    metrics = []
    stop_reason = "max_steps"
    step = start_step
    log_path = run_dir / "metrics.jsonl"
    with log_path.open("a" if resume_path else "w") as log:
        for step in range(start_step + 1, config.max_steps + 1):
            if time.monotonic() - start >= config.max_seconds:
                step -= 1
                stop_reason = "time_limit"
                break
            if fixed_batch is not None:
                clean, noise, times = fixed_batch
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                clean = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(config.device)
                noise = torch.randn_like(clean)
                times = torch.randint(scheduler.num_steps, (len(clean),), device=config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = scheduler.loss(model(scheduler.add_noise(clean, noise, times), times), noise)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("非有限训练损失")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters(), strict=True):
                    # EMA = decay × 旧值 + (1-decay) × 当前参数。
                    target.lerp_(source, 1 - config.ema_decay)
            row = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.monotonic() - start,
            }
            metrics.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % config.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / "latest.pt",
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step,
                        "config": asdict(config),
                        "rng": _rng_state(config.device),
                        "data_state": loader.state_dict()
                        if hasattr(loader, "state_dict")
                        else None,
                    },
                )
                print(
                    f"step={step} loss={row['loss']:.5f} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    final_probe = probe_loss(model, scheduler, probe, config.device) if probe is not None else None
    ema_probe = probe_loss(ema, scheduler, probe, config.device) if probe is not None else None
    checkpoint = run_dir / "latest.pt"
    save_checkpoint(
        checkpoint,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "step": step,
            "config": asdict(config),
            "rng": _rng_state(config.device),
            "data_state": loader.state_dict() if hasattr(loader, "state_dict") else None,
        },
    )
    result = {
        "config": asdict(config),
        "start_step": start_step,
        "final_step": step,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - start,
        "initial_probe_loss": initial_probe,
        "final_probe_loss": final_probe,
        "ema_probe_loss": ema_probe,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "resume_mode": "stateful_exact" if hasattr(loader, "state_dict") else "rng_only",
        "resume_limitation": (
            "Deterministic dataset, num_workers=0, same device/environment only"
            if hasattr(loader, "state_dict")
            else "No shuffled-loader mid-epoch order restoration"
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
