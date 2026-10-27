"""Bounded training with an explicitly supplied scheduler; no reference fallback."""

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainConfig:
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(device: str) -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    name, values, position, has_gauss, cached_gauss = np.random.get_state()
    state["numpy"] = [name, values.tolist(), position, has_gauss, cached_gauss]
    if device.startswith("mps"):
        state["mps"] = torch.mps.get_rng_state().cpu()
    if device.startswith("cuda"):
        state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state: dict) -> None:
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


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def probe_loss(
    model: nn.Module, scheduler, probe: tuple[Tensor, Tensor, Tensor], device: str
) -> float:
    was_training = model.training
    model.eval()
    clean, noise, times = (x.to(device) for x in probe)
    prediction = model(scheduler.add_noise(clean, noise, times), times)
    loss = scheduler.loss(prediction, noise).item()
    model.train(was_training)
    return loss


def train_denoiser(
    model: nn.Module,
    scheduler,
    loader,
    config: TrainConfig,
    *,
    probe: tuple[Tensor, Tensor, Tensor] | None = None,
    fixed_batch: tuple[Tensor, Tensor, Tensor] | None = None,
    resume_path: str | None = None,
) -> dict:
    """Exact order recovery requires StatefulBatchStream and deterministic dataset access."""
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
