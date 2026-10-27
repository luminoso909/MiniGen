"""Bounded VAE experiments with explicitly supplied learner/reference functions."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from minigen.training.denoising import _restore_rng, _rng_state, save_checkpoint, seed_all


@dataclass(frozen=True)
class VAEConfig:
    run_dir: str
    beta: float = 1.0
    max_steps: int = 1000
    max_seconds: float = 300
    learning_rate: float = 0.001
    seed: int = 0
    device: str = "cpu"
    checkpoint_every: int = 100
    core_source: str = "caller_supplied"


def reconstruction_loss(logits, images):
    """Soft-target BCE: sum pixels per image, then mean over batch."""
    if logits.shape != images.shape or images.min() < 0 or images.max() > 1:
        raise ValueError("Equal image shapes and targets in [0,1] required")
    return (
        F.binary_cross_entropy_with_logits(logits, images, reduction="none")
        .flatten(1)
        .sum(1)
        .mean()
    )


@torch.no_grad()
def evaluate_vae(model, images, *, reparameterize_fn, kl_fn, seed=42, batch_size=128):
    """Fixed CPU noise gives a reproducible probe without consuming training RNG."""
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
            # Ask the supplied KL for each coordinate; no second KL implementation here.
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
    model,
    stream,
    config,
    *,
    reparameterize_fn,
    kl_fn,
    validation,
    fixed_noise=None,
    resume_path=None,
):
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

    def checkpoint(path):
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
