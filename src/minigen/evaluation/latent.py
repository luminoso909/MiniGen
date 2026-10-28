"""Bounded frozen-VAE diagnostics; reports actual completion rather than simulated outputs."""

import time

import torch
from PIL import Image

from minigen.autoencoders.frozen import image_batch, to_image
from minigen.evaluation.icons import aggregate_scores, score_image


def reconstruct(adapter, records, data_root, deadline, batch_size=4):
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


def encode_pilot(adapter, records, data_root, deadline, batch_size=4):
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


def fit_fixed_target(model, inputs, times, target, deadline, max_steps=100):
    """Caller supplies the teaching model; this is a fixed-target optimization diagnostic."""
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
