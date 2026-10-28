"""Frozen model-specific adapter. Loading/downloading is explicit, never an import side effect."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def hash_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FrozenVAE:
    def __init__(self, model, identity, device="cpu"):
        self.model = model.eval().requires_grad_(False).to(device=device)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Frozen VAE must be loaded in FP32")
        self.identity = identity
        self.device = device
        self.scale = float(model.config.scaling_factor)
        if self.scale <= 0:
            raise ValueError("Scaling factor must be positive")

    @classmethod
    def load(cls, root, lock_path, device="cpu"):
        from diffusers import AutoencoderKL

        identity = json.loads(Path(lock_path).read_text())
        directory = Path(root) / identity["local_path"]
        for name, info in identity["files"].items():
            if hash_file(directory / name) != info["sha256"]:
                raise ValueError(f"Model hash mismatch: {name}")
        model = AutoencoderKL.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        return cls(model, identity, device)

    @torch.no_grad()
    def encode(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected [B,3,H,W] image batch")
        if not torch.isfinite(images).all() or images.min() < -1.001 or images.max() > 1.001:
            raise ValueError("Expected finite pixels in [-1,1]")
        return self.model.encode(images.to(self.device)).latent_dist.mode() * self.scale

    @torch.no_grad()
    def decode(self, scaled_latents):
        return self.model.decode(scaled_latents.to(self.device) / self.scale).sample


def image_batch(records, data_root):
    arrays = []
    for record in records:
        with Image.open(Path(data_root) / record["image_path"]) as image:
            arrays.append(np.array(image.convert("RGB"), dtype=np.float32))
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2) / 127.5 - 1


def to_image(tensor):
    pixels = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def cache_identity(adapter, records, manifest_hash):
    return {
        "schema_version": 1,
        "model_revision": adapter.identity["revision"],
        "model_files": adapter.identity["files"],
        "scale": adapter.scale,
        "encoder_mode": "posterior_mean",
        "preprocessing": "RGB/128/no-resize/[-1,1]",
        "dtype": "float32",
        "manifest_sha256": manifest_hash,
        "samples": [{"id": r["sample_id"], "sha256": r["sha256"]} for r in records],
    }


def load_cache(path, expected):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["identity"] != expected:
        raise ValueError("Incompatible latent cache identity")
    if hash_file(path) != meta["payload_sha256"]:
        raise ValueError("Latent payload hash mismatch")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != meta["shape"] or str(array.dtype) != "float32":
        raise ValueError("Invalid latent shape/dtype")
    return torch.from_numpy(array)


def write_cache(path, latents, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = latents.detach().cpu().to(torch.float32).numpy()
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)
    meta = {"identity": identity, "payload_sha256": hash_file(path), "shape": list(array.shape)}
    temp_meta = path.with_suffix(".json.partial")
    temp_meta.write_text(json.dumps(meta, indent=2))
    temp_meta.replace(path.with_suffix(".json"))
