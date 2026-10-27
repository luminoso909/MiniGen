"""Explicit MNIST acquisition and reproducible subsets; never touch test labels."""

import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity


def prepare_mnist(
    root: str | Path, *, train_size: int = 10000, validation_size: int = 1000, seed: int = 20261026
) -> tuple[Subset, Subset, dict]:
    root = Path(root)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST(root / "raw", train=True, transform=transform, download=True)
    if train_size <= 0 or validation_size <= 0 or train_size + validation_size > len(dataset):
        raise ValueError("MNIST 子集大小无效")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_ids, val_ids = order[:train_size], order[train_size : train_size + validation_size]
    payload = {"train": train_ids, "validation": val_ids}
    split_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    resources = []
    for filename, expected_md5 in dataset.resources:
        path = Path(dataset.raw_folder) / filename
        if not check_integrity(path, expected_md5):
            raise RuntimeError(f"MNIST 校验失败: {filename}")
        resources.append(
            {
                "file": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "expected_md5": expected_md5,
                "md5_verified": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "source": "torchvision.datasets.MNIST",
        "torchvision_version": torchvision.__version__,
        "mirrors": dataset.mirrors,
        "seed": seed,
        "split_sha256": split_hash,
        "indices": payload,
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "official_test_used": False,
        "transform": "ToTensor then Normalize(mean=0.5, std=0.5), shape [1,28,28]",
        "resources": resources,
        "disk_bytes": sum(p.stat().st_size for p in (root / "raw").rglob("*") if p.is_file()),
    }
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "mnist_week1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return Subset(dataset, train_ids), Subset(dataset, val_ids), manifest
