"""显式 MNIST 获取、官方资源校验和固定训练/验证划分。"""

import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity

from minigen._typing import JsonDict


def prepare_mnist(
    root: str | Path,
    *,
    train_size: int = 10000,
    validation_size: int = 1000,
    seed: int = 20261026,
) -> tuple[Subset, Subset, JsonDict]:
    """显式获取官方 MNIST 并冻结训练/验证子集与资源校验信息。

    Args:
        root: 本地 data 根目录，raw 和 manifests 写入其下。
        train_size: 从官方训练部分选择的训练样本数。
        validation_size: 从同一官方训练部分留出的验证样本数。
        seed: 固定随机排列的独立种子，测试集不参与。

    Raises:
        ValueError: 子集数量非正或总数超出官方训练集；RuntimeError: 原始资源 MD5 不符。
        下载异常及 OSError 原样透传；调用此函数本身即显式允许 MNIST 下载。

    Results:
        tuple[Subset, Subset, JsonDict]: 训练 Subset、验证 Subset、JSON 元数据；单样本为
            [1,28,28] 浮点图及数字标签，像素 [-1,1]。

    副作用:
        按需下载/解压官方 MNIST，写子集索引、资源校验与清单。
    """
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
