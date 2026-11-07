"""P2 配对、身份、持久化和审核门槛的离线回归。"""

import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.data.representation_cache import (
    CLIP_ENCODING,
    CacheSource,
    RepresentationCache,
    build_cache,
    cache_source,
    verify_cache_acceptance,
    verify_review,
)


class _VAE:
    """用输入红通道常数填满潜变量，便于独立检查图像对齐。"""

    scale = 0.125
    device = "cpu"
    identity: JsonDict = {"revision": "toy-vae", "files": {}}

    def encode(
        self,
        images: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b c_z h_z w_z"]:
        """让潜变量保留图像首个红色像素的数值。

        Args:
            images: [b,3,h,w] 映射到 [-1,1] 的颜色图。

        Results:
            Float[Tensor, "b c_z h_z w_z"]: 每行值均为对应的像素。
        """
        return images[:, :1, :1, :1].expand(-1, 4, 32, 32)


class _CLIP:
    """用可手算的数字描述编码到每个位置，并区分空条件。"""

    device = "cpu"
    identity: JsonDict = {"revision": "toy-clip", "files": {}, "encoding": CLIP_ENCODING}

    def encode(
        self,
        captions: list[str],
    ) -> SimpleNamespace:
        """把数字描述及空条件映射到独立固定张量。

        Args:
            captions: 每个条目是十进制数字字符串或空字符串。

        Raises:
            ValueError: 测试替身无法解析非数字非空输入。

        Results:
            SimpleNamespace: features 为 [b,77,512]，mask 为 [b,77]。
        """
        text = torch.tensor([float(s or "-1") for s in captions])
        lengths = torch.tensor([3 if s else 2 for s in captions])
        return SimpleNamespace(
            features=text[:, None, None].expand(-1, 77, 512),
            mask=torch.arange(77)[None, :] < lengths[:, None],
        )


def _write(
    path: Path,
    payload: JsonDict,
) -> Path:
    """写入测试锁文件并返回其路径。

    Args:
        path: 临时目录中的 JSON 路径。
        payload: 可序列化的对象。

    Raises:
        OSError/TypeError: 写入或序列化失败时透传。

    Results:
        Path: 供测试加载的同一个路径。

    副作用:
        创建或覆盖指定的测试文件。
    """
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def tiny_source(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """准备两个训练、两个验证和一个测试条目，规格与正式数据相同。

    Args:
        tmp_path: pytest 独立临时目录。

    Results:
        tuple: 数据目录、冻结摘要、VAE 锁与 CLIP 锁。

    副作用:
        创建五张小文件、JSONL 和对应身份文件。
    """
    vae = _write(tmp_path / "vae.json", {**_VAE.identity, "scaling_factor": _VAE.scale})
    clip = _write(tmp_path / "clip.json", _CLIP.identity)
    manifest = tmp_path / "manifest.jsonl"
    records = []
    for index, split in enumerate(("train", "validation", "train", "validation", "test")):
        file = tmp_path / f"{index}.png"
        Image.new("RGB", (256, 256), color=(32 * index, 0, 0)).save(file)
        records.append(
            {
                "sample_id": f"item-{index}",
                "group_id": str(index),
                "image_path": file.name,
                "sha256": hash_file(file),
                "source_id": "toy",
                "source_revision": "toy-v1",
                "attributes": {},
                "split": split,
                "caption": str(index),
                "caption_cleaned": str(100 + index),
                "caption_tokens": 1,
                "height": 256,
                "width": 256,
                "mode": "RGB",
            }
        )
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    freeze = _write(
        tmp_path / "freeze.json",
        {
            "manifest_sha256": hash_file(manifest),
            "dataset_version": "v1",
            "image_size": 256,
            "split_counts": {"train": 2, "validation": 2, "test": 1},
            "identity": {"token_lock_sha256": hash_file(clip), "caption_policy": "toy"},
        },
    )
    return tmp_path, freeze, vae, clip


def _build(
    source: CacheSource,
    target: Path,
) -> None:
    """显式把确定性测试编码器注入真实缓存写入器。

    Args:
        source: 小型冻结清单契约。
        target: 缓存输出目录。

    Raises:
        ValueError/OSError: 真实缓存写入器拒绝不合规输入。

    副作用:
        在测试目录写表示文件。
    """
    build_cache(source, _VAE(), _CLIP(), target, batch_size=1)


def test_pairing_batching_and_reopen(
    tiny_source: tuple[Path, Path, Path, Path],
) -> None:
    """验证不同划分的配对值、空条件、批处理与轻量恢复。

    Args:
        tiny_source: 可重建且带三个划分的小型来源。

    Raises:
        AssertionError: 读回的内容、样本 ID 或默认 collate 不符合契约。
    """
    source = cache_source(*tiny_source, 1)
    target = tiny_source[0] / "payload"
    _build(source, target)
    cache = RepresentationCache(target, source)
    train = cache.split("train")
    valid = cache.split("validation")
    assert len(train) == len(valid) == 1
    assert train[0].sample_id == "item-0"
    assert valid[0].sample_id == "item-1"
    assert cache.verify_rows(source) == {"train": 1, "validation": 1}
    assert train[0].latent[0, 0, 0] == -1
    assert float(valid[0].latent[0, 0, 0]) == pytest.approx(32 / 127.5 - 1, abs=5e-4)
    assert valid[0].text[0, 0] == 1
    assert valid[0].mask.sum() == 3
    assert next(iter(DataLoader(valid, batch_size=1))).latent.shape == (1, 4, 32, 32)
    assert len(pickle.dumps(valid)) < 4096
    assert torch.equal(pickle.loads(pickle.dumps(valid))[0].text, valid[0].text)
    blank, mask = cache.blank()
    assert blank.shape == (77, 512)
    assert torch.all(blank == -1) and mask.sum() == 2
    with pytest.raises(KeyError):
        cache.split("test")
    with pytest.raises(FileExistsError):
        _build(source, target)
    partial = target.rename(target.with_name(".payload.partial"))
    with pytest.raises(ValueError, match="unpublished partial"):
        RepresentationCache(partial, source)


def test_row_scan_rejects_mask_shift_with_valid_payload_hash(
    tiny_source: tuple[Path, Path, Path, Path],
) -> None:
    """确认完整扫描会拒绝哈希正确但已偏离 Caption 对齐约定的掩码。

    Args:
        tiny_source: 含已知 Caption 长度的小型冻结清单。

    Raises:
        AssertionError: 内容扫描没有拒绝实际 token 位置错误的数组。
    """
    source = cache_source(*tiny_source, 1)
    target = source.root / "payload"
    _build(source, target)
    mask_path = target / "validation_mask.npy"
    mask = np.load(mask_path)
    mask[0, 0] = False
    mask[0, 3] = True
    np.save(mask_path, mask)
    index_path = target / "index.json"
    index = json.loads(index_path.read_text())
    index["payload_sha256"][mask_path.name] = hash_file(mask_path)
    _write(index_path, index)
    reader = RepresentationCache(target, source)
    with pytest.raises(ValueError, match="Caption mask mismatch: validation"):
        reader.verify_rows(source)


def test_corrupt_inputs_or_payload_rejected(
    tiny_source: tuple[Path, Path, Path, Path],
) -> None:
    """确认身份变化、载荷损坏与源图替换均不会静默复用。

    Args:
        tiny_source: 真实清单格式的小数据。

    Raises:
        AssertionError: 未拒绝任一种损坏或留下半成品。
    """
    source = cache_source(*tiny_source, 1)
    target = source.root / "payload"
    _build(source, target)
    altered = replace(source, identity={**source.identity, "caption_policy": "changed"})
    with pytest.raises(ValueError, match="identity"):
        RepresentationCache(target, altered)
    (target / "validation_text.npy").write_bytes(b"broken")
    with pytest.raises(ValueError, match="payload hash"):
        RepresentationCache(target, source)
    (source.root / "1.png").write_bytes(b"broken")
    failed_target = source.root / "failed"
    with pytest.raises(ValueError, match="Image hash mismatch"):
        _build(source, failed_target)
    assert not failed_target.exists()
    assert not list(source.root.glob(".cache-*"))
    tiny_source[3].write_text("{}")
    with pytest.raises(ValueError, match="tokenizer"):
        cache_source(*tiny_source, 1)


def test_formal_requires_bound_review(
    tiny_source: tuple[Path, Path, Path, Path],
) -> None:
    """验证正式构建需明确审核，错误身份或后改报告均无效。

    Args:
        tiny_source: 不含真实审核数据的测试来源。

    Raises:
        AssertionError: 审核门槛或证据绑定条件未执行。
    """
    source = cache_source(*tiny_source, None)
    source = replace(
        source,
        records={
            **source.records,
            "train": [{"sample_id": f"reviewed-{i}"} for i in range(200)],
        },
    )
    with pytest.raises(ValueError, match="user approval"):
        verify_review(source, None, None)
    reconstruction = {
        "manifest_sha256": source.identity["manifest_sha256"],
        "random_sample_ids": [row["sample_id"] for row in source.records["train"]],
        "reconstruction": {
            "count_random": 200,
            "latent_shape": [4, 32, 32],
            "scale": _VAE.scale,
            "model": _VAE.identity,
            "samples_sha256": "0" * 64,
            "decoded_sha256": "1" * 64,
        },
        "aggregate": {
            "count": 200,
            **{
                key: {"nonfinite_count": 0, "mean": 0.5}
                for key in ("psnr_db", "ssim", "lpips_alex_v01")
            },
        },
    }
    evidence = _write(source.root / "reviewed.json", reconstruction)
    receipt = _write(
        source.root / "receipt.json",
        {
            "decision": "approved",
            "reviewed_by": "test fixture",
            "manifest_sha256": source.identity["manifest_sha256"],
            "vae_lock_sha256": hash_file(tiny_source[2]),
            "reconstruction_sha256": hash_file(evidence),
        },
    )
    assert verify_review(source, receipt, evidence) == hash_file(receipt)
    with pytest.raises(ValueError, match="20-image local FP32"):
        verify_cache_acceptance(source, receipt, evidence, None)
    evidence.write_text(evidence.read_text() + "\n")
    with pytest.raises(ValueError, match="approval does not match"):
        verify_review(source, receipt, evidence)
    _write(evidence, {**reconstruction, "manifest_sha256": "wrong"})
    with pytest.raises(ValueError, match="evidence does not match"):
        verify_review(source, receipt, evidence)


def test_formal_reader_requires_exact_receipt(
    tiny_source: tuple[Path, Path, Path, Path],
) -> None:
    """确保试点和不同审核的缓存不会被当成已核验的正式缓存。

    Args:
        tiny_source: 小型冻结清单。

    Raises:
        AssertionError: 读取器未拒绝缺失或不匹配的审核回执。
    """
    source = cache_source(*tiny_source, 1)
    target = source.root / "payload"
    _build(source, target)
    formal = replace(source, identity={**source.identity, "kind": "formal"})
    with pytest.raises(ValueError, match="identity"):
        RepresentationCache(
            target, formal, approval_sha256="a" * 64, local_acceptance_sha256="c" * 64
        )
    index = target / "index.json"
    metadata = json.loads(index.read_text())
    _write(
        index,
        {
            **metadata,
            "identity": formal.identity,
            "approval_sha256": "a" * 64,
            "local_acceptance_sha256": "c" * 64,
        },
    )
    with pytest.raises(ValueError, match="approval"):
        RepresentationCache(target, formal)
    with pytest.raises(ValueError, match="approval"):
        RepresentationCache(
            target, formal, approval_sha256="b" * 64, local_acceptance_sha256="c" * 64
        )
    with pytest.raises(ValueError, match="local_acceptance"):
        RepresentationCache(target, formal, approval_sha256="a" * 64)
    with pytest.raises(ValueError, match="local_acceptance"):
        RepresentationCache(
            target, formal, approval_sha256="a" * 64, local_acceptance_sha256="d" * 64
        )
    reader = RepresentationCache(
        target, formal, approval_sha256="a" * 64, local_acceptance_sha256="c" * 64
    )
    assert len(reader.split("train")) == 1
