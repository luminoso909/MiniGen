"""核对真实数据接入的抽样边界与重建指标的独立可算案例。"""

import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.evaluation.reconstruction import select_samples
from minigen.evaluation.reconstruction_metrics import LPIPSAlex, psnr, ssim
from minigen.evaluation.reconstruction_review import detail_box


def test_reconstruction_split_and_lock(
    tmp_path: Path,
) -> None:
    """检查均匀抽样只使用训练集，诊断组不重叠，清单被修改后立即拒绝。

    Args:
        tmp_path: pytest 隔离目录，容纳小型固定输入。

    Raises:
        AssertionError: 抽样、划分隔离、顺序或数据身份检查不符合约定。

    副作用:
        写入临时图像、数据清单和冻结身份。
    """
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (256, 256), (100, 50, 20)).save(image_path)
    digest = hash_file(image_path)
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "sample_id": f"ab-{idx:05d}",
            "image_path": image_path.name,
            "source_revision": "revision",
            "sha256": digest,
            "group_id": f"group-{idx}",
            "split": "train" if idx < 8 else "validation" if idx == 8 else "test",
        }
        for idx in reversed(range(10))
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    lock = tmp_path / "freeze.json"
    lock.write_text(
        json.dumps(
            {
                "manifest_sha256": hash_file(manifest),
                "source_revision": "revision",
                "manifest_rows": 10,
                "split_counts": {"train": 8, "validation": 1, "test": 1},
            }
        )
    )

    samples, manifest_hash = select_samples(
        tmp_path, lock, seed=42, count=2, risk_pool_size=6, risks_each=1
    )
    expected = random.Random(42).sample([f"ab-{idx:05d}" for idx in range(8)], 2)
    assert [sample.sample_id for sample in samples[:2]] == expected
    assert [sample.cohort for sample in samples] == ["random"] * 2 + ["diagnostic"] * 3
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert all(sample.sample_id < "ab-00008" for sample in samples)
    assert manifest_hash == hash_file(manifest)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="freeze"):
        select_samples(tmp_path, lock, seed=42, count=2, risk_pool_size=6, risks_each=1)


def test_psnr_ssim_match_constant_pixel_formula() -> None:
    """以可手算的常量图对检查逐图 PSNR、SSIM 与恒等输入。

    Raises:
        AssertionError: 数值、输出形状或范围拒绝不符合定义。
    """
    reference = torch.zeros((2, 3, 12, 12), dtype=torch.float64)
    estimated = reference.clone()
    estimated[1] = 0.25

    distances = psnr(reference, estimated)
    assert distances.shape == (2,)
    assert distances[0].item() == float("inf")
    assert distances[1].item() == pytest.approx(12.041199826559248)
    expected_ssim = torch.tensor([1, 0.01**2 / (0.25**2 + 0.01**2)], dtype=reference.dtype)
    torch.testing.assert_close(ssim(reference, estimated), expected_ssim, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="finite"):
        psnr(reference, estimated - 0.001)
    with pytest.raises(ValueError, match="same-size"):
        ssim(reference[:, :, :8], estimated[:, :, :8])


def test_detail_crop_can_reach_bottom_right() -> None:
    """检查局部图使用最大误差区域，且网格不跳过最右和最下边界。

    Raises:
        AssertionError: 未选择能完全覆盖差异的右下 8×8 区域。
    """
    reference = torch.zeros((3, 19, 21))
    estimated = reference.clone()
    estimated[:, -8:, -8:] = 1
    assert detail_box(reference, estimated, size=8, stride=4) == (13, 11, 21, 19)


def test_lpips_rejects_changed_local_weights(
    tmp_path: Path,
) -> None:
    """验证 LPIPS 在尝试反序列化前拒绝同大小但哈希不同的本地权重。

    Args:
        tmp_path: pytest 隔离目录，用于构造损坏的假权重。

    Raises:
        AssertionError: 未按预期拒绝权重身份错误。

    副作用:
        写入临时测试用权重及身份文件。
    """
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"changed")
    lock = tmp_path / "lpips.json"
    info = {"path": weights.name, "bytes": 7, "sha256": "a" * 64}
    lock.write_text(
        json.dumps({"lpips_alex": {"backbone": info, "calibration": {**info, "version": "0.1"}}})
    )
    with pytest.raises(ValueError, match="weight lock mismatch"):
        LPIPSAlex.load(tmp_path, lock)


def test_sdxl_rejects_unsafe_directory_and_wrong_hash(
    tmp_path: Path,
) -> None:
    """确认 VAE 在导入可选模型库或反序列化前拒绝非法资产身份。

    Args:
        tmp_path: 创建假模型文件的隔离目录。

    Raises:
        AssertionError: 越界目录或同大小的哈希错误没有被拒绝。

    副作用:
        写入临时资产目录和模型锁。
    """
    assets = tmp_path / "weights"
    assets.mkdir()
    files = {}
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        (assets / name).write_bytes(b"wrong")
        files[name] = {"sha256": "0" * 64, "bytes": 5}
    identity = {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "revision": "1" * 40,
        "files": files,
        "scaling_factor": 0.13025,
    }
    lock = tmp_path / "sdxl.json"
    lock.write_text(json.dumps({**identity, "local_path": ".."}))
    with pytest.raises(ValueError, match="locked local SDXL"):
        FrozenVAE.load(tmp_path, lock)
    lock.write_text(json.dumps({**identity, "local_path": "weights"}))
    with pytest.raises(ValueError, match="file identity mismatch"):
        FrozenVAE.load(tmp_path, lock)
