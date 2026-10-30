"""第四周分片缓存与恢复契约测试，显式替身不依赖真实图像生成模型。"""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from minigen._typing import JsonDict
from minigen.autoencoders.frozen import FrozenVAE, hash_file, image_batch
from minigen.data.latent import CachedLatentDataset, build_latent_cache, latent_cache_identity
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.training.denoising import TrainConfig
from minigen.training.latent import train_latent_denoiser


def test_resume_contract_survives_fresh_process(
    tmp_path: Path,
) -> None:
    """验证相同实现的训练契约跨真实解释器进程保持一致，并能恢复继续更新。

    Args:
        tmp_path: pytest 隔离目录，第一进程在此保存状态，第二进程加载。

    Raises:
        AssertionError: 任一进程失败、身份变化或恢复步数不正确。
        OSError: 无法启动项目解释器时透传。

    副作用:
        启动两个 CPU 子进程，写入临时训练日志和 Checkpoint；不下载资源。
    """
    script = """
import sys
import torch
from minigen.models.tiny_unet import TinyUNet
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.training.denoising import TrainConfig
from minigen.training.latent import train_latent_denoiser
torch.set_num_threads(2)
directory, mode = sys.argv[1:]
model = TinyUNet(in_channels=4, base=8, time_width=16)
latents = torch.randn(4, 4, 8, 8, generator=torch.Generator().manual_seed(3))
result = train_latent_denoiser(
    model, DDPMScheduler(num_steps=4), latents,
    TrainConfig(directory, max_steps=int(mode), scheduler_source="project_core"),
    data_identity={"name": "process_fixture"},
    model_identity={"source": "engineering_test", "source_sha256": "a" * 64,
                    "architecture": {"model": "TinyUNet", "base": 8}},
    batch_size=2, resume_path=directory + "/latest.pt" if mode == "2" else None,
)
assert result["start_step"] == int(mode) - 1
assert result["final_step"] == int(mode)
"""
    for mode in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class FakeSpatialVAE(nn.Module):
    """用池化提供确定空间张量，只验证分片、次序、缩放和恢复工程。

    副作用:
        初始化本地测试参数和编码样本计数，不代表真实 VAE 表示能力。
    """

    def __init__(
        self,
    ) -> None:
        """准备冻结适配器所需的参数、缩放与编码计数。

        副作用:
            注册局部测试参数，初始化 encoded_images 为 0。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=0.13)
        self.encoded_images = 0

    def encode(
        self,
        images: Float[Tensor, "b 3 h w"],
    ) -> SimpleNamespace:
        """以 8 倍池化构造四通道假后验均值。

        Args:
            images: [b,3,h,w] 浮点图像，测试采用 128 像素。

        Results:
            SimpleNamespace: 包含 latent_dist.mode 的对象；其结果为 [b,4,h/8,w/8]。

        副作用:
            累加已编码图像数，用于检查恢复时没有重算已提交分片。
        """
        self.encoded_images += len(images)
        pooled = F.avg_pool2d(images, 8)
        values = torch.cat([pooled, pooled[:, :1]], dim=1)
        # mode 是无参回调，返回本次已计算的 [b,4,h/8,w/8]，不重新抽样。
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: values))


@pytest.fixture
def adapter() -> FrozenVAE:
    """建立显式测试编码器，不读取真实模型或下载权重。

    Results:
        FrozenVAE: 使用假空间编码器和可追溯测试身份的冻结外壳。
    """
    return FrozenVAE(FakeSpatialVAE(), {"revision": "fake-1", "files": {}}, "cpu")


@pytest.fixture
def records(
    tmp_path: Path,
) -> list[JsonDict]:
    """创建有序六张小型 RGB 样本及精确哈希，供缓存契约测试。

    Args:
        tmp_path: pytest 隔离的本地文件夹。

    Results:
        list[JsonDict]: train 记录列表；不是正式几何图标数据集。

    副作用:
        在 pytest 临时目录写六张 128×128 PNG，不改项目数据。
    """
    result = []
    for index in range(6):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (128, 128), (index * 30, 100, 200)).save(path)
        result.append(
            {
                "sample_id": f"test_{index}",
                "image_path": path.name,
                "sha256": hash_file(path),
                "split": "train",
            }
        )
    return result


def test_cache_resume_preserves_order_and_requires_completion(
    tmp_path: Path,
    records: list[JsonDict],
    adapter: FrozenVAE,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """检查超时只保留完整分片，拒绝半份训练，续建不重复编码已有样本。

    Args:
        tmp_path: 隔离的图片和缓存目录。
        records: 六条有序测试记录。
        adapter: 计数式测试编码器。
        monkeypatch: pytest 临时补丁，确定性模拟在第一个分片之后超时。

    Raises:
        AssertionError: 完整性、样本次序、编码次数或重载结果不正确。

    副作用:
        编码并写临时缓存，在局部补丁中模拟时间，最后破坏一个测试分片。
    """
    cache = tmp_path / "cache"
    expected = latent_cache_identity(adapter, records, "a" * 64, "train")
    ticks = iter([0.0, 0.0, 0.0, 2.0])

    def clock() -> float:
        """产生可复现的分片间超时，不依赖机器快慢。

        Results:
            float: 前三个调用为 0，之后为 2 秒。

        副作用:
            推进本测试局部时间序列。
        """
        return next(ticks, 2.0)

    with monkeypatch.context() as context:
        context.setattr("minigen.data.latent.time.monotonic", clock)
        partial = build_latent_cache(
            adapter,
            records,
            tmp_path,
            "a" * 64,
            cache,
            split="train",
            batch_size=1,
            shard_size=2,
            max_seconds=1,
        )
    assert partial["status"] == "time_limit" and partial["completed_count"] == 2
    with pytest.raises(ValueError, match="complete"):
        CachedLatentDataset(cache, expected)
    assert adapter.model.encoded_images == 2
    complete = build_latent_cache(
        adapter,
        records,
        tmp_path,
        "a" * 64,
        cache,
        split="train",
        batch_size=2,
        shard_size=2,
    )
    assert complete["completed"] and complete["completed_count"] == 6
    assert adapter.model.encoded_images == 6
    dataset = CachedLatentDataset(cache, expected)
    assert len(dataset) == 6 and dataset.latents.shape == (6, 4, 16, 16)
    torch.testing.assert_close(dataset.latents, adapter.encode(image_batch(records, tmp_path)))
    torch.testing.assert_close(dataset[2], dataset.latents[2], rtol=0, atol=0)
    assert CachedLatentDataset(cache, expected).data_id == dataset.data_id
    payload = cache / "latents_000000_000002.npy"
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    with pytest.raises(ValueError, match="hash"):
        CachedLatentDataset(cache, expected)


@pytest.mark.parametrize("change", ["scale", "revision", "manifest", "order", "split"])
def test_cache_identity_changes_never_reuse(
    tmp_path: Path,
    records: list[JsonDict],
    adapter: FrozenVAE,
    change: str,
) -> None:
    """检查数据、模型和缩放变化会拒绝已有缓存，而不是静默重用。

    Args:
        tmp_path: 隔离目录。
        records: 六条样本记录。
        adapter: 显式测试编码器。
        change: 本次故意修改的身份维度。

    Raises:
        AssertionError: 变化后的身份未导致明确失败。

    副作用:
        写临时测试缓存，并修改局部测试适配器或清单。
    """
    cache = tmp_path / "cache"
    build_latent_cache(adapter, records, tmp_path, "a" * 64, cache, split="train")
    manifest = "a" * 64
    split = "train"
    if change == "scale":
        adapter.scale *= 2
    elif change == "revision":
        adapter.identity["revision"] = "fake-2"
    elif change == "manifest":
        manifest = "b" * 64
    elif change == "order":
        records = list(reversed(records))
    else:
        records = [{**record, "split": "validation"} for record in records]
        split = "validation"
    with pytest.raises(ValueError, match="identity"):
        build_latent_cache(adapter, records, tmp_path, manifest, cache, split=split)


def test_cache_rejects_source_tampering_and_false_completion(
    tmp_path: Path,
    records: list[JsonDict],
    adapter: FrozenVAE,
) -> None:
    """检查原图变化和伪造完整标记均不能成为训练缓存。

    Args:
        tmp_path: 隔离目录。
        records: 原始样本清单。
        adapter: 无真实权重的测试编码器。

    Raises:
        AssertionError: 源文件变化或完整标记不一致没有被拒绝。

    副作用:
        写缓存并故意破坏局部测试原图与索引。
    """
    cache = tmp_path / "cache"
    build_latent_cache(adapter, records, tmp_path, "a" * 64, cache, split="train")
    index_path = cache / "index.json"
    index = json.loads(index_path.read_text())
    index["completed_count"] = 5
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="completion"):
        CachedLatentDataset(cache, latent_cache_identity(adapter, records, "a" * 64, "train"))
    (tmp_path / records[0]["image_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        build_latent_cache(adapter, records, tmp_path, "a" * 64, tmp_path / "other", split="train")


class LinearNoiseModel(nn.Module):
    """用小卷积检验潜空间训练工程，不作为 DiT 实现或生成效果证据。

    副作用:
        初始化测试网络参数，消耗 Torch 随机状态。
    """

    def __init__(
        self,
    ) -> None:
        """建立四通道测试卷积。

        副作用:
            注册局部测试参数并消耗 Torch 随机状态。
        """
        super().__init__()
        self.layer = nn.Conv2d(4, 4, 1)

    def forward(
        self,
        image: Float[Tensor, "b 4 h w"],
        times: Int[Tensor, " b"],
    ) -> Float[Tensor, "b 4 h w"]:
        """将简单时间项加入卷积结果，检查显式条件形状。

        Args:
            image: [b,4,h,w] 测试潜变量。
            times: [b] 整数时间索引。

        Results:
            Float[Tensor, "b 4 h w"]: 同形预测，仅用于工程恢复测试。
        """
        return self.layer(image) + times[:, None, None, None] * 0.001


def model_provenance() -> JsonDict:
    """构造明确标为测试替身的源码与架构说明。

    Results:
        JsonDict: 测试来源，不冒称 notebook_reference 或个人 DiT。
    """
    return {
        "source": "test_stub",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "architecture": {"type": "conv1x1", "channels": 4},
    }


def test_latent_training_resume_matches_continuous(
    tmp_path: Path,
) -> None:
    """比较随机数据顺序、时间、噪声下连续训练与恢复训练的真实状态。

    Args:
        tmp_path: 隔离的三个训练输出目录。

    Raises:
        AssertionError: 参数、EMA、RNG、来源或数据位置未精确恢复。

    副作用:
        运行受限 CPU 工程训练，更新测试模型并写临时 Checkpoint 与日志。
    """
    torch.set_num_threads(2)
    torch.manual_seed(3)
    original = LinearNoiseModel()
    latents = torch.linspace(-1, 1, 40).reshape(5, 4, 1, 2)
    provenance = model_provenance()
    full_model = copy.deepcopy(original)
    full = train_latent_denoiser(
        full_model,
        DDPMScheduler(num_steps=8),
        latents,
        TrainConfig(str(tmp_path / "full"), max_steps=6, scheduler_source="project_core"),
        data_identity={"split": "train", "test": True},
        model_identity=provenance,
        batch_size=2,
        data_seed=19,
    )
    partial = train_latent_denoiser(
        copy.deepcopy(original),
        DDPMScheduler(num_steps=8),
        latents,
        TrainConfig(str(tmp_path / "partial"), max_steps=3, scheduler_source="project_core"),
        data_identity={"split": "train", "test": True},
        model_identity=provenance,
        batch_size=2,
        data_seed=19,
    )
    resumed_model = LinearNoiseModel()
    resumed = train_latent_denoiser(
        resumed_model,
        DDPMScheduler(num_steps=8),
        latents,
        TrainConfig(str(tmp_path / "resumed"), max_steps=6, scheduler_source="project_core"),
        data_identity={"split": "train", "test": True},
        model_identity=provenance,
        batch_size=2,
        data_seed=19,
        resume_path=partial["checkpoint"],
    )
    assert resumed["start_step"] == 3 and resumed["final_step"] == 6
    assert resumed["resume_mode"] == "stateful_exact" and resumed["model_source"] == "test_stub"
    assert resumed["contract_sha256"] == full["contract_sha256"]
    a = torch.load(full["checkpoint"], weights_only=True)
    b = torch.load(resumed["checkpoint"], weights_only=True)
    for group in ("model", "ema"):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
    torch.testing.assert_close(a["data_state"]["order"], b["data_state"]["order"], rtol=0, atol=0)
    assert a["data_state"]["cursor"] == b["data_state"]["cursor"]
    assert [row["loss"] for row in full["metrics"]][3:] == [
        row["loss"] for row in resumed["metrics"]
    ]


@pytest.mark.parametrize(
    "change",
    ["steps", "beta", "clip", "source", "source_hash", "architecture", "data", "tensor", "fixed"],
)
def test_latent_resume_rejects_contract_changes(
    tmp_path: Path,
    change: str,
) -> None:
    """相同形状或架构不足以续训，日程、真实数据、源码与监督也必须一致。

    Args:
        tmp_path: 隔离的旧训练和尝试恢复目录。
        change: 故意修改的身份维度。

    Raises:
        AssertionError: 未拒绝已改变的恢复契约。

    副作用:
        用 CPU 测试网络执行一次更新，保存后尝试不兼容恢复。
    """
    torch.set_num_threads(2)
    latents = torch.zeros(3, 4, 2, 2)
    provenance = model_provenance()
    original = train_latent_denoiser(
        LinearNoiseModel(),
        DDPMScheduler(num_steps=8),
        latents,
        TrainConfig(str(tmp_path / "first"), max_steps=1, scheduler_source="project_core"),
        data_identity={"split": "train"},
        model_identity=provenance,
        batch_size=2,
    )
    scheduler = DDPMScheduler(num_steps=9 if change == "steps" else 8)
    data_identity = {"split": "train"}
    fixed = None
    if change == "beta":
        scheduler = DDPMScheduler(num_steps=8, beta_end=0.03)
    elif change == "clip":
        scheduler.clip_denoised = True
    elif change == "source":
        provenance["source"] = "different_source_same_architecture"
    elif change == "source_hash":
        provenance["source_sha256"] = "0" * 64
    elif change == "architecture":
        provenance["architecture"] = {"type": "same_weights_different_declared_architecture"}
    elif change == "data":
        data_identity["split"] = "validation"
    elif change == "tensor":
        latents = latents + 0.1
    elif change == "fixed":
        fixed = (latents[:2], torch.ones_like(latents[:2]), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="contract changed"):
        train_latent_denoiser(
            LinearNoiseModel(),
            scheduler,
            latents,
            TrainConfig(str(tmp_path / "attempt"), max_steps=2, scheduler_source="project_core"),
            data_identity=data_identity,
            model_identity=provenance,
            batch_size=2,
            fixed_batch=fixed,
            resume_path=original["checkpoint"],
        )
