"""冻结图像表示的离线重建评测；本模块不会训练模型或写入正式缓存。"""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file


@dataclass(frozen=True, slots=True)
class ReconstructionSample:
    """记录一张训练图像的身份及其在本次审计中的角色。

    Args:
        sample_id: 冻结清单中的唯一图像标识。
        image_path: 相对数据版本根目录的文件路径。
        sha256: 冻结的图像字节摘要。
        group_id: 数据划分时使用的近似变体组。
        cohort: random 或 diagnostic，只有前者纳入主统计。
        reason: 抽样角色的详细说明。
    """

    sample_id: str
    image_path: str
    sha256: str
    group_id: str
    cohort: str
    reason: str


def load_image(
    data_root: Path,
    sample: ReconstructionSample,
) -> Float[Tensor, "3 h w"]:
    """验证图像文件身份并将原生 RGB 像素映射到 [0,1]。

    Args:
        data_root: 冻结数据版本目录。
        sample: 指定相对路径和 SHA-256 的训练样本。

    Raises:
        ValueError: 路径越界、图像哈希不符或图像不是原生 256×256 RGB。
        OSError: 本地图像读取或解码失败。

    Results:
        Float[Tensor, "3 h w"]: CPU FP32，h=w=256，不做重采样。
    """
    path = (data_root / sample.image_path).resolve()
    if not path.is_relative_to(data_root.resolve()) or hash_file(path) != sample.sha256:
        raise ValueError(f"Image path or hash mismatch: {sample.sample_id}")
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(f"Expected native 256x256 RGB: {sample.sample_id}")
        pixels = np.array(image, dtype=np.float32)
    return torch.from_numpy(pixels).permute(2, 0, 1) / 255


def _risk_scores(
    image: Float[Tensor, "3 h w"],
) -> tuple[float, float, float]:
    """从真实图像计算三类高风险候选的粗筛代理量。

    Args:
        image: [3,h,w]、[0,1] 原图；用像素差而非重建结果排序。

    Results:
        tuple[float, float, float]: 平均边缘强度、平均色彩跨度、稀疏高反差细节。
    """
    rgb = image.numpy()
    gray = rgb.mean(axis=0)
    edges = np.abs(np.diff(gray, axis=0, append=gray[-1:])) + np.abs(
        np.diff(gray, axis=1, append=gray[:, -1:])
    )
    dense = float(edges.mean())
    chroma = float((rgb.max(axis=0) - rgb.min(axis=0)).mean())
    sparse = float(np.quantile(edges, 0.99) / (dense + 1e-6))
    return dense, chroma, sparse


def select_samples(
    data_root: Path,
    data_lock: Path,
    *,
    seed: int,
    count: int = 200,
    risk_pool_size: int = 256,
    risks_each: int = 3,
) -> tuple[list[ReconstructionSample], str]:
    """从已冻结的训练切分抽样，并用未入选样本补充诊断案例。

    Args:
        data_root: 包含 manifest.jsonl 及其引用图像的版本目录。
        data_lock: P1 冻结回执，提供完整清单 SHA-256、源 revision、样本数及划分数。
        seed: Python random.Random 的确定性种子。
        count: 主统计均匀无放回抽样数量。
        risk_pool_size: 从剩余训练集无放回抽取的诊断候选数量。
        risks_each: 每类视觉代理最多保留的诊断样本数。

    Raises:
        ValueError: 清单哈希、版本、行数、训练行数或训练样本身份不符；抽样参数无效。
        OSError: 读取清单、冻结回执或诊断候选图像失败。

    Results:
        tuple[list[ReconstructionSample], str]: 主样本在前的清单，以及清单文件 SHA-256。
    """
    frozen = json.loads(data_lock.read_text())
    manifest = data_root / "manifest.jsonl"
    identity = hash_file(manifest)
    if identity != frozen["manifest_sha256"]:
        raise ValueError("ArtBench manifest does not match the P1 freeze receipt")

    train: list[ReconstructionSample] = []
    rows = 0
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            if row["source_revision"] != frozen["source_revision"]:
                raise ValueError("ArtBench manifest contains a different source revision")
            if row["split"] == "train":
                train.append(
                    ReconstructionSample(
                        row["sample_id"],
                        row["image_path"],
                        row["sha256"],
                        row["group_id"],
                        "random",
                        "uniform_without_replacement",
                    )
                )
    if (
        rows != frozen["manifest_rows"]
        or len(train) != frozen["split_counts"]["train"]
        or len({sample.sample_id for sample in train}) != len(train)
    ):
        raise ValueError("ArtBench row counts or train identities do not match the freeze receipt")
    if count <= 0 or risks_each < 0 or risk_pool_size < 3 * risks_each:
        raise ValueError("Invalid reconstruction cohort sizes")

    train.sort(key=lambda sample: sample.sample_id)
    rng = random.Random(seed)
    selected = rng.sample(train, count)
    seen = {sample.sample_id for sample in selected}
    pool = rng.sample([sample for sample in train if sample.sample_id not in seen], risk_pool_size)
    ranked = [(sample, _risk_scores(load_image(data_root, sample))) for sample in pool]
    reasons = ("dense_edges", "strong_chroma", "sparse_high_contrast_detail")
    for axis, reason in enumerate(reasons):
        # 先按风险代理量排序；ID 用于打破同分，重复案例归入第一个类别。
        ordered = sorted(ranked, key=lambda pair: (-pair[1][axis], pair[0].sample_id))
        retained = 0
        for sample, _ in ordered:
            if retained == risks_each:
                break
            if sample.sample_id not in seen:
                selected.append(
                    ReconstructionSample(
                        sample.sample_id,
                        sample.image_path,
                        sample.sha256,
                        sample.group_id,
                        "diagnostic",
                        reason,
                    )
                )
                seen.add(sample.sample_id)
                retained += 1
    return selected, identity


@torch.inference_mode()
def reconstruct(
    adapter: FrozenVAE,
    data_root: Path,
    samples: list[ReconstructionSample],
    output: Path,
    *,
    batch_size: int,
) -> JsonDict:
    """重建固定审计队列并留存原始解码张量与接口诊断。

    Args:
        adapter: 通过固定权重身份加载的 VAE。
        data_root: 训练图像数据版本根目录。
        samples: 主样本及补充案例的有序清单。
        output: 全新的审计目录，不能指向正式训练缓存。
        batch_size: 大于零的图像批量。

    Raises:
        ValueError: 批量或样本不合法、模型未冻结或重建形状、数值、缩放不符。
        OSError: 输入图像读取或本地输出写入失败；底层编解码错误原样透传。

    Results:
        JsonDict: 解码动态范围、缩放诊断、形状、组件与设备元数据。

    副作用:
        创建审计目录，写入样本清单与 FP32 原始解码张量，向终端输出批次进度。
    """
    if (
        not samples
        or batch_size <= 0
        or adapter.model.training
        or any(p.requires_grad for p in adapter.model.parameters())
    ):
        raise ValueError("Expected nonempty samples, positive batch size and a frozen eval VAE")
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples.json").write_text(
        json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2) + "\n"
    )

    started = time.perf_counter()
    decoded = []
    max_scale_delta = 0.0
    for offset in range(0, len(samples), batch_size):
        images = torch.stack(
            [load_image(data_root, sample) for sample in samples[offset : offset + batch_size]]
        )
        vae_input = images.mul(2).sub(1)
        scaled = adapter.encode(vae_input)
        reconstruction = adapter.decode(scaled).cpu()
        if scaled.shape != (len(images), 4, 32, 32) or reconstruction.shape != images.shape:
            raise ValueError("Unexpected 256px ArtBench reconstruction or latent shape")
        if not torch.isfinite(scaled).all() or not torch.isfinite(reconstruction).all():
            raise ValueError("Nonfinite VAE latent or reconstruction")
        if offset == 0:
            raw = adapter.model.encode(vae_input.to(adapter.device)).latent_dist.mode()
            delta = ((scaled / adapter.scale) - raw).abs().max().item()
            max_scale_delta = float(delta)
            if not torch.allclose(scaled / adapter.scale, raw, rtol=1e-5, atol=2e-6):
                raise ValueError("VAE posterior mean and inverse scaling do not agree")
        decoded.append(reconstruction)
        if offset == 0 or len(decoded) % 10 == 0 or offset + batch_size >= len(samples):
            print(
                f"Reconstructed {min(offset + batch_size, len(samples))}/{len(samples)}", flush=True
            )

    raw_decoded = torch.cat(decoded)
    torch.save(raw_decoded, output / "decoded.pt")
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": adapter.device,
        "torch": str(torch.__version__),
        "dtype": str(raw_decoded.dtype),
        "input": "native_rgb_0_255_to_float32_minus1_1",
        "posterior": "mean",
        "scale": adapter.scale,
        "first_batch_inverse_scale_max_abs_error": max_scale_delta,
        "latent_shape": [4, 32, 32],
        "raw_decode_min": raw_decoded.min().item(),
        "raw_decode_max": raw_decoded.max().item(),
        "raw_decode_out_of_range_fraction": ((raw_decoded < -1) | (raw_decoded > 1))
        .float()
        .mean()
        .item(),
        "decoded_sha256": hash_file(output / "decoded.pt"),
        "samples_sha256": hash_file(output / "samples.json"),
        "model": adapter.identity,
    }
