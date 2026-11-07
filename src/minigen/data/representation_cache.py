"""ArtBench 冻结表示缓存：分割内对齐、可校验、按行读取。"""

import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file, image_batch
from minigen.conditioning.clip import FrozenCLIPText
from minigen.data.manifest import load_manifest

SPLITS = ("train", "validation")
LAYOUT = {
    "latents": ((4, 32, 32), np.dtype("<f2")),
    "text": ((77, 512), np.dtype("<f2")),
    "mask": ((77,), np.dtype("?")),
}
SCHEMA = "minigen.artbench.representation.v1"
CLIP_ENCODING = {
    "version": "clip-text-single-v1",
    "context_tokens": 77,
    "max_content_tokens": 75,
    "hidden_width": 512,
    "features": "last_hidden_state",
    "compute_dtype": "float32",
    "mask": "true for BOS/content/EOS; false for right padding",
    "bos_token_id": 49406,
    "eos_token_id": 49407,
    "pad_token_id": 49407,
}


@dataclass(frozen=True)
class CacheSource:
    """保存生成及读取同一批 ArtBench 表示所需的数据契约。

    Args:
        root: 含 RGB 图像及 manifest.jsonl 的冻结数据根。
        records: 两个缓存划分的有序样本记录。
        identity: 数据、模型、预处理、存储精度与构建模式的固定身份。
    """

    root: Path
    records: dict[str, list[JsonDict]]
    identity: JsonDict


class Representation(NamedTuple):
    """一个可训练的图文配对；张量从映射文件复制，互不共享写入状态。

    Args:
        sample_id: 冻结清单中的样本标识。
        latent: [4,32,32] FP16 潜变量，已应用 VAE 缩放。
        text: [77,512] FP16 文本 token 表示。
        mask: [77] 布尔有效 token 掩码。
    """

    sample_id: str
    latent: Float[Tensor, "c_z h_z w_z"]
    text: Float[Tensor, "n d"]
    mask: Bool[Tensor, " n"]


def cache_source(
    data_root: Path,
    freeze_path: Path,
    vae_lock: Path,
    clip_lock: Path,
    pilot_rows: int | None,
) -> CacheSource:
    """用冻结记录重新派生预期身份；读取缓存时无需加载编码器。

    Args:
        data_root: 实际 ArtBench 数据根目录。
        freeze_path: P1 固定的数据清单和划分摘要。
        vae_lock: 包含 VAE revision、资产哈希和模型缩放的 JSON 锁。
        clip_lock: 包含 CLIP tokenizer 与模型资产的 JSON 锁。
        pilot_rows: 每个划分的试点上限；None 表示完整 train/validation。

    Raises:
        ValueError: 冻结身份、数据字段、划分或试点规模不符。
        OSError/JSONDecodeError/KeyError: 清单、锁定文件读取或解析失败。

    Results:
        CacheSource: 有序记录与独立派生的可比较身份。
    """
    freeze = json.loads(freeze_path.read_text())
    vae = json.loads(vae_lock.read_text())
    clip = json.loads(clip_lock.read_text())
    manifest_path = data_root / "manifest.jsonl"
    manifest_sha = hash_file(manifest_path)
    clip_sha = hash_file(clip_lock)
    if (
        manifest_sha != freeze["manifest_sha256"]
        or clip_sha != freeze["identity"]["token_lock_sha256"]
        or freeze["image_size"] != 256
    ):
        raise ValueError("Frozen manifest, image size or caption tokenizer identity mismatch")
    rows = load_manifest(manifest_path, data_root)
    if Counter(row["split"] for row in rows) != freeze["split_counts"]:
        raise ValueError("Frozen split counts mismatch")
    if pilot_rows is not None and (
        pilot_rows < 1 or any(pilot_rows >= freeze["split_counts"][split] for split in SPLITS)
    ):
        raise ValueError("Pilot rows must be positive and smaller than each complete split")

    records = {split: [] for split in SPLITS}
    for row in rows:
        split = row["split"]
        if split not in records:
            continue
        caption = row["caption"]
        content_tokens = row["caption_tokens"]
        if (
            row["mode"] != "RGB"
            or (row["width"], row["height"]) != (256, 256)
            or not isinstance(caption, str)
            or not caption.strip()
            or type(content_tokens) is not int
            or not 1 <= content_tokens <= 75
        ):
            raise ValueError(f"Invalid ArtBench representation input: {row['sample_id']}")
        if pilot_rows is None or len(records[split]) < pilot_rows:
            records[split].append(
                {
                    key: row[key]
                    for key in ("sample_id", "image_path", "sha256", "caption", "caption_tokens")
                }
            )
    return CacheSource(
        data_root,
        records,
        {
            "schema": SCHEMA,
            "kind": "formal" if pilot_rows is None else "pilot",
            "pilot_rows_per_split": pilot_rows,
            "manifest_sha256": manifest_sha,
            "freeze_sha256": hash_file(freeze_path),
            "dataset_version": freeze["dataset_version"],
            "caption_policy": freeze["identity"]["caption_policy"],
            "caption_field": "caption",
            "image_preprocessing": "RGB/256x256/no-resize/[-1,1]",
            "vae": {
                "lock_sha256": hash_file(vae_lock),
                "revision": vae["revision"],
                "files": vae["files"],
                "scaling_factor": vae["scaling_factor"],
                "inference": "fp32/posterior_mean",
            },
            "clip": {
                "lock_sha256": clip_sha,
                "revision": clip["revision"],
                "files": clip["files"],
                "encoding": CLIP_ENCODING.copy(),
                "blank": "empty caption",
            },
            "storage": {
                name: {"shape": list(shape), "dtype": dtype.str}
                for name, (shape, dtype) in LAYOUT.items()
            },
        },
    )


def verify_review(
    source: CacheSource,
    approval_path: Path | None,
    evidence_path: Path | None,
) -> str | None:
    """正式构建要求与当前数据、VAE 和 200 张报告一致的审核回执。

    Args:
        source: 本次构建的数据和编码契约。
        approval_path: 用户签署的审核 JSON；试点不需要。
        evidence_path: 本次签署的 200 张重建结构化报告；试点不需要。

    Raises:
        ValueError: 正式审核缺失、身份不匹配或试点错误携带审核。
        OSError/JSONDecodeError: 审核文件读取、解析失败。

    Results:
        str | None: 正式回执 SHA-256，试点返回 None。
    """
    if source.identity["kind"] == "pilot":
        if approval_path is not None or evidence_path is not None:
            raise ValueError("Pilot cache does not use formal approval")
        return None
    if approval_path is None or evidence_path is None:
        raise ValueError("Formal cache requires user approval of the 200-image VAE reconstruction")
    evidence = json.loads(evidence_path.read_text())
    reconstruction = evidence["reconstruction"]
    metrics = evidence["aggregate"]
    sample_ids = set(evidence["random_sample_ids"])
    train_ids = {row["sample_id"] for row in source.records["train"]}
    vae = source.identity["vae"]
    if (
        evidence["manifest_sha256"] != source.identity["manifest_sha256"]
        or reconstruction["count_random"] != 200
        or metrics["count"] != 200
        or len(evidence["random_sample_ids"]) != 200
        or len(sample_ids) != 200
        or not sample_ids <= train_ids
        or reconstruction["latent_shape"] != [4, 32, 32]
        or reconstruction["scale"] != vae["scaling_factor"]
        or any(reconstruction["model"][key] != vae[key] for key in ("revision", "files"))
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", reconstruction[key])
            for key in ("samples_sha256", "decoded_sha256")
        )
        or any(
            metrics[key]["nonfinite_count"] != 0 or not math.isfinite(metrics[key]["mean"])
            for key in ("psnr_db", "ssim", "lpips_alex_v01")
        )
    ):
        raise ValueError("Reconstruction evidence does not match the frozen VAE and data")
    approval = json.loads(approval_path.read_text())
    expected = {
        "decision": "approved",
        "manifest_sha256": source.identity["manifest_sha256"],
        "vae_lock_sha256": source.identity["vae"]["lock_sha256"],
        "reconstruction_sha256": hash_file(evidence_path),
    }
    if (
        any(approval.get(key) != value for key, value in expected.items())
        or not str(approval.get("reviewed_by") or "").strip()
    ):
        raise ValueError("VAE approval does not match the current cache inputs")
    return hash_file(approval_path)


def _inputs(
    source: CacheSource,
    records: list[JsonDict],
) -> Float[Tensor, "b c h w"]:
    """校验图像实际内容和规格，避免清单稳定但本地文件被替换。

    Args:
        source: 提供正式数据根。
        records: 当前批的清单记录。

    Raises:
        ValueError: 图像文件哈希、模式或尺寸不符。
        OSError: 图像缺失、不可读或无法解码时透传。

    Results:
        Float[Tensor, "b c h w"]: CPU FP32 输入 [b,3,256,256]，映射到 [-1,1]。
    """
    for row in records:
        path = source.root / row["image_path"]
        if hash_file(path) != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError(f"Image must be RGB 256x256: {row['sample_id']}")
    return image_batch(records, source.root)


def _store(
    arrays: dict[str, np.memmap],
    offset: int,
    count: int,
    values: dict[str, Tensor],
) -> None:
    """按顺序写一批表示；拒绝非法形状与 FP16 溢出。

    Args:
        arrays: 三个磁盘映射数组，首轴均为样本。
        offset: 该批在划分内的起始行。
        count: 当前输入样本数，输出必须逐项对应。
        values: latents/text/mask 张量。

    Raises:
        ValueError: 编码结果形状、dtype 或有限性不符。

    副作用:
        更新映射文件对应行；调用方负责 flush。
    """
    for name, value in values.items():
        shape, dtype = LAYOUT[name]
        if tuple(value.shape) != (count, *shape) or (
            value.dtype != (torch.bool if name == "mask" else torch.float32)
        ):
            raise ValueError(f"Unexpected representation contract: {name}")
        cpu = value.detach().cpu()
        if name != "mask" and (
            not torch.isfinite(cpu).all() or cpu.abs().amax() > torch.finfo(torch.float16).max
        ):
            raise ValueError(f"Non-finite or FP16-overflowed representation: {name}")
        arrays[name][offset : offset + count] = cpu.numpy().astype(dtype, copy=False)


def build_cache(
    source: CacheSource,
    vae: FrozenVAE,
    clip: FrozenCLIPText,
    output: Path,
    *,
    batch_size: int,
    approval_path: Path | None = None,
    evidence_path: Path | None = None,
) -> None:
    """分批生成图像和文本表示，完整写入后一次发布。

    Args:
        source: 来自 cache_source 的数据和锁定契约。
        vae: 具有匹配身份的冻结 FP32 VAE。
        clip: 具有匹配身份的冻结 FP32 CLIP 文本塔。
        output: 新缓存目录，不能覆盖已有目录。
        batch_size: 编解码的最大批大小。
        approval_path: 正式缓存必需的用户审核回执。
        evidence_path: 正式审核对应的重建报告。

    Raises:
        ValueError: 审核、组件、批大小或编码内容不符。
        FileExistsError/OSError: 目标已存在、图像或磁盘访问失败时透传。

    副作用:
        在目标的父目录写临时文件；成功时将其重命名为最终缓存，失败时清理。
    """
    receipt = verify_review(source, approval_path, evidence_path)
    if batch_size < 1 or any(not source.records[split] for split in SPLITS):
        raise ValueError("Cache requires positive batch size and nonempty splits")
    for component, expected in (
        (vae.identity, source.identity["vae"]),
        (clip.identity, source.identity["clip"]),
    ):
        if any(component[key] != expected[key] for key in ("revision", "files")):
            raise ValueError("Loaded encoder identity does not match cache source")
    if vae.scale != source.identity["vae"]["scaling_factor"]:
        raise ValueError("Loaded VAE scaling does not match cache source")
    if clip.identity.get("encoding") != source.identity["clip"]["encoding"]:
        raise ValueError("Loaded CLIP encoding does not match cache source")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cache-", dir=output.parent) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        blank = clip.encode([""])
        if (
            tuple(blank.features.shape) != (1, 77, 512)
            or tuple(blank.mask.shape) != (1, 77)
            or blank.features.dtype != torch.float32
            or blank.mask.dtype != torch.bool
            or not torch.equal(blank.mask.cpu(), torch.arange(77)[None, :] < 2)
            or not torch.isfinite(blank.features).all()
            or blank.features.abs().amax() > torch.finfo(torch.float16).max
        ):
            raise ValueError("Invalid blank CLIP conditioning")
        np.save(staging / "blank_text.npy", blank.features.cpu().numpy()[0].astype("<f2"))
        np.save(staging / "blank_mask.npy", blank.mask.cpu().numpy()[0])
        for split in SPLITS:
            records = source.records[split]
            arrays = {
                name: np.lib.format.open_memmap(
                    staging / f"{split}_{name}.npy",
                    mode="w+",
                    dtype=dtype,
                    shape=(len(records), *shape),
                )
                for name, (shape, dtype) in LAYOUT.items()
            }
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                latent = vae.encode(_inputs(source, batch))
                condition = clip.encode([row["caption"] for row in batch])
                lengths = torch.tensor([row["caption_tokens"] + 2 for row in batch])
                if not torch.equal(
                    condition.mask.cpu(), torch.arange(77)[None, :] < lengths[:, None]
                ):
                    raise ValueError(f"CLIP mask disagrees with frozen Caption lengths in {split}")
                _store(
                    arrays,
                    start,
                    len(batch),
                    {"latents": latent, "text": condition.features, "mask": condition.mask},
                )
            for array in arrays.values():
                array.flush()
            del arrays, array
        metadata = {
            "identity": source.identity,
            "approval_sha256": receipt,
            "execution": {
                "torch": str(torch.__version__),
                "device": str(vae.device),
                "batch_size": batch_size,
            },
            "sample_ids": {
                split: [row["sample_id"] for row in source.records[split]] for split in SPLITS
            },
            "payload_sha256": {
                path.name: hash_file(path) for path in sorted(staging.glob("*.npy"))
            },
        }
        (staging / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(output)


class RepresentationCache:
    """只读打开预先构建的缓存，提供后续训练使用的划分和独立空条件。

    Args:
        directory: 已完整发布的缓存目录。
        source: 由调用方从当前冻结源和组件锁独立派生的期待身份。
        approval_sha256: 正式模式中由 verify_review 返回的回执摘要；试点为 None。

    Raises:
        ValueError: 身份、顺序、文件摘要、dtype 或形状不符。
        OSError/JSONDecodeError: 文件读取或解析失败。
    """

    def __init__(
        self,
        directory: Path,
        source: CacheSource,
        *,
        approval_sha256: str | None = None,
    ) -> None:
        """校验全部数组后建立只读映射；初始化不把所有张量载入内存。

        Args:
            directory: 含 index.json 和数组的缓存目录。
            source: 当前期待的来源契约与样本顺序。
            approval_sha256: 本次已核验的正式审核回执摘要；试点不传。

        Raises:
            ValueError: 缓存身份、顺序或有效载荷不符。
            OSError/JSONDecodeError: 文件读取或解析失败。
        """
        metadata = json.loads((directory / "index.json").read_text())
        ids = {split: [row["sample_id"] for row in source.records[split]] for split in SPLITS}
        if metadata["identity"] != source.identity or metadata["sample_ids"] != ids:
            raise ValueError("Cache identity or sample order mismatch")
        formal = source.identity["kind"] == "formal"
        if (
            formal != (approval_sha256 is not None)
            or (formal and not re.fullmatch(r"[0-9a-f]{64}", approval_sha256))
            or metadata["approval_sha256"] != approval_sha256
        ):
            raise ValueError("Cache kind and approval receipt mismatch")
        self._ids = ids
        self._directory = directory.resolve()
        names = {"blank_text": LAYOUT["text"], "blank_mask": LAYOUT["mask"]}
        for split, rows in ids.items():
            for name, (shape, dtype) in LAYOUT.items():
                names[f"{split}_{name}"] = ((len(rows), *shape), dtype)
        if set(metadata["payload_sha256"]) != {f"{name}.npy" for name in names}:
            raise ValueError("Cache payload file set mismatch")
        for name, (shape, dtype) in names.items():
            path = directory / f"{name}.npy"
            if hash_file(path) != metadata["payload_sha256"][path.name]:
                raise ValueError(f"Cache payload hash mismatch: {path.name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Cache payload shape/dtype mismatch: {path.name}")

    def split(
        self,
        name: str,
    ) -> "CachedSplit":
        """为指定划分返回可随机访问的 PyTorch Dataset。

        Args:
            name: train 或 validation。

        Raises:
            KeyError: 请求了未缓存的划分。

        Results:
            CachedSplit: 与清单顺序相同的逐图表示。
        """
        ids = self._ids[name]
        paths = {field: self._directory / f"{name}_{field}.npy" for field in LAYOUT}
        return CachedSplit(ids, paths)

    def blank(
        self,
    ) -> tuple[Float[Tensor, "n d"], Bool[Tensor, " n"]]:
        """取得后续条件丢弃或 CFG 使用的独立空文本表示。

        Results:
            tuple: [77,512] FP16 特征和 [77] 布尔有效掩码。
        """
        text = np.load(self._directory / "blank_text.npy", allow_pickle=False)
        mask = np.load(self._directory / "blank_mask.npy", allow_pickle=False)
        return torch.from_numpy(text), torch.from_numpy(mask)


class CachedSplit(Dataset[Representation]):
    """将划分中的 ID 与磁盘数组绑定为可索引的只读数据集。

    Args:
        ids: 清单顺序的样本标识。
        paths: latents、text 和 mask 三个已验证的数组路径。
    """

    def __init__(
        self,
        ids: list[str],
        paths: dict[str, Path],
    ) -> None:
        """保留已校验的有序索引及其磁盘映射。

        Args:
            ids: 与数组首轴严格对齐的标识。
            paths: 该划分的数组路径。
        """
        self._ids = ids
        self._paths = paths
        self._arrays: dict[str, np.memmap] = {}

    def __getstate__(
        self,
    ) -> tuple[list[str], dict[str, Path]]:
        """向采用 spawn 的 DataLoader 工作进程只传 ID 和路径。

        Results:
            tuple: 样本 ID 与数组路径；不序列化大型映射载荷。
        """
        return self._ids, self._paths

    def __setstate__(
        self,
        state: tuple[list[str], dict[str, Path]],
    ) -> None:
        """从序列化状态恢复轻量索引，按需打开映射。

        Args:
            state: __getstate__ 返回的样本 ID 与数组路径。

        副作用:
            初始化当前工作进程的 Dataset 实例。
        """
        self._ids, self._paths = state
        self._arrays = {}

    def __len__(
        self,
    ) -> int:
        """返回划分中的缓存行数。

        Results:
            int: 可索引的样本总数。
        """
        return len(self._ids)

    def __getitem__(
        self,
        index: int,
    ) -> Representation:
        """从同一行复制潜变量、文本特征与掩码。

        Args:
            index: 划分内整数索引。

        Raises:
            IndexError: 索引超出该划分。
            OSError: 首次读取时映射文件不可访问。

        Results:
            Representation: 有样本 ID 的独立张量副本。
        """
        sample_id = self._ids[index]
        if not self._arrays:
            self._arrays = {
                name: np.load(path, mmap_mode="r", allow_pickle=False)
                for name, path in self._paths.items()
            }
        return Representation(
            sample_id,
            torch.from_numpy(self._arrays["latents"][index].copy()),
            torch.from_numpy(self._arrays["text"][index].copy()),
            torch.from_numpy(self._arrays["mask"][index].copy()),
        )

    def __iter__(
        self,
    ) -> Iterator[Representation]:
        """依照冻结清单顺序逐项返回表示。

        Results:
            Iterator[Representation]: 惰性迭代，不预载全部张量。
        """
        for index in range(len(self)):
            yield self[index]
