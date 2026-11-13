"""构建并回读独立 B1 动漫头像的冻结图文缓存。"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.conditioning.clip import FrozenCLIPText
from minigen.data.cache_writer import IncrementalCacheWriter
from minigen.data.manifest import load_manifest
from minigen.data.representation_cache import (
    CLIP_ENCODING,
    LAYOUT,
    SPLITS,
    CacheSource,
    RepresentationCache,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIRECTORY = Path("data/processed/anime_faces_256/b1_v1")
IMAGE_ROOT = Path("data/processed/anime_faces_256/color_front_v1")
VAE_LOCK = Path("configs/models/sdxl_vae.json")
CLIP_LOCK = Path("configs/models/clip_text.json")
SPLIT_SCHEMA = "minigen.anime_faces.b1.split.v1"
CACHE_SCHEMA = "minigen.anime_faces.representation.v1"
CONTENT_LIMIT = 48


def anime_source(
    manifest: Path,
    receipt_path: Path,
    image_root: Path,
    vae_lock: Path,
    clip_lock: Path,
    smoke_rows: int | None,
) -> CacheSource:
    """从已去重、已分组的动漫头像清单重新派生只读缓存身份。

    Args:
        manifest: 独立 B1 的训练及验证 JSONL 清单。
        receipt_path: 包含清单哈希与划分数量的分组回执。
        image_root: 清洗后图像根，清单路径相对于此目录。
        vae_lock: 锁定 VAE 资产及其缩放的 JSON。
        clip_lock: 锁定 CLIP 文本塔及词表的 JSON。
        smoke_rows: 每个划分取固定前缀的行数；None 取完整划分。

    Raises:
        OSError: 清单、回执或锁文件不可读。
        ValueError: 清单身份、行属性、去重或分组隔离不符。

    Results:
        CacheSource: 每个划分的有序行和与绝对路径无关的身份。
    """
    receipt = json.loads(receipt_path.read_text())
    manifest_sha = hash_file(manifest)
    if receipt["schema"] != SPLIT_SCHEMA or receipt["manifest_sha256"] != manifest_sha:
        raise ValueError("Anime split receipt does not match the manifest")
    rows = load_manifest(manifest, image_root)
    split_counts = Counter(row["split"] for row in rows)
    if set(split_counts) != set(SPLITS) or split_counts != receipt["split_counts"]:
        raise ValueError("Anime split counts do not match the receipt")
    if smoke_rows is not None and not 0 < smoke_rows < min(split_counts.values()):
        raise ValueError("Smoke rows must be positive and smaller than both full splits")

    image_hashes = set()
    groups = {}
    records = {split: [] for split in SPLITS}
    for row in rows:
        split = row["split"]
        if (
            row["sha256"] in image_hashes
            or groups.setdefault(row["group_id"], split) != split
            or (row["mode"], row["width"], row["height"]) != ("RGB", 256, 256)
            or not isinstance(row["caption"], str)
            or not row["caption"].strip()
            or type(row["caption_tokens"]) is not int
            or not 1 <= row["caption_tokens"] <= CONTENT_LIMIT
        ):
            raise ValueError(f"Invalid deduplicated anime cache row: {row['sample_id']}")
        image_hashes.add(row["sha256"])
        if smoke_rows is None or len(records[split]) < smoke_rows:
            records[split].append(row)

    vae = json.loads(vae_lock.read_text())
    clip = json.loads(clip_lock.read_text())
    return CacheSource(
        image_root,
        records,
        {
            "schema": CACHE_SCHEMA,
            "kind": "anime_full" if smoke_rows is None else "anime_smoke",
            "smoke_rows_per_split": smoke_rows,
            "manifest_sha256": manifest_sha,
            "split_receipt_sha256": hash_file(receipt_path),
            "caption_field": "caption",
            "caption_policy": "anime-cleaned-tags-v1",
            "max_content_tokens": CONTENT_LIMIT,
            "image_preprocessing": "RGB/256x256/no-resize/[-1,1]",
            "vae": {
                "lock_sha256": hash_file(vae_lock),
                "revision": vae["revision"],
                "files": vae["files"],
                "scaling_factor": vae["scaling_factor"],
                "inference": "fp32/posterior_mean",
            },
            "clip": {
                "lock_sha256": hash_file(clip_lock),
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


def image_inputs(
    root: Path,
    rows: list[JsonDict],
) -> Float[Tensor, "b 3 h w"]:  # noqa: F722
    """从同一份已核验字节解码图像，按缓存约定归一化。

    Args:
        root: 清洗后图像根。
        rows: 当前有序的非空批，图像均应为 RGB 256×256。

    Raises:
        OSError: 读取或解码图像失败。
        ValueError: 实际文件哈希、颜色模式或图像尺寸不符。

    Results:
        Float[Tensor, "b 3 h w"]: CPU FP32 图像，值域 [-1,1]。
    """
    batch = []
    for row in rows:
        data = (root / row["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['sample_id']}")
        with Image.open(BytesIO(data)) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError(f"Expected an RGB 256x256 image: {row['sample_id']}")
            batch.append(np.array(image))
    return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 127.5 - 1


def storage_array(
    name: str,
    values: Tensor,
    rows: int,
) -> np.ndarray:
    """核验真实编码的张量契约，再转换到冻结的磁盘布局。

    Args:
        name: latents、text 或 mask。
        values: 已按相应布局排列的编码输出，首轴为样本。
        rows: 当前编码批的预期样本数。

    Raises:
        ValueError: 形状、精度、有限性或 FP16 值域不符。

    Results:
        np.ndarray: CPU 数组，使用缓存声明的精度。
    """
    shape, dtype = LAYOUT[name]
    expected_dtype = torch.bool if name == "mask" else torch.float32
    if tuple(values.shape) != (rows, *shape) or values.dtype != expected_dtype:
        raise ValueError(f"Unexpected {name} representation layout")
    cpu = values.detach().cpu().numpy()
    if name != "mask" and (not np.isfinite(cpu).all() or np.abs(cpu).max() > np.finfo(dtype).max):
        raise ValueError(f"{name} representation is non-finite or outside FP16 range")
    return cpu.astype(dtype, copy=False)


def build_anime_cache(
    source: CacheSource,
    vae: FrozenVAE,
    clip: FrozenCLIPText,
    output: Path,
    batch_size: int,
) -> None:
    """增量编码动漫图文并原子发布，遇中断可按同一身份继续。

    Args:
        source: 已核验且含两个划分的动漫头像来源。
        vae: 匹配来源锁的冻结 SDXL VAE。
        clip: 匹配来源锁的冻结 CLIP 文本塔。
        output: 新的发布目录。
        batch_size: 单次处理的最大行数，必须为正。

    Raises:
        ValueError: 批大小、组件、编码输出或恢复身份不符。
        OSError: 读图、磁盘写入、空间或文件锁失败。

    副作用:
        按分割增量写入缓存，完成时一次发布，并向标准错误流显示进度。
    """
    if batch_size < 1:
        raise ValueError("Cache batch size must be positive")
    for name, model_identity in (("vae", vae.identity), ("clip", clip.identity)):
        if any(model_identity[key] != source.identity[name][key] for key in ("revision", "files")):
            raise ValueError(f"Loaded {name} does not match source")
    if (
        vae.scale != source.identity["vae"]["scaling_factor"]
        or clip.identity["encoding"] != source.identity["clip"]["encoding"]
    ):
        raise ValueError("Loaded VAE scale or CLIP token contract does not match source")
    metadata = {
        "identity": source.identity,
        "execution": {
            "torch": str(torch.__version__),
            "devices": {"vae": str(vae.device), "clip": str(clip.device)},
            "batch_size": batch_size,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    layouts = {
        split: {
            name: ((len(source.records[split]), *shape), dtype)
            for name, (shape, dtype) in LAYOUT.items()
        }
        for split in SPLITS
    }
    with IncrementalCacheWriter(output, resume_identity=metadata, layouts=layouts) as writer:
        remaining = sum(len(source.records[split]) - writer.progress(split) for split in SPLITS)
        row_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in LAYOUT.values())
        if shutil.disk_usage(output.parent).free < (remaining + 1) * row_bytes + 64 * 1024**2:
            raise OSError("Insufficient disk space for the remaining anime cache")
        if not writer.has_blank:
            blank = clip.encode([""])
            features = storage_array("text", blank.features, 1)
            mask = storage_array("mask", blank.mask, 1)
            if not np.array_equal(mask[0], np.arange(77) < 2):
                raise ValueError("Blank CLIP conditioning must contain just BOS and EOS")
            writer.write_blank(features[0], mask[0])

        # 相同 Tag 串只计算一次；复用已提交的磁盘行避免把所有文本特征放入内存。
        arrays = {split: writer.arrays(split) for split in SPLITS}
        text_locations: dict[str, tuple[str, int]] = {}
        for split in SPLITS:
            for index, row in enumerate(source.records[split][: writer.progress(split)]):
                text_locations.setdefault(row["caption"], (split, index))
        for split in SPLITS:
            rows = source.records[split]
            committed = writer.progress(split)
            print(f"{split}: {committed}/{len(rows)}", file=sys.stderr, flush=True)
            for start in range(committed, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                stop = start + len(batch)
                latent = vae.encode(image_inputs(source.root, batch))
                arrays[split]["latents"][start:stop] = storage_array("latents", latent, len(batch))
                new_captions = list(
                    dict.fromkeys(
                        row["caption"] for row in batch if row["caption"] not in text_locations
                    )
                )
                encoded = {}
                if new_captions:
                    condition = clip.encode(new_captions)
                    text = storage_array("text", condition.features, len(new_captions))
                    mask = storage_array("mask", condition.mask, len(new_captions))
                    encoded = dict(zip(new_captions, zip(text, mask, strict=True), strict=True))
                for index, row in enumerate(batch, start):
                    caption = row["caption"]
                    if caption in text_locations:
                        origin_split, origin_index = text_locations[caption]
                        text_row = arrays[origin_split]["text"][origin_index]
                        mask_row = arrays[origin_split]["mask"][origin_index]
                    else:
                        text_row, mask_row = encoded[caption]
                    if not np.array_equal(mask_row, np.arange(77) < row["caption_tokens"] + 2):
                        raise ValueError(f"CLIP mask does not match frozen row: {row['sample_id']}")
                    arrays[split]["text"][index] = text_row
                    arrays[split]["mask"][index] = mask_row
                    text_locations.setdefault(caption, (split, index))
                if stop - committed >= 256 or stop == len(rows):
                    writer.commit(split, stop)
                    committed = stop
                    print(f"{split}: {stop}/{len(rows)}", file=sys.stderr, flush=True)

        writer.publish(
            {
                **metadata,
                "sample_ids": {
                    split: [row["sample_id"] for row in source.records[split]] for split in SPLITS
                },
                "unique_captions": len(text_locations),
                "payload_sha256": {
                    path.name: hash_file(path) for path in sorted(writer.directory.glob("*.npy"))
                },
            }
        )


def readback(
    source: CacheSource,
    output: Path,
) -> JsonDict:
    """使用共用只读入口回扫每行，记录缓存的可重验摘要。

    Args:
        source: 由当前清单和组件锁重新派生的期待身份。
        output: 已发布的缓存目录。

    Raises:
        ValueError: 缓存身份、数组、样本顺序、值域或有效位不符。
        OSError: 缓存文件不可读。

    Results:
        JsonDict: 哈希、数组布局、总大小、样本数、空条件与执行信息。
    """
    cache = RepresentationCache(output, source)
    verified = cache.verify_rows(source)
    text, mask = cache.blank()
    index_path = output / "index.json"
    index = json.loads(index_path.read_text())
    return {
        "status": "passed",
        "schema": source.identity["schema"],
        "mode": source.identity["kind"],
        "manifest_sha256": source.identity["manifest_sha256"],
        "split_receipt_sha256": source.identity["split_receipt_sha256"],
        "cache_index_sha256": hash_file(index_path),
        "verified_rows": verified,
        "unique_captions": index["unique_captions"],
        "execution": index["execution"],
        "storage": source.identity["storage"],
        "payload_sha256": index["payload_sha256"],
        "payload_bytes": sum((output / name).stat().st_size for name in index["payload_sha256"]),
        "blank_text_shape": list(text.shape),
        "blank_valid_tokens": int(mask.sum()),
    }


def main() -> None:
    """选择试点、全量或只读验收；所有编码资产均从本地锁文件加载。

    Raises:
        SystemExit: 命令参数错误或编码设备不可用。
        OSError: 本地资产或缓存不可读写。
        ValueError: 数据、组件或缓存身份及内容不一致。

    副作用:
        设置进程 CPU 线程数。编码时创建缓存及回读回执，输出真实运行摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=SPLIT_DIRECTORY / "manifest.jsonl")
    parser.add_argument(
        "--split-receipt", type=Path, default=SPLIT_DIRECTORY / "split_receipt.json"
    )
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-rows", type=int, help="每个分割固定取此前缀数量，默认全量")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    devices = ("cpu", "mps", "cuda")
    parser.add_argument("--device", choices=devices, default="cpu")
    parser.add_argument("--vae-device", choices=devices)
    parser.add_argument("--clip-device", choices=devices)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.batch_size < 1:
        parser.error("batch-size and cpu-threads must be positive")
    start = perf_counter()
    root = args.root.resolve()
    output = root / args.output
    vae_lock, clip_lock = root / VAE_LOCK, root / CLIP_LOCK
    source = anime_source(
        root / args.manifest,
        root / args.split_receipt,
        root / args.image_root,
        vae_lock,
        clip_lock,
        args.smoke_rows,
    )
    torch.set_num_threads(args.cpu_threads)
    if not args.verify_only:
        vae_device = args.vae_device or args.device
        clip_device = args.clip_device or args.device
        selected = (vae_device, clip_device)
        if "cuda" in selected and not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this process")
        if "mps" in selected and not torch.backends.mps.is_available():
            parser.error("MPS is unavailable in this process")
        if "mps" in selected and any(
            os.environ.get(name, "0") != "0"
            for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")
        ):
            parser.error("MPS cache requires fallback and fast math disabled")
        vae = FrozenVAE.load(root, vae_lock, device=vae_device)
        clip = FrozenCLIPText.from_lock(clip_lock, root=root, device=clip_device)
        build_anime_cache(source, vae, clip, output, args.batch_size)
    result = readback(source, output)
    result["elapsed_seconds"] = round(perf_counter() - start, 3)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if not args.verify_only:
        temporary = output / ".readback.json.tmp"
        temporary.write_text(encoded)
        temporary.replace(output / "readback.json")
    print(encoded, end="")


if __name__ == "__main__":
    main()
