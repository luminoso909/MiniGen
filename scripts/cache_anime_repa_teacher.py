"""从冻结训练像素生成 B2 REPA 的 16×16 CLIP patch 缓存。"""

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import batched
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.evaluation.assets import asset_spec, verify_asset

ROOT = Path(__file__).resolve().parents[1]
ASSET_LOCK = Path("configs/models/evaluation.json")
PROBE_CONFIG = Path("configs/evaluation/anime_repa_teacher_probe.json")
PROBE_RECEIPT = Path("artifacts/exports/anime-faces-b1/repa_teacher_probe/receipt.json")
MANIFEST = Path("data/processed/anime_faces_256/b1_v1/manifest.jsonl")
REPRESENTATION_INDEX = Path("artifacts/cache/anime_faces_256/b1_v1/index.json")
OUTPUT = Path("artifacts/cache/anime_faces_256/b2_repa_clip_v1")
SOURCE = "puruchinera/anime-faces-256"
REVISION = "0f01ad9151f6d53f2ed7227d0a3d2373808f2abd"
RAW_DIR = Path("data/raw/anime_faces_256") / REVISION
SHARDS = (
    (399206445, "5845205365d67e1aeea14e1990995960a8d196c7ce475d42415702f31a64a90f"),
    (375805717, "9911d8b2c4637faff3eb2654982e5776f45f87127bd2b6fe762fb3753070712e"),
    (384047732, "b876b0b2828488acd8ef16f7f0ccfe54be006c69ac6eb64cb3edc063dbaf5f1f"),
    (362983536, "6e10a825e5ca810a15d3e69517515260047192e710d024fcb7792b1c123a1b0b"),
)
FEATURE_SPEC = "rgb256:bicubic336/clip-fp32:last_hidden_state:patch24x24:bilinear16x16:l2"


class FrozenPatchTeacher:
    """离线载入锁定的 CLIP 视觉塔并输出具有空间对应关系的 patch 特征。

    Args:
        root: 已有固定版本资产的 MiniGen 根目录。
        device: 单个视觉前向设备。
    """

    def __init__(
        self,
        root: Path,
        device: torch.device,
    ) -> None:
        """验证完整资产并初始化 FP32 视觉塔与归一化常数。

        Args:
            root: 已有固定版本资产的 MiniGen 根目录。
            device: 单个视觉前向设备。

        Raises:
            ValueError/OSError/RuntimeError: 资产失败、视觉塔权重不全或设备不可用。

        副作用:
            在指定设备分配模型与输入归一化常数。
        """
        from transformers import CLIPVisionModel

        directory = verify_asset(root, ASSET_LOCK, "clip")
        model, info = CLIPVisionModel.from_pretrained(
            directory, local_files_only=True, use_safetensors=False, output_loading_info=True
        )
        if info["missing_keys"] or info["mismatched_keys"]:
            raise ValueError(f"Frozen CLIP vision weights are incomplete: {info}")
        self.device = device
        self.model = model.eval().requires_grad_(False).float().to(device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[
            None, :, None, None
        ]
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[
            None, :, None, None
        ]

    @torch.inference_mode()
    def features(
        self,
        images: list[Image.Image],
    ) -> Float[Tensor, "b n d"]:  # noqa: F722
        """取最终视觉层的空间特征并对齐 B2 的图像网格。

        Args:
            images: 一批 256×256 RGB 训练输入，按缓存顺序排列。

        Raises:
            RuntimeError: 视觉前向、空间变换或设备失败。
            ValueError: 输出含非有限值。

        Results:
            Float[Tensor, "b n d"]: [b,256,1024] FP32，按行展开，沿最后一维归一化。
        """
        pixels = np.stack(
            [np.asarray(im.resize((336, 336), Image.Resampling.BICUBIC)) for im in images]
        )
        batch = torch.from_numpy(pixels).to(self.device, torch.float32).permute(0, 3, 1, 2)
        batch = (batch.div_(255.0) - self.mean) / self.std
        # 去掉 CLS 后仍是源图的行优先网格，在空间轴上插值。
        patches = self.model(pixel_values=batch, return_dict=True).last_hidden_state[:, 1:]
        grid = patches.reshape(-1, 24, 24, 1024).permute(0, 3, 1, 2)
        resized = F.interpolate(grid, size=(16, 16), mode="bilinear", align_corners=False)
        features = F.normalize(resized.flatten(2).transpose(1, 2), dim=-1)
        if not torch.isfinite(features).all().item():
            raise ValueError("Frozen CLIP returned a nonfinite patch feature")
        return features


def source_assets() -> dict[str, tuple[int, str]]:
    """提供固定版本源 Parquet 的逐分片文件身份。

    Results:
        dict: 下载日期的固定行源，文件名对应字节数和 SHA-256。
    """
    return {f"data/train-{i:05d}-of-00004.parquet": record for i, record in enumerate(SHARDS)}


def verify_file(
    path: Path,
    size: int,
    sha256: str,
) -> None:
    """核对本地公开文件和锁定身份。

    Args:
        path: 已存在或刚下载的文件。
        size: 固定文件字节数。
        sha256: 固定内容 SHA-256。

    Raises:
        ValueError: 文件大小或内容不匹配。
        OSError: 无法访问文件。
    """
    if path.stat().st_size != size or hash_file(path) != sha256:
        raise ValueError(f"Asset does not match fixed hash and size: {path}")


def fetch_file(
    url: str,
    path: Path,
    size: int,
    sha256: str,
) -> None:
    """明确下载一个公开固定资产，回读通过后发布。

    Args:
        url: 官方域或镜像域上的固定修订 URL。
        path: 校验后写入的资产目标。
        size: 预期文件字节数。
        sha256: 预期文件 SHA-256。

    Raises:
        ValueError/OSError/CalledProcessError: 网络、路径或资产校验失败。

    副作用:
        续写临时下载，或在临时文件完整时直接验收，通过后重命名。
    """
    if path.exists():
        verify_file(path, size, sha256)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if not partial.exists() or partial.stat().st_size < size:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
    verify_file(partial, size, sha256)
    partial.replace(path)


def fetch_assets(
    root: Path,
    base_url: str,
) -> None:
    """从远端直接取得固定图像分片及视觉塔，无需本机上传。

    Args:
        root: 写入原始分片和模型锁对应路径的项目根。
        base_url: 官方 Hugging Face 或 AutoDL 可访问的镜像地址。

    Raises:
        ValueError/OSError/CalledProcessError: 下载或验证不通过。

    副作用:
        向指定 Hugging Face 域并发请求固定文件并验证缓存，最多六个连接任务。
    """
    jobs = [
        (
            f"{base_url}/datasets/{SOURCE}/resolve/{REVISION}/{name}",
            root / RAW_DIR / name,
            size,
            sha256,
        )
        for name, (size, sha256) in source_assets().items()
    ]
    folder, spec = asset_spec(root, ASSET_LOCK, "clip")
    files = sorted(spec["files"].items(), key=lambda item: item[1]["bytes"], reverse=True)
    for name, identity in files:
        url = f"{base_url}/{spec['repo_id']}/resolve/{spec['revision']}/{name}"
        jobs.append((url, folder / name, identity["bytes"], identity["sha256"]))
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_file, *job) for job in jobs]
        for future in futures:
            future.result()


def verified_probe(
    root: Path,
    receipt: Path,
) -> str:
    """绑定已通过的教师探针与本次教师、清单。

    Args:
        root: 项目根目录。
        receipt: 教师探针回执，相对项目根或绝对路径。

    Raises:
        ValueError/OSError: 探针未通过、与锁定配置不符或无法读取。

    Results:
        str: 与本次运行绑定的探针回执 SHA-256。
    """
    path = root / receipt
    result = json.loads(path.read_text())
    config_path = root / PROBE_CONFIG
    config = json.loads(config_path.read_text())
    if (
        result["structure_passed"] is not True
        or result["probe_config_sha256"] != hash_file(config_path)
        or config["weights_sha256"] != hash_file(root / ASSET_LOCK)
        or config["manifest_sha256"] != hash_file(root / MANIFEST)
    ):
        raise ValueError("The frozen teacher probe must pass for this asset lock and source")
    return hash_file(path)


def training_rows(
    root: Path,
) -> tuple[list[JsonDict], str]:
    """验证动漫清单身份和已有 latent/text 缓存的训练行顺序。

    Args:
        root: 生成原始表示缓存用的项目根。

    Raises:
        ValueError/OSError: 清单哈希、训练样本 ID 或固定源身份不符，或文件不可读。

    Results:
        tuple: 32,892 条训练记录及其完整划分清单的 SHA-256。
    """
    manifest = root / MANIFEST
    index = json.loads((root / REPRESENTATION_INDEX).read_text())
    manifest_sha256 = hash_file(manifest)
    ordered_ids = index["sample_ids"]["train"]
    with manifest.open() as handle:
        rows = [row for row in map(json.loads, handle) if row["split"] == "train"]
    if (
        manifest_sha256 != index["identity"]["manifest_sha256"]
        or len(rows) != 32892
        or [row["sample_id"] for row in rows] != ordered_ids
        or any(row["source_id"] != SOURCE or row["source_revision"] != REVISION for row in rows)
    ):
        raise ValueError(
            "Training manifest differs from the frozen latent/text representation cache"
        )
    return rows, manifest_sha256


def training_image(
    data: bytes,
    row: JsonDict,
) -> Image.Image:
    """按一次清洗时的固定角度再现训练 RGB 像素。

    Args:
        data: 冻结源 Parquet 行中的图像字节。
        row: 包含原始字节和最终像素哈希、旋转角的训练记录。

    Raises:
        ValueError/OSError: 源字节、像素或尺寸不符，或图像无法解码。

    Results:
        Image.Image: 与训练缓存输入逐像素一致的 RGB256 图像。
    """
    if hashlib.sha256(data).hexdigest() != row["original_sha256"]:
        raise ValueError(f"Original Parquet bytes differ: {row['sample_id']}")
    with Image.open(BytesIO(data)) as decoded:
        image = decoded.convert("RGB")
    angle = row["rotation_degrees"]
    if angle:
        pixels = np.asarray(image)
        if abs(angle / 90 - round(angle / 90)) < 1e-6:
            pixels = np.rot90(pixels, round(angle / 90))
        else:
            import cv2

            # 与清洗时一致，旋转后放大以免引入边角补色。
            radians = math.radians(angle)
            scale = abs(math.cos(radians)) + abs(math.sin(radians))
            transform = cv2.getRotationMatrix2D((127.5, 127.5), angle, scale)
            pixels = cv2.warpAffine(
                pixels,
                transform,
                (256, 256),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        image = Image.fromarray(pixels)
    if (
        image.size != (256, 256)
        or hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]
    ):
        raise ValueError(f"Reconstructed training pixels differ: {row['sample_id']}")
    return image


def iter_pixels(
    root: Path,
    rows: list[JsonDict],
) -> Iterator[Image.Image]:
    """逐分片读取精确的训练像素，并保留现有训练缓存行序。

    Args:
        root: 有固定原始分片的项目根目录。
        rows: 需要读取的训练记录，按训练缓存顺序排列。

    Raises:
        OSError/ValueError: 分片无效、待处理行未出现或次序不同。

    Results:
        Iterator[Image.Image]: 按训练缓存顺序逐张产生通过校验的图像。
    """
    from pyarrow.parquet import ParquetFile

    selected = iter(rows)
    pending = next(selected, None)
    for shard, (size, sha256) in source_assets().items():
        if pending is None or pending["source_shard"] != shard:
            continue
        path = root / RAW_DIR / shard
        verify_file(path, size, sha256)
        offset = 0
        parquet = ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=["image"]):
            for index, item in enumerate(batch.column(0).to_pylist(), start=offset):
                if index == pending["source_shard_row"]:
                    yield training_image(item["bytes"], pending)
                    pending = next(selected, None)
                    if pending is None or pending["source_shard"] != shard:
                        break
            if pending is None or pending["source_shard"] != shard:
                break
            offset += batch.num_rows
    if pending is not None:
        raise ValueError(f"Missing canonical Parquet row for {pending['sample_id']}")


def cache_features(
    root: Path,
    output: Path,
    rows: list[JsonDict],
    manifest_sha256: str,
    probe_sha256: str,
    device: torch.device,
    batch_size: int,
) -> JsonDict:
    """按缓存行顺序提取和回读 patch 表示。

    Args:
        root: 项目根，对于视觉塔和源 Parquet 都只读。
        output: 新缓存目录的绝对路径。
        rows: 已验证的完整有序训练记录。
        manifest_sha256: 现有表示缓存对应的清单哈希。
        probe_sha256: 已通过教师探针的回执哈希。
        device: 冻结视觉模型的前向设备。
        batch_size: 单次教师前向的输入张数。

    Raises:
        OSError/ValueError/RuntimeError: 图像重建、教师前向、回读或发布失败。

    Results:
        JsonDict: 已发布索引，不含训练更新。

    副作用:
        写入暂存的 FP16 NumPy 数组，完整回读后发布同级目录。
    """
    if output.exists():
        raise FileExistsError(f"Teacher cache already exists: {output}")
    partial = output.with_name(f".{output.name}-staging")
    partial.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    _, spec = asset_spec(root, ASSET_LOCK, "clip")
    teacher = FrozenPatchTeacher(root, device)
    shape = (len(rows), 256, 1024)
    path = partial / "train.npy"
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    count = 0
    for images in batched(iter_pixels(root, rows), batch_size):
        encoded = teacher.features(list(images)).half().cpu().numpy()
        array[count : count + len(images)] = encoded
        count += len(images)
        if count % (batch_size * 40) == 0:
            print(
                f"[teacher] {count}/{len(rows)} images, {perf_counter() - start:.1f}s", flush=True
            )
    array.flush()
    del array
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        count != len(rows)
        or readback.shape != shape
        or readback.dtype != np.float16
        or any(not np.isfinite(readback[i : i + 128]).all() for i in range(0, count, 128))
    ):
        raise ValueError("Teacher patch cache readback failed")
    ids = [row["sample_id"] for row in rows]
    identity = {
        "schema": "minigen.anime_repa_clip.v1",
        "feature_spec": FEATURE_SPEC,
        "shape": list(shape),
        "dtype": "float16",
        "train_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "weights_sha256": spec["files"]["pytorch_model.bin"]["sha256"],
        "array_sha256": hash_file(path),
        "source_manifest_sha256": manifest_sha256,
        "probe_sha256": probe_sha256,
        "elapsed_seconds": perf_counter() - start,
        "device": str(device),
    }
    (partial / "index.json").write_text(json.dumps(identity, indent=2) + "\n")
    partial.rename(output)
    return identity


def main() -> None:
    """显式下载、预检源图像或缓存已通过探针的教师表示。

    Raises:
        OSError/ValueError/RuntimeError: 配置验证、读取、计算或发布失败。

    副作用:
        按所选动作下载公开资产或写入独立的教师缓存，输出 JSON 回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download", "check", "cache"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--probe", type=Path, default=PROBE_RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--base-url",
        default="https://hf-mirror.com",
        choices=("https://hf-mirror.com", "https://huggingface.co"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "download":
        verified_probe(root, args.probe)
        fetch_assets(root, args.base_url)
        print(json.dumps({"status": "downloaded", "shards": len(SHARDS)}))
        return
    rows, manifest_sha256 = training_rows(root)
    if args.operation == "check":
        counts = {}
        for shard in source_assets():
            cohort = [row for row in rows if row["source_shard"] == shard]
            angled = [row for row in cohort if row["rotation_degrees"] % 90 != 0]
            selected = sorted(cohort[:3] + angled[:3], key=lambda row: row["source_shard_row"])
            selected = list({row["sample_id"]: row for row in selected}.values())
            counts[shard] = sum(1 for _ in iter_pixels(root, selected))
        print(
            json.dumps(
                {
                    "status": "pixels_verified",
                    "manifest_sha256": manifest_sha256,
                    "samples_by_shard": counts,
                },
                indent=2,
            )
        )
        return
    probe_sha256 = verified_probe(root, args.probe)
    device = torch.device(args.device)
    identity = cache_features(
        root, root / args.output, rows, manifest_sha256, probe_sha256, device, args.batch_size
    )
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
