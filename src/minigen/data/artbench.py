"""固定 ArtBench Parquet 来源的离线准备、身份分组和审计。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import numpy as np
import yaml
from PIL import Image, ImageDraw

from minigen._typing import JsonDict
from minigen.data.artbench_caption import clean_caption
from minigen.data.manifest import load_manifest

IMAGE_SIZE = 256
SOURCE_FIELDS = ("image", "caption", "caption_jp", "artist", "url")
SCHEMA = "minigen.artbench.v1"
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def sha256_file(
    path: Path,
) -> str:
    """流式计算文件 SHA-256。

    Args:
        path: 需要读取的本地文件。

    Raises:
        OSError: 文件无法读取。

    Results:
        str: 十六进制 SHA-256。
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_url(
    value: str,
) -> str:
    """生成分组使用的图片来源键，合并 WikiArt 镜像主机别名。

    Args:
        value: 原始作品图像 URL；空值代表没有来源 URL。

    Results:
        str: 保留图片路径的主机和路径键；缺失时为空。
    """
    if not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname or ""
    except ValueError:
        return value.strip()
    if host.endswith(".wikiart.org") and host.split(".")[0].startswith("uploads"):
        host = "wikiart.org"
    return f"{host}{parsed.path}" if host else value.strip()


def open_source_shards(
    config: JsonDict,
    raw_dir: Path,
) -> list[tuple[str, Path]]:
    """逐文件核验固定版本的分片大小和哈希。

    Args:
        config: 含期望尺寸、样本数和分片锁的来源配置。
        raw_dir: 固定版本原始数据根。

    Raises:
        ValueError: 尺寸、行数、分片路径或大小/哈希与约定不符。
        OSError: 分片缺失或无法读取。

    Results:
        list[tuple[str, Path]]: 按锁顺序返回安全的分片路径。
    """
    if config["image_size"] != IMAGE_SIZE or config["expected_rows"] <= 0:
        raise ValueError("ArtBench requires square 256×256 images and a positive row count")
    root = raw_dir.resolve()
    paths = []
    names = set()
    for spec in config["files"]:
        name = spec["path"]
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or name in names:
            raise ValueError(f"Unsafe or duplicate shard path: {name}")
        names.add(name)
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"ArtBench shard size/hash mismatch: {name}")
        paths.append((name, path))
    if not paths:
        raise ValueError("ArtBench source lock has no shards")
    return paths


def read_source_rows(
    paths: list[tuple[str, Path]],
    expected_rows: int,
) -> Iterator[tuple[str, int, JsonDict]]:
    """按固定分片及行顺序，分批读取本地 Parquet。

    Args:
        paths: 已核验的分片锁路径。
        expected_rows: 所有分片的预期总行数。

    Raises:
        ImportError: 缺少 PyArrow。
        ValueError: 数据列或行数与固定来源约定不符。
        OSError: Parquet 分片无法读取。

    Results:
        Iterator[tuple[str, int, JsonDict]]: 分片名、分片内行号和原始记录。
    """
    from pyarrow.parquet import ParquetFile

    files = [(name, ParquetFile(path)) for name, path in paths]
    try:
        if sum(file.metadata.num_rows for _, file in files) != expected_rows:
            raise ValueError("ArtBench parquet row count mismatch")
        for name, file in files:
            if not set(SOURCE_FIELDS).issubset(file.schema_arrow.names):
                raise ValueError(f"ArtBench schema mismatch: {name}")
            index = 0
            for batch in file.iter_batches(batch_size=128, columns=SOURCE_FIELDS):
                for row in batch.to_pylist():
                    yield name, index, row
                    index += 1
    finally:
        for _, file in files:
            file.close()


def decode_image(
    contents: bytes,
) -> tuple[Image.Image, str, str]:
    """完整解码并校验单帧尺寸，记录源模式和格式。

    Args:
        contents: Parquet image.bytes 字段。

    Raises:
        OSError: Pillow 无法解码或字节损坏。
        ValueError: 尺寸或帧数异常。

    Results:
        tuple[Image.Image, str, str]: RGB 图像、源模式、源编码。
    """
    with Image.open(BytesIO(contents)) as image:
        if image.size != (IMAGE_SIZE, IMAGE_SIZE) or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Unexpected ArtBench image geometry: {image.size}")
        image.load()
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            background = Image.new("RGBA", image.size, "white")
            rgb = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
        else:
            rgb = image.convert("RGB")
        return rgb, image.mode, image.format or "unknown"


def visual_signature(
    image: Image.Image,
) -> tuple[int, np.ndarray]:
    """生成近重复分组用的 dHash 与低分辨率颜色参照。

    Args:
        image: 已校验的 256×256 RGB 图像。

    Results:
        tuple[int, np.ndarray]: 64 位 dHash 和 4×4 RGB 颜色数组。
    """
    grayscale = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    difference = np.packbits(grayscale[:, 1:] > grayscale[:, :-1])
    colors = np.asarray(image.resize((4, 4), Image.Resampling.BOX), dtype=np.int16)
    return int.from_bytes(difference.tobytes()), colors


def find_group(
    parents: list[int],
    index: int,
) -> int:
    """找到样本的组代表并压缩已访问的父索引。

    Args:
        parents: 由并查集维护的父索引。
        index: 样本行号。

    Results:
        int: 当前连通组代表行号。

    副作用:
        更新 parents 中访问过的索引。
    """
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def join_groups(
    parents: list[int],
    left: int,
    right: int,
) -> None:
    """合并两个重复候选所在的连通组。

    Args:
        parents: 并查集父索引。
        left: 第一个样本行号。
        right: 第二个样本行号。

    副作用:
        合并时把较小行号作为确定性组代表。
    """
    left, right = sorted((find_group(parents, left), find_group(parents, right)))
    parents[right] = left


def group_and_split(
    records: list[JsonDict],
    signatures: list[tuple[int, np.ndarray]],
    seed: int,
) -> JsonDict:
    """对精确图像、同一来源与近重复候选分组，再按组切分。

    Args:
        records: 按固定来源顺序排列的样本清单。
        signatures: 与清单一一对应的 dHash 和颜色摘要。
        seed: 用于组级切分的固定盐值。

    Results:
        JsonDict: 重复候选计数、组规模和各集合数量。

    副作用:
        写入各记录的 group_id 与 split。
    """
    parents = list(range(len(records)))
    seen_pixels = {}
    seen_urls = {}
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    near_candidates = near_merged = 0
    for index, (record, (dhash, colors)) in enumerate(zip(records, signatures, strict=True)):
        pixels = record["pixel_sha256"]
        join_groups(parents, index, seen_pixels.setdefault(pixels, index))
        url_key = canonical_url(record["source_url"])
        if url_key:
            join_groups(parents, index, seen_urls.setdefault(url_key, index))

        # Hamming ≤3 的两个 64 位哈希，至少有一个完整 16 位段相同。
        keys = [(offset, (dhash >> offset) & 0xFFFF) for offset in range(0, 64, 16)]
        candidates = {previous for key in keys for previous in bands[key]}
        for previous in candidates:
            if find_group(parents, previous) == find_group(parents, index):
                continue
            other_hash, other_colors = signatures[previous]
            if (dhash ^ other_hash).bit_count() > 3:
                continue
            near_candidates += 1
            if float(np.abs(colors - other_colors).mean()) <= 12:
                join_groups(parents, index, previous)
                near_merged += 1
        for key in keys:
            bands[key].append(index)

    groups = Counter()
    split_counts = Counter()
    for index, record in enumerate(records):
        root_id = records[find_group(parents, index)]["sample_id"]
        group_id = hashlib.sha256(root_id.encode()).hexdigest()
        bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        record["group_id"] = group_id
        record["split"] = split
        groups[group_id] += 1
        split_counts[split] += 1
    return {
        "split_seed": seed,
        "split_counts": dict(split_counts),
        "groups": len(groups),
        "groups_with_multiple_rows": sum(size > 1 for size in groups.values()),
        "largest_group": max(groups.values(), default=0),
        "exact_pixel_duplicate_rows": len(records) - len(seen_pixels),
        "same_url_duplicate_rows": sum(bool(canonical_url(r["source_url"])) for r in records)
        - len(seen_urls),
        "near_ungrouped_candidates_checked": near_candidates,
        "near_group_merges": near_merged,
        "near_method": "64-bit dHash Hamming <= 3 and 4x4 RGB mean absolute difference <= 12",
    }


def write_jsonl(
    path: Path,
    records: list[JsonDict],
) -> None:
    """按记录顺序生成 UTF-8 JSONL。

    Args:
        path: 输出路径，父目录已创建。
        records: 可序列化的逐行字典。

    Raises:
        OSError: 文件无法写入。

    副作用:
        创建或替换目标 JSONL。
    """
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_audit_sample(
    directory: Path,
    records: list[JsonDict],
    count: int,
    seed: int,
) -> list[str]:
    """从训练候选固定随机抽取样本，生成本地对照页、网格及人审空表。

    Args:
        directory: 含 images 的输出版本目录。
        records: 已分组、划分的全量清单。
        count: 抽取张数，训练候选不足时保留全部候选。
        seed: 仅用于抽样的独立随机种子。

    Raises:
        OSError: 本地图像无法读取或审计文件无法写入。

    Results:
        list[str]: 按抽样顺序排列的样本标识。

    副作用:
        在输出版本目录的 audit 子目录创建 HTML、JSONL 和接触表。
    """
    folder = directory / "audit"
    folder.mkdir()
    train = [r for r in records if r["split"] == "train"]
    sample = random.Random(seed).sample(train, min(count, len(train)))
    fields = ("sample_id", "image_path", "caption_raw", "caption", "caption_status", "artist")
    write_jsonl(
        folder / "review.jsonl",
        [{**{key: r[key] for key in fields}, "human_review": None} for r in sample],
    )
    html = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>ArtBench P1 审计</title>',
        "<style>body{font:16px system-ui;margin:2rem}main{display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}"
        "article{border:1px solid #ccc;padding:1rem}img{width:256px;height:256px}</style>",
        "<h1>ArtBench P1 图像与描述复核</h1>",
        "<p>以样本 ID 关联 review.jsonl。复核主体、颜色、数量、空间关系和场景，"
        "记录明确的技术性缺陷，并区分作品笔触与扫描失真。</p><main>",
    ]
    for record in sample:
        image_path = escape(record["image_path"], quote=True)
        sample_id = escape(record["sample_id"])
        raw = escape(record["caption_raw"])
        cleaned = escape(record["caption"])
        html.append(
            f'<article><h2>{sample_id}</h2><img loading="lazy" src="../{image_path}" '
            f'alt="{sample_id}"><p><b>原始</b> {raw}</p><p><b>训练</b> {cleaned}</p></article>'
        )
    (folder / "review.html").write_text("\n".join(html + ["</main></html>"]), encoding="utf-8")
    for start in range(0, len(sample), 16):
        grid = Image.new("RGB", (896, 976), "#eceef1")
        draw = ImageDraw.Draw(grid)
        for index, record in enumerate(sample[start : start + 16]):
            left = (index % 4) * 224
            top = (index // 4) * 244
            with Image.open(directory / record["image_path"]) as image:
                image.thumbnail((216, 216), Image.Resampling.LANCZOS)
                grid.paste(image, (left + 4, top + 4))
            draw.text((left + 6, top + 224), record["sample_id"], fill="#111827")
        grid.save(folder / f"contact_sheet_{start // 16 + 1:02d}.jpg", quality=90)
    return [r["sample_id"] for r in sample]


def prepare_artbench(
    config_path: Path,
    raw_dir: Path,
    output_dir: Path,
    count_tokens: Callable[[str], int],
    token_lock_sha256: str,
) -> JsonDict:
    """离线核验、解码、分组、清洗 Caption 并生成 ArtBench 版本产物。

    Args:
        config_path: 固定来源及准备参数的 YAML。
        raw_dir: 本地 Parquet 所在的来源版本目录。
        output_dir: 独立的处理版本目录；同身份重复运行执行只读校验。
        count_tokens: 已验证的 CLIP 内容 token 计数函数。
        token_lock_sha256: CLIP 组件锁的 SHA-256，用于绑定文本处理身份。

    Raises:
        ValueError: 来源、已有产物或锁定参数与预期不符。
        OSError: 本地数据或产物无法读写。
        ImportError: 缺少本地读取依赖。

    Results:
        JsonDict: 数据准备、图像及 Caption 的自动审计证据和正式门槛状态。

    副作用:
        生成独立版本的图像、清单、摘要和审计包；完整写入后再发布目录。
    """
    from minigen.data.artbench_caption import CAPTION_CLEANUP_VERSION

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shards = open_source_shards(config, raw_dir)
    identity = {
        "config_sha256": sha256_file(config_path),
        "token_lock_sha256": token_lock_sha256,
        "caption_policy": CAPTION_CLEANUP_VERSION,
        "schema_version": SCHEMA,
    }
    if output_dir.exists():
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        manifest = output_dir / "manifest.jsonl"
        if summary["identity"] != identity or sha256_file(manifest) != summary["manifest_sha256"]:
            raise ValueError("ArtBench preparation identity mismatch; use a new version directory")
        records = load_manifest(manifest, output_dir, verify_hashes=True)
        if len(records) != summary["manifest_rows"]:
            raise ValueError("ArtBench manifest row count changed")
        return summary

    seed = config.get("seed", 20261107)
    review_count = config.get("review_count", 240)
    if review_count <= 0:
        raise ValueError("ArtBench review_count must be positive")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    records: list[JsonDict] = []
    signatures = []
    errors: list[JsonDict] = []
    formats = Counter()
    modes = Counter()
    caption_statuses = Counter()
    caption_actions = Counter()
    token_lengths = []
    low_range_rows = 0
    raw_over_limit_rows = 0

    with TemporaryDirectory(prefix=".artbench-prep-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        for index, (shard, shard_row, row) in enumerate(
            read_source_rows(shards, config["expected_rows"])
        ):
            sample_id = f"ab-{index:05d}"
            raw_image = row["image"]
            contents = raw_image.get("bytes") if isinstance(raw_image, dict) else None
            try:
                if not isinstance(contents, bytes) or not contents:
                    raise ValueError("Missing embedded image bytes")
                image, mode, source_format = decode_image(contents)
            except (OSError, ValueError) as exc:
                errors.append(
                    {
                        "sample_id": sample_id,
                        "source_shard": shard,
                        "row": shard_row,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            formats[source_format] += 1
            modes[mode] += 1
            low_range_rows += all(high - low <= 2 for low, high in image.getextrema())
            pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            signature = visual_signature(image)

            # 保留可直接使用的源 RGB 字节，其他模式按相同尺寸写无损 RGB PNG。
            keep_source = mode == "RGB" and source_format in FORMATS
            suffix = FORMATS[source_format] if keep_source else ".png"
            relative = Path("images") / f"{index // 1000:03d}" / (sample_id + suffix)
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if keep_source:
                target.write_bytes(contents)
                encoded_sha256 = hashlib.sha256(contents).hexdigest()
            else:
                image.save(target, format="PNG")
                encoded_sha256 = sha256_file(target)

            raw_caption = row["caption"] or ""
            artist = row["artist"] or ""
            cleanup = clean_caption(raw_caption, artist, count_tokens=count_tokens)
            caption_statuses[cleanup.status] += 1
            caption_actions.update(cleanup.actions)
            raw_over_limit_rows += count_tokens(raw_caption) > 75
            if cleanup.budget_token_count is not None:
                token_lengths.append(cleanup.budget_token_count)
            records.append(
                {
                    "sample_id": sample_id,
                    "source_id": config["repo_id"],
                    "source_revision": config["revision"],
                    "source_shard": shard,
                    "source_row": shard_row,
                    "source_image_path": raw_image.get("path"),
                    "source_image_sha256": hashlib.sha256(contents).hexdigest(),
                    "source_url": row["url"] or "",
                    "artist": artist,
                    "caption_raw": raw_caption,
                    "caption_jp": row["caption_jp"],
                    "caption_cleaned": cleanup.text,
                    "caption_status": cleanup.status,
                    "caption_actions": list(cleanup.actions),
                    "caption_cleaned_tokens": cleanup.token_count,
                    "caption": cleanup.budget_text,
                    "caption_tokens": cleanup.budget_token_count,
                    "attributes": {},
                    "image_path": relative.as_posix(),
                    "sha256": encoded_sha256,
                    "pixel_sha256": pixel_sha256,
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE,
                    "mode": "RGB",
                    "schema_version": SCHEMA,
                }
            )
            signatures.append(signature)

        groups = group_and_split(records, signatures, seed)
        manifest = staged / "manifest.jsonl"
        write_jsonl(manifest, records)
        reviewed_ids = write_audit_sample(staged, records, review_count, seed)
        write_jsonl(staged / "decode_failures.jsonl", errors)
        missing_captions = sum(not r["caption"] for r in records)
        over_budget = sum((r["caption_tokens"] or 0) > 75 for r in records)
        tokens = sorted(token_lengths)
        manifest_sha256 = sha256_file(manifest)
        summary = {
            "identity": identity,
            "source_id": config["repo_id"],
            "source_revision": config["revision"],
            "dataset_version": config["version"],
            "image_size": IMAGE_SIZE,
            "source_fields": list(SOURCE_FIELDS),
            "raw_shards": config["files"],
            "source_rows": len(records) + len(errors),
            "manifest_rows": len(records),
            "manifest_sha256": manifest_sha256,
            "source_modes": dict(modes),
            "source_formats": dict(formats),
            "decode_failures": len(errors),
            "nearly_uniform_images": low_range_rows,
            "caption_statuses": dict(caption_statuses),
            "caption_actions": dict(caption_actions),
            "caption_raw_over_limit": raw_over_limit_rows,
            "caption_unresolved": missing_captions,
            "caption_over_budget": over_budget,
            "caption_tokens": {
                "p50": tokens[(len(tokens) - 1) // 2] if tokens else None,
                "max": max(tokens, default=None),
            },
            **groups,
            "audit_sample_size": len(reviewed_ids),
            "audit_sample_seed": seed,
            "audit_sample_split": "train",
            "audit_sample_ids": reviewed_ids,
            "automated_checks_passed": (
                len(records) == config["expected_rows"]
                and not errors
                and not missing_captions
                and not over_budget
                and set(groups["split_counts"]) == {"train", "validation", "test"}
            ),
            "human_reviewed": 0,
        }
        (staged / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged.rename(output_dir)
    return summary
