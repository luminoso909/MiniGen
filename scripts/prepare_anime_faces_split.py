"""冻结 B1 动漫头像的精确去重和近重复分组划分。"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from minigen._typing import JsonDict
from minigen.data.artbench import (
    find_group,
    join_groups,
    sha256_file,
    visual_signature,
    write_jsonl,
)
from minigen.data.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "anime_faces_256"
SPLIT_SCHEMA = "minigen.anime_faces.b1.split.v1"
SOURCE_ROWS = 37_250
UNIQUE_ROWS = 36_547
SPLIT_SEED = 20261113
HAIR_COLORS = frozenset(
    "black white gray silver brown red pink purple blue aqua teal cyan green yellow blonde orange "
    "amber gold hazel beige turquoise blue-gray gray-blue dark light".split()
)
HAIR_LENGTHS = frozenset(
    {"very long hair", "long hair", "medium hair", "short hair", "very short hair"}
)
EXPRESSIONS = frozenset(
    {"smile", "grin", "frown", "smirk", "pout", "surprised", "angry", "crying", "expressionless"}
)

type Signature = tuple[str, int, NDArray[np.int16]]
type Stratum = tuple[int, str, str]


def digest(
    text: str,
) -> str:
    """计算可复现的文本标识摘要。

    Args:
        text: 使用 UTF-8 编码的标识内容。

    Results:
        str: 十六进制 SHA-256。
    """
    return hashlib.sha256(text.encode()).hexdigest()


def hair_stratum(
    row: JsonDict,
) -> Stratum:
    """提取用于平衡划分的 Rank、发色和发长。

    Args:
        row: 含 Rank 和已清洗视觉标签的清单行。

    Results:
        Stratum: Rank、发色及发长。缺失或多标签时归入 other。
    """
    tags = set(row["cleaned_tags"])
    colors = {
        tag.split()[-2] for tag in tags if tag.endswith(" hair") and tag.split()[-2] in HAIR_COLORS
    }
    lengths = tags & HAIR_LENGTHS
    return (
        row["rank"],
        min(colors) if len(colors) == 1 else "other",
        min(lengths) if len(lengths) == 1 else "other",
    )


def caption_priority(
    row: JsonDict,
) -> tuple[int, int, int]:
    """为完全相同的图选择核心标签较完整的来源行。

    Args:
        row: 清洗后的源记录，含标签、舍弃理由和源行号。

    Results:
        tuple: 可排序的三元组，依次优先核心属性、少冲突和较早源行。
    """
    _, color, length = hair_stratum(row)
    tags = set(row["cleaned_tags"])
    core_count = sum((color != "other", length != "other", bool(tags & EXPRESSIONS)))
    conflicts = sum(reason.startswith("conflicting_") for reason in row["removed_tags"].values())
    return -core_count, conflicts, row["source_index"]


def read_signature(
    image_root: Path,
    row: JsonDict,
) -> Signature:
    """验证文件身份并计算像素、dHash 及低分辨率颜色摘要。

    Args:
        image_root: 源清洗目录，包含相对图片路径。
        row: 提供图像路径及 SHA-256 的去重后记录。

    Raises:
        ValueError: 图像越界、哈希错误或图片几何与约定不符。
        OSError: 图片文件无法读取或解码。

    Results:
        Signature: RGB 像素摘要、64 位 dHash 和 4×4 RGB 颜色数组。
    """
    path = (image_root / row["image_path"]).resolve()
    if not path.is_relative_to(image_root):
        raise ValueError(f"Image escapes data root: {row['sample_id']}")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != row["sha256"]:
        raise ValueError(f"Image hash mismatch: {row['sample_id']}")
    with Image.open(BytesIO(contents)) as image:
        if image.size != (256, 256) or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected one 256x256 RGB frame: {row['sample_id']}")
        pixels = hashlib.sha256(image.tobytes()).hexdigest()
        dhash, colors = visual_signature(image)
    return pixels, dhash, colors


def group_images(
    rows: list[JsonDict],
    signatures: list[Signature],
    aliases: list[set[str]],
) -> tuple[list[int], list[tuple[int, int]], int]:
    """合并原图、解码像素及满足双阈值的近重复头像。

    Args:
        rows: 按稳定源行顺序排列的去重清单。
        signatures: 与 rows 一一对齐的像素摘要和视觉签名。
        aliases: 每张保留图对应的全部原图字节摘要。

    Results:
        tuple: 并查集父索引、每对满足视觉阈值的边和 dHash 命中对数。

    副作用:
        为各行补充 pixel_sha256，供独立划分复验。
    """
    parents = list(range(len(rows)))
    seen_original: dict[str, int] = {}
    seen_pixels: dict[str, int] = {}
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    edges = []
    hash_matches = 0
    for index, (row, signature, sources) in enumerate(zip(rows, signatures, aliases, strict=True)):
        pixels, dhash, colors = signature
        row["pixel_sha256"] = pixels
        join_groups(parents, index, seen_pixels.setdefault(pixels, index))
        for original in sorted(sources):
            join_groups(parents, index, seen_original.setdefault(original, index))

        # 距离不超过 3 的两个 64 位哈希必然有一个完整 16 位段相同。
        keys = [(offset, (dhash >> offset) & 0xFFFF) for offset in range(0, 64, 16)]
        candidates = {other for key in keys for other in bands[key]}
        for other in sorted(candidates):
            _, other_hash, other_colors = signatures[other]
            if (dhash ^ other_hash).bit_count() <= 3:
                hash_matches += 1
                if float(np.abs(colors - other_colors).mean()) <= 12:
                    join_groups(parents, index, other)
                    edges.append((index, other))
        for key in keys:
            bands[key].append(index)
    return parents, edges, hash_matches


def choose_validation(
    rows: list[JsonDict],
    groups: dict[int, list[int]],
    seed: int,
) -> set[int]:
    """按组内主属性做分层配额，组级选取约 10% 的验证图。

    Args:
        rows: 去重后、尚未分割的全量记录。
        groups: 并查集代表到成员索引的映射。
        seed: 决定组级散列排序的固定种子。

    Results:
        set[int]: 验证侧的组代表索引。配额无法整组填充时保持组完整。
    """
    strata: dict[Stratum, list[int]] = defaultdict(list)
    for group, members in groups.items():
        labels = Counter(hair_stratum(rows[index]) for index in members)
        dominant = min(labels, key=lambda label: (-labels[label], label))
        strata[dominant].append(group)
    sizes = {
        label: sum(len(groups[group]) for group in members) for label, members in strata.items()
    }
    target = (len(rows) + 5) // 10
    quotas = {label: size // 10 for label, size in sizes.items()}
    remainder = target - sum(quotas.values())
    for label in sorted(sizes, key=lambda value: (-(sizes[value] % 10), value))[:remainder]:
        quotas[label] += 1

    order = {group: digest(f"{seed}:{rows[group]['group_id']}") for group in groups}
    validation: set[int] = set()
    assigned = 0
    for label, members in strata.items():
        remaining = quotas[label]
        for group in sorted(members, key=order.__getitem__):
            size = len(groups[group])
            if size <= remaining:
                validation.add(group)
                assigned += size
                remaining -= size
            if remaining == 0:
                break
    for group in sorted(groups.keys() - validation, key=order.__getitem__):
        size = len(groups[group])
        if size <= target - assigned:
            validation.add(group)
            assigned += size
        if assigned == target:
            break
    return validation


def prepare(
    source: Path,
    destination: Path,
    seed: int,
    workers: int,
) -> JsonDict:
    """验证清洗身份并发布动漫头像组级划分清单和回执。

    Args:
        source: 含清洗清单、回执和图片的目录。
        destination: 发布独立清单与划分回执的目录。
        seed: 组级切分盐值。
        workers: 图像签名读取的并发数。

    Raises:
        ValueError: 源清单不匹配、跨集重复或已有输出不一致。
        OSError: 清单和图像无法读取或输出无法写入。

    Results:
        JsonDict: 实际数量、重复组统计、校验和与回读口径。

    副作用:
        在目标目录发布清单和回执。相同产物可重复核验。
    """
    source = source.resolve()
    source_manifest = source / "manifest.jsonl"
    summary = json.loads((source / "summary.json").read_text())
    source_hash = sha256_file(source_manifest)
    if source_hash != summary["manifest_sha256"]:
        raise ValueError("Source cleaning manifest hash differs from its receipt")
    input_rows = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    if (
        len(input_rows) != SOURCE_ROWS
        or len({row["sample_id"] for row in input_rows}) != SOURCE_ROWS
    ):
        raise ValueError("Unexpected source row count or duplicate sample ID")

    by_hash: dict[str, list[JsonDict]] = defaultdict(list)
    for row in input_rows:
        if row["source_id"] != summary["source"] or row["source_revision"] != summary["revision"]:
            raise ValueError(f"Source identity differs from receipt: {row['sample_id']}")
        by_hash[row["sha256"]].append(row)
    if len(by_hash) != UNIQUE_ROWS:
        raise ValueError(f"Expected {UNIQUE_ROWS} unique images, found {len(by_hash)}")

    rows = sorted(
        (min(items, key=caption_priority).copy() for items in by_hash.values()),
        key=lambda row: row["source_index"],
    )
    aliases = [{original["original_sha256"] for original in by_hash[row["sha256"]]} for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        signatures = list(pool.map(lambda row: read_signature(source, row), rows))
    parents, edges, hash_matches = group_images(rows, signatures, aliases)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        representative = find_group(parents, index)
        groups[representative].append(index)
        row["group_id"] = digest(f"anime-faces-b1-v1:{rows[representative]['sample_id']}")
        row["mode"] = "RGB"
        row["attributes"] = {"tags": row["cleaned_tags"]}
        duplicates = sorted(original["sample_id"] for original in by_hash[row["sha256"]])
        if len(duplicates) > 1:
            row["exact_duplicate_sample_ids"] = duplicates
    validation = choose_validation(rows, groups, seed)
    split_counts: Counter[str] = Counter()
    rank_counts: dict[str, Counter[str]] = defaultdict(Counter)
    hair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_identities: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for index, row in enumerate(rows):
        split = "validation" if find_group(parents, index) in validation else "train"
        row["split"] = split
        split_counts[split] += 1
        rank, hair_color, _ = hair_stratum(row)
        rank_counts[split][str(rank)] += 1
        hair_counts[split][hair_color] += 1
        for field in ("group_id", "sha256", "pixel_sha256"):
            split_identities[field][split].add(row[field])
        split_identities["original_sha256"][split].update(aliases[index])
    overlaps = {
        field: len(keys["train"] & keys["validation"]) for field, keys in split_identities.items()
    }
    overlaps["near_pairs"] = sum(
        rows[left]["split"] != rows[right]["split"] for left, right in edges
    )
    if any(overlaps.values()) or not all(split_counts[split] for split in ("train", "validation")):
        raise ValueError(f"Unexpected empty split or cross-split image overlap: {overlaps}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".anime-split-") as temporary:
        staged = Path(temporary) / "split"
        staged.mkdir()
        manifest = staged / "manifest.jsonl"
        write_jsonl(manifest, rows)
        receipt: JsonDict = {
            "schema": SPLIT_SCHEMA,
            "source_id": summary["source"],
            "source_revision": summary["revision"],
            "source_manifest_sha256": source_hash,
            "manifest_sha256": sha256_file(manifest),
            "source_rows": len(input_rows),
            "deduplicated_rows": len(rows),
            "exact_duplicates_removed": len(input_rows) - len(rows),
            "split_seed": seed,
            "split_counts": dict(split_counts),
            "rank_counts": {split: dict(counts) for split, counts in rank_counts.items()},
            "hair_color_counts": {split: dict(counts) for split, counts in hair_counts.items()},
            "validation_target": (len(rows) + 5) // 10,
            "groups": len(groups),
            "groups_with_multiple_rows": sum(len(members) > 1 for members in groups.values()),
            "largest_group": max(map(len, groups.values())),
            "dhash_matches": hash_matches,
            "near_pairs": len(edges),
            "near_method": "64-bit dHash Hamming <= 3 and 4x4 RGB mean absolute difference <= 12",
            "cross_split_overlap": overlaps,
        }
        receipt_path = staged / "split_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        reread = load_manifest(manifest, source, verify_hashes=False)
        if reread != rows:
            raise ValueError("Split manifest does not match in-memory records after readback")
        if destination.exists():
            for name in ("manifest.jsonl", "split_receipt.json"):
                if sha256_file(destination / name) != sha256_file(staged / name):
                    raise ValueError(f"Existing split differs from regenerated output: {name}")
        else:
            staged.rename(destination)
    return receipt


def main() -> int:
    """按固定种子生成并核验 B1 动漫头像分组划分。

    Raises:
        SystemExit: 参数非法或请求帮助。
        ValueError/OSError: 数据或产物校验失败时透传。

    Results:
        int: 划分与回读通过返回 0。

    副作用:
        生成独立清单及回执并打印划分结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DATA_ROOT / "color_front_v1")
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "b1_v1")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    receipt = prepare(args.source_dir, args.output_dir, args.seed, args.workers)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
