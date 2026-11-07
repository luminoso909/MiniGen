"""确定性几何图标、结构化描述、分组划分与逐图清单审计。"""

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from minigen._typing import JsonDict, JsonValue
from minigen.data.manifest import load_manifest as load_manifest

PALETTE = {
    "red": (220, 45, 45),
    "blue": (40, 90, 220),
    "green": (30, 165, 80),
    "yellow": (235, 190, 25),
}
SHAPES = ("circle", "square", "triangle")
HELD = {("red", "triangle"), ("blue", "square")}


def digest(
    value: JsonValue,
) -> str:
    """对 JSON 兼容内容计算与键顺序无关的稳定 SHA-256。

    Args:
        value: 配置、场景属性等可 JSON 序列化对象。

    Raises:
        TypeError/ValueError: 输入不能 JSON 序列化时透传。

    Results:
        str: 十六进制摘要，用于冻结配置和场景组身份。
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def caption(
    objects: list[JsonDict],
    relations: list[JsonDict],
    held_template: bool = False,
    template: int = 0,
) -> str:
    """从真实对象和关系构造英文描述，不引入猜测标签。

    Args:
        objects: 顺序对象列表，至少含 color 与 shape；同图组合唯一以便指代。
        relations: 含 subject、predicate、object 的关系索引列表。
        held_template: 是否使用仅属于模板外集合的表达。
        template: 训练模板选择值，0 与非 0 对应两种固定措辞。

    Raises:
        KeyError/IndexError: 对象字段缺失或关系索引越界时透传。

    Results:
        str: 与结构化场景一致的英文描述。
    """
    nouns = [f"a {o['color']} {o['shape']}" for o in objects]
    names = [f"the {o['color']} {o['shape']}" for o in objects]
    clauses = [f"{names[r['subject']]} is {r['predicate']} {names[r['object']]}" for r in relations]
    contents = ", ".join(nouns)
    layout = "; ".join(clauses)
    if held_template:
        text = f"White canvas showing {contents}."
        return text + (f" Layout: {layout}." if layout else "")
    if template == 0:
        text = f"On a white background: {contents}."
    else:
        text = f"An icon with {contents} on white."
    return text + (f" Positions: {layout}." if layout else "")


def relations_for(
    objects: list[JsonDict],
) -> list[JsonDict]:
    """依据对象中心推导明确的左右和上下关系。

    Args:
        objects: 带 center=[x,y] 的有序对象列表；轴差至少 24 像素才记录关系。

    Raises:
        KeyError/TypeError: 中心坐标缺失或无法比较时透传。

    Results:
        list[JsonDict]: 关系字典列表，subject/object 保持原对象索引；对角布局可同时产生两类关系。
    """
    result = []
    for i, j in itertools.combinations(range(len(objects)), 2):
        a, b = objects[i]["center"], objects[j]["center"]
        if abs(a[0] - b[0]) >= 24:
            left, right = (i, j) if a[0] < b[0] else (j, i)
            result.append({"subject": left, "predicate": "left of", "object": right})
        if abs(a[1] - b[1]) >= 24:
            upper, lower = (i, j) if a[1] < b[1] else (j, i)
            result.append({"subject": upper, "predicate": "above", "object": lower})
    return result


def render(
    objects: list[JsonDict],
    image_size: int = 128,
) -> Image.Image:
    """把已定义场景渲染成白底 RGB 几何图标。

    Args:
        objects: 含 bbox、color、shape 的对象；bbox 为包含边界的像素坐标。
        image_size: 正方形画布边长，默认 128。

    Raises:
        ValueError: 未知 shape 主动抛出；KeyError: 未知颜色或必要字段缺失；Pillow 绘制异常透传。

    Results:
        Image.Image: Pillow RGB 图像；不自动保存文件。
    """
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox"]
        color = PALETTE[obj["color"]]
        if obj["shape"] == "circle":
            draw.ellipse((x0, y0, x1, y1), fill=color)
        elif obj["shape"] == "square":
            draw.rectangle((x0, y0, x1, y1), fill=color)
        elif obj["shape"] == "triangle":
            draw.polygon(((x0 + x1) // 2, y0, x0, y1, x1, y1), fill=color)
        else:
            raise ValueError("Unknown shape")
    return image


def bbox_gap(
    a: list[int],
    b: list[int],
) -> float:
    """计算两个含边界像素外接框之间的最小间隔。

    Args:
        a: [x0,y0,x1,y1] 第一个框。
        b: 同格式的第二个框；边界间减 1 以按空隙像素计数。

    Raises:
        IndexError/TypeError: 框缺少坐标或坐标不能做数值运算时透传。

    Results:
        float: 非负欧氏间距；接触或重叠时为 0。
    """
    dx = max(a[0] - b[2] - 1, b[0] - a[2] - 1, 0)
    dy = max(a[1] - b[3] - 1, b[1] - a[3] - 1, 0)
    return math.hypot(dx, dy)


def _scene(
    rng: np.random.Generator,
    count: int,
    config: JsonDict,
) -> tuple[list[JsonDict], str, str] | None:
    """采样对象属性和离散布局，先分组与分配集合，再进行后续渲染。

    Args:
        rng: 独立 NumPy 随机发生器。
        count: 本图对象数，当前生成配置为 1–3。
        config: 尺寸、画布和间距规则，决定候选能否保留。

    Raises:
        ValueError: 采样数量超出候选池时由 NumPy 透传；KeyError: 配置缺项。

    Results:
        tuple[list[JsonDict], str, str] | None: 有效时返回 (objects,group_id,split)；
            越界或间距不足返回 None。

    副作用:
        推进调用方传入的独立 NumPy 随机发生器。
    """
    pairs = [(c, s) for c in PALETTE for s in SHAPES]
    selected = rng.choice(len(pairs), size=count, replace=False)
    cells = sorted(rng.choice(9, size=count, replace=False).tolist())
    objects = []
    for index, cell in zip(selected, cells, strict=True):
        color, shape = pairs[index]
        size = int(rng.integers(config["min_size"], config["max_size"] + 1))
        cx = [20, 64, 108][cell % 3] + int(rng.integers(-2, 3))
        cy = [20, 64, 108][cell // 3] + int(rng.integers(-2, 3))
        x0, y0 = cx - size // 2, cy - size // 2
        bbox = [x0, y0, x0 + size - 1, y0 + size - 1]
        if min(bbox) < 0 or max(bbox) >= config["image_size"]:
            return None
        if any(bbox_gap(bbox, o["bbox"]) < config["min_gap"] for o in objects):
            return None
        objects.append(
            {
                "color": color,
                "shape": shape,
                "cell": cell,
                "bbox": bbox,
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "size": size,
            }
        )
    # 组身份不含尺寸和微小位移，让同一粗场景的近似变体留在同一集合。
    group = digest([(o["color"], o["shape"], o["cell"]) for o in objects])
    combo = any((o["color"], o["shape"]) in HELD for o in objects)
    if combo:
        split = "test_combo"
    else:
        bucket = int(group[:8], 16) % 100
        split = (
            "train"
            if bucket < 70
            else ("validation" if bucket < 80 else "test_id" if bucket < 90 else "test_template")
        )
    return objects, group, split


def build_icons(
    config: JsonDict,
    data_root: str | Path,
) -> Path:
    """显式生成确定性图标与清单；已冻结版本先校验而不覆盖。

    Args:
        config: 包含 dataset_id、generator_revision、种子、尺寸和各集合配额的 JSON 兼容配置。
        data_root: 输出根目录，图片写 processed，清单写 manifests。

    Raises:
        ValueError: 同版本配置变化、已有未完成目录或已冻结文件校验失败。
        RuntimeError: 超过候选尝试上限仍无法满足配额；OSError/配置错误透传。

    Results:
        Path: 真实 JSONL 清单路径；冻结元数据同时记录配置指纹与尝试数量。

    副作用:
        首次生成时写图片、JSONL 和冻结元数据；已有同版本只校验，不覆盖。
        生成过程使用独立 RNG，不推进 NumPy 全局 RNG。
    """
    root = Path(data_root)
    folder = root / "processed" / config["dataset_id"]
    manifest = root / "manifests" / f"{config['dataset_id']}.jsonl"
    info_path = folder / "dataset.json"
    identity = digest(config)
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info["config_sha256"] != identity:
            raise ValueError("Frozen data config mismatch; use a new dataset_id")
        load_manifest(manifest, root, verify_hashes=True)
        return manifest
    if folder.exists():
        raise ValueError(
            "Incomplete data directory; inspect it before explicit removal or new version"
        )
    folder.mkdir(parents=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    quotas = config["split_counts"]
    counts = Counter()
    strata = Counter()
    rng = np.random.default_rng(config["seed"])
    hashes = set()
    records = []
    attempts = 0
    while sum(counts.values()) < sum(quotas.values()):
        attempts += 1
        if attempts > 300 * sum(quotas.values()):
            raise RuntimeError("Unable to fill grouped quotas without duplicates")
        # Balanced object counts within each split, including rare single-object scenes.
        count = 1 + (attempts % 3)
        candidate = _scene(rng, count, config)
        if candidate is None:
            continue
        objects, group, split = candidate
        if split not in quotas or counts[split] >= quotas[split]:
            continue
        if strata[split, count] >= (quotas[split] // 3 + (count <= quotas[split] % 3)):
            continue
        image = render(objects, config["image_size"])
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash in hashes:
            continue
        hashes.add(pixel_hash)
        sample_id = f"{config['dataset_id']}-{split}-{counts[split]:05d}"
        path = folder / split / f"{sample_id}.png"
        path.parent.mkdir(exist_ok=True)
        image.save(path)
        relations = relations_for(objects)
        template_id = "held-0" if split == "test_template" else f"train-{counts[split] % 2}"
        records.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "image_path": path.relative_to(root).as_posix(),
                "source_id": "synthetic_icons_v1",
                "source_revision": config["generator_revision"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pixel_sha256": pixel_hash,
                "width": image.width,
                "height": image.height,
                "mode": "RGB",
                "caption": caption(objects, relations, split == "test_template", counts[split] % 2),
                "caption_origin": "structured_scene_attributes",
                "template_id": template_id,
                "attributes": {"objects": objects, "count": count, "relations": relations},
                "group_id": group,
                "split": split,
                "attribution_ref": "project_local_renderer",
            }
        )
        counts[split] += 1
        strata[split, count] += 1
    temporary = manifest.with_suffix(".partial")
    temporary.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    temporary.replace(manifest)
    info_path.write_text(
        json.dumps(
            {"config": config, "config_sha256": identity, "attempts": attempts, "counts": counts},
            indent=2,
        )
    )
    return manifest


def audit_icons(
    records: list[JsonDict],
    data_root: str | Path,
) -> JsonDict:
    """审查像素重复、场景分组、组合/模板隔离及图像格式。

    Args:
        records: 已加载的真实样本清单；通常为非空的完整冻结版本。
        data_root: 定位图像及生成采样元数据的根目录。

    Raises:
        ValueError: 完全重复、跨集合场景组、组合/模板泄漏、描述不符或图像格式不符。
        OSError/KeyError: 图像损坏或记录字段缺失时透传；空记录集的统计错误不另行拦截。

    Results:
        JsonDict: JSON 质量摘要: 数量、分布、最小尺寸/间距、采样拒绝总数和有边界的缩略图近似候选。
    """
    root = Path(data_root)
    hashes, groups = {}, {}
    thumbnails = defaultdict(list)
    counts = Counter()
    distribution = defaultdict(Counter)
    sizes, gaps = [], []
    for r in records:
        split = r["split"]
        counts[split] += 1
        for field, store in [("pixel_sha256", hashes), ("group_id", groups)]:
            key = r[field]
            if key in store and (field == "pixel_sha256" or store[key] != split):
                raise ValueError(f"Cross-split group or duplicate: {field}")
            store[key] = split
        attrs = r["attributes"]
        objs = attrs["objects"]
        seen = {(o["color"], o["shape"]) for o in objs}
        if len(seen) != len(objs):
            raise ValueError("Ambiguous object reference")
        has_held = bool(seen & HELD)
        if has_held != (split == "test_combo"):
            raise ValueError("Composition leakage")
        if (r["template_id"].startswith("held")) != (split == "test_template"):
            raise ValueError("Template leakage")
        if r["caption"] != caption(
            objs, attrs["relations"], split == "test_template", int(r["template_id"][-1])
        ):
            raise ValueError("Caption/scene mismatch")
        distribution[split][f"count_{len(objs)}"] += 1
        for o in objs:
            sizes.append(o["size"])
            distribution[split][o["color"]] += 1
            distribution[split][o["shape"]] += 1
        gaps.extend(bbox_gap(a["bbox"], b["bbox"]) for a, b in itertools.combinations(objs, 2))
        with Image.open(root / r["image_path"]) as image:
            image.load()
            if image.size != (128, 128) or image.mode != "RGB":
                raise ValueError("Image format mismatch")
            thumb = np.asarray(image.resize((16, 16), Image.Resampling.BOX)) // 32
        key = hashlib.sha256(thumb.tobytes()).hexdigest()
        thumbnails[key].append((split, r["sample_id"], r["group_id"]))
    cross = [entries for entries in thumbnails.values() if len({e[0] for e in entries}) > 1]
    sampling = None
    if records:
        metadata = (root / records[0]["image_path"]).parents[1] / "dataset.json"
        if metadata.exists():
            info = json.loads(metadata.read_text())
            accepted = sum(info["counts"].values())
            sampling = {
                "candidates": info["attempts"],
                "accepted": accepted,
                "rejected": info["attempts"] - accepted,
                "rejection_reasons": [
                    "out_of_bounds_or_insufficient_gap",
                    "split_quota_filled",
                    "object_count_quota_filled",
                    "duplicate_pixels",
                ],
                "per_reason_counts": "not_instrumented",
            }
    return {
        "counts": dict(counts),
        "generation_sampling": sampling,
        "unique_pixels": len(hashes),
        "groups": len(groups),
        "distribution": {k: dict(v) for k, v in distribution.items()},
        "min_object_size": min(sizes),
        "min_bbox_gap": min(gaps) if gaps else None,
        "cross_split_thumbnail_candidates": cross[:30],
        "cross_split_thumbnail_candidate_groups": len(cross),
        "near_duplicate_method": "16x16 RGB BOX thumbnail, 3-bit/channel bucket equality",
        "near_duplicate_limit": "Candidate screen only; not an exhaustive perceptual guarantee",
        "total_png_bytes": sum((root / r["image_path"]).stat().st_size for r in records),
    }


def select_validation(
    records: list[JsonDict],
    count: int = 96,
) -> list[JsonDict]:
    """固定抽取按对象数量分层的小对象和窄间距验证压力子集。

    Args:
        records: 完整清单，只从 split=validation 的记录选择，不读测试图调参。
        count: 总抽样数，当前使用 96，需能按 1/2/3 对象等分。

    Raises:
        ValueError: 验证分层数量不足或 count 无法形成要求的等分。

    Results:
        list[JsonDict]: 三类对象数量各 count/3 的记录，按小尺寸、窄间距、ID 稳定排序。
    """
    chosen = []
    for n in (1, 2, 3):
        candidates = [
            r for r in records if r["split"] == "validation" and r["attributes"]["count"] == n
        ]
        # key: Callable[[JsonDict], tuple[int, float, str]]；优先小尺寸和窄间距。
        candidates.sort(
            key=lambda r: (
                min(o["size"] for o in r["attributes"]["objects"]),
                min(
                    (
                        bbox_gap(a["bbox"], b["bbox"])
                        for a, b in itertools.combinations(r["attributes"]["objects"], 2)
                    ),
                    default=0,
                ),
                r["sample_id"],
            )
        )
        chosen.extend(candidates[: count // 3])
    if len(chosen) != count:
        raise ValueError("Insufficient validation strata")
    return chosen
