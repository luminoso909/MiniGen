"""Deterministic, grouped geometric scenes and audited JSONL manifests."""

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PALETTE = {
    "red": (220, 45, 45),
    "blue": (40, 90, 220),
    "green": (30, 165, 80),
    "yellow": (235, 190, 25),
}
SHAPES = ("circle", "square", "triangle")
HELD = {("red", "triangle"), ("blue", "square")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def caption(objects, relations, held_template=False, template=0):
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


def relations_for(objects):
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


def render(objects, image_size=128):
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


def bbox_gap(a, b):
    dx = max(a[0] - b[2] - 1, b[0] - a[2] - 1, 0)
    dy = max(a[1] - b[3] - 1, b[1] - a[3] - 1, 0)
    return math.hypot(dx, dy)


def _scene(rng, count, config):
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


def load_manifest(path, data_root, verify_hashes=False):
    root = Path(data_root).resolve()
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    ids = set()
    for record in records:
        if record.get("example_only") or not record.get("sha256"):
            raise ValueError("Example or missing image hash")
        required = {
            "sample_id",
            "image_path",
            "group_id",
            "split",
            "caption",
            "attributes",
            "source_id",
            "source_revision",
            "width",
            "height",
            "mode",
        }
        if not required.issubset(record):
            raise ValueError("Missing manifest field")
        path = (root / record["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes data root")
        if record["sample_id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(record["sample_id"])
        if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Image hash mismatch")
    return records


def build_icons(config, data_root):
    """Explicit generation only; frozen existing versions are verified, never overwritten."""
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


def audit_icons(records, data_root):
    """Full hash/group checks and a bounded color-aware thumbnail near-duplicate screen."""
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


def select_validation(records, count=96):
    """Deterministic count strata, small sizes and narrow gaps; no test images."""
    chosen = []
    for n in (1, 2, 3):
        candidates = [
            r for r in records if r["split"] == "validation" and r["attributes"]["count"] == n
        ]
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
