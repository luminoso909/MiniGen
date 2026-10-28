"""Rule-based reconstruction diagnostics; not yet an S3 generation evaluator."""

import itertools

import cv2
import numpy as np

from minigen.data.icons import PALETTE


def parse_icons(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    names = list(PALETTE)
    colors = np.asarray(list(PALETTE.values()), dtype=np.float32)
    distances = ((pixels[:, :, None, :] - colors) ** 2).sum(-1)
    closest = distances.argmin(-1)
    valid = (distances.min(-1) < 100**2) & (pixels.max(-1) - pixels.min(-1) > 50)
    objects = []
    for i, name in enumerate(names):
        mask = ((closest == i) & valid).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for j in range(1, count):
            x, y, width, height, area = stats[j]
            if area < 80:
                continue
            fill = area / (width * height)
            shape = "square" if fill >= 0.88 else "triangle" if fill <= 0.62 else "circle"
            objects.append(
                {
                    "color": name,
                    "shape": shape,
                    "center": [float(x + (width - 1) / 2), float(y + (height - 1) / 2)],
                    "bbox": [int(x), int(y), int(x + width - 1), int(y + height - 1)],
                    "fill": float(fill),
                    "area": int((labels == j).sum()),
                }
            )
    return sorted(objects, key=lambda o: (o["center"][1], o["center"][0]))


def score_image(image, record):
    truth = record["attributes"]["objects"]
    predicted = parse_icons(image)
    equal_count = len(predicted) == len(truth)
    if equal_count:
        order = min(
            itertools.permutations(predicted),
            key=lambda permutation: sum(
                sum((a - b) ** 2 for a, b in zip(t["center"], p["center"], strict=True))
                for t, p in zip(truth, permutation, strict=True)
            ),
        )
        colors = sum(t["color"] == p["color"] for t, p in zip(truth, order, strict=True))
        shapes = sum(t["shape"] == p["shape"] for t, p in zip(truth, order, strict=True))
    else:
        order, colors, shapes = [], 0, 0
    relations = record["attributes"]["relations"]
    relation_hits = 0
    if equal_count:
        for r in relations:
            a, b = order[r["subject"]]["center"], order[r["object"]]["center"]
            relation_hits += (a[0] < b[0]) if r["predicate"] == "left of" else (a[1] < b[1])
    all_correct = equal_count and colors == len(truth) and shapes == len(truth)
    all_correct = all_correct and relation_hits == len(relations)
    return {
        "count_correct": int(equal_count),
        "color_hits": colors,
        "shape_hits": shapes,
        "object_total": len(truth),
        "relation_hits": int(relation_hits),
        "relation_total": len(relations),
        "all_correct": int(all_correct),
        "parse_failure": int(not predicted or len(predicted) > 3),
    }


def aggregate_scores(scores):
    total = {k: sum(s[k] for s in scores) for k in scores[0]}
    n = len(scores)
    return {
        "images": n,
        "count_accuracy": total["count_correct"] / n,
        "color_accuracy": total["color_hits"] / total["object_total"],
        "shape_accuracy": total["shape_hits"] / total["object_total"],
        "relation_accuracy": total["relation_hits"] / max(1, total["relation_total"]),
        "all_accuracy": total["all_correct"] / n,
        "parse_failure_rate": total["parse_failure"] / n,
        "object_denominator": total["object_total"],
        "relation_denominator": total["relation_total"],
    }


def gate_passes(metrics):
    return (
        metrics["color_accuracy"] >= 0.98
        and metrics["shape_accuracy"] >= 0.98
        and metrics["count_accuracy"] >= 0.95
        and metrics["relation_accuracy"] >= 0.95
        and metrics["all_accuracy"] >= 0.95
    )
