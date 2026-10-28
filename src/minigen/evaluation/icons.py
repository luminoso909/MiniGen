"""规则图标的重建诊断；尚非正式 S3 生成评测器。"""

import itertools
from collections.abc import Mapping

import cv2
import numpy as np
from PIL import Image

from minigen._typing import JsonDict
from minigen.data.icons import PALETTE


def parse_icons(image: Image.Image) -> list[JsonDict]:
    """基于颜色距离、连通区域与填充比例解析规则几何图标。

    1 输入：
        image：Pillow 图像，内部转 RGB；只适用于本项目受控调色板。

    2 输出：
        预测对象列表，含 color、shape、center、bbox、fill、area；按位置排序，小区域被滤除。

    3 报错（Raises）：
        无显式 raise；图像转换或 OpenCV 连通区域运算异常透传。
    """
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
    # key: Callable[[JsonDict], tuple[float, float]]；对象中心按纵坐标、横坐标排序。
    return sorted(objects, key=lambda o: (o["center"][1], o["center"][0]))


def score_image(image: Image.Image, record: JsonDict) -> dict[str, int]:
    """先按空间位置匹配对象，再核对颜色、形状、数量和关系。

    1 输入：
        image：待评测的原图或重建 RGB 图。
        record：真实场景标注，含 attributes.objects 与 relations。

    2 输出：
        整数命中数、各分母及整图是否成功；数量不符时对象与关系均按失败计。

    3 报错（Raises）：
        KeyError/IndexError：真实标注不完整或关系索引无效时透传；解析器异常透传。
    """
    truth = record["attributes"]["objects"]
    predicted = parse_icons(image)
    equal_count = len(predicted) == len(truth)
    if equal_count:
        # key: Callable[[tuple[JsonDict, ...]], float]；仅按中心配对，避免按正确属性作弊。
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


def aggregate_scores(scores: list[dict[str, int]]) -> dict[str, float | int]:
    """用全部样本的命中和分母汇总属性指标。

    1 输入：
        scores：非空逐图评分列表；失败样本不能删除。

    2 输出：
        图像数、各类准确率、解析失败率、对象与关系分母；关系空时分母至少取 1。

    3 报错（Raises）：
        IndexError：scores 为空；KeyError：逐图字段不完整时透传。
    """
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


def gate_passes(metrics: Mapping[str, float | int]) -> bool:
    """检查预先固定的表示保留门槛，不随结果调整。

    1 输入：
        metrics：汇总指标，颜色/形状需 ≥98%，数量/关系/全部属性需 ≥95%。

    2 输出：
        全部达到门槛为 True，否则 False；不额外检查样本数。

    3 报错（Raises）：
        KeyError：所需指标缺失时透传。
    """
    return (
        metrics["color_accuracy"] >= 0.98
        and metrics["shape_accuracy"] >= 0.98
        and metrics["count_accuracy"] >= 0.95
        and metrics["relation_accuracy"] >= 0.95
        and metrics["all_accuracy"] >= 0.95
    )
