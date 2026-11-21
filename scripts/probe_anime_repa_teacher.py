# ruff: noqa: F722

"""用固定头像和局部对照测量冻结视觉编码器的空间响应。"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from cache_anime_repa_teacher import ASSET_LOCK, FrozenPatchTeacher
from jaxtyping import Float, UInt8
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from minigen.autoencoders.sdxl import hash_file

REGIONS = ("eye", "nose_mouth")


def elliptical_distance(
    size: int,
    center: Sequence[float],
    radius: Sequence[float],
) -> Float[np.ndarray, " h w"]:
    """计算像素或 patch 中心到固定椭圆中心的归一化距离平方。

    Args:
        size: 正方形网格边长。
        center: 原始 256 图像上的 x、y 坐标。
        radius: 原始图像上的 x、y 椭圆半径。

    Results:
        Float[np.ndarray, " h w"]: 行优先网格上的椭圆距离平方。
    """
    coordinates = (np.arange(size, dtype=np.float32) + 0.5) * (256 / size) - 0.5
    x = (coordinates - center[0]) / radius[0]
    y = (coordinates - center[1]) / radius[1]
    return x[None, :] ** 2 + y[:, None] ** 2


def variants(
    pixels: UInt8[np.ndarray, " h w c"],
    center: Sequence[float],
    radius: Sequence[float],
    offset: Sequence[float],
) -> tuple[list[Image.Image], float]:
    """生成同一区域的几何、重采样和像素变化量匹配的颜色对照。

    Args:
        pixels: 原始 256×256 RGB 像素。
        center: 椭圆中心的 x、y 坐标。
        radius: 椭圆横纵半径。
        offset: 椭圆中心峰值偏移的 x、y 分量。

    Results:
        tuple: 几何、重采样、颜色三图和颜色与几何的加权像素差之比。
    """
    distance = elliptical_distance(256, center, radius)
    mask = np.clip(1 - distance / 4, 0, 1) ** 2
    y, x = np.indices(distance.shape, dtype=np.float32)
    dx, dy = offset
    geometry = cv2.remap(pixels, x + dx * mask, y + dy * mask, cv2.INTER_LINEAR)
    sham = cv2.remap(
        pixels,
        x + math.copysign(0.5, dx) * mask,
        y + math.copysign(0.5, dy) * mask,
        cv2.INTER_LINEAR,
    )
    rgb = pixels.astype(np.float32)
    # 颜色对照改变同一窗口的像素值，同时保持原来的空间坐标。
    target_rgb = np.array([40, 210, 70], dtype=np.float32)
    delta = (target_rgb - rgb) * mask[..., None]
    geometric_energy = float((np.abs(geometry - rgb) * mask[..., None]).mean())
    strength = geometric_energy / float((np.abs(delta) * mask[..., None]).mean())
    color = np.clip(np.rint(rgb + strength * delta), 0, 255).astype(np.uint8)
    color_ratio = float((np.abs(color - rgb) * mask[..., None]).mean()) / geometric_energy
    return [Image.fromarray(value) for value in (geometry, sham, color)], color_ratio


def patch_response(
    features: Float[Tensor, " b h w d"],
    center: Sequence[float],
    radius: Sequence[float],
) -> dict[str, float]:
    """比较三个变体在目标区域和远处的 patch 变化。

    Args:
        features: 依次为原图、几何错位、重采样和等像素差颜色对照的特征。
        center: 椭圆中心的 x、y 坐标。
        radius: 椭圆横纵半径。

    Results:
        dict: 三组局部 1 减 cosine、几何远端响应和三个判别比值。
    """
    values = (1 - (features[1:] * features[0]).sum(-1)).clamp_min(0).numpy()
    distance = elliptical_distance(16, center, radius)
    region = distance <= 1.3**2
    exterior = distance >= 2.5**2
    geometry_roi, resample_roi, color_roi = values[:, region].mean(axis=-1)
    geometry_out = values[0, exterior].mean()
    return {
        "geometry_roi": float(geometry_roi),
        "resample_roi": float(resample_roi),
        "photometric_roi": float(color_roi),
        "geometry_exterior": float(geometry_out),
        "geometry_vs_resample_ratio": float(geometry_roi / max(resample_roi, 1e-8)),
        "roi_vs_exterior_ratio": float(geometry_roi / max(geometry_out, 1e-8)),
        "geometry_vs_photometric_ratio": float(geometry_roi / max(color_roi, 1e-8)),
    }


def render_example(
    destination: Path,
    examples: list[tuple[str, list[Image.Image]]],
) -> None:
    """输出前四个样本两类变形的原图和三个变体。

    Args:
        destination: PNG 输出路径。
        examples: 复现顺序中前四个样本的 ID 和七张图。

    副作用:
        保存接触表。
    """
    cell = 208
    header = 38
    canvas = Image.new("RGB", (7 * cell, header + len(examples) * (cell + 22)), "#f5f2ec")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    labels = (
        "Original",
        "Eye warp",
        "Eye resample",
        "Eye color",
        "Nose/mouth warp",
        "Nose/mouth resample",
        "Nose/mouth color",
    )
    for column, label in enumerate(labels):
        draw.text((column * cell + 6, 10), label, fill="#243443", font=font)
    for row, (sample_id, images) in enumerate(examples):
        y = header + row * (cell + 22)
        for column, image in enumerate(images):
            canvas.paste(image.resize((cell, cell), Image.Resampling.LANCZOS), (column * cell, y))
        draw.text((6, y + cell + 3), sample_id, fill="#243443", font=font)
    canvas.save(destination)


def main() -> None:
    """核对冻结数据，运行教师探针并输出逐项结果与资格判定。

    Raises:
        ValueError: 数据、教师或画像身份不合约，或检测到非有限特征。
        OSError: 输入或输出不可读写。

    副作用:
        运行本地教师前向并写出探针结果和对照图，向终端报告进度。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/evaluation/anime_repa_teacher_probe.json",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/exports/anime-faces-b1/repa_teacher_probe",
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    config = json.loads(args.config.read_text())
    for name in ("manifest", "benchmark", "benchmark_exclusions", "weights"):
        if hash_file(Path(config[name])) != config[f"{name}_sha256"]:
            raise ValueError(f"Probe {name} identity changed")
    manifest = {
        row["sample_id"]: row
        for line in Path(config["manifest"]).read_text().splitlines()
        if (row := json.loads(line))
    }
    bank = json.loads(Path(config["benchmark"]).read_text())
    exclusions = json.loads(Path(config["benchmark_exclusions"]).read_text())
    excluded_groups = {row["source"]["group_id"] for row in bank["regular"] + bank["pairs"]}
    excluded_groups.update(
        group for item in exclusions["groups"].values() for group in item["group_ids"]
    )
    items = config["items"]
    if len(items) != config["gate"]["sample_count"] or len(
        {item["group_id"] for item in items}
    ) != len(items):
        raise ValueError("Expected complete distinct-group selection")
    device = torch.device(args.device)
    if Path(config["weights"]) != ASSET_LOCK:
        raise ValueError("Probe must use the training teacher's asset lock")
    print(f"Loading locked CLIP on {device}", flush=True)
    teacher = FrozenPatchTeacher(Path.cwd(), device)
    scores = {region: [] for region in REGIONS}
    rows = []
    examples = []
    for index, item in enumerate(items):
        entry = manifest[item["sample_id"]]
        path = Path(config["image_root"]) / item["image_path"]
        if (
            entry["split"] != "validation"
            or entry["group_id"] in excluded_groups
            or entry["group_id"] != item["group_id"]
            or entry["image_path"] != item["image_path"]
            or entry["sha256"] != item["sha256"]
            or hash_file(path) != item["sha256"]
        ):
            raise ValueError(f"Probe source identity mismatch: {path}")
        with Image.open(path) as source:
            original = source.convert("RGB")
        pixels = np.array(original)
        images = [original]
        energy_ratios = {}
        for region in REGIONS:
            images_for_region, energy_ratio = variants(pixels, **item["regions"][region])
            images.extend(images_for_region)
            energy_ratios[region] = energy_ratio
        features = teacher.features(images).reshape(-1, 16, 16, 1024).cpu()
        results = {"sample_id": item["sample_id"]}
        for region, indices in zip(REGIONS, ((0, 1, 2, 3), (0, 4, 5, 6)), strict=True):
            region_config = item["regions"][region]
            values = patch_response(
                features[list(indices)],
                region_config["center"],
                region_config["radius"],
            )
            values["photometric_pixel_ratio"] = energy_ratios[region]
            results[region] = values
            scores[region].append(values)
        rows.append(results)
        if index < 4:
            examples.append((item["sample_id"], images))
        if (index + 1) % 4 == 0:
            print(f"Completed {index + 1}/{len(items)}", flush=True)
    gate = config["gate"]
    summary = {}
    low, high = gate["photometric_pixel_ratio_bounds"]
    thresholds = (
        "geometry_vs_resample_ratio",
        "roi_vs_exterior_ratio",
        "geometry_vs_photometric_ratio",
    )
    for region, values in scores.items():
        medians = {key: float(np.median([row[key] for row in values])) for key in values[0]}
        counts = {key: sum(row[key] >= gate[key] for row in values) for key in thresholds}
        counts["photometric_pixel_match"] = sum(
            low <= row["photometric_pixel_ratio"] <= high for row in values
        )
        minimum = gate["minimum_met_per_region"]
        summary[region] = {
            "medians": medians,
            "counts": counts,
            "structure_passed": all(counts[key] >= minimum for key in thresholds[:2]),
            "photometric_preference_passed": all(
                counts[key] >= minimum for key in (thresholds[2], "photometric_pixel_match")
            ),
            "passed": all(count >= minimum for count in counts.values()),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    render_example(args.output / "variants.png", examples)
    receipt = {
        "probe_config": args.config.as_posix(),
        "probe_config_sha256": hash_file(args.config),
        "device": str(device),
        "feature_dtype": "float32",
        "images": len(items) * (1 + 3 * len(REGIONS)),
        "optimizer_updates": 0,
        "sample_count": len(items),
        "gate": gate,
        "regions": summary,
        "structure_passed": all(value["structure_passed"] for value in summary.values()),
        "photometric_preference_passed": all(
            value["photometric_preference_passed"] for value in summary.values()
        ),
        "passed": all(value["passed"] for value in summary.values()),
        "rows": rows,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
