# ruff: noqa: F722

"""在冻结训练 RGB 上定位动漫五官并缓存 B2 区域加权 Flow 的软掩码。"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import batched
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from cache_anime_repa_teacher import fetch_file, iter_pixels, training_rows, verify_file
from jaxtyping import Float
from PIL import Image, ImageDraw

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/cache/anime_faces_256/b2_face_weights_v1")
PREVIEW = Path("artifacts/exports/anime-faces-b1/b2_face_weights_preview")
DETECTOR_REPO = "hysts/anime-face-detector-hrnetv2"
DETECTOR_REVISION = "9b3435248b26aeb82e2a8578fe9d86d5d57158af"
WEIGHTS = {
    "name": "model.safetensors",
    "bytes": 38917560,
    "sha256": "e71271376406a743c01528a0460637fcc06e72aeeea583f85007cc72dc8b7a4a",
}
WHEEL = {
    "name": "anime_face_detector-0.1.0-py3-none-any.whl",
    "bytes": 68626,
    "sha256": "9a6a8c1384b7a57fab8ce9988f814271ff88bac52a9dd871490a28b61dff7692",
}
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ff/42/"
    "a164a2431cff2acc0cace7448d7c61702ecab49a8ed44c7fdd3b704de3f3/"
    f"{WHEEL['name']}"
)
DETECTOR = {
    "repo_id": DETECTOR_REPO,
    "revision": DETECTOR_REVISION,
    "weights": WEIGHTS,
    "wheel": WHEEL,
}
DETECTOR_DIR = Path("artifacts/models/anime-face-detector-hrnetv2") / DETECTOR_REVISION
GRID_SIZE = 64
LATENT_SIZE = 32
GRID_Y, GRID_X = np.indices((GRID_SIZE, GRID_SIZE), dtype=np.float32)
REGIONS = ("face", "left_eye", "right_eye", "nose", "mouth")
RECIPE = {
    "name": "landmark_soft_regions_v1",
    "pixel_input": "frozen final RGB256, no geometric augmentation",
    "weights_before_normalization": {"background": 1, "face": 2, "eyes_nose_mouth": 3},
    "face_outline": "soft expanded hull of jaw, brow, features and estimated forehead",
    "feature_mask": "oriented ellipses with smoothstep transition",
    "face_confidence": {"mean_jaw": 0.30, "chin": 0.25, "stronger_eye": 0.38},
    "face_geometry_heatmap": {"eye_distance": [5, 46], "chin_drop_relative_to_eyes": [0.28, 2.1]},
    "feature_confidence": {"eye": 0.32, "nose": 0.25, "mouth": 0.30},
    "fallback": "ignore uncertain feature; use uniform weights if face localization is unreliable",
    "resize": "64x64 HRNet coordinates to 32x32 latent via 2x2 area mean",
    "normalization": "per-sample mean 1 before float16 storage",
}


class FaceLandmarks:
    """在固定设备上运行原清洗使用的 HRNet 热图前向。"""

    def __init__(
        self,
        root: Path,
        device: torch.device,
    ) -> None:
        """校验模型资产并准备整张头像的 RGB 归一化。

        Args:
            root: 已有固定版本公开资产的项目根。
            device: HRNet 的计算设备。

        Raises:
            OSError/ValueError: 权重或推理 wheel 文件身份不一致。
            ImportError/RuntimeError: 无法加载固定实现或模型。

        副作用:
            将校验过的 wheel 放入导入路径，并在设备分配模型。
        """
        folder = root / DETECTOR_DIR
        for asset in (WEIGHTS, WHEEL):
            verify_file(folder / asset["name"], asset["bytes"], asset["sha256"])
        sys.path.insert(0, str(folder / WHEEL["name"]))
        from anime_face_detector._landmark import load_landmark_model

        self.device = device
        self.model = load_landmark_model(str(folder / WEIGHTS["name"]), device=str(device))
        self.model = self.model.requires_grad_(False)
        if device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).reshape(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).reshape(1, 3, 1, 1)

    @torch.inference_mode()
    def landmarks(
        self,
        images: tuple[Image.Image, ...],
    ) -> Float[np.ndarray, "b n d"]:
        """返回与原 HRNet 清洗一致的 28 个热图峰值与响应。

        Args:
            images: 按缓存顺序排列的 256×256 最终 RGB 输入。

        Raises:
            RuntimeError: 模型或底层运算失败。

        Results:
            Float[np.ndarray, "b n d"]: [b,28,3]，各点为 64×64 热图的 x、y、置信度。
        """
        pixels = np.stack([np.asarray(image) for image in images])
        batch = torch.from_numpy(pixels).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        batch.div_(255.0).sub_(self.mean).div_(self.std)
        heatmap = self.model.keypoint_head(self.model.backbone(batch))
        confidence, locations = heatmap.flatten(2).max(dim=-1)
        x = locations.remainder(GRID_SIZE)
        y = locations.div(GRID_SIZE, rounding_mode="floor")
        return torch.stack((x, y, confidence), dim=-1).cpu().numpy()


def feature_mask(
    center: Float[np.ndarray, " d"],
    along: Float[np.ndarray, " d"],
    radius_x: float,
    radius_y: float,
) -> Float[np.ndarray, "h w"]:
    """在热图坐标系生成带羽化边缘的椭圆区域。

    Args:
        center: [2] 眼眶、鼻部或嘴部中心坐标。
        along: [2] 双眼连线单位向量。
        radius_x: 沿双眼连线的半径。
        radius_y: 垂直眼线方向的半径。

    Results:
        Float[np.ndarray, "h w"]: [64,64]，取值为 0 到 1。
    """
    dx = GRID_X - center[0]
    dy = GRID_Y - center[1]
    u = (dx * along[0] + dy * along[1]) / radius_x
    v = (-dx * along[1] + dy * along[0]) / radius_y
    level = np.clip((1.45 - np.hypot(u, v)) / 0.70, 0, 1)
    return level * level * (3 - 2 * level)


def region_weights(
    landmarks: Float[np.ndarray, "n d"],
) -> tuple[Float[np.ndarray, "h w"], tuple[bool, ...]]:
    """生成每张训练头像均值为一的潜空间权重。

    Args:
        landmarks: [28,3] 热图坐标与响应强度。

    Results:
        tuple: [32,32] FP32 权重，以及全脸、左眼、右眼、鼻部和嘴部分区是否有效。
    """
    points, scores = landmarks[:, :2], landmarks[:, 2]
    eyes = points[11:23].reshape(2, 6, 2)
    centers = eyes.mean(axis=1)
    eye_scores = scores[11:23].reshape(2, 6).mean(axis=1)
    midpoint = centers.mean(axis=0)
    vector = centers[1] - centers[0]
    distance = float(np.linalg.norm(vector))
    jaw = points[:5]
    valid_face = (
        np.isfinite(landmarks).all()
        and 5.0 <= distance <= 46.0
        and scores[:5].mean() >= 0.30
        and scores[2] >= 0.25
        and eye_scores.max() >= 0.38
    )
    if not valid_face:
        return np.ones((LATENT_SIZE, LATENT_SIZE), dtype=np.float32), (False,) * len(REGIONS)

    axis = vector / distance
    down = np.array((-axis[1], axis[0]), dtype=np.float32)
    chin_drop = float((jaw[2] - midpoint) @ down) / distance
    if not 0.28 <= chin_drop <= 2.1:
        return np.ones((LATENT_SIZE, LATENT_SIZE), dtype=np.float32), (False,) * len(REGIONS)

    mouth = points[24:28].mean(axis=0)
    nose = points[23]
    nose_drop = float((nose - midpoint) @ down) / distance
    mouth_drop = float((mouth - midpoint) @ down) / distance
    nose_side = abs(float((nose - midpoint) @ axis)) / distance
    mouth_side = abs(float((mouth - midpoint) @ axis)) / distance
    nose_ok = bool(scores[23] >= 0.25 and -0.1 <= nose_drop <= chin_drop and nose_side <= 0.6)
    mouth_ok = bool(
        scores[24:28].mean() >= 0.30
        and max(0.12, nose_drop if nose_ok else 0.0) <= mouth_drop <= chin_drop + 0.06
        and mouth_side <= 0.65
    )
    eye_ok = tuple(
        bool(conf >= 0.32 and np.ptp(eye @ axis) >= 1.1)
        for eye, conf in zip(eyes, eye_scores, strict=True)
    )
    forehead = midpoint - 0.52 * distance * down
    cap = np.stack((forehead - 0.67 * distance * axis, forehead + 0.67 * distance * axis))
    anchors = [jaw, points[5:11], cap, eyes.reshape(12, 2)]
    if nose_ok:
        anchors.append(nose[None])
    if mouth_ok:
        anchors.append(mouth[None])
    face_center = midpoint + 0.24 * distance * down
    expanded = face_center + 1.10 * (np.concatenate(anchors) - face_center)
    polygon = cv2.convexHull(np.rint(256 * expanded).astype(np.int32))
    face = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    cv2.fillConvexPoly(face, polygon, 1.0, shift=8)
    face = cv2.GaussianBlur(face, (0, 0), sigmaX=max(0.65, 0.065 * distance))

    local = np.zeros_like(face)
    for eye, center, include in zip(eyes, centers, eye_ok, strict=True):
        if include:
            rx = max(1.5, 0.73 * float(np.ptp(eye @ axis)))
            ry = max(1.30, 0.16 * distance, 0.72 * float(np.ptp(eye @ down)))
            np.maximum(local, feature_mask(center, axis, rx, ry), out=local)
    if nose_ok:
        rx, ry = max(1.3, 0.15 * distance), max(1.2, 0.18 * distance)
        np.maximum(local, feature_mask(nose, axis, rx, ry), out=local)
    if mouth_ok:
        rx = max(1.5, 0.19 * distance, 0.76 * float(np.ptp(points[24:28] @ axis)))
        ry = max(1.2, 0.15 * distance, 0.80 * float(np.ptp(points[24:28] @ down)))
        np.maximum(local, feature_mask(mouth, axis, rx, ry), out=local)
    unscaled = 1.0 + np.maximum(face, local) + local
    area = unscaled.reshape(LATENT_SIZE, 2, LATENT_SIZE, 2).mean(axis=(1, 3))
    area /= area.mean()
    return area, (True, *eye_ok, nose_ok, mouth_ok)


def render_preview(
    path: Path,
    examples: list[tuple[str, Image.Image, Float[np.ndarray, "h w"], tuple[bool, ...]]],
) -> None:
    """并排保存真实训练输入和热力图，检查局部权重是否位于五官。

    Args:
        path: 输出 PNG 路径。
        examples: 样本 ID、冻结输入、潜空间权重及五个区域状态。

    Raises:
        OSError: 图片写入失败。

    副作用:
        写入 PNG 对照图。
    """
    cell, caption, columns = 176, 36, 3
    canvas = Image.new(
        "RGB",
        (2 * cell * columns, (cell + caption) * ((len(examples) + columns - 1) // columns)),
        "#f6f3ee",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (sample_id, image, weights, flags) in enumerate(examples):
        x = index % columns * 2 * cell
        y = index // columns * (cell + caption)
        raw = image.resize((cell, cell), Image.Resampling.BICUBIC)
        upscaled = Image.fromarray(weights).resize((cell, cell), Image.Resampling.BILINEAR)
        scale = np.asarray(upscaled)
        alpha = np.clip(0.65 * (scale - 0.82), 0, 0.78)[..., None]
        heat = np.asarray(raw, dtype=np.float32) * (1 - alpha) + np.array([241, 68, 44]) * alpha
        canvas.paste(raw, (x, y))
        canvas.paste(Image.fromarray(heat.astype(np.uint8)), (x + cell, y))
        regions = " ".join(region for region, ok in zip(REGIONS, flags, strict=True) if ok)
        draw.text((x + 6, y + cell + 2), f"{sample_id}  {regions or 'uniform'}", fill="#2e2932")
        draw.text(
            (x + 6, y + cell + 17),
            f"Normalized weight {weights.min():.2f}–{weights.max():.2f}",
            fill="#2e2932",
        )
    canvas.save(path)


def build_cache(
    root: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    preview: bool,
) -> JsonDict:
    """在原训练顺序逐图验证并发布五官软权重及身份回执。

    Args:
        root: 含固定 Parquet、训练清单和检测器资产的项目根。
        output: 独立输出目录。
        device: 自动地标模型使用的执行设备。
        batch_size: 推理批量。
        preview: 只计算确定性的十二张实例，用作布局预览。

    Raises:
        OSError/RuntimeError/ValueError: 图像身份、推理、数组验证或写入失败。

    Results:
        JsonDict: 训练权重或预览的可复核身份及区域统计。

    副作用:
        写入独立权重数组、回执和对照图，并报告进度。
    """
    if output.exists():
        raise FileExistsError(f"Weight cache already exists: {output}")
    rows, manifest_sha256 = training_rows(root)
    if preview:
        indices = np.linspace(0, len(rows) - 1, 12, dtype=int)
        rows = [rows[index] for index in indices]
    partial = output.with_name(f".{output.name}-staging")
    partial.mkdir(parents=True, exist_ok=False)
    model = FaceLandmarks(root, device)
    start = perf_counter()
    shape = (len(rows), LATENT_SIZE, LATENT_SIZE)
    array_path = partial / "train.npy"
    array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float16, shape=shape)
    preview_indices = set(np.linspace(0, len(rows) - 1, 12, dtype=int))
    examples = []
    counts: Counter[str] = Counter({region: 0 for region in REGIONS})
    counts.update({"uniform": 0, "face_without_local_regions": 0})
    count = 0
    for images in batched(iter_pixels(root, rows), batch_size):
        for image, landmarks in zip(images, model.landmarks(images), strict=True):
            weights, flags = region_weights(landmarks)
            counts.update(
                region for region, included in zip(REGIONS, flags, strict=True) if included
            )
            if not flags[0]:
                counts["uniform"] += 1
            elif not any(flags[1:]):
                counts["face_without_local_regions"] += 1
            if count in preview_indices:
                examples.append((rows[count]["sample_id"], image, weights, flags))
            array[count] = weights
            count += 1
        if count % (batch_size * 16) == 0:
            print(
                f"[face weights] {count}/{len(rows)} images, {perf_counter() - start:.1f}s",
                flush=True,
            )
    array.flush()
    del array
    readback = np.load(array_path, mmap_mode="r", allow_pickle=False)
    maximum_deviation = 0.0
    for begin in range(0, count, 256):
        chunk = readback[begin : begin + 256]
        if not np.isfinite(chunk).all() or chunk.min() <= 0:
            raise ValueError("Nonpositive or nonfinite face weights in cache")
        mean = chunk.mean(axis=(1, 2), dtype=np.float32)
        maximum_deviation = max(maximum_deviation, float(np.max(np.abs(mean - 1))))
    if count != len(rows) or readback.shape != shape or maximum_deviation > 0.001:
        raise ValueError("Face weight cache readback failed")
    ids = [row["sample_id"] for row in rows]
    identity = {
        "schema": "minigen.anime_b2_face_weights.v1",
        "preview": preview,
        "recipe": RECIPE,
        "detector": DETECTOR,
        "shape": list(shape),
        "dtype": "float16",
        "train_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "array_sha256": hash_file(array_path),
        "source_manifest_sha256": manifest_sha256,
        "coverage_count": dict(counts),
        "coverage_fraction": {region: number / count for region, number in counts.items()},
        "max_mean_error": maximum_deviation,
        "preview_ids": [item[0] for item in examples],
        "elapsed_seconds": perf_counter() - start,
        "device": str(device),
    }
    render_preview(partial / "preview.png", examples)
    (partial / "index.json").write_text(json.dumps(identity, indent=2) + "\n")
    partial.rename(output)
    return identity


def main() -> None:
    """在 AutoDL 获取固定检测器，或生成带样本校验的权重缓存。

    Raises:
        OSError/RuntimeError/ValueError: 模型资产、数据、设备或输出异常。

    副作用:
        明确选择 download 时获取公开检测器。其余模式执行指定设备的推理并写入回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download", "preview", "cache"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--base-url",
        default="https://hf-mirror.com",
        choices=("https://hf-mirror.com", "https://huggingface.co"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "download":
        folder = root / DETECTOR_DIR
        weights_url = (
            f"{args.base_url}/{DETECTOR_REPO}/resolve/{DETECTOR_REVISION}/{WEIGHTS['name']}"
        )
        fetch_file(weights_url, folder / WEIGHTS["name"], WEIGHTS["bytes"], WEIGHTS["sha256"])
        fetch_file(WHEEL_URL, folder / WHEEL["name"], WHEEL["bytes"], WHEEL["sha256"])
        print(json.dumps({"status": "downloaded", "detector": DETECTOR}, indent=2))
        return
    preview = args.operation == "preview"
    output = root / (args.output or (PREVIEW if preview else OUTPUT))
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    result = build_cache(root, output, torch.device(args.device), args.batch_size, preview)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
