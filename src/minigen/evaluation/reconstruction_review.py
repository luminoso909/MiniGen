"""生成供人工审批的重建图表与可追溯数值；所有图像保持在本地运行目录。"""

import html
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image, ImageDraw
from torch import Tensor

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.evaluation.reconstruction import ReconstructionSample, load_image
from minigen.evaluation.reconstruction_metrics import LPIPSAlex, psnr, ssim

_METRICS = ("psnr_db", "ssim", "lpips_alex_v01")


class ImageScore(TypedDict):
    """记录一个可追溯图像对的量化结果和局部比较范围。"""

    sample_id: str
    cohort: str
    reason: str
    psnr_db: float | Literal["Infinity"]
    ssim: float
    lpips_alex_v01: float
    crop_xyxy: tuple[int, int, int, int]


def detail_box(
    reference: Float[Tensor, "3 h w"],
    estimated: Float[Tensor, "3 h w"],
    *,
    size: int = 64,
    stride: int = 32,
) -> tuple[int, int, int, int]:
    """用局部 RGB 绝对误差选择需要放大查看的区域。

    Args:
        reference: CPU [3,h,w]、[0,1] 原图。
        estimated: 同形 CPU 重建图。
        size: 正方形裁剪边长。
        stride: 网格移动步幅；额外包含最右、最下边界。

    Raises:
        ValueError: 尺寸、步长或图像配对不合法。

    Results:
        tuple[int, int, int, int]: (左, 上, 右, 下)，同分时选行优先首个区域。
    """
    if (
        reference.ndim != 3
        or reference.shape[0] != 3
        or reference.shape != estimated.shape
        or reference.device.type != "cpu"
        or estimated.device.type != "cpu"
    ):
        raise ValueError("Detail crop expects a matching single RGB image pair")
    h, w = reference.shape[-2:]
    if not 0 < size <= min(h, w) or stride <= 0:
        raise ValueError("Detail crop size or stride is invalid")
    error = (reference - estimated).abs().mean(dim=0).numpy()
    xs = sorted({*range(0, w - size + 1, stride), w - size})
    ys = sorted({*range(0, h - size + 1, stride), h - size})
    x, y = max(
        ((x, y) for y in ys for x in xs),
        key=lambda xy: error[xy[1] : xy[1] + size, xy[0] : xy[0] + size].mean(),
    )
    return x, y, x + size, y + size


def _make_row(
    original: Float[Tensor, "3 h w"],
    decoded: Float[Tensor, "3 h w"],
    score: ImageScore,
) -> Image.Image:
    """制作 256px 原图、重建及一组对齐的误差热点局部图。

    Args:
        original: [3,256,256] 原图，像素范围 [0,1]。
        decoded: 同形的裁剪后重建。
        score: 样本身份、三个指标和原图坐标下的裁剪范围。

    Results:
        Image.Image: 可放进审计联系表的 RGB 比较行。
    """
    src, rec = [
        Image.fromarray((pixels.permute(1, 2, 0).mul(255).round().to(torch.uint8)).numpy())
        for pixels in (original, decoded)
    ]
    canvas = Image.new("RGB", (800, 300), "#f5f5f5")
    canvas.paste(src, (8, 36))
    canvas.paste(rec, (272, 36))
    crop = score["crop_xyxy"]
    for x, image in ((536, src), (668, rec)):
        canvas.paste(image.crop(crop).resize((124, 124), Image.Resampling.NEAREST), (x, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{score['sample_id']} | original", fill="black")
    draw.text((272, 6), "reconstruction", fill="black")
    draw.text((536, 6), "detail: original", fill="black")
    draw.text((668, 6), "detail: reconstruction", fill="black")
    draw.text((536, 171), f"Crop {crop}", fill="black")
    draw.text((536, 191), f"PSNR {float(score['psnr_db']):.2f} dB", fill="black")
    draw.text((536, 211), f"SSIM {score['ssim']:.4f}", fill="black")
    draw.text((536, 231), f"LPIPS {score['lpips_alex_v01']:.4f}", fill="black")
    draw.text((536, 251), score["reason"], fill="black")
    return canvas


def _contact_sheet(
    paths: Sequence[Path],
    target: Path,
) -> None:
    """将完整比较行叠放，保留原生像素便于人工核对。

    Args:
        paths: 至少一个已生成的比较行 PNG，顺序即审计顺序。
        target: 输出 PNG 文件。

    Raises:
        OSError: 任一比较行读取失败或输出不可写。

    副作用:
        写入一张原图、重建与局部细节联系表。
    """
    sheet = Image.new("RGB", (800, 300 * len(paths)), "#f5f5f5")
    for index, path in enumerate(paths):
        with Image.open(path) as row:
            sheet.paste(row, (0, index * 300))
    sheet.save(target)


def _aggregate(
    rows: list[ImageScore],
) -> JsonDict:
    """汇总主队列的分布及各指标的低质极端样本。

    Args:
        rows: 每图指标，允许诊断样本混入但不会计入汇总。

    Results:
        JsonDict: 主队列数量、每指标有限值统计、无穷值数量和最差 5 个样本。
    """
    primary = [row for row in rows if row["cohort"] == "random"]
    aggregate: JsonDict = {"count": len(primary)}
    for metric in _METRICS:
        values = [(row["sample_id"], float(row[metric])) for row in primary]
        finite = np.asarray([value for _, value in values if math.isfinite(value)])
        quantiles = (
            np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(finite) else [None] * 5
        )
        worst = sorted(
            values, key=lambda item: (-item[1] if metric == "lpips_alex_v01" else item[1], item[0])
        )[:5]
        aggregate[metric] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "p05": quantiles[0],
            "p25": quantiles[1],
            "median": quantiles[2],
            "p75": quantiles[3],
            "p95": quantiles[4],
            "nonfinite_count": len(values) - len(finite),
            "worst_5": [
                {"sample_id": sample_id, "value": value if math.isfinite(value) else "Infinity"}
                for sample_id, value in worst
            ],
        }
    return aggregate


def _write_gallery(
    output: Path,
    rows: list[ImageScore],
    aggregate: JsonDict,
) -> list[str]:
    """写入联系表和可逐张打开的离线人工核对索引。

    Args:
        output: 已包含 pairs/ 比较图的审计目录。
        rows: 按样本清单顺序的所有评分。
        aggregate: 各指标汇总，提供最差样本 ID。

    Raises:
        OSError: 比较图读取失败或索引不能写入。

    Results:
        list[str]: 联系表相对输出目录的文件名。

    副作用:
        写入 contact_*.png、worst_*.png、diagnostic_*.png 和 review.html。
    """
    random_rows = [row["sample_id"] for row in rows if row["cohort"] == "random"]
    diagnostic = [row["sample_id"] for row in rows if row["cohort"] == "diagnostic"]
    worst = list(
        dict.fromkeys(item["sample_id"] for name in _METRICS for item in aggregate[name]["worst_5"])
    )
    files = []
    links = []
    for tag, ids in (("contact", random_rows), ("worst", worst), ("diagnostic", diagnostic)):
        for start in range(0, len(ids), 5):
            name = f"{tag}_{start // 5 + 1:02d}.png"
            _contact_sheet(
                [output / "pairs" / f"{sid}.png" for sid in ids[start : start + 5]], output / name
            )
            files.append(name)
            links.append(f'<li><a href="{name}">{name}</a></li>')
    pair_links = []
    for row in rows:
        sample_id = html.escape(row["sample_id"], quote=True)
        cohort = html.escape(row["cohort"])
        pair_links.append(f'<li><a href="pairs/{sample_id}.png">{sample_id}</a> ({cohort})</li>')
    (output / "review.html").write_text(
        "<!doctype html><html lang='zh'><meta charset='utf-8'>"
        "<title>ArtBench VAE 重建人工复核</title>"
        "<body><h1>ArtBench VAE 重建人工复核</h1>"
        "<p>每行从左到右为原图、重建、同一误差热点区域的原图与重建。"
        "人工查看轮廓、细线、小物体、颜色、背景伪影，分别记录轻微、明显、无法使用。"
        "诊断组单列；数值排名用于优先检查，不能替代人工结论。</p>"
        f"<h2>联系表</h2><ul>{''.join(links)}</ul><h2>逐图</h2><ul>{''.join(pair_links)}</ul></body></html>"
    )
    return files


@torch.inference_mode()
def score_reconstructions(
    data_root: Path,
    output: Path,
    lpips: LPIPSAlex,
    *,
    manifest_sha256: str,
    expected_samples: list[ReconstructionSample],
    batch_size: int,
) -> JsonDict:
    """在同一冻结队列上计算三项指标并生成可追溯复核材料。

    Args:
        data_root: 训练图像数据版本目录。
        output: 已完成重建的审计目录。
        lpips: 已冻结且从固定权重加载的 CPU LPIPS v0.1 AlexNet。
        manifest_sha256: 本次冻结数据清单摘要，用于防止误读其他版本。
        expected_samples: 由同一冻结训练切分和种子重新抽取的有序样本。
        batch_size: CPU 指标批量，必须大于零。

    Raises:
        ValueError: 队列、重建张量、摘要或批量不符。
        OSError: 图像、元数据或本地审计文件读写失败。

    Results:
        JsonDict: 完整重建元数据、指标定义、主队列汇总和逐图指标。

    副作用:
        写入 metrics.json、全尺寸图像对、联系表和离线人工复核入口。
    """
    run = json.loads((output / "reconstruction.json").read_text())
    samples_path, decoded_path = output / "samples.json", output / "decoded.pt"
    if (
        batch_size <= 0
        or run["manifest_sha256"] != manifest_sha256
        or hash_file(samples_path) != run["samples_sha256"]
        or hash_file(decoded_path) != run["decoded_sha256"]
    ):
        raise ValueError("Reconstruction metrics input does not match run identity")
    samples = [ReconstructionSample(**row) for row in json.loads(samples_path.read_text())]
    decoded = torch.load(decoded_path, map_location="cpu", weights_only=True)
    if (
        samples != expected_samples
        or not torch.is_tensor(decoded)
        or decoded.shape != (len(samples), 3, 256, 256)
        or decoded.dtype != torch.float32
        or not torch.isfinite(decoded).all()
        or [sample.cohort for sample in samples]
        != ["random"] * run["count_random"] + ["diagnostic"] * run["count_diagnostic"]
        or len({sample.sample_id for sample in samples}) != len(samples)
    ):
        raise ValueError("Invalid reconstruction cohort or decoded tensor")

    pairs = output / "pairs"
    pairs.mkdir(exist_ok=True)
    rows: list[ImageScore] = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        original = torch.stack([load_image(data_root, sample) for sample in batch])
        reconstruction = decoded[start : start + batch_size].clamp(-1, 1).add(1).div(2)
        scores = (
            psnr(original, reconstruction),
            ssim(original, reconstruction),
            lpips(original, reconstruction),
        )
        for sample, src, rec, p, s, distance in zip(
            batch, original, reconstruction, *scores, strict=True
        ):
            row: ImageScore = {
                "sample_id": sample.sample_id,
                "cohort": sample.cohort,
                "reason": sample.reason,
                "psnr_db": p.item() if torch.isfinite(p) else "Infinity",
                "ssim": s.item(),
                "lpips_alex_v01": distance.item(),
                "crop_xyxy": detail_box(src, rec),
            }
            _make_row(src, rec, row).save(pairs / f"{sample.sample_id}.png")
            rows.append(row)
    aggregate = _aggregate(rows)
    files = _write_gallery(output, rows, aggregate)
    report: JsonDict = {
        "manifest_sha256": manifest_sha256,
        "reconstruction": run,
        "metric_protocol": {
            "pixel_domain": (
                "float32 RGB [0,1]; reconstructed pixels clipped before scaling; "
                "no uint8 quantization"
            ),
            "psnr": "data_range=1; MSE over RGB and all pixels; dB",
            "ssim": (
                "per-channel 11x11 Gaussian sigma=1.5 valid window; "
                "K1=0.01, K2=0.03; RGB and window mean"
            ),
            "lpips": (
                "v0.1 AlexNet nonspatial; CPU FP32; five ReLU layers with pinned backbone "
                "and linear calibration"
            ),
            "crop": (
                "maximum local mean absolute RGB error; size=64, stride=32; 124px nearest display"
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "aggregate": aggregate,
        "rows": rows,
        "sheets": files,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report
