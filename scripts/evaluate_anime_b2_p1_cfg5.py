"""在原 208 题上配对评测 B2-p1 三份 30k EMA 的 CFG 3 与 5。"""

import argparse
from pathlib import Path

from evaluate_anime_four_benchmark import BENCHMARK, ROOT, digest, freeze, render, report, reveal
from render_anime_benchmark_blind import RECEIPT_FIELDS, read_json
from score_anime_scale_benchmark import PRIMARY_THRESHOLDS, SEEDS

from minigen._typing import JsonDict

SCOPE = Path("configs/evaluation/anime_b2_p1_cfg5.json")
RUBRIC = Path("reports/v2/anime/benchmark_v1_rubric.md")
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_p1_cfg5_benchmark")
VARIANTS = ("uniform", "mid60", "repa")
COMMON_FIELDS = (
    *(name for name in RECEIPT_FIELDS if name != "sampling"),
    "checkpoint_step",
    "condition_encoding",
    "condition_features_sha256",
    "condition_mask_sha256",
    "data_identity",
    "decode_batch_size",
    "device",
    "evaluation_torch",
    "model_config",
    "noise_shape",
    "sampling_batch_size",
    "status",
    "tf32",
    "trainable_parameters",
    "weights",
)
PAIRED_FIELDS = (
    *COMMON_FIELDS,
    "checkpoint_sha256",
    "run_id",
    "sampler_helper_sha256",
    "source_sha256",
)


def read_scope(
    root: Path,
) -> tuple[JsonDict, JsonDict]:
    """读取已登记的指导强度、同权重配对及定稿题库。

    Args:
        root: 项目根目录。

    Raises:
        ValueError: 范围、内容摘要、初噪或配对与预定计划不符。

    Results:
        tuple: 实验范围和基准题库读本。
    """
    scope = read_json(root / SCOPE)
    benchmark = read_json(root / BENCHMARK)
    candidates = {
        f"{variant}_cfg{cfg}": {"variant": variant, "cfg": cfg}
        for variant in VARIANTS
        for cfg in (3, 5)
    }
    pairs = [[f"{variant}_cfg3", f"{variant}_cfg5"] for variant in VARIANTS]
    if (
        scope["schema"] != "minigen.anime_faces.b2_p1_cfg5_scope.v1"
        or scope["benchmark_sha256"] != digest(root / BENCHMARK)
        or scope["rubric_sha256"] != digest(root / RUBRIC)
        or scope["final_step"] != 30000
        or scope["seeds"] != list(SEEDS)
        or benchmark["seeds"] != scope["seeds"]
        or benchmark["sampling"]["cfg"] != 3
        or tuple(scope["variants"]) != VARIANTS
        or tuple(scope["candidates"]) != tuple(candidates)
        or scope["candidates"] != candidates
        or scope["baseline_pairs"] != pairs
        or scope["thresholds"] != PRIMARY_THRESHOLDS
        or scope["both_seeds_positive"] is not True
    ):
        raise ValueError("CFG comparison does not match the registered scope")
    return scope, benchmark


def verified_receipts(
    root: Path,
    output: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, str]:
    """核对六组回执与三个已冻结的 CFG 3 来源。

    Args:
        root: 固定题库、范围配置和 CFG 3 原回执所在项目根目录。
        output: 本轮六组生成目录所在位置。

    Raises:
        ValueError: 快照、采样协议、初噪或配对端点身份不符。

    Results:
        tuple: 实验范围、题库、核验后的回执和范围摘要。
    """
    scope, benchmark = read_scope(root)
    scope_hash = digest(root / SCOPE)
    originals = {}
    for variant, spec in scope["variants"].items():
        path = root / spec["original_receipt"]
        original = read_json(path)
        if (
            digest(path) != spec["original_receipt_sha256"]
            or original["schema"] != "minigen.anime_faces.benchmark_generation.v1"
            or original["arm"] != spec["original_arm"]
            or original["checkpoint_sha256"] != spec["checkpoint_sha256"]
            or original["checkpoint_step"] != scope["final_step"]
            or original["benchmark_sha256"] != scope["benchmark_sha256"]
            or original["rubric_sha256"] != scope["rubric_sha256"]
            or original["seeds"] != scope["seeds"]
            or original["status"] != "completed"
            or original["sampling"] != benchmark["sampling"]
            or len(original["records"]) != 256
        ):
            raise ValueError(f"Frozen CFG 3 origin differs from scope: {variant}")
        originals[variant] = original
    reference = originals[VARIANTS[0]]
    if len({row["checkpoint_sha256"] for row in originals.values()}) != len(VARIANTS) or any(
        row[field] != reference[field] for row in originals.values() for field in COMMON_FIELDS
    ):
        raise ValueError("Expected distinct B2-p1 checkpoints and a common sampling protocol")
    receipts = {}
    for arm, candidate in scope["candidates"].items():
        variant, cfg = candidate["variant"], candidate["cfg"]
        spec, original = scope["variants"][variant], originals[variant]
        receipt = read_json(output / "generation" / arm / "receipt.json")
        if (
            receipt["arm"] != arm
            or receipt["comparison_scope_sha256"] != scope_hash
            or receipt["original_arm"] != spec["original_arm"]
            or receipt["original_receipt_sha256"] != spec["original_receipt_sha256"]
            or receipt.get("inference_source_sha256", receipt["source_sha256"])
            != spec["inference_source_sha256"]
            or receipt["sampling"] != (benchmark["sampling"] | {"cfg": cfg})
            or any(receipt[field] != original[field] for field in PAIRED_FIELDS)
            or len(receipt["records"]) != 256
        ):
            raise ValueError(f"Generation receipt differs from its registered CFG pair: {arm}")
        for actual, expected in zip(receipt["records"], original["records"], strict=True):
            paired = expected if cfg == 3 else expected | {"image_sha256": actual["image_sha256"]}
            if actual != paired:
                raise ValueError(f"Prompt, noise, or reused PNG fingerprint differs: {arm}")
        receipts[arm] = receipt
    return scope, benchmark, receipts, scope_hash


def main() -> None:
    """运行匿名渲染、正式锁分、解盲或报告阶段。

    副作用:
        按明确选择的阶段写评测产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "freeze", "reveal", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if args.command == "render":
        render(root, output, prepared=verified_receipts(root, output))
    elif args.command == "freeze":
        if args.scores is None:
            parser.error("freeze requires --scores")
        read_scope(root)
        freeze(output, root / args.scores)
    elif args.command == "reveal":
        read_scope(root)
        reveal(root, output, scope_path=SCOPE)
    else:
        read_scope(root)
        report(root, output, scope_path=SCOPE)


if __name__ == "__main__":
    main()
