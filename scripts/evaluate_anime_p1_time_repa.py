"""用原 208 题评测三个 patch1 消融及其同架构 Uniform 对照。"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from evaluate_anime_four_benchmark import digest, freeze, render, report, reveal, write_new_json
from render_anime_benchmark_blind import RECEIPT_FIELDS, read_json
from score_anime_scale_benchmark import PRIMARY_THRESHOLDS, SEEDS

from minigen._typing import JsonDict

ROOT = Path(__file__).resolve().parents[1]
SCOPE = Path("configs/evaluation/anime_p1_time_repa.json")
BENCHMARK = Path("configs/benchmark/anime_faces_v1.json")
RUBRIC = Path("reports/v2/anime/benchmark_v1_rubric.md")
FAMILY = Path("artifacts/exports/anime-faces-b1")
BASE_GENERATION = FAMILY / "four_50m_30k_benchmark/generation"
OUTPUT = FAMILY / "p1_time_repa_benchmark"
RUNS = Path("artifacts/runs/anime_faces_256")
SHARED = (
    *RECEIPT_FIELDS,
    "data_identity",
    "condition_features_sha256",
    "condition_mask_sha256",
    "condition_encoding",
    "sampler_helper_sha256",
    "evaluation_torch",
    "sampling_batch_size",
    "decode_batch_size",
    "noise_shape",
    "tf32",
)


def read_scope(
    root: Path,
) -> tuple[JsonDict, JsonDict]:
    """核对固定题库、盲评规则和各分支的对应基线。

    Args:
        root: 项目根目录。

    Raises:
        ValueError: 题库身份、训练范围、采样种子或阈值不符。

    Results:
        tuple: 已登记的消融范围与原题库。
    """
    scope = read_json(root / SCOPE)
    benchmark = read_json(root / BENCHMARK)
    branches = [
        (spec["baseline"], name) for name, spec in scope["candidates"].items() if "run" in spec
    ]
    if (
        scope["schema"] != "minigen.anime_faces.p1_time_repa_scope.v1"
        or scope["benchmark_sha256"] != digest(root / BENCHMARK)
        or scope["rubric_sha256"] != digest(root / RUBRIC)
        or benchmark["seeds"] != list(SEEDS)
        or scope["seeds"] != list(SEEDS)
        or scope["parent_step"] != 25000
        or scope["final_step"] != 30000
        or scope["thresholds"] != PRIMARY_THRESHOLDS
        or scope["both_seeds_positive"] is not True
        or len(scope["candidates"]) != 5
        or len(branches) != 3
        or scope["baseline_pairs"] != [list(pair) for pair in branches]
    ):
        raise ValueError("Ablation scope differs from the registered protocol")
    return scope, benchmark


def original_receipt(
    root: Path,
    spec: JsonDict,
) -> tuple[Path, JsonDict]:
    """读取先前完成的同架构 Uniform 生成回执。

    Args:
        root: 原始生成产物所在项目根目录。
        spec: 含原实验臂与原始回执摘要的基线登记。

    Raises:
        ValueError: 原回执身份不符。

    Results:
        tuple: 原图片目录和已核验的生成回执。
    """
    folder = root / BASE_GENERATION / spec["arm"]
    path = folder / "receipt.json"
    receipt = read_json(path)
    if digest(path) != spec["receipt_sha256"] or receipt["arm"] != spec["arm"]:
        raise ValueError(f"Uniform receipt identity changed: {path}")
    return folder, receipt


def copy_baselines(
    root: Path,
    output: Path,
) -> None:
    """核验 Uniform 原图并将其放入本轮生成目录。

    Args:
        root: 项目根目录。
        output: 本轮独立评测目录。

    Raises:
        ValueError: 某张 Uniform 图片与原始生成回执不同。

    副作用:
        发布两组复用图和本轮引用回执。
    """
    scope, _ = read_scope(root)
    generation = output / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    for arm, spec in scope["candidates"].items():
        if "receipt_sha256" not in spec or (generation / arm).exists():
            continue
        source, old = original_receipt(root, spec)
        with TemporaryDirectory(prefix=".uniform-", dir=generation) as temporary:
            pending = Path(temporary) / arm
            pending.mkdir()
            for row in old["records"]:
                filename = row["image_path"]
                image = source / filename
                if digest(image) != row["image_sha256"]:
                    raise ValueError(f"Original Uniform PNG changed: {image}")
                shutil.copyfile(image, pending / filename)
            write_new_json(
                pending / "receipt.json",
                old
                | {
                    "arm": arm,
                    "original_arm": spec["arm"],
                    "original_receipt_sha256": spec["receipt_sha256"],
                    "comparison_scope_sha256": digest(root / SCOPE),
                },
            )
            pending.rename(generation / arm)


def branch_acceptance(
    root: Path,
    spec: JsonDict,
    baseline: JsonDict,
) -> tuple[Path, JsonDict]:
    """读取恢复验收，检查同架构配对的父快照与 Uniform 终点。

    Args:
        root: 项目根目录。
        spec: 本分支输出位置及原父权重摘要。
        baseline: 已核验的同架构 Uniform 30k 回执。

    Raises:
        ValueError: 恢复轨迹、数据或快照身份不同。

    Results:
        tuple: 该分支的编号终点路径及通过配对检查的验收数据。
    """
    directory = root / RUNS / spec["run"]
    path = directory / "acceptance.json"
    accepted = read_json(path)
    expected = {
        "status": "verified",
        "step": 30000,
        "parent_step": 25000,
        "training_updates": 5000,
        "parent_sha256": spec["parent_sha256"],
        "parent_run_id": baseline["run_id"],
        "baseline_sha256": baseline["checkpoint_sha256"],
        "paired_stream_and_rng": True,
        "data_identity": baseline["data_identity"],
    }
    if any(accepted.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Training acceptance does not establish matched 25k–30k ancestry: {path}")
    return directory / "step_030000.pt", accepted


def generate(
    root: Path,
    output: Path,
    arm: str,
) -> None:
    """用冻结推理包还原分支 EMA 并生成原题库的 256 张图。

    Args:
        root: 项目根目录。
        output: 本轮独立评测目录。
        arm: 需要出图的消融分支名称。

    Raises:
        ValueError: 快照、真实缓存、条件张量或初噪身份不符。
        RuntimeError: CUDA 推理或 EMA 加载失败。

    副作用:
        在 CUDA 上加载推理模型，完成后发布 PNG 与回执。
    """
    import torch
    from evaluate_anime_time_ablation import tensor_digest
    from generate_anime_benchmark import (
        benchmark_noise,
        encode_benchmark_conditions,
        plan_endpoints,
        publish_benchmark_arm,
        read_protocol,
        validate_sources,
    )
    from run_anime_b1 import open_anime_cache
    from train_anime_six import read_arm_config

    from minigen.autoencoders.sdxl import FrozenVAE, hash_file
    from minigen.training.checkpoint import CHECKPOINT_SCHEMA, run_identity
    from minigen.training.config import build_model
    from minigen.training.runtime import resolve_device

    started = perf_counter()
    scope, _ = read_scope(root)
    spec = scope["candidates"][arm]
    baseline_spec = scope["candidates"][spec["baseline"]]
    _, baseline = original_receipt(root, baseline_spec)
    snapshot, accepted = branch_acceptance(root, spec, baseline)
    target = output / "generation" / arm
    if target.exists():
        verify_arm(target, arm, baseline, scope, digest(root / SCOPE), accepted)
        return
    checkpoint_sha = hash_file(snapshot)
    package = Path(inspect.getfile(build_model)).resolve().parents[1]
    source_sha = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        source_sha.update(str(file.relative_to(package)).encode() + b"\0" + file.read_bytes())
    if (
        checkpoint_sha != accepted["checkpoint_sha256"]
        or source_sha.hexdigest() != baseline["source_sha256"]
        or str(torch.__version__) != baseline["evaluation_torch"]
        or hash_file(root / "scripts/generate_anime_benchmark.py")
        != baseline["sampler_helper_sha256"]
    ):
        raise ValueError("Checkpoint or frozen inference implementation differs from its lock")
    benchmark, exclusions, protocol = read_protocol(root)
    config, data = read_arm_config(root, baseline_spec["arm"])
    source, _, blank, identity = open_anime_cache(root, data)
    validate_sources(benchmark, exclusions, source, identity)
    device = resolve_device("cuda", "fp32")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    endpoints = plan_endpoints(benchmark)
    with torch.inference_mode():
        condition = encode_benchmark_conditions(root, root / data["clip_lock"], endpoints)
        features_hash = tensor_digest(condition.features)
        mask_hash = tensor_digest(condition.mask)
        if (
            identity != baseline["data_identity"]
            or features_hash != baseline["condition_features_sha256"]
            or mask_hash != baseline["condition_mask_sha256"]
            or config.model != baseline["model_config"]
            or any(
                tensor_digest(benchmark_noise(seed)) != baseline["seed_noise_sha256"][str(seed)]
                for seed in SEEDS
            )
        ):
            raise ValueError(
                "The actual cache, tags, architecture or initial noise differs from Uniform"
            )
        state = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
        if (
            state["schema"] != CHECKPOINT_SCHEMA
            or state["step"] != 30000
            or state["objective"] != "flow_velocity"
            or state["config"]["model"] != config.model
            or state["data_identity"] != identity
            or run_identity(state["contract"]) != accepted["run_id"]
        ):
            raise ValueError(f"Checkpoint content differs from accepted training: {snapshot}")
        with torch.device("meta"):
            model = build_model(config.model)
        model.load_state_dict(state["ema"], strict=True, assign=True)
        model.eval().requires_grad_(False)
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != baseline["trainable_parameters"]
        ):
            raise ValueError(f"The EMA backbone size differs from Uniform: {arm}")
        del state
        vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
        publish_benchmark_arm(
            target,
            model.to(device),
            vae,
            condition.to(device),
            blank.to(device),
            endpoints,
            benchmark["sampling"],
            protocol
            | {
                "arm": arm,
                "checkpoint_step": 30000,
                "cumulative_training_updates": 30000,
                "checkpoint_sha256": checkpoint_sha,
                "run_id": accepted["run_id"],
                "source_sha256": accepted["source_sha256"],
                "inference_source_sha256": source_sha.hexdigest(),
                "data_identity": identity,
                "model_config": config.model,
                "trainable_parameters": baseline["trainable_parameters"],
                "weights": "ema",
                "comparison_scope_sha256": digest(root / SCOPE),
                "parent_sha256": spec["parent_sha256"],
                "training_acceptance_sha256": digest(snapshot.parent / "acceptance.json"),
                "condition_features_sha256": features_hash,
                "condition_mask_sha256": mask_hash,
                "sampler_helper_sha256": baseline["sampler_helper_sha256"],
                "sampler_sha256": digest(Path(__file__)),
            },
            started,
        )
    verify_arm(target, arm, baseline, scope, digest(root / SCOPE), accepted)


def verify_arm(
    directory: Path,
    arm: str,
    baseline: JsonDict,
    scope: JsonDict,
    scope_hash: str,
    accepted: JsonDict | None = None,
) -> JsonDict:
    """逐图核对生成回执内的端点、初噪、文件摘要与权重身份。

    Args:
        directory: 单个实验臂的生成目录。
        arm: 本轮实验臂名称。
        baseline: 冻结的原 Uniform 回执。
        scope: 本轮预登记范围。
        scope_hash: 预登记范围的内容摘要。
        accepted: 分支终点验收。Uniform 原图复用时省略。

    Raises:
        ValueError: 生成来源、样图或初始化噪声不一致。

    Results:
        JsonDict: 本轮通过核验的生成回执。
    """
    receipt = read_json(directory / "receipt.json")
    spec = scope["candidates"][arm]
    fields = (*SHARED, "model_config", "trainable_parameters", "weights", "checkpoint_step")
    checkpoint = baseline if accepted is None else accepted
    if (
        receipt["arm"] != arm
        or receipt["schema"] != "minigen.anime_faces.benchmark_generation.v1"
        or receipt["status"] != "completed"
        or receipt["comparison_scope_sha256"] != scope_hash
        or receipt["benchmark_sha256"] != scope["benchmark_sha256"]
        or receipt["rubric_sha256"] != scope["rubric_sha256"]
        or receipt["seeds"] != scope["seeds"]
        or receipt["checkpoint_step"] != scope["final_step"]
        or receipt["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or receipt["run_id"] != checkpoint["run_id"]
        or len(receipt["records"]) != 256
        or len(baseline["records"]) != 256
        or any(receipt[field] != baseline[field] for field in fields)
    ):
        raise ValueError(f"Generation receipt does not match the paired evaluation scope: {arm}")
    if accepted is None and receipt["original_receipt_sha256"] != spec["receipt_sha256"]:
        raise ValueError(f"Uniform source receipt differs from the lock: {arm}")
    if accepted is not None and (
        receipt["parent_sha256"] != spec["parent_sha256"]
        or receipt["inference_source_sha256"] != baseline["source_sha256"]
    ):
        raise ValueError(f"Branch ancestry or inference source differs from the lock: {arm}")
    record_fields = (
        "id",
        "seed",
        "endpoint",
        "image_path",
        "prompt",
        "noise_index",
        "noise_sha256",
    )
    for actual, reference in zip(receipt["records"], baseline["records"], strict=True):
        filename = actual["image_path"]
        if (
            any(actual[key] != reference[key] for key in record_fields)
            or digest(directory / filename) != actual["image_sha256"]
            or (accepted is None and actual["image_sha256"] != reference["image_sha256"])
        ):
            raise ValueError(f"Image hash, prompts or paired noise changed: {arm}/{filename}")
    return receipt


def verify_generation(
    root: Path,
    output: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, str]:
    """核对五组完整生成，以及相同的题本、初噪和采样条件。

    Args:
        root: 项目根目录。
        output: 本轮独立评测目录。

    Raises:
        ValueError: 权重、样图、文本条件或采样协议不符。

    Results:
        tuple: 评测范围、题库、五组回执和范围摘要。
    """
    scope, benchmark = read_scope(root)
    scope_hash = digest(root / SCOPE)
    receipts = {}
    for arm, spec in scope["candidates"].items():
        uniform_spec = scope["candidates"][spec["baseline"]] if "run" in spec else spec
        _, baseline = original_receipt(root, uniform_spec)
        accepted = branch_acceptance(root, spec, baseline)[1] if "run" in spec else None
        if baseline["sampling"] != benchmark["sampling"]:
            raise ValueError(f"Uniform sampling does not match the benchmark: {arm}")
        receipts[arm] = verify_arm(
            output / "generation" / arm, arm, baseline, scope, scope_hash, accepted
        )
    first = next(iter(receipts.values()))
    if (
        any(any(receipt[key] != first[key] for key in SHARED) for receipt in receipts.values())
        or len({receipt["checkpoint_sha256"] for receipt in receipts.values()}) != 5
    ):
        raise ValueError("Five distinct paired EMA checkpoints are required")
    return scope, benchmark, receipts, scope_hash


def main() -> None:
    """运行采样、匿名渲染、评分锁、解盲或报告阶段。

    副作用:
        按选择写生成和评测产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("baselines", "generate", "render", "freeze", "reveal", "report")
    )
    parser.add_argument("--arm", choices=("b1_p1_mid60", "b1_p1_repa", "b2_p1_repa"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if args.command == "baselines":
        copy_baselines(root, output)
    elif args.command == "generate":
        if args.arm is None:
            parser.error("generate requires --arm")
        generate(root, output, args.arm)
    elif args.command == "render":
        render(root, output, prepared=verify_generation(root, output))
    elif args.command == "freeze":
        if args.scores is None:
            parser.error("freeze requires --scores")
        freeze(output, root / args.scores)
    elif args.command == "reveal":
        reveal(root, output, scope_path=SCOPE)
    else:
        report(root, output, scope_path=SCOPE)


if __name__ == "__main__":
    main()
