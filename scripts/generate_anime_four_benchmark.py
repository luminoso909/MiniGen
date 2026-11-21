"""只读四组 50M 30k EMA，复用原题库生成 1,024 张固定配对评测图。"""

import argparse
import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
from evaluate_anime_six import load_ema
from evaluate_anime_time_ablation import tensor_digest
from generate_anime_benchmark import (
    BATCH_SIZE,
    DECODE_BATCH_SIZE,
    NOISE_SHAPE,
    SEEDS,
    benchmark_noise,
    encode_benchmark_conditions,
    plan_endpoints,
    publish_benchmark_arm,
    read_protocol,
    validate_sources,
)
from generate_anime_scale_benchmark import verify_vae_files
from run_anime_b1 import ROOT, open_anime_cache
from train_anime_six import read_arm_config

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.contracts import Condition, FlowModel
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device

ARMS = ("b1_50m_p2", "b2_50m_p2", "b1_50m_p1", "b2_50m_p1")
SCOPE = Path("configs/benchmark/anime_faces_v1_four_50m_30k.json")
OUTPUT = Path("artifacts/exports/anime-faces-b1/four_50m_30k_benchmark")


def read_scope(
    root: Path,
) -> tuple[JsonDict, JsonDict, list[JsonDict], JsonDict]:
    """核对本轮四组身份和原题库内容锁，不改变历史评测资料。

    Args:
        root: 题库、组件和训练回执所在项目根目录。

    Raises:
        OSError/ValueError: 资料不可读取，或配置与已验收权重身份不同。

    Results:
        tuple: 四组范围、带来源排除项的题库、128 个端点与共同协议摘要。
    """
    scope = json.loads((root / SCOPE).read_text())
    benchmark, exclusions, protocol = read_protocol(root)
    acceptance_path = root / scope["training_acceptance"]
    acceptance = json.loads(acceptance_path.read_text())
    if (
        scope["schema"] != "minigen.anime_faces.four_50m_30k_scope.v1"
        or tuple(scope["candidates"]) != ARMS
        or scope["seeds"] != SEEDS
        or scope["sampling"] != benchmark["sampling"]
        or any(scope[key] != value for key, value in protocol.items() if key in scope)
        or hash_file(acceptance_path) != scope["training_acceptance_sha256"]
        or acceptance["status"] != "verified"
    ):
        raise ValueError("Four-model scope differs from the accepted benchmark or training")
    # 与训练契约采用同一摘要算法，确认真正加载的是六组训练的冻结包。
    source = Path(inspect.getfile(build_model)).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(source.rglob("*.py")):
        digest.update(str(file.relative_to(source)).encode() + b"\0" + file.read_bytes())
    for arm, spec in scope["candidates"].items():
        accepted = acceptance["arms"][arm]
        snapshot = next(row for row in accepted["snapshots"] if row["file"] == "step_030000.pt")
        if (
            spec["checkpoint_sha256"] != snapshot["sha256"]
            or spec["run_id"] != accepted["result"]["run_id"]
            or spec["parameters"] != accepted["result"]["trainable_parameters"]
            or spec["model_config"] != accepted["result"]["model_config"]
            or spec["source_sha256"] != digest.hexdigest()
        ):
            raise ValueError(f"Accepted weight or frozen package changed: {arm}")
    return (
        scope,
        benchmark | {"source_exclusions": exclusions},
        plan_endpoints(benchmark),
        protocol
        | {
            "comparison": scope["comparison"],
            "comparison_scope_sha256": hash_file(root / SCOPE),
            "sampler_sha256": hash_file(Path(__file__)),
            "sampler_helper_sha256": hash_file(root / "scripts/generate_anime_benchmark.py"),
        },
    )


def read_inputs(
    root: Path,
    benchmark: JsonDict,
    endpoints: list[JsonDict],
) -> tuple[JsonDict, Condition, Condition, JsonDict]:
    """复用已验收头像缓存的空条件，按原协议编码全部题目标签。

    Args:
        root: 冻结组件和头像缓存所在根目录。
        benchmark: 原题单与来源组排除清单。
        endpoints: 依原题顺序展开的 128 个端点。

    Raises:
        OSError/ValueError/RuntimeError: 缓存、来源身份或条件编码不符合原协议。

    Results:
        tuple: 数据配置、CPU 空条件、CPU 的 [128,77,512] 条件及数据身份。
    """
    _, data = read_arm_config(root, ARMS[0])
    source, _, blank, identity = open_anime_cache(root, data)
    validate_sources(benchmark, benchmark["source_exclusions"], source, identity)
    condition = encode_benchmark_conditions(root, root / data["clip_lock"], endpoints)
    if not torch.isfinite(condition.features).all():
        raise ValueError("Nonfinite benchmark text features")
    return data, blank, condition, identity


def read_model(
    root: Path,
    arm: str,
    scope: JsonDict,
    identity: JsonDict,
) -> tuple[FlowModel, JsonDict]:
    """核对已登记的编号快照，并只加载其 FP32 EMA 速度网络。

    Args:
        root: 训练输出和配方所在根目录。
        arm: 本轮四个实验臂之一。
        scope: 含准确权重摘要及参数量的冻结范围。
        identity: 当前真实缓存的数据身份。

    Raises:
        OSError/ValueError/RuntimeError: 训练完成状态、权重或模型规格不匹配。

    Results:
        tuple: CPU EMA 模型及其生成来源记录。
    """
    model, metadata = load_ema(root, arm, identity)
    spec = scope["candidates"][arm]
    expected = {
        "run_id": spec["run_id"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_step": 30000,
        "source_sha256": spec["source_sha256"],
        "trainable_parameters": spec["parameters"],
        "model_config": spec["model_config"],
    }
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError(f"EMA differs from this evaluation scope: {arm}")
    return model, metadata | {"cumulative_training_updates": 30000, "objective": "flow_velocity"}


@torch.inference_mode()
def preflight(
    args: argparse.Namespace,
) -> None:
    """串行执行四次 CPU EMA 前向，验收真实输入而不产生正式图。

    Args:
        args: 项目根目录和预检回执路径。

    Raises:
        OSError/ValueError/RuntimeError: 资产或真实速度预测不符合固定协议。

    副作用:
        设置 CPU 线程数，读取真实权重和组件并写预检回执；优化器更新为零。
    """
    root = args.root.resolve()
    torch.set_num_threads(1)
    scope, benchmark, endpoints, protocol = read_scope(root)
    data, _, condition, identity = read_inputs(root, benchmark, endpoints)
    latent = benchmark_noise(SEEDS[0])[:1]
    sample_condition = Condition(condition.features[:1], condition.mask[:1])
    arms = {}
    for arm in ARMS:
        model, metadata = read_model(root, arm, scope, identity)
        prediction = model(latent, torch.full((1,), 0.5), sample_condition)
        if prediction.shape != latent.shape or not torch.isfinite(prediction).all():
            raise ValueError(f"Invalid CPU velocity prediction: {arm}")
        arms[arm] = metadata | {"cpu_forward_sha256": tensor_digest(prediction)}
        del model
    report = protocol | {
        "schema": "minigen.anime_faces.four_50m_30k_preflight.v1",
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "device": "cpu",
        "arms": arms,
        "seeds": SEEDS,
        "seed_noise_sha256": {str(seed): tensor_digest(benchmark_noise(seed)) for seed in SEEDS},
        "condition_features_sha256": tensor_digest(condition.features),
        "condition_mask_sha256": tensor_digest(condition.mask),
        "vae": verify_vae_files(root, root / data["vae_lock"]),
        "formal_images_generated": 0,
        "optimizer_updates": 0,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Four EMA CPU forwards passed; formal images=0: {output}", flush=True)


@torch.inference_mode()
def generate(
    args: argparse.Namespace,
) -> None:
    """在 CUDA FP32 上生成单个模型的两种子共 256 张正式图。

    Args:
        args: 项目根目录、实验臂及 generation 输出根目录。

    Raises:
        OSError/ValueError/RuntimeError: 输入或设备不符合协议，或正式输出已存在。

    副作用:
        设置 CUDA 精度及 CPU 线程，写入样图和生成回执；不创建优化器或更新参数。
    """
    started = perf_counter()
    root = args.root.resolve()
    target = root / args.output / args.arm
    if target.exists():
        raise FileExistsError(target)
    device = resolve_device("cuda", "fp32")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scope, benchmark, endpoints, protocol = read_scope(root)
    data, blank, condition, identity = read_inputs(root, benchmark, endpoints)
    model, metadata = read_model(root, args.arm, scope, identity)
    model = model.to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    metadata |= protocol | {
        "condition_features_sha256": tensor_digest(condition.features),
        "condition_mask_sha256": tensor_digest(condition.mask),
    }
    publish_benchmark_arm(
        target,
        model,
        vae,
        condition.to(device),
        blank.to(device),
        endpoints,
        benchmark["sampling"],
        metadata,
        started,
    )


def verify(
    args: argparse.Namespace,
) -> None:
    """回读已完成实验臂的全部图片摘要和配对初噪，供中断后跳过完整分支。

    Args:
        args: 项目根目录、实验臂及 generation 输出根目录。

    Raises:
        OSError/ValueError: 回执、图片或本轮身份不一致。

    副作用:
        读取本轮生成文件并打印验收结论，不生成或修改任何图片。
    """
    root = args.root.resolve()
    scope, benchmark, endpoints, protocol = read_scope(root)
    folder = root / args.output / args.arm
    receipt = json.loads((folder / "receipt.json").read_text())
    spec = scope["candidates"][args.arm]
    expected = protocol | {
        "arm": args.arm,
        "status": "completed",
        "checkpoint_step": 30000,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "run_id": spec["run_id"],
        "weights": "ema",
        "source_sha256": spec["source_sha256"],
        "model_config": spec["model_config"],
        "trainable_parameters": spec["parameters"],
        "seeds": SEEDS,
        "sampling": benchmark["sampling"],
        "sampling_batch_size": BATCH_SIZE,
        "decode_batch_size": DECODE_BATCH_SIZE,
        "noise_shape": list(NOISE_SHAPE),
        "condition_encoding": "cpu_clip_fp16_roundtrip",
        "tf32": False,
    }
    if len(receipt["records"]) != 256 or any(
        receipt[key] != value for key, value in expected.items()
    ):
        raise ValueError(f"Completed generation does not match this scope: {args.arm}")
    for offset, seed in enumerate(SEEDS):
        noise = benchmark_noise(seed)
        if receipt["seed_noise_sha256"][str(seed)] != tensor_digest(noise):
            raise ValueError(f"Fixed initial noise differs: {args.arm}/{seed}")
        records = receipt["records"][offset * 128 : (offset + 1) * 128]
        for row, point in zip(records, endpoints, strict=True):
            if (
                any(row[key] != value for key, value in point.items())
                or row["seed"] != seed
                or row["noise_sha256"] != tensor_digest(noise[point["noise_index"]][None])
                or hash_file(folder / row["image_path"]) != row["image_sha256"]
            ):
                raise ValueError(
                    f"Incomplete or changed generated endpoint: {args.arm}/{point['id']}"
                )
    print(f"Verified {args.arm}: 256 images, paired noise, 30k EMA", flush=True)


def main() -> None:
    """提供 CPU 预检、CUDA 采样及完成分支核验三个明确入口。

    副作用:
        读取命令行并执行选定的只读权重评测流程。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--output", type=Path, default=OUTPUT / "preflight.json")
    check.set_defaults(action=preflight)
    for name, action in (("generate", generate), ("verify", verify)):
        command = sub.add_parser(name)
        command.add_argument("--arm", choices=ARMS, required=True)
        command.add_argument("--output", type=Path, default=OUTPUT / "generation")
        command.set_defaults(action=action)
    for command in sub.choices.values():
        command.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
