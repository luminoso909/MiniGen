"""用固定 B2 patch1 30k EMA 权重发布 CFG 3 和 CFG 5 的配对样图。"""

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import generate_anime_benchmark as benchmark_tools
import torch
from evaluate_anime_time_ablation import tensor_digest
from run_anime_b1 import ROOT, open_anime_cache
from train_anime_six import read_arm_config

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.contracts import FlowModel
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device

SCOPE = Path("configs/evaluation/anime_b2_p1_cfg5.json")
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_p1_cfg5_benchmark")
RECORD_FIELDS = ("id", "seed", "endpoint", "image_path", "prompt", "noise_index", "noise_sha256")


def verify_output(
    target: Path,
    metadata: JsonDict,
    original: JsonDict,
    sampling: JsonDict,
) -> None:
    """逐图核对本轮回执、原题顺序、初噪和实际 PNG 字节。

    Args:
        target: 已发布的实验臂目录。
        metadata: 本轮实验臂与原生成回执的绑定字段。
        original: CFG 3 冻结生成回执。
        sampling: 本轮预期设置，CFG 为 3 或 5。

    Raises:
        OSError: 回执或样图不能读取。
        ValueError: 元数据、样本顺序、采样设置或图像摘要不匹配。
    """
    receipt = json.loads((target / "receipt.json").read_text())
    paired = (
        "checkpoint_sha256",
        "run_id",
        "data_identity",
        "model_config",
        "seeds",
        "seed_noise_sha256",
        "condition_features_sha256",
        "condition_mask_sha256",
        "evaluation_torch",
    )
    if (
        any(receipt[key] != value for key, value in metadata.items())
        or any(receipt[key] != original[key] for key in paired)
        or receipt["sampling"] != sampling
        or len(receipt["records"]) != 256
    ):
        raise ValueError(f"Published receipt is outside the comparison scope: {target}")
    for image, old in zip(receipt["records"], original["records"], strict=True):
        if (
            any(image[key] != old[key] for key in RECORD_FIELDS)
            or hash_file(target / image["image_path"]) != image["image_sha256"]
            or (sampling["cfg"] == 3 and image["image_sha256"] != old["image_sha256"])
        ):
            raise ValueError(f"Invalid paired image: {target / image['image_path']}")


def copy_cfg3(
    original_path: Path,
    target: Path,
    original: JsonDict,
    metadata: JsonDict,
) -> None:
    """从原始生成回执逐字节复用 CFG 3 图片并原子发布。

    Args:
        original_path: 既有 CFG 3 原始生成回执路径。
        target: 本次实验的 CFG 3 输出目录。
        original: 已锁的既有生成回执。
        metadata: 本轮实验臂与原生成回执的绑定字段。

    Raises:
        OSError: PNG 或回执无法读写。
        ValueError: 既有 CFG 3 图片与锁定的回执不匹配。

    副作用:
        将已回读的图像和本轮引用回执发布到独立目录。
    """
    with TemporaryDirectory(prefix=".cfg3-", dir=target.parent) as temporary:
        pending = Path(temporary) / target.name
        pending.mkdir()
        for record in original["records"]:
            image = original_path.parent / record["image_path"]
            if hash_file(image) != record["image_sha256"]:
                raise ValueError(f"Original CFG 3 image changed: {image}")
            shutil.copyfile(image, pending / image.name)
        (pending / "receipt.json").write_text(
            json.dumps(original | metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        pending.rename(target)


def load_ema(
    snapshot: Path,
    original: JsonDict,
    architecture: JsonDict,
) -> FlowModel:
    """核对训练终点契约并只还原冻结推理模型的 EMA 权重。

    Args:
        snapshot: 原实验臂的 30k 编号快照。
        original: 已锁 CFG 3 回执，绑定训练身份和检查点摘要。
        architecture: 原 B2 patch1 配置。

    Raises:
        OSError: 权重文件无法读取。
        ValueError: 检查点、模型规格或训练契约不符。
        RuntimeError: EMA 张量与模型结构不符。

    Results:
        FlowModel: 仅用于推理的 CPU EMA 模型。
    """
    if hash_file(snapshot) != original["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint digest changed: {snapshot}")
    state = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["step"] != 30000
        or state["objective"] != "flow_velocity"
        or state["config"]["model"] != architecture
        or state["data_identity"] != original["data_identity"]
        or state["contract"]["source_sha256"] != original["source_sha256"]
        or run_identity(state["contract"]) != original["run_id"]
    ):
        raise ValueError(f"Checkpoint content differs from the CFG 3 receipt: {snapshot}")
    with torch.device("meta"):
        model = build_model(architecture)
    model.load_state_dict(state["ema"], strict=True, assign=True)
    model.eval().requires_grad_(False)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != original["trainable_parameters"]
    ):
        raise ValueError(f"EMA parameter count changed: {snapshot}")
    return model


@torch.inference_mode()
def generate(
    root: Path,
    output: Path,
) -> None:
    """只改引导系数，分三组完成固定题库的 768 张新图并验收全部配对。

    Args:
        root: 项目、冻结源码及正式缓存的根目录。
        output: 本轮独立输出目录，含 generation 下的六条实验臂。

    Raises:
        OSError: 锁定资产、旧图或本轮输出无法读写。
        ValueError: 缓存、题库、运行协议、检查点或配对信息不符。
        RuntimeError: CUDA、EMA 装载或采样失败。

    副作用:
        调整本进程的 PyTorch CUDA 数值设置，读取锁定身份与模型，
        发布复用的 CFG 3 图和本轮新生成的 CFG 5 图及回执。
    """
    scope_path = root / SCOPE
    scope = json.loads(scope_path.read_text())
    scope_sha = hash_file(scope_path)
    benchmark, exclusions, protocol = benchmark_tools.read_protocol(root)
    sampling = benchmark["sampling"]
    if (
        scope["schema"] != "minigen.anime_faces.b2_p1_cfg5_scope.v1"
        or scope["seeds"] != benchmark_tools.SEEDS
        or scope["final_step"] != 30000
        or any(scope[key] != protocol[key] for key in ("benchmark_sha256", "rubric_sha256"))
    ):
        raise ValueError("This CFG comparison does not match the locked benchmark")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device("cuda", "fp32")
    config, data = read_arm_config(root, "b2_50m_p1")
    source, _, blank, data_identity = open_anime_cache(root, data)
    benchmark_tools.validate_sources(benchmark, exclusions, source, data_identity)
    endpoints = benchmark_tools.plan_endpoints(benchmark)
    condition = benchmark_tools.encode_benchmark_conditions(
        root, root / data["clip_lock"], endpoints
    )
    package = Path(inspect.getfile(build_model)).resolve().parents[1]
    source_digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        source_digest.update(str(file.relative_to(package)).encode() + b"\0" + file.read_bytes())
    inference_source = source_digest.hexdigest()
    helper_sha = hash_file(Path(benchmark_tools.__file__))
    actual = protocol | {
        "schema": "minigen.anime_faces.benchmark_generation.v1",
        "status": "completed",
        "checkpoint_step": 30000,
        "weights": "ema",
        "seeds": scope["seeds"],
        "sampler_helper_sha256": helper_sha,
        "evaluation_torch": str(torch.__version__),
        "condition_encoding": "cpu_clip_fp16_roundtrip",
        "condition_features_sha256": tensor_digest(condition.features),
        "condition_mask_sha256": tensor_digest(condition.mask),
        "noise_shape": list(benchmark_tools.NOISE_SHAPE),
        "sampling_batch_size": benchmark_tools.BATCH_SIZE,
        "decode_batch_size": benchmark_tools.DECODE_BATCH_SIZE,
        "tf32": False,
        "sampling": sampling,
        "data_identity": data_identity,
        "model_config": config.model,
        "seed_noise_sha256": {
            str(seed): tensor_digest(benchmark_tools.benchmark_noise(seed))
            for seed in scope["seeds"]
        },
    }
    originals = {}
    for variant, spec in scope["variants"].items():
        original_path = root / spec["original_receipt"]
        original = json.loads(original_path.read_text())
        if (
            hash_file(original_path) != spec["original_receipt_sha256"]
            or original["arm"] != spec["original_arm"]
            or original["checkpoint_sha256"] != spec["checkpoint_sha256"]
            or inference_source != spec["inference_source_sha256"]
            or original.get("inference_source_sha256", original["source_sha256"])
            != inference_source
            or any(original[key] != value for key, value in actual.items())
            or len(original["records"]) != 256
        ):
            raise ValueError(f"Original CFG 3 receipt does not match the scope: {original_path}")
        originals[variant] = (original_path, original)

    generation = output / "generation"
    generation.mkdir(parents=True, exist_ok=True)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    condition, blank = condition.to(device), blank.to(device)
    for arm, spec in scope["candidates"].items():
        variant = scope["variants"][spec["variant"]]
        original_path, original = originals[spec["variant"]]
        target = generation / arm
        cfg = spec["cfg"]
        paired_sampling = sampling | {"cfg": cfg}
        metadata = {
            "arm": arm,
            "comparison_scope_sha256": scope_sha,
            "original_receipt_sha256": variant["original_receipt_sha256"],
            "original_arm": variant["original_arm"],
            "inference_source_sha256": inference_source,
            "training_updates": 0,
        }
        if not target.exists() and cfg == 3:
            copy_cfg3(original_path, target, original, metadata)
        if not target.exists() and cfg == 5:
            started = perf_counter()
            model = load_ema(root / variant["checkpoint"], original, config.model).to(device)
            # 旧入口的内容摘要属于历史锁。在此进程中，仅为新图覆盖采样系数。
            old_guidance = benchmark_tools.GUIDANCE
            benchmark_tools.GUIDANCE = float(cfg)
            try:
                benchmark_tools.publish_benchmark_arm(
                    target,
                    model,
                    vae,
                    condition,
                    blank,
                    endpoints,
                    paired_sampling,
                    original | metadata | {"sampler_sha256": hash_file(Path(__file__))},
                    started,
                )
            finally:
                benchmark_tools.GUIDANCE = old_guidance
            del model
        verify_output(target, metadata, original, paired_sampling)
    print("Verified 1,536 PNGs with matched checkpoints, endpoints and initial noise.", flush=True)


def main() -> None:
    """解析根目录和输出目录，执行不含参数更新的 CUDA 采样。

    副作用:
        按锁定配方将六组配对样图和回执写到独立输出目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    generate(args.root.resolve(), args.root.resolve() / args.output)


if __name__ == "__main__":
    main()
