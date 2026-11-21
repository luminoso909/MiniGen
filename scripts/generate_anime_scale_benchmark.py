"""为动漫三模型评测执行无卡预检或生成一个 50M 实验臂。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
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
from run_anime_b1 import CONFIG, ROOT, open_anime_cache

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import SDXL_FILES, FrozenVAE, hash_file
from minigen.commands import read_experiment
from minigen.contracts import Condition, FlowModel
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, FLOW_OBJECTIVE, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device

SCALE_ARMS = ("b1_50m_20k", "b2_50m_20k")
SCOPE = Path("configs/benchmark/anime_faces_v1_scale_50m.json")
BASELINE_RECEIPT = Path(
    "artifacts/exports/anime-faces-b1/uniform_continuation_20000/generation/uniform20k/receipt.json"
)
OUTPUT = Path("artifacts/exports/anime-faces-b1/scale_50m_benchmark")
STEP = 20000


def read_scale_scope(
    root: Path,
    baseline_path: Path,
) -> tuple[JsonDict, JsonDict, list[JsonDict], JsonDict, JsonDict]:
    """读取独立三组预登记及原题库，核对原 30M 基线回执的身份和每个初噪行。

    Args:
        root: 项目根目录，包含原题库内容锁和当前三组范围。
        baseline_path: 已验收的 30M 正式生成回执，以其哈希核验历史文件。

    Raises:
        OSError: 预登记、题库或历史回执文件不可读。
        ValueError: 预登记范围、基线身份、端点顺序或初噪哈希不符。

    Results:
        tuple: 三组范围、原题库、固定顺序端点、历史基线与当前生成身份。
    """
    scope_path = root / SCOPE
    scope = json.loads(scope_path.read_text())
    benchmark, exclusions, protocol = read_protocol(root)
    baseline_sha256 = hash_file(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    if (
        scope["schema"] != "minigen.anime_faces.scale_50m_scope.v1"
        or set(scope["candidates"]) != set(SCALE_ARMS)
        or hash_file(root / scope["plan"]) != scope["plan_sha256"]
        or baseline_sha256 != scope["baseline"]["receipt_sha256"]
        or baseline["schema"] != "minigen.anime_faces.benchmark_generation.v1"
        or baseline["status"] != "completed"
        or baseline["arm"] != scope["baseline"]["arm"]
        or baseline["checkpoint_sha256"] != scope["baseline"]["checkpoint_sha256"]
        or baseline["cumulative_training_updates"] != STEP
        or baseline["sampling"] != benchmark["sampling"]
        or baseline["seeds"] != SEEDS
        or baseline["noise_shape"] != list(NOISE_SHAPE)
        or baseline["condition_encoding"] != "cpu_clip_fp16_roundtrip"
        or baseline["sampling_batch_size"] != BATCH_SIZE
        or baseline["decode_batch_size"] != DECODE_BATCH_SIZE
        or baseline["tf32"] is not False
        or any(baseline[key] != value for key, value in protocol.items())
        or any(scope[key] != protocol[key] for key in ("benchmark_sha256", "rubric_sha256"))
    ):
        raise ValueError("Scale benchmark scope and historical 30M generation do not match")
    endpoints = plan_endpoints(benchmark)
    verified_rows = []
    for seed in SEEDS:
        noise = benchmark_noise(seed)
        if tensor_digest(noise) != baseline["seed_noise_sha256"][str(seed)]:
            raise ValueError(f"Fixed seed tensor differs from 30M baseline: {seed}")
        rows = [tensor_digest(row[None]) for row in noise]
        verified_rows.extend(
            endpoint | {"seed": seed, "noise_sha256": rows[endpoint["noise_index"]]}
            for endpoint in endpoints
        )
    for expected, recorded in zip(verified_rows, baseline["records"], strict=True):
        if any(recorded[key] != value for key, value in expected.items()):
            raise ValueError(f"Fixed endpoint differs from 30M baseline: {expected['id']}")
    return (
        scope,
        benchmark | {"source_exclusions": exclusions},
        endpoints,
        baseline,
        protocol
        | {
            "comparison": scope["comparison"],
            "comparison_scope_sha256": hash_file(scope_path),
            "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
            "baseline_generation_receipt_sha256": baseline_sha256,
        },
    )


def open_scale_cache(
    root: Path,
    benchmark: JsonDict,
    baseline: JsonDict,
) -> tuple[JsonDict, Condition]:
    """验证实际缓存来源、排除项及验证子集，复用与 30M 相同的空条件。

    Args:
        root: 训练资产所在项目根目录。
        benchmark: 原固定题库及其已用来源排除项。
        baseline: 已核验的既有 30M 生成回执。

    Raises:
        OSError: 缓存或组件锁不可读。
        ValueError: 缓存、来源组或训练与验证子集身份不符。

    Results:
        tuple: 共同的数据设置、[1,n,d] 空条件及布尔掩码。
    """
    _, data = read_experiment(root, CONFIG)
    source, _, blank, identity = open_anime_cache(root, data)
    validate_sources(benchmark, benchmark["source_exclusions"], source, identity)
    if identity != baseline["data_identity"]:
        raise ValueError("Scale benchmark data differs from the 30M baseline cache")
    return data, blank


def read_scale_checkpoint(
    root: Path,
    arm: str,
    spec: JsonDict,
    baseline: JsonDict,
) -> tuple[FlowModel, JsonDict]:
    """只读映射编号检查点，用 meta 构造与 assign 加载准确的 EMA 网络。

    Args:
        root: 训练配方、检查点与其回执的项目根目录。
        arm: 本次预登记的候选实验臂名称。
        spec: 当前候选的固定配置、检查点、运行标识和参数量。
        baseline: 固定 30M 基线，定义实验使用的共同数据身份。

    Raises:
        OSError: 训练产物或配置不可读。
        ValueError/RuntimeError: 权重、训练身份、结构或 EMA 张量不匹配。

    Results:
        tuple: CPU 上只含映射 EMA 的速度网络，与可公开的候选生成回执知识字段。
    """
    path = root / spec["checkpoint"]
    checkpoint_sha256 = hash_file(path)
    if checkpoint_sha256 != spec["checkpoint_sha256"]:
        raise ValueError(f"Scale checkpoint does not match the accepted snapshot: {arm}")
    result_path = path.with_name("result.json")
    result = json.loads(result_path.read_text())
    config, _ = read_experiment(root, Path(spec["config"]))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = state["contract"]
    expected = {
        "objective": FLOW_OBJECTIVE,
        "architecture": spec["architecture"],
        "run_id": spec["run_id"],
        "model_config": config.model,
        "data_identity": baseline["data_identity"],
        "source_sha256": contract["source_sha256"],
        "trainable_parameters": spec["parameters"],
    }
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["step"] != STEP
        or state["scheduler"]["step"] != STEP
        or state["objective"] != FLOW_OBJECTIVE
        or state["config"]["model"] != config.model
        or state["data_identity"] != expected["data_identity"]
        or contract["data"] != expected["data_identity"]
        or run_identity(contract) != spec["run_id"]
        or result["status"] != "completed"
        or result["step"] != STEP
        or result["performance"]["optimizer_steps"] != STEP
        or any(result[key] != value for key, value in expected.items())
    ):
        raise ValueError(f"Scale checkpoint and completed training receipt differ: {arm}")
    weights = state["ema"]
    with torch.device("meta"):
        model = build_model(config.model)
    layout = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    if (
        sum(param.numel() for param in model.parameters()) != spec["parameters"]
        or {key: (value.shape, value.dtype) for key, value in weights.items()} != layout
        or any(
            value.is_floating_point() and value.dtype != torch.float32 for value in weights.values()
        )
    ):
        raise ValueError(f"Scale EMA shape, dtype or parameter count differs: {arm}")
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, expected | {
        "arm": arm,
        "objective_source": "explicit",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": STEP,
        "cumulative_training_updates": STEP,
        "training_result_sha256": hash_file(result_path),
    }


def verify_vae_files(
    root: Path,
    lock_path: Path,
) -> JsonDict:
    """逐个核对 VAE 本地文件及解码规格，预检阶段不实例化整个 VAE。

    Args:
        root: 解析本地模型目录的项目根目录。
        lock_path: 与训练缓存身份一致的 VAE 锁路径。

    Raises:
        OSError: 冻结组件文件不可读。
        ValueError: 文件集合、摘要、路径或解码配置不符。

    Results:
        JsonDict: 本次实际核对的组件锁与文件摘要。
    """
    lock = json.loads(lock_path.read_text())
    directory = (root / lock["local_path"]).resolve()
    if set(lock["files"]) != SDXL_FILES or not directory.is_relative_to(root):
        raise ValueError("Preflight requires the locked local SDXL VAE")
    hashes = {}
    for name, entry in lock["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Locked VAE file changed: {name}")
        actual = hash_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Locked VAE file changed: {name}")
        hashes[name] = actual
    config = json.loads((directory / "config.json").read_text())
    if (
        config["latent_channels"] != 4
        or len(config["block_out_channels"]) != 4
        or config["scaling_factor"] != lock["scaling_factor"]
    ):
        raise ValueError("VAE compression or scaling does not match the cache lock")
    return {"mode": "files_and_config", "lock_sha256": hash_file(lock_path), "files": hashes}


@torch.inference_mode()
def preflight(
    args: argparse.Namespace,
) -> None:
    """在单线程 CPU 上验收两组 EMA、原题条件和先前基线的固定初噪。

    Args:
        args: 项目根目录、历史生成回执和验收输出文件。

    Raises:
        OSError: 输入产物不可读或验收回执无法写入。
        ValueError/RuntimeError: 条件、检查点或真实模型前向不符合冻结协议。

    副作用:
        设置本进程的计算线程数，写入独立的预检回执，不写正式采样图。
    """
    root = args.root.resolve()
    torch.set_num_threads(1)
    scope, benchmark, endpoints, baseline, metadata = read_scale_scope(
        root, root / args.baseline_receipt
    )
    data, _ = open_scale_cache(root, benchmark, baseline)
    clip_lock = root / data["clip_lock"]
    condition = encode_benchmark_conditions(root, clip_lock, endpoints)
    if not torch.isfinite(condition.features).all():
        raise ValueError("CLIP produced nonfinite benchmark conditions")
    vae_files = verify_vae_files(root, root / data["vae_lock"])
    first_condition = Condition(condition.features[:1], condition.mask[:1])
    latent = benchmark_noise(SEEDS[0])[:1]
    time = torch.full((1,), 0.5, dtype=torch.float32)
    arms = {}
    for arm in SCALE_ARMS:
        model, checkpoint = read_scale_checkpoint(root, arm, scope["candidates"][arm], baseline)
        velocity = model(latent, time, first_condition)
        if velocity.shape != latent.shape or not torch.isfinite(velocity).all():
            raise ValueError(f"Scale checkpoint CPU forward differs from velocity contract: {arm}")
        arms[arm] = checkpoint | {
            "cpu_forward": {"shape": list(velocity.shape), "output_sha256": tensor_digest(velocity)}
        }
        del model
    report = metadata | {
        "schema": "minigen.anime_faces.scale_50m_cpu_preflight.v1",
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
        "seeds": SEEDS,
        "seed_noise_sha256": baseline["seed_noise_sha256"],
        "noise_shape": list(NOISE_SHAPE),
        "checked_baseline_endpoints": len(baseline["records"]),
        "clip": {
            "lock_sha256": hash_file(clip_lock),
            "encoded_endpoints": len(endpoints),
            "shape": list(condition.features.shape),
            "features_sha256": tensor_digest(condition.features),
            "mask_sha256": tensor_digest(condition.mask),
        },
        "vae": vae_files,
        "arms": arms,
        "formal_images_generated": 0,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".preflight-", dir=output.parent) as temp:
        pending = Path(temp) / "preflight.json"
        pending.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        pending.replace(output)
    print(
        f"CPU preflight passed: 2 EMA forwards, {len(baseline['records'])} matched rows: {output}"
    )


@torch.inference_mode()
def generate_scale_arm(
    args: argparse.Namespace,
) -> None:
    """在正式 CUDA 设置下生成一个候选臂，并沿用原题库的采样和验收流程。

    Args:
        args: 项目根目录、历史生成回执、候选实验臂和输出根目录。

    Raises:
        OSError: 检查点、组件或产物不可读取，或目标目录已存在。
        ValueError/RuntimeError: 目录身份、CUDA 设备、采样协议或运行结果不合法。

    副作用:
        设置本进程的 CUDA 精度和 CPU 线程，写入该实验臂的全部 PNG 及回执。
    """
    started = perf_counter()
    root = args.root.resolve()
    target = root / args.output / args.arm
    if target.exists():
        raise FileExistsError(target)
    device = resolve_device("cuda", "fp32")
    config, _ = read_experiment(root, CONFIG)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scope, benchmark, endpoints, baseline, metadata = read_scale_scope(
        root, root / args.baseline_receipt
    )
    if str(torch.__version__) != baseline["evaluation_torch"]:
        raise ValueError("CUDA benchmark must use the same Torch release as 30M generation")
    data, blank = open_scale_cache(root, benchmark, baseline)
    condition = encode_benchmark_conditions(root, root / data["clip_lock"], endpoints).to(device)
    model, checkpoint = read_scale_checkpoint(
        root, args.arm, scope["candidates"][args.arm], baseline
    )
    model = model.to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    publish_benchmark_arm(
        target,
        model,
        vae,
        condition,
        blank.to(device),
        endpoints,
        benchmark["sampling"],
        checkpoint | metadata,
        started,
    )


def main() -> None:
    """解析无卡预检和 CUDA 正式采样的独立入口。

    副作用:
        从命令行读取路径及实验臂并执行选中的流程。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="核对冻结输入并逐个执行一次 CPU EMA 前向")
    check.add_argument("--output", type=Path, default=OUTPUT / "preflight.json")
    check.set_defaults(action=preflight)
    generate = sub.add_parser("generate", help="在 CUDA 正式生成一个预登记的候选臂")
    generate.add_argument("--output", type=Path, default=OUTPUT / "generation")
    generate.add_argument("--arm", choices=SCALE_ARMS, required=True)
    generate.set_defaults(action=generate_scale_arm)
    for command in (check, generate):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--baseline-receipt", type=Path, default=BASELINE_RECEIPT)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
