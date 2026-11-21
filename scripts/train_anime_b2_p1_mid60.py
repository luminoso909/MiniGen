"""从 B2 50M patch1 的完整 25k 快照派生 Mid60 分支并核对训练配对。"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from accept_anime_six import check_checkpoint, same_stream_and_rng
from run_anime_b1 import open_anime_cache
from torch import nn
from torch.utils.data import Subset, default_collate
from train_anime_six import ROOT, read_arm_config
from train_anime_six import RUNS as SIX_RUNS

from minigen._typing import Checkpoint, JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.checkpoint import run_identity, save_checkpoint
from minigen.training.config import TrainConfig, build_model
from minigen.training.engine import _contract, train_flow
from minigen.training.runtime import autocast_context, hardware_description, resolve_device
from minigen.training.time_sampling import flow_training_times

ARM = "b2_50m_p1"
START = 25000
FINAL = 30000
MID_PROBABILITY = 0.6
RUNS = Path("artifacts/runs/anime_faces_256/b2_50m_p1_mid60_25k_30k")
SCOPE = Path("configs/evaluation/anime_b2_p1_mid60.json")


class ParentRun(NamedTuple):
    """已核验的 Uniform 父谱系、固定范围及 25k 训练状态。"""

    scope: JsonDict
    directory: Path
    run: JsonDict
    state: Checkpoint


def write_json(
    path: Path,
    report: JsonDict,
) -> None:
    """原子发布一个预检或验收回执。

    Args:
        path: 本轮独立运行目录下的目标文件。
        report: 已整理的运行或验收元数据。

    副作用:
        创建回执目录并替换目标 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_parent(
    root: Path,
) -> ParentRun:
    """读取原 25k 检查点，确认它属于已验收的 Uniform 训练分支。

    Args:
        root: 冻结范围和六组训练资产所在的项目根目录。

    Raises:
        ValueError/AssertionError: 父检查点或既有 30k 对照与冻结身份不符。
        OSError/RuntimeError: 原始回执或检查点无法读取。

    Results:
        ParentRun: 冻结范围、原运行目录、运行回执和完整 CPU 恢复状态。
    """
    scope = json.loads((root / SCOPE).read_text())
    parent_dir = root / SIX_RUNS / ARM
    run = json.loads((parent_dir / "run.json").read_text())
    result = json.loads((parent_dir / "result.json").read_text())
    if (
        scope["schema"] != "minigen.anime_b2_p1_mid60.scope.v1"
        or scope["arm"] != ARM
        or scope["parent_step"] != START
        or scope["final_step"] != FINAL
        or scope["middle_probability"] != MID_PROBABILITY
        or scope["middle_range"] != [0.3, 0.7]
        or root / scope["parent_checkpoint"] != parent_dir / f"step_{START:06d}.pt"
        or root / scope["baseline_checkpoint"] != parent_dir / f"step_{FINAL:06d}.pt"
        or result["hardware"] != scope["hardware"]
        or run["run_id"] != scope["parent_run_id"]
        or run_identity(run["contract"]) != run["run_id"]
        or run["contract"]["source_sha256"] != scope["source_sha256"]
        or run["contract"]["recipe"].get("time_mid_probability") is not None
        or result["run_id"] != run["run_id"]
        or result["attempt_id"] != run["attempt_id"]
        or result["status"] != "completed"
        or result["step"] != FINAL
    ):
        raise ValueError("Uniform parent run differs from the frozen ablation scope")
    parent, metadata = check_checkpoint(parent_dir / f"step_{START:06d}.pt", START, run)
    if (
        metadata["sha256"] != scope["parent_sha256"]
        or hash_file(parent_dir / f"step_{FINAL:06d}.pt") != scope["baseline_sha256"]
    ):
        raise ValueError("Uniform checkpoints differ from the accepted hashes")
    return ParentRun(scope, parent_dir, run, parent)


def preflight(
    parent: ParentRun,
    model: nn.Module,
    train: Subset,
    config: TrainConfig,
    device: torch.device,
) -> JsonDict:
    """用真实缓存微批与父在线权重执行零更新的 CUDA BF16 前向。

    Args:
        parent: 已核验的父训练状态和冻结范围。
        model: 与父快照张量规格相同的速度预测模型。
        train: 与 Uniform 共享的训练缓存子集。
        config: 继承父配置并指定 Mid60 的训练配方。
        device: 已检查硬件型号和精度的 CUDA 设备。

    Raises:
        ValueError/RuntimeError: 真实前向形状、数值或张量规格不合法。

    Results:
        JsonDict: 零更新前向、模型参数量及配方来源记录。

    副作用:
        将模型加载到 CUDA 并设为推理模式。
    """
    batch = default_collate([train[index] for index in range(config.batch_size)])
    latent = batch.latent.float().to(device)
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    cpu_times = flow_training_times(torch.rand(len(latent), generator=rng), MID_PROBABILITY)
    noise = torch.randn(latent.shape, generator=rng).to(device)
    times = cpu_times.to(device)
    noisy, target = flow_interpolate(latent, noise, times)
    condition = Condition(batch.text.float(), batch.mask).to(device)
    model.load_state_dict(parent.state["model"], strict=True)
    model = model.eval().to(device)
    with torch.inference_mode(), autocast_context(device, config.precision):
        prediction = model(noisy, times, condition)
        loss = flow_loss(prediction, target)
    if prediction.shape != target.shape or not torch.isfinite(loss):
        raise ValueError("Mid60 BF16 preflight returned invalid shape or values")
    return {
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "parent_step": START,
        "parent_sha256": parent.scope["parent_sha256"],
        "parent_run_id": parent.run["run_id"],
        "baseline_sha256": parent.scope["baseline_sha256"],
        "hardware": hardware_description(device),
        "torch": str(torch.__version__),
        "precision": config.precision,
        "mid_probability": config.time_mid_probability,
        "optimizer_updates": 0,
        "latent_shape": list(latent.shape),
        "prediction_shape": list(prediction.shape),
        "flow_loss": float(loss),
    }


def run_branch(
    root: Path,
    operation: str,
) -> JsonDict:
    """预检父状态，或在独立目录以严格恢复方式运行本轮分支。

    Args:
        root: 已同步小文件及已有权重和缓存的项目根目录。
        operation: check 做零更新前向，train 恢复并训练至 30k。

    Raises:
        ValueError/RuntimeError: 原配置、冻结包、实际设备或训练结果不匹配。
        OSError: 回执、缓存或检查点不能读取或写入。

    Results:
        JsonDict: 本次实际的预检或训练回执。

    副作用:
        设置 CPU 线程，写入预检或分支检查点，train 运行优化更新。
    """
    base, data = read_arm_config(root, ARM)
    torch.set_num_threads(base.cpu_threads)
    parent = read_parent(root)
    device = resolve_device(base.device, base.precision)
    source, cache, blank, identity = open_anime_cache(root, data)
    train = Subset(cache.split("train"), selected_indices(source, data, "train"))
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    model = build_model(base.model)
    original = parent.run["contract"]
    if (
        _contract(base, identity, model, device, original["initialization"]) != original
        or hardware_description(device) != parent.state["hardware"]
    ):
        raise ValueError("Original recipe, data, frozen package, Torch or GPU differs")
    directory = root / RUNS
    config = replace(base, output_dir=str(directory), time_mid_probability=MID_PROBABILITY)
    if operation == "check":
        report = preflight(parent, model, train, config, device)
        write_json(directory / "preflight.json", report)
        return report

    initialization = {
        "kind": "time_sampling_fork",
        "parent_step": START,
        "parent_checkpoint_sha256": parent.scope["parent_sha256"],
        "parent_run_id": parent.run["run_id"],
        "fork_runner_sha256": hash_file(Path(__file__)),
        "restored_state": ["model", "ema", "optimizer", "scaler", "stream", "rng", "scheduler"],
    }
    contract = _contract(config, identity, model, device, initialization)
    resume = directory / "last.pt"
    if not resume.exists():
        resume = directory / "fork_025000.pt"
        # 只给独立分支改身份元数据，父权重、优化器和随机状态原样继承。
        fork = parent.state | {
            "contract": contract,
            "config": config.to_dict(),
            "stream": parent.state["stream"] | {"data_id": run_identity(contract)},
        }
        save_checkpoint(resume, fork)
        del fork
    del parent
    result = train_flow(
        config,
        train,
        blank,
        identity,
        model=model,
        resume=resume,
        validation=validation,
        initialization=initialization,
    )
    if result["status"] != "completed" or result["step"] != FINAL:
        raise RuntimeError(f"Mid60 stopped before 30k: {result['status']} at {result['step']}")
    return result


def same_state(
    first: Any,
    second: Any,
) -> bool:
    """逐项比较恢复状态中的容器、标量和实际张量值。

    Args:
        first: 原父检查点中要比较的异构状态。
        second: 分支首次保存中相对应的状态。

    Results:
        bool: 状态张量及标量逐项一致时为真。
    """
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return first.dtype == second.dtype and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            same_state(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            same_state(a, b) for a, b in zip(first, second, strict=True)
        )
    return first == second


def accept(
    root: Path,
) -> JsonDict:
    """验收完整 5,000 次更新及其与原 Uniform 的训练前缀、日程和随机流配对。

    Args:
        root: 包含两份分支及父检查点的项目根目录。

    Raises:
        ValueError/AssertionError: 分支轨迹、计数或随机状态未满足对照条件。
        OSError/RuntimeError: 检查点、日志或回执无法读取。

    Results:
        JsonDict: 已发布的可供正式采样验证的无卡回读回执。

    副作用:
        设置 CPU 线程数并写入独立运行目录的 acceptance.json。
    """
    torch.set_num_threads(2)
    parent = read_parent(root)
    directory = root / RUNS
    run = json.loads((directory / "run.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    restored, _ = check_checkpoint(directory / f"step_{START:06d}.pt", START, run)
    for key in ("model", "ema", "optimizer", "scaler", "scheduler"):
        if not same_state(parent.state[key], restored[key]):
            raise ValueError(f"Restored 25k {key} differs from the original parent")
    same_stream_and_rng(parent.state, restored, "restored_mid60_25k")
    del restored
    path = directory / f"step_{FINAL:06d}.pt"
    final, final_metadata = check_checkpoint(path, FINAL, run)
    baseline, baseline_metadata = check_checkpoint(
        parent.directory / f"step_{FINAL:06d}.pt", FINAL, parent.run
    )
    original = parent.run["contract"]
    fork_contract = final["contract"]
    initialization = fork_contract["initialization"]
    expected_config = parent.run["config"] | {
        "output_dir": str(directory),
        "time_mid_probability": MID_PROBABILITY,
    }
    if (
        run["config"] != expected_config
        or fork_contract
        != original
        | {
            "recipe": original["recipe"] | {"time_mid_probability": MID_PROBABILITY},
            "initialization": initialization,
        }
        or initialization["kind"] != "time_sampling_fork"
        or initialization["parent_step"] != START
        or initialization["parent_run_id"] != parent.scope["parent_run_id"]
        or initialization["parent_checkpoint_sha256"] != parent.scope["parent_sha256"]
        or initialization["fork_runner_sha256"] != hash_file(Path(__file__))
        or result["status"] != "completed"
        or result["step"] != FINAL
        or result["attempt_id"] != run["attempt_id"]
        or result["run_id"] != run["run_id"]
        or run_identity(fork_contract) != run["run_id"]
        or final["hardware"] != baseline["hardware"]
        or final["scheduler"] != baseline["scheduler"]
        or final["history"][:START] != parent.state["history"]
        or baseline["history"][:START] != parent.state["history"]
    ):
        raise ValueError("Final states do not share the original 25k parent trajectory")
    same_stream_and_rng(baseline, final, "uniform_vs_mid60_30k")
    tail = final["history"][START:]
    if any(
        any(mid[key] != uniform[key] for key in ("step", "samples", "lr"))
        for mid, uniform in zip(tail, baseline["history"][START:], strict=True)
    ):
        raise ValueError("Mid60 learning rate or sampled batch lengths diverged")
    samples = sum(row["samples"] for row in tail)
    mid_samples = sum(row["mid_time_samples"] for row in tail)
    if abs(mid_samples / samples - MID_PROBABILITY) > 0.01:
        raise ValueError("Observed time samples differ from Mid60")
    report = {
        "schema": "minigen.anime_b2_p1_mid60.acceptance.v1",
        "status": "verified",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "arm": ARM,
        "parent_step": START,
        "step": FINAL,
        "training_updates": FINAL - START,
        "new_updates": FINAL - START,
        "parent_sha256": parent.scope["parent_sha256"],
        "parent_run_id": parent.scope["parent_run_id"],
        "baseline_sha256": baseline_metadata["sha256"],
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": final_metadata["sha256"],
        "run_id": run["run_id"],
        "source_sha256": fork_contract["source_sha256"],
        "data_identity": fork_contract["data"],
        "paired_stream_and_rng": True,
        "parent_online_ema_adamw_scheduler_verified": True,
        "matched_learning_rates_and_batch_lengths": True,
        "shared_history_prefix_steps": START,
        "branch_samples": samples,
        "mid_time_samples": mid_samples,
        "observed_mid_probability": mid_samples / samples,
        "initial_learning_rate": tail[0]["lr"],
        "final_learning_rate": tail[-1]["lr"],
        "branch_seconds": final["elapsed_seconds"] - parent.state["elapsed_seconds"],
        "uniform_validation_flow_mse": baseline_metadata["validation_flow_mse"],
        "mid60_validation_flow_mse": final_metadata["validation_flow_mse"],
    }
    write_json(directory / "acceptance.json", report)
    return report


def main() -> None:
    """提供零更新预检、分支训练及 CPU 验收三个入口。

    副作用:
        读取命令行参数，执行所选操作并输出回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "train", "accept"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result = accept(root) if args.operation == "accept" else run_branch(root, args.operation)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
