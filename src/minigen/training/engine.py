"""单卡 Latent Flow 训练闭环；冻结编码缓存通过数据接口注入。"""

import copy
import fcntl
import hashlib
import json
import math
import random
import resource
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset, default_collate

from minigen._typing import Checkpoint, JsonDict
from minigen.contracts import Condition
from minigen.data.representation_cache import Representation
from minigen.data.stateful import StatefulBatchStream
from minigen.objectives.flow import drop_condition, flow_interpolate, flow_loss
from minigen.training.checkpoint import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    restore_rng,
    rng_state,
    run_identity,
    save_checkpoint,
)
from minigen.training.config import TrainConfig, build_model, learning_rate_for_step
from minigen.training.runtime import (
    autocast_context,
    hardware_description,
    resolve_device,
    synchronize,
)
from minigen.training.time_sampling import flow_training_times

MicrobatchLoss = Callable[
    [
        nn.Module,
        Float[Tensor, "b c_z h_z w_z"],
        Float[Tensor, " b"],
        Condition,
        Float[Tensor, "b c_z h_z w_z"],
        Sequence[str],
    ],
    tuple[Float[Tensor, ""], Float[Tensor, ""]],
]


def _json_write(
    path: Path,
    payload: JsonDict,
) -> None:
    """原子更新小型运行元数据，拒绝 NaN 等非标准 JSON。

    Args:
        path: 输出 JSON 路径。
        payload: 可序列化的有限数值与元数据。

    Raises:
        OSError/ValueError/TypeError: 写入或 JSON 编码失败。

    副作用:
        写入临时文件并替换目标。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _contract(
    config: TrainConfig,
    identity: JsonDict,
    model: nn.Module,
    device: torch.device,
    initialization: JsonDict | None,
    auxiliary_module: nn.Module | None = None,
) -> JsonDict:
    """冻结会影响数值轨迹的配置、数据身份和模型张量规格。

    Args:
        config: 完整训练配置。
        identity: 调用方传入的数据、缓存和子集身份。
        model: 实际注入的模型。
        device: 实际训练设备。
        initialization: 显式权重初始化的来源，默认随机初始化。
        auxiliary_module: 仅在训练使用的独立模块，其张量规格纳入恢复身份。

    Results:
        JsonDict: 不含输出路径及可延长停止上限的兼容性契约。
    """
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(source.rglob("*.py")):
        digest.update(str(file.relative_to(source)).encode() + b"\0" + file.read_bytes())
    recipe = config.to_dict()
    for key in ("output_dir", "max_steps", "max_seconds"):
        recipe.pop(key)
    if recipe["time_mid_probability"] is None:
        recipe.pop("time_mid_probability")
    contract: JsonDict = {
        "source_sha256": digest.hexdigest(),
        "recipe": recipe,
        "data": identity,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tensors": {
            key: [list(value.shape), str(value.dtype)] for key, value in model.state_dict().items()
        },
        "torch": str(torch.__version__),
        "device": str(device),
        **({"initialization": initialization} if initialization is not None else {}),
    }
    if auxiliary_module is not None:
        contract["auxiliary_tensors"] = {
            key: [list(value.shape), str(value.dtype)]
            for key, value in auxiliary_module.state_dict().items()
        }
    # JSON 规范化让 YAML 中的 list 与 dataclass 默认 tuple 使用同一身份。
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    dataset: Dataset[Representation],
    config: TrainConfig,
    device: torch.device,
) -> float:
    """用独立固定噪声和时间测量验证损失，不消耗训练 RNG。

    Args:
        model: 待评估的速度预测网络。
        dataset: 调用方明确提供的验证集；不由本函数寻找测试集。
        config: 批大小、种子和计算精度。
        device: 已验证的执行设备。

    Raises:
        ValueError: 验证集为空或结果非有限。

    Results:
        float: 按所有图像元素求平均的 Flow 噪声路径预测损失。

    副作用:
        临时切换模型为 eval，结束后恢复原 train/eval 状态。
    """
    if len(dataset) == 0:
        raise ValueError("Validation dataset must be nonempty")
    generator = torch.Generator().manual_seed(config.seed + 10000)
    training = model.training
    model.eval()
    total = 0.0
    try:
        for start in range(0, len(dataset), config.batch_size):
            batch = default_collate(
                [dataset[i] for i in range(start, min(start + config.batch_size, len(dataset)))]
            )
            data = batch.latent.detach().float().to(device)
            noise = torch.randn(data.shape, generator=generator).to(device)
            times = torch.rand(len(data), generator=generator).to(device)
            zt, target = flow_interpolate(data, noise, times)
            condition = Condition(batch.text.detach(), batch.mask).to(device)
            with autocast_context(device, config.precision):
                loss = flow_loss(model(zt, times, condition), target)
            total += float(loss) * len(data)
    finally:
        model.train(training)
    result = total / len(dataset)
    if not math.isfinite(result):
        raise ValueError("Nonfinite validation loss")
    return result


def _reconcile_history(
    directory: Path,
    history: list[JsonDict],
) -> None:
    """从检查点恢复权威历史，剔除上次中断时尚未提交的日志行。

    Args:
        directory: 独立运行目录。
        history: 检查点内的完整更新历史，编号必须从 1 连续。

    Raises:
        ValueError: 历史行编号不连续。
        OSError: 日志写入失败。

    副作用:
        原子替换 metrics.jsonl。
    """
    if [row["step"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Checkpoint history is not contiguous")
    temporary = directory / "metrics.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    temporary.replace(directory / "metrics.jsonl")


def train_flow(
    config: TrainConfig,
    dataset: Dataset[Representation],
    blank: Condition,
    data_identity: JsonDict,
    *,
    model: nn.Module | None = None,
    resume: Path | None = None,
    validation: Dataset[Representation] | None = None,
    initialization: JsonDict | None = None,
    microbatch_loss: MicrobatchLoss | None = None,
    auxiliary_module: nn.Module | None = None,
) -> JsonDict:
    """对注入的缓存做有步数和时间边界的单卡 Flow 训练。

    Args:
        config: 模型、优化和运行目录配置。
        dataset: 可确定索引的训练表示；包含 sample_id、latent、text 和 mask。
        blank: 同一编码器的单条独立空条件，用于条件丢弃。
        data_identity: 已验收缓存及子集的身份，恢复时逐字段一致。
        model: 显式注入的速度预测模型；省略时由配置构造 B0/B1/B2。
        resume: 同设备兼容检查点，允许延长停止上限。
        validation: 可选验证表示；使用独立固定噪声和时间。
        initialization: 显式传入模型的初始化来源，纳入恢复身份。
        microbatch_loss: 可选辅助目标，返回未加权的 Flow 和已加权的辅助标量。
            调用方在 initialization 记录辅助目标及数据身份。
        auxiliary_module: 辅助目标独用的可训练模块，使用独立的 AdamW 并单独保存。

    Raises:
        ValueError: 数据、设备、恢复身份或运行目录不兼容。
        BlockingIOError: 同一输出目录已有训练进程。
        OSError/RuntimeError: 文件、模型、非有限梯度或底层执行失败。

    Results:
        JsonDict: 实际更新次数、停止原因、设备、计时和产物路径。

    副作用:
        设置进程线程数和随机状态，更新模型、EMA、优化器与数据游标，写日志和检查点。
        恢复原模型 train/eval 状态，但保留学习后的参数。
    """
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(
            config,
            dataset,
            blank,
            data_identity,
            model,
            resume,
            validation,
            initialization,
            microbatch_loss,
            auxiliary_module,
        )


def _train(
    config: TrainConfig,
    dataset: Dataset[Representation],
    blank: Condition,
    data_identity: JsonDict,
    model: nn.Module | None,
    resume: Path | None,
    validation: Dataset[Representation] | None,
    initialization: JsonDict | None,
    microbatch_loss: MicrobatchLoss | None,
    auxiliary_module: nn.Module | None,
) -> JsonDict:
    """在独占运行目录中执行训练，检查点只在完整优化更新后保存。

    Args:
        config: 已校验的训练配置。
        dataset: 确定性训练数据。
        blank: 一条空条件。
        data_identity: 数据和所选行的版本身份。
        model: 可选显式模型对象。
        resume: 可选恢复路径。
        validation: 可选固定验证集。
        initialization: 显式权重初始化的来源。
        microbatch_loss: 可选逐微批 Flow 和辅助损失计算。
        auxiliary_module: 可选仅训练时更新的辅助模块，不纳入主模型裁剪或 EMA。

    Raises:
        ValueError: 身份、检查点、样本或损失不合法。
        OSError/RuntimeError: 训练和持久化失败。

    Results:
        JsonDict: 保存到 result.json 的实际运行摘要。

    副作用:
        更新模型、优化状态及随机数，写入独立目录；身份核验后的训练失败另存回执。
    """
    device = resolve_device(config.device, config.precision)
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    directory = Path(config.output_dir)
    if resume is None and any((directory / name).exists() for name in ("last.pt", "metrics.jsonl")):
        raise ValueError("Run directory already contains training; use explicit resume")
    model = (model if model is not None else build_model(config.model)).to(device)
    original_mode = model.training
    contract = _contract(config, data_identity, model, device, initialization, auxiliary_module)
    identity_digest = run_identity(contract)
    stream = StatefulBatchStream(
        dataset, config.batch_size * config.grad_accum, data_id=identity_digest, seed=config.seed
    )
    blank = blank.to(device)
    if len(blank.features) != 1:
        raise ValueError("Training requires one independent blank condition")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    auxiliary = None
    if auxiliary_module is not None:
        auxiliary_module.to(device).train()
        auxiliary_optimizer = torch.optim.AdamW(
            auxiliary_module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        auxiliary = (auxiliary_module, auxiliary_optimizer)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    history: list[JsonDict] = []
    step = 0
    elapsed_before = 0.0
    if resume is not None:
        state = load_checkpoint(resume)
        if state["contract"] != contract:
            raise ValueError(
                "Checkpoint recipe/data/model/runtime mismatch; exact resume needs the same device"
            )
        if (
            config.max_steps < state["config"]["max_steps"]
            or config.max_seconds < state["config"]["max_seconds"]
        ):
            raise ValueError("Resume may only extend stopping limits")
        step = state["step"]
        history = state["history"]
        if step != len(history) or state["scheduler"]["step"] != step:
            raise ValueError("Checkpoint update/history/scheduler mismatch")
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if auxiliary is not None:
            auxiliary[0].load_state_dict(state["auxiliary"]["model"], strict=True)
            auxiliary[1].load_state_dict(state["auxiliary"]["optimizer"])
        scaler.load_state_dict(state["scaler"])
        stream.load_state_dict(state["stream"])
        elapsed_before = state["elapsed_seconds"]
        restore_rng(state["rng"], device)
    if (directory / "run.json").exists() and json.loads((directory / "run.json").read_text())[
        "contract"
    ] != contract:
        raise ValueError("Output directory belongs to another run identity")
    attempt_id = uuid4().hex
    started_at = datetime.now(UTC).isoformat()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    _json_write(
        directory / "run.json",
        {
            "schema": "minigen.run.v1",
            "hardware": hardware_description(device),
            "run_id": identity_digest,
            "attempt_id": attempt_id,
            "started_at": started_at,
            "objective": config.objective,
            "architecture": config.model.get("architecture", "b0"),
            "contract": contract,
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "config": config.to_dict(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "resume": str(resume) if resume else None,
        },
    )
    _reconcile_history(directory, history)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    reason = "completed"
    try:
        _save(
            directory,
            config,
            contract,
            model,
            ema,
            optimizer,
            scaler,
            stream,
            history,
            step,
            device,
            elapsed_before,
            snapshot=step in config.checkpoint_steps,
            auxiliary=auxiliary,
        )
        while step < config.max_steps:
            if perf_counter() - started >= config.max_seconds:
                reason = "time_limit"
                break
            update_start = perf_counter()
            batch = next(stream)
            data_seconds = perf_counter() - update_start
            data = batch.latent.detach().float().to(device)
            if not torch.isfinite(data).all():
                raise ValueError("Nonfinite cached latent")
            # 每次更新先生成整个有效 batch 的随机量；微批划分不改变抽样顺序。
            noise = torch.randn(data.shape).to(device)
            cpu_times = flow_training_times(torch.rand(len(data)), config.time_mid_probability)
            times = cpu_times.to(device)
            dropped = (torch.rand(len(data)) < config.condition_dropout).to(device)
            condition = drop_condition(
                Condition(batch.text.detach(), batch.mask).to(device), blank, dropped
            )
            zt, target = flow_interpolate(data, noise, times)
            lr = learning_rate_for_step(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if auxiliary is not None:
                for group in auxiliary[1].param_groups:
                    group["lr"] = lr
                auxiliary[1].zero_grad(set_to_none=True)
            total_loss = 0.0
            total_aux = 0.0
            for start in range(0, len(data), config.batch_size):
                stop = min(start + config.batch_size, len(data))
                micro = Condition(condition.features[start:stop], condition.mask[start:stop])
                with autocast_context(device, config.precision):
                    if microbatch_loss is None:
                        loss = flow_loss(
                            model(zt[start:stop], times[start:stop], micro), target[start:stop]
                        )
                    else:
                        flow, aux = microbatch_loss(
                            model,
                            zt[start:stop],
                            times[start:stop],
                            micro,
                            target[start:stop],
                            batch.sample_id[start:stop],
                        )
                        loss = flow + aux
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                weighted = loss * (stop - start) / len(data)
                total_loss += float(weighted.detach())
                if microbatch_loss is not None:
                    total_aux += float(aux.detach()) * (stop - start) / len(data)
                scaler.scale(weighted).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            if auxiliary is not None:
                scaler.unscale_(auxiliary[1])
                torch.nn.utils.clip_grad_norm_(
                    auxiliary[0].parameters(), config.grad_clip, error_if_nonfinite=True
                )
            scaler.step(optimizer)
            if auxiliary is not None:
                scaler.step(auxiliary[1])
            scaler.update()
            with torch.no_grad():
                for avg, current in zip(ema.parameters(), model.parameters(), strict=True):
                    avg.lerp_(current, 1 - config.ema_decay)
                for avg, current in zip(ema.buffers(), model.buffers(), strict=True):
                    avg.copy_(current)
            synchronize(device)
            step += 1
            row: JsonDict = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "grad_norm": float(norm),
                "samples": len(data),
                "data_seconds": data_seconds,
                "step_seconds": perf_counter() - update_start,
                "elapsed_seconds": elapsed_before + perf_counter() - started,
            }
            if config.time_mid_probability is not None:
                row["mid_time_samples"] = int(((cpu_times >= 0.3) & (cpu_times < 0.7)).sum())
            if microbatch_loss is not None:
                row["aux_loss"] = total_aux
                row["flow_loss"] = total_loss - total_aux
            if validation is not None and (
                step % config.validation_every == 0 or step == config.max_steps
            ):
                row["val_loss"] = validation_loss(ema, validation, config, device)
            history.append(row)
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
            if step % config.checkpoint_every == 0 or step in config.checkpoint_steps:
                _save(
                    directory,
                    config,
                    contract,
                    model,
                    ema,
                    optimizer,
                    scaler,
                    stream,
                    history,
                    step,
                    device,
                    elapsed_before + perf_counter() - started,
                    snapshot=step in config.checkpoint_steps,
                    auxiliary=auxiliary,
                )
        synchronize(device)
        elapsed = elapsed_before + perf_counter() - started
        _save(
            directory,
            config,
            contract,
            model,
            ema,
            optimizer,
            scaler,
            stream,
            history,
            step,
            device,
            elapsed,
            snapshot=step in config.checkpoint_steps,
            auxiliary=auxiliary,
        )
        result: JsonDict = {
            "schema": "minigen.training-result.v1",
            "hardware": hardware_description(device),
            "run_id": identity_digest,
            "attempt_id": attempt_id,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "objective": config.objective,
            "architecture": config.model.get("architecture", "b0"),
            "model_config": config.model,
            "source_sha256": contract["source_sha256"],
            "data_identity": data_identity,
            "status": reason,
            "step": step,
            "elapsed_seconds": elapsed,
            "device": str(device),
            "precision": config.precision,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "checkpoint": str(directory / "last.pt"),
            "metrics": str(directory / "metrics.jsonl"),
            "train_samples": len(dataset),
            "validation_samples": len(validation) if validation is not None else 0,
            "performance": {
                "optimizer_steps": len(history),
                "samples_seen": sum(row["samples"] for row in history),
                "optimizer_step_seconds": sum(row["step_seconds"] for row in history),
                "samples_per_optimizer_second": (
                    sum(row["samples"] for row in history)
                    / sum(row["step_seconds"] for row in history)
                )
                if history
                else None,
                "data_seconds": sum(row["data_seconds"] for row in history),
                "scope": "累计更新耗时含数据访问/同步，不含验证和检查点写盘；非预热性能基准",
            },
            "memory": {
                "cpu_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "cuda_peak_allocated": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "cuda_peak_reserved": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None,
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory()
                if device.type == "mps"
                else None,
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()
                if device.type == "mps"
                else None,
                "scope": "CPU 为进程峰值；CUDA 为本次调用峰值；MPS 为当前值，非峰值",
            },
        }
        _json_write(directory / "result.json", result)
        return result
    except Exception as error:
        # 仅在身份核验后记录失败，不覆写旧成功结果；时间与 attempt_id 区分不同尝试。
        try:
            _json_write(
                directory / "failure.json",
                {
                    "schema": "minigen.training-failure.v1",
                    "run_id": identity_digest,
                    "attempt_id": attempt_id,
                    "started_at": started_at,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "step_at_failure": step,
                    "last_published_checkpoint": str(directory / "last.pt")
                    if (directory / "last.pt").exists()
                    else None,
                    "resume": str(resume) if resume else None,
                },
            )
        except (OSError, TypeError, ValueError):
            # 磁盘故障可能同时阻止回执写入，原始训练异常仍必须向上传播。
            pass
        raise
    finally:
        model.train(original_mode)


def _save(
    directory: Path,
    config: TrainConfig,
    contract: JsonDict,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stream: StatefulBatchStream,
    history: list[JsonDict],
    step: int,
    device: torch.device,
    elapsed: float,
    *,
    snapshot: bool,
    auxiliary: tuple[nn.Module, torch.optim.Optimizer] | None = None,
) -> None:
    """保存完成更新后的全部恢复状态及按需的保留快照。

    Args:
        directory: 运行目录。
        config: 完整运行配置。
        contract: 恢复兼容性身份。
        model: 在线模型。
        ema: EMA 模型。
        optimizer: AdamW 参数及动量状态。
        scaler: 混合精度梯度缩放状态。
        stream: 数据排列与游标。
        history: 已完成更新的权威日志。
        step: 已完成的优化更新数。
        device: 实际训练设备。
        elapsed: 累计训练调用墙钟秒数。
        snapshot: 是否额外保留该步快照。
        auxiliary: 需要单独保存的训练辅助模块和其 AdamW 状态。

    Raises:
        OSError/RuntimeError: 状态序列化或写盘失败。

    副作用:
        原子写入 last.pt，并按需保留命名快照。
    """
    payload: Checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "hardware": hardware_description(device),
        "objective": config.objective,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "contract": contract,
        "data_identity": contract["data"],
        "stream": stream.state_dict(),
        "rng": rng_state(device),
        "scheduler": {
            "step": step,
            "schedule_steps": config.schedule_steps,
            "warmup_steps": config.warmup_steps,
            "min_lr_ratio": config.min_lr_ratio,
        },
        "history": history,
        "elapsed_seconds": elapsed,
    }
    if auxiliary is not None:
        payload["auxiliary"] = {
            "model": auxiliary[0].state_dict(),
            "optimizer": auxiliary[1].state_dict(),
        }
    save_checkpoint(directory / "last.pt", payload)
    snapshot_path = directory / f"step_{step:06d}.pt"
    if snapshot and not snapshot_path.exists():
        save_checkpoint(snapshot_path, payload)
