"""可恢复的 Benchmark 生成任务；图像身份与随机初噪独立于批次划分。"""

import fcntl
import hashlib
import json
import math
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn
from torchvision.utils import save_image

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import _loaded_run
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition
from minigen.models.attention import set_attention_backend
from minigen.samplers.flow import integrate
from minigen.training.runtime import synchronize


def write_json(
    path: Path,
    value: JsonDict,
) -> None:
    """原子发布有限 JSON，读者不会看到半条结果。

    Args:
        path: 目标文件，父目录必须存在。
        value: JSON 元数据。

    Raises:
        ValueError/OSError: 数据非有限或写盘失败。

    副作用:
        写临时文件并替换目标。
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def content_hash(
    value: JsonDict,
) -> str:
    """规范化任务或配置以构造可审计身份。

    Args:
        value: 有限 JSON 对象。

    Raises:
        ValueError/TypeError: 内容不能表示为有限 JSON。

    Results:
        str: SHA-256 摘要。
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def integration_steps(
    method: str,
    solver_nfe: int,
) -> int:
    """将速度场求值预算转换为积分步数，不混入 CFG 分支。

    Args:
        method: euler 或 heun。
        solver_nfe: 正整数求值次数；Heun 必须是偶数。

    Raises:
        ValueError: 方法或预算不合法。

    Results:
        int: 与指定预算完全对应的积分步数。
    """
    if type(solver_nfe) is not int or solver_nfe < 1 or method not in {"euler", "heun"}:
        raise ValueError("Expected Euler/Heun and positive solver NFE")
    if method == "heun" and solver_nfe % 2:
        raise ValueError("Heun requires an even solver NFE")
    return solver_nfe // (2 if method == "heun" else 1)


def make_tasks(
    benchmark: JsonDict,
    split: str,
    count: int | None,
) -> list[JsonDict]:
    """从冻结题目构造逐初噪任务，改写题与原题同 seed 使用同一初噪。

    Args:
        benchmark: 已验证的 Benchmark 内容。
        split: validation、test、distribution 或明确的三个测试子集。
        count: 可选提示词数量上限，不是最终图片数。

    Raises:
        ValueError: 子集、数量或任务身份无效。

    Results:
        list: 含 sample_id/task_id、prompt_id、caption、seed 和要求的有序任务。
    """
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("Prompt count must be positive")
    if split == "distribution":
        prompts = [
            {
                "prompt_id": row["sample_id"],
                "caption": row["caption"],
                "seeds": [20261110],
                "requirements": [],
                "source_sample_id": row["sample_id"],
            }
            for row in benchmark["references"]
        ]
    elif split in {"validation", "test", "id", "composition_stress", "paraphrase"}:
        prompts = [
            row
            for row in benchmark["prompts"]
            if (row["split"] != "validation" if split == "test" else row["split"] == split)
        ]
    else:
        raise ValueError("Unknown benchmark split")
    if not prompts or (count is not None and count > len(prompts)):
        raise ValueError("Requested prompt count exceeds split")
    tasks = []
    for prompt in prompts[:count]:
        for seed in prompt["seeds"]:
            if type(seed) is not int or seed < 0 or seed >= 2**63:
                raise ValueError("Noise seed must be an integer in [0, 2**63)")
            record = {
                "prompt_id": prompt["prompt_id"],
                "caption": prompt["caption"],
                "seed": seed,
                "requirements": prompt["requirements"],
            }
            task_id = content_hash(record)[:24]
            tasks.append({**record, "task_id": task_id, "sample_id": task_id})
    if len({row["task_id"] for row in tasks}) != len(tasks):
        raise ValueError("Duplicate benchmark tasks")
    return tasks


def initial_noise(
    tasks: list[JsonDict],
    shape: tuple[int, int, int],
) -> Float[Tensor, "b c h w"]:
    """逐任务使用独立 CPU RNG，使恢复和改批大小不改变初噪。

    Args:
        tasks: 每条含 seed 的任务，同 seed 跨提示词共享初噪。
        shape: 单图的 c/h/w 潜变量形状。

    Results:
        Float[Tensor, "b c h w"]: CPU FP32 高斯张量。
    """
    return torch.stack(
        [torch.randn(shape, generator=torch.Generator().manual_seed(row["seed"])) for row in tasks]
    )


@dataclass
class GenerationSession:
    """持有一次运行共用的模型与冻结组件，避免每个任务重复载入。

    Args:
        model: 已 strict 加载的 eval 速度主干。
        text: 冻结文本编码器。
        vae: 冻结 VAE 解码器。
        blank: 同编码器的单条空条件。
        metadata: 原始训练、检查点与数据身份。
    """

    model: nn.Module
    text: FrozenCLIPText
    vae: FrozenVAE
    blank: Condition
    metadata: JsonDict


def verify_benchmark_source(
    definition: JsonDict,
    metadata: JsonDict,
) -> None:
    """将评测清单绑定到检查点实际训练数据，拒绝混用另一冻结数据版本。

    Args:
        definition: 已通过内容锁检查的 Benchmark。
        metadata: 已加载检查点的数据身份。

    Raises:
        ValueError: 清单摘要缺失或不同于训练数据清单。
    """
    expected = metadata.get("data_identity", {}).get("source", {}).get("manifest_sha256")
    if not expected or definition.get("manifest_sha256") != expected:
        raise ValueError("Benchmark manifest differs from checkpoint training data")


def load_session(
    root: Path,
    config: Path,
    checkpoint: Path,
    device: str,
    weights: str,
    attention_backend: str,
) -> GenerationSession:
    """复用原有检查点及缓存核验，只读载入生成所需组件。

    Args:
        root: 项目/资产根目录。
        config: 与检查点匹配的训练配置。
        checkpoint: 已训练的本地权重。
        device: cpu、mps、cuda 或 auto，当前采样固定 FP32。
        weights: ema 或 model。
        attention_backend: 主干的 sdpa 或 naive。

    Raises:
        ValueError/OSError/RuntimeError: 身份、设备、资产或权重错误。

    Results:
        GenerationSession: 已加载的单次生成会话。

    副作用:
        分配模型与组件内存，设置主干运行时后端。
    """
    if weights not in {"ema", "model"}:
        raise ValueError("Weights must be ema or model")
    args = Namespace(
        root=root, config=config, checkpoint=checkpoint, device=device, weights=weights
    )
    _, data, _, _, blank, model, metadata = _loaded_run(args)
    resolved = next(model.parameters()).device
    set_attention_backend(model, attention_backend)
    text = FrozenCLIPText.from_lock(root / data["clip_lock"], root=root, device=resolved)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=resolved)
    return GenerationSession(model, text, vae, blank, metadata)


@torch.no_grad()
def render_batch(
    session: GenerationSession,
    tasks: list[JsonDict],
    method: str,
    solver_nfe: int,
    guidance: float,
) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
    """执行一批文本编码、Flow 积分和解码，分别计时主干及端到端路径。

    Args:
        session: 已加载且冻结的会话。
        tasks: 含 caption 和 seed 的逐图任务。
        method: euler 或 heun。
        solver_nfe: 速度场求值预算。
        guidance: CFG 系数，条件和空分支额外调用另计。

    Raises:
        ValueError/FloatingPointError/RuntimeError: 编码、积分、解码或数值异常。

    Results:
        tuple: CPU [b,3,h,w] 的 [0,1] 图像与本批真实计时/调用统计。
    """
    device = next(session.model.parameters()).device
    synchronize(device)
    started = perf_counter()
    encoded = session.text.encode([row["caption"] for row in tasks])
    condition = Condition(encoded.features, encoded.mask).to(device)
    config = session.model.config
    initial = initial_noise(tasks, (config.in_channels, config.latent_size, config.latent_size)).to(
        device
    )
    synchronize(device)
    denoise_start = perf_counter()
    generated = integrate(
        session.model,
        initial,
        condition,
        session.blank,
        integration_steps(method, solver_nfe),
        guidance,
        method,
    )
    synchronize(device)
    denoise_seconds = perf_counter() - denoise_start
    decoded = session.vae.decode(generated.samples)
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Nonfinite VAE output")
    images = (decoded.clamp(-1, 1).cpu() + 1) / 2
    synchronize(device)
    return images, {
        "denoise_seconds": denoise_seconds,
        "pipeline_seconds": perf_counter() - started,
        "solver_nfe": generated.solver_nfe,
        "model_evaluations": generated.model_evaluations,
        "images": len(tasks),
        "timing_scope": (
            "pipeline includes encoding/noise/decode/CPU transfer; excludes loading and disk writes"
        ),
    }


def publish_receipt(
    path: Path,
    report: JsonDict,
) -> None:
    """每次发布前刷新计数，使进程中断后的部分回执仍可独立读取。

    Args:
        path: 目标 samples.json。
        report: 含全部 records 的运行回执。

    Raises:
        OSError/ValueError: 结果写入或有限 JSON 校验失败。

    副作用:
        更新状态统计并原子写回。
    """
    records = report["records"]
    report.update(
        completed_samples=sum(row["status"] == "completed" for row in records),
        failed_samples=sum(row["status"] == "failed" for row in records),
        pending_samples=sum(row["status"] == "pending" for row in records),
    )
    report["complete"] = report["completed_samples"] == len(records)
    report["status"] = "completed" if report["complete"] else "partial"
    write_json(path, report)


def execute_tasks(
    tasks: list[JsonDict],
    identity: JsonDict,
    output: Path,
    batch_size: int,
    render: Callable[[list[JsonDict]], tuple[Float[Tensor, "b 3 h w"], JsonDict]],
    resume: bool = False,
    retry_failed: bool = False,
    max_seconds: float = 1800,
) -> JsonDict:
    """逐批原子发布任务结果；失败保留分母，恢复核对已完成图片哈希。

    Args:
        tasks: 具有唯一 task_id 的有序任务。
        identity: 生成协议与模型身份，必须包含 scope。
        output: 独立产物目录。
        batch_size: 正整数批大小，改变批大小不改变任务初噪。
        render: 将任务批映射为图像与计时统计的显式回调。
        resume: 仅在身份匹配时继续已存在的 samples.json。
        retry_failed: 恢复时是否重试已记录失败的任务。
        max_seconds: 本次调用运行时限，批之间检查。

    Raises:
        ValueError: 参数、身份、记录或已发布图片校验失败。
        FileExistsError: 没有明确 resume 却已有输出，或另一进程占用目录。
        OSError: 发布结果失败。

    Results:
        JsonDict: 完成/失败/待处理的全量记录；partial 不能作为完整评测。

    副作用:
        写 PNG、samples.json 和执行锁；耗时回调不修改训练参数。
    """
    if (
        type(batch_size) is not int
        or batch_size < 1
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Expected positive batch size and finite time budget")
    if not tasks or len({row["task_id"] for row in tasks}) != len(tasks):
        raise ValueError("Expected unique nonempty tasks")
    if identity.get("scope") not in {"real_local", "formal", "simulated"}:
        raise ValueError("Explicit evidence scope is required")
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "samples.json"
    with (output / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError("Generation output is in use") from error
        records = [
            {**row, "status": "pending", "image_path": None, "sha256": None, "error": None}
            for row in tasks
        ]
        timings = []
        if receipt.exists():
            if not resume:
                raise FileExistsError(receipt)
            old = json.loads(receipt.read_text())
            if old["identity"] != identity or old["tasks_hash"] != content_hash({"tasks": tasks}):
                raise ValueError("Generation identity or tasks differ")
            records, timings = old["records"], old["batches"]
            if [row["task_id"] for row in records] != [row["task_id"] for row in tasks]:
                raise ValueError("Generation records are incomplete or reordered")
            for row, task in zip(records, tasks, strict=True):
                if any(row.get(key) != value for key, value in task.items()):
                    raise ValueError("Generation record content differs from original task")
                if row["status"] == "completed":
                    file = (output / row["image_path"]).resolve()
                    if (
                        not file.is_relative_to(output.resolve())
                        or not file.is_file()
                        or hash_file(file) != row["sha256"]
                    ):
                        raise ValueError("Completed image is missing, modified or outside output")
                    with Image.open(file) as image:
                        image.verify()
                elif row["status"] not in {"failed", "pending"}:
                    raise ValueError("Unknown task status")
        elif any(p.name != ".generation.lock" for p in output.iterdir()):
            raise ValueError("Cannot use nonempty output without a generation receipt")
        report = {
            "schema": "minigen.bench-generation.v1",
            "scope": identity["scope"],
            "identity": identity,
            "output": str(output.resolve()),
            "tasks_hash": content_hash({"tasks": tasks}),
            "records": records,
            "batches": timings,
            "expected_samples": len(tasks),
            "stop_reason": "running_or_interrupted",
        }
        started = perf_counter()
        pending = [
            i
            for i, row in enumerate(records)
            if row["status"] == "pending" or (retry_failed and row["status"] == "failed")
        ]
        publish_receipt(receipt, report)
        for offset in range(0, len(pending), batch_size):
            if perf_counter() - started >= max_seconds:
                report["stop_reason"] = "time_limit"
                break
            indices = pending[offset : offset + batch_size]
            batch = [tasks[i] for i in indices]
            try:
                images, timing = render(batch)
                if (
                    images.ndim != 4
                    or images.shape[:2] != (len(batch), 3)
                    or not torch.isfinite(images).all()
                    or images.min() < 0
                    or images.max() > 1
                ):
                    raise ValueError("Renderer must return finite [b,3,h,w] images in [0,1]")
                for index, image in zip(indices, images, strict=True):
                    row = records[index]
                    relative = f"{row['task_id']}.png"
                    temporary = output / (relative + ".tmp")
                    save_image(image, temporary, format="PNG")
                    temporary.replace(output / relative)
                    row.update(
                        status="completed",
                        image_path=relative,
                        sha256=hash_file(output / relative),
                        error=None,
                    )
                timings.append(
                    {
                        **timing,
                        "task_ids": [row["task_id"] for row in batch],
                        "batch_size": len(batch),
                    }
                )
            except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
                for index in indices:
                    records[index].update(
                        status="failed",
                        image_path=None,
                        sha256=None,
                        error=f"{type(error).__name__}: {error}",
                    )
            publish_receipt(receipt, report)
        report["last_attempt_seconds"] = perf_counter() - started
        if report["stop_reason"] != "time_limit":
            report["stop_reason"] = (
                "completed"
                if all(row["status"] == "completed" for row in records)
                else "item_failures"
            )
        publish_receipt(receipt, report)
        return report


def run_generation(
    root: Path,
    config: Path,
    checkpoint: Path,
    benchmark: Path,
    output: Path,
    split: str = "validation",
    count: int | None = None,
    device: str = "cpu",
    batch_size: int = 2,
    method: str = "euler",
    solver_nfe: int = 10,
    guidance: float = 1.0,
    weights: str = "ema",
    attention_backend: str = "sdpa",
    scope: str = "real_local",
    resume: bool = False,
    retry_failed: bool = False,
    approval: Path | None = None,
    max_seconds: float = 1800,
) -> JsonDict:
    """将冻结题目、既有检查点与可恢复生成器连接为真正的 Benchmark 入口。

    Args:
        root: 项目根目录。
        config: 原始训练配置。
        checkpoint: 已训练检查点，不执行续训。
        benchmark: 已锁定的 benchmark.json。
        output: 新生成目录或明确恢复的目录。
        split: validation、test、distribution 或具体测试子集。
        count: 本地烟测提示词数上限，正式运行不得缩减清单。
        device: cpu、mps、cuda 或 auto。
        batch_size: 同时生成的图像数。
        method: euler 或 heun。
        solver_nfe: 不含 CFG 附加分支的求值预算。
        guidance: 非负 CFG 系数。
        weights: ema 或 model。
        attention_backend: sdpa 或 naive。
        scope: real_local 或 formal；此入口不会创建模拟成绩。
        resume: 是否核对身份后恢复。
        retry_failed: 是否重试已记录的失败任务。
        approval: 正式运行所需的真实人工批准记录。
        max_seconds: 本次调用的推理时限。

    Raises:
        ValueError/FileExistsError/OSError/RuntimeError: 协议、审批、身份或资产无效。

    Results:
        JsonDict: 全量任务及实际完成状态，未完成时保留原分母。

    副作用:
        读取冻结模型与数据，执行有限推理并保存图像和回执。
    """
    from minigen.evaluation.benchmark import content_hash as benchmark_content_hash
    from minigen.evaluation.benchmark import load_benchmark

    if scope not in {"real_local", "formal"} or (scope == "formal" and count is not None):
        raise ValueError("Formal runs require full tasks; only real_local/formal are allowed")
    if scope != "formal" and split != "validation":
        raise ValueError("Local debugging uses validation only; test tasks require formal approval")
    integration_steps(method, solver_nfe)
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Guidance must be finite and nonnegative")
    root = root.resolve()
    benchmark = root / benchmark
    definition = load_benchmark(
        benchmark,
        require_approved=scope == "formal",
        approval=(root / approval) if approval else None,
    )
    tasks = make_tasks(definition, split, count)
    session = load_session(root, config, checkpoint, device, weights, attention_backend)
    verify_benchmark_source(definition, session.metadata)
    actual_device = str(next(session.model.parameters()).device)
    identity = {
        **session.metadata,
        "benchmark_sha256": hash_file(benchmark),
        "benchmark_content_sha256": benchmark_content_hash(definition),
        "approval_verified": scope == "formal",
        "approval_sha256": hash_file(root / approval) if scope == "formal" and approval else None,
        "scope": scope,
        "split": split,
        "method": method,
        "solver_nfe": solver_nfe,
        "guidance": guidance,
        "weights": weights,
        "attention_backend": attention_backend,
        "device": actual_device,
        "precision": "fp32",
        "generator_version": "minigen-bench-generator-v1",
        "evaluation_source_sha256": content_hash(
            {
                str(p.relative_to(root)): hash_file(p)
                for p in sorted((root / "src/minigen").rglob("*.py"))
            }
        ),
    }

    def render(
        batch: list[JsonDict],
    ) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
        """为当前冻结会话执行一个批次。

        Args:
            batch: 有序的提示词与种子任务。

        Results:
            tuple: CPU 图像和本批耗时/求值次数。
        """
        return render_batch(session, batch, method, solver_nfe, guidance)

    return execute_tasks(
        tasks, identity, root / output, batch_size, render, resume, retry_failed, max_seconds
    )
