"""同一检查点的采样预算和局部注意力对照，不声称本机数据代表 CUDA。"""

import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import torch

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.evaluation.generation import (
    load_session,
    make_tasks,
    render_batch,
    verify_benchmark_source,
    write_json,
)
from minigen.models.attention import set_attention_backend
from minigen.training.runtime import hardware_description, synchronize


def sampling_matrix() -> list[JsonDict]:
    """生成先 NFE、再 CFG 的单变量扫描，另加固定设置的 naive 对照。

    Results:
        list: 九项设置；Heun 使用相同 solver NFE，CFG 额外调用单独报告。
    """
    rows = [
        {"method": method, "solver_nfe": nfe, "guidance": 1.0, "attention_backend": "sdpa"}
        for method in ("euler", "heun")
        for nfe in (10, 20, 40)
    ]
    rows += [
        {"method": "euler", "solver_nfe": 40, "guidance": scale, "attention_backend": "sdpa"}
        for scale in (2.0, 3.0)
    ]
    rows.append(
        {"method": "euler", "solver_nfe": 40, "guidance": 1.0, "attention_backend": "naive"}
    )
    return rows


def run_profile(
    root: Path,
    config: Path,
    checkpoint: Path,
    benchmark: Path,
    output: Path,
    device: str = "cpu",
    count: int = 1,
    warmup: int = 1,
    repeats: int = 3,
    matrix: Path | None = None,
    weights: str = "ema",
) -> JsonDict:
    """在固定验证提示和初噪上实测主干与端到端耗时，逐设置保存失败。

    Args:
        root: 项目根。
        config: 与检查点匹配的配置。
        checkpoint: 固定模型检查点。
        benchmark: 锁定的题目文件，仅取 validation。
        output: 不覆盖的性能 JSON 文件。
        device: cpu、mps、cuda 或 auto。
        count: 固定验证提示词数，每题使用已登记种子。
        warmup: 每种设置的预热次数，至少一次。
        repeats: 正式计时重复次数。
        matrix: 可选 JSON 设置列表；None 使用九项单变量扫描。
        weights: ema 或 model。

    Raises:
        ValueError/FileExistsError/OSError/RuntimeError: 参数、协议、资产或输出错误。

    Results:
        JsonDict: 每次耗时、实际求值次数与分设备内存；未测 CUDA 字段为 null。

    副作用:
        加载冻结组件、执行有限采样并写性能回执，不保存正式评测图片。
    """
    from minigen.evaluation.benchmark import load_benchmark

    if type(warmup) is not int or warmup < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("Profile requires positive warmup and repeat counts")
    root = root.resolve()
    output = root / output
    if output.exists():
        raise FileExistsError(output)
    definition = load_benchmark(root / benchmark)
    tasks = make_tasks(definition, "validation", count)
    configurations = json.loads((root / matrix).read_text()) if matrix else sampling_matrix()
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("Profile matrix must be a nonempty list")
    started = perf_counter()
    session = load_session(root, config, checkpoint, device, weights, "sdpa")
    verify_benchmark_source(definition, session.metadata)
    resolved = next(session.model.parameters()).device
    synchronize(resolved)
    load_seconds = perf_counter() - started
    results = []
    for setting in configurations:
        row = {"setting": setting, "status": "failed", "measurements": [], "error": None}
        try:
            if set(setting) != {"method", "solver_nfe", "guidance", "attention_backend"}:
                raise ValueError("Unknown/missing profile setting fields")
            set_attention_backend(session.model, setting["attention_backend"])
            for _ in range(warmup):
                render_batch(
                    session, tasks, setting["method"], setting["solver_nfe"], setting["guidance"]
                )
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)
            for _ in range(repeats):
                _, measured = render_batch(
                    session, tasks, setting["method"], setting["solver_nfe"], setting["guidance"]
                )
                row["measurements"].append(measured)
            row.update(
                status="completed",
                images_per_repeat=len(tasks),
                denoise_seconds_mean=sum(x["denoise_seconds"] for x in row["measurements"])
                / repeats,
                pipeline_seconds_mean=sum(x["pipeline_seconds"] for x in row["measurements"])
                / repeats,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(resolved)
                if resolved.type == "cuda"
                else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(resolved)
                if resolved.type == "cuda"
                else None,
                mps_current_allocated_bytes=torch.mps.current_allocated_memory()
                if resolved.type == "mps"
                else None,
                mps_peak_allocated_bytes=None,
                memory_reason="MPS API exposes current allocation only; CUDA peak only on CUDA",
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        results.append(row)
    report = {
        "schema": "minigen.bench-profile.v1",
        "scope": "simulated"
        if session.metadata.get("scope") == "simulated"
        else "real_local"
        if resolved.type != "cuda"
        else "real_pilot",
        "identity": session.metadata,
        "benchmark_sha256": hash_file(root / benchmark),
        "device": str(resolved),
        "precision": "fp32",
        "torch": str(torch.__version__),
        "hardware": hardware_description(resolved),
        "warmup": warmup,
        "repeats": repeats,
        "tasks": tasks,
        "load_seconds": load_seconds,
        "cpu_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "results": results,
        "cost_actual_cny": None,
        "cost_reason": "No price or billed GPU time measured",
        "complete": all(row["status"] == "completed" for row in results),
        "scope_note": (
            "FP32 inference; pipeline excludes asset load and disk I/O; "
            "SDPA kernel selection not forced or inferred"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report
