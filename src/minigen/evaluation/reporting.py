"""从已存在的真实回执生成分范围结果表、图表及完整样本索引。"""

import csv
import html
import json
import math
from pathlib import Path

from minigen._typing import JsonDict
from minigen.evaluation.benchmark import content_hash


def comparison_rows(
    scores: list[JsonDict],
    training: list[JsonDict],
) -> list[JsonDict]:
    """按 run_id 连接训练与评分；缺测 GPU 和成本保留为空。

    Args:
        scores: score 回执列表，每个包含 scope、metrics 和身份。
        training: 对应训练 result 回执；未提供时不推测训练资源。

    Raises:
        ValueError: 同一 run_id 出现冲突的训练身份。

    Results:
        list: 可写入 CSV 的每检查点指标与资源行，不生成架构排名。
    """
    runs = {}
    for run in training:
        key = (run.get("run_id"), run.get("step"))
        if key in runs and runs[key] != run:
            raise ValueError("Duplicate run identities require distinct checkpoint receipts")
        runs[key] = run
    result = []
    for score in scores:
        run = runs.get((score.get("run_id"), score.get("checkpoint_step")), {})
        if run.get("checkpoint_sha256") and run["checkpoint_sha256"] != score.get(
            "checkpoint_sha256"
        ):
            raise ValueError("Training receipt and score checkpoint hashes differ")
        # 只有配对检查点步数一致的资源记录可连用，不能把末次训练总耗时给早期模型。
        same_step = run.get("step") == score.get("checkpoint_step")
        elapsed = run.get("elapsed_seconds") if same_step else None
        gpu_hours = elapsed / 3600 if elapsed is not None and run.get("device") == "cuda" else None
        row = {
            "scope": score.get("scope"),
            "architecture": score.get("architecture"),
            "run_id": score.get("run_id"),
            "checkpoint_step": score.get("checkpoint_step"),
            "checkpoint_sha256": score.get("checkpoint_sha256"),
            "benchmark_sha256": score.get("benchmark_sha256"),
            "complete": score.get("complete"),
            "exploratory": score.get("exploratory"),
            "valid": score.get("coverage", {}).get("valid"),
            "attempted": score.get("coverage", {}).get("attempted"),
            "parameters": run.get("trainable_parameters"),
            "device": run.get("device"),
            "precision": run.get("precision"),
            "hardware": run.get("hardware"),
            "evaluation_identity": content_hash(
                {k: v for k, v in score.get("identity", {}).items() if k != "generation_sha256"}
            ),
            "solver_nfe": score.get("sampling", {}).get("solver_nfe"),
            "sampler": score.get("sampling", {}).get("method"),
            "guidance": score.get("sampling", {}).get("guidance"),
            "training_seconds": elapsed,
            "gpu_hours": gpu_hours,
            "gpu_hours_reason": None if gpu_hours is not None else "no_matching_cuda_measurement",
            "cost_actual_cny": None,
            "cost_reason": "billing_receipt_not_provided",
            "training_flops": run.get("training_flops_estimate") if same_step else None,
            "flops_reason": run.get("training_flops_method")
            or "requires_explicit_estimator_and_matching_checkpoint",
            "samples_per_second": run.get("performance", {}).get("samples_per_optimizer_second")
            if same_step
            else None,
            "cuda_peak_bytes": run.get("memory", {}).get("cuda_peak_allocated")
            if same_step
            else None,
            "generation_path": score.get("generation_path"),
        }
        for name in ("kid", "cmmd", "clipscore", "vqa", "lpips"):
            metric = score.get("metrics", {}).get(name, {})
            simulated = score.get("scope") == "simulated" or metric.get("scope") == "simulated"
            row[name] = (
                metric.get("value")
                if metric.get("status") == "completed" and not simulated
                else None
            )
            row[f"{name}_status"] = (
                "simulated_excluded" if simulated else metric.get("status", "not_run")
            )
        result.append(row)
    return result


def select_comparison(
    rows: list[JsonDict],
    mode: str,
    target: float,
    tolerance: float = 0.1,
) -> JsonDict:
    """选取同更新步数或同一范围内最接近但不超过预算的已有检查点。

    Args:
        rows: 单一 scope、Benchmark 和硬件口径的结果行。
        mode: steps、gpu_hours 或 training_flops。
        target: 明确的更新步数或计算预算。
        tolerance: 同预算时允许相对目标的最大缺口，默认 10%。

    Raises:
        ValueError: 对照类型、目标无效，或混用范围/Benchmark/硬件。

    Results:
        dict: 选中行和缺失架构；不插值或补造任何检查点。
    """
    if (
        mode not in {"steps", "gpu_hours", "training_flops"}
        or not math.isfinite(target)
        or target <= 0
    ):
        raise ValueError("A positive explicit comparison target is required")
    if not 0 <= tolerance < 1:
        raise ValueError("Tolerance must be in [0,1)")
    if any(r.get("scope") == "simulated" or not r.get("complete") for r in rows):
        raise ValueError("Comparison excludes simulated or incomplete results")
    if mode == "gpu_hours" and any(not r.get("hardware") for r in rows):
        raise ValueError("GPU budget comparison requires actual hardware identity")
    identities = {
        (
            r.get("scope"),
            r.get("benchmark_sha256"),
            r.get("device"),
            r.get("precision"),
            r.get("hardware"),
            r.get("evaluation_identity"),
            r.get("solver_nfe"),
            r.get("sampler"),
            r.get("guidance"),
        )
        for r in rows
    }
    if len(identities) > 1:
        raise ValueError("Comparison requires matching scope, benchmark, device and precision")
    column = "checkpoint_step" if mode == "steps" else mode
    selected = []
    for architecture in ("b0", "b1", "b2"):
        available = [
            r
            for r in rows
            if r.get("architecture") == architecture
            and r.get(column) is not None
            and r[column] <= target
        ]
        if not available:
            continue
        choice = max(available, key=lambda row: row[column])  # 输入结果行，输出预算轴值。
        delta = target - choice[column]
        if (mode == "steps" and delta != 0) or delta / target > tolerance:
            continue
        selected.append(
            {
                "record": choice,
                "target": target,
                "actual": choice[column],
                "relative_gap": delta / target,
            }
        )
    return {
        "mode": mode,
        "target": target,
        "selected": selected,
        "missing": sorted({"b0", "b1", "b2"} - {r["record"]["architecture"] for r in selected}),
    }


def build_report(
    scores: list[Path],
    training: list[Path],
    output: Path,
    profiles: list[Path] | None = None,
    comparison_mode: str | None = None,
    target: float | None = None,
    tolerance: float = 0.1,
) -> JsonDict:
    """输出真实回执汇总和全部样本画廊，缺数据的图明确标记未测量。

    Args:
        scores: 指标回执文件路径列表。
        training: 训练 result.json 路径列表。
        output: 新报告目录，不覆盖既有结果。
        profiles: 可选 profile JSON 路径，只按原值保存，不推测 CUDA 数据。
        comparison_mode: 可选 steps、gpu_hours 或 training_flops 同预算对照。
        target: 使用对照时必须提供的目标步数或预算。
        tolerance: 已有检查点与目标的最大相对预算缺口。

    Raises:
        FileExistsError: 目标目录已存在。
        OSError/ValueError/KeyError: 输入无法读取或身份不一致。

    Results:
        dict: 报告目录和实际结果数。

    副作用:
        创建 JSON、CSV、Markdown、PNG 和 HTML，读取原始生成图用于嵌入画廊。
    """
    import base64

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(output)
    score_values = [json.loads(p.read_text()) for p in scores]
    run_values = [json.loads(p.read_text()) for p in training]
    rows = comparison_rows(score_values, run_values)
    if not rows:
        raise ValueError("At least one real or explicitly simulated score receipt is required")
    if len({row["scope"] for row in rows}) != 1:
        raise ValueError("Separate reports are required for different result scopes")
    if (comparison_mode is None) != (target is None):
        raise ValueError("Comparison mode and target must be specified together")
    comparison = select_comparison(rows, comparison_mode, target, tolerance) if target else None
    output.mkdir(parents=True)
    payload = {
        "schema": "minigen.benchmark-report.v1",
        "rows": rows,
        "comparison": comparison,
        "score_sources": [str(p) for p in scores],
        "training_sources": [str(p) for p in training],
        "profiles": [json.loads(p.read_text()) for p in profiles or []],
        "conclusion": "工程汇总；探索性/模拟/未执行与正式结果分开，未自动选择优胜架构",
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    pairs = [
        ("checkpoint_step", "kid"),
        ("checkpoint_step", "clipscore"),
        ("gpu_hours", "kid"),
        ("training_flops", "kid"),
        ("training_seconds", "cmmd"),
        ("solver_nfe", "cmmd"),
    ]
    for ax, (x, y) in zip(axes.flat, pairs, strict=True):
        for row in rows:
            if row.get(x) is not None and row.get(y) is not None:
                label = f"{row['architecture']}/{row['scope']}"
                ax.scatter(row[x], row[y], label=label, marker="o" if row["complete"] else "x")
        ax.set(xlabel=x, ylabel=y)
        if ax.collections:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "Not measured", ha="center", transform=ax.transAxes)
        ax.grid(alpha=0.2)
    fig.savefig(output / "quality_curves.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for profile in payload["profiles"]:
        if profile.get("scope") == "simulated":
            continue
        for row in profile.get("results", []):
            if row.get("status") != "completed":
                continue
            setting = row["setting"]
            label = (
                f"{setting['method']} NFE{setting['solver_nfe']} CFG{setting['guidance']} "
                f"{setting['attention_backend']}"
            )
            ax.scatter(
                label,
                row["pipeline_seconds_mean"],
                label=f"{profile.get('hardware')}/{profile.get('scope')}",
            )
    ax.set(ylabel="pipeline seconds (model + text + decoder)")
    ax.tick_params(axis="x", rotation=30)
    if not ax.collections:
        ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes)
    else:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
    fig.savefig(output / "performance.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for path, run in zip(training, run_values, strict=True):
        logs = Path(run.get("metrics") or path.with_name("metrics.jsonl"))
        if not logs.exists():
            continue
        entries = [json.loads(line) for line in logs.read_text().splitlines() if line]
        ax.plot(
            [r["step"] for r in entries],
            [r["loss"] for r in entries],
            label=f"{run.get('architecture')}:train",
        )
        valid = [r for r in entries if "val_loss" in r]
        ax.plot(
            [r["step"] for r in valid],
            [r["val_loss"] for r in valid],
            "--",
            label=f"{run.get('architecture')}:validation",
        )
    ax.set(xlabel="optimizer step", ylabel="velocity MSE")
    if ax.lines:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes)
    fig.savefig(output / "loss.png", dpi=120)
    plt.close(fig)
    cards = []
    for score in score_values:
        for record in score.get("records", []):
            path = Path(record.get("image_path") or "")
            if not path.is_absolute() and score.get("generation_path"):
                path = Path(score["generation_path"]).parent / path
            image = ""
            if path.is_file() and path.suffix.lower() == ".png":
                image = (
                    '<img width="256" src="data:image/png;base64,'
                    + base64.b64encode(path.read_bytes()).decode()
                    + '">'
                )
            label = html.escape(
                str(
                    {
                        "architecture": score.get("architecture"),
                        "scope": score.get("scope"),
                        "prompt": record.get("caption"),
                        "status": record.get("status"),
                        "error": record.get("validation_error") or record.get("error"),
                    }
                )
            )
            cards.append(f"<article>{image}<p>{label}</p></article>")
    (output / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MiniGen complete sample index</title>'
        "<style>main{display:flex;flex-wrap:wrap}"
        "article{width:280px;margin:10px;overflow-wrap:anywhere}</style>"
        "<h1>All samples, including failures</h1><main>" + "".join(cards) + "</main>"
    )
    lines = [
        "## **一、实际结果汇总**",
        "",
        "所有数值来自所列回执。模拟 VQA 不进入质量数值表；没有实测的 CUDA 时间和费用为空。",
        "",
        "| 模型 | 范围 | 有效/尝试 | KID | CMMD | CLIPScore |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['architecture']} | {r['scope']} | {r['valid']}/{r['attempted']} | "
            f"{r['kid']} | {r['cmmd']} | {r['clipscore']} |"
        )
    lines += [
        "",
        "![真实曲线](quality_curves.png)",
        "",
        "![训练日志](loss.png)",
        "",
        "![设备实测](performance.png)",
        "",
        "[全部样本及失败](gallery.html) · [CSV](metrics.csv) · [身份与原始来源](summary.json)",
        "",
        "## **二、解释限制**",
        "",
        "real_local 为本地工程验证；simulated 为人工构造测试；"
        "formal 仍需有效 Benchmark、人审和正式检查点。"
        "小样本指标不支持模型优劣结论。费用缺少账单时不推算实际消费，"
        "缺少实测训练 FLOPs 时不伪造该轴。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return {"output": str(output), "records": len(rows), "status": "completed"}
