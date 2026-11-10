"""MiniGen-Bench 独立命令，保留原 evaluate 的配对 Flow MSE 语义。"""

import argparse
import json
from pathlib import Path

from minigen._typing import JsonDict


def _review_records(
    paths: list[Path],
) -> list[JsonDict]:
    """从生成清单收集全部匿名审核任务并展开结构化问题。

    Args:
        paths: 一个或多个 samples.json。

    Raises:
        OSError/KeyError/ValueError: 清单无法解析或记录缺少必要身份。

    Results:
        list: 图像路径绝对化、问题可独立审核的记录；失败项仍保留。
    """
    from minigen.evaluation.vqa import make_questions

    records = []
    for path in paths:
        data = json.loads(path.read_text())
        base = path.parent
        identity = data.get("identity", {})
        for record in data["records"]:
            row = dict(record)
            row["task_id"] = (
                f"{identity.get('checkpoint_sha256', path.parent.name)}:{row['task_id']}"
            )
            row["image_path"] = (
                str((base / row["image_path"]).resolve()) if row.get("image_path") else None
            )
            row["questions"] = [
                {**q, "text": q["question"] + " Expected: " + q["expected"]}
                for q in make_questions(row.get("requirements", []), row["prompt_id"])
            ]
            records.append(row)
    return records


def main(
    argv: list[str] | None = None,
) -> int:
    """按子命令延迟导入评测组件，返回实际完成状态。

    Args:
        argv: bench 后的命令参数；None 使用进程参数。

    Raises:
        SystemExit: 帮助或参数错误。
        OSError/ValueError/KeyError/RuntimeError: 输入身份、资产或计算错误，交由外层 CLI 处理。

    Results:
        int: 完整执行为 0；部分失败或未完成为 1。

    副作用:
        根据显式命令生成评测产物，并打印 JSON 回执；本入口不训练。
    """
    parser = argparse.ArgumentParser(prog="minigen bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="建立候选题目及参考集内容锁")
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path)
    for name in ("generate", "profile"):
        cmd = sub.add_parser(
            name, help="分批恢复生成" if name == "generate" else "固定验证任务性能对照"
        )
        cmd.add_argument("--root", type=Path, default=Path.cwd())
        for field in ("config", "checkpoint", "benchmark", "output"):
            cmd.add_argument(f"--{field}", type=Path, required=True)
        cmd.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
        cmd.add_argument("--count", type=int, default=None if name == "generate" else 1)
        cmd.add_argument("--weights", choices=("ema", "model"), default="ema")
        if name == "generate":
            cmd.add_argument(
                "--split",
                choices=(
                    "validation",
                    "test",
                    "distribution",
                    "id",
                    "composition_stress",
                    "paraphrase",
                ),
                default="validation",
            )
            cmd.add_argument("--batch-size", type=int, default=2)
            cmd.add_argument("--method", choices=("euler", "heun"), default="euler")
            cmd.add_argument("--solver-nfe", type=int, default=10)
            cmd.add_argument("--guidance", type=float, default=1.0)
            cmd.add_argument("--attention-backend", choices=("sdpa", "naive"), default="sdpa")
            cmd.add_argument("--scope", choices=("real_local", "formal"), default="real_local")
            cmd.add_argument("--resume", action="store_true")
            cmd.add_argument("--retry-failed", action="store_true")
            cmd.add_argument("--approval", type=Path)
            cmd.add_argument("--max-seconds", type=float, default=1800)
        else:
            cmd.add_argument("--warmup", type=int, default=1)
            cmd.add_argument("--repeats", type=int, default=3)
            cmd.add_argument("--matrix", type=Path)
    score = sub.add_parser("score", help="独立离线图像质量和条件评分")
    score.add_argument("--root", type=Path, default=Path.cwd())
    score.add_argument("--generation", type=Path, required=True)
    score.add_argument("--reference", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--asset-lock", type=Path, default=Path("configs/models/evaluation.json"))
    score.add_argument(
        "--metrics",
        nargs="+",
        choices=("kid", "cmmd", "clipscore", "vqa", "lpips"),
        default=["kid", "cmmd", "clipscore"],
    )
    score.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    score.add_argument("--batch-size", type=int, default=4)
    score.add_argument(
        "--scope", choices=("real_local", "formal", "simulated"), default="real_local"
    )
    score.add_argument("--vqa-backend", choices=("blip", "mock"), default="blip")
    score.add_argument("--kid-subset-size", type=int)
    score.add_argument("--kid-subsets", type=int)
    score.add_argument("--resume", action="store_true")
    review = sub.add_parser("review", help="人工匿名审核及校准")
    actions = review.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export")
    export.add_argument("--generations", type=Path, nargs="+", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--seed", type=int, default=20261110)
    imp = actions.add_parser("import")
    for field in ("package", "answers", "output"):
        imp.add_argument(f"--{field}", type=Path, required=True)
    calibration = actions.add_parser("calibrate")
    calibration.add_argument("--root", type=Path, default=Path.cwd())
    calibration.add_argument("--benchmark", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--count", type=int, default=8)
    calibration.add_argument("--annotations", type=Path)
    compare = actions.add_parser("compare")
    for field in ("human", "automatic", "output"):
        compare.add_argument(f"--{field}", type=Path, required=True)
    report = sub.add_parser("report", help="汇总真实、模拟和缺测结果")
    report.add_argument("--scores", type=Path, nargs="+", required=True)
    report.add_argument("--training", type=Path, nargs="*", default=[])
    report.add_argument("--profiles", type=Path, nargs="*", default=[])
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--comparison-mode", choices=("steps", "gpu_hours", "training_flops"))
    report.add_argument("--target", type=float)
    report.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop("command")
    if command == "prepare":
        from minigen.evaluation.benchmark import prepare_benchmark

        result = prepare_benchmark(**options)
    elif command == "generate":
        from minigen.evaluation.generation import run_generation

        result = run_generation(**options)
    elif command == "profile":
        from minigen.evaluation.profiling import run_profile

        result = run_profile(**options)
    elif command == "score":
        from minigen.evaluation.scoring import run_scoring

        options["metrics"] = tuple(options["metrics"])
        result = run_scoring(**options)
    elif command == "report":
        from minigen.evaluation.reporting import build_report

        result = build_report(**options)
    else:
        from minigen.evaluation.review import (
            compare_calibration,
            export_review,
            import_review,
            prepare_calibration,
        )

        action = options.pop("action")
        if action == "export":
            options["records"] = _review_records(options.pop("generations"))
            result = export_review(**options)
        elif action == "import":
            result = import_review(**options)
        elif action == "calibrate":
            from minigen.evaluation.benchmark import load_benchmark

            options["benchmark"] = load_benchmark(options["root"] / options["benchmark"])
            result = prepare_calibration(**options)
        else:
            if options["output"].exists():
                raise FileExistsError(options["output"])
            automatic = json.loads(options["automatic"].read_text())
            if isinstance(automatic, dict):
                if "metrics" in automatic:
                    metric = automatic["metrics"]["vqa"]
                    if metric.get("scope") == "simulated":
                        raise ValueError("Simulated VQA cannot calibrate real human judgments")
                    namespace = automatic.get("checkpoint_sha256") or "calibration"
                    automatic = [
                        {
                            **row,
                            "task_id": f"{namespace}:{row['sample_id']}",
                            "verdict": {"wrong": "incorrect", "indeterminate": "undecidable"}.get(
                                row["status"], row["status"]
                            ),
                        }
                        for row in metric["questions"]
                    ]
                else:
                    automatic = automatic["records"]
            result = compare_calibration(json.loads(options["human"].read_text()), automatic)
            options["output"].parent.mkdir(parents=True, exist_ok=True)
            options["output"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("complete") is False)
