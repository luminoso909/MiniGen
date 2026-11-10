"""由真实 CUDA Pilot 和用户填写的预算生成正式训练配置，不启动训练。"""

import argparse
import json
import math
from pathlib import Path

from minigen.commands import read_experiment


def prepare_config(
    root: Path,
    architecture: str,
    pilot: Path,
    output: Path,
    run_dir: str,
    steps: int,
    max_seconds: float,
    hourly_cny: float,
    budget_cny: float,
) -> dict[str, object]:
    """继承已测量 Pilot 配方，构造全训练集配置及独立成本估算回执。

    Args:
        root: 仓库根目录。
        architecture: b0、b1 或 b2。
        pilot: 对应架构实际 CUDA pilot.json。
        output: 新训练配置路径，禁止覆盖。
        run_dir: 新训练运行目录。
        steps: 用户批准的完整训练日程步数。
        max_seconds: 本次允许的训练循环时限。
        hourly_cny: 当前实际 GPU 时价。
        budget_cny: 用户填写的单次训练预算，不包含未登记附加服务。

    Raises:
        ValueError: 不是对应 CUDA Pilot、参数非法或估算超预算。
        FileExistsError: 配置或运行目录已存在。
        OSError/KeyError: Pilot 或模型配置不可读。

    Results:
        dict: 配置路径、更新步数及含 25% 运行预留的费用估算。

    副作用:
        写入新配置及 .budget.json 回执，不请求 GPU、不训练、不替用户批准预算。
    """
    if architecture not in {"b0", "b1", "b2"} or type(steps) is not int or steps < 1:
        raise ValueError("Expected one architecture and positive steps")
    if any(not math.isfinite(x) or x <= 0 for x in (max_seconds, hourly_cny, budget_cny)):
        raise ValueError("Runtime and cost settings must be finite and positive")
    report = json.loads((root / pilot).read_text())
    result = report["training_result"]
    if (
        report.get("status") != "measured"
        or not report.get("cuda_device")
        or result.get("device") != "cuda"
        or result.get("architecture") != architecture
        or report.get("measured_steps", 0) < 1
    ):
        raise ValueError("An actual matching CUDA pilot is required")
    step_seconds = float(report["step_seconds_mean"])
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Pilot timing must be finite and positive")
    estimated_seconds = step_seconds * steps * 1.25
    estimated_cny = estimated_seconds / 3600 * hourly_cny
    if estimated_cny > budget_cny or estimated_seconds > max_seconds:
        raise ValueError("Estimated full schedule exceeds declared runtime or budget")
    run_record = json.loads((root / pilot).with_name("run.json").read_text())
    training = dict(run_record["config"])
    model_spec = json.loads((root / f"configs/models/{architecture}_30m.json").read_text())
    if (
        not result.get("run_id")
        or run_record.get("run_id") != result.get("run_id")
        or training.get("device") != "cuda"
        or training.get("precision") != report.get("precision")
        or training.get("model") != model_spec
    ):
        raise ValueError("Pilot run.json disagrees with measured CUDA run identity")
    training.pop("model")
    interval = max(1, steps // 20)
    training.update(
        output_dir=run_dir,
        max_steps=steps,
        max_seconds=max_seconds,
        schedule_steps=steps,
        warmup_steps=min(500, steps - 1, max(1, steps // 20)),
        checkpoint_every=interval,
        checkpoint_steps=list(range(interval, steps + 1, interval)),
        validation_every=interval,
    )
    payload = {
        "model": f"configs/models/{architecture}_30m.json",
        "training": training,
        "data": {"train_limit": None, "validation_limit": 256},
    }
    path = root / output
    budget_path = path.with_suffix(".budget.json")
    if path.exists() or budget_path.exists() or (root / run_dir).exists():
        raise FileExistsError("Use new config and run paths")
    receipt = {
        "config": str(path),
        "architecture": architecture,
        "steps": steps,
        "pilot": str(pilot),
        "hourly_cny": hourly_cny,
        "declared_budget_cny": budget_cny,
        "estimated_seconds_with_reserve": estimated_seconds,
        "estimated_cny_with_reserve": estimated_cny,
        "actual_cost_cny": None,
        "scope": "估算含25%运行预留，不含存储传输等费用；预算值由调用者填写，非自动批准",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        read_experiment(root, path)
    except (ValueError, TypeError, OSError, KeyError):
        path.unlink()
        raise
    budget_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    """解析生成正式配置所需的实测 Pilot 和用户资源设置。

    Raises:
        SystemExit: 缺少必要参数。
        ValueError/OSError/KeyError: 配方或资源约束不成立。

    副作用:
        写配置与预算回执，打印实际路径，不执行训练。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--architecture", choices=("b0", "b1", "b2"), required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--hourly-cny", type=float, required=True)
    parser.add_argument("--budget-cny", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_config(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
