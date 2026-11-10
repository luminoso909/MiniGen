"""从真实 P3 训练日志绘制损失、梯度和学习率，不补造缺失步数。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(
    run: Path,
    output: Path,
) -> dict[str, object]:
    """绘制一次有界训练的真实曲线，同时保留原始与移动平均损失。

    Args:
        run: 含 metrics.jsonl 与 result.json 的实际运行目录。
        output: 曲线 PNG 的输出路径。

    Raises:
        ValueError: 日志为空或步数重复、逆序。
        OSError/KeyError: 日志字段缺失或文件不可读写。

    Results:
        dict: 实际步数、损失、停止状态及原始结果。

    副作用:
        创建输出父目录并写入 PNG。
    """
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    steps = [row["step"] for row in rows]
    if not rows or steps != sorted(set(steps)):
        raise ValueError("Expected nonempty, strictly increasing step records")
    losses = [row["loss"] for row in rows]
    smooth = [sum(losses[max(0, i - 19) : i + 1]) / min(i + 1, 20) for i in range(len(rows))]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    axes[0].plot(steps, losses, alpha=0.3, label="stochastic train loss")
    axes[0].plot(steps, smooth, label="20-update moving mean")
    valid = [row for row in rows if row.get("val_loss") is not None]
    if valid:
        axes[0].plot(
            [r["step"] for r in valid],
            [r["val_loss"] for r in valid],
            "o-",
            label="fixed validation pairs",
        )
    axes[0].set_ylabel("velocity MSE")
    axes[0].legend()
    axes[1].plot(steps, [row["grad_norm"] for row in rows])
    axes[1].set_ylabel("gradient norm")
    axes[2].plot(steps, [row["lr"] for row in rows])
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("optimizer update")
    for axis in axes:
        axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return {
        "updates": len(rows),
        "last_step": steps[-1],
        "first_20_mean_loss": sum(losses[:20]) / min(20, len(rows)),
        "last_20_mean_loss": sum(losses[-20:]) / min(20, len(rows)),
        "result": json.loads((run / "result.json").read_text()),
    }


def main() -> None:
    """解析实际运行与输出路径并打印摘要。

    Raises:
        SystemExit: 缺少命令参数。
        ValueError/OSError: 日志不合法或图像保存失败。

    副作用:
        写入曲线并向终端输出实际训练摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot_run(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
