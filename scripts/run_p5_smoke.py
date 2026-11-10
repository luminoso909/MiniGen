"""用既有 P4 检查点运行有总时限的 P5 本机工程验收，不训练或下载权重。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def run_smoke(
    root: Path,
    output: Path,
    device: str,
    max_seconds: float,
) -> dict[str, object]:
    """以真实 CLI 顺序生成、评分、剖析和汇总，每步原子记录结果。

    Args:
        root: 含 P4 检查点和冻结 Benchmark 的项目根目录。
        output: 新验收目录；不覆盖已有实验。
        device: cpu 或 mps，禁止本入口启动 CUDA 或训练。
        max_seconds: 所有子命令共享的执行上限，最多 1800 秒。

    Raises:
        ValueError: 时限或设备不属于本机有限验收范围。
        FileExistsError: 输出目录已存在。
        OSError: 目录或子进程启动失败。

    Results:
        dict: 逐命令状态、耗时、日志和未执行任务；失败不伪造指标。

    副作用:
        执行离线本机推理并写图像、特征缓存、评分和报告；不修改历史产物。
    """
    if device not in {"cpu", "mps"} or not 0 < max_seconds <= 1800:
        raise ValueError("Local smoke requires cpu/mps and at most 1800 seconds")
    output = root / output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    benchmark = "reports/v2/benchmark/artbench_v1/benchmark.json"
    commands = []
    prefix = [sys.executable, "-m", "minigen", "bench"]
    for arch in ("b0", "b1", "b2"):
        folder = output / arch
        commands.append(
            prefix
            + [
                "generate",
                "--config",
                f"configs/training/p4_{arch}_resume.json",
                "--checkpoint",
                f"artifacts/runs/p4_{arch}_local/last.pt",
                "--benchmark",
                benchmark,
                "--output",
                str(folder),
                "--device",
                device,
                "--count",
                "4",
                "--batch-size",
                "2",
                "--method",
                "euler",
                "--solver-nfe",
                "10",
                "--guidance",
                "1",
                "--max-seconds",
                "300",
            ]
        )
        commands.append(
            prefix
            + [
                "score",
                "--generation",
                str(folder / "samples.json"),
                "--reference",
                benchmark,
                "--output",
                str(folder / "scores.json"),
                "--device",
                device,
                "--metrics",
                "kid",
                "cmmd",
                "clipscore",
                "lpips",
                "--batch-size",
                "4",
            ]
        )
    commands.append(
        prefix
        + [
            "score",
            "--generation",
            str(output / "b0/samples.json"),
            "--output",
            str(output / "vqa_mock.json"),
            "--metrics",
            "vqa",
            "--vqa-backend",
            "mock",
            "--device",
            "cpu",
        ]
    )
    commands.append(
        prefix
        + [
            "profile",
            "--config",
            "configs/training/p4_b0_resume.json",
            "--checkpoint",
            "artifacts/runs/p4_b0_local/last.pt",
            "--benchmark",
            benchmark,
            "--output",
            str(output / "profile.json"),
            "--device",
            device,
            "--count",
            "1",
            "--warmup",
            "1",
            "--repeats",
            "2",
        ]
    )
    commands.append(
        prefix
        + [
            "review",
            "export",
            "--generations",
            *[str(output / arch / "samples.json") for arch in ("b0", "b1", "b2")],
            "--output",
            str(output / "blind_review"),
        ]
    )
    commands.append(
        prefix
        + [
            "report",
            "--scores",
            *[str(output / a / "scores.json") for a in ("b0", "b1", "b2")],
            "--training",
            *[f"artifacts/runs/p4_{a}_local/result.json" for a in ("b0", "b1", "b2")],
            "--profiles",
            str(output / "profile.json"),
            "--output",
            str(output / "report"),
        ]
    )
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    started = perf_counter()
    entries = []
    for i, command in enumerate(commands):
        remaining = max_seconds - (perf_counter() - started)
        row = {"command": command, "status": "not_run", "log": f"command_{i:02d}.log"}
        if remaining > 0:
            before = perf_counter()
            try:
                with (output / row["log"]).open("w") as log:
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                        check=False,
                    )
                row.update(
                    returncode=completed.returncode,
                    status="completed" if completed.returncode == 0 else "failed",
                )
            except subprocess.TimeoutExpired:
                row["status"] = "time_limit"
            row["seconds"] = perf_counter() - before
        entries.append(row)
        print(json.dumps(row), flush=True)
        receipt = {
            "scope": "real_local",
            "device": device,
            "max_seconds": max_seconds,
            "elapsed_seconds": perf_counter() - started,
            "commands": entries,
            "complete": len(entries) == len(commands)
            and all(r["status"] == "completed" for r in entries),
        }
        temporary = output / "execution.json.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output / "execution.json")
    return receipt


def main() -> None:
    """读取显式设备和新输出目录，执行本机小样本验收。

    Raises:
        SystemExit: 参数缺失，或本轮包含失败/未完成命令时退出 1。
        OSError/ValueError: 路径或资源设置错误。

    副作用:
        调用有总时限的离线推理命令并写真实执行回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=1800)
    result = run_smoke(**vars(parser.parse_args()))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
