"""MiniGen 项目命令：训练、采样、固定噪声评估与缓存管理。"""

import argparse
import json
import sys
from pathlib import Path

from minigen import __version__


def main(
    argv: list[str] | None = None,
) -> int:
    """按需导入工程组件并执行命令，保持 doctor 与帮助不依赖可选模型包。

    Args:
        argv: 不含程序名的命令参数；None 使用进程参数。

    Raises:
        SystemExit: argparse 帮助、版本或参数错误触发退出。

    Results:
        int: 成功为 0；诊断失败为 1；执行错误转换为标准错误文本和退出码 2。

    副作用:
        输出命令结果；训练、采样和缓存按参数创建本地文件。
    """
    values = sys.argv[1:] if argv is None else argv
    if values and values[0] == "cache":
        from minigen.cache_command import main as cache_main

        try:
            cache_main(values[1:])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            print(f"minigen cache: {error}", file=sys.stderr)
            return 2
        return 0
    parser = argparse.ArgumentParser(prog="minigen", description="MiniGen 开发与实验工具")
    parser.add_argument("--version", action="version", version=f"minigen {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查依赖、设备和张量前向/反向")
    doctor.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    doctor.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train = commands.add_parser("train", help="从固定缓存训练，支持精确恢复")
    train.add_argument("--root", type=Path, default=Path.cwd())
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    commands.add_parser("cache", help="构建或独立回读固定组件缓存")
    for name in ("sample", "evaluate"):
        child = commands.add_parser(
            name, help="Flow 采样" if name == "sample" else "固定配对 Flow MSE 诊断"
        )
        child.add_argument("--root", type=Path, default=Path.cwd())
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        child.add_argument("--weights", choices=("ema", "model"), default="ema")
        child.add_argument(
            "--split",
            choices=("train", "validation"),
            default="train" if name == "sample" else "validation",
        )
        child.add_argument(
            "--condition-mode", choices=("correct", "blank", "shuffled"), default="correct"
        )
        child.add_argument(
            "--count", "--limit", dest="count", type=int, default=8 if name == "sample" else 16
        )
        child.add_argument("--start-index", type=int, default=0, help="已配置子集内的起点")
        child.add_argument("--seed", type=int, default=1234)
        if name == "sample":
            child.add_argument("--prompts", type=Path, help="可选 JSON 字符串列表，替代缓存条件")
            child.add_argument("--steps", type=int, default=40)
            child.add_argument("--method", choices=("euler", "heun"), default="euler")
            child.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args(values)
    if args.command != "doctor":
        from minigen.commands import evaluate_command, sample_command, train_command

        try:
            report = {
                "train": train_command,
                "sample": sample_command,
                "evaluate": evaluate_command,
            }[args.command](args)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError, TypeError) as error:
            print(f"minigen {args.command}: {error}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    from minigen.doctor import diagnose

    report = diagnose(args.device)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python {report['python']} | MiniGen {__version__}")
        for name, result in report["packages"].items():
            print(f"{name}: {result.get('version', result.get('error'))}")
        for name, result in report["devices"].items():
            print(f"{name}: {result['status']}")
        print("检查通过" if report["ok"] else "检查失败；使用 --json 查看细节")
    return 0 if report["ok"] else 1
