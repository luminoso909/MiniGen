"""Honest command surface: only doctor is implemented in the bootstrap."""

import argparse
import json
import sys

from minigen import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minigen", description="MiniGen 开发与学习工具")
    parser.add_argument("--version", action="version", version=f"minigen {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查依赖、设备和小张量前向/反向")
    doctor.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    doctor.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    for name in ("train", "sample", "evaluate", "cache"):
        commands.add_parser(name, help="尚未实现；保留未来命令入口")
    args = parser.parse_args(argv)
    if args.command != "doctor":
        print(
            f"minigen {args.command}: 尚未实现。请按任务清单推进；不会下载数据或启动训练。",
            file=sys.stderr,
        )
        return 2

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
