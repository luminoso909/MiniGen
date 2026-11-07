"""准备固定版本的 ArtBench 256×256 图像、清单和本地审计包。"""

import argparse
import json
from pathlib import Path

import yaml

from minigen.data.artbench import prepare_artbench, sha256_file
from minigen.data.artbench_caption import load_clip_token_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "datasets" / "artbench_pd_256.yaml"
DEFAULT_CLIP_LOCK = ROOT / "configs" / "models" / "clip_text.json"


def main() -> int:
    """读取本地 ArtBench Parquet，生成可直接核对的全量数据审计。

    Raises:
        SystemExit: 参数格式不合法或请求命令帮助。
        ValueError/OSError/ImportError: 输入身份、读取或依赖异常时透传。

    Results:
        int: 自动验收通过为 0；完成审计但未通过为 1。

    副作用:
        创建独立的处理版本目录与审计包，并在终端输出摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clip-lock", type=Path, default=DEFAULT_CLIP_LOCK)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_dir = args.raw_dir or ROOT / "data" / "raw" / "artbench_pd_256" / config["revision"]
    output_dir = (
        args.output_dir or ROOT / "data" / "processed" / "artbench_pd_256" / config["version"]
    )
    tokenizer = load_clip_token_counter(ROOT, args.clip_lock)
    result = prepare_artbench(
        args.config, raw_dir, output_dir, tokenizer, sha256_file(args.clip_lock)
    )
    brief = {key: value for key, value in result.items() if key != "audit_sample_ids"}
    print(json.dumps({"output_dir": str(output_dir), **brief}, ensure_ascii=False, indent=2))
    return 0 if result["automated_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
