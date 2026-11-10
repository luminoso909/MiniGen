"""按已提交身份锁显式下载或离线校验评测权重。"""

import argparse
import json
from pathlib import Path

from minigen.evaluation.assets import download_assets, verify_asset


def main() -> None:
    """选择模型后执行下载；VQA 只有显式指定才会获取。

    Raises:
        SystemExit: 参数不合法。
        OSError/ValueError/KeyError/CalledProcessError: 资产缺失、身份错误或下载失败。

    副作用:
        可按参数访问网络和写本地模型目录；打印校验回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=Path("configs/models/evaluation.json"))
    parser.add_argument("--models", nargs="+", choices=("inception", "clip", "vqa"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = {name: str(verify_asset(args.root, args.lock, name)) for name in args.models}
    else:
        result = download_assets(args.root, args.lock, args.models)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
