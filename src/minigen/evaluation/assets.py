"""显式下载并校验评测资产；推理路径只允许读取固定的本地文件。"""

import json
import subprocess
from pathlib import Path

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file


def asset_spec(
    root: Path,
    lock: Path,
    name: str,
) -> tuple[Path, JsonDict]:
    """解析单个评测模型的固定目录及文件身份。

    Args:
        root: 项目或同步后的资产根目录。
        lock: 相对 root 或绝对的模型锁文件。
        name: models 中登记的模型名。

    Raises:
        ValueError: 路径逃出资产根或身份缺失。
        OSError/KeyError: 锁文件不能读取或模型未登记。

    Results:
        tuple: 本地模型目录和含 files 的身份字典。
    """
    root = root.resolve()
    spec = json.loads((root / lock).read_text())["models"][name]
    folder = (root / spec["local_path"]).resolve()
    if not folder.is_relative_to(root) or not spec.get("files"):
        raise ValueError("Asset directory or file identity is invalid")
    for relative, identity in spec["files"].items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or len(identity["sha256"]) != 64:
            raise ValueError("Unsafe asset path or missing SHA-256")
    return folder, spec


def verify_asset(
    root: Path,
    lock: Path,
    name: str,
) -> Path:
    """验证已下载文件的大小和哈希，不隐式访问网络。

    Args:
        root: 资产根目录。
        lock: 固定模型锁。
        name: 评测模型名。

    Raises:
        ValueError: 文件大小、SHA-256 或目录身份不匹配。
        OSError/KeyError: 文件缺失或锁不能读取。

    Results:
        Path: 可以传给离线模型加载器的目录。
    """
    folder, spec = asset_spec(root, lock, name)
    for relative, identity in spec["files"].items():
        path = folder / relative
        if path.stat().st_size != identity["bytes"] or hash_file(path) != identity["sha256"]:
            raise ValueError(f"Asset hash mismatch: {name}/{relative}")
    return folder


def download_assets(
    root: Path,
    lock: Path,
    names: list[str],
) -> JsonDict:
    """仅获取明确指定的资产，已匹配文件复用，部分下载可续传。

    Args:
        root: 资产根目录。
        lock: 具有固定 HTTPS URL、字节数与 SHA-256 的锁。
        names: 本次显式获取的模型名；不自动追加 VQA。

    Raises:
        ValueError: URL、已存在文件或完成下载的哈希不符。
        OSError/KeyError/CalledProcessError: 文件、身份或下载失败。

    Results:
        JsonDict: 模型目录、实际新下载数和校验通过状态。

    副作用:
        请求公开网络，在忽略目录写入 partial 文件并在校验通过后原子发布。
    """
    receipts = {}
    for name in names:
        folder, spec = asset_spec(root, lock, name)
        downloaded = 0
        for relative, identity in spec["files"].items():
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if (
                    destination.stat().st_size != identity["bytes"]
                    or hash_file(destination) != identity["sha256"]
                ):
                    raise ValueError(f"Existing asset differs: {destination}")
                continue
            url = identity["url"]
            if not url.startswith("https://"):
                raise ValueError("Explicit asset downloads require HTTPS")
            partial = destination.with_suffix(destination.suffix + ".partial")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
            if (
                partial.stat().st_size != identity["bytes"]
                or hash_file(partial) != identity["sha256"]
            ):
                raise ValueError(f"Downloaded asset differs: {destination}")
            partial.replace(destination)
            downloaded += 1
        receipts[name] = {"path": str(verify_asset(root, lock, name)), "downloaded": downloaded}
    return {"status": "passed", "models": receipts}
