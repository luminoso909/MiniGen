"""通用逐图清单读取、路径隔离和图像身份校验。"""

import hashlib
import json
from pathlib import Path

from minigen._typing import JsonDict


def load_manifest(
    path: str | Path,
    data_root: str | Path,
    verify_hashes: bool = False,
) -> list[JsonDict]:
    """读取逐图 JSONL，拒绝示例、越界路径、重复标识和可选的哈希错误。

    Args:
        path: JSONL 清单路径，每行一个样本。
        data_root: 所有相对图像路径必须位于此根目录内。
        verify_hashes: 是否实际读取文件并校验 SHA-256。

    Raises:
        ValueError: 示例记录、哈希缺失、必要字段缺失、重复 ID、路径越界或哈希不符。
        JSONDecodeError/OSError: 清单或图像文件读取失败时透传。

    Results:
        list[JsonDict]: 按文件顺序排列的样本字典列表，不执行模型加载。
    """
    root = Path(data_root).resolve()
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    ids = set()
    for record in records:
        if record.get("example_only") or not record.get("sha256"):
            raise ValueError("Example or missing image hash")
        required = {
            "sample_id",
            "image_path",
            "group_id",
            "split",
            "caption",
            "attributes",
            "source_id",
            "source_revision",
            "width",
            "height",
            "mode",
        }
        if not required.issubset(record):
            raise ValueError("Missing manifest field")
        path = (root / record["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes data root")
        if record["sample_id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(record["sample_id"])
        if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Image hash mismatch")
    return records
