"""有时限、可续建的分片潜变量缓存；只编码训练与验证集合。"""

import hashlib
import json
import time
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from minigen._typing import JsonDict
from minigen.autoencoders.frozen import (
    FrozenVAE,
    cache_identity,
    hash_file,
    image_batch,
    load_cache,
    write_cache,
)


def identity_digest(
    identity: JsonDict,
) -> str:
    """对 JSON 身份做稳定序列化并计算摘要。

    Args:
        identity: 仅含 JSON 值的完整数据或训练身份。

    Raises:
        TypeError: 身份含不可序列化对象时透传。

    Results:
        str: 对字典键排序后计算的 SHA-256，不受字典插入顺序影响。
    """
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def latent_cache_identity(
    adapter: FrozenVAE,
    records: list[JsonDict],
    manifest_hash: str,
    split: str,
) -> JsonDict:
    """冻结单个训练或验证集合的编码与样本顺序约定。

    Args:
        adapter: 已显式加载、冻结的 VAE，提供模型哈希与缩放规则。
        records: 已按清单顺序筛为同一集合的非空记录，含样本 ID 和图像哈希。
        manifest_hash: 完整原始 JSONL 清单的 SHA-256，包含描述与划分版本。
        split: train 或 validation；缓存构建不读取测试集。

    Raises:
        ValueError: 集合非法、记录为空、混入其他划分、样本 ID 重复或清单摘要格式错误。
        KeyError: 样本或编码器缺少必要身份字段时透传。

    Results:
        JsonDict: 严格缓存身份，包含有序样本表、VAE、预处理、缩放和划分。
    """
    if split not in {"train", "validation"} or not records:
        raise ValueError("A nonempty train or validation split is required")
    if any(record["split"] != split for record in records):
        raise ValueError("Records must belong to exactly the requested split")
    ids = [record["sample_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs are not allowed")
    if len(manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in manifest_hash):
        raise ValueError("A SHA-256 manifest digest is required")
    return {**cache_identity(adapter, records, manifest_hash), "split": split}


def _write_index(
    path: Path,
    index: JsonDict,
) -> None:
    """以临时文件替换缓存索引，防止半个 JSON 被当作完整版本。

    Args:
        path: 缓存目录中的 index.json 路径。
        index: 当前已验证分片及全局身份。

    Raises:
        OSError: 索引不可写或替换失败时透传；TypeError: 非 JSON 数据无法序列化。

    副作用:
        创建或替换索引文件；不会把尚未保存完成的分片写入索引。
    """
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _shard_identity(
    identity: JsonDict,
    start: int,
    end: int,
) -> JsonDict:
    """从全局身份派生某个连续分片的身份。

    Args:
        identity: 包含有序 samples 表的全局缓存身份。
        start: 分片包含的第一条样本下标。
        end: 不包含的结束下标。

    Raises:
        KeyError: 全局身份没有 samples 表时透传。

    Results:
        JsonDict: 全局摘要、下标范围及对应有序样本；避免每片复制全量样本表。
    """
    return {
        "schema_version": 1,
        "cache_id": identity_digest(identity),
        "start": start,
        "end": end,
        "samples": identity["samples"][start:end],
    }


def _validated_shards(
    directory: Path,
    index: JsonDict,
    identity: JsonDict,
) -> list[Float[Tensor, "b c h w"]]:
    """验证索引的连续分片、身份、文件哈希与有限值。

    Args:
        directory: 分片所在目录，所有文件名必须符合确定性命名。
        index: 已从 JSON 读取的索引，可能仍不完整。
        identity: 本次期望的全局缓存身份。

    Raises:
        ValueError: 全局身份、分片顺序、数量、状态、形状或有限性不符。
        OSError/KeyError: 缺失文件或元数据字段时透传。

    Results:
        list[Float[Tensor, "b c h w"]]: 已核验的 CPU FP32 分片，按样本顺序排列。
    """
    if index["schema_version"] != 1 or index["identity"] != identity:
        raise ValueError("Incompatible full latent cache identity")
    total = len(identity["samples"])
    if index["total_count"] != total:
        raise ValueError("Cache total count differs from the manifest")
    cursor = 0
    arrays = []
    spatial_shape = None
    for shard in index["shards"]:
        start, end = shard["start"], shard["end"]
        name = f"latents_{start:06d}_{end:06d}.npy"
        if start != cursor or not start < end <= total or shard["file"] != name:
            raise ValueError("Invalid shard order, range or filename")
        values = load_cache(directory / name, _shard_identity(identity, start, end))
        if values.ndim != 4 or len(values) != end - start or not torch.isfinite(values).all():
            raise ValueError("Invalid latent shard shape or nonfinite values")
        shape = tuple(values.shape[1:])
        if spatial_shape is not None and shape != spatial_shape:
            raise ValueError("Latent spatial shape changed between shards")
        spatial_shape = shape
        arrays.append(values)
        cursor = end
    if index["completed_count"] != cursor or bool(index["completed"]) != (cursor == total):
        raise ValueError("Cache completion marker disagrees with verified shards")
    return arrays


def build_latent_cache(
    adapter: FrozenVAE,
    records: list[JsonDict],
    data_root: str | Path,
    manifest_hash: str,
    cache_dir: str | Path,
    *,
    split: str,
    batch_size: int = 8,
    shard_size: int = 256,
    max_seconds: float = 900,
) -> JsonDict:
    """在时间预算内编码缓存，复用已校验的完整分片并原子更新索引。

    Args:
        adapter: 显式冻结 VAE；不触发下载，不改变其后验均值策略。
        records: 已按原清单顺序筛为 train 或 validation 的记录。
        data_root: 原图根目录；编码前重新核验文件哈希和路径边界。
        manifest_hash: 完整 JSONL 清单的 SHA-256。
        cache_dir: 本集合独立缓存目录，写分片和 index.json。
        split: train 或 validation，禁止测试数据参与本周缓存。
        batch_size: 每次 VAE 编码的图像数，必须为正。
        shard_size: 完整提交的分片最大样本数，必须为正。
        max_seconds: 本次调用时间上限，必须为正；仅能在编码 batch 之间停止。

    Raises:
        ValueError: 参数、源文件哈希、路径、身份、已有缓存或编码结果非法。
        OSError/KeyError: 原图或索引缺失、不可读或必要字段缺项；模型错误透传。

    Results:
        JsonDict: 本次执行摘要。
            status/completed: complete 或 time_limit，以及是否完成整份集合。
            completed_count/total_count: 已提交样本数和应有样本数。
            index_path/identity: 缓存入口和完整身份，用于独立重载。
            discarded_partial_count: 超时前已编码但尚未成片的样本数，下次重新编码。
            elapsed_seconds: 包含已有缓存校验的实测耗时。

    副作用:
        读取原图和已有缓存，执行冻结编码，创建目录并写原子分片与索引。
        只保留完整分片；超时不把部分结果标为完整数据集。
    """
    started = time.monotonic()
    if batch_size <= 0 or shard_size <= 0 or max_seconds <= 0:
        raise ValueError("Batch size, shard size and time budget must be positive")
    identity = latent_cache_identity(adapter, records, manifest_hash, split)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "index.json"
    if path.exists():
        index = json.loads(path.read_text())
        _validated_shards(directory, index, identity)
    else:
        index = {
            "schema_version": 1,
            "identity": identity,
            "total_count": len(records),
            "completed_count": 0,
            "completed": False,
            "shards": [],
        }
        _write_index(path, index)
    root = Path(data_root).resolve()
    partial_count = 0
    for start in range(index["completed_count"], len(records), shard_size):
        end = min(start + shard_size, len(records))
        chunks = []
        for offset in range(start, end, batch_size):
            if time.monotonic() - started >= max_seconds:
                break
            subset = records[offset : min(offset + batch_size, end)]
            for record in subset:
                image_path = (root / record["image_path"]).resolve()
                if not image_path.is_relative_to(root) or hash_file(image_path) != record["sha256"]:
                    raise ValueError("Source image path or hash differs from the manifest")
            values = adapter.encode(image_batch(subset, root)).detach().cpu().float()
            if values.ndim != 4 or len(values) != len(subset) or not torch.isfinite(values).all():
                raise ValueError("Encoder returned invalid latent shapes or nonfinite values")
            chunks.append(values)
        encoded = sum(len(chunk) for chunk in chunks)
        if encoded != end - start:
            partial_count = encoded
            break
        values = torch.cat(chunks)
        if index["shards"]:
            previous = index["shards"][0]
            first_meta = json.loads((directory / previous["file"]).with_suffix(".json").read_text())
            if list(values.shape[1:]) != first_meta["shape"][1:]:
                raise ValueError("Encoder spatial shape changed between shards")
        name = f"latents_{start:06d}_{end:06d}.npy"
        write_cache(directory / name, values, _shard_identity(identity, start, end))
        index["shards"].append({"start": start, "end": end, "file": name})
        index["completed_count"] = end
        index["completed"] = end == len(records)
        _write_index(path, index)
    return {
        "status": "complete" if index["completed"] else "time_limit",
        "completed": index["completed"],
        "completed_count": index["completed_count"],
        "total_count": len(records),
        "index_path": str(path),
        "identity": identity,
        "discarded_partial_count": partial_count,
        "elapsed_seconds": time.monotonic() - started,
    }


class CachedLatentDataset(Dataset[Float[Tensor, "c h w"]]):
    """只有完整、身份一致并通过载荷校验的缓存才能成为训练数据集。

    Args:
        cache_dir: 包含 index.json 与 NPY 分片的缓存目录。
        expected_identity: 调用方从本次原始清单和 VAE 重建的预期身份。

    Raises:
        ValueError: 缓存不完整、身份或分片不合法。
        OSError/KeyError: 文件或必要元数据缺失时透传。

    Results:
        CachedLatentDataset: CPU FP32 数据集；latents 为完整 [n_s,c,h,w]，data_id 为稳定身份摘要。

    副作用:
        读取全部分片到内存，不下载数据或编码器。
    """

    def __init__(
        self,
        cache_dir: str | Path,
        expected_identity: JsonDict,
    ) -> None:
        """核验索引和全部分片，再建立按原顺序访问的潜变量张量。

        Args:
            cache_dir: 本地缓存目录，入口必须是已完成的 index.json。
            expected_identity: 本次希望使用的完整版本，不能直接盲信索引内的旧身份。

        Raises:
            ValueError: 身份、完整性、顺序、哈希、形状或有限值校验失败。
            OSError/KeyError: 文件读取或必要字段缺失时透传。

        副作用:
            从磁盘加载已验证分片，建立只读使用约定的 CPU 数据张量。
        """
        directory = Path(cache_dir)
        index = json.loads((directory / "index.json").read_text())
        chunks = _validated_shards(directory, index, expected_identity)
        if not index["completed"] or not chunks:
            raise ValueError("A complete latent cache is required before training")
        self.latents = torch.cat(chunks)
        self.identity = json.loads(json.dumps(expected_identity))
        self.data_id = identity_digest(expected_identity)

    def __len__(
        self,
    ) -> int:
        """返回完整缓存的样本数。

        Results:
            int: 与冻结身份有序 samples 数量一致的 n_s。
        """
        return len(self.latents)

    def __getitem__(
        self,
        index: int,
    ) -> Float[Tensor, "c h w"]:
        """按冻结顺序取得一个潜变量视图，调用方不应原地修改。

        Args:
            index: Python 整数样本下标，负下标沿用 Tensor 语义。

        Raises:
            IndexError: 下标越界时由 Tensor 透传。

        Results:
            Float[Tensor, "c h w"]: 单样本 CPU FP32 潜变量，不含 batch 轴。
        """
        return self.latents[index]
