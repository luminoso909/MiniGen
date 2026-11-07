"""可中断续跑的本地 NumPy 缓存写入器。"""

import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import numpy as np

from minigen._typing import JsonDict

CacheLayouts = dict[str, dict[str, tuple[tuple[int, ...], np.dtype]]]
JOURNAL = ".progress.json"


def _sync_file(
    path: Path,
) -> None:
    """把已经写入的文件刷新到磁盘。

    Args:
        path: 需要同步的普通文件路径。

    Raises:
        OSError: 打开或同步失败。

    副作用:
        触发文件数据同步。
    """
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json(
    path: Path,
    value: JsonDict,
) -> None:
    """原子替换 JSON；意外退出时保留上次完整的进度。

    Args:
        path: 目标 JSON 文件。
        value: 待序列化的记录。

    Raises:
        OSError/TypeError: 写入、同步、替换或序列化失败。

    副作用:
        写入同目录临时文件并替换目标。
    """
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _chunk_digest(
    arrays: dict[str, np.memmap],
    start: int,
    stop: int,
) -> str:
    """对相同行段的所有数组按固定字段顺序计算摘要。

    Args:
        arrays: 一个划分的逐字段连续映射。
        start: 首行下标（含）。
        stop: 末行下标（不含）。

    Results:
        str: 该段实际存储字节的 SHA-256。
    """
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(name.encode() + b"\0")
        digest.update(memoryview(array[start:stop]).cast("B"))
    return digest.hexdigest()


class IncrementalCacheWriter:
    """持独占锁增量写入缓存，恢复时核对身份与已提交的字节。

    Args:
        output: 发布目录，不能覆盖。
        resume_identity: 来源、编码器、设备及运行选项等恢复契约。
        layouts: 各划分、字段的完整行数、形状和 dtype。

    Raises:
        ValueError: 映射布局、恢复身份或已写入内容不符。
        OSError: 获取锁、访问文件或发布失败。

    副作用:
        在同级目录保存锁、暂存数组及进度；发布时原子重命名。
    """

    def __init__(
        self,
        output: Path,
        *,
        resume_identity: JsonDict,
        layouts: CacheLayouts,
    ) -> None:
        """固定写入契约；进入上下文前不创建文件。

        Args:
            output: 最终缓存目录。
            resume_identity: 必须与前次写入完全一致的 JSON 身份。
            layouts: 按划分和字段声明的形状与存储精度。

        Raises:
            ValueError: 布局为空、含不安全名称、无效行数或空条件布局不一致。
            TypeError: 恢复身份不能表示为 JSON。
        """
        self.output = output.resolve()
        self.directory = self.partial_path(self.output)
        self._lock: BinaryIO | None = None
        self._maps: dict[str, dict[str, np.memmap]] = {}
        self._journal: JsonDict = {}
        self._layouts = layouts
        self._rows: dict[str, int] = {}
        self._blank_layout: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        if not layouts:
            raise ValueError("Cache needs at least one split")
        for split, fields in layouts.items():
            if not re.fullmatch(r"[a-z][a-z0-9]*", split) or split == "blank":
                raise ValueError(f"Invalid cache split name: {split}")
            if not {"text", "mask"} <= fields.keys() or any(
                not re.fullmatch(r"[a-z][a-z0-9]*", name) for name in fields
            ):
                raise ValueError(f"Invalid cache field names: {split}")
            rows = {shape[0] for shape, _ in fields.values() if shape}
            if len(rows) != 1 or any(
                not shape or any(axis < 1 for axis in shape) or dtype.hasobject
                for shape, dtype in fields.values()
            ):
                raise ValueError(f"Invalid cache array layout: {split}")
            self._rows[split] = rows.pop()
            blank = {
                f"blank_{name}": (fields[name][0][1:], fields[name][1]) for name in ("text", "mask")
            }
            if self._blank_layout and self._blank_layout != blank:
                raise ValueError("Cache splits disagree on blank layouts")
            self._blank_layout = blank
        self._contract = {
            "schema": "minigen.incremental-cache.v1",
            "identity": json.loads(json.dumps(resume_identity)),
            "layouts": {
                split: {name: [list(shape), dtype.str] for name, (shape, dtype) in fields.items()}
                for split, fields in layouts.items()
            },
        }

    @staticmethod
    def partial_path(
        output: Path,
    ) -> Path:
        """从发布目录派生唯一的同级暂存目录。

        Args:
            output: 最终缓存目录。

        Results:
            Path: 重启时仍使用的同级暂存目录。
        """
        return output.with_name(f".{output.name}.partial")

    def __enter__(self) -> "IncrementalCacheWriter":
        """获取独占锁并新建或恢复缓存写入。

        Raises:
            BlockingIOError: 另一进程正在写入同一缓存。
            FileExistsError: 正式缓存已存在。
            RuntimeError: 写入器已经处于上下文中。
            ValueError: 暂存身份、布局或进度不符。
            OSError: 锁定或暂存访问失败。

        Results:
            IncrementalCacheWriter: 持锁且可以写入的实例。

        副作用:
            打开锁文件和可写映射，首次创建暂存目录。
        """
        if self._lock is not None:
            raise RuntimeError("Cache writer already open")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        lock = (self.output.parent / f".{self.output.name}.lock").open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.output.exists():
                raise FileExistsError(self.output)
            if not self.directory.exists():
                self._create()
            self._open()
        except BaseException:
            self._maps.clear()
            lock.close()
            raise
        self._lock = lock
        return self

    def _create(self) -> None:
        """创建一套完整的初始数组与空进度后，再使其可被恢复。

        Raises:
            OSError: 文件创建、同步或重命名失败。

        副作用:
            创建零进度的暂存目录。
        """
        with tempfile.TemporaryDirectory(prefix=".cache-init-", dir=self.output.parent) as root:
            staging = Path(root) / "payload"
            staging.mkdir()
            for split, fields in self._layouts.items():
                for name, (shape, dtype) in fields.items():
                    path = staging / f"{split}_{name}.npy"
                    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
                    del array
                    _sync_file(path)
            _write_json(
                staging / JOURNAL,
                {
                    "contract": self._contract,
                    "chunks": {split: [] for split in self._layouts},
                    "blank": {},
                },
            )
            staging.rename(self.directory)

    def _open(self) -> None:
        """打开暂存数组，重算并检查所有已提交的段与空条件摘要。

        Raises:
            ValueError: 身份、布局、进度、已提交数据或空条件不符。
            OSError: 文件读取或映射失败。

        副作用:
            更新实例持有的映射与进度。
        """
        state = json.loads((self.directory / JOURNAL).read_text(encoding="utf-8"))
        if state.get("contract") != self._contract:
            raise ValueError("Partial cache resume identity or layout mismatch")
        chunks = state.get("chunks")
        blank = state.get("blank")
        if (
            not isinstance(chunks, dict)
            or chunks.keys() != self._layouts.keys()
            or not isinstance(blank, dict)
            or (blank and blank.keys() != self._blank_layout.keys())
        ):
            raise ValueError("Invalid partial cache journal")
        for split, fields in self._layouts.items():
            arrays = {}
            for name, (shape, dtype) in fields.items():
                path = self.directory / f"{split}_{name}.npy"
                # r+ 会在底层扩展文件；先用只读映射验证文件没有被截断。
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError(f"Partial cache array layout mismatch: {path.name}")
                if path.stat().st_size != array.offset + array.nbytes:
                    raise ValueError(f"Partial cache array byte size mismatch: {path.name}")
                arrays[name] = np.load(path, mmap_mode="r+", allow_pickle=False)
            start = 0
            if not isinstance(chunks[split], list):
                raise ValueError(f"Invalid partial cache chunk list: {split}")
            for item in chunks[split]:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or type(item[0]) is not int
                    or not start < item[0] <= self._rows[split]
                    or _chunk_digest(arrays, start, item[0]) != item[1]
                ):
                    raise ValueError(f"Partial cache committed data mismatch: {split}/{start}")
                start = item[0]
            self._maps[split] = arrays
        for name, digest in blank.items():
            path = self.directory / f"{name}.npy"
            array = np.load(path, allow_pickle=False)
            if (array.shape, array.dtype) != self._blank_layout[name] or hashlib.sha256(
                path.read_bytes()
            ).hexdigest() != digest:
                raise ValueError(f"Partial cache blank data mismatch: {name}")
        self._journal = state

    def _require_open(self) -> None:
        """确认持有有效写入上下文。

        Raises:
            RuntimeError: 尚未进入上下文、已经发布或已经退出。
        """
        if self._lock is None or not self._maps:
            raise RuntimeError("Cache writer is not open")

    def arrays(
        self,
        split: str,
    ) -> dict[str, np.memmap]:
        """获取指定划分的可写映射，调用方只写尚未提交的行。

        Args:
            split: 已声明的划分名称。

        Raises:
            RuntimeError: 当前没有有效写入上下文。
            KeyError: 划分没有声明。

        Results:
            dict[str, np.memmap]: 字段名到可写 NumPy 映射的字典。
        """
        self._require_open()
        return self._maps[split].copy()

    def progress(
        self,
        split: str,
    ) -> int:
        """读取该划分的已提交前缀长度。

        Args:
            split: 已声明的划分名称。

        Raises:
            RuntimeError: 当前没有有效写入上下文。
            KeyError: 划分没有声明。

        Results:
            int: 恢复后第一条需要写入的行下标。
        """
        self._require_open()
        chunks = self._journal["chunks"][split]
        return chunks[-1][0] if chunks else 0

    def commit(
        self,
        split: str,
        next_offset: int,
    ) -> None:
        """先同步数组再提交连续行段；未提交行在恢复时可重写。

        Args:
            split: 已声明的划分名称。
            next_offset: 本次写入后的首条未处理行下标。

        Raises:
            RuntimeError: 当前没有有效写入上下文。
            KeyError: 划分没有声明。
            ValueError: 行下标不在已提交位置与划分结尾之间。
            OSError: 数组或进度同步失败。

        副作用:
            同步指定划分的映射并原子更新进度。
        """
        start = self.progress(split)
        if type(next_offset) is not int or not start < next_offset <= self._rows[split]:
            raise ValueError(f"Invalid cache progress for {split}: {start} -> {next_offset}")
        arrays = self._maps[split]
        for name, array in arrays.items():
            array.flush()
            _sync_file(self.directory / f"{split}_{name}.npy")
        chunk = [next_offset, _chunk_digest(arrays, start, next_offset)]
        chunks = self._journal["chunks"]
        updated = {**self._journal, "chunks": {**chunks, split: [*chunks[split], chunk]}}
        _write_json(self.directory / JOURNAL, updated)
        self._journal = updated

    @property
    def has_blank(self) -> bool:
        """判断空条件是否已持久化并记入进度。

        Raises:
            RuntimeError: 当前没有有效写入上下文。

        Results:
            bool: 是否已提交空条件。
        """
        self._require_open()
        return bool(self._journal["blank"])

    def write_blank(
        self,
        text: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """写入独立的空条件并登记两个文件的摘要。

        Args:
            text: 与逐样本文本表示相同形状和 dtype 的空条件。
            mask: 与逐样本掩码相同形状和 dtype 的空条件掩码。

        Raises:
            RuntimeError: 当前没有有效写入上下文。
            ValueError: 空条件已提交，或数组形状和 dtype 不符。
            OSError: 写入或同步失败。

        副作用:
            写入空条件映射并原子更新进度。
        """
        if self.has_blank:
            raise ValueError("Blank conditioning was already committed")
        values = {"blank_text": text, "blank_mask": mask}
        if any(
            (value.shape, value.dtype) != self._blank_layout[name] for name, value in values.items()
        ):
            raise ValueError("Invalid blank conditioning layout")
        hashes = {}
        for name, value in values.items():
            path = self.directory / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            _sync_file(path)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        updated = {**self._journal, "blank": hashes}
        _write_json(self.directory / JOURNAL, updated)
        self._journal = updated

    def publish(
        self,
        metadata: JsonDict,
    ) -> None:
        """所有划分与空条件完成后，才对读者原子发布正式目录。

        Args:
            metadata: 包含来源、样本顺序和完整文件摘要的索引。

        Raises:
            RuntimeError: 当前没有有效写入上下文。
            ValueError: 存在未提交行或缺少空条件。
            FileExistsError: 发布目标已经存在。
            OSError/TypeError: 无法写入索引或原子重命名。

        副作用:
            写入 index.json，将暂存目录改名为正式目录，并结束写入。
        """
        if not self.has_blank or any(
            self.progress(split) != rows for split, rows in self._rows.items()
        ):
            raise ValueError("Cannot publish an incomplete cache")
        if self.output.exists():
            raise FileExistsError(self.output)
        _write_json(self.directory / "index.json", metadata)
        self.directory.rename(self.output)
        self._maps.clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """释放进程锁与自身的数组映射；保留未发布进度。

        Args:
            exc_type: 上下文异常类型，无异常时为 None。
            exc: 上下文异常实例，无异常时为 None。
            traceback: 上下文异常追踪，无异常时为 None。

        Raises:
            OSError: 关闭锁文件失败时透传。

        副作用:
            释放锁并丢弃实例持有的映射；不吞掉原异常。
        """
        self._maps.clear()
        if self._lock is not None:
            self._lock.close()
            self._lock = None
