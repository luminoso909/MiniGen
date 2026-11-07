"""测试缓存写入中断、恢复身份、持久数据和发布边界。"""

import json
from pathlib import Path

import numpy as np
import pytest

from minigen.data.cache_writer import CacheLayouts, IncrementalCacheWriter

LAYOUTS: CacheLayouts = {
    split: {"text": ((3, 2), np.dtype("<f2")), "mask": ((3, 1), np.dtype("?"))}
    for split in ("train", "validation")
}


def test_interrupt_resume_and_publish(
    tmp_path: Path,
) -> None:
    """验证恢复只跳过已提交前缀，并在两划分完整后发布。

    Args:
        tmp_path: pytest 提供的独立工作目录。

    Raises:
        AssertionError: 行、空条件、索引或发布位置不符合约定。
    """
    target = tmp_path / "cache"
    writer = IncrementalCacheWriter(target, resume_identity={"version": 1}, layouts=LAYOUTS)
    with writer:
        assert not writer.has_blank
        writer.write_blank(np.zeros(2, dtype="<f2"), np.array([True]))
        train = writer.arrays("train")
        train["text"][:] = [[1, 2], [90, 90], [90, 90]]
        train["mask"][0] = True
        writer.commit("train", 1)
        with pytest.raises(ValueError, match="incomplete"):
            writer.publish({"verified": True})
        with pytest.raises(BlockingIOError):
            with IncrementalCacheWriter(target, resume_identity={"version": 1}, layouts=LAYOUTS):
                pytest.fail("A second writer acquired the same lock")
    assert writer.directory.exists()
    assert not target.exists()

    with IncrementalCacheWriter(target, resume_identity={"version": 1}, layouts=LAYOUTS) as resumed:
        assert resumed.has_blank
        assert resumed.progress("train") == 1
        assert resumed.progress("validation") == 0
        for split in LAYOUTS:
            start = resumed.progress(split)
            arrays = resumed.arrays(split)
            arrays["text"][start:] = [[1, 2], [3, 4], [5, 6]][start:]
            arrays["mask"][start:] = True
            resumed.commit(split, 3)
        resumed.publish({"verified": True})

    assert not writer.directory.exists()
    assert json.loads((target / "index.json").read_text()) == {"verified": True}
    for split in LAYOUTS:
        assert np.array_equal(np.load(target / f"{split}_text.npy"), [[1, 2], [3, 4], [5, 6]])
        assert np.load(target / f"{split}_mask.npy").all()
    assert np.array_equal(np.load(target / "blank_text.npy"), [0, 0])
    assert np.array_equal(np.load(target / "blank_mask.npy"), [True])
    with pytest.raises(FileExistsError):
        with writer:
            pytest.fail("Published cache was reopened for writing")


def test_resume_rejects_identity_and_committed_corruption(
    tmp_path: Path,
) -> None:
    """验证变化的来源与被篡改的已提交数据都不能续跑。

    Args:
        tmp_path: pytest 提供的独立工作目录。

    Raises:
        AssertionError: 发生不安全的续跑。
    """
    target = tmp_path / "cache"
    original = IncrementalCacheWriter(target, resume_identity={"version": 1}, layouts=LAYOUTS)
    with original:
        original.arrays("train")["text"][0] = 1
        original.commit("train", 1)

    with pytest.raises(ValueError, match="resume identity"):
        with IncrementalCacheWriter(target, resume_identity={"version": 2}, layouts=LAYOUTS):
            pytest.fail("Cache resumed under a different identity")

    corrupted = np.load(original.directory / "train_text.npy", mmap_mode="r+")
    corrupted[0, 0] = 2
    corrupted.flush()
    with pytest.raises(ValueError, match="committed data"):
        with original:
            pytest.fail("Cache resumed with changed committed bytes")
