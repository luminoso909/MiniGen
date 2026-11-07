"""测试本地 ArtBench 的数据锁、完整写入和重复样本隔离。"""

from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image

from minigen.data.artbench import prepare_artbench, sha256_file
from minigen.data.manifest import load_manifest


def fixture_token_count(
    text: str,
) -> int:
    """返回集成测试用的单词数，真实运行使用固定 CLIP tokenizer。

    Args:
        text: 用于合成 Parquet 样本的描述。

    Results:
        int: 以空白分词后的数量。
    """
    return len(text.split())


def test_prepare_artbench_preserves_grouped_rows_and_rechecks_identity(
    tmp_path: Path,
) -> None:
    """用实际 Parquet 校验 RGB 图像、近重分组、缓存和文件身份检查。

    Args:
        tmp_path: pytest 提供的临时数据目录。

    Raises:
        AssertionError: 重复数据跨集合、清单丢行或身份错误未拦截时失败。

    副作用:
        在测试临时目录写入 Parquet 及数据准备产物。
    """
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    buffer = BytesIO()
    Image.new("RGB", (256, 256), "navy").save(buffer, format="JPEG")
    row = {
        "image": {"bytes": buffer.getvalue(), "path": None},
        "caption": "The image shows a blue painting.",
        "caption_jp": "青い絵",
        "artist": "example-artist",
        "url": "https://uploads1.wikiart.org/example/art.jpg",
    }
    shard = tmp_path / "train.parquet"
    parquet.write_table(pa.Table.from_pylist([row, row]), shard)
    config = {
        "repo_id": "example/artbench",
        "revision": "fixture-only",
        "version": "fixture",
        "image_size": 256,
        "expected_rows": 2,
        "files": [
            {"path": shard.name, "bytes": shard.stat().st_size, "sha256": sha256_file(shard)}
        ],
    }
    config_path = tmp_path / "source.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_dir = tmp_path / "prepared"
    summary = prepare_artbench(config_path, tmp_path, output_dir, fixture_token_count, "fixture")
    rows = load_manifest(output_dir / "manifest.jsonl", output_dir, verify_hashes=True)
    assert len(rows) == summary["manifest_rows"] == 2
    assert len({(r["split"], r["group_id"]) for r in rows}) == 1
    assert rows[0]["caption"] == "A blue painting."
    assert summary["exact_pixel_duplicate_rows"] == 1
    assert summary["decode_failures"] == 0
    assert prepare_artbench(config_path, tmp_path, output_dir, fixture_token_count, "fixture") == (
        summary
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_artbench(config_path, tmp_path, output_dir, fixture_token_count, "different")
    (output_dir / rows[0]["image_path"]).write_bytes(b"unexpected bytes")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        prepare_artbench(config_path, tmp_path, output_dir, fixture_token_count, "fixture")
