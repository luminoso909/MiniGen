"""Benchmark 题量、隔离、内容锁与正式人工审核门槛。"""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from minigen.evaluation.benchmark import (
    build_prompts,
    content_hash,
    load_benchmark,
    prepare_benchmark,
    validate_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_counts_requirements_and_pairing() -> None:
    """验证 160 主测试、80 验证、五类要求及改写约束复制。

    Raises:
        AssertionError: 数量、类别、配对或题族隔离错误。
    """
    catalog = json.loads((ROOT / "configs/benchmark/prompt_catalog.json").read_text())
    prompts = build_prompts(catalog, [])
    payload = {"prompts": prompts, "references": [], "validation_references": []}
    validate_benchmark(payload)
    assert len(prompts) == 240
    assert {req["kind"] for row in prompts for req in row["requirements"]} == {
        "object",
        "count",
        "color",
        "binding",
        "relation",
    }
    pairs = [row for row in prompts if row["split"] == "paraphrase"]
    assert len(pairs) == 40
    assert not any(row["ood_verified"] for row in prompts)
    assert all(row["semantic_review"] == "pending_human_review" for row in prompts)
    pairs[0]["requirements"] = []
    with pytest.raises(ValueError, match="requirements"):
        validate_benchmark(payload)


def make_data(
    root: Path,
) -> Path:
    """创建彼此隔离、拥有真实哈希的极小 256px 测试图片和清单。

    Args:
        root: pytest 临时项目根。

    Results:
        Path: 测试清单路径。

    副作用:
        写临时图片、清单与配置副本。
    """
    import hashlib

    data = root / "data/processed/artbench_pd_256/v1"
    data.mkdir(parents=True)
    (root / "configs/benchmark").mkdir(parents=True)
    shutil.copy(ROOT / "configs/benchmark/prompt_catalog.json", root / "configs/benchmark")
    shutil.copy(ROOT / "configs/benchmark/sampling_matrix.json", root / "configs/benchmark")
    (root / "configs/models").mkdir()
    shutil.copy(ROOT / "configs/models/evaluation.json", root / "configs/models")
    rows = []
    for index in range(12):
        image = data / f"{index}.png"
        Image.new("RGB", (256, 256), (index * 20, 30, 50)).save(image)
        rows.append(
            {
                "sample_id": str(index),
                "image_path": image.name,
                "group_id": str(index),
                "split": ("train", "validation", "test")[index // 4],
                "caption": "A painting of a dog.",
                "attributes": {},
                "source_id": "fixture",
                "source_revision": "fixture-v1",
                "width": 256,
                "height": 256,
                "mode": "RGB",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "artist": "fixture",
                "fixture": True,
            }
        )
    path = data / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_prepare_content_lock_and_manual_approval(
    tmp_path: Path,
) -> None:
    """验证内容可重建、缺少人工批准拒绝正式读取、篡改被发现。

    Args:
        tmp_path: 临时项目根。

    Raises:
        AssertionError: 版本锁、参考顺序或审核门槛行为不正确。

    副作用:
        写入测试图片、Benchmark 与模拟审核回执，不代表真实人工审核。
    """
    make_data(tmp_path)
    output = tmp_path / "bench"
    first = prepare_benchmark(
        tmp_path,
        output,
        reference_count=2,
        validation_reference_count=2,
        calibration_reference_count=2,
    )
    second = prepare_benchmark(
        tmp_path,
        output,
        reference_count=2,
        validation_reference_count=2,
        calibration_reference_count=2,
    )
    assert first["benchmark_sha256"] == second["benchmark_sha256"]
    payload = load_benchmark(output / "benchmark.json")
    assert len(payload["references"]) == 2
    assert {row["split"] for row in payload["references"]} == {"test"}
    with pytest.raises(ValueError, match="human approval"):
        load_benchmark(output / "benchmark.json", require_approved=True)
    approval = tmp_path / "fixture_approval.json"
    approval.write_text(
        json.dumps(
            {
                "benchmark_sha256": first["benchmark_sha256"],
                "approved": True,
                "reviewer": "fixture_only",
                "reviewed_at": "fixture",
            }
        )
    )
    load_benchmark(output / "benchmark.json", require_approved=True, approval=approval)
    matrix = tmp_path / "configs/benchmark/sampling_matrix.json"
    original = matrix.read_bytes()
    matrix.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="bound asset hash"):
        load_benchmark(output / "benchmark.json", require_approved=True, approval=approval)
    matrix.write_bytes(original)
    payload["prompts"][0]["caption"] += " changed"
    (output / "benchmark.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_benchmark(output / "benchmark.json")


def test_reference_leakage_and_corruption_rejected(
    tmp_path: Path,
) -> None:
    """同组跨划分或参考图片损坏时，准备阶段不能冻结内容。

    Args:
        tmp_path: 临时项目根。

    Raises:
        AssertionError: 泄漏或哈希错误未被拒绝。

    副作用:
        只篡改临时测试清单与图片。
    """
    manifest = make_data(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[8]["group_id"] = rows[0]["group_id"]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        prepare_benchmark(
            tmp_path, tmp_path / "bench", reference_count=4, validation_reference_count=2
        )
    rows[8]["group_id"] = "8"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (manifest.parent / "8.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_benchmark(
            tmp_path, tmp_path / "bench", reference_count=4, validation_reference_count=2
        )


def test_canonical_json_identity() -> None:
    """JSON 键顺序不影响内容哈希，NaN 则拒绝。

    Raises:
        AssertionError: 内容身份受键顺序影响或接纳非有限数值。
    """
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        content_hash({"invalid": float("nan")})
