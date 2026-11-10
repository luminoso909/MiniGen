"""独立验证汇总不把模拟、残缺与不同计算口径当作正式可比结果。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from minigen._typing import JsonDict
from minigen.evaluation.reporting import build_report, comparison_rows, select_comparison


def score_fixture() -> JsonDict:
    """建立显式构造的评分回执，不代表任何模型成绩。

    Results:
        JsonDict: 便于修改证据边界的最小结果。
    """
    return {
        "scope": "real_local",
        "architecture": "b0",
        "run_id": "fixture",
        "checkpoint_step": 10,
        "checkpoint_sha256": "checkpoint",
        "benchmark_sha256": "benchmark",
        "complete": True,
        "exploratory": True,
        "coverage": {"valid": 4, "attempted": 4},
        "identity": {"asset_lock_sha256": "locked", "metrics": ["kid"]},
        "metrics": {"kid": {"status": "completed", "value": -0.01}},
    }


def comparison_fixture() -> list[JsonDict]:
    """创建同一实际硬件口径的三架构合成比较行。

    Results:
        list: 每行已明确完整性及评测身份，预算为测试数字。
    """
    return [
        {
            "scope": "formal",
            "architecture": name,
            "benchmark_sha256": "benchmark",
            "evaluation_identity": "same-evaluator-sha256",
            "device": "cuda",
            "hardware": "fixture-GPU-A",
            "precision": "bf16",
            "complete": True,
            "checkpoint_step": 10,
            "gpu_hours": value,
        }
        for name, value in [("b0", 1.0), ("b1", 0.95), ("b2", 0.5)]
    ]


def test_comparison_excludes_simulation_and_mismatched_training_steps() -> None:
    """模拟指标不进入数值列，末次资源不能冒充早期检查点资源。

    Raises:
        AssertionError: 模拟分数泄漏或错误连接训练资源。
    """
    score = score_fixture()
    training = {
        "run_id": "fixture",
        "step": 20,
        "device": "cuda",
        "precision": "bf16",
        "elapsed_seconds": 3600,
        "performance": {"samples_per_optimizer_second": 100},
        "memory": {"cuda_peak_allocated": 999},
    }
    row = comparison_rows([score], [training])[0]
    assert row["kid"] == -0.01
    assert row["training_seconds"] is None and row["gpu_hours"] is None
    assert row["samples_per_second"] is None and row["cuda_peak_bytes"] is None
    score["scope"] = "simulated"
    row = comparison_rows([score], [])[0]
    assert row["kid"] is None and row["kid_status"] == "simulated_excluded"
    score["scope"] = "real_local"
    score["metrics"]["kid"]["scope"] = "simulated"
    assert comparison_rows([score], [])[0]["kid"] is None


def test_comparison_uses_existing_budget_points_and_rejects_mixed_evidence() -> None:
    """预算只选择已存在的检查点，不能比较残缺、模拟或异构评测。

    Raises:
        AssertionError: 超预算、插值或不匹配口径未被拒绝。
    """
    rows = comparison_fixture()
    selected = select_comparison(rows, "gpu_hours", 1.0, tolerance=0.1)
    assert len(selected["selected"]) == 2 and selected["missing"] == ["b2"]
    assert selected["selected"][1]["actual"] == 0.95
    assert not select_comparison(rows, "steps", 11)["selected"]
    changes = [
        ("scope", "simulated"),
        ("complete", False),
        ("hardware", "fixture-GPU-B"),
        ("evaluation_identity", "other-evaluator-sha256"),
    ]
    for key, value in changes:
        altered = [dict(r) for r in rows]
        altered[0][key] = value
        with pytest.raises(ValueError):
            select_comparison(altered, "gpu_hours", 1.0)
    for row in rows:
        row["hardware"] = None
    with pytest.raises(ValueError):
        select_comparison(rows, "gpu_hours", 1.0)


def test_report_preserves_failed_gallery_rows_and_relative_images(
    tmp_path: Path,
) -> None:
    """报告展示全部成功与失败条目，图片相对生成清单定位。

    Args:
        tmp_path: 独立图像与报告目录。

    Raises:
        AssertionError: 图片丢失、失败被过滤或覆盖已有报告。

    副作用:
        产生合成 PNG、指标回执及独立 HTML/CSV/图表，不运行模型。
    """
    images = tmp_path / "generation"
    images.mkdir()
    Image.new("RGB", (8, 8), "red").save(images / "sample.png")
    (images / "samples.json").write_text("{}")
    score = score_fixture()
    score.update(
        complete=False,
        generation_path=str(images / "samples.json"),
        coverage={"valid": 1, "attempted": 2},
        records=[
            {"caption": "fixture image", "status": "completed", "image_path": "sample.png"},
            {
                "caption": "failed fixture",
                "status": "failed",
                "image_path": None,
                "validation_error": "fixture error",
            },
        ],
    )
    path = tmp_path / "score.json"
    path.write_text(json.dumps(score))
    output = tmp_path / "report"
    result = build_report([path], [], output)
    assert result["status"] == "completed"
    gallery = (output / "gallery.html").read_text()
    assert "data:image/png;base64," in gallery
    assert "failed fixture" in gallery and "fixture error" in gallery
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["rows"][0]["complete"]
    assert (output / "metrics.csv").is_file() and (output / "quality_curves.png").is_file()
    with pytest.raises(FileExistsError):
        build_report([path], [], output)
