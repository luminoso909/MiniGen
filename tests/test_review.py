"""匿名页面、人工结果导入和独立校准的契约测试。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from minigen.evaluation.review import (
    compare_calibration,
    export_review,
    import_review,
    prepare_calibration,
)


def review_records(
    root: Path,
) -> list[dict[str, object]]:
    """建立包含私有架构身份与转义测试文本的图片记录。

    Args:
        root: 临时目录。

    Results:
        list: 用于审核工具的两条模拟输入。

    副作用:
        写入测试图片。
    """
    path = root / "b0_secret_model.png"
    Image.new("RGB", (256, 256), "red").save(path)
    return [
        {
            "task_id": str(i),
            "image_path": str(path),
            "architecture": "b0_secret_model",
            "caption": "A red painting </script>",
            "requirements": [{"kind": "color", "subject": "painting", "expected": "red"}],
        }
        for i in range(2)
    ]


def test_anonymity_escape_and_partial_import(
    tmp_path: Path,
) -> None:
    """网页不泄露私有架构/文件名，人工缺项不补分，身份错误被拒绝。

    Args:
        tmp_path: 测试目录。

    Raises:
        AssertionError: 匿名隔离、脚本转义或人工输入验证失效。

    副作用:
        创建模拟审核包和固定测试回答，不记录真实人评。
    """
    package = tmp_path / "package"
    report = export_review(review_records(tmp_path), package)
    page = (package / "review.html").read_text()
    assert "b0_secret_model" not in page
    assert str(tmp_path) not in page
    assert "painting </script>" not in page
    assert "data:image/png;base64," in page and "localStorage" in page
    assert report["status"] == "pending_human_review"
    mapping = json.loads((package / "private_mapping.json").read_text())
    first = mapping["items"][0]["item_id"]
    answers = tmp_path / "fixture_answers.json"
    data = {
        "package_id": report["package_id"],
        "reviewer": "fixture",
        "reviewed_at": "fixture",
        "answers": {
            first: {
                "quality": 3,
                "alignment": None,
                "errors": ["texture"],
                "questions": {"q-000": "correct"},
                "notes": "fixture",
            }
        },
    }
    answers.write_text(json.dumps(data))
    imported = import_review(package, answers, tmp_path / "import.json")
    assert imported["status"] == "partial"
    assert imported["expected_items"] == 2 and imported["answered_items"] == 1
    assert imported["records"][0]["alignment"] is None
    data["answers"][first]["quality"] = 6
    answers.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="scores"):
        import_review(package, answers, tmp_path / "bad.json")
    with pytest.raises(FileExistsError):
        export_review(review_records(tmp_path), package)


def test_calibration_coverage_and_disagreement() -> None:
    """人工未知与自动漏项不计作正确，保留逐类混淆。

    Raises:
        AssertionError: 分母、一致率或漏项数错误。
    """
    human = {
        "source": "user_import",
        "records": [
            {
                "original_id": "a",
                "questions": {
                    "q0": "correct",
                    "q1": "incorrect",
                    "q2": "undecidable",
                    "q3": "correct",
                },
            }
        ],
    }
    automatic = [
        {"task_id": "a", "question_id": "q0", "kind": "count", "verdict": "correct"},
        {"task_id": "a", "question_id": "q1", "kind": "count", "verdict": "correct"},
        {"task_id": "a", "question_id": "q2", "kind": "count", "verdict": "undecidable"},
    ]
    result = compare_calibration(human, automatic)
    assert result["decidable_pairs"] == 2
    assert result["agreement_on_decidable"] == 0.5
    assert result["missing_automatic"] == 1
    with pytest.raises(ValueError, match="separately imported"):
        compare_calibration({"source": "mock"}, automatic)


def test_calibration_pairs_keep_lineage(
    tmp_path: Path,
) -> None:
    """退化图另存、源图保持不变且人工审核仍为空。

    Args:
        tmp_path: 临时目录。

    Raises:
        AssertionError: 来源图被覆盖、派生谱系丢失或审核状态错误。

    副作用:
        在临时目录创建一张测试图和对应退化审核包。
    """
    record = review_records(tmp_path)[0]
    original = Path(record["image_path"]).read_bytes()
    benchmark = {
        "data_root": ".",
        "validation_references": [
            {
                "sample_id": "a",
                "image_path": Path(record["image_path"]).name,
                "sha256": "fixture",
                "caption": "A red painting",
            }
        ],
    }
    result = prepare_calibration(benchmark, tmp_path, tmp_path / "calibration", count=1)
    assert result["real_images"] == result["degraded_images"] == 1
    assert result["status"] == "pending_human_review"
    assert Path(record["image_path"]).read_bytes() == original
    lineage = json.loads((tmp_path / "calibration/lineage.json").read_text())
    assert {row["lineage"]["kind"] for row in lineage} == {"real", "degraded"}
    assert not any(row["caption_is_visual_truth"] for row in lineage)


def test_failed_generation_remains_anonymous_review_item(
    tmp_path: Path,
) -> None:
    """无图片的生成失败保留匿名审核位置，不暴露私有错误或架构。

    Args:
        tmp_path: 临时审核目录。

    Raises:
        AssertionError: 失败样本被删除或泄露内部身份。

    副作用:
        写入包含失败占位的测试页面。
    """
    records = [
        {
            "task_id": "private-b2-run",
            "caption": "A painting of a bird.",
            "image_path": None,
            "status": "failed",
            "error": "b2-private-exception",
        }
    ]
    result = export_review(records, tmp_path / "failed")
    page = (tmp_path / "failed/review.html").read_text()
    assert result["count"] == 1
    assert "private-b2" not in page and "b2-private-exception" not in page
    assert '"image": null' in page
