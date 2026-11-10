"""人工要求、真实适配器契约替身、审核导入和校准的模拟端到端测试。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from minigen.autoencoders.sdxl import hash_file
from minigen.bench_command import _review_records
from minigen.evaluation.review import compare_calibration, import_review, prepare_calibration
from minigen.evaluation.scoring import run_scoring


class FixedAnswerBackend:
    """占据真实适配器接口位置的契约替身；答案来自测试常量，非真实推理。

    Args:
        root: 测试根目录。
        asset_lock: 测试资产锁。
        device: 请求设备。
        batch_size: 请求批量。
    """

    simulated = False

    def __init__(
        self,
        root: Path,
        asset_lock: Path,
        device: str,
        batch_size: int,
    ) -> None:
        """接受真实适配器参数但不加载模型。

        Args:
            root: 测试根目录。
            asset_lock: 测试资产锁。
            device: 测试设备。
            batch_size: 测试批量。
        """

    def answer(
        self,
        paths: list[Path],
        questions: list[str],
    ) -> list[dict[str, str | None]]:
        """逐项返回固定 red，只验证身份串联，不计算视觉准确率。

        Args:
            paths: 测试图片。
            questions: 与图片同序的测试问题。

        Results:
            list: 固定 answer/error 字段。
        """
        return [{"answer": "red", "error": None} for _ in questions]


def test_annotation_score_review_calibration_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模拟标签验证整个校准数据流，同时证明默认 Mock 不产生正确答案。

    Args:
        tmp_path: 不进入项目实验报告的临时测试目录。
        monkeypatch: 显式替换真实 BLIP，避免下载或伪称真实模型执行。

    Raises:
        AssertionError: 问题 ID、文件身份、审核导入或配对统计不一致。

    副作用:
        生成标记为 simulated 的测试输入与固定回答，不构成真实人工校准。
    """
    source = tmp_path / "red.png"
    Image.new("RGB", (256, 256), "red").save(source)
    benchmark = {
        "data_root": ".",
        "validation_references": [
            {
                "sample_id": "fixture-red",
                "image_path": source.name,
                "sha256": hash_file(source),
                "split": "validation",
                "caption": "A red painting.",
            }
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "source": "human_annotations",
                "test_evidence_scope": "simulated",
                "samples": [
                    {
                        "sample_id": "fixture-red",
                        "requirements": [
                            {"kind": "color", "subject": "painting", "expected": "red"}
                        ],
                    }
                ],
            }
        )
    )
    output = tmp_path / "calibration"
    prepared = prepare_calibration(benchmark, tmp_path, output, count=1, annotations=annotations)
    samples = Path(prepared["samples"])
    receipt = json.loads(samples.read_text())
    assert receipt["expected_samples"] == 2
    assert {r["sample_id"] for r in receipt["records"]} == {
        "real:fixture-red",
        "degraded:fixture-red",
    }
    mapping = json.loads((output / "review/private_mapping.json").read_text())
    assert {r["original_id"] for r in mapping["items"]} == {
        r["task_id"] for r in _review_records([samples])
    }
    assert all("Expected: red" in r["questions"][0]["text"] for r in mapping["items"])
    assets = tmp_path / "assets.json"
    assets.write_text('{"models":{}}')
    mock = run_scoring(
        tmp_path, samples, None, output / "mock.json", assets, metrics=("vqa",), vqa_backend="mock"
    )
    assert mock["metrics"]["vqa"]["scope"] == "simulated"
    assert mock["metrics"]["vqa"]["categories"]["color"]["indeterminate"] == 2
    monkeypatch.setattr("minigen.evaluation.scoring.BlipVQA", FixedAnswerBackend)
    score = run_scoring(
        tmp_path, samples, None, output / "fake-real.json", assets, metrics=("vqa",)
    )
    score["test_evidence_scope"] = "simulated"
    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps(
            {
                "package_id": mapping["package_id"],
                "reviewer": "SIMULATED_TEST_FIXTURE",
                "reviewed_at": "SIMULATED_TEST_FIXTURE",
                "test_evidence_scope": "simulated",
                "answers": {
                    row["item_id"]: {
                        "quality": 3,
                        "alignment": 3,
                        "questions": {row["questions"][0]["question_id"]: "correct"},
                    }
                    for row in mapping["items"]
                },
            }
        )
    )
    human = import_review(output / "review", answers, output / "human-fixture.json")
    automatic = [
        {
            **row,
            "task_id": f"calibration:{row['sample_id']}",
            "verdict": row["status"],
        }
        for row in score["metrics"]["vqa"]["questions"]
    ]
    compared = compare_calibration(human, automatic)
    compared["test_evidence_scope"] = "simulated"
    assert compared["paired_questions"] == 2 and compared["missing_automatic"] == 0
    assert compared["agreement_on_decidable"] == 1
    bad = json.loads(annotations.read_text())
    bad["samples"][0]["sample_id"] = "wrong-id"
    annotations.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly cover"):
        prepare_calibration(benchmark, tmp_path, tmp_path / "bad", 1, annotations)
