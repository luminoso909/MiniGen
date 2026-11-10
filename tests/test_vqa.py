"""VQA 的要求转问题、判定分母和真实适配器调用协议；不声称运行预训练 BLIP。"""

from pathlib import Path
from typing import Any

import pytest
import torch
from jaxtyping import Int
from PIL import Image
from torch import Tensor

from minigen.evaluation.vqa import BlipVQA, MockVQA, evaluate_vqa, make_questions, normalize_answer


@pytest.mark.parametrize("subject", ["woman", "child", "sheep", "horses"])
def test_count_question_preserves_category_and_answer(
    subject: str,
) -> None:
    """数量问句兼容单复数类别，对象问句避免错误冠词，要求及答案不变。

    Args:
        subject: 普通单数、不规则单数、单复数同形或旧夹具复数类别。

    Raises:
        AssertionError: 问句仍需猜测复数，或修改了要求及预期答案。
    """
    requirement = {"kind": "count", "subject": subject, "expected": "2"}
    question = make_questions([requirement], "count-contract")[0]
    assert question["question"] == f"How many instances of {subject} are visible in this painting?"
    assert question["question_id"] == "count-contract:q00"
    assert {key: question[key] for key in requirement} == requirement
    presence = make_questions([{"kind": "object", "subject": "apple", "expected": "yes"}], "p")
    assert presence[0]["question"] == "Are any instances of apple visible in this painting?"
    assert presence[0]["expected"] == "yes"


def test_questions_unknowns_and_failed_denominators() -> None:
    """未知答复和生成失败保留分母，mock 不允许产出正式结果。

    Raises:
        AssertionError: 要求、解析、分母或模拟边界错误。
    """
    requirements = [
        {"kind": "object", "subject": "horse", "expected": "yes"},
        {"kind": "count", "subject": "horses", "expected": "2"},
        {"kind": "relation", "subject": "horse", "object": "tree", "expected": "left"},
    ]
    questions = make_questions(requirements, "p1")
    answers = {
        questions[0]["question"]: "yes",
        questions[1]["question"]: "two",
        questions[2]["question"]: "not sure",
    }
    records = [
        {
            "sample_id": "s1",
            "prompt_id": "p1",
            "status": "completed",
            "image_path": "fake.png",
            "requirements": requirements,
        },
        {"sample_id": "s2", "prompt_id": "p1", "status": "failed", "requirements": requirements},
    ]
    result = evaluate_vqa(records, Path("."), MockVQA(answers), "real_local")
    assert result["scope"] == "simulated"
    assert result["categories"]["object"]["applicable"] == 2
    assert result["categories"]["object"]["conservative_success_rate"] == 0.5
    assert result["categories"]["relation"]["indeterminate"] == 1
    assert result["categories"]["relation"]["error"] == 1
    assert result["all_requirements_correct"] == 0
    assert normalize_answer("grey", "color") == "gray"
    assert normalize_answer("red and blue", "color") is None
    with pytest.raises(ValueError, match="formal"):
        evaluate_vqa(records, Path("."), MockVQA(), "formal")


class FakeProcessor:
    """模拟 Transformers processor 的 I/O，不模拟真实模型准确率。"""

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        """接受真实适配器批输入并返回可迁移张量容器。

        Args:
            kwargs: images/text/return_tensors/padding 调用参数。

        Results:
            Any: 模拟 BatchEncoding。
        """
        from transformers import BatchEncoding

        return BatchEncoding({"input_ids": torch.ones(len(kwargs["text"]), 3, dtype=torch.long)})

    def batch_decode(
        self,
        output: Int[Tensor, "b n"],
        skip_special_tokens: bool,
    ) -> list[str]:
        """根据模拟批大小返回固定答案。

        Args:
            output: [b,1] 模型输出占位。
            skip_special_tokens: 实际适配器传入的解码参数。

        Results:
            list[str]: 每项为 two，不是视觉识别结果。
        """
        return ["two"] * len(output)


class FakeModel:
    """只检查 generate 被指定为确定性解码。"""

    def generate(
        self,
        **kwargs: Any,
    ) -> Int[Tensor, "b n"]:
        """验证真实模型适配器的推理选项。

        Args:
            kwargs: 张量与解码配置。

        Raises:
            AssertionError: 解码选项不符合固定协议。

        Results:
            torch.Tensor: [b,1] 的模拟 token。
        """
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        assert kwargs["max_new_tokens"] == 16
        return kwargs["input_ids"][:, :1]


def test_real_adapter_batching_and_failure_contract(
    tmp_path: Path,
) -> None:
    """通过显式假模型检查真实适配器代码路径、尾批和坏图错误。

    Args:
        tmp_path: 图片测试目录。

    Raises:
        AssertionError: 批处理或错误传播不符合契约。

    副作用:
        保存测试图；不加载、下载或运行真实 VQA 权重。
    """
    image = tmp_path / "image.png"
    Image.new("RGB", (256, 256)).save(image)
    backend = BlipVQA.__new__(BlipVQA)
    backend.model, backend.processor = FakeModel(), FakeProcessor()
    backend.device, backend.batch_size = "cpu", 2
    result = backend.answer([image] * 3, ["How many?"] * 3)
    assert len(result) == 3 and all(r["answer"] == "two" for r in result)
    failed = backend.answer([tmp_path / "missing.png"], ["How many?"])
    assert failed[0]["answer"] is None and "FileNotFoundError" in failed[0]["error"]
