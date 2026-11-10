"""结构化视觉问答及诚实的判定分母；真实 BLIP 和模拟后端不混用。"""

import re
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image


class VQABackend(Protocol):
    """为评分器提供逐项问答结果，不把模型输出称为人工真值。"""

    simulated: bool

    def answer(
        self,
        paths: list[Path],
        questions: list[str],
    ) -> list[dict[str, str | None]]:
        """按原顺序回答每个图像问题。

        Args:
            paths: 每个问题对应的图像路径，允许重复。
            questions: 与路径同长的问题。

        Results:
            list: 每项含 answer 和 error；错误不可静默丢弃。
        """
        ...


def make_questions(
    requirements: list[dict[str, str]],
    prompt_id: str,
) -> list[dict[str, str]]:
    """把提示词要求变成可复查问题；预期答案来自任务要求而非图像标注。

    Args:
        requirements: kind/subject/expected，可选 object 的结构化要求。
        prompt_id: 用于稳定问题身份的提示词编号。

    Raises:
        ValueError/KeyError: 要求类型、主体、答案或关系对象不合法。

    Results:
        list: question_id、question、kind、expected 等逐问题信息。
    """
    result = []
    for i, requirement in enumerate(requirements):
        kind, subject, expected = (requirement[k] for k in ("kind", "subject", "expected"))
        if not subject.strip() or not expected.strip():
            raise ValueError("Question subject and expected answer must be nonempty")
        if kind == "object":
            # 类别前不直接拼接 a/an，避免 apple 等类别产生错误冠词。
            question = f"Are any instances of {subject} visible in this painting?"
        elif kind == "count":
            # subject 是类别名，可能为 woman/sheep 等；不猜测英语复数或改写任务身份。
            question = f"How many instances of {subject} are visible in this painting?"
        elif kind in {"color", "binding"}:
            question = f"What color is the {subject} in this painting?"
        elif kind == "relation":
            other = requirement["object"]
            question = (
                f"Where is the {subject} relative to the {other}: left, right, above, or below?"
            )
        else:
            raise ValueError(f"Unsupported requirement {kind}")
        result.append({**requirement, "question_id": f"{prompt_id}:q{i:02d}", "question": question})
    return result


def normalize_answer(
    answer: str,
    kind: str,
) -> str | None:
    """只接受明确的有限答案词汇，模糊或复合回答保留不可判断。

    Args:
        answer: 模型原始文本。
        kind: 对象、数量、颜色、属性绑定或空间关系类型。

    Results:
        str | None: 规范答案；None 表示不能可靠解析，不能当作正确。
    """
    value = re.sub(r"[^a-z0-9 ]", "", answer.lower()).strip()
    numbers = dict(
        zip(
            "zero one two three four five six seven eight nine ten".split(),
            map(str, range(11)),
            strict=True,
        )
    )
    aliases = {
        "grey": "gray",
        "violet": "purple",
        "to the left": "left",
        "to the right": "right",
        "on the left": "left",
        "on the right": "right",
        "over": "above",
        "under": "below",
    }
    value = aliases.get(value, value)
    if kind == "object":
        return value if value in {"yes", "no"} else None
    if kind == "count":
        value = numbers.get(value, value)
        return str(int(value)) if value.isdigit() and int(value) <= 100 else None
    if kind in {"color", "binding"}:
        return (
            value
            if value
            in {
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
                "brown",
                "black",
                "white",
                "gray",
                "gold",
                "silver",
                "beige",
            }
            else None
        )
    if kind == "relation":
        return value if value in {"left", "right", "above", "below"} else None
    return None


class MockVQA:
    """只用于显式模拟验收；未设置的答案返回 unknown，不伪造满分。

    Args:
        answers: 按问题文本索引的固定答案字典。
    """

    simulated = True

    def __init__(
        self,
        answers: dict[str, str] | None = None,
    ) -> None:
        """保留固定模拟回答。

        Args:
            answers: 显式提供的测试答案。
        """
        self.answers = answers or {}

    def answer(
        self,
        paths: list[Path],
        questions: list[str],
    ) -> list[dict[str, str | None]]:
        """不读取图像，返回明确属于模拟后端的测试值。

        Args:
            paths: 协议占位图像路径。
            questions: 与路径同长的问题。

        Raises:
            ValueError: 路径和问题数量不同。

        Results:
            list: 各问题固定 answer 与空 error。
        """
        if len(paths) != len(questions):
            raise ValueError("VQA paths/questions count mismatch")
        return [{"answer": self.answers.get(q, "unknown"), "error": None} for q in questions]


class BlipVQA:
    """离线 BLIP VQA 适配器；绘画域评分仍须独立人工校准。

    Args:
        root: 仓库根目录。
        asset_lock: 包含 models.vqa 的资产锁。
        device: 推理设备。
        batch_size: 每批问题数。
    """

    simulated = False

    def __init__(
        self,
        root: Path,
        asset_lock: Path,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        """校验资产并直接构造真实 BLIP 类，不使用旧 VQA pipeline。

        Args:
            root: 项目根目录。
            asset_lock: 锁定模型及 processor 文件。
            device: CPU/MPS/CUDA。
            batch_size: 正整数批量。

        Raises:
            ValueError/OSError/RuntimeError: 配置、资产或加载失败。
            ImportError: Transformers 依赖缺失。

        副作用:
            从本地载入冻结模型和 processor，分配实际设备内存。
        """
        from transformers import BlipForQuestionAnswering, BlipProcessor

        from minigen.evaluation.assets import verify_asset

        if batch_size < 1:
            raise ValueError("VQA batch_size must be positive")
        directory = verify_asset(root, root / asset_lock, "vqa")
        self.model = (
            BlipForQuestionAnswering.from_pretrained(
                directory, local_files_only=True, use_safetensors=True
            )
            .eval()
            .requires_grad_(False)
            .float()
            .to(device)
        )
        self.processor = BlipProcessor.from_pretrained(directory, local_files_only=True)
        self.device, self.batch_size = device, batch_size

    @torch.inference_mode()
    def answer(
        self,
        paths: list[Path],
        questions: list[str],
    ) -> list[dict[str, str | None]]:
        """确定性解码所有问题；批失败显式保留到每个受影响问题。

        Args:
            paths: 问题对应的图片路径。
            questions: 同序问题列表。

        Raises:
            ValueError: 输入数量不匹配。

        Results:
            list: answer/error 列表；捕获的读图及模型异常转换为 error，不冒充未知答案。
        """
        if len(paths) != len(questions):
            raise ValueError("VQA paths/questions count mismatch")
        results = []
        for start in range(0, len(paths), self.batch_size):
            selected = paths[start : start + self.batch_size]
            try:
                images = []
                for path in selected:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(
                    images=images,
                    text=questions[start : start + len(selected)],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                output = self.model.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=16
                )
                answers = self.processor.batch_decode(output, skip_special_tokens=True)
                if len(answers) != len(selected):
                    raise ValueError("VQA returned a different answer count")
                results.extend({"answer": answer, "error": None} for answer in answers)
            except (OSError, ValueError, RuntimeError) as error:
                results.extend(
                    {"answer": None, "error": f"{type(error).__name__}: {error}"} for _ in selected
                )
        return results


def evaluate_vqa(
    records: list[dict[str, Any]],
    output_root: Path,
    backend: VQABackend,
    scope: str,
) -> dict[str, Any]:
    """为全部预期样本保留逐问题判定及失败分母。

    Args:
        records: 含要求、状态和相对图像路径的完整生成清单。
        output_root: 图片相对路径根。
        backend: 真实或显式模拟后端。
        scope: real_local、simulated 或 formal。

    Raises:
        ValueError: 正式评分请求模拟后端，或后端答案数量不符。

    Results:
        dict: 逐问题原始结果、每类适用分母及保守成功率、可判断覆盖率。
    """
    if scope == "formal" and backend.simulated:
        raise ValueError("Mock VQA cannot produce formal evaluation")
    entries, paths, questions, indices = [], [], [], []
    for record in records:
        for item in make_questions(record.get("requirements", []), record["prompt_id"]):
            row = {
                **item,
                "sample_id": record["sample_id"],
                "raw_answer": None,
                "normalized_answer": None,
                "status": "error",
                "error": record.get("error") or "generation_not_completed",
            }
            entries.append(row)
            if record["status"] == "completed":
                indices.append(len(entries) - 1)
                paths.append(output_root / record["image_path"])
                questions.append(item["question"])
    outputs = backend.answer(paths, questions) if paths else []
    if len(outputs) != len(indices):
        raise ValueError("VQA backend changed result count")
    for index, output in zip(indices, outputs, strict=True):
        row = entries[index]
        row["raw_answer"], row["error"] = output["answer"], output["error"]
        if output["error"] is not None:
            continue
        normalized = normalize_answer(output["answer"] or "", row["kind"])
        row["normalized_answer"] = normalized
        expected = normalize_answer(row["expected"], row["kind"])
        row["status"] = (
            "indeterminate"
            if normalized is None or expected is None
            else ("correct" if normalized == expected else "wrong")
        )
    categories = {}
    for kind in sorted({row["kind"] for row in entries}):
        selected = [row for row in entries if row["kind"] == kind]
        counts = {
            name: sum(row["status"] == name for row in selected)
            for name in ("correct", "wrong", "indeterminate", "error")
        }
        categories[kind] = {
            **counts,
            "applicable": len(selected),
            "conservative_success_rate": counts["correct"] / len(selected),
            "judgment_coverage": (counts["correct"] + counts["wrong"]) / len(selected),
        }
    sample_success = sum(
        bool(selected := [q for q in entries if q["sample_id"] == r["sample_id"]])
        and all(q["status"] == "correct" for q in selected)
        for r in records
    )
    return {
        "scope": "simulated" if backend.simulated else scope,
        "calibrated": False,
        "overall_applicable_samples": sum(bool(r.get("requirements")) for r in records),
        "expected_samples": len(records),
        "generation_failures": sum(r["status"] != "completed" for r in records),
        "all_requirements_correct": sample_success,
        "categories": categories,
        "questions": entries,
        "interpretation": "自动问题判断；未经绘画域人工校准，不作为正式视觉真值",
    }
