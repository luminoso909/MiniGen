"""离线匿名人工审核、结果导入与自动评测校准；从不自动填写人工答案。"""

import base64
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter

from minigen._typing import JsonDict
from minigen.evaluation.benchmark import content_hash

ERROR_TYPES = (
    "missing_subject",
    "count",
    "color",
    "binding",
    "relation",
    "structure",
    "texture",
    "unrecognizable",
    "other",
)
VERDICTS = ("correct", "incorrect", "undecidable")


def _dump(
    path: Path,
    value: object,
) -> None:
    """保存有限 JSON 内容。

    Args:
        path: 输出路径。
        value: JSON 对象。

    Raises:
        OSError/TypeError/ValueError: 文件写入或序列化失败。

    副作用:
        写入 JSON 文件。
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _image_uri(
    path: Path,
) -> str:
    """以重新编码 PNG 生成内嵌图，去除原文件名和图片元数据。

    Args:
        path: 本地图片。

    Raises:
        OSError: 图片无法读取或解码。

    Results:
        str: 不包含源路径的 PNG data URI。
    """
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def export_review(
    records: list[JsonDict],
    output: Path,
    seed: int = 20261110,
    purpose: str = "blind_review",
) -> JsonDict:
    """创建匿名离线网页及独立私有映射；只有用户操作才产生评分。

    Args:
        records: 含 image_path、caption 的记录，可附 requirements、questions、task_id。
        output: 不存在的审核包目录；image_path 相对当前工作目录或为绝对路径。
        seed: 匿名展示顺序的固定种子。
        purpose: blind_review 或 calibration，记录审核用途。

    Raises:
        ValueError: 记录为空、重复标识或审核用途非法。
        FileExistsError: 目标目录存在，避免覆写已有人工工作。
        OSError/KeyError: 图片或字段不可读。

    Results:
        JsonDict: 包身份、待审数量和网页位置；不含人工得分。

    副作用:
        创建 HTML、私有映射与 pending 回执，不请求网络。
    """
    if not records or purpose not in {"blind_review", "calibration"}:
        raise ValueError("Expected review records and known purpose")
    if output.exists():
        raise FileExistsError(output)
    identities = [str(row.get("task_id", row.get("sample_id", i))) for i, row in enumerate(records)]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate review identity")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    public = []
    private = []
    for index in order:
        row = records[index]
        item_id = hashlib.sha256(f"{seed}:{identities[index]}".encode()).hexdigest()[:16]
        questions = row.get("questions")
        if questions is None:
            questions = [
                {
                    "question_id": f"q-{j:03d}",
                    "text": (
                        f"{requirement['kind']}: {requirement['subject']} "
                        f"→ {requirement['expected']}"
                    )
                    + (
                        f" relative to {requirement['object']}" if requirement.get("object") else ""
                    ),
                }
                for j, requirement in enumerate(row.get("requirements", []))
            ]
        image_uri = None
        image_status = "generation_failed"
        if row.get("image_path") and row.get("status") != "failed":
            try:
                image_uri = _image_uri(Path(row["image_path"]))
                image_status = "available"
            except OSError:
                image_status = "image_unreadable"
        public.append(
            {
                "item_id": item_id,
                "caption": row["caption"],
                "image": image_uri,
                "image_status": image_status,
                "questions": [
                    {"question_id": q["question_id"], "text": q["text"]} for q in questions
                ],
            }
        )
        private.append(
            {
                "item_id": item_id,
                "original_id": identities[index],
                "record": row,
                "questions": questions,
            }
        )
    package_id = content_hash({"purpose": purpose, "seed": seed, "public": public})
    output.mkdir(parents=True)
    _dump(output / "private_mapping.json", {"package_id": package_id, "items": private})
    _dump(
        output / "review.json",
        {
            "schema": "minigen.human-review.v1",
            "package_id": package_id,
            "purpose": purpose,
            "count": len(public),
            "status": "pending_human_review",
            "seed": seed,
        },
    )
    # JSON 数据必须转义 <，避免 Caption 中的结束标签突破脚本元素。
    embedded = json.dumps(
        {"package_id": package_id, "items": public, "error_types": ERROR_TYPES}, ensure_ascii=False
    ).replace("<", "\\u003c")
    page = (
        Path(__file__).with_name("review_template.html").read_text().replace("__DATA__", embedded)
    )
    (output / "review.html").write_text(page)
    return {
        "package_id": package_id,
        "count": len(public),
        "html": str(output / "review.html"),
        "status": "pending_human_review",
    }


def import_review(
    package: Path,
    answers: Path,
    output: Path,
) -> JsonDict:
    """验证用户提供的审核 JSON 并恢复私有 ID；未填写项保持缺失。

    Args:
        package: export_review 输出目录。
        answers: 用户导出的回答文件。
        output: 新的结构化回执路径。

    Raises:
        ValueError: 包身份、审核人、分数、错误标签或问题 ID 不合法。
        FileExistsError: 输出已存在。
        OSError/KeyError: 输入不可读或缺少必要字段。

    Results:
        JsonDict: 原始人工回答、覆盖数量与独立来源，不自动填充空答案。

    副作用:
        写入结构化人工审核回执。
    """
    if output.exists():
        raise FileExistsError(output)
    mapping = json.loads((package / "private_mapping.json").read_text())
    response = json.loads(answers.read_text())
    if (
        response.get("package_id") != mapping["package_id"]
        or not str(response.get("reviewer", "")).strip()
        or not response.get("reviewed_at")
    ):
        raise ValueError("Missing reviewer/time or wrong review package")
    by_id = {row["item_id"]: row for row in mapping["items"]}
    imported = []
    for item_id, values in response["answers"].items():
        if item_id not in by_id:
            raise ValueError("Unknown anonymous item")
        for key in ("quality", "alignment"):
            score = values.get(key)
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError("Human scores must be integers in [1,5] or null")
        if set(values.get("errors", [])) - set(ERROR_TYPES):
            raise ValueError("Unknown human error category")
        valid_questions = {question["question_id"] for question in by_id[item_id]["questions"]}
        for question_id, verdict in values.get("questions", {}).items():
            if question_id not in valid_questions or verdict not in (*VERDICTS, ""):
                raise ValueError("Unknown question or human verdict")
        imported.append(
            {"item_id": item_id, "original_id": by_id[item_id]["original_id"], **values}
        )
    result = {
        "schema": "minigen.human-review-result.v1",
        "package_id": mapping["package_id"],
        "reviewer": response["reviewer"],
        "reviewed_at": response["reviewed_at"],
        "source": "user_import",
        "expected_items": len(by_id),
        "answered_items": len(imported),
        "records": imported,
        "status": "complete"
        if len(imported) == len(by_id)
        and all(
            row.get("quality") is not None
            and row.get("alignment") is not None
            and all(
                row.get("questions", {}).get(question["question_id"]) in VERDICTS
                for question in by_id[row["item_id"]]["questions"]
            )
            for row in imported
        )
        else "partial",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _dump(output, result)
    return result


def compare_calibration(
    human: JsonDict,
    automatic: list[JsonDict],
) -> JsonDict:
    """按原始样本与问题 ID 比较自动判定和真实导入的人评，保留不可判与漏项。

    Args:
        human: import_review 返回的人工记录。
        automatic: 含 task_id、question_id、kind、verdict 的自动逐问题记录。

    Raises:
        ValueError: 输入不是人工导入结果或自动结果 ID 重复。

    Results:
        JsonDict: 总体及逐类一致性、混淆计数和覆盖率；没有配对则均值为 null。
    """
    if human.get("source") != "user_import":
        raise ValueError("Calibration requires separately imported human answers")
    auto = {(row["task_id"], row["question_id"]): row for row in automatic}
    if len(auto) != len(automatic):
        raise ValueError("Duplicate automatic question identity")
    counts: dict[str, Counter[str]] = {}
    compared = []
    missing = 0
    for row in human["records"]:
        for question_id, verdict in row.get("questions", {}).items():
            key = (row["original_id"], question_id)
            if not verdict:
                continue
            if key not in auto:
                missing += 1
                continue
            candidate = auto[key]
            kind = candidate["kind"]
            count = counts.setdefault(kind, Counter())
            count[f"human_{verdict}__auto_{candidate['verdict']}"] += 1
            compared.append(
                {
                    "task_id": key[0],
                    "question_id": question_id,
                    "kind": kind,
                    "human": verdict,
                    "automatic": candidate["verdict"],
                }
            )
    judged = [
        row
        for row in compared
        if row["human"] in {"correct", "incorrect"} and row["automatic"] in {"correct", "incorrect"}
    ]
    return {
        "schema": "minigen.evaluator-calibration.v1",
        "paired_questions": len(compared),
        "decidable_pairs": len(judged),
        "missing_automatic": missing,
        "agreement_on_decidable": sum(row["human"] == row["automatic"] for row in judged)
        / len(judged)
        if judged
        else None,
        "by_kind_confusion": {key: dict(value) for key, value in counts.items()},
        "records": compared,
        "status": "exploratory_requires_adequate_human_coverage",
    }


def prepare_calibration(
    benchmark: JsonDict,
    root: Path,
    output: Path,
    count: int = 8,
    annotations: Path | None = None,
) -> JsonDict:
    """为固定验证图生成独立退化副本及匿名审核页，不把 Caption 当视觉真值。

    Args:
        benchmark: 已加载的 Benchmark；优先使用固定 calibration_references。
        root: 本地资产根。
        output: 新输出目录，保存派生图片与审核包。
        count: 真实图片数量，默认 8；可扩展到正式校准规模。
        annotations: 可选人工事实要求 JSON；必须恰好覆盖选中的验证样本。

    Raises:
        ValueError: 样本数、人工标注、图像身份或要求答案无效。
        FileExistsError: 输出目录存在。
        OSError: 图片读取或写入失败。

    Results:
        JsonDict: 配对图谱及待审回执；有标注时另含可直接 VQA 评分的 samples.json。

    副作用:
        复制原图并写退化图片、谱系、标注身份和匿名 HTML；不填写人工判断。
    """
    import shutil

    from minigen.autoencoders.sdxl import hash_file
    from minigen.evaluation.vqa import make_questions, normalize_answer

    references = benchmark.get("calibration_references", benchmark["validation_references"])
    if count < 1 or count > len(references):
        raise ValueError("Insufficient validation references for calibration")
    if output.exists():
        raise FileExistsError(output)
    selected = references[:count]
    requirements = {}
    if annotations is not None:
        annotations = root / annotations
        annotated = json.loads(annotations.read_text())
        rows = annotated.get("samples", [])
        identifiers = [row["sample_id"] for row in rows]
        if (
            annotated.get("source") != "human_annotations"
            or len(identifiers) != len(set(identifiers))
            or set(identifiers) != {row["sample_id"] for row in selected}
        ):
            raise ValueError("Human annotations must exactly cover selected validation sample IDs")
        for row in rows:
            questions = make_questions(row["requirements"], row["sample_id"])
            if not questions or any(
                normalize_answer(q["expected"], q["kind"]) is None for q in questions
            ):
                raise ValueError("Human requirements need nonempty, parseable expected answers")
            requirements[row["sample_id"]] = row["requirements"]
    output.mkdir(parents=True)
    records, sample_records = [], []
    for row in selected:
        source = root / benchmark["data_root"] / row["image_path"]
        if annotations is not None and (
            row.get("split") != "validation" or hash_file(source) != row["sha256"]
        ):
            raise ValueError("Calibration source must match its frozen validation identity")
        file_id = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:20]
        original = output / f"{file_id}_real.png"
        shutil.copyfile(source, original)
        with Image.open(source) as im:
            image = im.convert("RGB")
            degraded = (
                image.filter(ImageFilter.GaussianBlur(radius=3)).resize((64, 64)).resize((256, 256))
            )
            target = output / f"{file_id}_degraded.png"
            degraded.save(target)
        for variant, path in (("real", original), ("degraded", target)):
            sample_id = f"{variant}:{row['sample_id']}"
            requested = requirements.get(row["sample_id"], [])
            questions = [
                {**q, "text": q["question"] + " Expected: " + q["expected"]}
                for q in make_questions(requested, row["sample_id"])
            ]
            sample_records.append(
                {
                    "task_id": sample_id,
                    "sample_id": sample_id,
                    "prompt_id": row["sample_id"],
                    "seed": 0,
                    "caption": row["caption"],
                    "requirements": requested,
                    "status": "completed",
                    "image_path": path.name,
                    "sha256": hash_file(path),
                    "error": None,
                }
            )
            records.append(
                {
                    **sample_records[-1],
                    "task_id": f"calibration:{sample_id}",
                    "image_path": str(path.resolve()),
                    "caption": row["caption"],
                    "lineage": {
                        "source_id": row["sample_id"],
                        "source_sha256": row["sha256"],
                        "kind": variant,
                        "transform": "gaussian_blur_radius3_resize64_then256"
                        if variant == "degraded"
                        else None,
                    },
                    "questions": questions
                    or [
                        {
                            "question_id": "caption-claim",
                            "text": (
                                "The visible subject and scene broadly agree "
                                "with the displayed caption."
                            ),
                        }
                    ],
                    "caption_is_visual_truth": False,
                }
            )
    _dump(output / "lineage.json", records)
    review = export_review(records, output / "review", purpose="calibration")
    samples_path = None
    if annotations is not None:
        samples_path = output / "samples.json"
        _dump(
            samples_path,
            {
                "schema": "minigen.bench-generation.v1",
                "scope": "real_local",
                "output": str(output.resolve()),
                "identity": {
                    "checkpoint_sha256": "calibration",
                    "benchmark_content_sha256": content_hash(benchmark),
                    "annotations_sha256": hash_file(annotations),
                    "source": "human_annotations",
                    "purpose": "evaluator_calibration",
                },
                "expected_samples": len(sample_records),
                "completed_samples": len(sample_records),
                "failed_samples": 0,
                "pending_samples": 0,
                "records": sample_records,
                "complete": True,
                "status": "completed",
            },
        )
    return {
        "real_images": count,
        "degraded_images": count,
        "status": "pending_human_review",
        "review": review,
        "samples": str(samples_path) if samples_path else None,
        "annotations_source": "human_annotations" if annotations is not None else None,
        "purpose": "evaluator_calibration" if annotations is not None else "review_tool_smoke_only",
    }
