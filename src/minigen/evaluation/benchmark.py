"""ArtBench 评测题目、独立图片索引及不冒充人工批准的内容锁。"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from minigen._typing import JsonDict
from minigen.data.manifest import load_manifest


def content_hash(
    value: object,
) -> str:
    """计算规范 JSON 内容摘要，与缩进和字典插入顺序无关。

    Args:
        value: 可序列化的有限 JSON 内容。

    Raises:
        TypeError/ValueError: 内容无法写为有限 JSON。

    Results:
        str: SHA-256 内容身份。
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _write_json(
    path: Path,
    value: object,
) -> None:
    """写入新的小型 JSON 产物。

    Args:
        path: 目标文件。
        value: JSON 内容。

    Raises:
        OSError/TypeError/ValueError: 写入或序列化失败。

    副作用:
        写入目标文件。
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _requirement(
    kind: str,
    subject: str,
    expected: str,
    other: str | None = None,
) -> JsonDict:
    """构造 Prompt 请求的视觉约束，不声称图片已经满足约束。

    Args:
        kind: object、count、color、binding 或 relation。
        subject: 需要判断的对象或属性绑定目标。
        expected: 预期规范答案。
        other: 关系中的第二对象。

    Results:
        JsonDict: 交给 VQA 与人工校准的问题要求。
    """
    result: JsonDict = {"kind": kind, "subject": subject, "expected": expected}
    if other is not None:
        result["object"] = other
    return result


def build_prompts(
    catalog: JsonDict,
    training: list[JsonDict],
) -> list[JsonDict]:
    """按显式题目目录建立 160 主测试与 80 验证题，登记训练描述词证据。

    Args:
        catalog: 含两组主体、场景与固定种子的配置。
        training: 训练清单；只读取 Caption 作为概念词支持，不作视觉真值。

    Raises:
        ValueError: 主体配置重复或规模不符。

    Results:
        list[JsonDict]: 每题包含语义族、要求、成对关系及证据边界。
    """
    test = catalog["test_subjects"]
    validation = catalog["validation_subjects"]
    subjects = [row[0] for row in test + validation]
    if len(test) != 20 or len(validation) != 20 or len(set(subjects)) != 40:
        raise ValueError("Expected two disjoint 20-subject catalogs")
    support: dict[str, JsonDict] = {}
    for subject, plural, *_ in test + validation:
        pattern = re.compile(r"\b(?:" + re.escape(subject) + "|" + re.escape(plural) + r")\b", re.I)
        found = [row for row in training if pattern.search(row["caption"])]
        support[subject] = {
            "caption_matches": len(found),
            "example_ids": [row["sample_id"] for row in found[:5]],
            "evidence_type": "training_caption_lexical_only",
            "visual_occurrence_verified": False,
        }
    prompts: list[JsonDict] = []
    for split, rows, seeds in (
        ("id", test, catalog["main_seeds"]),
        ("validation", validation, catalog["validation_seeds"]),
    ):
        for index, (subject, plural, setting, color_subject, color) in enumerate(rows):
            variants = (
                (
                    f"A painting showing one {subject} {setting}.",
                    [_requirement("count", subject, "1")],
                ),
                (
                    f"A painting showing two {plural} {setting}.",
                    [_requirement("count", subject, "2")],
                ),
                (
                    f"A painting showing one {subject} {setting}. The {color_subject} is {color}.",
                    [
                        _requirement("color", color_subject, color),
                        _requirement("binding", color_subject, color),
                    ],
                ),
                (
                    (
                        f"A painting showing one {subject} on the left side of the composition, "
                        "with empty space on the right."
                    ),
                    [_requirement("relation", subject, "left", "image center")],
                ),
            )
            for variant, (caption, requirements) in enumerate(variants):
                prompts.append(
                    {
                        "prompt_id": f"{split}-{index:02d}-{variant}",
                        "caption": caption,
                        "split": split,
                        "seeds": seeds,
                        "family_id": f"{split}-{subject}",
                        "paired_prompt_id": None,
                        "ood_verified": False,
                        "requirements": [_requirement("object", subject, "yes"), *requirements],
                        "evidence": support[subject],
                        "semantic_review": "pending_human_review",
                    }
                )
    for index, (subject, _, _, color_subject, color) in enumerate(test):
        other, _, _, other_color_subject, other_color = test[(index + 7) % len(test)]
        for variant in range(2):
            relation = "left" if variant == 0 else "above"
            caption = (
                f"An artistic painting showing one {subject} to the {relation} of one {other}."
            )
            if relation == "above":
                caption = f"An artistic painting showing one {subject} above one {other}."
            requirements = [
                _requirement("object", subject, "yes"),
                _requirement("object", other, "yes"),
                _requirement("relation", subject, relation, other),
            ]
            if variant == 1:
                caption += (
                    f" The {color_subject} is {color}, "
                    f"and the {other_color_subject} is {other_color}."
                )
                requirements.extend(
                    [
                        _requirement("binding", color_subject, color),
                        _requirement("binding", other_color_subject, other_color),
                    ]
                )
            prompts.append(
                {
                    "prompt_id": f"composition-{index:02d}-{variant}",
                    "caption": caption,
                    "split": "composition_stress",
                    "seeds": catalog["main_seeds"],
                    "family_id": f"composition-{subject}-{other}",
                    "paired_prompt_id": None,
                    "ood_verified": False,
                    "requirements": requirements,
                    "evidence": {
                        "components": [support[subject], support[other]],
                        "unseen_visual_combination": "unverified",
                    },
                    "semantic_review": "pending_human_review",
                }
            )
    # 每个主体只改写单数/双数两题；约束直接复制，不靠不同表达重新猜真值。
    originals = [
        row for row in prompts if row["split"] == "id" and row["prompt_id"].endswith(("-0", "-1"))
    ]
    for original in originals:
        caption = original["caption"].removeprefix("A painting showing ").removesuffix(".")
        prompts.append(
            {
                **original,
                "prompt_id": "paraphrase-" + original["prompt_id"],
                "caption": f"Depict {caption} as a painted artwork.",
                "split": "paraphrase",
                "paired_prompt_id": original["prompt_id"],
            }
        )
    return prompts


def validate_benchmark(
    payload: JsonDict,
) -> None:
    """验证题量、唯一标识、配对约束、题族隔离和数据划分身份。

    Args:
        payload: 待验证的完整 Benchmark 内容。

    Raises:
        ValueError: 规模、配对、隔离、字段或种子约定不成立。
    """
    prompts = payload["prompts"]
    if Counter(row["split"] for row in prompts) != {
        "id": 80,
        "composition_stress": 40,
        "paraphrase": 40,
        "validation": 80,
    }:
        raise ValueError("Benchmark prompt counts differ from protocol")
    by_id = {row["prompt_id"]: row for row in prompts}
    if len(by_id) != len(prompts) or len({row["caption"] for row in prompts}) != len(prompts):
        raise ValueError("Duplicate prompt identity or caption")
    main_families = {row["family_id"] for row in prompts if row["split"] != "validation"}
    val_families = {row["family_id"] for row in prompts if row["split"] == "validation"}
    if main_families & val_families:
        raise ValueError("Validation family overlaps main test")
    for row in prompts:
        seeds = [20261112, 20261113] if row["split"] == "validation" else [20261110, 20261111]
        if row["seeds"] != seeds or not row["requirements"] or row["ood_verified"]:
            raise ValueError("Prompt seeds, requirements or OOD evidence invalid")
        if row["split"] == "paraphrase":
            original = by_id.get(row["paired_prompt_id"])
            if (
                original is None
                or original["split"] != "id"
                or original["requirements"] != row["requirements"]
                or original["family_id"] != row["family_id"]
            ):
                raise ValueError("Paraphrase must preserve ID requirements and family")
    for split, key in (
        ("test", "references"),
        ("validation", "validation_references"),
        ("validation", "calibration_references"),
    ):
        records = payload.get(key, [])
        if len({row["sample_id"] for row in records}) != len(records) or any(
            row["split"] != split for row in records
        ):
            raise ValueError("Reference IDs or split invalid")


def prepare_benchmark(
    root: Path,
    output: Path,
    manifest: Path | None = None,
    reference_count: int = 2000,
    validation_reference_count: int = 16,
    calibration_reference_count: int = 100,
) -> JsonDict:
    """从冻结清单建立可复现参考索引和待人工审题的内容锁，不计算正式指标。

    Args:
        root: 仓库及本地资产根。
        output: 新输出目录；存在时只验证同一版本，不覆写。
        manifest: 可选清单路径，默认正式 ArtBench v1。
        reference_count: Test 参考数量，正式为 2000。
        validation_reference_count: 本地烟测验证参考数量，默认 16。
        calibration_reference_count: 独立人工校准验证参考数量，默认 100。

    Raises:
        ValueError: 数据泄漏、尺寸、数量、文件哈希或已有身份不符。
        OSError/KeyError: 本地清单、目录或配置不可访问。

    Results:
        JsonDict: 已锁定内容与摘要，不自动提供人工批准。

    副作用:
        新建 Benchmark JSON 与独立锁；读取参考图作文件完整性检查。
    """
    root = root.resolve()
    source = root / (manifest or Path("data/processed/artbench_pd_256/v1/manifest.jsonl"))
    data_root = source.parent
    rows = load_manifest(source, data_root)
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        if (row["width"], row["height"], row["mode"]) != (256, 256, "RGB"):
            raise ValueError("ArtBench image specification changed")
        for key, seen in (("group_id", groups), ("sha256", hashes)):
            previous = seen.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError("Cross-split group or file leakage")
    catalog = json.loads((root / "configs/benchmark/prompt_catalog.json").read_text())
    prompts = build_prompts(catalog, [row for row in rows if row["split"] == "train"])
    references: dict[str, list[JsonDict]] = {}
    for split, key, count in (
        ("test", "references", reference_count),
        ("validation", "validation_references", validation_reference_count),
        ("validation", "calibration_references", calibration_reference_count),
    ):
        if count < 1:
            raise ValueError("Reference count must be positive")
        subset = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: content_hash([catalog["selection_seed"], row["sample_id"]]),
        )
        if len(subset) < count:
            raise ValueError("Insufficient frozen reference samples")
        selected = subset[:count]
        for row in selected:
            if (
                hashlib.sha256((data_root / row["image_path"]).read_bytes()).hexdigest()
                != row["sha256"]
            ):
                raise ValueError("Reference image hash mismatch")
        references[key] = [
            {
                key: row[key]
                for key in (
                    "sample_id",
                    "image_path",
                    "sha256",
                    "group_id",
                    "caption",
                    "split",
                    "artist",
                )
            }
            for row in selected
        ]
    bound_files = {}
    for relative in ("configs/models/evaluation.json", "configs/benchmark/sampling_matrix.json"):
        asset = root / relative
        bound_files[relative] = (
            hashlib.sha256(asset.read_bytes()).hexdigest() if asset.exists() else None
        )
    evaluator = root / "configs/models/evaluation.json"
    vqa = json.loads(evaluator.read_text())["models"]["vqa"] if evaluator.exists() else {}
    payload: JsonDict = {
        "schema": "minigen.benchmark.v1",
        "version": catalog["version"],
        "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "catalog_sha256": content_hash(catalog),
        "data_root": str(data_root.relative_to(root)),
        "selection_seed": catalog["selection_seed"],
        "prompts": prompts,
        **references,
        "protocol": {
            "kid_subsets": 100,
            "kid_subset_size": 1000,
            "kid_kernel": "(x dot y / feature_dimension + 1)^3",
            "kid_estimator": "unbiased MMD; negative estimates retained",
            "kid_features": "torch-fidelity Inception 2048; unnormalized",
            "cmmd_sigma": 10.0,
            "cmmd_scale": 1000,
            "cmmd_estimator": "biased MMD; unit-normalized CLIP image projection",
            "cmmd_kernel": "exp(-squared_distance/(2*sigma^2))",
            "clipscore": "L14-336-no-prefix-weight2.5",
            "clipscore_formula": "2.5*max(unit_image dot unit_text,0)",
            "vqa": {
                "repo_id": vqa.get("repo_id"),
                "revision": vqa.get("revision"),
                "verdicts": ["correct", "wrong", "indeterminate", "error"],
                "denominator": (
                    "all applicable questions including unknown/error; "
                    "generated failures fail overall"
                ),
                "confidence": "raw generation probability is not calibrated confidence",
                "calibration": "pending real inference and human annotation",
            },
            "human_rubrics": {
                "quality": {
                    "1": "unrecognizable/severe artifacts",
                    "3": "recognizable with clear defects",
                    "5": "coherent, clear, few visible defects",
                },
                "alignment": {
                    "1": "most requested facts fail",
                    "3": "some requested facts met",
                    "5": "all requested visual facts met",
                },
                "intermediate_scores": "2 and 4 interpolate adjacent anchors",
                "claim_verdicts": ["correct", "incorrect", "undecidable"],
                "incomplete": "missing human answers remain pending, not zero or inferred",
            },
            "bound_files": bound_files,
            "assets_locked": all(bound_files.values()),
            "distribution_seed": 20261110,
            "human_review": "pending",
            "ood": "composition stress only; visual absence unverified",
            "semantic_family_check": (
                "catalog subject families disjoint; "
                "related concepts and phrasing await human review"
            ),
            "reference_count": reference_count,
            "validation_reference_count": validation_reference_count,
            "calibration_reference_count": calibration_reference_count,
        },
    }
    validate_benchmark(payload)
    digest = content_hash(payload)
    output = root / output
    if output.exists():
        existing = load_benchmark(output / "benchmark.json")
        if content_hash(existing) != digest:
            raise ValueError("Existing benchmark differs; use a new version/output")
        return {
            "benchmark_sha256": digest,
            "benchmark": str(output / "benchmark.json"),
            "status": "verified",
            "human_approval": "pending",
        }
    output.mkdir(parents=True)
    _write_json(output / "benchmark.json", payload)
    _write_json(
        output / "lock.json",
        {
            "schema": "minigen.benchmark-lock.v1",
            "benchmark_sha256": digest,
            "human_approval": "pending",
        },
    )
    return {
        "benchmark_sha256": digest,
        "benchmark": str(output / "benchmark.json"),
        "status": "content_frozen",
        "human_approval": "pending",
    }


def load_benchmark(
    path: Path,
    require_approved: bool = False,
    approval: Path | None = None,
) -> JsonDict:
    """读取内容锁，正式执行时额外要求绑定该内容的真实审核回执。

    Args:
        path: benchmark.json 路径。
        require_approved: 正式运行时为 True，本地烟测可为 False。
        approval: 用户人工审核产生的独立 JSON 回执；程序不会自动创建批准。

    Raises:
        ValueError: 内容被改写、协议不符或正式执行缺少合法批准回执。
        OSError/KeyError: 文件不可读或必要字段缺失。

    Results:
        JsonDict: 通过内容校验的 Benchmark 定义。
    """
    payload = json.loads(path.read_text())
    lock = json.loads(path.with_name("lock.json").read_text())
    digest = content_hash(payload)
    if lock["benchmark_sha256"] != digest:
        raise ValueError("Benchmark content hash mismatch")
    validate_benchmark(payload)
    if require_approved:
        if approval is None:
            raise ValueError("Formal benchmark requires human approval receipt")
        receipt = json.loads(approval.read_text())
        if (
            receipt.get("benchmark_sha256") != digest
            or receipt.get("approved") is not True
            or not receipt.get("reviewer")
            or not receipt.get("reviewed_at")
        ):
            raise ValueError("Human approval does not match benchmark content")
        roots = [
            parent
            for parent in path.resolve().parents
            if (parent / "configs/benchmark/prompt_catalog.json").exists()
        ]
        if not roots:
            raise ValueError("Formal benchmark cannot locate its bound project assets")
        verify_protocol_assets(roots[0], payload)
    return payload


def verify_protocol_assets(
    root: Path,
    payload: JsonDict,
) -> None:
    """核对正式协议绑定的评测资产锁与采样矩阵，拒绝身份漂移。

    Args:
        root: 包含固定配置的项目根。
        payload: 已通过内容哈希核验的 Benchmark。

    Raises:
        ValueError: 缺少正式资产绑定，或当前文件哈希不同。
        OSError/KeyError: 配置无法读取或协议缺少必要字段。
    """
    protocol = payload["protocol"]
    bound = protocol.get("bound_files", {})
    required = {"configs/models/evaluation.json", "configs/benchmark/sampling_matrix.json"}
    if not protocol.get("assets_locked") or set(bound) != required:
        raise ValueError("Formal benchmark has no complete evaluator/sampling asset lock")
    for relative, expected in bound.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Benchmark bound asset hash mismatch: {relative}")
