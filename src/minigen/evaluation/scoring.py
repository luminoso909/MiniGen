"""生成清单驱动的离线指标执行，保留缺图、失败、模拟来源及正式分母。"""

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torchvision.transforms.functional import to_tensor

from minigen.autoencoders.sdxl import hash_file
from minigen.evaluation.features import FeatureSuite
from minigen.evaluation.metrics import clip_score, cmmd, kid
from minigen.evaluation.reconstruction_metrics import LPIPSAlex
from minigen.evaluation.vqa import BlipVQA, MockVQA, evaluate_vqa


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """原子保存有限 JSON，避免中断暴露半份报告。

    Args:
        path: 目标文件。
        payload: 可序列化数据。

    Raises:
        OSError/ValueError/TypeError: 写盘或序列化失败。

    副作用:
        创建父目录并原子替换 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _valid_images(
    records: list[dict[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
    """验证清单中已完成图片，读图失败保留原条目并排除特征计算。

    Args:
        records: 完整预期样本，含 status/image_path/sha256。
        image_root: 允许的相对路径根。

    Raises:
        TypeError: 清单路径字段类型非法。

    Results:
        tuple: 有效条目、有序图片路径、包含全部失败的审核清单。
    """
    valid, paths, audited = [], [], []
    for item in records:
        row = dict(item)
        try:
            if row["status"] != "completed":
                audited.append(row)
                continue
            path = (image_root / row["image_path"]).resolve()
            if not path.is_relative_to(image_root.resolve()) or hash_file(path) != row["sha256"]:
                raise ValueError("Image path or SHA-256 mismatch")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (256, 256):
                    raise ValueError("Expected RGB256 image")
                image.load()
            valid.append(row)
            paths.append(path)
        except (KeyError, OSError, ValueError) as error:
            row["status"], row["error"] = "failed", f"{type(error).__name__}: {error}"
        audited.append(row)
    return valid, paths, audited


def _reference_records(
    root: Path,
    reference: Path,
    scope: str,
) -> tuple[list[dict[str, Any]], Path]:
    """读取 Benchmark 的参考清单，不将真实测试集用于本地 smoke。

    Args:
        root: 仓库根目录。
        reference: Benchmark 或显式参考清单 JSON。
        scope: formal 使用 test references，其余使用 validation_references。

    Raises:
        ValueError/KeyError/OSError: 清单格式、用途或文件不合法。

    Results:
        tuple: 参考记录及图片根目录。
    """
    payload = json.loads((root / reference).read_text())
    key = "references" if scope == "formal" else "validation_references"
    records = payload[key]
    expected_split = "test" if scope == "formal" else "validation"
    if not records or any(r.get("split") != expected_split for r in records):
        raise ValueError("Reference split does not match evaluation scope")
    return [{**row, "status": "completed"} for row in records], root / payload["data_root"]


def _lpips_pairs(
    root: Path,
    records: list[dict[str, Any]],
    image_root: Path,
    device: str,
) -> dict[str, Any]:
    """只比较相同 Prompt 不同 seed，另报缺失配对而不奖励失败图多样性。

    Args:
        root: 仓库根目录。
        records: 全部有效图片记录。
        image_root: 图片根目录。
        device: 实际推理设备。

    Raises:
        ValueError/OSError/RuntimeError: 权重、配对图片或计算失败。

    Results:
        dict: 均值、逐对距离及覆盖情况；未声明条件遵循已经成立。
    """
    model = LPIPSAlex.load(root, root / "reports/v2/components/p2_assets.json").to(device)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["prompt_id"], []).append(row)
    pairs = []
    for prompt_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["seed"])
        if len(rows) < 2:
            continue
        tensors = []
        for row in rows[:2]:
            with Image.open(image_root / row["image_path"]) as image:
                tensors.append(to_tensor(image).unsqueeze(0).to(device))
        value = float(model(tensors[0], tensors[1]).cpu()[0])
        pairs.append(
            {"prompt_id": prompt_id, "seeds": [r["seed"] for r in rows[:2]], "distance": value}
        )
    if not pairs:
        raise ValueError("LPIPS diversity requires at least one valid same-prompt two-seed pair")
    mean = sum(p["distance"] for p in pairs) / len(pairs)
    return {
        "value": mean,
        "mean": mean,
        "pairs": pairs,
        "paired_prompts": len(pairs),
        "valid_prompt_count": len(grouped),
        "conditioning_verified": False,
        "interpretation": "双种子差异，不作为独立质量或遵循得分",
    }


def run_scoring(
    root: Path,
    generation: Path,
    reference: Path | None,
    output: Path,
    asset_lock: Path = Path("configs/models/evaluation.json"),
    metrics: tuple[str, ...] = ("kid", "cmmd", "clipscore"),
    device: str = "cpu",
    batch_size: int = 4,
    scope: str = "real_local",
    vqa_backend: str = "blip",
    kid_subset_size: int | None = None,
    kid_subsets: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """运行请求的指标，单项失败保存错误，不伪造整体成功或模拟成绩。

    Args:
        root: 仓库根目录。
        generation: samples.json 生成清单。
        reference: 含正式/验证参考索引的 Benchmark；纯语义指标可为 None。
        output: 指标报告 JSON 文件。
        asset_lock: 独立评测模型锁。
        metrics: kid/cmmd/clipscore/vqa/lpips 子集。
        device: 实际推理设备。
        batch_size: 特征与问答批量。
        scope: real_local、simulated 或 formal。
        vqa_backend: blip 或显式 mock。
        kid_subset_size: None 使用正式 1000、smoke 4。
        kid_subsets: None 使用正式 100、smoke 10。
        resume: 仅相同身份可重用完成结果或重试失败指标。

    Raises:
        ValueError/FileExistsError: 请求、身份、用途或恢复模式非法。
        OSError/KeyError: 输入清单无法读取或结构错误。

    Results:
        dict: 指标值、原始判断、完整性、覆盖率及错误；模拟 VQA 单独标记。

    副作用:
        加载冻结模型，写特征缓存及原子评分回执；不更新模型参数。
    """
    root = root.resolve()
    generation, output, asset_lock = root / generation, root / output, root / asset_lock
    if scope not in {"real_local", "simulated", "formal"} or vqa_backend not in {"blip", "mock"}:
        raise ValueError("Invalid scoring scope or VQA backend")
    if (
        not metrics
        or len(set(metrics)) != len(metrics)
        or set(metrics) - {"kid", "cmmd", "clipscore", "vqa", "lpips"}
    ):
        raise ValueError("Unsupported or repeated metrics")
    if scope == "formal" and vqa_backend == "mock" and "vqa" in metrics:
        raise ValueError("Mock VQA cannot be used for formal results")
    payload = json.loads(generation.read_text())
    if payload["scope"] != scope:
        raise ValueError("Generation and evaluation scopes differ")
    if payload.get("expected_samples") != len(payload["records"]):
        raise ValueError("Generation omitted expected records")
    if not payload["records"] or len({r["sample_id"] for r in payload["records"]}) != len(
        payload["records"]
    ):
        raise ValueError("Expected nonempty unique sample IDs")
    if any(r["status"] not in {"completed", "failed", "pending"} for r in payload["records"]):
        raise ValueError("Unknown generation status")
    if scope == "formal":
        approved = payload["identity"]
        if not approved.get("approval_verified") or len(approved.get("approval_sha256", "")) != 64:
            raise ValueError("Formal scoring requires an approved generation receipt")
        if reference is None:
            raise ValueError("Formal scoring requires the locked benchmark")
        if set(metrics) & {"kid", "cmmd"} and (
            approved.get("split") != "distribution" or payload["expected_samples"] != 2000
        ):
            raise ValueError("Formal distribution metrics require all 2000 distribution tasks")
    if scope != "simulated" and reference is not None:
        from minigen.evaluation.benchmark import content_hash, load_benchmark

        definition = load_benchmark(root / reference)
        if payload["identity"].get("benchmark_sha256") != hash_file(root / reference):
            raise ValueError("Generation and reference benchmark identities differ")
        if scope == "formal":
            from minigen.evaluation.generation import make_tasks

            if payload["identity"].get("benchmark_content_sha256") != content_hash(definition):
                raise ValueError("Approved benchmark content identity differs")
            expected_tasks = make_tasks(definition, payload["identity"]["split"], None)
            if len(expected_tasks) != len(payload["records"]) or any(
                any(row.get(key) != value for key, value in task.items())
                for row, task in zip(payload["records"], expected_tasks, strict=True)
            ):
                raise ValueError("Formal generation records differ from the frozen task list")
            from minigen.evaluation.benchmark import verify_protocol_assets

            verify_protocol_assets(root, definition)
            if (
                hash_file(asset_lock)
                != definition["protocol"]["bound_files"]["configs/models/evaluation.json"]
            ):
                raise ValueError("Evaluator override differs from the approved protocol")
    subset_size = (
        kid_subset_size if kid_subset_size is not None else (1000 if scope == "formal" else 4)
    )
    subsets = kid_subsets if kid_subsets is not None else (100 if scope == "formal" else 10)
    if scope == "formal" and "kid" in metrics and (subset_size != 1000 or subsets != 100):
        raise ValueError("Formal KID requires the frozen 100 subsets of 1000 samples")
    identity = {
        "generation_sha256": hash_file(generation),
        "asset_lock_sha256": hash_file(asset_lock),
        "reference_sha256": hash_file(root / reference) if reference else None,
        "metrics": list(metrics),
        "scope": scope,
        "device": device,
        "batch_size": batch_size,
        "vqa_backend": vqa_backend,
        "kid_subset_size": subset_size,
        "kid_subsets": subsets,
        "implementation": "minigen-bench-score-v1",
        "source_sha256": {
            path.name: hash_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    previous = None
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        previous = json.loads(output.read_text())
        if previous["identity"] != identity:
            raise ValueError("Scoring identity changed")
    image_root = generation.parent
    valid, paths, audited = _valid_images(payload["records"], image_root)
    reference_paths: list[Path] = []
    reference_audit: list[dict[str, Any]] = []
    if set(metrics) & {"kid", "cmmd"}:
        if reference is None:
            raise ValueError("Distribution metrics require a reference manifest")
        references, directory = _reference_records(root, reference, scope)
        _, reference_paths, reference_audit = _valid_images(references, directory)
    if (
        previous
        and previous["complete"]
        and len(valid) == len(audited)
        and all(r["status"] == "completed" for r in reference_audit)
    ):
        return previous
    train_identity = payload.get("identity", {})
    result: dict[str, Any] = {
        "schema": "minigen.bench-scores.v1",
        "scope": scope,
        "identity": identity,
        "generation_path": str(generation),
        "generation_sha256": identity["generation_sha256"],
        "architecture": train_identity.get("architecture"),
        "sampling": {
            key: train_identity.get(key, payload.get(key))
            for key in (
                "method",
                "solver_nfe",
                "model_evaluations",
                "guidance",
                "device",
                "precision",
                "attention_backend",
            )
        },
        "run_id": train_identity.get("run_id"),
        "checkpoint_step": train_identity.get("checkpoint_step"),
        "checkpoint_sha256": train_identity.get("checkpoint_sha256"),
        "benchmark_sha256": train_identity.get("benchmark_sha256"),
        "metrics": {},
        "records": audited,
        "reference_records": reference_audit,
        "coverage": {
            "attempted": len(audited),
            "expected": len(audited),
            "valid": len(valid),
            "failed": sum(r["status"] == "failed" for r in audited),
            "pending": sum(r["status"] == "pending" for r in audited),
            "reference_valid": len(reference_paths),
            "reference_expected": len(reference_audit),
        },
        "exploratory": scope != "formal"
        or len(paths) < 1000
        or (bool(reference_audit) and len(reference_paths) < 1000),
        "complete": False,
    }
    calls = {
        batch["model_evaluations"]
        for batch in payload.get("batches", [])
        if "model_evaluations" in batch
    }
    if len(calls) == 1:
        result["sampling"]["model_evaluations"] = calls.pop()
    suite = FeatureSuite(root, asset_lock, device, batch_size)
    for metric in metrics:
        try:
            if metric == "kid":
                details = kid(
                    suite.inception(reference_paths), suite.inception(paths), subsets, subset_size
                )
                details["value"] = details["mean"]
                suite.close()
            elif metric == "cmmd":
                details = {
                    "value": cmmd(suite.clip_images(reference_paths), suite.clip_images(paths)),
                    "sigma": 10,
                    "scale": 1000,
                    "estimator": "biased",
                    "unit_normalized": True,
                }
            elif metric == "clipscore":
                details = clip_score(
                    suite.clip_images(paths), suite.clip_text([r["caption"] for r in valid])
                )
                details["value"] = details["mean"]
                details["text_lengths"] = suite.text_lengths
                details["sample_ids"] = [r["sample_id"] for r in valid]
            elif metric == "lpips":
                suite.close()
                details = _lpips_pairs(root, valid, image_root, device)
            else:
                suite.close()
                backend = (
                    MockVQA()
                    if vqa_backend == "mock"
                    else BlipVQA(root, asset_lock, device, batch_size)
                )
                details = evaluate_vqa(audited, image_root, backend, scope)
                questions = details["questions"]
                details["value"] = (
                    sum(q["status"] == "correct" for q in questions) / len(questions)
                    if questions
                    else None
                )
            question_errors = (
                sum(q["status"] == "error" for q in details["questions"]) if metric == "vqa" else 0
            )
            result["metrics"][metric] = {
                "status": (
                    "failed"
                    if question_errors and question_errors == len(details["questions"])
                    else "partial"
                    if question_errors
                    else "completed"
                ),
                "scope": scope,
                **details,
                "coverage": {"expected": len(audited), "valid": len(valid)},
                "complete": len(valid) == len(audited)
                and all(r["status"] == "completed" for r in reference_audit)
                and question_errors == 0,
                "error": f"{question_errors} VQA questions failed; see preserved question errors"
                if question_errors
                else None,
            }
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
            result["metrics"][metric] = {
                "status": "failed",
                "scope": scope,
                "value": None,
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
            }
        _atomic_json(output, result)
    suite.close()
    result["complete"] = (
        len(valid) == len(audited)
        and all(r["status"] == "completed" for r in reference_audit)
        and all(m["status"] == "completed" and m["complete"] for m in result["metrics"].values())
    )
    result["status"] = "completed" if result["complete"] else "partial"
    _atomic_json(output, result)
    return result
