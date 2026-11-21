"""在冻结头像题库上配对评测 B2 patch1 的 Uniform 与 Mid60。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import shutil
import textwrap
from collections import Counter
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter

from evaluate_anime_four_benchmark import (
    DEFECT_STATES,
    DEFECTS,
    METRIC_LABELS,
    digest,
    write_new_json,
)
from PIL import Image, ImageDraw
from render_anime_benchmark_blind import HTML, build_review, read_json
from score_anime_scale_benchmark import (
    ATTRIBUTE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FACE,
    PRESERVATION,
    SEEDS,
    SUCCESS,
    compare,
    summarize,
    swap_outcome,
)

from minigen._typing import JsonDict

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("artifacts/runs/anime_faces_256/b2_50m_p1_mid60_25k_30k")
SCOPE = Path("configs/evaluation/anime_b2_p1_mid60.json")
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_p1_mid60_benchmark")
BENCHMARK = Path("configs/benchmark/anime_faces_v1.json")
COLUMNS = ("left", "right")
ARMS = ("uniform", "mid60")
SCORE_SCHEMA = "minigen.anime_b2_p1_mid60.blind_scores.v1"
SHARED = (
    "data_identity",
    "model_config",
    "source_sha256",
    "checkpoint_step",
    "weights",
    "trainable_parameters",
    "condition_features_sha256",
    "condition_mask_sha256",
    "sampler_helper_sha256",
    "evaluation_torch",
    "condition_encoding",
    "seed_noise_sha256",
    "noise_shape",
    "tf32",
    "sampling_batch_size",
    "decode_batch_size",
    "sampling",
    "seeds",
)
RECORD_KEYS = ("id", "seed", "endpoint", "prompt", "noise_index", "noise_sha256", "image_path")


def read_scope(
    root: Path,
) -> JsonDict:
    """核对本轮题库、生成设置和配对评测的成功门槛。

    Args:
        root: 本次消融所用的 MiniGen 根目录。

    Raises:
        ValueError: 范围中的题库、步数、设置或门槛与定稿不符。

    Results:
        JsonDict: 预登记的实验范围。
    """
    scope = read_json(root / SCOPE)
    benchmark = read_json(root / BENCHMARK)
    if (
        scope["schema"] != "minigen.anime_b2_p1_mid60.scope.v1"
        or scope["benchmark_sha256"] != digest(root / BENCHMARK)
        or scope["seeds"] != list(SEEDS)
        or scope["sampling"] != benchmark["sampling"]
        or scope["parent_step"] != 25000
        or scope["final_step"] != 30000
        or scope["middle_probability"] != 0.6
        or scope["evaluation"]
        != {
            "questions": 208,
            "images_per_arm": 256,
            "new_images": 256,
            "face_min_net": 13,
            "iris_min_net": 8,
            "both_seeds_strictly_positive": True,
            "freeze_before_reveal": True,
        }
    ):
        raise ValueError("Mid60 scope differs from the registered benchmark or recipe")
    return scope


def read_uniform(
    root: Path,
    scope: JsonDict,
) -> JsonDict:
    """核对可直接复用的原 Uniform 30k 生成回执。

    Args:
        root: 旧四组评测产物所在根目录。
        scope: 记载旧回执及检查点摘要的预登记配置。

    Raises:
        ValueError: 旧回执或原推理设置不符合本次配对要求。

    Results:
        JsonDict: 原 Uniform 生成回执。
    """
    path = root / scope["baseline_receipt"]
    baseline = read_json(path)
    expected = {
        "arm": scope["arm"],
        "run_id": scope["parent_run_id"],
        "checkpoint_sha256": scope["baseline_sha256"],
        "source_sha256": scope["source_sha256"],
        "status": "completed",
        "checkpoint_step": scope["final_step"],
        "weights": "ema",
        "benchmark_sha256": scope["benchmark_sha256"],
        "rubric_sha256": scope["rubric_sha256"],
        "sampling": scope["sampling"],
        "seeds": scope["seeds"],
        "condition_encoding": "cpu_clip_fp16_roundtrip",
        "sampling_batch_size": 16,
        "decode_batch_size": 4,
        "tf32": False,
    }
    if (
        digest(path) != scope["baseline_receipt_sha256"]
        or len(baseline["records"]) != 256
        or any(baseline[key] != value for key, value in expected.items())
    ):
        raise ValueError("The existing Uniform generation is outside the registered scope")
    return baseline


def generate(
    root: Path,
    output: Path,
) -> None:
    """用验收后的 Mid60 EMA 生成 256 张图，并复用已验收的旧基线图。

    Args:
        root: 训练、组件和题库所在项目根目录。
        output: 本次配对评测根目录。

    Raises:
        ValueError: 训练验收、权重、文本条件或采样初噪与冻结协议不符。
        RuntimeError: CUDA、EMA 装载或推理失败。

    副作用:
        在 CUDA 上只执行推理，在独立评测目录发布 PNG 和生成回执。
    """
    import torch
    from evaluate_anime_time_ablation import tensor_digest
    from generate_anime_benchmark import (
        benchmark_noise,
        encode_benchmark_conditions,
        plan_endpoints,
        publish_benchmark_arm,
        read_protocol,
        validate_sources,
    )
    from run_anime_b1 import open_anime_cache
    from train_anime_six import read_arm_config

    from minigen.autoencoders.sdxl import FrozenVAE, hash_file
    from minigen.training.checkpoint import CHECKPOINT_SCHEMA, run_identity
    from minigen.training.config import build_model
    from minigen.training.runtime import resolve_device

    started = perf_counter()
    scope = read_scope(root)
    baseline = read_uniform(root, scope)
    acceptance_path = root / RUNS / "acceptance.json"
    acceptance = read_json(acceptance_path)
    expected = {
        "status": "verified",
        "step": 30000,
        "parent_step": 25000,
        "training_updates": 5000,
        "parent_sha256": scope["parent_sha256"],
        "parent_run_id": scope["parent_run_id"],
        "baseline_sha256": scope["baseline_sha256"],
        "source_sha256": scope["source_sha256"],
        "paired_stream_and_rng": True,
        "data_identity": baseline["data_identity"],
    }
    if any(acceptance[key] != value for key, value in expected.items()):
        raise ValueError("Mid60 training acceptance does not establish the matched 5k branch")
    package = Path(inspect.getfile(build_model)).resolve().parents[1]
    source_hash = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        source_hash.update(str(file.relative_to(package)).encode() + b"\0" + file.read_bytes())
    if (
        source_hash.hexdigest() != scope["source_sha256"]
        or str(torch.__version__) != baseline["evaluation_torch"]
        or hash_file(root / "scripts/generate_anime_benchmark.py")
        != baseline["sampler_helper_sha256"]
    ):
        raise ValueError("The inference environment differs from the accepted Uniform samples")
    benchmark, exclusions, protocol = read_protocol(root)
    device = resolve_device("cuda", "fp32")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, data = read_arm_config(root, scope["arm"])
    source, _, blank, identity = open_anime_cache(root, data)
    validate_sources(benchmark, exclusions, source, identity)
    endpoints = plan_endpoints(benchmark)
    with torch.inference_mode():
        condition = encode_benchmark_conditions(root, root / data["clip_lock"], endpoints)
        features_hash = tensor_digest(condition.features)
        mask_hash = tensor_digest(condition.mask)
        if (
            identity != baseline["data_identity"]
            or features_hash != baseline["condition_features_sha256"]
            or mask_hash != baseline["condition_mask_sha256"]
            or config.model != baseline["model_config"]
        ):
            raise ValueError("Actual data, model shape or condition tensors differ from Uniform")
        for seed in SEEDS:
            noise = benchmark_noise(seed)
            if tensor_digest(noise) != baseline["seed_noise_sha256"][str(seed)]:
                raise ValueError(f"Initial noise digest differs for seed {seed}")

        generation = output / "generation"
        generation.mkdir(parents=True, exist_ok=True)
        uniform = generation / "uniform"
        if not uniform.exists():
            base_dir = (root / scope["baseline_receipt"]).parent
            with TemporaryDirectory(prefix=".uniform-", dir=generation) as temp:
                pending = Path(temp) / "uniform"
                pending.mkdir()
                for row in baseline["records"]:
                    filename = row["image_path"]
                    old_image = base_dir / filename
                    if digest(old_image) != row["image_sha256"]:
                        raise ValueError(
                            f"Uniform PNG does not match its original receipt: {filename}"
                        )
                    shutil.copyfile(old_image, pending / filename)
                write_new_json(
                    pending / "receipt.json",
                    baseline
                    | {
                        "arm": "uniform",
                        "original_arm": scope["arm"],
                        "original_receipt_sha256": scope["baseline_receipt_sha256"],
                    },
                )
                pending.rename(uniform)

        target = generation / "mid60"
        if not target.exists():
            snapshot = root / RUNS / "step_030000.pt"
            checkpoint_sha = hash_file(snapshot)
            if checkpoint_sha != acceptance["checkpoint_sha256"]:
                raise ValueError("Mid60 checkpoint differs from the verified snapshot")
            state = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
            if (
                state["schema"] != CHECKPOINT_SCHEMA
                or state["step"] != scope["final_step"]
                or state["objective"] != "flow_velocity"
                or state["config"]["model"] != config.model
                or state["config"]["time_mid_probability"] != 0.6
                or state["data_identity"] != identity
                or run_identity(state["contract"]) != acceptance["run_id"]
            ):
                raise ValueError("Final Mid60 checkpoint identity is invalid")
            with torch.device("meta"):
                model = build_model(config.model)
            model.load_state_dict(state["ema"], strict=True, assign=True)
            model.eval().requires_grad_(False)
            parameter_count = sum(param.numel() for param in model.parameters())
            if parameter_count != baseline["trainable_parameters"]:
                raise ValueError("The Mid60 and Uniform model sizes differ")
            del state
            vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
            publish_benchmark_arm(
                target,
                model.to(device),
                vae,
                condition.to(device),
                blank.to(device),
                endpoints,
                benchmark["sampling"],
                protocol
                | {
                    "arm": "mid60",
                    "checkpoint_step": scope["final_step"],
                    "checkpoint_sha256": checkpoint_sha,
                    "run_id": acceptance["run_id"],
                    "source_sha256": scope["source_sha256"],
                    "data_identity": identity,
                    "model_config": config.model,
                    "trainable_parameters": parameter_count,
                    "weights": "ema",
                    "comparison_scope_sha256": digest(root / SCOPE),
                    "parent_sha256": scope["parent_sha256"],
                    "training_acceptance_sha256": digest(acceptance_path),
                    "condition_features_sha256": features_hash,
                    "condition_mask_sha256": mask_hash,
                    "sampler_helper_sha256": baseline["sampler_helper_sha256"],
                    "sampler_sha256": digest(Path(__file__)),
                },
                started,
            )
    paired_receipts(root, output)


def paired_receipts(
    root: Path,
    output: Path,
) -> dict[str, JsonDict]:
    """核对两组回执的真实身份及同题、双初噪配对。

    Args:
        root: 含冻结实验范围和原基线回执的项目根目录。
        output: 本次评测的两组生成目录所在路径。

    Raises:
        ValueError: 基线或候选身份、推理条件或逐图配对不符。

    Results:
        dict[str, JsonDict]: 按 Uniform、Mid60 索引的生成回执。
    """
    scope = read_scope(root)
    old = read_uniform(root, scope)
    receipts = {arm: read_json(output / "generation" / arm / "receipt.json") for arm in ARMS}
    uniform, mid60 = (receipts[arm] for arm in ARMS)
    if (
        uniform
        != old
        | {
            "arm": "uniform",
            "original_arm": scope["arm"],
            "original_receipt_sha256": scope["baseline_receipt_sha256"],
        }
        or mid60["arm"] != "mid60"
        or mid60["status"] != "completed"
        or mid60["run_id"] == uniform["run_id"]
        or mid60["parent_sha256"] != scope["parent_sha256"]
        or mid60["comparison_scope_sha256"] != digest(root / SCOPE)
        or any(uniform[key] != mid60[key] for key in SHARED)
        or [tuple(row[k] for k in RECORD_KEYS) for row in uniform["records"]]
        != [tuple(row[k] for k in RECORD_KEYS) for row in mid60["records"]]
    ):
        raise ValueError("Generated arms are not a valid, scope-matched pair")
    return receipts


def card_image(
    card: JsonDict,
    path: Path,
) -> None:
    """绘制去元数据的匿名双列题卡。

    Args:
        card: 已去掉模型归属的题卡及图片数据。
        path: PNG 写入路径。

    副作用:
        写入独立视觉评分用 PNG。
    """
    rows = card["endpoints"]
    canvas = Image.new("RGB", (534, 56 + len(rows) * 325), "#f7f5ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 8), card["id"], fill="#1c3038")
    for index, side in enumerate(COLUMNS):
        draw.text((9 + index * 262, 31), side, fill="#1c3038")
        for row_index, point in enumerate(rows):
            raw = base64.b64decode(card["images"][side][point["key"]].split(",", 1)[1])
            y = 52 + row_index * 325
            with Image.open(BytesIO(raw)) as picture:
                canvas.paste(picture.convert("RGB"), (4 + index * 262, y))
            caption = f"{point['key']}: {point['prompt']}"
            for line_index, line in enumerate(textwrap.wrap(caption, width=42)[:5]):
                draw.text((7 + index * 262, y + 258 + line_index * 11), line, fill="#1c3038")
    canvas.save(path)


def render(
    root: Path,
    output: Path,
) -> None:
    """为 512 张配对图生成不显示模型身份的 208 张双列题卡。

    Args:
        root: 包含范围配置和题库的项目根目录。
        output: 已有两组生成目录的本次评测根目录。

    Raises:
        ValueError: 回执、配对或图片摘要不符。

    副作用:
        创建独立匿名评分包、私有映射与导出清单，保留已发布的评分。
    """
    paired_receipts(root, output)
    review = output / "review"
    key_path = output / "private/key.json"
    if (review / "export_manifest.json").exists():
        verify_export(root, output)
        return
    if review.exists():
        if any((review / name).exists() for name in ("scores.json", "scores.lock.json")):
            raise FileExistsError("Cannot replace a review that already has scores")
        shutil.rmtree(review)
    key_path.unlink(missing_ok=True)
    deck, key = build_review(root / BENCHMARK, output / "generation", None)
    write_new_json(key_path, key)
    deck.update(
        columns=list(COLUMNS),
        defects=DEFECTS,
        score_schema=SCORE_SCHEMA,
        mapping_sha256=digest(key_path),
        comparison_scope_sha256=digest(root / SCOPE),
    )
    (review / "cards").mkdir(parents=True)
    template = {"schema": SCORE_SCHEMA, "review_id": deck["review_id"], "reviewer": "", "items": {}}
    for card in deck["items"]:
        template["items"][card["id"]] = {
            side: {
                "endpoints": {
                    point["key"]: {
                        "face": "",
                        "iris": "" if point["iris_required"] else "not_applicable",
                        "targets": {label: "" for label in point["targets"]},
                        "defects": {label: "" for label in DEFECTS},
                    }
                    for point in card["endpoints"]
                },
                **({"preservation": ""} if card["kind"] == "pair" else {}),
            }
            for side in COLUMNS
        } | {"preferences": {"left-right": ""}}
        card_image(card, review / "cards" / f"{card['id']}.png")
    public = deck | {
        "items": [{k: v for k, v in card.items() if k != "images"} for card in deck["items"]]
    }
    write_new_json(review / "deck.json", public)
    write_new_json(review / "score_template.json", template)
    document = HTML.replace("repeat(3,minmax(0,1fr))", "repeat(2,minmax(0,1fr))")
    (review / "blind_review.html").write_text(
        document.replace(
            "__ANONYMOUS_DATA__", json.dumps(deck, ensure_ascii=False).replace("</", r"<\/")
        )
    )
    files = [
        review / "deck.json",
        review / "score_template.json",
        review / "blind_review.html",
        *sorted((review / "cards").glob("*.png")),
        key_path,
    ]
    write_new_json(
        review / "export_manifest.json",
        {
            "schema": "minigen.anime_b2_p1_mid60.export_manifest.v1",
            "review_id": deck["review_id"],
            "image_count": 512,
            "card_count": 208,
            "files": {path.relative_to(output).as_posix(): digest(path) for path in files},
        },
    )
    print(f"Rendered 208 anonymous paired cards: {review}", flush=True)


def verify_export(
    root: Path,
    output: Path,
) -> JsonDict:
    """验证回传的双组生成图、题卡和匿名材料字节摘要。

    Args:
        root: 含原基线生成回执和本次范围的项目根目录。
        output: 本次评测的导出路径。

    Raises:
        ValueError: 数量、逐文件哈希或题卡标识不符。

    Results:
        JsonDict: 可作为 AutoDL 回传确认的值，不解析身份映射。
    """
    receipts = paired_receipts(root, output)
    manifest_path = output / "review/export_manifest.json"
    manifest = read_json(manifest_path)
    deck_path = output / "review/deck.json"
    deck = read_json(deck_path)
    expected = {
        "review/blind_review.html",
        "review/deck.json",
        "review/score_template.json",
        "private/key.json",
        *(f"review/cards/{card['id']}.png" for card in deck["items"]),
    }
    if (
        manifest["review_id"] != deck["review_id"]
        or manifest["card_count"] != 208
        or manifest["image_count"] != 512
        or len(deck["items"]) != 208
        or len(expected) != 212
        or set(manifest["files"]) != expected
        or manifest["files"]["private/key.json"] != deck["mapping_sha256"]
    ):
        raise ValueError("Export does not cover the exact anonymous card set")
    for name, expected_hash in manifest["files"].items():
        if digest(output / name) != expected_hash:
            raise ValueError(f"Export file digest differs: {name}")
    for arm, receipt in receipts.items():
        for row in receipt["records"]:
            path = output / "generation" / arm / row["image_path"]
            if digest(path) != row["image_sha256"]:
                raise ValueError(f"Export image digest differs: {path}")
    return {
        "schema": "minigen.anime_b2_p1_mid60.local_export_verified.v1",
        "review_id": deck["review_id"],
        "scope_sha256": digest(root / SCOPE),
        "deck_sha256": digest(deck_path),
        "export_manifest_sha256": digest(manifest_path),
        "all_png_hashes_verified": True,
        "generation_png_count": 512,
        "anonymous_card_count": 208,
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> None:
    """核对匿名评分身份、题目覆盖、字段和取值。

    Args:
        deck: 不含身份的公开题卡。
        scores: 完整的匿名双列视觉评分。

    Raises:
        ValueError: 评分不完整，或同一端点的属性与虹膜状态矛盾。
    """
    if (
        scores.get("schema") != SCORE_SCHEMA
        or scores.get("review_id") != deck["review_id"]
        or not scores.get("reviewer", "").strip()
        or set(scores["items"]) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Scores are incomplete or belong to another review")
    for card in deck["items"]:
        item = scores["items"][card["id"]]
        preferences = item["preferences"]
        if set(preferences) != {"left-right"} or preferences["left-right"] not in (
            *COLUMNS,
            "tie",
            "not_comparable",
        ):
            raise ValueError(f"Missing or invalid preference: {card['id']}")
        for side in COLUMNS:
            rating = item[side]
            paired = card["kind"] == "pair"
            if paired and rating["preservation"] not in PRESERVATION:
                raise ValueError(f"Invalid preservation: {card['id']}/{side}")
            if set(rating["endpoints"]) != {point["key"] for point in card["endpoints"]}:
                raise ValueError(f"Incomplete endpoints: {card['id']}/{side}")
            for point in card["endpoints"]:
                row = rating["endpoints"][point["key"]]
                iris = (
                    ("readable", "unreadable", "failure")
                    if point["iris_required"]
                    else ("not_applicable",)
                )
                if (
                    set(row) != {"face", "iris", "targets", "defects"}
                    or row["face"] not in FACE
                    or row["iris"] not in iris
                    or set(row["targets"]) != set(point["targets"])
                    or any(value not in ATTRIBUTE for value in row["targets"].values())
                    or set(row["defects"]) != set(DEFECTS)
                    or any(value not in DEFECT_STATES for value in row["defects"].values())
                    or (
                        "eye_color" in row["targets"]
                        and row["iris"] in {"unreadable", "failure"}
                        and row["targets"]["eye_color"] != row["iris"]
                    )
                ):
                    raise ValueError(f"Invalid endpoint scores: {card['id']}/{side}/{point['key']}")


def score_fingerprint(
    root: Path,
    output: Path,
    scores_path: Path,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """验证题卡、评分及公开身份承诺，不读取私有映射的内容。

    Args:
        root: 本轮冻结范围配置的根目录。
        output: 已发布的匿名评测目录。
        scores_path: 待锁定或已锁定的评分 JSON。

    Raises:
        ValueError: 题卡、更改后的文件或匿名评分与发布记录不符。

    Results:
        tuple: 公开题卡、完整评分及锁定所需的摘要。
    """
    provenance = verify_export(root, output)
    deck = read_json(output / "review/deck.json")
    scores = read_json(scores_path)
    validate_scores(deck, scores)
    return (
        deck,
        scores,
        {
            "schema": "minigen.anime_b2_p1_mid60.blind_scores_lock.v1",
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "scores_sha256": digest(scores_path),
            "deck_sha256": provenance["deck_sha256"],
            "html_sha256": digest(output / "review/blind_review.html"),
            "mapping_sha256": deck["mapping_sha256"],
            "comparison_scope_sha256": provenance["scope_sha256"],
            "export_manifest_sha256": provenance["export_manifest_sha256"],
            "counts": {"cards": 208, "images": 512, "iris": 304, "eye_color": 256, "swaps": 96},
        },
    )


def freeze(
    root: Path,
    output: Path,
    scores_path: Path,
) -> None:
    """在解盲前保存评分文件和锁定摘要。

    Args:
        root: 本轮范围及基线材料所在项目根目录。
        output: 本轮匿名评分目录。
        scores_path: 所有题目完成评分后的原始文件。

    Raises:
        FileExistsError: 本轮已锁分，须保留既有评分与锁。

    副作用:
        写入独立的不可覆盖评分文件及锁。
    """
    _, _, fingerprint = score_fingerprint(root, output, scores_path)
    review = output / "review"
    lock_path = review / "scores.lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    frozen = review / "scores.json"
    if not frozen.exists():
        with frozen.open("xb") as copy:
            copy.write(scores_path.read_bytes())
    if digest(frozen) != fingerprint["scores_sha256"]:
        raise ValueError("Existing frozen scores differ from the submitted file")
    write_new_json(lock_path, fingerprint | {"locked_at": datetime.now(UTC).isoformat()})
    print(f"Locked 208 cards / 512 scores: {lock_path}", flush=True)


def locked_inputs(
    root: Path,
    output: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """先通过评分锁核验，再解读已承诺摘要的模型身份。

    Args:
        root: 冻结范围配置的项目根目录。
        output: 本轮评分和解盲目录。

    Raises:
        ValueError: 锁分后评分、题卡、映射或实验范围改变。

    Results:
        tuple: 公开题卡、锁定评分、模型身份及锁分回执。
    """
    review = output / "review"
    lock = read_json(review / "scores.lock.json")
    deck, scores, current = score_fingerprint(root, output, review / "scores.json")
    if not lock.get("locked_at") or any(lock.get(k) != v for k, v in current.items()):
        raise ValueError("Scores or provenance changed after locking")
    key = read_json(output / "private/key.json")
    if (
        key["review_id"] != deck["review_id"]
        or key["benchmark_sha256"] != digest(root / BENCHMARK)
        or set(key["assignments"]) != set(scores["items"])
    ):
        raise ValueError("Identity mapping does not match the locked review")
    return deck, scores, key, lock


def reveal(
    root: Path,
    output: Path,
) -> None:
    """解盲并计算原六项指标、眼色干预子集、缺陷和双种子门槛。

    Args:
        root: 冻结范围及原评测协议所在项目根目录。
        output: 本轮已锁定的评分和显示文件。

    副作用:
        生成独立的解盲结果 JSON，保留原评分文件。
    """
    _, scores, key, lock = locked_inputs(root, output)
    observations = {arm: {metric: {} for metric in SUCCESS} for arm in ARMS}
    defects = {
        arm: {label: {str(seed): Counter() for seed in SEEDS} for label in DEFECTS} for arm in ARMS
    }
    eye_change = {arm: {str(seed): Counter() for seed in SEEDS} for arm in ARMS}
    preferences = {str(seed): Counter() for seed in SEEDS}
    for card_id, card in scores["items"].items():
        row = key["assignments"][card_id]
        seed = str(row["seed"])
        prefix = row["group_id"], row["question_id"], row["seed"]
        for side in COLUMNS:
            arm, rating = row[side], card[side]
            model = observations[arm]
            for label, score in rating["endpoints"].items():
                point = (*prefix, label)
                model["face"][point] = score["face"]
                model["iris"][point] = score["iris"]
                model["eye_color"][point] = score["targets"].get("eye_color", "not_applicable")
                for name, status in score["defects"].items():
                    defects[arm][name][seed][status] += 1
            if row["kind"] == "pair":
                point, changed = (*prefix, "swap"), row["changed_attribute"]
                first = rating["endpoints"]["a"]["targets"][changed]
                second = rating["endpoints"]["b"]["targets"][changed]
                preserved = rating["preservation"]
                model["swap_switch"][point] = swap_outcome(first, second, "preserved")
                model["swap_preservation"][point] = preserved
                model["swap_joint"][point] = swap_outcome(first, second, preserved)
                if changed == "eye_color":
                    bucket = eye_change[arm][seed]
                    bucket["denominator"] += 1
                    bucket["switch_correct"] += model["swap_switch"][point] == "correct"
                    bucket["preserved"] += preserved == "preserved"
                    bucket["joint_correct"] += model["swap_joint"][point] == "correct"
        picked = card["preferences"]["left-right"]
        preferences[seed][row[picked] if picked in COLUMNS else picked] += 1

    groups = sorted({point[0] for point in observations["uniform"]["face"]})
    rng = Random(BOOTSTRAP_SEED)
    draws = [
        tuple(rng.choices(range(len(groups)), k=len(groups))) for _ in range(BOOTSTRAP_REPLICATES)
    ]
    models = {
        arm: {
            "overall": {metric: summarize(points, metric) for metric, points in model.items()},
            "by_seed": {
                str(seed): {
                    metric: summarize(points, metric, seed) for metric, points in model.items()
                }
                for seed in SEEDS
            },
            "defects_by_seed": defects[arm],
            "eye_color_interventions_by_seed": eye_change[arm],
        }
        for arm, model in observations.items()
    }
    paired = {
        metric: compare(observations["uniform"][metric], points, metric, groups, draws)
        for metric, points in observations["mid60"].items()
    }
    write_new_json(
        output / "review/deblinded_metrics.json",
        {
            "schema": "minigen.anime_b2_p1_mid60.metrics.v1",
            "locked_review": lock,
            "sampling": key["sampling"],
            "checkpoint_identities": key["checkpoint_identities"],
            "models": models,
            "paired": paired,
            "preferences_by_seed": preferences,
            "passes": paired["face"]["passes"] and paired["iris"]["passes"],
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "source_groups": len(groups),
            },
            "scope": "固定复用 208 题，单训练种子，两初噪，视觉盲评。",
        },
    )
    print("Revealed and summarized 208 locked cards", flush=True)


def report(
    root: Path,
    output: Path,
) -> None:
    """汇总解盲结果和逐题双列对照，生成独立 HTML 报告。

    Args:
        root: 本次定稿范围和原权重回执所在根目录。
        output: 已解盲的评测根目录。

    Raises:
        ValueError: 成绩文件对应的评分锁与当前评分不符。

    副作用:
        写入报告，内含全量配对图和代表性端点的本地链接。
    """
    deck, scores, key, lock = locked_inputs(root, output)
    metrics = read_json(output / "review/deblinded_metrics.json")
    if metrics["locked_review"] != lock:
        raise ValueError("The metrics are not from this locked review")
    table_rows = []
    for name, paired in metrics["paired"].items():
        totals = [metrics["models"][arm]["overall"][name] for arm in ARMS]
        rates = "".join(f"<td>{item['success']}/{item['applicable']}</td>" for item in totals)
        seed_deltas = "".join(f"<td>{item['net']:+d}</td>" for item in paired["by_seed"].values())
        low, high = paired["bootstrap_95_percent_pp"]
        table_rows.append(
            f"<tr><th>{METRIC_LABELS[name]}</th>{rates}<td>{paired['net']:+d}</td>"
            f"{seed_deltas}<td>{low:+.2f} 至 {high:+.2f}</td></tr>"
        )

    defect_rows = []
    for label, name in DEFECTS.items():
        counts = [
            sum(
                outcomes.get("present", 0)
                for outcomes in metrics["models"][arm]["defects_by_seed"][label].values()
            )
            for arm in ARMS
        ]
        defect_rows.append(
            f"<tr><th>{name}</th><td>{counts[0]}/256</td><td>{counts[1]}/256</td>"
            f"<td>{counts[1] - counts[0]:+d}</td></tr>"
        )

    eye_rows = []
    for name, label in (
        ("switch_correct", "两端眼色正确"),
        ("preserved", "非目标内容保持"),
        ("joint_correct", "眼色换词联合成功"),
    ):
        counts = [
            sum(
                outcomes.get(name, 0)
                for outcomes in metrics["models"][arm]["eye_color_interventions_by_seed"].values()
            )
            for arm in ARMS
        ]
        eye_rows.append(
            f"<tr><th>{label}</th><td>{counts[0]}/32</td><td>{counts[1]}/32</td>"
            f"<td>{counts[1] - counts[0]:+d}</td></tr>"
        )

    detail_rows = []
    examples: dict[str, list[str]] = {"improved": [], "regressed": []}
    for card in deck["items"]:
        row = key["assignments"][card["id"]]
        rating = scores["items"][card["id"]]
        sides = {row[side]: side for side in COLUMNS}
        images = []
        for point in card["endpoints"]:
            label = point["key"]
            suffix = "" if label == "single" else f"_{label}"
            filename = f"{row['question_id']}_{row['seed']}{suffix}.png"
            parts = []
            for arm in ARMS:
                score = rating[sides[arm]]["endpoints"][label]
                parts.append(
                    f'<figure><img loading="lazy" src="generation/{arm}/{filename}" '
                    f'alt="{arm}"><figcaption>{arm} · {score["face"]} · {score["iris"]}'
                    "</figcaption></figure>"
                )
            images.append(
                f"<p>{escape(point['prompt'])}</p><div class='pair'>{''.join(parts)}</div>"
            )
        title = f"{row['question_id']} · {row['seed']}"
        body = f"<details><summary>{title}</summary>{''.join(images)}</details>"
        detail_rows.append((row["question_id"], row["seed"], body))
        for point in card["endpoints"]:
            usable = [
                rating[sides[arm]]["endpoints"][point["key"]]["face"] == "usable" for arm in ARMS
            ]
            if usable[0] != usable[1]:
                direction = "improved" if usable[1] else "regressed"
                if len(examples[direction]) < 2 and body not in examples[direction]:
                    examples[direction].append(body)
    by_seed = {
        arm: {
            "scores": model["by_seed"],
            "defects": model["defects_by_seed"],
            "eye_color_interventions": model["eye_color_interventions_by_seed"],
        }
        for arm, model in metrics["models"].items()
    }
    supplementary = escape(
        json.dumps(
            by_seed | {"preferences": metrics["preferences_by_seed"]}, ensure_ascii=False, indent=2
        )
    )
    outcome = "通过预登记门槛" if metrics["passes"] else "未达到预登记门槛"
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B2 patch1 时间采样对照</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1040px;margin:2rem auto;padding:0 1rem;
background:#faf9f5;color:#152e36}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:.55rem;border:1px solid #bdd0d2;
text-align:left}} .pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}
figure{{margin:0}} img{{width:100%;max-width:256px}} details{{padding:.4rem 0}}
summary{{cursor:pointer}} pre{{overflow:auto}}
</style></head><body>
<h1>B2 50M patch1 · Uniform 对 Mid60</h1>
<p>从同一个 25k 完整检查点分支，到 30k。原 208 题、相同双初噪，CFG3、Euler40。</p>
<p><strong>{outcome}</strong>。头像可用净增至少 13/256，虹膜可辨净增至少 8/152，
且每项在两个初噪下均提升。</p>
<table><thead><tr><th>指标</th><th>Uniform</th><th>Mid60</th><th>净增</th>
<th>初噪 1</th><th>初噪 2</th><th>配对差值 95% 区间（百分点）</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table>
<h2>五官缺陷</h2><p>分母为所有计划端点，表内计数为明确出现缺陷。</p>
<table><tr><th>缺陷</th><th>Uniform</th><th>Mid60</th><th>净增</th></tr>
{"".join(defect_rows)}</table>
<h2>眼色换词</h2><p>16 个眼色干预题，每题两个初噪。</p>
<table><tr><th>指标</th><th>Uniform</th><th>Mid60</th><th>净增</th></tr>
{"".join(eye_rows)}</table>
<details><summary>分种子计数、眼色干预、五官缺陷和头像偏好</summary>
<pre>{supplementary}</pre></details>
<h2>头像改善示例</h2>{"".join(examples["improved"]) or "<p>没有净改善端点。</p>"}
<h2>头像退步示例</h2>{"".join(examples["regressed"]) or "<p>没有退步端点。</p>"}
<h2>逐题双列图</h2>{"".join(body for _, _, body in sorted(detail_rows))}
<p><a href="review/deblinded_metrics.json">详细成绩</a> ·
<a href="review/scores.lock.json">评分锁</a></p>
</body></html>
"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Report: {output / 'index.html'}", flush=True)


def main() -> None:
    """按阶段执行固定题库生成、匿名评分锁定和配对结果发布。

    副作用:
        执行选定阶段并写入独立实验目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate", "render", "verify-export", "freeze", "reveal", "report")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if args.action == "verify-export":
        marker = verify_export(root, output)
        print(json.dumps(marker, ensure_ascii=False, indent=2))
    elif args.action == "freeze":
        if args.scores is None:
            parser.error("freeze requires --scores")
        freeze(root, output, args.scores)
    else:
        actions = {"generate": generate, "render": render, "reveal": reveal, "report": report}
        actions[args.action](root, output)


if __name__ == "__main__":
    main()
