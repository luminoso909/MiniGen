"""采样六组 30k 动漫模型，匿名锁分后汇总结构、虹膜和采样设置。"""

from __future__ import annotations

import argparse
import base64
import html
import json
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import wrap
from time import perf_counter

import torch
from evaluate_anime_time_ablation import tensor_digest
from PIL import Image, ImageDraw, ImageFont
from render_anime_benchmark_blind import image_data, order_key, read_json
from render_anime_repa_blind import scoring_endpoint
from render_anime_roi_blind import assignment_digest
from run_anime_b1 import ROOT, open_anime_cache
from torch import nn
from torchvision.utils import save_image
from train_anime_six import ARMS, RUNS, read_arm_config

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import selected_indices
from minigen.contracts import Condition
from minigen.samplers.flow import integrate
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device, synchronize

OUTPUT = Path("artifacts/exports/anime-faces-b1/six_30k_v1")
STEP = 30000
COUNT = 16
SEED = 5001
GUIDANCE = (0, 1, 3, 5)
SOLVERS = (("euler", 40), ("heun", 20))
COLUMNS = ("A", "B", "C", "D", "E", "F")
SCHEMA = "minigen.anime_six.blind_scores.v1"
FACE = ("usable", "distorted", "unrecognizable", "failure")
IRIS = ("readable", "unreadable", "not_applicable", "failure")
HEAD = ("clear", "broken", "uncertain", "failure")
CLARITY = ("clear", "unclear", "uncertain", "failure")
QUALITY = ("good", "acceptable", "poor", "failure")
DEFECT = ("present", "absent", "uncertain", "not_applicable")
ATTRIBUTE = ("correct", "wrong", "unreadable", "failure")
FACE_FIELDS = ("face", "iris", "head_contour", "face_clarity", "image_quality")
DEFECTS = {
    "eye_fragmentation": "眼周断裂",
    "nose_mouth_missing": "鼻口缺失或移位",
    "contour_fusion": "脸缘粘连",
}
RUBRIC = {
    "face": "头像可用要求头部明确，五官及轮廓连贯。明显错位或断裂记 distorted。",
    "iris": "至少一只应睁开可见的眼睛有独立虹膜，能与眼皮、肤色及阴影分开。",
    "head_contour": "头部及下颌轮廓可追踪且没有无法解释的断裂记 clear。",
    "face_clarity": "可见五官的位置与边缘清楚记 clear；自然遮挡按可见部分判断。",
    "image_quality": "good 为清楚完整，acceptable 为主体完整且有轻微伪影，poor 有明显生成伪影。",
    "eye_fragmentation": "可见眼皮、眼球或眼眶出现断裂、重复、错位或与旁物粘连。",
    "nose_mouth_missing": "无遮挡面部的鼻口无法定位、熔化或移位。正常动漫简笔鼻不算缺陷。",
    "contour_fusion": "脸颊或下颌与头发、衣服或背景不合理融为一体。",
    "uncertainty": "自然遮挡、侧脸、裁切和正常画风本身不算缺陷，证据不足时记 uncertain。",
    "cfg0": "CFG 0 为无条件生成。文本属性不评分，虹膜按实际可见睁眼判断，双眼自然闭合可记不适用。",
    "preferred": "选择同题中最适合做头像的列，可同时选择并列最佳；六列无明确差异时全选。",
}


def atomic_json(
    path: Path,
    payload: JsonDict,
) -> None:
    """原子发布可恢复流程的小型元数据。

    Args:
        path: 目标 JSON 路径。
        payload: 有限 JSON 对象。

    Raises:
        OSError/ValueError/TypeError: 写入或 JSON 序列化失败。

    副作用:
        写入临时文件后替换目标。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def condition_name(
    guidance: int,
    method: str,
    steps: int,
) -> str:
    """为一个采样设置返回稳定的目录名。

    Args:
        guidance: 当前 CFG 系数。
        method: euler 或 heun。
        steps: 对应积分步数。

    Results:
        str: 同时含 CFG、方法和积分步数的名称。
    """
    return f"cfg{guidance}_{method}{steps}"


def load_ema(
    root: Path,
    arm: str,
    identity: JsonDict,
) -> tuple[nn.Module, JsonDict]:
    """核对真实完成的 30k 分支，仅映射其 FP32 EMA 主干。

    Args:
        root: AutoDL 项目根目录。
        arm: 六组中的分支名。
        identity: 已验收训练和验证缓存身份。

    Raises:
        OSError/ValueError/RuntimeError: 文件、训练状态、模型或数据身份不符。

    Results:
        tuple: CPU EMA 网络及该分支的公开生成来源。
    """
    config, _ = read_arm_config(root, arm)
    checkpoint = root / RUNS / arm / f"step_{STEP:06d}.pt"
    result = read_json(checkpoint.with_name("result.json"))
    run = read_json(checkpoint.with_name("run.json"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    run_id = run_identity(state["contract"])
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["objective"] != "flow_velocity"
        or state["scheduler"]["step"] != STEP
        or state["step"] != STEP
        or state["config"]["model"] != config.model
        or state["config"]["seed"] != 42
        or state["config"]["max_steps"] != STEP
        or state["contract"]["initialization"]["kind"] != "random"
        or state["contract"]["initialization"]["seed"] != 42
        or state["data_identity"] != identity
        or result["status"] != "completed"
        or result["step"] != STEP
        or result["run_id"] != run_id
        or result["attempt_id"] != run["attempt_id"]
        or result["performance"]["optimizer_steps"] != STEP
        or run["contract"] != state["contract"]
    ):
        raise ValueError(f"Checkpoint does not match a completed six-arm run: {arm}")
    with torch.device("meta"):
        model = build_model(config.model)
    model.load_state_dict(state["ema"], strict=True, assign=True)
    model.eval().requires_grad_(False)
    metadata = {
        "arm": arm,
        "run_id": run_id,
        "model_config": config.model,
        "checkpoint_sha256": hash_file(checkpoint),
        "checkpoint_step": STEP,
        "weights": "ema",
        "training_seed": 42,
        "source_sha256": state["contract"]["source_sha256"],
        "trainable_parameters": result["trainable_parameters"],
        "data_identity": identity,
    }
    return model, metadata


def generate(
    root: Path,
    output: Path,
    arm: str,
    device_name: str,
    batch_size: int,
) -> None:
    """为一组最终模型生成可按采样设置恢复的 128 张固定样图。

    Args:
        root: 原训练缓存和权重所在项目根目录。
        output: 六组采样、匿名评分及报告目录。
        arm: 需要执行的实验分支。
        device_name: cuda 或 cpu，使用同样 FP32 协议。
        batch_size: 积分微批，CPU 可显式使用 1。

    Raises:
        OSError/ValueError/RuntimeError: 训练身份、已有采样身份或执行设备不符。

    副作用:
        加载冻结模型，设置 CPU 线程数和 TF32 选项，写 PNG 和采样回执。
    """
    config, data = read_arm_config(root, arm)
    device = resolve_device(device_name, "fp32")
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, cache, blank, identity = open_anime_cache(root, data)
    indices = selected_indices(source, data, "validation")[:COUNT]
    entries = [cache.split("validation")[index] for index in indices]
    if len(entries) != COUNT:
        raise ValueError("Six-arm sampling needs the original 16 development labels")
    conditions = Condition(
        torch.stack([row.text for row in entries]).float(),
        torch.stack([row.mask for row in entries]),
    ).to(device)
    blank = blank.to(device)
    noise = torch.randn((COUNT, 4, 32, 32), generator=torch.Generator().manual_seed(SEED))
    labels = [
        {
            "index": index,
            "sample_id": source.records["validation"][source_index]["sample_id"],
            "caption": source.records["validation"][source_index]["caption"],
            "noise_sha256": tensor_digest(noise[index : index + 1]),
        }
        for index, source_index in enumerate(indices)
    ]
    model, metadata = load_ema(root, arm, identity)
    model = model.to(device=device, dtype=torch.float32)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    destination = output / "generation" / arm
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for guidance in GUIDANCE:
        for method, steps in SOLVERS:
            name = condition_name(guidance, method, steps)
            directory = destination / name
            expected = metadata | {
                "seed": SEED,
                "guidance": guidance,
                "method": method,
                "steps": steps,
                "precision": "fp32",
                "device": str(device),
                "sampling_batch_size": batch_size,
                "sampler_sha256": hash_file(Path(__file__)),
                "torch_version": torch.__version__,
            }
            receipt = directory / "receipt.json"
            if receipt.exists():
                existing = read_json(receipt)
                if existing["identity"] != expected:
                    raise ValueError(f"Existing samples have another identity: {directory}")
                for row in existing["records"]:
                    if hash_file(destination / row["image_path"]) != row["image_sha256"]:
                        raise ValueError(f"Existing PNG differs: {row['image_path']}")
                rows.extend(existing["records"])
                continue
            directory.mkdir(exist_ok=True)
            started = perf_counter()
            records = []
            with torch.inference_mode():
                for first in range(0, COUNT, batch_size):
                    last = min(first + batch_size, COUNT)
                    condition = Condition(
                        conditions.features[first:last], conditions.mask[first:last]
                    )
                    result = integrate(
                        model,
                        noise[first:last].to(device),
                        condition,
                        blank,
                        steps=steps,
                        guidance=float(guidance),
                        method=method,
                    )
                    images = (vae.decode(result.samples).clamp(-1, 1).cpu() + 1) / 2
                    for offset, image in enumerate(images):
                        index = first + offset
                        path = directory / f"sample_{index:02d}.png"
                        pending = path.with_name(path.stem + ".pending.png")
                        save_image(image, pending)
                        pending.replace(path)
                        records.append(
                            labels[index]
                            | {
                                "guidance": guidance,
                                "method": method,
                                "steps": steps,
                                "seed": SEED,
                                "image_path": str(path.relative_to(destination)),
                                "image_sha256": hash_file(path),
                            }
                        )
            synchronize(device)
            atomic_json(
                receipt,
                {
                    "identity": expected,
                    "status": "completed",
                    "records": records,
                    "solver_nfe": steps * (2 if method == "heun" else 1),
                    "model_evaluations": 40 * (1 if guidance in (0, 1) else 2),
                    "elapsed_seconds": perf_counter() - started,
                },
            )
            rows.extend(records)
            print(f"{arm} {name}: {len(records)} images", flush=True)
    atomic_json(
        destination / "receipt.json",
        metadata
        | {
            "schema": "minigen.anime_six.generation.v1",
            "status": "completed",
            "seed": SEED,
            "precision": "fp32",
            "records": rows,
            "condition_order": [condition_name(g, m, s) for g in GUIDANCE for m, s in SOLVERS],
            "image_count": len(rows),
        },
    )


def verify_generation(
    output: Path,
) -> dict[str, JsonDict]:
    """核对六组完整 PNG、来源及同标签同初噪的排列。

    Args:
        output: 已有六组生成回执的目录。

    Raises:
        OSError/ValueError: 生成缺失、身份、顺序或 PNG 哈希不符。

    Results:
        dict: 六组已核对的生成回执。
    """
    receipts = {arm: read_json(output / "generation" / arm / "receipt.json") for arm in ARMS}
    fields = (
        "guidance",
        "method",
        "steps",
        "seed",
        "index",
        "sample_id",
        "caption",
        "noise_sha256",
    )
    reference = next(iter(receipts.values()))
    expected = [(g, m, s, SEED, i) for g in GUIDANCE for m, s in SOLVERS for i in range(COUNT)]
    reference_rows = [tuple(row[key] for key in fields) for row in reference["records"]]
    for arm, receipt in receipts.items():
        if (
            receipt["status"] != "completed"
            or receipt["arm"] != arm
            or receipt["checkpoint_step"] != STEP
            or receipt["data_identity"] != reference["data_identity"]
            or receipt["seed"] != SEED
            or receipt["precision"] != "fp32"
            or [tuple(row[key] for key in fields) for row in receipt["records"]] != reference_rows
            or [tuple(row[key] for key in fields[:5]) for row in receipt["records"]] != expected
        ):
            raise ValueError(f"Six-arm generation is incomplete or unpaired: {arm}")
        for row in receipt["records"]:
            image_data(output / "generation" / arm / row["image_path"], row["image_sha256"], 256)
    return receipts


def draw_card(
    path: Path,
    card: JsonDict,
    pictures: dict[str, Image.Image],
) -> None:
    """绘制六列匿名原尺寸评分图，便于离线视觉评审。

    Args:
        path: 合成 PNG 路径。
        card: 公开匿名题义和条件信息。
        pictures: A–F 的 RGB256 图像。

    Raises:
        OSError: 图像无法保存。

    副作用:
        写一张 3 列两行的匿名题卡 PNG。
    """
    canvas = Image.new("RGB", (816, 652), "#f5f3ec")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=17)
    font = ImageFont.load_default(size=14)
    title = f"{card['id']} | CFG {card['guidance']} | {card['method']} {card['steps']}"
    draw.text((10, 9), title, "#243544", font=heading)
    prompt = "Unconditional generation" if card["guidance"] == 0 else card["prompt"]
    draw.multiline_text((10, 33), "\n".join(wrap(prompt, 105)), "#243544", font=font)
    for index, column in enumerate(COLUMNS):
        x, y = 10 + 269 * (index % 3), 80 + 284 * (index // 3)
        draw.text((x, y), column, "#243544", font=heading)
        canvas.paste(pictures[column], (x, y + 22))
    canvas.save(path)


def render(
    output: Path,
) -> None:
    """将 768 张图打包为 128 道隐藏模型身份的六列题卡。

    Args:
        output: 六组生成图和评分工作区。

    Raises:
        OSError/ValueError: 来源不完整、题卡已存在或文件不可写。

    副作用:
        写独立的匿名页面、题卡、评分模板和私有映射。
    """
    receipts = verify_generation(output)
    reference = next(iter(receipts.values()))
    review, private = output / "review", output / "private"
    if review.exists() or private.exists():
        raise FileExistsError("Six-arm anonymous review already exists")
    secret = secrets.token_bytes(32)
    salt = secrets.token_hex(32)
    review_id = order_key(secret, "six30k:5001")[:24]
    cards, mapping, embedded = [], {}, {}
    with TemporaryDirectory(dir=output) as pending:
        staged_review, staged_private = Path(pending) / "review", Path(pending) / "private"
        pictures = staged_review / "cards"
        pictures.mkdir(parents=True)
        staged_private.mkdir(mode=0o700)
        for guidance in GUIDANCE:
            for method, steps in SOLVERS:
                positions = [
                    i
                    for i, row in enumerate(reference["records"])
                    if (row["guidance"], row["method"]) == (guidance, method)
                ]
                positions.sort(key=lambda i: order_key(secret, f"side:{guidance}:{method}:{i}"))
                base_order = sorted(
                    ARMS, key=lambda arm: order_key(secret, f"base:{guidance}:{method}:{arm}")
                )
                for ordinal, position in enumerate(positions):
                    row = reference["records"][position]
                    opaque = order_key(secret, f"card:{guidance}:{method}:{row['index']}")[:16]
                    endpoint = scoring_endpoint("single", row["caption"])
                    card = {
                        "id": opaque,
                        "guidance": guidance,
                        "method": method,
                        "steps": steps,
                        "prompt": row["caption"] if guidance else "Unconditional generation",
                        "targets": endpoint["targets"] if guidance else {},
                        "iris_required": endpoint["iris_required"] if guidance else None,
                        "image_sha256": {},
                    }
                    layout = base_order[ordinal % 6 :] + base_order[: ordinal % 6]
                    assignment = {
                        "index": row["index"],
                        "seed": SEED,
                        "guidance": guidance,
                        "method": method,
                        "steps": steps,
                        "columns": dict(zip(COLUMNS, layout, strict=True)),
                    }
                    card_pictures, images = {}, {}
                    for column, arm in assignment["columns"].items():
                        record = receipts[arm]["records"][position]
                        uri = image_data(
                            output / "generation" / arm / record["image_path"],
                            record["image_sha256"],
                            256,
                        )
                        target = pictures / f"{opaque}_{column}.png"
                        target.write_bytes(base64.b64decode(uri.partition(",")[2]))
                        card["image_sha256"][column] = hash_file(target)
                        images[column] = uri
                        with Image.open(target) as image:
                            card_pictures[column] = image.convert("RGB").copy()
                    draw_card(pictures / f"{opaque}.png", card, card_pictures)
                    card["card_sha256"] = hash_file(pictures / f"{opaque}.png")
                    cards.append(card)
                    mapping[opaque] = assignment
                    embedded[opaque] = images
        cards.sort(key=lambda card: order_key(secret, f"order:{card['id']}"))
        deck = {
            "review_id": review_id,
            "score_schema": SCHEMA,
            "seed": SEED,
            "columns": COLUMNS,
            "assignment_sha256": assignment_digest(mapping, salt),
            "rubric": RUBRIC,
            "items": cards,
        }
        atomic_json(staged_review / "deck.json", deck)
        atomic_json(
            staged_review / "score_template.json",
            {
                "schema": SCHEMA,
                "review_id": review_id,
                "reviewer": "",
                "items": {
                    card["id"]: {
                        "preferred": [],
                        "columns": {
                            column: {
                                **dict.fromkeys(FACE_FIELDS, ""),
                                "defects": dict.fromkeys(DEFECTS, ""),
                                "targets": dict.fromkeys(card["targets"], ""),
                            }
                            for column in COLUMNS
                        },
                        "note": "",
                    }
                    for card in cards
                },
            },
        )
        page_data = deck | {"images": embedded, "rating_options": rating_options()}
        serialized = json.dumps(page_data, ensure_ascii=False).replace("<", "\\u003c")
        (staged_review / "blind_review.html").write_text(BLIND_HTML.replace("__DATA__", serialized))
        atomic_json(
            staged_private / "assignment.json",
            {
                "review_id": review_id,
                "salt": salt,
                "assignments": mapping,
                "deck_sha256": hash_file(staged_review / "deck.json"),
                "review_html_sha256": hash_file(staged_review / "blind_review.html"),
            },
        )
        staged_private.rename(private)
        staged_review.rename(review)


def rating_options() -> JsonDict:
    """提供页面与校验器共享的评分枚举。

    Results:
        JsonDict: 每个评分字段的允许状态。
    """
    return {
        "face": FACE,
        "iris": IRIS,
        "head_contour": HEAD,
        "face_clarity": CLARITY,
        "image_quality": QUALITY,
        "defects": {key: DEFECT for key in DEFECTS},
        "attribute": ATTRIBUTE,
    }


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> None:
    """核对完整匿名评分的身份、覆盖、状态及虹膜与眼色一致性。

    Args:
        deck: 公开冻结题卡。
        scores: 评审导出的六列评分。

    Raises:
        ValueError: 评分遗漏、适用性或枚举与冻结题卡不符。
    """
    if (
        scores.get("schema") != SCHEMA
        or scores.get("review_id") != deck["review_id"]
        or not str(scores.get("reviewer", "")).strip()
        or set(scores.get("items", {})) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Scores must cover this exact six-arm deck with a named reviewer")
    options = rating_options()
    for card in deck["items"]:
        rating = scores["items"][card["id"]]
        preferred = rating.get("preferred", [])
        if not preferred or len(set(preferred)) != len(preferred) or set(preferred) - set(COLUMNS):
            raise ValueError(f"Missing or invalid preferred columns: {card['id']}")
        if set(rating.get("columns", {})) != set(COLUMNS):
            raise ValueError(f"Missing columns: {card['id']}")
        for column in COLUMNS:
            point = rating["columns"][column]
            for field in ("face", "head_contour", "face_clarity", "image_quality"):
                if point.get(field) not in options[field]:
                    raise ValueError(f"Invalid {field}: {card['id']}/{column}")
            iris = (
                IRIS
                if card["iris_required"] is None
                else IRIS[:2] + ("failure",)
                if card["iris_required"]
                else ("not_applicable",)
            )
            if point.get("iris") not in iris:
                raise ValueError(f"Invalid iris applicability: {card['id']}/{column}")
            if set(point.get("defects", {})) != set(DEFECTS) or any(
                value not in DEFECT for value in point["defects"].values()
            ):
                raise ValueError(f"Missing structural defect scores: {card['id']}/{column}")
            if set(point.get("targets", {})) != set(card["targets"]) or any(
                value not in ATTRIBUTE for value in point["targets"].values()
            ):
                raise ValueError(f"Incomplete attribute scores: {card['id']}/{column}")
            eye = point["targets"].get("eye_color")
            if (
                eye is not None
                and point["iris"] in ("unreadable", "failure")
                and eye != point["iris"]
            ):
                raise ValueError(f"Eye color must match iris visibility: {card['id']}/{column}")


def lock(
    output: Path,
    score_path: Path,
) -> None:
    """验证全部匿名评分和 PNG 后冻结评分摘要，尚不读取身份。

    Args:
        output: 含匿名评分页面的目录。
        score_path: 已完成的评分 JSON。

    Raises:
        OSError/ValueError: 评分无效、图片变化或已有锁分。

    副作用:
        写入完整评分副本和哈希锁。
    """
    review = output / "review"
    deck = read_json(review / "deck.json")
    scores = read_json(score_path)
    validate_scores(deck, scores)
    if (review / "scores.lock.json").exists() or (review / "scores.json").exists():
        raise FileExistsError("Six-arm scores already frozen")
    for card in deck["items"]:
        if hash_file(review / "cards" / f"{card['id']}.png") != card["card_sha256"]:
            raise ValueError(f"Anonymous card changed: {card['id']}")
        for column in COLUMNS:
            if (
                hash_file(review / "cards" / f"{card['id']}_{column}.png")
                != card["image_sha256"][column]
            ):
                raise ValueError(f"Anonymous PNG changed: {card['id']}/{column}")
    atomic_json(review / "scores.json", scores)
    atomic_json(
        review / "scores.lock.json",
        {
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "scores_sha256": hash_file(review / "scores.json"),
            "deck_sha256": hash_file(review / "deck.json"),
            "review_html_sha256": hash_file(review / "blind_review.html"),
        },
    )


def locked_inputs(
    output: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """核对评分锁和私有映射承诺，供解盲及解盲报告共用。

    Args:
        output: 含公开评分和独立私有映射的工作区。

    Raises:
        OSError/ValueError: 锁、评分或映射发生变化。

    Results:
        tuple: 题卡、评分、身份映射及评分锁。
    """
    review = output / "review"
    frozen = read_json(review / "scores.lock.json")
    for name in ("scores", "deck", "review_html"):
        filename = "blind_review.html" if name == "review_html" else name + ".json"
        if frozen[name + "_sha256"] != hash_file(review / filename):
            raise ValueError(f"Frozen {name} changed")
    deck, scores = read_json(review / "deck.json"), read_json(review / "scores.json")
    validate_scores(deck, scores)
    private = read_json(output / "private/assignment.json")
    if (
        any(
            private[key] != frozen[key]
            for key in ("review_id", "deck_sha256", "review_html_sha256")
        )
        or assignment_digest(private["assignments"], private["salt"]) != deck["assignment_sha256"]
    ):
        raise ValueError("Private assignment does not match the frozen commitment")
    return deck, scores, private["assignments"], frozen


def summarize(
    records: list[JsonDict],
) -> JsonDict:
    """按明确分母汇总单组评分，保留难判与不适用数。

    Args:
        records: 同一模型及采样条件下的逐图评分记录。

    Results:
        JsonDict: 各指标原始计数、成功数及可判分母。
    """
    metrics = {}
    benefits = {
        "face": {"usable"},
        "iris": {"readable"},
        "head_contour": {"clear"},
        "face_clarity": {"clear"},
        "image_quality": {"good", "acceptable"},
        **{key: {"absent"} for key in DEFECTS},
    }
    for field, success in benefits.items():
        values = [
            row["score"][field] if field not in DEFECTS else row["score"]["defects"][field]
            for row in records
        ]
        counts = Counter(values)
        metrics[field] = {
            "total": len(values),
            "counts": dict(counts),
            "success": sum(counts[value] for value in success),
            "applicable": len(values) - counts["not_applicable"],
            "determinate": len(values) - counts["not_applicable"] - counts["uncertain"],
        }
    labels = sorted({label for row in records for label in row["score"]["targets"]})
    metrics["attributes"] = {
        label: dict(
            Counter(
                row["score"]["targets"][label]
                for row in records
                if label in row["score"]["targets"]
            )
        )
        for label in labels
    }
    metrics["preference"] = {
        "exclusive_best": sum(row["preferred"] and row["preferred_count"] == 1 for row in records),
        "tied_best": sum(row["preferred"] and row["preferred_count"] > 1 for row in records),
    }
    return metrics


def deblind(
    output: Path,
) -> None:
    """解盲已锁评分，按模型与八种采样条件输出独立统计。

    Args:
        output: 已完成匿名锁分的工作区。

    Raises:
        OSError/ValueError: 锁分或身份不符，或已有解盲结果。

    副作用:
        写入解盲逐图评分及模型条件汇总。
    """
    destination = output / "review/deblinded_metrics.json"
    if destination.exists():
        raise FileExistsError(destination)
    deck, scores, mapping, frozen = locked_inputs(output)
    rows = []
    groups: dict[str, list[JsonDict]] = defaultdict(list)
    for card in deck["items"]:
        identity, rated = mapping[card["id"]], scores["items"][card["id"]]
        condition = condition_name(card["guidance"], card["method"], card["steps"])
        for column, arm in identity["columns"].items():
            row = {
                "id": card["id"],
                "arm": arm,
                "column": column,
                "index": identity["index"],
                "seed": SEED,
                "condition": condition,
                "guidance": card["guidance"],
                "method": card["method"],
                "steps": card["steps"],
                "score": rated["columns"][column],
                "preferred": column in rated["preferred"],
                "preferred_count": len(rated["preferred"]),
                "note": rated.get("note", ""),
            }
            rows.append(row)
            groups[arm + "/" + condition].append(row)
    atomic_json(
        destination,
        {
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "score_lock": frozen,
            "summaries": {key: summarize(group) for key, group in groups.items()},
            "items": rows,
        },
    )


def report(
    output: Path,
) -> None:
    """将解盲结果整理成条件可筛选的完整图像与指标页面。

    Args:
        output: 已完成解盲的工作区。

    Raises:
        OSError/ValueError: 解盲身份、图像或评分锁不匹配。

    副作用:
        写带模型名称的比较图、指标 JSON 和浏览页面。
    """
    deck, _, mapping, frozen = locked_inputs(output)
    result = read_json(output / "review/deblinded_metrics.json")
    if result["score_lock"] != frozen:
        raise ValueError("Report requires the locked deblinded result")
    destination = output / "comparisons"
    destination.mkdir(exist_ok=True)
    font, heading = ImageFont.load_default(size=14), ImageFont.load_default(size=18)
    card_by_case = {
        (card["guidance"], card["method"], mapping[card["id"]]["index"]): card
        for card in deck["items"]
    }
    figures = []
    for guidance in GUIDANCE:
        for method, steps in SOLVERS:
            condition = condition_name(guidance, method, steps)
            for first in range(0, COUNT, 4):
                canvas = Image.new("RGB", (1608, 1280), "#f5f3ec")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (10, 8),
                    f"Seed {SEED} | {condition} | prompts {first + 1:02d}-{first + 4:02d}",
                    "#243544",
                    font=heading,
                )
                for col, arm in enumerate(ARMS):
                    draw.text((10 + 266 * col, 36), arm, "#243544", font=font)
                for row_number, index in enumerate(range(first, first + 4)):
                    card = card_by_case[guidance, method, index]
                    identity = mapping[card["id"]]
                    y = 62 + 302 * row_number
                    for col, arm in enumerate(ARMS):
                        column = next(
                            label
                            for label, assigned in identity["columns"].items()
                            if assigned == arm
                        )
                        path = output / "review/cards" / f"{card['id']}_{column}.png"
                        if hash_file(path) != card["image_sha256"][column]:
                            raise ValueError(f"Report PNG differs: {path}")
                        with Image.open(path) as image:
                            canvas.paste(image.convert("RGB"), (10 + 266 * col, y))
                    draw.multiline_text(
                        (10, y + 262),
                        "\n".join(wrap(f"{index + 1:02d}. {card['prompt']}", 180)),
                        "#243544",
                        font=font,
                    )
                path = destination / f"{condition}_{first + 1:02d}-{first + 4:02d}.png"
                canvas.save(path)
                figures.append(
                    {
                        "condition": condition,
                        "path": str(path.relative_to(output)),
                        "first_index": first,
                    }
                )
    table_rows = []
    for key, summary in sorted(result["summaries"].items()):
        cells = [html.escape(key)]
        for field in FACE_FIELDS:
            metric = summary[field]
            cells.append(f"{metric['success']}/{metric['applicable']}")
        for field in ("hair_color", "eye_color", "eye_state"):
            counts = summary["attributes"].get(field, {})
            cells.append(
                f"{counts.get('correct', 0)}/{sum(counts.values())}" if counts else "不适用"
            )
        for field in DEFECTS:
            metric = summary[field]
            cells.append(
                f"{metric['counts'].get('present', 0)} / {metric['determinate']}；"
                f"难判 {metric['counts'].get('uncertain', 0)}"
            )
        table_rows.append(
            '<tr data-condition="'
            + key.split("/", 1)[1]
            + '">'
            + "".join("<td>" + cell + "</td>" for cell in cells)
            + "</tr>"
        )
    figure_html = "".join(
        '<figure data-condition="'
        + row["condition"]
        + '"><img loading="lazy" src="'
        + row["path"]
        + '"></figure>'
        for row in figures
    )
    selectors = '<option value="all">全部采样设置</option>' + "".join(
        "<option>" + condition_name(g, m, s) + "</option>" for g in GUIDANCE for m, s in SOLVERS
    )
    page = (
        REPORT_HTML.replace("__ROWS__", "".join(table_rows))
        .replace("__FIGURES__", figure_html)
        .replace("__OPTIONS__", selectors)
    )
    (output / "comparison.html").write_text(page)
    atomic_json(output / "comparison.json", {"score_lock": frozen, "figures": figures})


def main() -> None:
    """运行采样、匿名准备、锁分、解盲或图像报告动作。

    Raises:
        SystemExit: 命令行缺少分支或评分文件。

    副作用:
        按显式动作加载模型或写入评测产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "render", "lock", "deblind", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    output = args.root / args.output
    if args.action == "generate":
        if args.arm is None or args.batch_size < 1:
            parser.error("generate needs --arm and a positive batch size")
        generate(args.root, output, args.arm, args.device, args.batch_size)
    elif args.action == "render":
        render(output)
    elif args.action == "lock":
        if args.scores is None:
            parser.error("lock needs --scores")
        lock(output, args.scores)
    elif args.action == "deblind":
        deblind(output)
    else:
        report(output)
    print(f"{args.action}: {output}", flush=True)


BLIND_HTML = r"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>动漫头像匿名评测</title>
<style>
body{
margin:0;
background:#f4f2eb;
color:#243544;
font:15px system-ui,sans-serif}
header{
position:sticky;
top:0;
background:#fff;
padding:12px 20px;
z-index:2;
border-bottom:1px solid #ccc}
button,select,input{
font:inherit;
padding:5px;
margin:3px}
main{
padding:18px}
summary{
cursor:pointer}
details{
background:#fff;
padding:10px;
margin-bottom:12px}
.cards{
display:grid;
grid-template-columns:repeat(6,minmax(225px,1fr));
gap:10px;
overflow:auto}
.column{
background:#fff;
padding:8px;
border-radius:6px}
.column img{
width:100%;
max-width:256px;
display:block;
margin:auto}
.field{
display:flex;
justify-content:space-between;
align-items:center}
.field select{
max-width:140px}
textarea{
width:100%;
min-height:50px;
box-sizing:border-box}
#status{
margin-left:12px}
h2{
font-size:18px}
#prompt{
min-height:30px}
</style>
<header>
<button id="prev">上一题</button>
<select id="position">
</select>
<button id="next">下一题</button>
<span id="status">
</span>
<br>
评审 <input id="reviewer" placeholder="填写姓名或评审标识">
<button id="tie">六列并列最佳</button>
<button id="export">导出评分</button>
<button id="import">导入评分</button>
<input id="file" type="file" accept="application/json" hidden>
</header>
<main>
<details>
<summary>评分规则</summary>
<div id="rubric">
</div>
</details>
<h2 id="title">
</h2>
<p id="prompt">
</p>
<div id="cards" class="cards">
</div>
<p>观察记录</p>
<textarea id="note">
</textarea>
</main>
<script>
const deck=__DATA__, fields={
face:'头像可用',iris:'虹膜可辨',head_contour:'头部轮廓',face_clarity:'面部清晰',image_quality:'整体画质'}
, defects={
eye_fragmentation:'眼周断裂',nose_mouth_missing:'鼻口缺失或移位',contour_fusion:'脸缘粘连'}
;
const labels={
usable:'可用',distorted:'可辨但变形',unrecognizable:'不可辨',failure:'生成失败',readable:'可辨',unreadable:'难辨',not_applicable:'不适用',clear:'清楚完整',broken:'断裂',uncertain:'难判',unclear:'不清楚',good:'好',acceptable:'可接受',poor:'差',present:'有',absent:'无',correct:'正确',wrong:'错误'}
;
const $=id=>document.getElementById(id), key='minigen-six:'+deck.review_id;
let scores=JSON.parse(localStorage.getItem(key)||'null')||{
schema:deck.score_schema,review_id:deck.review_id,reviewer:'',items:{
}
}
, index=0;
function persist(){
scores.reviewer=$('reviewer').value;
localStorage.setItem(key,JSON.stringify(scores));
$('status').textContent=Object.keys(scores.items).length+'/'+deck.items.length+' 题已记录';
}
function selectField(container,title,options,value,save){
const label=document.createElement('label');
label.className='field';
label.append(document.createTextNode(title));
const select=document.createElement('select');
const empty=document.createElement('option');
empty.value='';
empty.textContent='请选择';
select.append(empty);
for(const option of options){
const item=document.createElement('option');
item.value=option;
item.textContent=labels[option]||option;
select.append(item)}
select.value=value||'';
select.onchange=()=>{
save(select.value);
persist()}
;
label.append(select);
container.append(label)}
function draw(){
const card=deck.items[index], item=scores.items[card.id]||{
preferred:[],columns:{
}
,note:''}
;
$('position').value=index;
$('title').textContent=(index+1)+' / '+deck.items.length+' · '+card.id
  +' · CFG '+card.guidance+' · '+card.method+' '+card.steps;
$('prompt').textContent=card.prompt;
$('cards').replaceChildren();
for(const column of deck.columns){
const rating=item.columns[column]||{
defects:{
}
,targets:{
}
}
;
const box=document.createElement('section');
box.className='column';
const heading=document.createElement('label'),check=document.createElement('input');
check.type='checkbox';
check.checked=item.preferred.includes(column);
check.onchange=()=>{
item.preferred=check.checked?[...new Set([...item.preferred,column])]
  :item.preferred.filter(c=>c!==column);
scores.items[card.id]=item;
persist()}
;
heading.append(check,document.createTextNode(column+' 最适合做头像'));
box.append(heading);
const picture=document.createElement('img');
picture.src=deck.images[card.id][column];
box.append(picture);
function store(){
item.columns[column]=rating;
scores.items[card.id]=item}
for(const field of Object.keys(fields)){
let options=deck.rating_options[field];
if(field==='iris'&&card.iris_required===false)options=['not_applicable'];
else if(field==='iris'&&card.iris_required===true)options=options.filter(v=>v!=='not_applicable');
selectField(box,fields[field],options,rating[field],value=>{
rating[field]=value;
store()}
)}
for(const field of Object.keys(defects)){
selectField(box,defects[field],deck.rating_options.defects[field],rating.defects[field],value=>{
rating.defects[field]=value;
store()}
)}
for(const [field,target]of Object.entries(card.targets)){
selectField(box,target,deck.rating_options.attribute,rating.targets[field],value=>{
rating.targets[field]=value;
store()}
)}
$('cards').append(box)}
$('note').value=item.note||'';
$('note').oninput=()=>{
item.note=$('note').value;
scores.items[card.id]=item;
persist()}
;
persist()}
for(let i=0;
i<deck.items.length;
i++){
const option=document.createElement('option');
option.value=i;
option.textContent=(i+1)+' · '+deck.items[i].id;
$('position').append(option)}
for(const [name,text]of Object.entries(deck.rubric)){
const p=document.createElement('p');
p.textContent=text;
$('rubric').append(p)}
$('reviewer').value=scores.reviewer||'';
$('reviewer').oninput=persist;
$('position').onchange=()=>{
index=Number($('position').value);
draw()}
;
$('prev').onclick=()=>{
index=Math.max(0,index-1);
draw()}
;
$('next').onclick=()=>{
index=Math.min(deck.items.length-1,index+1);
draw()}
;
$('tie').onclick=()=>{
const card=deck.items[index];
scores.items[card.id]=scores.items[card.id]||{
columns:{
}
,note:''}
;
scores.items[card.id].preferred=[...deck.columns];
draw()}
;
$('export').onclick=()=>{
persist();
const blob=new Blob([JSON.stringify(scores,null,2)],{
type:'application/json'}
),a=document.createElement('a');
a.href=URL.createObjectURL(blob);
a.download='blind_scores_'+deck.review_id+'.json';
a.click();
URL.revokeObjectURL(a.href)}
;
$('import').onclick=()=>$('file').click();
$('file').onchange=async()=>{
try{
const loaded=JSON.parse(await $('file').files[0].text());
if(loaded.review_id!==deck.review_id||loaded.schema!==deck.score_schema)
  throw Error('评分不属于当前题库');
scores=loaded;
$('reviewer').value=scores.reviewer||'';
draw()}
catch(error){
alert(error.message)}
finally{
$('file').value=''}
}
;
draw();
</script>
</html>"""

REPORT_HTML = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>六组 30k 动漫头像对照</title>
<style>body{
margin:24px;
background:#f4f2eb;
color:#243544;
font:15px system-ui,sans-serif}
table{
border-collapse:collapse;
background:#fff}
th,td{
padding:8px;
border:1px solid #ccc}
section{
overflow:auto}
figure{
margin:22px 0}
img{
max-width:100%;
height:auto}
select{
font:inherit;
padding:8px}
</style>
<h1>六组 30k 动漫头像对照</h1>
<p>相同 16 条开发标签与初噪 seed 5001。每项保留 16 张图。CFG 0 不评价文本遵循。
评分先锁定再解盲，每张图的主观分数依赖对应评审口径。</p>
<section>
<table>
<thead>
<tr>
<th>模型 / 采样设置</th>
<th>头像可用</th>
<th>虹膜可辨</th>
<th>轮廓清楚</th>
<th>面部清楚</th>
<th>整体可接受</th>
<th>发色正确</th>
<th>眼色正确</th>
<th>开闭眼正确</th>
<th>眼周断裂 / 可判</th>
<th>鼻口缺失 / 可判</th>
<th>脸缘粘连 / 可判</th>
</tr>
</thead>
<tbody>__ROWS__</tbody>
</table>
</section>
<p>
<select id="filter">__OPTIONS__</select>
</p>__FIGURES__<script>
document.getElementById('filter').onchange=function(){
  for(const fig of document.querySelectorAll('[data-condition]'))
    fig.hidden=this.value!=='all'&&fig.dataset.condition!==this.value;
};
</script>
</html>"""

if __name__ == "__main__":
    main()
