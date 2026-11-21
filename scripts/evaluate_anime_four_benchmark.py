"""为四组 50M 30k 固定题库建立匿名评分、锁分、解盲统计与对照报告。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import secrets
import shutil
import textwrap
from collections import Counter
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw
from render_anime_benchmark_blind import (
    HTML,
    RECEIPT_FIELDS,
    endpoint,
    image_data,
    order_key,
    read_json,
)
from score_anime_scale_benchmark import (
    ATTRIBUTE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FACE,
    IRIS,
    PRESERVATION,
    SUCCESS,
    ModelScores,
    compare,
    summarize,
    swap_outcome,
)

from minigen._typing import JsonDict

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/exports/anime-faces-b1/four_50m_30k_benchmark")
SCOPE = Path("configs/benchmark/anime_faces_v1_four_50m_30k.json")
BENCHMARK = Path("configs/benchmark/anime_faces_v1.json")
ARMS = ("b1_50m_p2", "b2_50m_p2", "b1_50m_p1", "b2_50m_p1")
COLUMNS = ("alpha", "beta", "gamma", "delta")
COLUMN_PAIRS = tuple(itertools.combinations(COLUMNS, 2))
ARM_PAIRS = tuple(itertools.combinations(ARMS, 2))
ABLATION_COLUMNS = (*COLUMNS, "epsilon")
CFG_COLUMNS = (*ABLATION_COLUMNS, "zeta")
SEEDS = (2026111401, 2026111402)
SCORE_SCHEMA = "minigen.anime_four.blind_scores.v1"
KEY_SCHEMA = "minigen.anime_four.blind_mapping.v1"
DEFECTS = {
    "eye_fragmentation": "眼周断裂、重复或错位",
    "nose_mouth_missing": "鼻口缺失、熔化或移位",
    "contour_fusion": "脸缘不合理粘连",
}
DEFECT_STATES = ("present", "absent", "uncertain", "not_applicable")
RANKING = ("face", "iris", "eye_color", "swap_joint")
METRIC_LABELS = {
    "face": "头像可用",
    "iris": "虹膜可辨",
    "eye_color": "眼色正确",
    "swap_switch": "换词两端正确",
    "swap_preservation": "非目标保持",
    "swap_joint": "换词联合成功",
}


def digest(
    path: Path,
) -> str:
    """计算文件原始字节的摘要。

    Args:
        path: 待读取文件。

    Results:
        str: SHA-256 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(
    path: Path,
    data: JsonDict,
) -> None:
    """写入新 JSON，避免覆盖正式评分和映射。

    Args:
        path: 输出文件。
        data: 需要持久化的对象。

    Raises:
        FileExistsError: 目标已经存在。

    副作用:
        创建父目录和 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_inputs(
    root: Path,
) -> tuple[JsonDict, JsonDict, str]:
    """核对本次四组范围与保持原样的题单。

    Args:
        root: 项目根目录。

    Raises:
        ValueError: 四组范围、题库摘要、种子或采样设置改变。

    Results:
        tuple: 四组范围、题库和范围摘要。
    """
    scope, benchmark = read_json(root / SCOPE), read_json(root / BENCHMARK)
    if (
        scope["schema"] != "minigen.anime_faces.four_50m_30k_scope.v1"
        or set(scope["candidates"]) != set(ARMS)
        or scope["benchmark_sha256"] != digest(root / BENCHMARK)
        or scope["rubric_sha256"] != digest(root / scope["rubric"])
        or benchmark["seeds"] != list(SEEDS)
        or scope["seeds"] != list(SEEDS)
        or scope["sampling"] != benchmark["sampling"]
        or len(benchmark["regular"]) != 80
        or len(benchmark["pairs"]) != 24
    ):
        raise ValueError("Four-model scope differs from the frozen benchmark")
    return scope, benchmark, digest(root / SCOPE)


def verified_receipts(
    root: Path,
    output: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, str]:
    """核对四份正式生成回执和共同的采样、权重及数据身份。

    Args:
        root: 配置所在项目根目录。
        output: 本次生成根目录。

    Raises:
        ValueError: 某组不属于固定的 30k EMA 比较，或端点不完整。

    Results:
        tuple: 四组范围、题库、生成回执和范围摘要。
    """
    scope, benchmark, scope_hash = read_inputs(root)
    receipts = {arm: read_json(output / "generation" / arm / "receipt.json") for arm in ARMS}
    first = receipts[ARMS[0]]
    shared = (
        *RECEIPT_FIELDS,
        "source_sha256",
        "data_identity",
        "evaluation_torch",
        "sampler_sha256",
        "sampler_helper_sha256",
        "condition_features_sha256",
        "condition_mask_sha256",
        "condition_encoding",
        "sampling_batch_size",
        "decode_batch_size",
        "tf32",
    )
    for arm, receipt in receipts.items():
        spec = scope["candidates"][arm]
        if (
            receipt["schema"] != "minigen.anime_faces.benchmark_generation.v1"
            or receipt["status"] != "completed"
            or receipt["arm"] != arm
            or receipt["checkpoint_step"] != 30000
            or receipt["cumulative_training_updates"] != 30000
            or receipt["weights"] != "ema"
            or receipt["checkpoint_sha256"] != spec["checkpoint_sha256"]
            or receipt["run_id"] != spec["run_id"]
            or receipt["trainable_parameters"] != spec["parameters"]
            or receipt["model_config"] != spec["model_config"]
            or receipt["source_sha256"] != spec["source_sha256"]
            or receipt["comparison_scope_sha256"] != scope_hash
            or receipt["benchmark_sha256"] != scope["benchmark_sha256"]
            or receipt["rubric_sha256"] != scope["rubric_sha256"]
            or receipt["seeds"] != list(SEEDS)
            or receipt["sampling"] != benchmark["sampling"]
            or any(receipt[field] != first[field] for field in shared)
            or len(receipt["records"]) != 256
        ):
            raise ValueError(f"Generation is outside the four-model protocol: {arm}")
    if len({row["checkpoint_sha256"] for row in receipts.values()}) != 4:
        raise ValueError("Four distinct checkpoint identities are required")
    return scope, benchmark, receipts, scope_hash


def card_image(
    card: JsonDict,
    path: Path,
    columns: tuple[str, ...] = COLUMNS,
) -> None:
    """把匿名模型列和换词两端绘成便于独立视觉评分的图片。

    Args:
        card: 含匿名图片数据 URI 的公开题卡。
        path: 组合卡 PNG 输出位置。
        columns: 从左到右的匿名列名。

    副作用:
        创建不含原模型名称或文件元数据的 PNG。
    """
    rows = card["endpoints"]
    canvas = Image.new("RGB", (6 + 262 * len(columns), 56 + len(rows) * 322), "#f7f5ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 8), card["id"], fill="#1c3038")
    for index, column in enumerate(columns):
        draw.text((9 + index * 262, 31), column, fill="#1c3038")
        for row_index, point in enumerate(rows):
            raw = base64.b64decode(card["images"][column][point["key"]].split(",", 1)[1])
            with Image.open(BytesIO(raw)) as picture:
                canvas.paste(picture.convert("RGB"), (4 + index * 262, 52 + row_index * 322))
            caption = f"{point['key']}: {point['prompt']}"
            for line_index, line in enumerate(textwrap.wrap(caption, width=43)[:5]):
                draw.text(
                    (7 + index * 262, 310 + row_index * 322 + line_index * 11), line, fill="#1c3038"
                )
    canvas.save(path)


def render(
    root: Path,
    output: Path,
    *,
    prepared: tuple[JsonDict, JsonDict, JsonDict, str] | None = None,
) -> None:
    """从真实生成文件建立新匿名页、公开题卡、模板及私有映射。

    Args:
        root: 项目根目录。
        output: 本次生成和评分目录。
        prepared: 扩展评测入口已核验的范围、题库、回执和范围摘要。

    Raises:
        FileExistsError: 评分目录已经存在。
        ValueError: 端点、哈希或跨模型初噪不一致。

    副作用:
        创建公开评分包、组合卡、私有映射及导出摘要。
    """
    _, benchmark, receipts, scope_hash = prepared or verified_receipts(root, output)
    arms = tuple(receipts)
    columns = CFG_COLUMNS[: len(arms)]
    column_pairs = tuple(itertools.combinations(columns, 2))
    review = output / "review"
    if any(
        (review / name).exists()
        for name in ("export_manifest.json", "scores.json", "scores.lock.json")
    ):
        raise FileExistsError(f"Published or scored review must be preserved: {review}")
    # export_manifest 是最后写入的发布标记；只重建尚未发布、未评分的中断产物。
    if review.exists():
        shutil.rmtree(review)
    key_path = output / "private/key.json"
    key_path.unlink(missing_ok=True)
    review.mkdir()
    (review / "cards").mkdir()
    records = {
        arm: {(row["id"], row["seed"], row["endpoint"]): row for row in receipt["records"]}
        for arm, receipt in receipts.items()
    }
    secret = secrets.token_bytes(32)
    rotations = [arms[index:] + arms[:index] for index in range(len(arms))]
    layouts = rotations + [layout[::-1] for layout in rotations]
    cards, assignments, expected = [], {}, set()
    questions = [("regular", row) for row in benchmark["regular"]]
    questions += [("pair", row) for row in benchmark["pairs"]]
    questions.sort(key=lambda pair: order_key(secret, f"layout:{pair[1]['id']}"))
    for index, (kind, question) in enumerate(questions):
        if columns == CFG_COLUMNS:
            # 允许指导配对在六列中的任意位置，同一题两个初噪共用排列。
            layout = sorted(arms, key=lambda arm: order_key(secret, f"arm:{question['id']}:{arm}"))
        else:
            layout = layouts[index % len(layouts)]
        placement = dict(zip(columns, layout, strict=True))
        points = (
            [endpoint("single", question["prompt"], question["targets"])]
            if kind == "regular"
            else [
                endpoint(label, question[f"prompt_{label}"], question[f"targets_{label}"])
                for label in ("a", "b")
            ]
        )
        for seed in SEEDS:
            card = {
                "id": order_key(secret, f"card:{question['id']}:{seed}")[:16],
                "kind": kind,
                "category": question["category"],
                "endpoints": points,
                "images": {column: {} for column in columns},
            }
            if kind == "pair":
                card["change"] = question["changed"]
            for point in points:
                key = (question["id"], seed, None if point["key"] == "single" else point["key"])
                expected.add(key)
                noise = set()
                for column, arm in placement.items():
                    row = records[arm][key]
                    suffix = "" if point["key"] == "single" else f"_{point['key']}"
                    filename = f"{question['id']}_{seed}{suffix}.png"
                    if row["prompt"] != point["prompt"] or row["image_path"] != filename:
                        raise ValueError(f"Incorrect prompt or endpoint file: {arm}/{key}")
                    noise.add(row["noise_sha256"])
                    card["images"][column][point["key"]] = image_data(
                        output / "generation" / arm / filename, row["image_sha256"], 256
                    )
                # 同题的换词两端也必须复用相同初噪。
                if len(noise) != 1:
                    raise ValueError(f"Models use different initial noise: {key}")
            first_arm = records[arms[0]]
            if (
                kind == "pair"
                and first_arm[question["id"], seed, "a"]["noise_sha256"]
                != (first_arm[question["id"], seed, "b"]["noise_sha256"])
            ):
                raise ValueError(f"Intervention endpoints use different noise: {question['id']}")
            cards.append(card)
            assignments[card["id"]] = {
                "question_id": question["id"],
                "seed": seed,
                "kind": kind,
                "category": question["category"],
                "group_id": question["source"]["group_id"],
                "changed_attribute": question.get("changed", {}).get("attribute"),
                **placement,
            }
    if len(cards) != 208 or any(set(rows) != expected for rows in records.values()):
        raise ValueError("Expected 208 cards and all planned image endpoints for each model")
    cards.sort(key=lambda card: order_key(secret, f"position:{card['id']}"))
    review_id = order_key(secret, f"review:{scope_hash}")[:24]
    key = {
        "schema": KEY_SCHEMA,
        "review_id": review_id,
        "arms": list(arms),
        "columns": list(columns),
        "benchmark_sha256": digest(root / BENCHMARK),
        "comparison_scope_sha256": scope_hash,
        "sampling": benchmark["sampling"],
        "checkpoint_identities": {
            arm: {
                field: receipt[field]
                for field in (
                    "checkpoint_sha256",
                    "checkpoint_step",
                    "run_id",
                    "trainable_parameters",
                    "model_config",
                )
            }
            for arm, receipt in receipts.items()
        },
        "assignments": assignments,
    }
    if columns == CFG_COLUMNS:
        key["sampling_by_arm"] = {arm: receipt["sampling"] for arm, receipt in receipts.items()}
    write_new_json(key_path, key)
    deck = {
        "review_id": review_id,
        "columns": list(columns),
        "score_schema": SCORE_SCHEMA,
        "mapping_sha256": digest(key_path),
        "comparison_scope_sha256": scope_hash,
        "defects": DEFECTS,
        "items": cards,
    }
    public = deck | {"items": [{k: v for k, v in card.items() if k != "images"} for card in cards]}
    write_new_json(review / "deck.json", public)
    template = {"schema": SCORE_SCHEMA, "review_id": review_id, "reviewer": "", "items": {}}
    for card in cards:
        point_scores = {
            point["key"]: {
                "face": "",
                "iris": "" if point["iris_required"] else "not_applicable",
                "targets": {target: "" for target in point["targets"]},
                "defects": {name: "" for name in DEFECTS},
            }
            for point in card["endpoints"]
        }
        template["items"][card["id"]] = {
            column: {
                "endpoints": point_scores,
                **({"preservation": ""} if card["kind"] == "pair" else {}),
            }
            for column in columns
        }
        template["items"][card["id"]]["preferences"] = {
            f"{one}-{two}": "" for one, two in column_pairs
        }
        card_image(card, review / "cards" / f"{card['id']}.png", columns)
    write_new_json(review / "score_template.json", template)
    document = HTML.replace('gamma:"丙"', 'gamma:"丙",delta:"丁",epsilon:"戊",zeta:"己"').replace(
        "grid-template-columns:repeat(3,minmax(0,1fr));",
        f"grid-template-columns:repeat({len(columns)},minmax(0,1fr));",
    )
    document = document.replace(
        "__ANONYMOUS_DATA__", json.dumps(deck, ensure_ascii=False).replace("</", r"<\/")
    )
    (review / "blind_review.html").write_text(document)
    files = [review / name for name in ("blind_review.html", "deck.json", "score_template.json")]
    files += sorted((review / "cards").glob("*.png")) + [key_path]
    write_new_json(
        review / "export_manifest.json",
        {
            "schema": "minigen.anime_four.export_manifest.v1",
            "review_id": review_id,
            "image_count": 256 * len(arms),
            "card_count": 208,
            "files": {str(path.relative_to(output)): digest(path) for path in files},
        },
    )
    print(f"Rendered 208 anonymous cards with {len(arms)} models: {review}")


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> JsonDict:
    """按公开题卡验证匿名评分的固定分母和字段。

    Args:
        deck: 公开题卡元数据。
        scores: 待锁定的完整匿名评分。

    Raises:
        ValueError: 评分身份、字段、状态或适用性不一致。

    Results:
        JsonDict: 题数、图数及虹膜、眼色、干预对适用数量。
    """
    columns = tuple(deck["columns"])
    if (
        scores.get("schema") != SCORE_SCHEMA
        or not scores.get("reviewer", "").strip()
        or scores.get("review_id") != deck["review_id"]
        or set(scores.get("items", {})) != {card["id"] for card in deck["items"]}
        or columns not in (COLUMNS, ABLATION_COLUMNS, CFG_COLUMNS)
        or len(deck["items"]) != 208
    ):
        raise ValueError("Scores must name a reviewer and cover this exact anonymous deck")
    pairs = {f"{one}-{two}": (one, two) for one, two in itertools.combinations(columns, 2)}
    counts = dict(cards=0, images=0, iris=0, eye_color=0, swaps=0)
    for card in deck["items"]:
        rating = scores["items"][card["id"]]
        if (
            set(rating) - {"note"} != {*columns, "preferences"}
            or rating["preferences"].keys() != pairs.keys()
        ):
            raise ValueError(f"Incomplete columns or preferences: {card['id']}")
        for label, pair in pairs.items():
            if rating["preferences"][label] not in (*pair, "tie", "not_comparable"):
                raise ValueError(f"Invalid preference: {card['id']}")
        for column in columns:
            rated = rating[column]
            expected_fields = (
                {"endpoints", "preservation"} if card["kind"] == "pair" else {"endpoints"}
            )
            if set(rated) != expected_fields or set(rated["endpoints"]) != {
                point["key"] for point in card["endpoints"]
            }:
                raise ValueError(f"Incomplete endpoints: {card['id']}/{column}")
            if card["kind"] == "pair" and rated["preservation"] not in PRESERVATION:
                raise ValueError(f"Invalid preservation: {card['id']}/{column}")
            for point in card["endpoints"]:
                score = rated["endpoints"][point["key"]]
                allowed_iris = IRIS if point["iris_required"] else ("not_applicable",)
                if (
                    set(score) != {"face", "iris", "targets", "defects"}
                    or score["face"] not in FACE
                    or score["iris"] not in allowed_iris
                    or (point["iris_required"] and score["iris"] == "not_applicable")
                    or set(score["targets"]) != set(point["targets"])
                    or any(value not in ATTRIBUTE for value in score["targets"].values())
                    or set(score["defects"]) != set(DEFECTS)
                    or any(value not in DEFECT_STATES for value in score["defects"].values())
                ):
                    raise ValueError(
                        f"Invalid endpoint scores: {card['id']}/{column}/{point['key']}"
                    )
                if "eye_color" in point["targets"] and score["iris"] in {"unreadable", "failure"}:
                    if score["targets"]["eye_color"] != score["iris"]:
                        raise ValueError(
                            f"Eye color requires a readable iris: {card['id']}/{column}"
                        )
                counts["images"] += 1
                counts["iris"] += point["iris_required"]
                counts["eye_color"] += "eye_color" in point["targets"]
            counts["swaps"] += card["kind"] == "pair"
        counts["cards"] += 1
    count = len(columns)
    if counts != dict(
        cards=208, images=256 * count, iris=152 * count, eye_color=128 * count, swaps=48 * count
    ):
        raise ValueError(f"Unexpected blind review counts: {counts}")
    return counts


def fingerprints(
    output: Path,
    scores_path: Path,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """检查匿名页、公开元数据及评分，并返回尚不涉及身份的摘要。

    Args:
        output: 本次匿名评分根目录。
        scores_path: 已完成的匿名评分文件。

    Raises:
        ValueError: 网页内公开题卡与元数据或评分不一致。

    Results:
        tuple: 公开题卡、匿名评分、锁分摘要字段。
    """
    deck_path, html_path = output / "review/deck.json", output / "review/blind_review.html"
    deck, scores = read_json(deck_path), read_json(scores_path)
    document = html_path.read_text()
    marker = '<script id="deck" type="application/json">'
    embedded = json.loads(document.split(marker, 1)[1].split("</script>", 1)[0])
    for card in embedded["items"]:
        card.pop("images")
    if embedded != deck:
        raise ValueError("The HTML deck differs from public deck metadata")
    return (
        deck,
        scores,
        {
            "schema": "minigen.anime_four.blind_scores_lock.v1",
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "scores_sha256": digest(scores_path),
            "deck_sha256": digest(deck_path),
            "html_sha256": digest(html_path),
            "mapping_sha256": deck["mapping_sha256"],
            "comparison_scope_sha256": deck["comparison_scope_sha256"],
            "counts": validate_scores(deck, scores),
        },
    )


def freeze(
    output: Path,
    scores_path: Path,
) -> None:
    """在不读取私有映射的情况下发布不可覆盖的评分锁。

    Args:
        output: 本次评测输出目录。
        scores_path: 完整匿名评分路径。

    副作用:
        写入 scores.json 评分副本及 scores.lock.json。
    """
    _, _, lock = fingerprints(output, scores_path)
    frozen = output / "review/scores.json"
    if scores_path.resolve() != frozen.resolve():
        with frozen.open("xb") as stream:
            stream.write(scores_path.read_bytes())
    write_new_json(
        output / "review/scores.lock.json", lock | {"locked_at": datetime.now(UTC).isoformat()}
    )
    lock_path = output / "review/scores.lock.json"
    print(f"Locked 208 cards / {lock['counts']['images']} images: {lock_path}")


def locked_inputs(
    root: Path,
    output: Path,
    *,
    scope_path: Path = SCOPE,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """先重验评分锁，随后读取并逐题校验已承诺摘要的私有映射。

    Args:
        root: 冻结题库和范围所在项目根目录。
        output: 含评分锁和私有映射的评测目录。
        scope_path: 固定实验范围配置路径。

    Raises:
        ValueError: 锁后输入改变、映射摘要不符或题目身份不一致。

    Results:
        tuple: 公开题卡、锁定分数、身份映射和评分锁。
    """
    deck, scores, current = fingerprints(output, output / "review/scores.json")
    lock = read_json(output / "review/scores.lock.json")
    if scope_path == SCOPE:
        _, benchmark, scope_hash = read_inputs(root)
        arms = ARMS
    else:
        scope = read_json(root / scope_path)
        benchmark = read_json(root / BENCHMARK)
        scope_hash = digest(root / scope_path)
        if scope["benchmark_sha256"] != digest(root / BENCHMARK):
            raise ValueError("Benchmark differs from the registered scope")
        arms = tuple(scope["candidates"])
    columns = tuple(deck["columns"])
    if not lock.get("locked_at") or any(lock.get(name) != value for name, value in current.items()):
        raise ValueError("Scores or deck changed after lock")
    if (
        scope_hash != lock["comparison_scope_sha256"]
        or digest(output / "private/key.json") != lock["mapping_sha256"]
    ):
        raise ValueError("Private mapping or scope changed after commitment")
    key = read_json(output / "private/key.json")
    if (
        key["schema"] != KEY_SCHEMA
        or key["review_id"] != deck["review_id"]
        or key["arms"] != list(arms)
        or key["columns"] != list(columns)
        or key["benchmark_sha256"] != digest(root / BENCHMARK)
        or set(key["assignments"]) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Mapping does not cover this exact locked review")
    questions = {row["id"]: (kind, row) for kind in ("regular", "pairs") for row in benchmark[kind]}
    seen, layouts = set(), {}
    for card in deck["items"]:
        row = key["assignments"][card["id"]]
        kind, question = questions[row["question_id"]]
        kind = "pair" if kind == "pairs" else kind
        layout = tuple(row[column] for column in columns)
        expected_points = (
            [endpoint("single", question["prompt"], question["targets"])]
            if kind == "regular"
            else [
                endpoint(label, question[f"prompt_{label}"], question[f"targets_{label}"])
                for label in ("a", "b")
            ]
        )
        if (
            row["seed"] not in SEEDS
            or (row["question_id"], row["seed"]) in seen
            or set(layout) != set(arms)
            or layouts.setdefault(row["question_id"], layout) != layout
            or row["group_id"] != question["source"]["group_id"]
            or row["kind"] != kind
            or row["category"] != question["category"]
            or row["changed_attribute"] != question.get("changed", {}).get("attribute")
            or card["kind"] != kind
            or card["category"] != question["category"]
            or card["endpoints"] != expected_points
            or card.get("change") != question.get("changed")
        ):
            raise ValueError(f"Private assignment does not match the question: {card['id']}")
        seen.add((row["question_id"], row["seed"]))
    if seen != set(itertools.product(questions, SEEDS)):
        raise ValueError("Mapping must cover both noise seeds for every question")
    return deck, scores, key, lock


def resolve_scores(
    scores: JsonDict,
    key: JsonDict,
) -> tuple[dict[str, ModelScores], JsonDict, JsonDict]:
    """把锁后评分映射至配对观测，并记录缺陷和头像偏好。

    Args:
        scores: 完整锁定匿名评分。
        key: 已核对身份的私有映射。

    Results:
        tuple: 原六指标配对观测、逐缺陷状态和偏好计数。
    """
    arms, columns = tuple(key["arms"]), tuple(key["columns"])
    arm_pairs = tuple(itertools.combinations(arms, 2))
    column_pairs = tuple(itertools.combinations(columns, 2))
    observations = {arm: {metric: {} for metric in SUCCESS} for arm in arms}
    defects = {
        arm: {name: {str(seed): Counter() for seed in SEEDS} for name in DEFECTS} for arm in arms
    }
    preferences = {
        f"{one}__{two}": {
            "first_arm": one,
            "second_arm": two,
            "overall": Counter(),
            "by_seed": {str(seed): Counter() for seed in SEEDS},
        }
        for one, two in arm_pairs
    }
    for card_id, card in scores["items"].items():
        row = key["assignments"][card_id]
        prefix = row["group_id"], row["question_id"], row["seed"]
        for column in columns:
            arm, rated = row[column], card[column]
            model = observations[arm]
            for label, score in rated["endpoints"].items():
                point = (*prefix, label)
                model["face"][point], model["iris"][point] = score["face"], score["iris"]
                model["eye_color"][point] = score["targets"].get("eye_color", "not_applicable")
                for name, value in score["defects"].items():
                    defects[arm][name][str(row["seed"])][value] += 1
            if row["kind"] == "pair":
                point, changed = (*prefix, "swap"), row["changed_attribute"]
                first = rated["endpoints"]["a"]["targets"][changed]
                second = rated["endpoints"]["b"]["targets"][changed]
                model["swap_switch"][point] = swap_outcome(first, second, "preserved")
                model["swap_preservation"][point] = rated["preservation"]
                model["swap_joint"][point] = swap_outcome(first, second, rated["preservation"])
        for one, two in column_pairs:
            pair = next(pair for pair in arm_pairs if set(pair) == {row[one], row[two]})
            bucket = preferences["__".join(pair)]
            picked = card["preferences"][f"{one}-{two}"]
            result = (
                ("first_wins" if row[picked] == pair[0] else "second_wins")
                if picked in columns
                else picked
            )
            bucket["overall"][result] += 1
            bucket["by_seed"][str(row["seed"])][result] += 1
    for bucket in preferences.values():
        bucket["overall"] = {
            state: bucket["overall"][state]
            for state in ("first_wins", "second_wins", "tie", "not_comparable")
        }
        bucket["by_seed"] = {
            seed: {
                state: counts[state]
                for state in ("first_wins", "second_wins", "tie", "not_comparable")
            }
            for seed, counts in bucket["by_seed"].items()
        }
    return observations, defects, preferences


def reveal(
    root: Path,
    output: Path,
    *,
    scope_path: Path = SCOPE,
) -> None:
    """核验评分锁后计算各组成绩和预登记的配对差异。

    Args:
        root: 冻结输入所在项目根目录。
        output: 本次评测目录。
        scope_path: 固定实验范围配置路径。

    副作用:
        创建解盲统计及排名文件，不修改原评分。
    """
    _, scores, key, lock = locked_inputs(root, output, scope_path=scope_path)
    observations, defects, preferences = resolve_scores(scores, key)
    arms = tuple(key["arms"])
    pairs = ARM_PAIRS if scope_path == SCOPE else read_json(root / scope_path)["baseline_pairs"]
    groups = sorted({point[0] for point in observations[arms[0]]["face"]})
    rng = Random(BOOTSTRAP_SEED)
    draws = [
        tuple(rng.choices(range(len(groups)), k=len(groups))) for _ in range(BOOTSTRAP_REPLICATES)
    ]
    models = {
        arm: {
            "overall": {name: summarize(points, name) for name, points in model.items()},
            "by_seed": {
                str(seed): {name: summarize(points, name, seed) for name, points in model.items()}
                for seed in SEEDS
            },
            "defects": defects[arm],
        }
        for arm, model in observations.items()
    }
    comparisons = {
        f"{one}__{two}": {
            "baseline": one,
            "candidate": two,
            "metrics": {
                metric: compare(
                    observations[one][metric], observations[two][metric], metric, groups, draws
                )
                for metric in SUCCESS
            },
        }
        for one, two in pairs
    }
    ranking = {"ranking": rank_models(models)} if scope_path == SCOPE else {}
    write_new_json(
        output / "review/deblinded_metrics.json",
        {
            "schema": "minigen.anime_four.blind_metrics.v1",
            "locked_review": lock,
            "sampling": key["sampling"],
            **({"sampling_by_arm": key["sampling_by_arm"]} if "sampling_by_arm" in key else {}),
            "checkpoint_identities": key["checkpoint_identities"],
            "models": models,
            "comparisons": comparisons,
            "pairwise_preferences": preferences,
            **ranking,
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "groups": len(groups),
                "unit": "source group including both seeds and endpoints",
            },
            "scope": (
                "fixed reused 208-question regression benchmark; "
                "one training seed; AI visual review"
            ),
        },
    )
    print(f"Revealed 208 locked cards: {output / 'review/deblinded_metrics.json'}")


def rank_models(
    models: JsonDict,
) -> JsonDict:
    """按原四组评测的优先级选出前两名并保留截止并列。

    Args:
        models: 解盲后的四组逐项统计。

    Results:
        JsonDict: 排名依据、顺序及含并列的候选。
    """
    ordered = sorted(
        ARMS,
        key=lambda arm: tuple(-models[arm]["overall"][metric]["success"] for metric in RANKING),
    )
    vectors = {
        arm: tuple(models[arm]["overall"][metric]["success"] for metric in RANKING) for arm in ARMS
    }
    ranking = [
        {
            "arm": arm,
            "rank": 1 + sum(vectors[other] > vectors[arm] for other in ARMS),
            "counts": list(vectors[arm]),
        }
        for arm in ordered
    ]
    cutoff = ranking[1]["rank"]
    top_two = [row["arm"] for row in ranking if row["rank"] <= cutoff]
    return {
        "priority": list(RANKING),
        "ordered": ranking,
        "top_two_with_cutoff_ties": top_two,
        "meaning": "relative candidates, not an absolute usability certificate",
    }


def report(
    root: Path,
    output: Path,
    *,
    scope_path: Path = SCOPE,
) -> None:
    """生成解盲后的模型表、分种子指标及全部题目的对照页。

    Args:
        root: 冻结输入所在项目根目录。
        output: 含已解盲统计的本次评测目录。
        scope_path: 固定实验范围配置路径。

    Raises:
        ValueError: 统计不属于当前锁定评分。

    副作用:
        写入 HTML 总览及带模型标签的逐题对照 PNG。
    """
    deck, scores, key, lock = locked_inputs(root, output, scope_path=scope_path)
    arms, columns = tuple(key["arms"]), tuple(key["columns"])
    data = read_json(output / "review/deblinded_metrics.json")
    if data["locked_review"] != lock:
        raise ValueError("Metrics are not from the current locked review")
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    rows, seed_rows, model_rows, questions = [], [], [], []
    for arm in arms:
        model = data["models"][arm]
        cells = [
            f"{model['overall'][metric]['success']}/{model['overall'][metric]['applicable']}"
            for metric in SUCCESS
        ]
        rows.append(
            "<tr><th>" + arm + "</th>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
        for seed in SEEDS:
            values = model["by_seed"][str(seed)]
            seed_rows.append(
                f"<tr><th>{arm}</th><td>{seed}</td>"
                + "".join(
                    f"<td>{values[metric]['success']}/{values[metric]['applicable']}</td>"
                    for metric in SUCCESS
                )
                + "</tr>"
            )
        identity = key["checkpoint_identities"][arm]
        model_rows.append(
            f"<tr><th>{arm}</th><td>{identity['trainable_parameters']:,}</td>"
            f"<td><code>{escape(json.dumps(identity['model_config']))}</code></td>"
            f"<td><code>{identity['checkpoint_sha256']}</code></td></tr>"
        )
    for index, card in enumerate(deck["items"], start=1):
        row = key["assignments"][card["id"]]
        folder = output / "generation"
        canvas = Image.new(
            "RGB", (6 + 262 * len(arms), 50 + 278 * len(card["endpoints"])), "#f7f5ef"
        )
        draw = ImageDraw.Draw(canvas)
        for column_index, arm in enumerate(arms):
            draw.text((7 + 262 * column_index, 8), arm, fill="#1c3038")
            for point_index, point in enumerate(card["endpoints"]):
                suffix = "" if point["key"] == "single" else f"_{point['key']}"
                filename = f"{row['question_id']}_{row['seed']}{suffix}.png"
                with Image.open(folder / arm / filename) as picture:
                    canvas.paste(
                        picture.convert("RGB"), (4 + 262 * column_index, 30 + 278 * point_index)
                    )
                draw.text(
                    (7 + 262 * column_index, 288 + 278 * point_index), point["key"], fill="#1c3038"
                )
        name = f"{row['question_id']}_{row['seed']}.png"
        canvas.save(comparisons / name)
        notes = "；".join(f"{point['key']}: {point['prompt']}" for point in card["endpoints"])
        scores_for_card = scores["items"][card["id"]]
        score_table = {
            arm: scores_for_card[column]
            for column, arm in ((column, row[column]) for column in columns)
        }
        questions.append(
            {
                "title": f"{index}. {row['question_id']} · {row['seed']} · {card['category']}",
                "kind": card["kind"],
                "prompt": notes,
                "image": f"comparisons/{name}",
                "scores": score_table,
            }
        )
    heads = "".join(f"<th>{METRIC_LABELS[name]}</th>" for name in SUCCESS)
    failures = {
        arm: [
            f"{row['question_id']}:{row['seed']}:{point}"
            for card_id, row in key["assignments"].items()
            for column in columns
            if row[column] == arm
            for point, score in scores["items"][card_id][column]["endpoints"].items()
            if score["face"] != "usable"
        ]
        for arm in arms
    }
    comparisons_json = escape(json.dumps(data["comparisons"], ensure_ascii=False, indent=2))
    defects_json = escape(
        json.dumps(
            {arm: data["models"][arm]["defects"] for arm in arms}, ensure_ascii=False, indent=2
        )
    )
    failures_json = escape(json.dumps(failures, ensure_ascii=False, indent=2))
    questions_json = json.dumps(questions, ensure_ascii=False).replace("</", r"<\/")
    if scope_path == SCOPE:
        title = "四组 50M 30k · 固定 Benchmark 复测"
        top = ", ".join(data["ranking"]["top_two_with_cutoff_ties"])
        verdict = (
            f"<h2>相对前两名候选</h2><p><strong>{escape(top)}</strong>。"
            "按头像可用、虹膜、眼色、换词联合成功依次排序，边界并列保留。</p>"
        )
    else:
        scope = read_json(root / scope_path)
        title = escape(scope.get("report_title", "Mid60 与 REPA · B1 / B2 patch1 30k"))
        baseline_label = "同权重 CFG 3" if columns == CFG_COLUMNS else "同架构 Uniform"
        paired_rows = []
        for pair in data["comparisons"].values():
            for metric in ("face", "iris"):
                result = pair["metrics"][metric]
                low, high = result["bootstrap_95_percent_pp"]
                seed_cells = "".join(
                    f"<td>{result['by_seed'][str(seed)]['net']:+d}</td>" for seed in SEEDS
                )
                paired_rows.append(
                    f"<tr><th>{pair['candidate']}</th><td>{pair['baseline']}</td>"
                    f"<td>{METRIC_LABELS[metric]}</td><td>{result['net']:+d}/{result['applicable']}</td>"
                    f"{seed_cells}<td>{low:+.2f} 至 {high:+.2f}</td>"
                    f"<td>{'通过' if result['passes'] else '未通过'}</td></tr>"
                )
        verdict = (
            "<h2>配对提升门槛</h2><p>头像可用净增至少 13/256，虹膜可辨至少 8/152。"
            "各项在两个初噪均需正向。</p><div class='scroll'><table>"
            f"<tr><th>分支</th><th>{baseline_label}</th><th>指标</th><th>总净增</th>"
            f"<th>{SEEDS[0]}</th><th>{SEEDS[1]}</th><th>95% 区间（百分点）</th><th>门槛</th></tr>"
            + "".join(paired_rows)
            + "</table></div>"
        )
    cfg_label = "CFG 3 / 5" if columns == CFG_COLUMNS else "CFG3"
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{font:16px/1.6 system-ui;background:#f5f3ed;color:#17323c;
margin:24px auto;max-width:1450px;padding:0 18px}}
table{{border-collapse:collapse;width:100%;background:white}}
td,th{{border:1px solid #c5d2d4;padding:8px;text-align:left}}
code{{overflow-wrap:anywhere}}img{{width:100%;height:auto}}
button,select{{font:inherit;padding:5px;margin:6px}}
.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>{title}</h1>
<p>30k EMA · {cfg_label} · Euler40 · 两个初噪 · 208 题 · {256 * len(arms):,} 图。
复用原题库，以单个训练种子做回归比较。AI 视觉评分锁定后解盲。</p>
{verdict}
<div class="scroll"><table><tr><th>模型</th>{heads}</tr>{"".join(rows)}</table></div>
<h2>两个初噪分别统计</h2><div class="scroll"><table>
<tr><th>模型</th><th>初噪</th>{heads}</tr>{"".join(seed_rows)}</table></div>
<h2>参数与架构</h2><div class="scroll"><table>
<tr><th>模型</th><th>参数</th><th>模型配置</th><th>检查点SHA256</th></tr>
{"".join(model_rows)}</table></div>
<h2>配对差值与分种子结果</h2><p>正值代表分支优于对照。
按来源组保留同题两个种子和端点进行配对 bootstrap。</p>
<details><summary>查看完整统计</summary><pre>{comparisons_json}</pre></details>
<h2>五官缺陷</h2><details><summary>各模型、种子的有缺陷／无缺陷／难判／不适用计数</summary>
<pre>{defects_json}</pre></details>
<h2>失败样本</h2><details><summary>头像不可用的全部题号与端点</summary>
<pre>{failures_json}</pre></details>
<h2>全部208道对照</h2><label>题型 <select id="kind"><option value="">全部</option>
<option value="regular">常规</option><option value="pair">换词</option></select></label>
<button id="prev">上一题</button><span id="position"></span><button id="next">下一题</button>
<section id="question"></section><p><a href="review/deblinded_metrics.json">统计 JSON</a> ·
<a href="review/scores.lock.json">评分锁</a> · <a href="review/blind_review.html">原匿名页</a></p>
<script id="questions" type="application/json">{questions_json}</script>
<script>const all=JSON.parse(document.getElementById('questions').textContent);let n=0;
const byId=id=>document.getElementById(id);function show(){{
const rows=all.filter(q=>!byId('kind').value||q.kind===byId('kind').value);
n=Math.max(0,Math.min(n,rows.length-1));const q=rows[n],box=byId('question');box.replaceChildren();
const title=document.createElement('h3');title.textContent=q.title;
const p=document.createElement('p');p.textContent=q.prompt;
const img=document.createElement('img');img.src=q.image;img.alt=q.title;
const detail=document.createElement('details');const summary=document.createElement('summary');
summary.textContent='逐模型评分';const pre=document.createElement('pre');
pre.textContent=JSON.stringify(q.scores,null,2);
detail.append(summary,pre);box.append(title,p,img,detail);byId('position').textContent=`${{n+1}}/${{rows.length}}`;
byId('prev').disabled=n===0;byId('next').disabled=n===rows.length-1;}}
byId('prev').onclick=()=>{{n--;show()}};byId('next').onclick=()=>{{n++;show()}};byId('kind').onchange=()=>{{n=0;show()}};show();</script></html>"""
    (output / "index.html").write_text(document)
    print(f"Wrote comparison report: {output / 'index.html'}")


def main() -> None:
    """解析四组匿名渲染、锁分、解盲和报告入口。

    副作用:
        按所选阶段写评测产物；不会训练模型或调用关机。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "freeze", "reveal", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if args.command == "render":
        render(root, output)
    elif args.command == "freeze":
        if args.scores is None:
            parser.error("freeze requires --scores")
        freeze(output, root / args.scores)
    elif args.command == "reveal":
        reveal(root, output)
    else:
        report(root, output)


if __name__ == "__main__":
    main()
