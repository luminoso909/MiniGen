"""采样 B2 五官加权配对开发图，匿名锁分后按种子统计五官缺陷。"""

import argparse
import base64
import hashlib
import json
import secrets
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import wrap

import torch
from evaluate_anime_time_ablation import COUNT, sample_variant, tensor_digest
from PIL import Image, ImageDraw, ImageFont
from render_anime_benchmark_blind import HTML, image_data, order_key, read_json
from render_anime_repa_blind import (
    CONFIG,
    SAMPLING,
    SEEDS,
    SIDES,
    read_ema,
    scoring_endpoint,
    write_json,
)
from run_anime_b1 import ROOT, open_anime_cache
from score_anime_scale_benchmark import FACE

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.training.runtime import resolve_device

ARMS = ("control", "weighted")
STEP = 5000
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_face_weights_v1/dev")
RUNS = Path("artifacts/runs/anime_faces_256/b2_face_weights_v1")
SCORE_SCHEMA = "minigen.anime_faces.face_weights_blind_scores.v1"
DEFECTS = {
    "eye_fragmentation": "眼周断裂",
    "nose_mouth_missing": "鼻口缺失",
    "contour_fusion": "脸缘粘连",
}
DEFECT_STATES = ("present", "absent", "uncertain", "not_applicable")
IRIS_STATES = ("readable", "unreadable", "failure", "not_applicable")
PREFERENCES = (*SIDES, "tie", "not_comparable")


@torch.inference_mode()
def generate(
    root: Path,
    output: Path,
    checkpoints: Path,
) -> None:
    """用原有 16 条开发标签和两套固定初噪采样两组最终 EMA。

    Args:
        root: AutoDL 项目目录。
        output: 评测目录，完整采样后发布 generation。
        checkpoints: 含 control 与 weighted 训练分支的目录。

    Raises:
        OSError: 输出已存在，或输入不可读。
        ValueError/RuntimeError: 冻结输入、检查点、训练身份或 CUDA 执行不符。

    副作用:
        设置 FP32 CUDA 采样配置并写入生成图及回执。
    """
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "generation"
    if destination.exists():
        raise FileExistsError(destination)
    device = resolve_device("cuda", "fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, data = read_experiment(root, CONFIG)
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, identity = open_anime_cache(root, data)
    indices = selected_indices(source, data, "validation")[:COUNT]
    if len(indices) != COUNT:
        raise ValueError(f"Expected {COUNT} fixed development labels")
    validation = cache.split("validation")
    entries = [validation[index] for index in indices]
    condition = Condition(
        torch.stack([entry.text for entry in entries]).float(),
        torch.stack([entry.mask for entry in entries]),
    ).to(device)
    blank = blank.to(device)
    records = []
    for position, index in enumerate(indices):
        row = source.records["validation"][index]
        records.append(
            {
                "index": position,
                "sample_id": row["sample_id"],
                "condition_sample_id": row["sample_id"],
                "caption": row["caption"],
            }
        )
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    side = config.model["latent_size"]
    noise_by_seed = {
        seed: torch.randn(
            (COUNT, config.model["in_channels"], side, side),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        )
        for seed in SEEDS
    }
    with TemporaryDirectory(dir=output) as pending:
        generation = Path(pending) / "generation"
        for arm in ARMS:
            path = checkpoints / arm / f"step_{STEP:06d}.pt"
            model, receipt = read_ema(path, arm, config.model, identity, step=STEP)
            if ("face_weights" in receipt["initialization"]) != (arm == "weighted"):
                raise ValueError(f"Incorrect face weighting identity in {path}")
            model = model.to(device)
            rows = []
            for seed, noise in noise_by_seed.items():
                sample_dir = generation / arm / str(seed) / "correct"
                group = [
                    record | {"noise_sha256": tensor_digest(noise[index : index + 1])}
                    for index, record in enumerate(records)
                ]
                sample_variant(
                    sample_dir,
                    model,
                    vae,
                    noise.to(device),
                    condition,
                    blank,
                    group,
                    receipt,
                    seed=seed,
                )
                for row in read_json(sample_dir / "samples.json")["records"]:
                    image = sample_dir / row["image_path"]
                    rows.append(
                        row
                        | {
                            "seed": seed,
                            "image_sha256": hash_file(image),
                            "image_path": str(image.relative_to(generation / arm)),
                        }
                    )
            del model
            write_json(
                generation / arm / "receipt.json",
                receipt
                | {
                    "arm": arm,
                    "data_identity": identity,
                    "sampling": SAMPLING,
                    "status": "completed",
                    "records": rows,
                },
            )
        generation.rename(destination)


def assignment_digest(
    mapping: JsonDict,
    salt: str,
) -> str:
    """计算左右映射的加盐摘要，供公开题卡承诺分组身份。

    Args:
        mapping: 匿名题号到真实臂、初噪和标签位置的映射。
        salt: 私有十六进制随机盐。

    Raises:
        TypeError/ValueError: 映射或盐无法序列化。

    Results:
        str: 公开可验证的 SHA-256 摘要。
    """
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(bytes.fromhex(salt) + encoded).hexdigest()


def render(
    output: Path,
) -> None:
    """验证配对采样输入和像素摘要，生成 32 道匿名题卡及独立映射。

    Args:
        output: 生成回执和评测文件所在目录。

    Raises:
        OSError: 输入不可读或盲评已存在。
        ValueError: 标签、初噪、步数、模型身份或图片摘要不符。

    副作用:
        写入匿名 HTML、单独 PNG、公开题卡元数据及私有映射。
    """
    generation = output / "generation"
    first, second = [read_json(generation / arm / "receipt.json") for arm in ARMS]
    fields = ("seed", "index", "sample_id", "condition_sample_id", "caption", "noise_sha256")
    keys = [(row["seed"], row["index"]) for row in first["records"]]
    expected_keys = [(seed, index) for seed in SEEDS for index in range(COUNT)]
    if (
        keys != expected_keys
        or [tuple(row[key] for key in fields) for row in first["records"]]
        != [tuple(row[key] for key in fields) for row in second["records"]]
        or [first["arm"], second["arm"]] != list(ARMS)
        or first["checkpoint_sha256"] == second["checkpoint_sha256"]
        or first["data_identity"] != second["data_identity"]
        or any(
            item["sampling"] != SAMPLING
            or item["checkpoint_step"] != STEP
            or item["status"] != "completed"
            for item in (first, second)
        )
    ):
        raise ValueError("Paired generation receipts do not match the fixed development protocol")
    review, private = output / "review", output / "private"
    if review.exists() or private.exists():
        raise FileExistsError("An anonymous review already exists")
    secret = secrets.token_bytes(32)
    review_id = order_key(secret, f"b2-region:{first['run_id']}:{second['run_id']}")[:24]
    cards, mapping = [], {}
    receipts = dict(zip(ARMS, (first, second), strict=True))
    with TemporaryDirectory(dir=output) as pending:
        staged_review = Path(pending) / "review"
        staged_private = Path(pending) / "private"
        pictures = staged_review / "cards"
        pictures.mkdir(parents=True)
        staged_private.mkdir(mode=0o700)
        for seed in SEEDS:
            positions = sorted(
                (position for position, key in enumerate(keys) if key[0] == seed),
                key=lambda position: order_key(secret, f"side:{keys[position]}"),
            )
            for ordinal, position in enumerate(positions):
                _, index = keys[position]
                opaque = order_key(secret, f"card:{seed}:{index}")[:16]
                layout = ARMS if ordinal % 2 == 0 else ARMS[::-1]
                endpoint = scoring_endpoint("single", first["records"][position]["caption"])
                # 保留完整提示词与虹膜适用性，评分专注于结构和虹膜。
                endpoint["targets"] = {}
                card = {
                    "id": opaque,
                    "kind": "regular",
                    "category": "五官结构",
                    "endpoints": [endpoint],
                    "images": {},
                    "image_sha256": {},
                }
                for side, arm in zip(SIDES, layout, strict=True):
                    row = receipts[arm]["records"][position]
                    uri = image_data(generation / arm / row["image_path"], row["image_sha256"], 256)
                    card["images"][side] = {"single": uri}
                    picture = pictures / f"{opaque}_{side}_single.png"
                    picture.write_bytes(base64.b64decode(uri.partition(",")[2]))
                    card["image_sha256"][side] = hash_file(picture)
                cards.append(card)
                mapping[opaque] = {"seed": seed, "development_index": index} | dict(
                    zip(SIDES, layout, strict=True)
                )
        cards.sort(key=lambda card: order_key(secret, f"position:{card['id']}"))
        salt = secrets.token_hex(32)
        metadata = {
            "review_id": review_id,
            "score_schema": SCORE_SCHEMA,
            "defects": DEFECTS,
            "assignment_sha256": assignment_digest(mapping, salt),
        }
        embed = json.dumps(metadata | {"items": cards}, ensure_ascii=False).replace("<", "\\u003c")
        (staged_review / "blind_review.html").write_text(
            HTML.replace("__ANONYMOUS_DATA__", embed), encoding="utf-8"
        )
        write_json(
            staged_review / "deck.json",
            metadata
            | {
                "items": [
                    {key: value for key, value in card.items() if key != "images"} for card in cards
                ]
            },
        )
        write_json(
            staged_private / "assignment.json",
            {
                "review_id": review_id,
                "deck_sha256": hash_file(staged_review / "deck.json"),
                "review_html_sha256": hash_file(staged_review / "blind_review.html"),
                "salt": salt,
                "assignments": mapping,
            },
        )
        staged_private.rename(private)
        staged_review.rename(review)


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> None:
    """严格检查完整评分和各缺陷字段，锁分无需知道训练臂。

    Args:
        deck: 冻结匿名题卡元数据。
        scores: 评审导出的匿名逐图评分。

    Raises:
        ValueError: 题卡身份、评分覆盖、适用性或状态无效。
    """
    if (
        scores.get("schema") != SCORE_SCHEMA
        or deck["score_schema"] != SCORE_SCHEMA
        or deck["defects"] != DEFECTS
        or scores.get("review_id") != deck["review_id"]
        or not str(scores.get("reviewer", "")).strip()
        or set(scores.get("items", {})) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Scores must cover this exact anonymous deck with a named reviewer")
    for card in deck["items"]:
        rating = scores["items"][card["id"]]
        if rating.get("preference") not in PREFERENCES:
            raise ValueError(f"Missing preference for {card['id']}")
        iris_states = (
            IRIS_STATES[:-1] if card["endpoints"][0]["iris_required"] else ("not_applicable",)
        )
        for side in SIDES:
            point = rating.get(side, {}).get("endpoints", {}).get("single", {})
            defects = point.get("defects", {})
            if (
                point.get("face") not in FACE
                or point.get("iris") not in iris_states
                or set(defects) != set(DEFECTS)
                or any(state not in DEFECT_STATES for state in defects.values())
            ):
                raise ValueError(f"Invalid or incomplete scores for {card['id']}/{side}")


def lock(
    output: Path,
    score_path: Path,
) -> None:
    """先验证并冻结匿名评分，同时锁定题卡和图像。

    Args:
        output: 含公开匿名题卡的评测目录。
        score_path: 评审导出的完整匿名评分路径。

    Raises:
        OSError: 输入不可读或已有锁分。
        ValueError: 评分或匿名 PNG 与题卡不符。

    副作用:
        写入冻结评分与哈希回执。
    """
    review = output / "review"
    deck_path = review / "deck.json"
    deck, scores = read_json(deck_path), read_json(score_path)
    validate_scores(deck, scores)
    for card in deck["items"]:
        for side in SIDES:
            name = f"{card['id']}_{side}_single.png"
            if hash_file(review / "cards" / name) != card["image_sha256"][side]:
                raise ValueError(f"Anonymous image differs from the rendered deck: {name}")
    frozen = review / "scores.json"
    write_json(frozen, scores)
    write_json(
        review / "scores.lock.json",
        {
            "scores_sha256": hash_file(frozen),
            "deck_sha256": hash_file(deck_path),
            "review_html_sha256": hash_file(review / "blind_review.html"),
            "review_id": scores["review_id"],
            "reviewer": scores["reviewer"],
        },
    )


def summarize(
    pairs: list[JsonDict],
) -> JsonDict:
    """在每个指标上计数单图状态并统计可比较配对改善。

    Args:
        pairs: 对应题卡的 control、weighted 两臂逐图评分。

    Results:
        JsonDict: 各臂原始计数、分母及配对净增，正值代表 weighted 改善。
    """
    counts = {}
    transitions = {}
    for metric in ("face", "iris", *DEFECTS):
        benefit = {"face": "usable", "iris": "readable"}.get(metric, "absent")
        results = {
            arm: [
                pair[arm][metric] if metric in ("face", "iris") else pair[arm]["defects"][metric]
                for pair in pairs
            ]
            for arm in ARMS
        }
        states = {"face": FACE, "iris": IRIS_STATES}.get(metric, DEFECT_STATES)
        excluded = {"not_applicable", "uncertain"}
        counts[metric] = {}
        for arm in ARMS:
            count = Counter(results[arm])
            determinate = len(pairs) - sum(count[state] for state in excluded)
            numerator = count["present"] if metric in DEFECTS else count[benefit]
            counts[metric][arm] = {
                "total": len(pairs),
                "applicable": len(pairs) - count["not_applicable"],
                "determinate": determinate,
                "numerator": numerator,
                "counts": {state: count[state] for state in states},
            }
        gain, loss, tie, undecided = 0, 0, 0, 0
        for old, new in zip(*results.values(), strict=True):
            if {old, new} & excluded:
                undecided += 1
            elif (old == benefit) == (new == benefit):
                tie += 1
            elif new == benefit:
                gain += 1
            else:
                loss += 1
        transitions[metric] = {
            "comparable": gain + loss + tie,
            "gain": gain,
            "loss": loss,
            "tie": tie,
            "excluded": undecided,
            "net_gain": gain - loss,
        }
    return {"counts": counts, "paired_changes": transitions}


def read_assignment(
    output: Path,
    lock_receipt: JsonDict,
    deck: JsonDict,
) -> JsonDict:
    """验证私有解盲映射符合锁定题卡中的分组承诺。

    Args:
        output: 含私有分组映射的评测目录。
        lock_receipt: 公开锁分回执。
        deck: 已锁定的匿名题卡。

    Raises:
        OSError/ValueError: 私有映射不可读或不符合题卡。

    Results:
        JsonDict: 匿名题号到真实配对身份的映射。
    """
    key = read_json(output / "private/assignment.json")
    mapping = key["assignments"]
    if (
        any(
            key[field] != lock_receipt[field]
            for field in ("review_id", "deck_sha256", "review_html_sha256")
        )
        or set(mapping) != {card["id"] for card in deck["items"]}
        or assignment_digest(mapping, key["salt"]) != deck["assignment_sha256"]
    ):
        raise ValueError("Private mapping does not match the locked deck commitment")
    return mapping


def deblind(
    output: Path,
) -> None:
    """读取已经锁定的评分，再按各初噪及合计汇总。

    Args:
        output: 同时含锁分回执及私有分组映射的评测目录。

    Raises:
        OSError: 输入不可读或解盲产物已存在。
        ValueError: 评分或映射与锁分摘要不符。

    副作用:
        写入解盲指标和逐题评分与模型的对应关系。
    """
    review = output / "review"
    frozen = read_json(review / "scores.lock.json")
    for digest, filename in (
        ("scores_sha256", "scores.json"),
        ("deck_sha256", "deck.json"),
        ("review_html_sha256", "blind_review.html"),
    ):
        if frozen[digest] != hash_file(review / filename):
            raise ValueError(f"Locked digest does not match {filename}")
    deck, scores = read_json(review / "deck.json"), read_json(review / "scores.json")
    validate_scores(deck, scores)
    mapping = read_assignment(output, frozen, deck)
    pairs = []
    for card in deck["items"]:
        opaque = card["id"]
        assignment, points = mapping[opaque], scores["items"][opaque]
        pairs.append(
            {
                "id": opaque,
                "seed": assignment["seed"],
                "development_index": assignment["development_index"],
                "preference": assignment.get(points["preference"], points["preference"]),
                "note": points.get("note", ""),
            }
            | {assignment[side]: points[side]["endpoints"]["single"] for side in SIDES}
        )
    summaries = {}
    for seed in ("all", *SEEDS):
        group = [pair for pair in pairs if seed == "all" or pair["seed"] == seed]
        summaries[str(seed)] = summarize(group) | {
            "preference": Counter(pair["preference"] for pair in group)
        }
    write_json(
        review / "deblinded_metrics.json",
        {
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "score_lock": frozen,
            "summaries": summaries,
            "items": pairs,
        },
    )


def compare(
    output: Path,
) -> None:
    """按种子和原标签顺序生成每四题一张的解盲配对图。

    Args:
        output: 含已解盲结果、匿名图片及左右分组身份的评测目录。

    Raises:
        OSError: 输入不可读或对照图目录已存在。
        ValueError: 评分锁、逐题身份或图片摘要不匹配。

    副作用:
        写入带臂名称的对照 PNG。匿名 HTML 与题卡文件保持原样。
    """
    review = output / "review"
    frozen = read_json(review / "scores.lock.json")
    result = read_json(review / "deblinded_metrics.json")
    deck_path = review / "deck.json"
    if result["score_lock"] != frozen or hash_file(deck_path) != frozen["deck_sha256"]:
        raise ValueError("Comparison requires the deblinded result matching the locked scores")
    metadata = read_json(deck_path)
    deck = {row["id"]: row for row in metadata["items"]}
    mapping = read_assignment(output, frozen, metadata)
    pairs = {(row["seed"], row["development_index"]): row for row in result["items"]}
    if len(pairs) != len(SEEDS) * COUNT:
        raise ValueError("Deblinded comparison requires the complete fixed development set")
    destination = output / "comparisons"
    if destination.exists():
        raise FileExistsError(destination)
    font = ImageFont.load_default(size=15)
    heading = ImageFont.load_default(size=19)
    with TemporaryDirectory(dir=output) as pending:
        staged = Path(pending) / "comparisons"
        staged.mkdir()
        for seed in SEEDS:
            for first in range(0, COUNT, 4):
                last = first + 4
                canvas = Image.new("RGB", (1080, 702), "#f5f4ef")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (12, 11),
                    f"Seed {seed}  |  Captions {first + 1:02d}-{last:02d}",
                    "#22313b",
                    font=heading,
                )
                for slot, index in enumerate(range(first, last)):
                    row = pairs[seed, index]
                    opaque = row["id"]
                    card, layout = deck[opaque], mapping[opaque]
                    if (layout["seed"], layout["development_index"]) != (seed, index):
                        raise ValueError(f"Comparison mapping differs for {opaque}")
                    x, y = 12 + 536 * (slot % 2), 43 + 327 * (slot // 2)
                    for column, (arm, label) in enumerate(
                        zip(ARMS, ("Original Flow", "Face-weighted Flow"), strict=True)
                    ):
                        side = next(side for side in SIDES if layout[side] == arm)
                        path = review / "cards" / f"{opaque}_{side}_single.png"
                        if hash_file(path) != card["image_sha256"][side]:
                            raise ValueError(
                                f"Anonymous image differs from the rendered deck: {path}"
                            )
                        draw.text((x + 264 * column, y), label, "#22313b", font=font)
                        with Image.open(path) as sample:
                            canvas.paste(sample.convert("RGB"), (x + 264 * column, y + 23))
                    prompt = card["endpoints"][0]["prompt"]
                    draw.multiline_text(
                        (x, y + 284),
                        "\n".join(wrap(f"{index + 1:02d}. {prompt}", width=70)),
                        "#22313b",
                        font=font,
                        spacing=2,
                    )
                canvas.save(staged / f"seed_{seed}_{first + 1:02d}-{last:02d}.png")
        staged.rename(destination)


def main() -> None:
    """运行配对采样、匿名渲染、锁分、解盲或出对照图。

    副作用:
        按动作生成、验证和写入评测产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "render", "lock", "deblind", "compare"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=RUNS)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    output = args.root / args.output
    if args.action == "generate":
        generate(args.root, output, args.root / args.checkpoint_root)
    elif args.action == "render":
        render(output)
    elif args.action == "lock":
        if args.scores is None:
            parser.error("lock needs --scores")
        lock(output, args.scores)
    elif args.action == "deblind":
        deblind(output)
    else:
        compare(output)
    print(f"{args.action}: {output}")


if __name__ == "__main__":
    main()
