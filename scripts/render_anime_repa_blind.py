"""在 AutoDL 采样 B2 双臂开发图，生成匿名题卡，并锁分解盲。"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from evaluate_anime_time_ablation import (
    COUNT,
    GUIDANCE,
    STEPS,
    eye_records,
    sample_variant,
    tensor_digest,
)
from render_anime_benchmark_blind import HTML, endpoint, image_data, order_key, read_json
from run_anime_b1 import ROOT, open_anime_cache
from score_anime_scale_benchmark import ATTRIBUTE, FACE, PRESERVATION

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, FLOW_OBJECTIVE, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device

ARMS = ("control", "repa")
MODES = ("correct", "eye_control")
SEEDS = (1234, 2345)
STEP = 2000
SIDES = ("left", "right")
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_repa_v1/dev")
RUNS = Path("artifacts/runs/anime_faces_256/b2_repa_v1")
CONFIG = Path("configs/training/anime_b2_50m_20000.json")
PLAN = Path("configs/evaluation/anime_b1_time_eyes.json")
PARENT_SHA = "289db105636596e42731e8be7d013fdb384a66d0b572489e0151dd7e53baff12"
SCORE_SCHEMA = "minigen.anime_faces.blind_scores.v1"
SAMPLING = {"precision": "fp32", "method": "euler", "steps": STEPS, "guidance": GUIDANCE}


def write_json(
    path: Path,
    data: JsonDict,
) -> None:
    """独占写入一份 JSON 回执并拒绝覆盖已有结果。

    Args:
        path: 目标文件路径。
        data: 可编码成 JSON 的对象。

    Raises:
        OSError: 目标存在或不可写。

    副作用:
        写入指定文件。
    """
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_ema(
    path: Path,
    arm: str,
    model_config: JsonDict,
    data_identity: JsonDict,
    *,
    step: int = STEP,
) -> tuple[torch.nn.Module, JsonDict]:
    """只加载本轮已完成的 B2 学生 EMA，不加载优化器和训练专用投影头。

    Args:
        path: 当前分支的编号检查点。
        arm: 消融臂名称，repa 的 EMA 包含训练专用投影头。
        model_config: 本轮共用的 B2 50M 网络规格。
        data_identity: 共同的验证缓存及数据规格。
        step: 本轮应完成的优化器更新次数。

    Raises:
        OSError: 检查点或完成回执不可读。
        ValueError/RuntimeError: 运行身份、初始检查点、步数或权重规格不吻合。

    Results:
        tuple: CPU 上的 B2 EMA 网络及检查点身份。
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = state["contract"]
    result_path = path.with_name("result.json")
    result = read_json(result_path)
    run_id = run_identity(contract)
    init = contract["initialization"]
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["objective"] != FLOW_OBJECTIVE
        or state["step"] != step
        or state["scheduler"]["step"] != step
        or state["config"]["model"] != model_config
        or state["data_identity"] != data_identity
        or init["checkpoint_sha256"] != PARENT_SHA
        or init["checkpoint_step"] != 20_000
        or init["weights"] != "ema"
        or result["status"] != "completed"
        or result["step"] != step
        or result["run_id"] != run_id
        or result["performance"]["optimizer_steps"] != step
    ):
        raise ValueError(f"Checkpoint does not match the completed B2 ablation: {path}")
    weights = state["ema"]
    projector_keys = {key for key in weights if key.startswith("repa_projector.")}
    if bool(projector_keys) != (arm == "repa"):
        raise ValueError(f"Checkpoint has unexpected projector state: {path}")
    student_weights = {key: value for key, value in weights.items() if key not in projector_keys}
    with torch.device("meta"):
        model = build_model(model_config)
    expected = {name: (value.shape, value.dtype) for name, value in model.state_dict().items()}
    if {key: (value.shape, value.dtype) for key, value in student_weights.items()} != expected:
        raise ValueError(f"Checkpoint student EMA weights differ from B2: {path}")
    model.load_state_dict(student_weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, {
        "run_id": run_id,
        "checkpoint_sha256": hash_file(path),
        "checkpoint_step": step,
        "initialization": init,
        "training_result_sha256": hash_file(result_path),
    }


@torch.inference_mode()
def generate(
    root: Path,
    output: Path,
    checkpoints: Path,
) -> None:
    """用冻结开发标签、同批初噪和相同采样协议逐臂解码。

    Args:
        root: AutoDL 上的 MiniGen 项目根目录。
        output: 开发评测目录，写入新的 generation 子目录。
        checkpoints: 同时包含 control 和 repa 的训练运行目录。

    Raises:
        OSError: 生成目录已存在，或缓存、权重、结果不可读写。
        ValueError/RuntimeError: CUDA、权重、固定开发标签或采样校验失败。

    副作用:
        设置推理精度与线程，写入 PNG、记录及每臂正式生成回执。
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
        raise ValueError(f"Expected {COUNT} fixed development rows")
    validation = cache.split("validation")
    entries = [validation[index] for index in indices]
    labels = []
    for position, index in enumerate(indices):
        row = source.records["validation"][index]
        labels.append(
            {
                "index": position,
                "sample_id": row["sample_id"],
                "condition_sample_id": row["sample_id"],
                "caption": row["caption"],
            }
        )
    eye_plan = root / PLAN
    eye_labels = eye_records(read_json(eye_plan)["rows"], labels)
    clip = FrozenCLIPText.from_lock(root / data["clip_lock"], root=root, device="cpu")
    encoded = clip.encode([row["caption"] for row in eye_labels])
    del clip
    conditions = {
        "correct": Condition(
            torch.stack([row.text for row in entries]).float(),
            torch.stack([row.mask for row in entries]),
        ).to(device),
        "eye_control": Condition(encoded.features.half().float(), encoded.mask).to(device),
    }
    blank = blank.to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    size = config.model["latent_size"]
    noise_by_seed = {
        seed: torch.randn(
            (COUNT, config.model["in_channels"], size, size),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        )
        for seed in SEEDS
    }
    with TemporaryDirectory(dir=output) as pending:
        generation = Path(pending) / "generation"
        generation.mkdir()
        for arm in ARMS:
            checkpoint = checkpoints / arm / "step_002000.pt"
            model, receipt = read_ema(checkpoint, arm, config.model, identity)
            model = model.to(device)
            rows = []
            for seed, noise in noise_by_seed.items():
                sources = zip(MODES, (labels, eye_labels), strict=True)
                for mode, records in sources:
                    variant_dir = generation / arm / str(seed) / mode
                    positions = [row["index"] for row in records]
                    group = [
                        row
                        | {"noise_sha256": tensor_digest(noise[row["index"] : row["index"] + 1])}
                        for row in records
                    ]
                    sample_variant(
                        variant_dir,
                        model,
                        vae,
                        noise[positions].to(device),
                        conditions[mode],
                        blank,
                        group,
                        receipt,
                        seed=seed,
                    )
                    report = read_json(variant_dir / "samples.json")
                    for row in report["records"]:
                        path = variant_dir / row["image_path"]
                        rows.append(
                            row
                            | {
                                "image_path": str(path.relative_to(generation / arm)),
                                "image_sha256": hash_file(path),
                                "seed": seed,
                                "mode": mode,
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
                    "eye_plan_sha256": hash_file(eye_plan),
                    "records": rows,
                },
            )
        generation.rename(destination)


def scoring_endpoint(
    key: str,
    caption: str,
) -> JsonDict:
    """将固定开发标签中的眼色、发色和开闭状态转为评分项。

    Args:
        key: 匿名题卡的 single、a 或 b 端点。
        caption: 该图生成时的逗号分隔标签。

    Results:
        JsonDict: 提示词、评分标签和虹膜适用性。
    """
    tags = caption.split(", ")
    colors = {
        "black",
        "blue",
        "brown",
        "dark blue",
        "green",
        "light brown",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
        "blonde",
        "gray",
        "pink",
        "orange",
    }
    targets = {}
    for tag in tags:
        value, _, part = tag.rpartition(" ")
        if value in colors and part in {"eyes", "hair"}:
            targets["eye_color" if part == "eyes" else "hair_color"] = tag
        if tag in {"closed eyes", "one eye closed"}:
            targets["eye_state"] = tag
    return endpoint(key, caption, targets) | {"iris_required": "closed eyes" not in tags}


def render(
    output: Path,
) -> None:
    """验证已生成的配对图，输出公开匿名题卡和独立的私有身份表。

    Args:
        output: 包含两组 generation 回执的开发评测目录。

    Raises:
        OSError: 输入不全、输出已存在或不可写。
        ValueError: 两臂配置、提示词、初噪、干预配对或图像摘要不一致。

    副作用:
        写入匿名 PNG、离线 HTML 题库、公开元数据与私有解盲表。
    """
    generation = output / "generation"
    receipts = {arm: read_json(generation / arm / "receipt.json") for arm in ARMS}
    first, second = receipts.values()
    common = ("data_identity", "sampling", "eye_plan_sha256")
    fields = ("seed", "mode", "index", "sample_id", "caption", "noise_sha256")
    if (
        any(first[key] != second[key] for key in common)
        or first["sampling"] != SAMPLING
        or first["checkpoint_sha256"] == second["checkpoint_sha256"]
        or [row["arm"] for row in receipts.values()] != list(ARMS)
        or [row["status"] for row in receipts.values()] != ["completed"] * 2
        or [tuple(row[key] for key in fields) for row in first["records"]]
        != [tuple(row[key] for key in fields) for row in second["records"]]
    ):
        raise ValueError("Paired generation receipts disagree")
    baseline = {
        (row["seed"], row["index"]): row for row in first["records"] if row["mode"] == "correct"
    }
    groups: dict[tuple[int, str, int], list[int]] = {}
    for position, row in enumerate(first["records"]):
        seed, mode, index = row["seed"], row["mode"], row["index"]
        if seed not in SEEDS or mode not in MODES or index not in range(COUNT):
            raise ValueError("Unexpected development prompt or initial noise seed")
        reference = baseline[seed, index]
        if (
            row["noise_sha256"] != reference["noise_sha256"]
            or row["sample_id"] != reference["sample_id"]
        ):
            raise ValueError(
                "Eye-color intervention differs from the paired initial noise or subject"
            )
        groups.setdefault((seed, mode, index), []).append(position)
    if (
        len(groups) != 44
        or len(baseline) != 32
        or len(first["records"]) != 56
        or any(
            len(positions) != (1 if mode == "correct" else 2)
            for (_, mode, _), positions in groups.items()
        )
    ):
        raise ValueError("Expected 16 development labels and six eye swaps with two seeds")

    secret = secrets.token_bytes(32)
    review_id = order_key(
        secret, f"b2-repa-dev:{first['checkpoint_sha256']}:{second['checkpoint_sha256']}"
    )[:24]
    cards, mapping = [], {}
    review = output / "review"
    private = output / "private"
    if review.exists() or private.exists():
        raise FileExistsError("An anonymous review already exists")
    review.mkdir()
    private.mkdir(mode=0o700)
    picture_dir = review / "cards"
    picture_dir.mkdir()
    for seed in SEEDS:
        # 每个初噪内分别平衡普通题和换词题在左右位置上的出现次数。
        for mode in MODES:
            questions = sorted(
                (question for question in groups if question[:2] == (seed, mode)),
                key=lambda question: order_key(secret, f"side:{question}"),
            )
            for index, question in enumerate(questions):
                positions = groups[question]
                paired = mode == "eye_control"
                keys = ("a", "b") if paired else ("single",)
                layout = ARMS if index % 2 == 0 else ARMS[::-1]
                opaque = order_key(secret, f"card:{question}")[:16]
                endpoints = [
                    scoring_endpoint(key, first["records"][position]["caption"])
                    for key, position in zip(keys, positions, strict=True)
                ]
                card = {
                    "id": opaque,
                    "endpoints": endpoints,
                    "kind": "pair" if paired else "regular",
                    "category": "eyes",
                    "images": {},
                }
                if paired:
                    old, new = (meta["targets"]["eye_color"] for meta in endpoints)
                    tags_a, tags_b = (meta["prompt"].split(", ") for meta in endpoints)
                    if old == new or [new if tag == old else tag for tag in tags_a] != tags_b:
                        raise ValueError(
                            "Only one known eye-color tag may change in the paired view"
                        )
                    card["change"] = {"from": old, "to": new}
                for side, arm in zip(SIDES, layout, strict=True):
                    pictures = {}
                    for key, position in zip(keys, positions, strict=True):
                        row = receipts[arm]["records"][position]
                        uri = image_data(
                            generation / arm / row["image_path"], row["image_sha256"], 256
                        )
                        pictures[key] = uri
                        (picture_dir / f"{opaque}_{side}_{key}.png").write_bytes(
                            base64.b64decode(uri.partition(",")[2])
                        )
                    card["images"][side] = pictures
                cards.append(card)
                mapping[opaque] = {"seed": seed, "development_index": question[2]} | dict(
                    zip(SIDES, layout, strict=True)
                )
    cards.sort(key=lambda card: order_key(secret, f"position:{card['id']}"))
    data = json.dumps({"review_id": review_id, "items": cards}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    html = (
        HTML.replace("__ANONYMOUS_DATA__", data)
        .replace("return triple && [", "return [")
        .replace('if (triple && field === "iris"', 'if (field === "iris"')
        .replace('$("eye-color-link").hidden = !triple;', '$("eye-color-link").hidden = false;')
    )
    (review / "blind_review.html").write_text(html, encoding="utf-8")
    write_json(
        review / "deck.json",
        {
            "review_id": review_id,
            "items": [
                {key: value for key, value in card.items() if key != "images"} for card in cards
            ],
        },
    )
    write_json(
        private / "assignment.json",
        {
            "review_id": review_id,
            "deck_sha256": hash_file(review / "deck.json"),
            "review_html_sha256": hash_file(review / "blind_review.html"),
            "assignments": mapping,
        },
    )


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> None:
    """验证一名评审的完整匿名评分，锁分时无需身份映射。

    Args:
        deck: 冻结匿名题卡元数据。
        scores: 评分页面导出的逐题数据。

    Raises:
        ValueError: 题卡、适用性、打分枚举或完整性不符。
    """
    if (
        scores.get("schema") != SCORE_SCHEMA
        or scores.get("review_id") != deck["review_id"]
        or not str(scores.get("reviewer", "")).strip()
        or set(scores.get("items", {})) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Scores do not cover this exact deck with a named reviewer")
    for card in deck["items"]:
        item = scores["items"][card["id"]]
        paired = card["kind"] == "pair"
        if item.get("preference") not in (*SIDES, "tie", "not_comparable"):
            raise ValueError(f"Missing preference for {card['id']}")
        for side in SIDES:
            rating = item[side]
            if paired and rating.get("preservation") not in PRESERVATION:
                raise ValueError(f"Missing preservation for {card['id']}/{side}")
            for point in card["endpoints"]:
                score = rating["endpoints"][point["key"]]
                allowed = (
                    ("readable", "unreadable", "failure")
                    if point["iris_required"]
                    else ("not_applicable",)
                )
                if (
                    score["face"] not in FACE
                    or score["iris"] not in allowed
                    or set(score["targets"]) != set(point["targets"])
                    or any(value not in ATTRIBUTE for value in score["targets"].values())
                ):
                    raise ValueError(f"Invalid scores for {card['id']}/{side}/{point['key']}")
                eye = score["targets"].get("eye_color")
                if (
                    eye is not None
                    and score["iris"] in ("unreadable", "failure")
                    and eye != score["iris"]
                ):
                    raise ValueError(
                        f"Eye color contradicts the iris score for {card['id']}/{side}"
                    )


def lock(
    output: Path,
    score_path: Path,
) -> None:
    """独立验证匿名评分并锁定评分、题卡和 HTML 摘要。

    Args:
        output: 开发评测目录。
        score_path: 评审导出的匿名 JSON。

    Raises:
        OSError: 输入缺失或已有锁分。
        ValueError: 评分与题卡不一致。

    副作用:
        写入评分副本和锁分回执。
    """
    review = output / "review"
    scores = read_json(score_path)
    deck_path = review / "deck.json"
    validate_scores(read_json(deck_path), scores)
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


def deblind(
    output: Path,
) -> None:
    """解盲已锁评分并按初噪和模型报告逐项计数。

    Args:
        output: 含公开锁分和私有解盲映射的开发评测目录。

    Raises:
        OSError: 评分或映射不可读，或汇总文件已存在。
        ValueError: 锁分摘要、题卡或映射身份不符。

    副作用:
        写入解盲汇总，保留常规标签和换词题的独立分母。
    """
    review = output / "review"
    frozen = read_json(review / "scores.lock.json")
    deck = read_json(review / "deck.json")
    scores = read_json(review / "scores.json")
    for name, path in (
        ("scores", "scores.json"),
        ("deck", "deck.json"),
        ("review_html", "blind_review.html"),
    ):
        if frozen[f"{name}_sha256"] != hash_file(review / path):
            raise ValueError(f"Locked {name} differs from observed file")
    validate_scores(deck, scores)
    key = read_json(output / "private/assignment.json")
    if any(
        key[field] != frozen[field] for field in ("review_id", "deck_sha256", "review_html_sha256")
    ) or set(key["assignments"]) != set(scores["items"]):
        raise ValueError("Private mapping does not correspond to the locked review")

    counts: dict[str, Counter[str]] = {}
    preferences: dict[str, Counter[str]] = {}
    changes: dict[str, Counter[str]] = {}
    for card in deck["items"]:
        group = key["assignments"][card["id"]]
        seed = str(group["seed"])
        rated = scores["items"][card["id"]]
        kind = card["kind"]
        preference = rated["preference"]
        by_arm = {group[side]: rated[side]["endpoints"] for side in SIDES}
        transitions = Counter()
        for point in card["endpoints"]:
            first, second = (by_arm[arm][point["key"]] for arm in ARMS)
            targets = {"face": "usable"}
            if point["iris_required"]:
                targets["iris"] = "readable"
            for metric, success in targets.items():
                was, now = first[metric] == success, second[metric] == success
                label = {(False, True): "gain", (True, False): "loss"}.get((was, now), "tie")
                transitions[f"{metric}:{label}"] += 1
        for noise in (seed, "all"):
            preferred = preferences.setdefault(f"{noise}/{kind}", Counter())
            preferred[group.get(preference, preference)] += 1
            changes.setdefault(f"{noise}/{kind}", Counter()).update(transitions)
            for side in SIDES:
                arm = group[side]
                count = counts.setdefault(f"{arm}/{noise}/{kind}", Counter())
                for score in rated[side]["endpoints"].values():
                    count[f"face:{score['face']}"] += 1
                    count[f"iris:{score['iris']}"] += 1
                    for label, state in score["targets"].items():
                        count[f"{label}:{state}"] += 1
                if kind == "pair":
                    endpoints = rated[side]["endpoints"].values()
                    switch = all(end["targets"]["eye_color"] == "correct" for end in endpoints)
                    preservation = rated[side]["preservation"]
                    count[f"switch:{'pass' if switch else 'fail'}"] += 1
                    count[f"preservation:{preservation}"] += 1
                    count[
                        f"swap_joint:{'pass' if switch and preservation == 'preserved' else 'fail'}"
                    ] += 1
    write_json(
        review / "deblinded_metrics.json",
        {
            "review_id": deck["review_id"],
            "reviewer": scores["reviewer"],
            "score_lock": frozen,
            "counts": counts,
            "paired_changes_control_to_repa": changes,
            "preferences": preferences,
        },
    )


def main() -> None:
    """调用开发采样、匿名渲染、评分冻结或解盲步骤。

    副作用:
        根据子命令执行 AutoDL CUDA 采样或读写评测文件。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "render", "lock", "deblind"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=RUNS)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    output = args.root / args.output
    if args.action == "generate":
        generate(args.root, output, args.root / args.checkpoint_root)
    elif args.action == "lock":
        if args.scores is None:
            parser.error("lock needs --scores")
        lock(output, args.scores)
    elif args.action == "render":
        render(output)
    else:
        deblind(output)
    print(f"{args.action}: {output}")


if __name__ == "__main__":
    main()
