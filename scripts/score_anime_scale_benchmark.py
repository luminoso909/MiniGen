"""验证并锁定三列头像盲评分，解盲后计算与固定 30M 基线的配对结果。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from random import Random

from render_anime_benchmark_blind import endpoint, read_json

from minigen._typing import JsonDict

ARMS = ("uniform20k", "b1_50m_20k", "b2_50m_20k")
COLUMNS = ("alpha", "beta", "gamma")
COLUMN_PAIRS = tuple(itertools.combinations(COLUMNS, 2))
ARM_PAIRS = tuple(itertools.combinations(ARMS, 2))
SEEDS = (2026111401, 2026111402)
BENCHMARK = Path("configs/benchmark/anime_faces_v1.json")
DECK_MARKER = b'<script id="deck" type="application/json">'
SCORE_SCHEMA = "minigen.anime_faces.blind_scores.v2"
MAP_SCHEMA = "minigen.anime_faces.blind_mapping.v2"
LOCK_SCHEMA = "minigen.anime_faces.blind_scores_lock.v2"
RESULT_SCHEMA = "minigen.anime_faces.scale_blind_metrics.v1"

FACE = ("usable", "distorted", "unrecognizable", "failure")
IRIS = ("readable", "unreadable", "not_applicable", "failure")
ATTRIBUTE = ("correct", "wrong", "unreadable", "failure")
PRESERVATION = ("preserved", "changed", "unreadable", "failure")
SUCCESS = {
    "face": "usable",
    "iris": "readable",
    "eye_color": "correct",
    "swap_switch": "correct",
    "swap_preservation": "preserved",
    "swap_joint": "correct",
}
STATUS = {
    "face": FACE,
    "iris": IRIS,
    "eye_color": (*ATTRIBUTE, "not_applicable"),
    "swap_switch": ATTRIBUTE,
    "swap_preservation": PRESERVATION,
    "swap_joint": ATTRIBUTE,
}
DENOMINATORS = {
    "face": 256,
    "iris": 152,
    "eye_color": 128,
    "swap_switch": 48,
    "swap_preservation": 48,
    "swap_joint": 48,
}
PRIMARY_THRESHOLDS = {"face": 13, "iris": 8}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261117

# 以来源组为抽样单元，同组的两个种子和两个端点保留独立判定。
type PointKey = tuple[str, str, int, str]
type Observations = dict[PointKey, str]
type ModelScores = dict[str, Observations]


def read_deck(
    path: Path,
) -> tuple[JsonDict, str]:
    """读取自包含 HTML 中的公开题卡元数据并计算整页摘要。

    Args:
        path: 渲染器生成的三列匿名评分页面。

    Raises:
        OSError: HTML 不可读取。
        ValueError: 页面缺少三列数据或 208 张唯一题卡，或 JSON 解析失败。

    Results:
        tuple: 去掉图片负载的公开题卡与 HTML SHA-256。
    """
    document = path.read_bytes()
    start = document.index(DECK_MARKER) + len(DECK_MARKER)
    deck = json.loads(document[start : document.index(b"</script>", start)])
    if deck.get("columns") != list(COLUMNS) or (
        len(deck["items"]) != 208 or len({card["id"] for card in deck["items"]}) != 208
    ):
        raise ValueError("Expected three anonymous columns and 208 distinct cards")
    for card in deck["items"]:
        card.pop("images", None)
    return deck, hashlib.sha256(document).hexdigest()


def validate_scores(
    deck: JsonDict,
    scores: JsonDict,
) -> dict[str, int]:
    """在不读取身份映射的情况下核对所有行、字段和适用性。

    Args:
        deck: 公开题卡元数据。
        scores: 一名评分者导出的完整三列评分。

    Raises:
        ValueError: 评分身份、适用性或字段不完整，或评分枚举不合法。

    Results:
        dict[str, int]: 已完成的题卡数、评分图数、适用虹膜与眼色数。
    """
    if (
        scores.get("schema") != SCORE_SCHEMA
        or not isinstance(scores.get("reviewer"), str)
        or not scores["reviewer"].strip()
        or scores.get("review_id") != deck["review_id"]
        or not isinstance(scores.get("items"), dict)
        or set(scores["items"]) != {card["id"] for card in deck["items"]}
    ):
        raise ValueError("Scores must cover this exact deck, with a named reviewer")

    counts = dict(cards=0, images=0, iris=0, eye_color=0, swaps=0)
    for card in deck["items"]:
        card_id = card["id"]
        points = {point["key"]: point for point in card["endpoints"]}
        paired = card["kind"] == "pair"
        expected_points = {"a", "b"} if paired else {"single"}
        rating = scores["items"][card_id]
        if (
            set(points) != expected_points
            or set(rating) - {"note"} != {*COLUMNS, "preferences"}
            or set(rating["preferences"]) != {f"{one}-{two}" for one, two in COLUMN_PAIRS}
            or any(
                rating["preferences"][f"{one}-{two}"] not in (one, two, "tie", "not_comparable")
                for one, two in COLUMN_PAIRS
            )
        ):
            raise ValueError(f"Unexpected columns, endpoints or preferences for card {card_id}")

        for column in COLUMNS:
            rated = rating[column]
            expected_fields = {"endpoints", "preservation"} if paired else {"endpoints"}
            if set(rated) != expected_fields or set(rated["endpoints"]) != expected_points:
                raise ValueError(f"Incomplete anonymous column {card_id}/{column}")
            if paired and rated["preservation"] not in PRESERVATION:
                raise ValueError(f"Invalid preservation for card {card_id}/{column}")
            for key, point in points.items():
                score = rated["endpoints"][key]
                allowed_iris = (
                    ("readable", "unreadable", "failure")
                    if point["iris_required"]
                    else ("not_applicable",)
                )
                if (
                    set(score) != {"face", "iris", "targets"}
                    or score["face"] not in FACE
                    or score["iris"] not in allowed_iris
                    or set(score["targets"]) != set(point["targets"])
                    or any(value not in ATTRIBUTE for value in score["targets"].values())
                ):
                    raise ValueError(f"Missing or invalid scores for {card_id}/{column}/{key}")
                eye_color = score["targets"].get("eye_color")
                if eye_color is not None and score["iris"] in {"unreadable", "failure"}:
                    if eye_color != score["iris"]:
                        raise ValueError(
                            f"Eye color contradicts iris score for {card_id}/{column}/{key}"
                        )
                counts["images"] += 1
                counts["iris"] += point["iris_required"]
                counts["eye_color"] += "eye_color" in point["targets"]
            counts["swaps"] += paired
        counts["cards"] += 1

    if counts != {"cards": 208, "images": 768, "iris": 456, "eye_color": 384, "swaps": 144}:
        raise ValueError(f"Unexpected counts for the fixed benchmark: {counts}")
    return counts


def score_fingerprints(
    deck_path: Path,
    scores_path: Path,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """核对匿名评分并用原始字节计算可复查的摘要。

    Args:
        deck_path: 三列匿名评分页面。
        scores_path: 浏览器导出的完整评分。

    Results:
        tuple: 公开题卡、完整评分、供锁分和解盲重验的身份记录。
    """
    deck, deck_sha256 = read_deck(deck_path)
    scores_bytes = scores_path.read_bytes()
    scores = json.loads(scores_bytes)
    receipt = {
        "schema": LOCK_SCHEMA,
        "review_id": deck["review_id"],
        "reviewer": scores.get("reviewer"),
        "deck_sha256": deck_sha256,
        "scores_sha256": hashlib.sha256(scores_bytes).hexdigest(),
        "counts": validate_scores(deck, scores),
    }
    return deck, scores, receipt


def validate_key(
    deck: JsonDict,
    key: JsonDict,
    benchmark_path: Path,
) -> None:
    """确认私有映射覆盖冻结题库且同题两种子沿用模型列位。

    Args:
        deck: 已锁分页中的公开题卡。
        key: 汇总者保管的私有映射。
        benchmark_path: 原始冻结 Benchmark JSON。

    Raises:
        OSError: 题库不可读取。
        ValueError: 解盲映射不属于这次评分，或题意与冻结记录不符。
    """
    content = benchmark_path.read_bytes()
    benchmark = json.loads(content)
    assignments = key.get("assignments", {})
    if (
        key.get("schema") != MAP_SCHEMA
        or key.get("review_id") != deck["review_id"]
        or key.get("benchmark_sha256") != hashlib.sha256(content).hexdigest()
        or key.get("arms") != list(ARMS)
        or set(assignments) != {card["id"] for card in deck["items"]}
        or benchmark["seeds"] != list(SEEDS)
    ):
        raise ValueError("Private assignments disagree with the locked review or benchmark")

    questions = {
        question["id"]: (kind, question)
        for kind, questions in (("regular", benchmark["regular"]), ("pair", benchmark["pairs"]))
        for question in questions
    }
    seen = set()
    orders = {}
    for card in deck["items"]:
        row = assignments[card["id"]]
        question_id, seed = row["question_id"], row["seed"]
        kind, question = questions[question_id]
        order = tuple(row[column] for column in COLUMNS)
        order_for_question = orders.setdefault(question_id, order)
        expected_points = (
            [endpoint("single", question["prompt"], question["targets"])]
            if kind == "regular"
            else [
                endpoint(label, question[f"prompt_{label}"], question[f"targets_{label}"])
                for label in ("a", "b")
            ]
        )
        if (
            seed not in SEEDS
            or (question_id, seed) in seen
            or set(order) != set(ARMS)
            or order_for_question != order
            or question["source"]["group_id"] != row["group_id"]
            or row["kind"] != kind
            or card["kind"] != kind
            or card["category"] != row["category"]
            or row["category"] != question["category"]
            or row["changed_attribute"] != question.get("changed", {}).get("attribute")
            or card.get("change") != question.get("changed")
            or card["endpoints"] != expected_points
        ):
            raise ValueError(f"Inconsistent private assignment for card {card['id']}")
        seen.add((question_id, seed))
    if seen != set(itertools.product(questions, SEEDS)):
        raise ValueError("Expected both seeds for each frozen question")


def swap_outcome(
    target_a: str,
    target_b: str,
    preservation: str,
) -> str:
    """把两个端点及非目标保持合成一个干预结果。

    Args:
        target_a: 原标签端点的目标属性得分。
        target_b: 替换端点的目标属性得分。
        preservation: 非目标保持得分，计算单独切换时传入 preserved。

    Results:
        str: 联合正确、错误、难辨或生成失败。
    """
    ratings = (target_a, target_b, preservation)
    if "failure" in ratings:
        return "failure"
    if "wrong" in ratings or preservation == "changed":
        return "wrong"
    return "unreadable" if "unreadable" in ratings else "correct"


def resolve_scores(
    scores: JsonDict,
    assignments: JsonDict,
) -> tuple[dict[str, ModelScores], dict[str, JsonDict]]:
    """依据锁分后的映射，将匿名评分归到相同的配对观测索引。

    Args:
        scores: 已冻结的匿名评分。
        assignments: 已验证的每张题卡私有映射。

    Results:
        tuple: 各组逐指标状态及按真实模型对归类的头像偏好原始计数。
    """
    observations = {arm: {metric: {} for metric in SUCCESS} for arm in ARMS}
    preferences = {
        f"{one}__{two}": {
            "first_arm": one,
            "second_arm": two,
            "overall": dict(first_wins=0, second_wins=0, tie=0, not_comparable=0),
            "by_seed": {
                str(seed): dict(first_wins=0, second_wins=0, tie=0, not_comparable=0)
                for seed in SEEDS
            },
        }
        for one, two in ARM_PAIRS
    }
    for card_id, card in scores["items"].items():
        row = assignments[card_id]
        prefix = row["group_id"], row["question_id"], row["seed"]
        for column in COLUMNS:
            model = observations[row[column]]
            rated = card[column]
            for label, score in rated["endpoints"].items():
                point = (*prefix, label)
                model["face"][point] = score["face"]
                model["iris"][point] = score["iris"]
                model["eye_color"][point] = score["targets"].get("eye_color", "not_applicable")
            if row["kind"] == "pair":
                point = (*prefix, "swap")
                changed = row["changed_attribute"]
                target_a = rated["endpoints"]["a"]["targets"][changed]
                target_b = rated["endpoints"]["b"]["targets"][changed]
                preservation = rated["preservation"]
                model["swap_switch"][point] = swap_outcome(target_a, target_b, "preserved")
                model["swap_preservation"][point] = preservation
                model["swap_joint"][point] = swap_outcome(target_a, target_b, preservation)

        for one, two in COLUMN_PAIRS:
            pair = next(arms for arms in ARM_PAIRS if set(arms) == {row[one], row[two]})
            bucket = preferences["__".join(pair)]
            picked = card["preferences"][f"{one}-{two}"]
            if picked in COLUMNS:
                picked = "first_wins" if row[picked] == pair[0] else "second_wins"
            bucket["overall"][picked] += 1
            bucket["by_seed"][str(row["seed"])][picked] += 1

    for bucket in preferences.values():
        bucket["overall"]["comparable"] = 208 - bucket["overall"]["not_comparable"]
        for by_seed in bucket["by_seed"].values():
            by_seed["comparable"] = 104 - by_seed["not_comparable"]
    return observations, preferences


def summarize(
    observations: Observations,
    metric: str,
    seed: int | None = None,
) -> JsonDict:
    """按冻结分母统计单组指标和每类原始状态。

    Args:
        observations: 单组单指标的全部观测。
        metric: 已定义的指标名称。
        seed: 指定初噪或汇总两个初噪。

    Raises:
        ValueError: 观测适用分母与冻结题库不符。

    Results:
        JsonDict: 成功数、分母、百分比、各状态计数。
    """
    counts = Counter(
        value for point, value in observations.items() if seed is None or point[2] == seed
    )
    eligible = sum(counts.values()) - counts["not_applicable"]
    expected = DENOMINATORS[metric] if seed is None else DENOMINATORS[metric] // 2
    if eligible != expected:
        raise ValueError(f"Unexpected denominator for {metric} at {seed}: {eligible} vs {expected}")
    success = counts[SUCCESS[metric]]
    return {
        "success": success,
        "applicable": eligible,
        "rate_percent": 100 * success / eligible,
        "counts": {state: counts[state] for state in STATUS[metric]},
    }


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    """对排序后的 bootstrap 分布做线性百分位插值。

    Args:
        values: 非空升序的百分比差值。
        fraction: 在零至一之间的百分位。

    Results:
        float: 线性插值后的百分位。
    """
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def compare(
    baseline: Observations,
    candidate: Observations,
    metric: str,
    group_ids: list[str],
    draws: list[tuple[int, ...]],
) -> JsonDict:
    """按同题端点计算改善、退步和来源组配对区间。

    Args:
        baseline: 30M 的逐项状态。
        candidate: 某个 50M 的逐项状态。
        metric: 冻结的主指标或次指标名称。
        group_ids: 完整来源组列表，顺序与抽样索引一致。
        draws: 共享的来源组 bootstrap 抽样索引。

    Raises:
        ValueError: 两组样本索引、适用性或配对分母与冻结题库不符。

    Results:
        JsonDict: 总计与分种子净增，配对百分比差和百分位区间。
    """
    if baseline.keys() != candidate.keys():
        raise ValueError(f"Unpaired observations for {metric}")
    changes = {seed: Counter() for seed in SEEDS}
    group_denominators = Counter()
    group_net = Counter()
    success = SUCCESS[metric]
    for point, before in baseline.items():
        after = candidate[point]
        before_included = before != "not_applicable"
        if before_included != (after != "not_applicable"):
            raise ValueError(f"Different applicability for {metric} at {point}")
        if not before_included:
            continue
        group, _, seed, _ = point
        delta = int(after == success) - int(before == success)
        group_denominators[group] += 1
        group_net[group] += delta
        changes[seed][delta] += 1

    denominator = DENOMINATORS[metric]
    if any(change.total() != denominator // 2 for change in changes.values()):
        raise ValueError(f"Unexpected paired denominators for {metric}")
    per_seed = {
        str(seed): {
            "improved": change[1],
            "worsened": change[-1],
            "unchanged": change[0],
            "net": change[1] - change[-1],
        }
        for seed, change in changes.items()
    }
    improved = sum(change[1] for change in changes.values())
    worsened = sum(change[-1] for change in changes.values())
    net = improved - worsened
    denominators = [group_denominators[group] for group in group_ids]
    deltas = [group_net[group] for group in group_ids]
    bootstrap = []
    for draw in draws:
        applicable = sum(denominators[index] for index in draw)
        if applicable:
            bootstrap.append(100 * sum(deltas[index] for index in draw) / applicable)
    bootstrap.sort()
    result = {
        "applicable": denominator,
        "improved": improved,
        "worsened": worsened,
        "unchanged": denominator - improved - worsened,
        "net": net,
        "delta_pp": 100 * net / denominator,
        "by_seed": per_seed,
        "bootstrap_95_percent_pp": [percentile(bootstrap, p) for p in (0.025, 0.975)],
        "valid_bootstrap_replicates": len(bootstrap),
    }
    if metric in PRIMARY_THRESHOLDS:
        minimum = PRIMARY_THRESHOLDS[metric]
        result["minimum_net"] = minimum
        result["passes"] = net >= minimum and all(value["net"] > 0 for value in per_seed.values())
    return result


def build_report(
    scores: JsonDict,
    key: JsonDict,
    lock: JsonDict,
) -> JsonDict:
    """生成三组盲评成绩、候选逐项比对和提升门槛判定。

    Args:
        scores: 已锁定并验证的匿名评分。
        key: 独立保存且已校验的身份映射。
        lock: 已校验的锁分回执。

    Results:
        JsonDict: 可持久化的正式汇总及可查证的输入身份。
    """
    observations, preferences = resolve_scores(scores, key["assignments"])
    baseline = ARMS[0]
    groups = sorted({point[0] for point in observations[baseline]["face"]})
    rng = Random(BOOTSTRAP_SEED)
    draws = [
        tuple(rng.choices(range(len(groups)), k=len(groups))) for _ in range(BOOTSTRAP_REPLICATES)
    ]
    models = {
        arm: {
            "overall": {metric: summarize(points, metric) for metric, points in metrics.items()},
            "by_seed": {
                str(seed): {
                    metric: summarize(points, metric, seed) for metric, points in metrics.items()
                }
                for seed in SEEDS
            },
        }
        for arm, metrics in observations.items()
    }
    candidates = {}
    for arm in ARMS[1:]:
        metrics = {
            metric: compare(observations[baseline][metric], points, metric, groups, draws)
            for metric, points in observations[arm].items()
        }
        candidates[arm] = {
            "metrics": metrics,
            "passes": all(metrics[metric]["passes"] for metric in PRIMARY_THRESHOLDS),
        }
    return {
        "schema": RESULT_SCHEMA,
        "locked_review": lock,
        "benchmark_sha256": key["benchmark_sha256"],
        "comparison_scope_sha256": key.get("comparison_scope_sha256"),
        "sampling": key["sampling"],
        "checkpoint_identities": key["checkpoint_identities"],
        "baseline": baseline,
        "models": models,
        "comparisons_vs_baseline": candidates,
        "pairwise_preferences": preferences,
        "bootstrap": {
            "resampling": "source_group_with_all_seeds_endpoints_and_arms",
            "groups": len(groups),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
    }


def write_new_json(
    path: Path,
    data: JsonDict,
) -> None:
    """创建新的 JSON 结果文件并保护既有锁分与报告。

    Args:
        path: 要创建的本地文件路径。
        data: 可序列化的结果字典。

    Raises:
        FileExistsError: 该结果文件已存在。
        OSError: 文件不能写入。

    副作用:
        写入一个新的 JSON 文件。
    """
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def main() -> None:
    """通过分离的命令先锁定匿名评分，再校验身份映射并汇总。

    Raises:
        SystemExit: 参数不合法。
        OSError/ValueError: 评分无法锁定、输入摘要不符或映射不完整。

    副作用:
        创建锁分回执或已解盲的 JSON 汇总，并打印产物路径。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    freeze = actions.add_parser("freeze", help="校验并锁定匿名评分")
    reveal = actions.add_parser("reveal", help="核验锁分后读取私有映射并计算")
    for action in (freeze, reveal):
        action.add_argument("--deck", type=Path, required=True)
        action.add_argument("--scores", type=Path, required=True)
        action.add_argument("--lock", type=Path, required=True)
    reveal.add_argument("--key", type=Path, required=True)
    reveal.add_argument("--benchmark", type=Path, default=BENCHMARK)
    reveal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    deck, scores, fingerprints = score_fingerprints(args.deck, args.scores)
    if args.action == "freeze":
        write_new_json(args.lock, {**fingerprints, "locked_at": datetime.now(UTC).isoformat()})
        print(f"Locked {fingerprints['counts']['cards']} cards: {args.lock}")
        return
    lock = read_json(args.lock)
    if not lock.get("locked_at") or {key: lock.get(key) for key in fingerprints} != fingerprints:
        raise ValueError("Blind scores or deck changed after the lock was recorded")
    key = read_json(args.key)
    validate_key(deck, key, args.benchmark)
    write_new_json(args.output, build_report(scores, key, lock))
    print(f"Revealed 208 locked cards: {args.output}")


if __name__ == "__main__":
    main()
