"""仅根据文本请求的属性评测；不使用未被提示的绝对位置匹配生成对象。"""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from minigen._typing import JsonDict
from minigen.evaluation.icons import parse_icons


def semantic_key(
    record: JsonDict,
) -> str:
    """把请求规范化为对象与有向关系，排除措辞和隐藏绝对坐标。

    Args:
        record: 含 attributes.objects/relations 的场景标注。

    Raises:
        KeyError/IndexError: 标注不完整或关系索引非法时透传。

    Results:
        str: 可比较的语义键；同义描述必须得到同一键。
    """
    objects = record["attributes"]["objects"]
    names = [(o["color"], o["shape"]) for o in objects]
    relations = sorted(
        (names[r["subject"]], r["predicate"], names[r["object"]])
        for r in record["attributes"]["relations"]
    )
    return repr((sorted(names), relations))


def shuffled_indices(
    records: list[JsonDict],
) -> list[int]:
    """寻找与自身语义均不同的一一置换，避免伪打乱对照。

    Args:
        records: 同一评价组的请求列表，至少需要两个不同语义。

    Raises:
        ValueError: 没有满足全员语义不同的循环置换；调用者应重建评价组。

    Results:
        list[int]: 每条请求对应的打乱条件索引，保持条件边际分布。
    """
    keys = [semantic_key(r) for r in records]
    for shift in range(1, len(keys)):
        order = [(i + shift) % len(keys) for i in range(len(keys))]
        if all(keys[i] != keys[j] for i, j in enumerate(order)):
            return order
    raise ValueError("No semantically distinct derangement available")


def score_request(
    image: Image.Image,
    record: JsonDict,
) -> dict[str, int]:
    """评分对象多重集、颜色形状绑定及文本指定关系，失败样本保持在分母。

    Args:
        image: 生成或控制图像；解析器仅适用受控色板与 1–3 个几何对象。
        record: 请求属性，不能使用其中未被描述的中心坐标作为生成评分条件。

    Raises:
        ValueError: 请求含重复颜色形状组合或不支持的关系，无法唯一指代。
        KeyError/IndexError: 请求字段不完整时透传。

    Results:
        dict[str,int]: 逐图二元命中、关系数及解析失败标记。
            color/shape: 颜色/形状多重集完全一致且数量正确。
            binding: 颜色与形状的配对多重集完全一致。
            all: binding 与全部关系同时满足；无关系时不增加额外要求。
    """
    objects = record["attributes"]["objects"]
    relations = record["attributes"]["relations"]
    names = [(o["color"], o["shape"]) for o in objects]
    if len(set(names)) != len(names) or any(
        r["predicate"] not in {"left of", "above"} for r in relations
    ):
        raise ValueError("Unique referring pairs and supported relations required")
    predicted = parse_icons(image)
    failure = not 1 <= len(predicted) <= 3
    count = not failure and len(predicted) == len(objects)
    color = count and Counter(o["color"] for o in objects) == Counter(o["color"] for o in predicted)
    shape = count and Counter(o["shape"] for o in objects) == Counter(o["shape"] for o in predicted)
    predicted_names = [(o["color"], o["shape"]) for o in predicted]
    binding = count and Counter(names) == Counter(predicted_names)
    hits = 0
    if binding:
        lookup = dict(zip(predicted_names, predicted, strict=True))
        for relation in relations:
            a = lookup[names[relation["subject"]]]["center"]
            b = lookup[names[relation["object"]]]["center"]
            axis = 0 if relation["predicate"] == "left of" else 1
            hits += int(a[axis] < b[axis])
    return {
        "count": int(count),
        "color": int(color),
        "shape": int(shape),
        "binding": int(binding),
        "relation_hits": hits,
        "relation_total": len(relations),
        "has_relation": int(bool(relations)),
        "relation_all": int(bool(relations) and binding and hits == len(relations)),
        "all": int(binding and hits == len(relations)),
        "parse_failure": int(failure),
    }


def summarize(
    scores: list[dict[str, int]],
) -> JsonDict:
    """按图汇总完整分母，附综合成功率的 Wilson 95% 区间。

    Args:
        scores: 非空逐图结果；解析失败不得过滤。

    Raises:
        ValueError: 样本为空。
        KeyError: 评分字段缺失时透传。

    Results:
        JsonDict: 逐属性均值、解析失败率、关系分母及 all 的 Wilson 区间。
            relation: 无含关系请求时为 None，不用零伪装一个实际测量。
    """
    if not scores:
        raise ValueError("Nonempty scores required")
    n = len(scores)
    result = {
        key: sum(s[key] for s in scores) / n
        for key in ("count", "color", "shape", "binding", "all", "parse_failure")
    }
    relation_n = sum(s["has_relation"] for s in scores)
    result.update(
        images=n,
        relation_images=relation_n,
        relation=sum(s["relation_all"] for s in scores) / relation_n if relation_n else None,
        relation_denominator=sum(s["relation_total"] for s in scores),
    )
    p, z = result["all"], 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["all_wilson95"] = [float(center - half), float(center + half)]
    return result


def paired_difference(
    correct: list[int],
    shuffled: list[int],
    seed: int = 0,
) -> JsonDict:
    """对相同请求与初始噪声的二元结果做配对 bootstrap。

    Args:
        correct: 正确条件下逐图成功位。
        shuffled: 与其逐项对齐的打乱条件成功位。
        seed: 独立 NumPy 发生器种子。

    Raises:
        ValueError: 数量不一致、为空或不是二元标记。

    Results:
        JsonDict: 成功率差、样本级 95% 区间及配对数；不能替代训练重复。
    """
    if (
        len(correct) != len(shuffled)
        or not correct
        or set(itertools.chain(correct, shuffled)) - {0, 1}
    ):
        raise ValueError("Aligned nonempty binary outcomes required")
    delta = np.asarray(correct, dtype=float) - np.asarray(shuffled, dtype=float)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(-1)
    return {
        "difference": float(delta.mean()),
        "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "pairs": len(delta),
        "limitation": "sample-level only; report each training seed separately",
    }


def prepare_blind_review(
    rows: list[JsonDict],
    directory: "Path",
) -> "Path":
    """生成隐藏模型/条件标签和解析结果的独立人工审核包，保护已有人工填写。

    Args:
        rows: 机器分析记录，image_path 相对 directory，含 request/parser/review_id。
        directory: 机器图像目录；blind 子目录只暴露匿名图像与人工表。

    Raises:
        ValueError: ID 重复，或已有审核包对应的图像/请求身份改变。
        OSError: 图像或清单不可读写时透传。

    Results:
        Path: blind/human_review.jsonl；解析与来源映射另存 review_key.json。

    副作用:
        复制匿名图像，创建机器私有映射和随机排序的人工表；已有同身份表不覆盖。
    """
    import hashlib
    import json
    import random
    import shutil

    from minigen.autoencoders.frozen import hash_file
    from minigen.data.latent import identity_digest

    if len({r["review_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source review IDs")
    key_rows = []
    for row in rows:
        digest = identity_digest(
            {
                "request": row["request"],
                "image": hash_file(directory / row["image_path"]),
                "id": row["review_id"],
            }
        )
        opaque = hashlib.sha256(digest.encode()).hexdigest()[:16]
        key_rows.append(
            {
                "opaque_id": opaque,
                "image_sha256": hash_file(directory / row["image_path"]),
                "record": row,
            }
        )
    key = {"schema_version": 1, "rows": key_rows}
    key_path = directory / "review_key.json"
    target = directory / "blind/human_review.jsonl"
    if key_path.exists():
        if json.loads(key_path.read_text()) != key or not target.exists():
            raise ValueError("Review identity changed; use a new run directory")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    public_rows = []
    for row in key_rows:
        name = row["opaque_id"] + ".png"
        shutil.copyfile(directory / row["record"]["image_path"], target.parent / name)
        public_rows.append(
            {
                "review_id": row["opaque_id"],
                "image_path": name,
                "request": row["record"]["request"],
                "human": None,
                "reviewer": None,
                "uncertain": None,
            }
        )
    random.Random(20261105).shuffle(public_rows)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in public_rows) + "\n")
    return target


def merge_human_review(
    review_file: "Path",
    key_file: "Path",
) -> list[JsonDict]:
    """先校验匿名审核的图像和请求，再由 Agent 合并隐藏解析结果。

    Args:
        review_file: 用户只填写 human/reviewer/uncertain 的匿名 JSONL。
        key_file: Agent 保存的源图像、来源分组与解析结果映射。

    Raises:
        ValueError: ID 缺失/重复、请求或图像身份被修改。
        OSError/KeyError: 文件或字段缺失时透传。

    Results:
        list[JsonDict]: 还原来源 ID 并附真实人工标签的记录，可送给门槛检查。
    """
    import json

    from minigen.autoencoders.frozen import hash_file

    rows = [json.loads(line) for line in review_file.read_text().splitlines() if line.strip()]
    keys = {r["opaque_id"]: r for r in json.loads(key_file.read_text())["rows"]}
    if len({r["review_id"] for r in rows}) != len(rows) or {r["review_id"] for r in rows} != set(
        keys
    ):
        raise ValueError("Missing, duplicate or unknown anonymous review IDs")
    merged = []
    for row in rows:
        key = keys[row["review_id"]]
        if (
            row["image_path"] != row["review_id"] + ".png"
            or row["request"] != key["record"]["request"]
            or hash_file(review_file.parent / row["image_path"]) != key["image_sha256"]
        ):
            raise ValueError("Review image or request identity changed")
        merged.append(
            {
                **key["record"],
                "human": row.get("human"),
                "reviewer": row.get("reviewer"),
                "uncertain": row.get("uncertain"),
            }
        )
    return merged
