"""门槛通过后才打开 S3 测试；显式教学模式，不提供绕过门槛选项。"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from minigen.autoencoders.frozen import FrozenVAE, hash_file, to_image
from minigen.conditioning.text import FrozenTextEncoder
from minigen.data.icons import load_manifest
from minigen.evaluation.conditional import (
    merge_human_review,
    prepare_blind_review,
    score_request,
    summarize,
)
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.samplers.conditional import sample_conditional

ROOT = Path(__file__).resolve().parents[1]


def review_gate(
    rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    """检查至少 100 原图与 100 生成/退化图的独立人工标签和逐属性一致性。

    Args:
        rows: 盲审 JSONL 记录，human 为人工填写的二元属性字典，reviewer 非空。

    Raises:
        KeyError/TypeError: 评分记录缺必要结构时透传。

    Results:
        tuple[bool,str]: 是否通过以及具体原因；阈值是本轮预定的工程门槛。
    """
    if len({r["review_id"] for r in rows}) != len(rows):
        return False, "Duplicate review IDs are not independent reviewed images"
    fields = ("count", "color", "shape", "binding", "all", "parse_failure")
    for original in (True, False):
        group = [r for r in rows if r["review_id"].startswith("original_") == original]
        reviewed = [
            r
            for r in group
            if isinstance(r.get("human"), dict)
            and not r.get("uncertain")
            and r.get("reviewer")
            and all(type(r["human"].get(k)) is int and r["human"][k] in (0, 1) for k in fields)
        ]
        if len(reviewed) < 100:
            return False, "Need 100 reviewed originals and 100 generated/degraded images"
        for key in fields:
            if sum(r["parser"][key] == r["human"][key] for r in reviewed) / len(reviewed) < 0.95:
                return False, f"Human/parser agreement below 95%: {key}"
        relation_rows = [r for r in reviewed if r["parser"].get("has_relation")]
        if any(
            type(r["human"].get("relation_all")) is not int
            or r["human"]["relation_all"] not in (0, 1)
            for r in relation_rows
        ):
            return False, "Review relation_all for all relation-bearing requests"
        if (
            relation_rows
            and sum(
                r["parser"]["relation_all"] == r["human"]["relation_all"] for r in relation_rows
            )
            / len(relation_rows)
            < 0.95
        ):
            return False, "Human/parser relation agreement below 95%"
    return True, "review counts and per-attribute agreement passed"


def reference_namespace() -> tuple[dict[str, Any], str]:
    """只提取被授权的标记单元，不执行 Notebook 训练；返回新条件核心哈希。

    Raises:
        ValueError: 标记单元不唯一。
        OSError/JSONDecodeError: 笔记不可读时透传。

    Results:
        tuple: 教学类/函数命名空间及条件源码 SHA-256。
    """
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    digest = ""
    for name, tags in [
        ("S2_第3周_图标数据与潜空间DiT基础.ipynb", ["dit_reference"]),
        ("S3_文本条件CFG与可信评测.ipynb", ["conditional_reference", "cfg_reference"]),
    ]:
        notebook = json.loads((ROOT / "docs/learning" / name).read_text())
        for tag in tags:
            found = [
                "".join(c["source"])
                for c in notebook["cells"]
                if tag in c.get("metadata", {}).get("tags", [])
            ]
            if len(found) != 1:
                raise ValueError("Expected a unique reference cell")
            source = found[0]
            exec(compile(source, "<" + tag + ">", "exec"), namespace)
            if tag == "dit_reference":
                namespace["base_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            if tag == "cfg_reference":
                namespace["cfg_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            if tag == "conditional_reference":
                digest = hashlib.sha256(source.encode()).hexdigest()
    return namespace, digest


def main() -> int:
    """先检查真实门槛，满足后加载明确教学权重评测三个封存划分。

    Raises:
        ValueError: 来源、配置、完成状态或身份不一致。
        OSError/JSONDecodeError/模型异常: 文件和执行失败时透传。

    Results:
        int: 未过门槛为 2；完整正式执行为 0；没有隐式跳过或伪造结果。

    副作用:
        检查个人核心的 CPU 子进程；门槛通过后才读取测试清单并编码测试描述、采样及保存结果。
        不下载、不收费；教学权重仍明确记录 notebook_reference。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--prepare-review", action="store_true")
    parser.add_argument("--notebook-reference", action="store_true", required=True)
    args = parser.parse_args()
    configuration = yaml.safe_load((ROOT / "configs/experiments/s3.yaml").read_text())
    result = json.loads((ROOT / "reports/stage3/results.json").read_text())
    selection = json.loads((args.run_dir / "selection_frozen.json").read_text())
    configuration_hash = hash_file(ROOT / "configs/experiments/s3.yaml")
    expected_run = ROOT / "artifacts/runs" / configuration["run_id"]
    if (
        args.run_dir.resolve() != expected_run.resolve()
        or configuration_hash != result["config_sha256"]
        or configuration_hash != selection["config_sha256"]
    ):
        raise ValueError("Run or frozen selection configuration changed")
    if args.prepare_review:
        machine_file = args.run_dir / "review/blind_review.jsonl"
        machine_rows = [
            json.loads(line) for line in machine_file.read_text().splitlines() if line.strip()
        ]
        path = prepare_blind_review(machine_rows, args.run_dir / "review")
        print(json.dumps({"status": "prepared", "human_review": str(path), "test_opened": False}))
        return 0
    if args.review is None:
        args.review = args.run_dir / "review/blind/human_review.jsonl"
    gains = [r for r in result["controls"] if "paired_gain" in r]
    if len(gains) != 2 or any(r["paired_gain"]["difference"] <= 0 for r in gains):
        print(
            json.dumps(
                {
                    "status": "待做",
                    "reason": "Positive conditional gain in both training seeds is required",
                    "test_opened": False,
                }
            )
        )
        return 2
    rows = merge_human_review(args.review, args.review.parent.parent / "review_key.json")
    passed, reason = review_gate(rows)
    if not passed:
        print(json.dumps({"status": "待做", "reason": reason, "test_opened": False}))
        return 2
    for script in ("check_dit_core.py", "check_conditioning_core.py", "check_cfg_core.py"):
        checked = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=False
        )
        if checked.returncode:
            print(json.dumps({"status": "待做", "reason": script, "test_opened": False}))
            return 2
    namespace, digest = reference_namespace()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = FrozenTextEncoder.load(ROOT, ROOT / "configs/models/clip_text.json", device)
    vae = FrozenVAE.load(ROOT, ROOT / "configs/models/sdxl_vae.json", device)
    records = load_manifest(ROOT / "data/manifests/synthetic_icons_week3_v1.jsonl", ROOT / "data")
    empty, empty_mask, _ = encoder.encode([""])
    empty, empty_mask = empty.half().float().to(device), empty_mask.to(device)
    output = []
    for seed in configuration["training"]["seeds"]:
        saved = torch.load(
            args.run_dir / f"seed_{seed}/latest.pt", map_location="cpu", weights_only=True
        )
        if (
            saved["contract"]["model"]["declared"]["source_sha256"] != digest
            or saved["contract"]["model"]["declared"]["base_dit_sha256"]
            != namespace["base_source_sha256"]
            or saved["contract"]["identity"]["drop_source_sha256"] != namespace["cfg_source_sha256"]
        ):
            raise ValueError(
                "Teaching source changed; cannot evaluate old weights as current model"
            )
        model = namespace["ConditionalDiTReference"]().to(device).eval()
        model.load_state_dict(saved["ema"], strict=True)
        for split in configuration["holdout"]["splits"]:
            requests = [r for r in records if r["split"] == split][
                : configuration["holdout"]["prompts_per_split"]
            ]
            text, mask, _ = encoder.encode([r["caption"] for r in requests])
            initial = torch.randn(
                (len(requests), 4, 16, 16), generator=torch.Generator().manual_seed(41)
            ).to(device)
            generated = sample_conditional(
                model,
                DDPMScheduler(1000).to(device),
                initial,
                text.half().float().to(device),
                mask.to(device),
                empty,
                empty_mask,
                namespace["cfg_reference"],
                method="ddim",
                steps=40,
                scale=selection["selected_scale"],
            )
            if not generated["completed"]:
                raise ValueError("Partial sampling cannot be scored as final generation")
            images = [to_image(image) for image in vae.decode(generated["images"])]
            scores = [
                score_request(image, record) for image, record in zip(images, requests, strict=True)
            ]
            image_dir = args.run_dir / "holdout" / f"seed_{seed}" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                image.save(image_dir / f"{index:03d}.png")
            (image_dir / "requests_and_scores.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": record["sample_id"],
                            "prompt": record["caption"],
                            "score": score,
                        }
                        for record, score in zip(requests, scores, strict=True)
                    ],
                    indent=2,
                )
                + "\n"
            )
            output.append(
                {
                    "seed": seed,
                    "split": split,
                    "metrics": summarize(scores),
                    "sample_ids": [r["sample_id"] for r in requests],
                    "source": "notebook_reference",
                }
            )
    target = args.run_dir / "holdout_results.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
