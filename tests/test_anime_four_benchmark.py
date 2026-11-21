"""仅供远端 CPU 执行的四组评测合成测试，不使用真实权重、缓存或评分。"""

import hashlib
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import evaluate_anime_four_benchmark as bench
import pytest
from PIL import Image

from minigen._typing import JsonDict


def write_json(
    path: Path,
    value: JsonDict,
) -> None:
    """在测试临时目录写入合成 JSON。

    Args:
        path: pytest 临时根目录下的目标文件。
        value: 合成元数据。

    副作用:
        创建父目录并写入 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def mock_hash(
    label: str,
) -> str:
    """根据测试标签生成与真实权重无关的确定摘要。

    Args:
        label: 标识合成来源或噪声的字符串。

    Results:
        str: 合成标签的 SHA-256。
    """
    return hashlib.sha256(f"SYNTHETIC_ONLY:{label}".encode()).hexdigest()


def synthetic_benchmark(
    sampling: JsonDict,
) -> JsonDict:
    """生成用于验证分母的合成提示词、来源分组与单属性替换题。

    Args:
        sampling: 评测范围规定的种子之外的共同采样参数。

    Results:
        JsonDict: 80 条常规题和 24 条替换题，每个初噪有 76 个虹膜端点和 64 个眼色端点。
    """
    regular_templates = (
        (24, "eyes", {"eye_color": "blue eyes"}),
        (20, "hair", {"hair_color": "silver hair"}),
        (20, "accessories", {"accessory": "paper crown"}),
        (8, "closure", {"eye_state": "closed eyes"}),
        (8, "closure", {"eye_state": "one eye closed"}),
    )
    regular_specs = [
        (category, targets) for count, category, targets in regular_templates for _ in range(count)
    ]
    regular = [
        {
            "id": f"SYN-R{index:03d}",
            "category": category,
            "prompt": f"Synthetic regular {index}: {', '.join(targets.values())}",
            "targets": targets,
            "source": {"group_id": mock_hash(f"regular-group:{index}")},
        }
        for index, (category, targets) in enumerate(regular_specs, start=1)
    ]

    pair_templates = (
        (16, "eyes", "eye_color", "blue eyes", "green eyes", {}),
        (4, "hair", "hair_color", "silver hair", "pink hair", {"eye_color": "blue eyes"}),
        (2, "closure", "eye_state", "closed eyes", "one eye closed", {}),
        (2, "closure", "eye_state", "one eye closed", "closed eyes", {}),
    )
    pairs = []
    for count, category, attribute, before, after, unchanged in pair_templates:
        for _ in range(count):
            index = len(pairs) + 1
            targets_a = unchanged | {attribute: before}
            targets_b = unchanged | {attribute: after}
            pairs.append(
                {
                    "id": f"SYN-P{index:03d}",
                    "category": category,
                    "prompt_a": f"Synthetic pair {index}: {', '.join(targets_a.values())}",
                    "prompt_b": f"Synthetic pair {index}: {', '.join(targets_b.values())}",
                    "targets_a": targets_a,
                    "targets_b": targets_b,
                    "changed": {"attribute": attribute, "from": before, "to": after},
                    "source": {"group_id": mock_hash(f"pair-group:{index}")},
                }
            )
    return {
        "schema": "minigen.anime_faces.benchmark.v1",
        "seeds": list(bench.SEEDS),
        "sampling": sampling,
        "regular": regular,
        "pairs": pairs,
    }


@pytest.fixture
def mock_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """创建合成题单、1,024 张硬链接纯色图及四份合成回执。

    Args:
        tmp_path: pytest 隔离的临时目录，不使用正式输出路径。
        monkeypatch: 仅将此测试的 bootstrap 次数降低为 20。

    Results:
        tuple: 合成项目根目录与明确标注 SYNTHETIC_ONLY 的输出目录。

    副作用:
        读取评测范围和评分量表，在临时目录写入合成题单、配置、图像和回执。
    """
    monkeypatch.setattr(bench, "BOOTSTRAP_REPLICATES", 20)
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "mock_project"
    scope = json.loads((project / bench.SCOPE).read_text())
    rubric = Path(scope["rubric"])
    (root / rubric).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(project / rubric, root / rubric)
    benchmark = synthetic_benchmark(scope["sampling"])
    write_json(root / bench.BENCHMARK, benchmark)
    scope["benchmark_sha256"] = bench.digest(root / bench.BENCHMARK)
    scope["comparison"] = "SYNTHETIC_ONLY_four_benchmark_fixture"
    scope["evidence_boundary"] = "Synthetic fixture; never use as formal model evidence."
    for arm, spec in scope["candidates"].items():
        spec["run_id"] = mock_hash(f"run:{arm}")
        spec["checkpoint_sha256"] = mock_hash(f"checkpoint:{arm}")
        spec["source_sha256"] = mock_hash("package")
    write_json(root / bench.SCOPE, scope)
    output = root / "artifacts/exports/SYNTHETIC_ONLY_four_benchmark"
    output.mkdir(parents=True)
    (output / "MOCK_ONLY.txt").write_text(
        "Synthetic fixture. Not trained or evaluated model output.\n"
    )
    questions = [("regular", row) for row in benchmark["regular"]]
    questions += [("pair", row) for row in benchmark["pairs"]]
    for arm_index, arm in enumerate(bench.ARMS):
        folder = output / "generation" / arm
        folder.mkdir(parents=True)
        prototype = root / f"synthetic_image_{arm_index}.png"
        Image.new("RGB", (256, 256), (40 + 35 * arm_index, 80, 170)).save(prototype)
        records = []
        for seed in bench.SEEDS:
            for noise_index, (kind, question) in enumerate(questions):
                labels = (None,) if kind == "regular" else ("a", "b")
                for label in labels:
                    suffix = "" if label is None else f"_{label}"
                    filename = f"{question['id']}_{seed}{suffix}.png"
                    (folder / filename).hardlink_to(prototype)
                    records.append(
                        {
                            "id": question["id"],
                            "seed": seed,
                            "endpoint": label,
                            "prompt": question["prompt" if label is None else f"prompt_{label}"],
                            "noise_index": noise_index,
                            "noise_sha256": mock_hash(f"noise:{seed}:{question['id']}"),
                            "image_path": filename,
                            "image_sha256": bench.digest(prototype),
                        }
                    )
        spec = scope["candidates"][arm]
        write_json(
            folder / "receipt.json",
            {
                "schema": "minigen.anime_faces.benchmark_generation.v1",
                "status": "completed",
                "fixture_only": True,
                "arm": arm,
                "checkpoint_step": 30000,
                "cumulative_training_updates": 30000,
                "weights": "ema",
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "run_id": spec["run_id"],
                "trainable_parameters": spec["parameters"],
                "model_config": spec["model_config"],
                "source_sha256": spec["source_sha256"],
                "comparison_scope_sha256": bench.digest(root / bench.SCOPE),
                "benchmark_sha256": scope["benchmark_sha256"],
                "rubric_sha256": scope["rubric_sha256"],
                "approval_sha256": scope["approval_sha256"],
                "benchmark_lock_sha256": scope["benchmark_lock_sha256"],
                "seeds": list(bench.SEEDS),
                "sampling": benchmark["sampling"],
                "seed_noise_sha256": {
                    str(seed): mock_hash(f"noise:{seed}") for seed in bench.SEEDS
                },
                "data_identity": {"fixture_only": True},
                "evaluation_torch": "synthetic-no-torch-used",
                "sampler_sha256": mock_hash("sampler"),
                "sampler_helper_sha256": mock_hash("sampler-helper"),
                "condition_features_sha256": mock_hash("condition-features"),
                "condition_mask_sha256": mock_hash("condition-mask"),
                "condition_encoding": "cpu_clip_fp16_roundtrip",
                "sampling_batch_size": 16,
                "decode_batch_size": 4,
                "tf32": False,
                "records": records,
            },
        )
    return root, output


def fill_synthetic_scores(
    output: Path,
) -> Path:
    """依据已知合成身份构造可手算的分数，不用于任何真实盲评。

    Args:
        output: 只包含本测试纯色图的临时输出目录。

    Results:
        Path: 完整合成评分文件。

    副作用:
        读取测试生成的私有映射，在临时目录写合成评分。
    """
    deck = bench.read_json(output / "review/deck.json")
    key = bench.read_json(output / "private/key.json")
    scores = bench.read_json(output / "review/score_template.json")
    scores["reviewer"] = "SYNTHETIC_ONLY_known_answer_fixture"
    for card in deck["items"]:
        assignment = key["assignments"][card["id"]]
        usable = {}
        for column in deck["columns"]:
            arm = assignment[column]
            usable[column] = arm in bench.ARMS[2:] or (
                arm == bench.ARMS[1] and assignment["seed"] == bench.SEEDS[0]
            )
            rated = scores["items"][card["id"]][column]
            for point in card["endpoints"]:
                rated["endpoints"][point["key"]] = {
                    "face": "usable" if usable[column] else "distorted",
                    "iris": "readable" if point["iris_required"] else "not_applicable",
                    "targets": {name: "correct" for name in point["targets"]},
                    "defects": {
                        name: "absent" if usable[column] else "present" for name in bench.DEFECTS
                    },
                }
            if card["kind"] == "pair":
                rated["preservation"] = "preserved"
        scores["items"][card["id"]]["preferences"] = {
            f"{one}-{two}": ("tie" if usable[one] == usable[two] else one if usable[one] else two)
            for one, two in itertools.combinations(deck["columns"], 2)
        }
    path = output / "review/scores.json"
    write_json(path, scores)
    return path


@pytest.mark.parametrize("count", (5, 6))
def test_ablation_uses_only_registered_pairs(
    mock_benchmark: tuple[Path, Path],
    count: int,
) -> None:
    """核对新增列的锁分、配对解盲与单初噪失败门槛。

    Args:
        mock_benchmark: 原四组纯色图和回执组成的隔离根目录。
        count: 指定本轮匿名题卡列数。

    Raises:
        AssertionError: 列覆盖、固定配对或主指标判定不一致。

    副作用:
        在隔离目录增加纯色图并生成匿名评分和报告。
    """
    root, output = mock_benchmark
    original_scope, benchmark, receipts, _ = bench.verified_receipts(root, output)
    first, second, third, fourth = bench.ARMS
    extra_arms = tuple(f"synthetic_extra_{index}" for index in range(count - len(bench.ARMS)))
    for arm in extra_arms:
        shutil.copytree(output / "generation" / fourth, output / "generation" / arm)
        receipts[arm] = receipts[fourth] | {
            "arm": arm,
            "sampling": benchmark["sampling"] | {"cfg": 5},
        }
    scope = original_scope.copy()
    scope["candidates"] = original_scope["candidates"] | dict.fromkeys(extra_arms, {})
    scope["baseline_pairs"] = [[first, second], [first, third], [fourth, extra_arms[0]]]
    scope_path = Path("configs/benchmark/synthetic_ablation.json")
    if count == 6:
        scope["report_title"] = "Synthetic CFG pair test"
    write_json(root / scope_path, scope)
    bench.render(
        root, output, prepared=(scope, benchmark, receipts, bench.digest(root / scope_path))
    )
    scores = fill_synthetic_scores(output)
    bench.freeze(output, scores)
    bench.reveal(root, output, scope_path=scope_path)
    bench.report(root, output, scope_path=scope_path)
    result = bench.read_json(output / "review/deblinded_metrics.json")
    assert result["locked_review"]["counts"]["images"] == 256 * count
    assert result["locked_review"]["counts"]["iris"] == 152 * count
    assert set(result["comparisons"]) == {f"{one}__{two}" for one, two in scope["baseline_pairs"]}
    one_seed = result["comparisons"][f"{first}__{second}"]["metrics"]["face"]
    both_seeds = result["comparisons"][f"{first}__{third}"]["metrics"]["face"]
    assert one_seed["net"] == 128 and one_seed["passes"] is False
    assert both_seeds["net"] == 256 and both_seeds["passes"] is True
    html = (output / "index.html").read_text()
    assert "配对提升门槛" in html
    if count == 6:
        assert "Synthetic CFG pair test" in html
        assert "CFG 3 / 5" in html
        assert result["sampling_by_arm"][extra_arms[-1]]["cfg"] == 5


def test_synthetic_pipeline_preserves_pairing_denominators_and_tied_top_two(
    mock_benchmark: tuple[Path, Path],
) -> None:
    """以可手算分数验收匿名化、锁分、解盲、配对统计和报告完整闭环。

    Args:
        mock_benchmark: 没有真实权重或数据的临时项目与合成输出目录。

    Raises:
        AssertionError: 列位、分母、已知模型排名或产物覆盖错误。

    副作用:
        在 pytest 临时目录渲染和锁定合成评分，生成合成统计与对照图。
    """
    root, output = mock_benchmark
    bench.render(root, output)
    deck = bench.read_json(output / "review/deck.json")
    key_path = output / "private/key.json"
    key = bench.read_json(key_path)
    public = (output / "review/deck.json").read_text()
    document = (output / "review/blind_review.html").read_text()
    for arm, identity in key["checkpoint_identities"].items():
        for secret in (arm, identity["run_id"], identity["checkpoint_sha256"]):
            assert secret not in public and secret not in document
    placements = Counter(
        (row[column], column) for row in key["assignments"].values() for column in bench.COLUMNS
    )
    assert set(placements.values()) == {52} and len(placements) == 16
    question_layouts = {}
    for row in key["assignments"].values():
        layout = tuple(row[column] for column in bench.COLUMNS)
        assert question_layouts.setdefault(row["question_id"], layout) == layout
    assert len(list((output / "review/cards").glob("*.png"))) == 208
    scores_path = fill_synthetic_scores(output)
    with pytest.raises(FileNotFoundError, match="scores.lock.json"):
        bench.reveal(root, output)
    hidden = key_path.with_suffix(".hidden")
    key_path.rename(hidden)
    bench.freeze(output, scores_path)
    hidden.rename(key_path)
    assert bench.validate_scores(deck, bench.read_json(scores_path)) == {
        "cards": 208,
        "images": 1024,
        "iris": 608,
        "eye_color": 512,
        "swaps": 192,
    }
    bench.reveal(root, output)
    bench.report(root, output)
    result = bench.read_json(output / "review/deblinded_metrics.json")
    assert set(result["ranking"]["top_two_with_cutoff_ties"]) == set(bench.ARMS[2:])
    assert [row["rank"] for row in result["ranking"]["ordered"]] == [1, 1, 3, 4]
    assert len(result["comparisons"]) == len(result["pairwise_preferences"]) == 6
    expected_faces = dict(zip(bench.ARMS, (0, 128, 256, 256), strict=True))
    for arm, model in result["models"].items():
        assert model["overall"]["face"]["success"] == expected_faces[arm]
        for metric, denominator in (
            ("face", 256),
            ("iris", 152),
            ("eye_color", 128),
            ("swap_joint", 48),
        ):
            assert model["overall"][metric]["applicable"] == denominator
            assert all(
                model["by_seed"][str(seed)][metric]["applicable"] == denominator // 2
                for seed in bench.SEEDS
            )
    pair = result["comparisons"][f"{bench.ARMS[0]}__{bench.ARMS[2]}"]["metrics"]["face"]
    assert (pair["improved"], pair["worsened"], pair["net"]) == (256, 0, 256)
    assert all(row["net"] == 128 for row in pair["by_seed"].values())
    for preference in result["pairwise_preferences"].values():
        assert sum(preference["overall"].values()) == 208
        assert all(sum(row.values()) == 104 for row in preference["by_seed"].values())
    assert result["bootstrap"]["groups"] == 104 and result["bootstrap"]["replicates"] == 20
    assert len(list((output / "comparisons").glob("*.png"))) == 208
    assert (output / "index.html").is_file() and "SYNTHETIC_ONLY" in str(output)
    assert "Synthetic fixture" in (output / "MOCK_ONLY.txt").read_text()


@pytest.mark.parametrize("changed", ["scores", "mapping"])
def test_locked_inputs_reject_changed_scores_or_mapping(
    mock_benchmark: tuple[Path, Path],
    changed: str,
) -> None:
    """锁分后改变分数或映射必须失败，不能生成貌似有效的解盲成绩。

    Args:
        mock_benchmark: 隔离的合成项目与输出目录。
        changed: 分别篡改评分或私有映射。

    Raises:
        AssertionError: 篡改后的数据仍发布了解盲统计。

    副作用:
        在测试临时目录渲染并锁定合成分数，随后刻意篡改一个输入。
    """
    root, output = mock_benchmark
    bench.render(root, output)
    scores_path = fill_synthetic_scores(output)
    bench.freeze(output, scores_path)
    path = scores_path if changed == "scores" else output / "private/key.json"
    path.write_text(path.read_text() + " \n")
    with pytest.raises(ValueError, match="changed after"):
        bench.reveal(root, output)
    assert not (output / "review/deblinded_metrics.json").exists()


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_render_rejects_missing_or_changed_generation(
    mock_benchmark: tuple[Path, Path],
    damage: str,
) -> None:
    """缺图或图像内容偏离回执时，不允许发布正式匿名评测包。

    Args:
        mock_benchmark: 隔离合成项目与输出目录。
        damage: 删除一张图，或将其替换成未登记的 PNG。

    Raises:
        AssertionError: 损坏输入仍生成完成标记。

    副作用:
        破坏测试临时目录中一张图，并尝试渲染不完整输入。
    """
    root, output = mock_benchmark
    folder = output / "generation" / bench.ARMS[0]
    path = folder / bench.read_json(folder / "receipt.json")["records"][0]["image_path"]
    path.unlink()
    if damage == "changed":
        Image.new("RGB", (256, 256), "red").save(path)
    with pytest.raises((FileNotFoundError, ValueError)):
        bench.render(root, output)
    assert not (output / "review/export_manifest.json").exists()


def test_unpublished_render_recovers_without_overwriting_published_review(
    mock_benchmark: tuple[Path, Path],
) -> None:
    """渲染中断残留可以重建，已发布匿名页则保持原有身份。

    Args:
        mock_benchmark: 隔离的合成项目与输出目录。

    Raises:
        AssertionError: 中断无法恢复，或正式评测被重新盲化覆盖。

    副作用:
        创建不完整渲染残留并重建合成评分包。
    """
    root, output = mock_benchmark
    review = output / "review"
    review.mkdir()
    (review / "incomplete.txt").write_text("unfinished mock render")
    write_json(output / "private/key.json", {"unfinished": True})
    bench.render(root, output)
    assert not (review / "incomplete.txt").exists()
    expected = bench.digest(review / "deck.json")
    with pytest.raises(FileExistsError, match="Published or scored"):
        bench.render(root, output)
    assert bench.digest(review / "deck.json") == expected
