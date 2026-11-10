"""评分清单完整性和可恢复报告测试；显式模拟特征不代表真实质量成绩。"""

import json
from pathlib import Path

import pytest
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

from minigen.autoencoders.sdxl import hash_file
from minigen.evaluation.features import FeatureSuite
from minigen.evaluation.scoring import run_scoring


class PixelMeanFeatures:
    """用像素均值检查提取器的批顺序和缓存，不代表真实 Inception。"""

    def __call__(
        self,
        images: UInt8[Tensor, "b c h w"],
    ) -> tuple[Float[Tensor, "b 2048"]]:
        """按图像内容生成确定性、批大小无关的测试向量。

        Args:
            images: 原生 uint8 图像批。

        Results:
            tuple: 一个 [b,2048] 浮点张量。
        """
        return (images.float().mean((1, 2, 3))[:, None].expand(-1, 2048),)


def test_feature_batch_order_cache_and_corruption(
    tmp_path: Path,
) -> None:
    """真实提取循环在不同批大小、重复图像与缓存命中下保持顺序。

    Args:
        tmp_path: 独立图片和缓存目录。

    Raises:
        AssertionError: 输出顺序变化或损坏缓存未拒绝。

    副作用:
        写入测试图片、身份与缓存；模型使用显式测试替身。
    """
    lock = tmp_path / "feature-lock.json"
    lock.write_text('{"models":{"inception":{"fixture":"pixelmean"}}}')
    paths = []
    for i in range(3):
        path = tmp_path / f"f{i}.png"
        Image.new("RGB", (256, 256), (30 * i, 30 * i, 30 * i)).save(path)
        paths.append(path)
    suite = FeatureSuite(tmp_path, lock, batch_size=2, cache_dir=tmp_path / "cache")
    suite.models["inception"] = PixelMeanFeatures()
    first = suite.inception([paths[2], paths[0], paths[1], paths[2]])
    assert first[:, 0].tolist() == [60, 0, 30, 60]
    suite.batch_size = 1
    torch.testing.assert_close(suite.inception(paths), first[[1, 2, 0]])
    files = list((tmp_path / "cache").rglob("*.pt"))
    assert len(files) == 3
    torch.save(torch.ones(2), files[0])
    with pytest.raises(ValueError, match="cached feature"):
        suite.inception(paths)


def test_formal_scores_require_approval_and_distribution_identity(
    tmp_path: Path,
) -> None:
    """改 scope 不能把烟测变成正式结果，正式分布任务还须满足固定规模。

    Args:
        tmp_path: 独立生成和参考目录。

    Raises:
        AssertionError: 未审批或非分布任务被正式评分接受。

    副作用:
        创建并修改模拟回执以验证拒绝路径。
    """
    generation, reference, assets = scoring_fixture(tmp_path)
    payload = json.loads(generation.read_text())
    payload["scope"] = "formal"
    generation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="approved generation"):
        run_scoring(tmp_path, generation, reference, tmp_path / "out.json", assets, scope="formal")


def test_formal_prompt_content_must_match_frozen_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使回执已有审批标志与哈希，改写实际题目仍须拒绝。

    Args:
        tmp_path: 独立模拟目录。
        monkeypatch: 替换 Benchmark 加载器，只检查评分任务校验层。

    Raises:
        AssertionError: 生成清单中的描述篡改未被拒绝。

    副作用:
        写测试回执；不提取特征或生成正式成绩。
    """
    from minigen.evaluation.benchmark import content_hash

    generation, reference, assets = scoring_fixture(tmp_path)
    definition = json.loads(reference.read_text())
    payload = json.loads(generation.read_text())
    tasks = [{"sample_id": r["sample_id"], "caption": r["caption"]} for r in payload["records"]]

    def load_fixture(
        path: Path,
    ) -> dict[str, object]:
        """返回已知的测试协议以隔离内容校验。

        Args:
            path: 待加载路径。

        Results:
            dict: 固定测试内容。
        """
        return definition

    def fixture_tasks(
        benchmark: dict[str, object],
        split: str,
        count: int | None,
    ) -> list[dict[str, str]]:
        """保留原始题目，模拟冻结任务生成器。

        Args:
            benchmark: 测试协议。
            split: 测试用途。
            count: 正式请求不截断。

        Results:
            list: 原始有序样本与描述。
        """
        return tasks

    monkeypatch.setattr("minigen.evaluation.benchmark.load_benchmark", load_fixture)
    monkeypatch.setattr("minigen.evaluation.generation.make_tasks", fixture_tasks)
    payload["scope"] = "formal"
    payload["identity"].update(
        {
            "approval_verified": True,
            "approval_sha256": "a" * 64,
            "benchmark_sha256": hash_file(reference),
            "benchmark_content_sha256": content_hash(definition),
            "split": "test",
        }
    )
    payload["records"][0]["caption"] = "A different caption."
    generation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen task list"):
        run_scoring(
            tmp_path,
            generation,
            reference,
            tmp_path / "out.json",
            assets,
            metrics=("clipscore",),
            scope="formal",
        )
    payload["identity"].update({"approval_verified": True, "approval_sha256": "a" * 64})
    generation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="2000 distribution"):
        run_scoring(tmp_path, generation, reference, tmp_path / "out.json", assets, scope="formal")


class FixtureFeatures:
    """用确定性小特征替代昂贵提取器，仅测试编排和身份。

    Args:
        root: 测试根。
        asset_lock: 测试资产锁。
        device: 请求设备。
        batch_size: 请求批量。
    """

    def __init__(
        self,
        root: Path,
        asset_lock: Path,
        device: str,
        batch_size: int,
    ) -> None:
        """记录测试范围并建立空截断信息。

        Args:
            root: 测试根。
            asset_lock: 测试资产锁。
            device: 测试设备。
            batch_size: 测试批量。
        """
        self.text_lengths: list[dict[str, object]] = []

    def inception(
        self,
        paths: list[Path],
    ) -> Float[Tensor, "n 3"]:
        """返回顺序敏感的固定特征用于检查子集计算。

        Args:
            paths: 有序图片列表。

        Results:
            Float[Tensor, "n 3"]: 模拟特征，不是 Inception 输出。
        """
        return torch.arange(len(paths) * 3, dtype=torch.float32).reshape(-1, 3) + 1

    def clip_images(
        self,
        paths: list[Path],
    ) -> Float[Tensor, "n 3"]:
        """生成非零投影占位，不读取图像内容。

        Args:
            paths: 图片列表。

        Results:
            Float[Tensor, "n 3"]: 模拟投影。
        """
        return self.inception(paths)

    def clip_text(
        self,
        texts: list[str],
    ) -> Float[Tensor, "n 3"]:
        """返回固定文本特征并记录未截断。

        Args:
            texts: 有序描述。

        Results:
            Float[Tensor, "n 3"]: 模拟文本特征。

        副作用:
            更新测试截断清单。
        """
        self.text_lengths = [{"truncated": False} for _ in texts]
        return torch.ones(len(texts), 3)

    def close(self) -> None:
        """测试替身无需释放真实模型资源。"""


class FailingVQABackend(FixtureFeatures):
    """模拟真实后端执行错误，不将推理失败误判为未知答案或成功评分。"""

    simulated = False
    failure_count = 3

    def answer(
        self,
        paths: list[Path],
        questions: list[str],
    ) -> list[dict[str, str | None]]:
        """按测试指定数量返回失败，其余返回明确的 yes。

        Args:
            paths: 完整有效图像路径。
            questions: 与图像同序的对象问题。

        Results:
            list: 含原始执行错误或成功回答的逐问题回执。
        """
        return [
            {"answer": None, "error": "RuntimeError: simulated backend failure"}
            if i < self.failure_count
            else {"answer": "yes", "error": None}
            for i in range(len(questions))
        ]


@pytest.mark.parametrize("failure_count", [1, 3])
def test_vqa_execution_errors_keep_evidence_and_fail_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_count: int,
) -> None:
    """有效生成图不掩盖问答执行失败；保留问题分母且命令返回非零。

    Args:
        tmp_path: 临时模拟目录。
        monkeypatch: 显式替换真实 VQA 后端。
        failure_count: 本例失败问题数，覆盖部分及全部失败。

    Raises:
        AssertionError: 执行错误被标完成、分母丢失或命令退出码为零。

    副作用:
        写模拟评分回执并执行真实 CLI 分发，不运行真实 VQA 模型。
    """
    from minigen.bench_command import main

    generation, _, assets = scoring_fixture(tmp_path)
    monkeypatch.setattr("minigen.evaluation.scoring.BlipVQA", FailingVQABackend)
    monkeypatch.setattr(FailingVQABackend, "failure_count", failure_count)
    output = tmp_path / "vqa-error.json"
    code = main(
        [
            "score",
            "--root",
            str(tmp_path),
            "--generation",
            str(generation),
            "--output",
            str(output),
            "--asset-lock",
            str(assets),
            "--metrics",
            "vqa",
            "--scope",
            "simulated",
        ]
    )
    report = json.loads(output.read_text())
    metric = report["metrics"]["vqa"]
    assert code == 1 and report["complete"] is False
    assert report["coverage"]["valid"] == 3
    assert metric["complete"] is False
    assert metric["status"] == ("failed" if failure_count == 3 else "partial")
    assert len(metric["questions"]) == metric["categories"]["object"]["applicable"] == 3
    assert metric["categories"]["object"]["error"] == failure_count
    assert metric["value"] == pytest.approx((3 - failure_count) / 3)
    assert all(
        "simulated backend failure" in q["error"]
        for q in metric["questions"]
        if q["status"] == "error"
    )


def scoring_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """建立完整生成清单、验证参考及不含权重的模拟资产锁。

    Args:
        tmp_path: 测试根目录。

    Results:
        tuple: 生成、参考、资产锁路径。

    副作用:
        保存三张测试图与 JSON 文件。
    """
    records = []
    for i in range(3):
        image = tmp_path / f"image{i}.png"
        Image.new("RGB", (256, 256), (i * 70, 20, 30)).save(image)
        records.append(
            {
                "sample_id": f"s{i}",
                "prompt_id": f"p{i}",
                "seed": 0,
                "caption": "A painting of a horse.",
                "status": "completed",
                "image_path": image.name,
                "sha256": hash_file(image),
                "requirements": [{"kind": "object", "subject": "horse", "expected": "yes"}],
            }
        )
    generation, reference, assets = (
        tmp_path / n for n in ["samples.json", "benchmark.json", "assets.json"]
    )
    generation.write_text(
        json.dumps(
            {
                "scope": "simulated",
                "expected_samples": 3,
                "identity": {"architecture": "b0"},
                "records": records,
            }
        )
    )
    reference.write_text(
        json.dumps(
            {
                "data_root": ".",
                "validation_references": [{**r, "split": "validation"} for r in records],
            }
        )
    )
    assets.write_text('{"models": {}}')
    return generation, reference, assets


def test_scoring_resume_corruption_and_mock_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整报告可恢复，坏图仍在分母，模拟 VQA 的来源不被掩盖。

    Args:
        tmp_path: 独立运行目录。
        monkeypatch: 显式替换特征器的测试工具。

    Raises:
        AssertionError: 身份、完整性或失败分母错误。

    副作用:
        写图片、清单和指标回执；全流程为模拟验收。
    """
    monkeypatch.setattr("minigen.evaluation.scoring.FeatureSuite", FixtureFeatures)
    generation, reference, assets = scoring_fixture(tmp_path)
    output = tmp_path / "score.json"
    options = {
        "root": tmp_path,
        "generation": generation,
        "reference": reference,
        "output": output,
        "asset_lock": assets,
        "scope": "simulated",
        "metrics": ("kid", "cmmd", "clipscore", "vqa"),
        "vqa_backend": "mock",
        "kid_subset_size": 2,
    }
    report = run_scoring(**options)
    assert report["complete"] and report["exploratory"]
    assert report["metrics"]["vqa"]["scope"] == "simulated"
    assert report["metrics"]["vqa"]["categories"]["object"]["indeterminate"] == 3
    assert run_scoring(**options, resume=True) == report
    with pytest.raises(FileExistsError):
        run_scoring(**options)
    with pytest.raises(ValueError, match="identity changed"):
        run_scoring(**{**options, "kid_subset_size": 3}, resume=True)
    stale = json.loads(output.read_text())
    stale["identity"]["source_sha256"]["scoring.py"] = "0" * 64
    output.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="identity changed"):
        run_scoring(**options, resume=True)
    output.write_text(json.dumps(report))
    (tmp_path / "image0.png").write_bytes(b"corrupt")
    failed = run_scoring(**options, resume=True)
    assert failed["complete"] is False
    assert failed["coverage"]["attempted"] == 3 and failed["coverage"]["valid"] == 2
    assert failed["metrics"]["vqa"]["categories"]["object"]["error"] == 1
    assert failed["metrics"]["vqa"]["categories"]["object"]["applicable"] == 3


def test_distribution_split_and_formal_mock_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """数据集用途与模拟后端不能靠改名称升级为正式验收。

    Args:
        tmp_path: 独立测试根目录。
        monkeypatch: 测试特征替身。

    Raises:
        AssertionError: 错误用途未被拒绝。

    副作用:
        写入测试清单。
    """
    monkeypatch.setattr("minigen.evaluation.scoring.FeatureSuite", FixtureFeatures)
    generation, reference, assets = scoring_fixture(tmp_path)
    with pytest.raises(ValueError, match="Mock VQA"):
        run_scoring(
            tmp_path,
            generation,
            reference,
            tmp_path / "score.json",
            assets,
            ("vqa",),
            scope="formal",
            vqa_backend="mock",
        )
    payload = json.loads(reference.read_text())
    payload["validation_references"][0]["split"] = "test"
    reference.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Reference split"):
        run_scoring(
            tmp_path,
            generation,
            reference,
            tmp_path / "score.json",
            assets,
            ("kid",),
            scope="simulated",
            kid_subset_size=2,
        )
