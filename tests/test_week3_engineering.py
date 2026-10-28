"""第三周数据、冻结适配与缓存契约测试，无网络或预训练权重需求。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn

from minigen._typing import JsonDict
from minigen.autoencoders.frozen import FrozenVAE, load_cache, write_cache
from minigen.data.icons import audit_icons, build_icons, caption, load_manifest, render

cv2 = pytest.importorskip("cv2", reason="Install --extra latent for image diagnostics")
icon_eval = pytest.importorskip("minigen.evaluation.icons")
aggregate_scores = icon_eval.aggregate_scores
gate_passes = icon_eval.gate_passes
score_image = icon_eval.score_image


@pytest.fixture
def dataset(
    tmp_path: Path,
) -> tuple[JsonDict, Path, list[JsonDict], Path]:
    """在 pytest 临时目录构建五集合各九张的规则图标 fixture。

    Args:
        tmp_path: pytest 隔离目录，不改正式数据。

    Raises:
        生成配置、文件或清单验证错误原样透传，使 fixture 创建失败。

    Results:
        tuple[JsonDict, Path, list[JsonDict], Path]: 配置、清单路径、校验后的记录列表和 data
            根目录。

    副作用:
        在 pytest 临时目录创建隔离的图标与清单。
    """
    config = {
        "dataset_id": "test",
        "generator_revision": "test-1",
        "seed": 1,
        "image_size": 128,
        "min_size": 24,
        "max_size": 32,
        "min_gap": 8,
        "split_counts": {
            s: 9 for s in ["train", "validation", "test_id", "test_combo", "test_template"]
        },
    }
    path = build_icons(config, tmp_path)
    return config, path, load_manifest(path, tmp_path, verify_hashes=True), tmp_path


def test_generation_is_deterministic_and_splits_are_grouped(
    dataset: tuple[JsonDict, Path, list[JsonDict], Path],
    tmp_path: Path,
) -> None:
    """验证图标重建确定性、冻结配置、唯一像素与分组隔离。

    Args:
        dataset: 图标 fixture，依次为配置、清单路径、真实记录和临时 data 根。
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    config, path, records, root = dataset
    before = path.read_bytes()
    assert build_icons(config, root) == path
    assert path.read_bytes() == before
    assert audit_icons(records, root)["unique_pixels"] == 45
    second = build_icons(config, tmp_path / "other")
    assert second.read_bytes() == before
    changed = {**config, "seed": 2}
    with pytest.raises(ValueError, match="Frozen"):
        build_icons(changed, root)
    scores = []
    for r in records:
        with Image.open(root / r["image_path"]) as image:
            scores.append(score_image(image, r))
    assert aggregate_scores(scores)["all_accuracy"] == 1


def test_manifest_rejects_fake_escape_tamper_and_leakage(
    dataset: tuple[JsonDict, Path, list[JsonDict], Path],
) -> None:
    """验证示例、路径越界、哈希篡改和分组泄漏不能进入数据流程。

    Args:
        dataset: 图标 fixture，依次为配置、清单路径、真实记录和临时 data 根。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    _, path, records, root = dataset
    wrong = copy.deepcopy(records)
    wrong[0]["example_only"] = True
    bad_path = root / "bad.jsonl"
    bad_path.write_text(json.dumps(wrong[0]))
    with pytest.raises(ValueError, match="Example"):
        load_manifest(bad_path, root)
    wrong[0].pop("example_only")
    wrong[0]["image_path"] = "../../outside.png"
    bad_path.write_text(json.dumps(wrong[0]))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(bad_path, root)
    wrong = copy.deepcopy(records)
    wrong[0]["group_id"] = wrong[-1]["group_id"]
    wrong[0]["split"] = "some_other_split"
    with pytest.raises(ValueError):
        audit_icons(wrong, root)
    (root / records[0]["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path, root, verify_hashes=True)


def test_parser_detects_missing_recolored_and_blank_objects(
    dataset: tuple[JsonDict, Path, list[JsonDict], Path],
) -> None:
    """验证属性解析器将删对象、换色与空白图判为失败。

    Args:
        dataset: 图标 fixture，依次为配置、清单路径、真实记录和临时 data 根。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    _, _, records, root = dataset
    record = next(r for r in records if r["attributes"]["count"] == 3)
    objects = copy.deepcopy(record["attributes"]["objects"])
    assert score_image(render(objects), record)["all_correct"] == 1
    assert score_image(render(objects[:-1]), record)["all_correct"] == 0
    objects[0]["color"] = "red" if objects[0]["color"] != "red" else "blue"
    assert score_image(render(objects), record)["all_correct"] == 0
    blank = score_image(Image.new("RGB", (128, 128), "white"), record)
    assert blank["parse_failure"] == 1
    assert aggregate_scores([blank])["all_accuracy"] == 0
    assert not gate_passes(aggregate_scores([blank]))
    original = record["attributes"]
    assert caption(original["objects"], original["relations"]) != caption(
        original["objects"], original["relations"], True
    )


class FakeVAE(nn.Module):
    """以成对倍乘/除法模拟 VAE 外壳，只验证缩放适配器。

    Results:
        FakeVAE: 带 scaling_factor、可冻结参数和 encode/decode 结果包装的测试模块。

    副作用:
        注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
    """

    def __init__(
        self,
    ) -> None:
        """创建测试缩放系数与标量参数，以便检查冻结。

        副作用:
            注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
        """
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.13)
        self.parameter = nn.Parameter(torch.ones(()))

    def encode(
        self,
        value: Float[Tensor, "b c h w"],
    ) -> SimpleNamespace:
        """包装两倍输入作为假后验均值，不进行表示学习。

        Args:
            value: [b,c,h,w] 浮点测试输入。

        Results:
            SimpleNamespace: SimpleNamespace，其中 latent_dist.mode 为无参函数，返回同形 value×2。
        """
        # mode 的契约为 Callable[[], Float[Tensor, "b c h w"]]。
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value * 2))

    def decode(
        self,
        value: Float[Tensor, "b c h w"],
    ) -> SimpleNamespace:
        """包装输入的一半作为假解码输出，形成解析往返。

        Args:
            value: 与假编码相匹配的同形张量。

        Results:
            SimpleNamespace: SimpleNamespace，sample 字段为 value/2 的同形张量。
        """
        return SimpleNamespace(sample=value / 2)


def test_scaling_roundtrip_freeze_and_pixel_contract() -> None:
    """使用假 VAE 验证缩放往返、参数冻结、模式和像素范围检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
    """
    adapter = FrozenVAE(FakeVAE(), {"revision": "test"})
    image = torch.rand(2, 3, 8, 8) * 2 - 1
    torch.testing.assert_close(adapter.encode(image), image * 2 * 0.13)
    torch.testing.assert_close(adapter.decode(adapter.encode(image)), image)
    assert not adapter.model.training
    assert all(not p.requires_grad for p in adapter.model.parameters())
    with pytest.raises(ValueError):
        adapter.encode(image * 20)


def test_cache_identity_and_payload_are_checked(
    tmp_path: Path,
) -> None:
    """验证缓存重新加载一致，并拒绝身份变化与载荷篡改。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    path = tmp_path / "latents.npy"
    latents = torch.randn(3, 4, 2, 2)
    identity = {"revision": "one", "scale": 0.13, "preprocessing": "A", "data": "hash"}
    write_cache(path, latents, identity)
    torch.testing.assert_close(load_cache(path, identity), latents)
    for key in identity:
        changed = {**identity, key: "different"}
        with pytest.raises(ValueError, match="identity"):
            load_cache(path, changed)
    np.save(path, np.zeros_like(latents))
    with pytest.raises(ValueError, match="payload"):
        load_cache(path, identity)
