"""使用本地小张量验证 CLI 链路；仅冻结组件和资产读取替换为显式测试夹具。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from minigen import commands
from minigen.cli import main
from minigen.contracts import Condition
from minigen.data.representation_cache import Representation
from minigen.training.checkpoint import load_checkpoint


class FixtureCache:
    """提供两条固定表示的测试缓存，不访问真实数据或权重。"""

    def split(
        self,
        split: str,
    ) -> list[Representation]:
        """返回确定表示；测试中训练和验证的重复不代表正式划分。

        Args:
            split: 测试请求的划分名称。

        Results:
            list[Representation]: 两条 [4,4,4] 潜变量与 [3,8] 文本表示。
        """
        return [
            Representation(
                str(i),
                torch.full((4, 4, 4), i / 2),
                torch.full((3, 8), float(i)),
                torch.ones(3, dtype=torch.bool),
            )
            for i in range(2)
        ]


class FixtureVAE:
    """显式替代冻结解码器，只验证 CLI 图片落盘，不能作为 VAE 验收。"""

    def decode(
        self,
        latents: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b 3 h_o w_o"]:
        """插值前三个通道，输出供 PNG 保存检查。

        Args:
            latents: [b,4,h,w] 测试潜变量。

        Results:
            Float[Tensor, "b 3 h_o w_o"]: [b,3,16,16] 测试图像。
        """
        return F.interpolate(latents[:, :3], size=(16, 16)).tanh()


def fixture_cache(
    root: Path,
    data: dict,
) -> tuple[SimpleNamespace, FixtureCache, Condition, dict]:
    """替代正式资产核验，保持测试输入身份固定且可审查。

    Args:
        root: CLI 传入的测试目录。
        data: 已解析的数据配置。

    Results:
        tuple: 仅供测试的来源、缓存、空条件与身份。
    """
    rows = [{"sample_id": str(i), "caption": f"fixture {i}"} for i in range(2)]
    source = SimpleNamespace(
        records={"train": rows, "validation": rows}, identity={"fixture": True}
    )
    return (
        source,
        FixtureCache(),
        Condition(torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)),
        {"fixture": True},
    )


def fixture_vae(
    *args: object,
    **kwargs: object,
) -> FixtureVAE:
    """创建标记明确的测试解码对象。

    Args:
        args: 真实加载器的位置参数，在测试中不使用。
        kwargs: 真实加载器的关键字参数，在测试中不使用。

    Results:
        FixtureVAE: 不下载权重的测试解码器。
    """
    return FixtureVAE()


@pytest.mark.parametrize("architecture", ["b0", "b1", "b2"])
def test_train_resume_sample_and_evaluate_cli(
    architecture: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过真实 CLI 验证三个主干共享的训练、恢复、采样和诊断。

    Args:
        architecture: b0、b1 或 b2，切换工厂而不复制训练流程。
        tmp_path: 独立输出目录。
        monkeypatch: 只替换资产读取及冻结 VAE，保留实际 B0、优化器和采样。

    Raises:
        AssertionError: 命令退出码、步数、求值次数、条件或文件结果不符。

    副作用:
        写入测试配置、检查点、生成图和 JSON。
    """
    monkeypatch.setattr(commands, "open_cache", fixture_cache)
    monkeypatch.setattr(commands.FrozenVAE, "load", fixture_vae)
    model = {
        "architecture": architecture,
        "hidden_size": 16,
        "num_heads": 2,
        "depth": 1,
        "latent_size": 4,
        "in_channels": 4,
        "text_dim": 8,
        "patch_size": 2,
    }
    if architecture == "b2":
        model.pop("depth")
        model.update(double_depth=1, single_depth=1)
    (tmp_path / "model.json").write_text(json.dumps(model))
    config = {
        "model": "model.json",
        "training": {
            "output_dir": "run",
            "device": "cpu",
            "max_steps": 2,
            "checkpoint_steps": [],
            "cpu_threads": 1,
        },
        "data": {"train_limit": 2, "validation_limit": 2},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    common = ["--root", str(tmp_path), "--config", "config.json", "--device", "cpu"]
    assert main(["train", *common]) == 0
    assert main(["train", *common]) == 2
    config["training"]["max_steps"] = 3
    path.write_text(json.dumps(config))
    assert main(["train", *common, "--resume", "run/last.pt"]) == 0
    assert load_checkpoint(tmp_path / "run/last.pt")["step"] == 3
    ckpt = ["--checkpoint", "run/last.pt", "--count", "2"]
    assert (
        main(
            [
                "sample",
                *common,
                *ckpt,
                "--output",
                "samples",
                "--steps",
                "2",
                "--method",
                "heun",
                "--guidance",
                "3",
                "--condition-mode",
                "shuffled",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "samples/samples.json").read_text())
    assert report["schema"] == "minigen.samples.v1"
    assert report["architecture"] == architecture
    assert report["objective"] == "flow_velocity"
    assert report["objective_source"] == "explicit"
    assert report["run_id"] == json.loads((tmp_path / "run/result.json").read_text())["run_id"]
    assert report["solver_nfe"] == 4 and report["model_evaluations"] == 8
    assert report["records"][0]["condition_sample_id"] == "1"
    assert (tmp_path / "samples/grid.png").exists()
    assert (tmp_path / "samples" / report["records"][0]["image_path"]).exists()
    assert (
        main(
            ["evaluate", *common, *ckpt, "--output", "evaluation.json", "--condition-mode", "blank"]
        )
        == 0
    )
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert not evaluation["generation_quality_evaluated"] and evaluation["loss"] > 0
    assert evaluation["architecture"] == architecture
    assert evaluation["run_id"] == report["run_id"]
    assert evaluation["schema"] == "minigen.flow-evaluation.v1"
    assert main(["evaluate", *common, *ckpt, "--output", "evaluation.json"]) == 2
    assert main(["sample", *common, *ckpt, "--output", "bad", "--steps", "0"]) == 2
    assert not (tmp_path / "bad").exists()


def test_explicit_selection_and_configuration_errors(
    tmp_path: Path,
) -> None:
    """检验子集顺序及拒绝重复索引、未知配置，防止恢复身份静默变化。

    Args:
        tmp_path: 独立配置目录。

    Raises:
        AssertionError: 非法配置或不合法索引未拒绝。

    副作用:
        写入一个错误配置文件。
    """
    source, _, _, _ = fixture_cache(tmp_path, {})
    assert commands.selected_indices(source, {"train_indices": [1, 0]}, "train") == [1, 0]
    for indices in ([0, 0], [-1], [2], [], [True]):
        with pytest.raises(ValueError, match="indices"):
            commands.selected_indices(source, {"train_indices": indices}, "train")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model": "missing.json", "training": {}, "data": {}, "typo": True}))
    with pytest.raises(ValueError, match="exactly"):
        commands.read_experiment(tmp_path, path)
