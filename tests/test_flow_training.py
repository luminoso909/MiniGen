"""Flow 训练的数值轨迹、有效 batch、日志修复与恢复边界。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset

from minigen.contracts import Condition
from minigen.data.representation_cache import Representation
from minigen.training.checkpoint import load_checkpoint
from minigen.training.config import TrainConfig, learning_rate_for_step
from minigen.training.engine import train_flow, validation_loss
from minigen.training.runtime import resolve_device


class TinyData(Dataset[Representation]):
    """用独立 RNG 生成四条可精确重放的图文表示。"""

    def __init__(self) -> None:
        """创建与训练全局 RNG 无关的固定数据。

        副作用:
            分配实例内 CPU 张量。
        """
        rng = torch.Generator().manual_seed(123)
        self.latents = torch.randn((4, 4, 2, 2), generator=rng)
        self.text = torch.randn((4, 2, 3), generator=rng)

    def __len__(self) -> int:
        """返回固定样本数。

        Results:
            int: 四条样本。
        """
        return 4

    def __getitem__(
        self,
        index: int,
    ) -> Representation:
        """取一条有明确 ID 的独立图文表示。

        Args:
            index: 样本下标。

        Raises:
            IndexError: 超出数据范围。

        Results:
            Representation: [4,2,2] 潜变量、[2,3] 文本和 [2] 掩码。
        """
        return Representation(
            str(index), self.latents[index], self.text[index], torch.ones(2, dtype=torch.bool)
        )


class TinyVelocity(nn.Module):
    """可手查的条件速度模型，用于检查真实训练循环。"""

    def __init__(self) -> None:
        """初始化三个确定性可训练权重。

        副作用:
            注册模型参数。
        """
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """结合状态、时间与文本给出速度。

        Args:
            z: [b,c,h,w] 路径状态。
            t: [b] 连续时间。
            condition: [b,n,d] 条件及其掩码。

        Results:
            Float[Tensor, "b c h w"]: 输入同形的预测速度。
        """
        text = condition.features.mean((1, 2))
        return (
            self.weights[0] * z
            + (self.weights[1] * t + self.weights[2] * text)[:, None, None, None]
        )


def blank_condition() -> Condition:
    """创建测试编码器的独立空条件。

    Results:
        Condition: [1,2,3] 零特征和两个有效 token。
    """
    return Condition(torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))


def test_exact_resume_and_history_reconciliation(
    tmp_path: Path,
) -> None:
    """验证真实 CPU 更新可精确恢复，并修复超前的日志尾部。

    Args:
        tmp_path: 独立测试目录。

    Raises:
        AssertionError: 权重、EMA、优化器、数据位置或历史不能对应。

    副作用:
        运行短训练并写测试检查点和日志。
    """
    data = TinyData()
    config = TrainConfig(
        output_dir=str(tmp_path / "full"),
        device="cpu",
        batch_size=2,
        max_steps=8,
        max_seconds=20,
        validation_every=2,
        checkpoint_steps=(),
        checkpoint_every=2,
        cpu_threads=1,
    )
    model = TinyVelocity().eval()
    train_flow(config, data, blank_condition(), {"fixture": 1}, model=model, validation=data)
    assert not model.training
    full = load_checkpoint(tmp_path / "full/last.pt")
    split = replace(config, output_dir=str(tmp_path / "split"), max_steps=4)
    train_flow(
        split, data, blank_condition(), {"fixture": 1}, model=TinyVelocity(), validation=data
    )
    with (tmp_path / "split/metrics.jsonl").open("a") as handle:
        handle.write('{"step": 999}\n')
    train_flow(
        replace(split, max_steps=8),
        data,
        blank_condition(),
        {"fixture": 1},
        model=TinyVelocity(),
        resume=tmp_path / "split/last.pt",
        validation=data,
    )
    restored = load_checkpoint(tmp_path / "split/last.pt")
    for name in ("model", "ema"):
        assert all(torch.equal(value, restored[name][key]) for key, value in full[name].items())
    for key, states in full["optimizer"]["state"].items():
        assert all(
            torch.equal(value, restored["optimizer"]["state"][key][name])
            for name, value in states.items()
        )
    assert torch.equal(full["rng"]["cpu"], restored["rng"]["cpu"])
    assert torch.equal(full["stream"]["order"], restored["stream"]["order"])
    assert full["stream"]["cursor"] == restored["stream"]["cursor"]
    for left, right in zip(full["history"], restored["history"], strict=True):
        for field in ("step", "loss", "lr", "grad_norm", "samples", "val_loss"):
            if field in left:
                assert left[field] == right[field]
    rows = [json.loads(row) for row in (tmp_path / "split/metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(1, 9))
    with pytest.raises(ValueError, match="mismatch"):
        train_flow(
            replace(split, max_steps=8, condition_dropout=0.2),
            data,
            blank_condition(),
            {"fixture": 1},
            model=TinyVelocity(),
            resume=tmp_path / "split/last.pt",
        )


def test_accumulation_and_validation_rng(
    tmp_path: Path,
) -> None:
    """确认有效 batch 的微批划分保持抽样及优化一致，验证不消耗训练 RNG。

    Args:
        tmp_path: 独立测试目录。

    Raises:
        AssertionError: 微批梯度缩放、优化结果或验证 RNG 隔离失败。

    副作用:
        执行两段短训练并写本地临时产物。
    """
    data = TinyData()
    config = TrainConfig(
        output_dir=str(tmp_path / "large"),
        device="cpu",
        batch_size=4,
        max_steps=3,
        max_seconds=20,
        checkpoint_steps=(),
        cpu_threads=1,
    )
    models = [TinyVelocity(), TinyVelocity()]
    train_flow(config, data, blank_condition(), {"fixture": 1}, model=models[0])
    train_flow(
        replace(config, output_dir=str(tmp_path / "micro"), batch_size=1, grad_accum=4),
        data,
        blank_condition(),
        {"fixture": 1},
        model=models[1],
    )
    torch.testing.assert_close(models[0].weights, models[1].weights, atol=1e-7, rtol=1e-6)
    state = torch.get_rng_state().clone()
    models[0].train()
    first = validation_loss(models[0], data, config, torch.device("cpu"))
    assert first == validation_loss(models[0], data, config, torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state()) and models[0].training


def test_limits_schedule_and_device_errors(
    tmp_path: Path,
) -> None:
    """固定日程不随停止上限改变，超时在更新边界留下可恢复检查点。

    Args:
        tmp_path: 独立测试目录。

    Raises:
        AssertionError: 上限、日程、设备或已有运行保护失效。

    副作用:
        写入零步超时检查点。
    """
    config = TrainConfig(
        output_dir=str(tmp_path / "timed"),
        device="cpu",
        max_steps=2,
        max_seconds=1e-12,
        checkpoint_steps=(),
        cpu_threads=1,
    )
    result = train_flow(config, TinyData(), blank_condition(), {"fixture": 1}, model=TinyVelocity())
    assert result["status"] == "time_limit" and result["step"] == 0
    assert load_checkpoint(tmp_path / "timed/last.pt")["step"] == 0
    assert learning_rate_for_step(config, 0) == config.learning_rate / config.warmup_steps
    assert (
        learning_rate_for_step(config, config.schedule_steps)
        == config.learning_rate * config.min_lr_ratio
    )
    assert learning_rate_for_step(config, 50) == learning_rate_for_step(
        replace(config, max_steps=100), 50
    )
    with pytest.raises(ValueError, match="already contains"):
        train_flow(config, TinyData(), blank_condition(), {"fixture": 1}, model=TinyVelocity())
    with pytest.raises(ValueError, match="Unknown training"):
        TrainConfig.from_dict({"typo": 1})
    with pytest.raises(ValueError, match="requires fp32"):
        resolve_device("cpu", "fp16")


@pytest.mark.parametrize("architecture", ["b0", "b1", "b2"])
def test_three_architecture_exact_resume(
    architecture: str,
    tmp_path: Path,
) -> None:
    """实际三种主干共享训练器，CPU 的连续更新与中断恢复保持相同轨迹。

    Args:
        architecture: b0、b1 或 b2 工厂分支。
        tmp_path: 独立结果目录。

    Raises:
        AssertionError: 模型/EMA、优化状态、数据序列或日志无法精确恢复。

    副作用:
        构造小规格真实主干并执行 6 步以及 3+3 步 CPU 训练。
    """
    specification = {
        "architecture": architecture,
        "hidden_size": 32,
        "num_heads": 4,
        "latent_size": 2,
        "in_channels": 4,
        "text_dim": 3,
        "patch_size": 2,
        "mlp_ratio": 2,
    }
    specification.update(
        {"double_depth": 1, "single_depth": 1} if architecture == "b2" else {"depth": 2}
    )
    config = TrainConfig(
        output_dir=str(tmp_path / "full"),
        model=specification,
        device="cpu",
        batch_size=2,
        max_steps=6,
        max_seconds=30,
        checkpoint_steps=(),
        checkpoint_every=3,
        cpu_threads=1,
        warmup_steps=1,
        schedule_steps=8,
    )
    data = TinyData()
    result = train_flow(config, data, blank_condition(), {"fixture": "three-model"})
    split = replace(config, output_dir=str(tmp_path / "split"), max_steps=3)
    train_flow(split, data, blank_condition(), {"fixture": "three-model"})
    train_flow(
        replace(split, max_steps=6),
        data,
        blank_condition(),
        {"fixture": "three-model"},
        resume=tmp_path / "split/last.pt",
    )
    full, resumed = (load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "split"))
    for name in ("model", "ema"):
        for key, tensor in full[name].items():
            assert torch.equal(tensor, resumed[name][key])
    for key, states in full["optimizer"]["state"].items():
        assert all(
            torch.equal(tensor, resumed["optimizer"]["state"][key][name])
            for name, tensor in states.items()
        )
    assert torch.equal(full["stream"]["order"], resumed["stream"]["order"])
    assert torch.equal(full["rng"]["cpu"], resumed["rng"]["cpu"])
    assert [row["loss"] for row in full["history"]] == [row["loss"] for row in resumed["history"]]
    assert result["architecture"] == architecture and result["objective"] == "flow_velocity"
    assert result["performance"]["samples_seen"] == 12
    assert result["performance"]["samples_per_optimizer_second"] > 0
    assert result["memory"]["mps_current_allocated_bytes"] is None


def test_checkpoint_objective_and_legacy_boundary(
    tmp_path: Path,
) -> None:
    """拒绝错误速度目标与 v2 缺失字段，明确兼容历史 v1 而不重写文件。

    Args:
        tmp_path: 测试检查点目录。

    Raises:
        AssertionError: 非 Flow 目标被接受，或旧文件读取产生内容变更。

    副作用:
        写入隔离的最小 schema 验证文件。
    """
    path = tmp_path / "checkpoint.pt"
    legacy = {"schema": "minigen.flow-training.v1", "config": {}, "contract": {"recipe": {}}}
    torch.save(legacy, path)
    original = path.read_bytes()
    state = load_checkpoint(path)
    assert state["objective"] == "flow_velocity"
    assert state["objective_source"] == "legacy_flow_schema"
    assert path.read_bytes() == original
    for bad in (
        {**legacy, "objective": "epsilon"},
        {**legacy, "config": {"objective": "epsilon"}},
        {**legacy, "contract": {"recipe": {"objective": "epsilon"}}},
        {**legacy, "schema": "minigen.flow-training.v2"},
    ):
        torch.save(bad, path)
        with pytest.raises(ValueError, match="objective"):
            load_checkpoint(path)
    with pytest.raises(ValueError, match="objective"):
        TrainConfig(objective="epsilon")


def test_failure_receipt_preserves_valid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """训练数值失败保留最后有效检查点和旧结果，身份错误不能覆写失败回执。

    Args:
        tmp_path: 独立结果目录。
        monkeypatch: 显式注入非有限输出，验证失败路径。

    Raises:
        AssertionError: 失败状态、尝试身份、旧结果或有效恢复点丢失。

    副作用:
        执行成功与失败的短训练，并保存 failure.json。
    """
    config = TrainConfig(
        output_dir=str(tmp_path),
        device="cpu",
        max_steps=1,
        max_seconds=30,
        checkpoint_steps=(),
        cpu_threads=1,
    )
    train_flow(config, TinyData(), blank_condition(), {"fixture": 1}, model=TinyVelocity())
    previous = (tmp_path / "result.json").read_bytes()

    def nonfinite_forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """模拟设备计算产生 NaN，测试训练器的有限值拒绝。

        Args:
            z: [b,c,h,w] 输入状态。
            t: [b] 连续时间，测试故障不使用。
            condition: 文本条件，测试故障不使用。

        Results:
            Float[Tensor, "b c h w"]: 含 NaN 的同形错误输出。
        """
        return z * self.weights[0] * float("nan")

    monkeypatch.setattr(TinyVelocity, "forward", nonfinite_forward)
    with pytest.raises(ValueError, match="Nonfinite"):
        train_flow(
            replace(config, max_steps=2),
            TinyData(),
            blank_condition(),
            {"fixture": 1},
            model=TinyVelocity(),
            resume=tmp_path / "last.pt",
        )
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["error_type"] == "ValueError"
    assert failure["attempt_id"] != json.loads(previous)["attempt_id"]
    assert failure["run_id"] == json.loads(previous)["run_id"]
    assert load_checkpoint(tmp_path / "last.pt")["step"] == 1
    assert (tmp_path / "result.json").read_bytes() == previous
    previous_failure = (tmp_path / "failure.json").read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        train_flow(
            replace(config, max_steps=2),
            TinyData(),
            blank_condition(),
            {"fixture": 2},
            model=TinyVelocity(),
            resume=tmp_path / "last.pt",
        )
    assert (tmp_path / "failure.json").read_bytes() == previous_failure
