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
