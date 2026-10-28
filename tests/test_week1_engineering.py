"""第一周训练/采样工程契约；解析替身不冒充 DDPM。"""

import copy
from pathlib import Path

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from minigen._learning import CoreImplementationRequired
from minigen.models.tiny_unet import TinyUNet
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.samplers.ddpm_loop import sample_ddpm
from minigen.training.denoising import TrainConfig, seed_all, train_denoiser


class AnalyticTrainingStub:
    """只供训练工程测试的可解析目标替身，不是 DDPM 算法。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        提供 num_steps、简单加法输入和 MSE，检查训练/恢复链路。

    3 报错（Raises）：
        无显式 raise；数值算子异常透传。
    """

    num_steps = 4

    def add_noise(
        self,
        clean: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
        timesteps: Int[Tensor, " batch"],
    ) -> Float[Tensor, "batch channels height width"]:
        """以简单相加提供测试输入，不实现扩散日程。

        1 输入：
            clean：[batch,channels,height,width] 原始输入。
            noise：同形噪声，供工程更新检查。
            timesteps：[batch] 整数索引；本替身不据其改变公式，部分测试记录副本。

        2 输出：
            clean+noise，同形浮点张量。

        3 报错（Raises）：
            RuntimeError：输入形状或设备不兼容时由算子透传。
        """
        return clean + noise

    def loss(
        self,
        prediction: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """为工程替身计算所有元素的均方误差。

        1 输入：
            prediction：[batch,channels,height,width] 模型输出。
            noise：同形监督噪声。

        2 输出：
            零维浮点 MSE；只用于测试训练工程。

        3 报错（Raises）：
            张量运算异常原样透传。
        """
        return (prediction - noise).square().mean()


class SmallModel(nn.Module):
    """仅用 1×1 卷积检查训练与恢复的轻量测试网络。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        nn.Module 实例，输入输出均为 [batch,1,height,width]；不学习时间条件。

    3 报错（Raises）：
        无显式 raise；下层模块异常透传。
    """

    def __init__(self) -> None:
        """注册单通道 1×1 测试卷积。

        1 输入：
            无显式参数；使用对象状态、闭包或命令行上下文。

        2 输出：
            无返回值；创建待训练参数。

        3 报错（Raises）：
            模块构造异常原样透传。
        """
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(
        self, image: Float[Tensor, "batch 1 height width"], times: Int[Tensor, " batch"]
    ) -> Float[Tensor, "batch 1 height width"]:
        """执行简单卷积，隔离检查优化器和状态恢复。

        1 输入：
            image：[batch,1,height,width] 浮点测试图。
            times：为保持训练器接口而接收的 [batch] 时间，本替身有意不使用。

        2 输出：
            与输入同形的卷积结果，不代表真实去噪能力。

        3 报错（Raises）：
            RuntimeError：通道、维数或设备不匹配时由卷积透传。
        """
        return self.layer(image)


def test_unet_shapes_and_time_condition_gradient() -> None:
    """验证 U-Net 保持图像形状且时间路径真实获得有限梯度。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    torch.set_num_threads(2)
    model = TinyUNet(base=8, time_width=16)
    image = torch.randn(2, 1, 28, 28)
    result = model(image, torch.tensor([0, 999]))
    assert result.shape == image.shape
    result.square().mean().backward()
    assert model.time_embedding.mlp[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cpu_checkpoint_resume_matches_continuous_with_fixed_loader(tmp_path: Path) -> None:
    """比较固定 loader 下连续训练与恢复训练的参数、EMA 和步数。

    1 输入：
        tmp_path：pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    torch.set_num_threads(2)
    seed_all(4)
    original = SmallModel()
    batches = [(torch.zeros(2, 1, 4, 4),)]
    full = copy.deepcopy(original)
    train_denoiser(
        full,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "full"), max_steps=6, checkpoint_every=6),
    )
    partial = copy.deepcopy(original)
    first = train_denoiser(
        partial,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "partial"), max_steps=3, checkpoint_every=3),
    )
    resumed = SmallModel()
    train_denoiser(
        resumed,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "resume"), max_steps=6, checkpoint_every=6),
        resume_path=first["checkpoint"],
    )
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=True)
    for key in a["ema"]:
        torch.testing.assert_close(a["ema"][key], b["ema"][key], rtol=0, atol=0)
    assert a["step"] == b["step"] == 6


def test_pending_project_scheduler_is_not_replaced(tmp_path: Path) -> None:
    """验证训练器不会为个人调度器偷偷选择教学实现。

    1 输入：
        tmp_path：pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    with pytest.raises(CoreImplementationRequired):
        train_denoiser(
            SmallModel(),
            DDPMScheduler(),
            [(torch.zeros(2, 1, 4, 4),)],
            TrainConfig(str(tmp_path), max_steps=1),
        )


def test_full_reverse_loop_and_timeout_are_distinguished() -> None:
    """验证完整反向链与超时中间状态的标记不同。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """

    class SamplingStub:
        """每次把状态加一的离散采样替身，用于计数与超时测试。

        1 输入：
            无显式参数；使用对象状态、闭包或命令行上下文。

        2 输出：
            提供四步 num_steps 与单步 step 的对象，不提供真实采样数学。

        3 报错（Raises）：
            无显式 raise。
        """

        num_steps = 4

        def step(
            self,
            predicted: Float[Tensor, "batch channels height width"],
            timestep: int,
            sample: Float[Tensor, "batch channels height width"],
            noise: Float[Tensor, "batch channels height width"],
        ) -> Float[Tensor, "batch channels height width"]:
            """用确定的加一更新检查反向循环是否执行完整。

            1 输入：
                predicted：接口兼容的预测，测试中不使用。
                timestep：循环索引，测试中不改变更新公式。
                sample：[batch,channels,height,width] 当前计数状态。
                noise：接口兼容的随机项，本替身不使用。

            2 输出：
                sample+1 的同形张量，每次调用贡献一次更新。

            3 报错（Raises）：
                底层张量运算异常透传。
            """
            return sample + 1

    model = SmallModel()
    initial = torch.zeros(2, 1, 4, 4)
    result = sample_ddpm(model, SamplingStub(), initial, capture=[0])
    assert result["completed"] and result["completed_steps"] == 4
    torch.testing.assert_close(result["images"], torch.full_like(initial, 4))
    short = sample_ddpm(model, SamplingStub(), initial, max_seconds=1e-12)
    assert not short["completed"] and short["completed_steps"] == 0
    assert model.training
