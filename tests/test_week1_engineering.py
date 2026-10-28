"""第一周训练/采样工程契约；解析替身不冒充 DDPM。"""

import copy
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from minigen.models.tiny_unet import TinyUNet
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.samplers.ddpm_loop import sample_ddpm
from minigen.training.denoising import TrainConfig, seed_all, train_denoiser


class AnalyticTrainingStub:
    """只供训练工程测试的可解析目标替身，不是 DDPM 算法。

    Results:
        AnalyticTrainingStub: 提供 num_steps、简单加法输入和 MSE，检查训练/恢复链路。
    """

    num_steps = 4

    def add_noise(
        self,
        clean: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
        timesteps: Int[Tensor, " b"],
    ) -> Float[Tensor, "b c h w"]:
        """以简单相加提供测试输入，不实现扩散日程。

        Args:
            clean: [b,c,h,w] 原始输入。
            noise: 同形噪声，供工程更新检查。
            timesteps: [b] 整数索引；本替身不据其改变公式，部分测试记录副本。

        Raises:
            RuntimeError: 输入形状或设备不兼容时由算子透传。

        Results:
            Float[Tensor, "b c h w"]: clean+noise，同形浮点张量。
        """
        return clean + noise

    def loss(
        self,
        prediction: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, ""]:
        """为工程替身计算所有元素的均方误差。

        Args:
            prediction: [b,c,h,w] 模型输出。
            noise: 同形监督噪声。

        Results:
            Float[Tensor, ""]: 零维浮点 MSE；只用于测试训练工程。
        """
        return (prediction - noise).square().mean()


class SmallModel(nn.Module):
    """仅用 1×1 卷积检查训练与恢复的轻量测试网络。

    Results:
        SmallModel: nn.Module 实例，输入输出均为 [b,1,h,w]；不学习时间条件。

    副作用:
        注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
    """

    def __init__(
        self,
    ) -> None:
        """注册单通道 1×1 测试卷积。

        副作用:
            注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
        """
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(
        self,
        image: Float[Tensor, "b 1 h w"],
        times: Int[Tensor, " b"],
    ) -> Float[Tensor, "b 1 h w"]:
        """执行简单卷积，隔离检查优化器和状态恢复。

        Args:
            image: [b,1,h,w] 浮点测试图。
            times: 为保持训练器接口而接收的 [b] 时间，本替身有意不使用。

        Raises:
            RuntimeError: 通道、维数或设备不匹配时由卷积透传。

        Results:
            Float[Tensor, "b 1 h w"]: 与输入同形的卷积结果，不代表真实去噪能力。
        """
        return self.layer(image)


def test_unet_shapes_and_time_condition_gradient() -> None:
    """验证 U-Net 保持图像形状且时间路径真实获得有限梯度。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
    """
    torch.set_num_threads(2)
    model = TinyUNet(base=8, time_width=16)
    image = torch.randn(2, 1, 28, 28)
    result = model(image, torch.tensor([0, 999]))
    assert result.shape == image.shape
    result.square().mean().backward()
    assert model.time_embedding.mlp[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cpu_checkpoint_resume_matches_continuous_with_fixed_loader(
    tmp_path: Path,
) -> None:
    """比较固定 loader 下连续训练与恢复训练的参数、EMA 和步数。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
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


def test_project_scheduler_trains_resumes_and_samples(
    tmp_path: Path,
) -> None:
    """验证个人 DDPM 在同一日程下能训练、恢复并执行完整短反向链。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    Raises:
        AssertionError: 参数、EMA、来源、步数或完整采样结果不符合约定。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    torch.set_num_threads(2)
    seed_all(7)
    original = SmallModel()
    batches = [(torch.linspace(-1, 1, 32).reshape(2, 1, 4, 4),)]
    full = copy.deepcopy(original)
    full_result = train_denoiser(
        full,
        DDPMScheduler(num_steps=8),
        batches,
        TrainConfig(str(tmp_path / "full"), max_steps=6, scheduler_source="project_core"),
    )
    partial = copy.deepcopy(original)
    first = train_denoiser(
        partial,
        DDPMScheduler(num_steps=8),
        batches,
        TrainConfig(str(tmp_path / "partial"), max_steps=3, scheduler_source="project_core"),
    )
    resumed = SmallModel()
    result = train_denoiser(
        resumed,
        DDPMScheduler(num_steps=8),
        batches,
        TrainConfig(str(tmp_path / "resume"), max_steps=6, scheduler_source="project_core"),
        resume_path=first["checkpoint"],
    )
    assert result["start_step"] == 3 and result["final_step"] == 6
    assert result["config"]["scheduler_source"] == "project_core"
    assert result["resume_mode"] == "rng_only"  # 固定单批数据；不声称恢复随机 loader 的游标。
    assert full_result["metrics"][-1]["loss"] == result["metrics"][-1]["loss"]
    assert any(
        not torch.equal(p, q) for p, q in zip(original.parameters(), full.parameters(), strict=True)
    )
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    saved_full = torch.load(full_result["checkpoint"], weights_only=True)
    saved_resumed = torch.load(result["checkpoint"], weights_only=True)
    for key in saved_full["ema"]:
        torch.testing.assert_close(
            saved_full["ema"][key], saved_resumed["ema"][key], rtol=0, atol=0
        )
    initial = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(11))
    sampled = sample_ddpm(resumed, DDPMScheduler(num_steps=8), initial, capture=[7, 0])
    assert sampled["completed"] and sampled["completed_steps"] == sampled["model_calls"] == 8
    assert sampled["images"].shape == initial.shape and torch.isfinite(sampled["images"]).all()
    assert resumed.training


def test_full_reverse_loop_and_timeout_are_distinguished() -> None:
    """验证完整反向链与超时中间状态的标记不同。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
    """

    class SamplingStub:
        """每次把状态加一的离散采样替身，用于计数与超时测试。

        Results:
            SamplingStub: 提供四步 num_steps 与单步 step 的对象，不提供真实采样数学。
        """

        num_steps = 4

        def step(
            self,
            predicted: Float[Tensor, "b c h w"],
            timestep: int,
            sample: Float[Tensor, "b c h w"],
            noise: Float[Tensor, "b c h w"],
        ) -> Float[Tensor, "b c h w"]:
            """用确定的加一更新检查反向循环是否执行完整。

            Args:
                predicted: 接口兼容的预测，测试中不使用。
                timestep: 循环索引，测试中不改变更新公式。
                sample: [b,c,h,w] 当前计数状态。
                noise: 接口兼容的随机项，本替身不使用。

            Results:
                Float[Tensor, "b c h w"]: sample+1 的同形张量，每次调用贡献一次更新。
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
