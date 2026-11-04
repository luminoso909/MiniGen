"""第二周 VAE 教学数学及单进程数据顺序恢复测试。"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from minigen._typing import JsonDict, KLFn, ReparameterizeFn
from minigen.autoencoders import vae_core
from minigen.data.stateful import StatefulBatchStream
from minigen.models.tiny_vae import TinyVAE
from minigen.training.denoising import TrainConfig, seed_all, train_denoiser
from minigen.training.vae import VAEConfig, evaluate_vae, reconstruction_loss, train_vae


@pytest.fixture
def reference() -> tuple[ReparameterizeFn, KLFn]:
    """显式装载第二周 Notebook 的唯一教学分布核心。

    Raises:
        AssertionError: 教学单元不唯一；文件、JSON 或执行装载错误透传。

    Results:
        tuple[ReparameterizeFn, KLFn]: 重参数化函数和 KL 函数的二元组；不会写回个人实现。
    """
    notebook = json.loads(Path("docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb").read_text())
    cells = [
        c for c in notebook["cells"] if "vae_reference" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1
    scope = {"torch": torch}
    exec("".join(cells[0]["source"]), scope)
    return scope["teaching_reparameterize"], scope["teaching_kl"]


def test_reference_math_gradients_and_reduction(
    reference: tuple[ReparameterizeFn, KLFn],
) -> None:
    """用独立分布 KL、gradcheck 和重复 batch 检查 VAE 公式与归一化。

    Args:
        reference: 第二周 Notebook 中显式提取的重参数化与 KL 函数。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
    """
    reparam, kl = reference
    mean = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    noise = torch.randn_like(mean)
    assert torch.autograd.gradcheck(reparam, (mean, logvar, noise))
    assert torch.autograd.gradcheck(kl, (mean, logvar))
    q = torch.distributions.Normal(mean, (logvar / 2).exp())
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    independent = torch.distributions.kl_divergence(q, p).sum(1).mean()
    torch.testing.assert_close(kl(mean, logvar), independent)
    torch.testing.assert_close(kl(mean.repeat(3, 1), logvar.repeat(3, 1)), independent)
    assert kl(torch.zeros_like(mean), torch.zeros_like(logvar)) == 0
    with pytest.raises(ValueError):
        reparam(mean, logvar, noise[:, :1])


def test_project_core_matches_independent_gaussian_contract() -> None:
    """复验已完成的个人 VAE，与独立高斯分布对照而不读取教学答案。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    mean = torch.tensor([[1.0, -0.5], [0.2, 2.0]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([[0.5, -0.5], [1.0, 0.0]], dtype=torch.float64)
    q = torch.distributions.Normal(mean, (logvar / 2).exp())
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    expected = torch.distributions.kl_divergence(q, p).sum(-1).mean()
    torch.testing.assert_close(vae_core.reparameterize(mean, logvar, noise), mean + q.scale * noise)
    torch.testing.assert_close(vae_core.kl_standard_normal(mean, logvar), expected)
    assert torch.autograd.gradcheck(vae_core.reparameterize, (mean, logvar, noise))
    assert torch.autograd.gradcheck(vae_core.kl_standard_normal, (mean, logvar))


def test_data_stream_short_batch_epoch_and_recovery() -> None:
    """验证短尾 batch、完整轮次、游标恢复和数据身份检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    dataset = TensorDataset(torch.arange(19))
    stream = StatefulBatchStream(dataset, 7, data_id="v1", seed=8)
    first = [next(stream)[0] for _ in range(3)]
    assert [len(x) for x in first] == [7, 7, 5]
    assert torch.equal(torch.cat(first).sort().values, torch.arange(19))
    state = stream.state_dict()
    following = [next(stream)[0] for _ in range(5)]
    other = StatefulBatchStream(dataset, 7, data_id="v1", seed=999)
    other.load_state_dict(state)
    assert all(torch.equal(a, next(other)[0]) for a in following)
    with pytest.raises(ValueError, match="identity"):
        StatefulBatchStream(dataset, 7, data_id="v2").load_state_dict(state)
    with pytest.raises(ValueError, match="configuration"):
        StatefulBatchStream(dataset, 8, data_id="v1").load_state_dict(state)


def compare_nested(
    a: Any,
    b: Any,
) -> None:
    """递归比较 Checkpoint 中混合的张量、字典、序列与标量。

    Args:
        a: 连续训练的异构状态，可含任意可保存对象。
        b: 恢复训练的对应状态，结构与数值应一致。

    Raises:
        AssertionError: 任意层级结构、标量或张量不一致。
    """
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare_nested(x, y)
    else:
        assert a == b


class EngineeringObjectiveStub:
    """只供训练工程测试的可解析目标替身，不是 DDPM 算法。

    Results:
        EngineeringObjectiveStub: 提供 num_steps、简单加法输入和 MSE，检查训练/恢复链路。

    副作用:
        注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
    """

    num_steps = 7

    def __init__(
        self,
    ) -> None:
        """创建噪声与时间记录列表，用于比较中断前后随机过程。

        副作用:
            注册测试模块或记录列表；带随机参数的模块初始化会消耗 Torch RNG。
        """
        self.noises = []
        self.times = []

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

        副作用:
            向实例 noises、times 追加张量副本，供连续/恢复对照。
        """
        self.noises.append(noise.clone())
        self.times.append(timesteps.clone())
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


class EngineeringModel(nn.Module):
    """仅用 1×1 卷积检查训练与恢复的轻量测试网络。

    Results:
        EngineeringModel: nn.Module 实例，输入输出均为 [b,1,h,w]；不学习时间条件。

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


@pytest.mark.parametrize("split", [2, 3, 5])
def test_denoiser_exact_resume_order_noise_optimizer_and_ema(
    tmp_path: Path,
    split: int,
) -> None:
    """验证单进程断点恢复后的数据、噪声、参数、EMA 与损失和连续训练一致。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。
        split: 参数化的中断步数，用于与连续训练对照。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    torch.set_num_threads(2)
    seed_all(11)
    dataset = TensorDataset(torch.arange(19.0).reshape(19, 1, 1, 1) / 19)
    original = EngineeringModel()

    def stream() -> StatefulBatchStream:
        """为同一闭包数据集构建确定的随机顺序流。

        Results:
            StatefulBatchStream: 新 StatefulBatchStream，固定 data_id、seed 和 batch
                大小用于公平恢复对照。
        """
        return StatefulBatchStream(dataset, 7, data_id="19-v1", seed=6)

    full_scheduler = EngineeringObjectiveStub()
    full = train_denoiser(
        copy.deepcopy(original),
        full_scheduler,
        stream(),
        TrainConfig(str(tmp_path / "full"), max_steps=8),
    )
    first = train_denoiser(
        copy.deepcopy(original),
        EngineeringObjectiveStub(),
        stream(),
        TrainConfig(str(tmp_path / "partial"), max_steps=split),
    )
    next_scheduler = EngineeringObjectiveStub()
    resumed = train_denoiser(
        EngineeringModel(),
        next_scheduler,
        stream(),
        TrainConfig(str(tmp_path / "resumed"), max_steps=8, require_exact_resume=True),
        resume_path=first["checkpoint"],
    )
    a = torch.load(full["checkpoint"], weights_only=True)
    b = torch.load(resumed["checkpoint"], weights_only=True)
    for key in ("model", "optimizer", "ema", "rng", "data_state", "step"):
        compare_nested(a[key], b[key])
    compare_nested(full_scheduler.noises[split:], next_scheduler.noises)
    compare_nested(full_scheduler.times[split:], next_scheduler.times)
    assert [r["loss"] for r in full["metrics"][split:]] == [r["loss"] for r in resumed["metrics"]]


def test_old_checkpoint_cannot_claim_exact_recovery(
    tmp_path: Path,
) -> None:
    """验证旧 Checkpoint 没有数据顺序状态时不能宣称精确恢复。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    first = train_denoiser(
        EngineeringModel(),
        EngineeringObjectiveStub(),
        [(torch.zeros(2, 1, 2, 2),)],
        TrainConfig(str(tmp_path / "first"), max_steps=1),
    )
    with pytest.raises(ValueError, match="no exact"):
        train_denoiser(
            EngineeringModel(),
            EngineeringObjectiveStub(),
            [(torch.zeros(2, 1, 2, 2),)],
            TrainConfig(str(tmp_path / "next"), max_steps=2, require_exact_resume=True),
            resume_path=first["checkpoint"],
        )


def test_vae_training_resume_and_validation_rng(
    tmp_path: Path,
    reference: tuple[ReparameterizeFn, KLFn],
) -> None:
    """验证 VAE 恢复一致性、配置拒绝及验证不消耗训练 RNG。

    Args:
        tmp_path: pytest 提供的隔离临时目录，用于文件、Checkpoint 或缓存检查。
        reference: 第二周 Notebook 中显式提取的重参数化与 KL 函数。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        构造测试张量/模块或调整进程内 Torch 随机状态；相关设置仅用于该测试。
        在 pytest 隔离临时目录中写入或修改测试文件。
    """
    reparam, kl = reference
    seed_all(31)
    images = torch.rand(19, 1, 28, 28)
    dataset = TensorDataset(images)
    original = TinyVAE(2)

    def stream() -> StatefulBatchStream:
        """对固定验证实验创建同配置的数据流。

        Results:
            StatefulBatchStream: 以固定种子排列闭包数据集的 StatefulBatchStream。
        """
        return StatefulBatchStream(dataset, 7, data_id="images01", seed=4)

    def run(
        model: TinyVAE,
        folder: str,
        steps: int,
        resume: str | None = None,
    ) -> JsonDict:
        """以统一配置执行测试中的极短 VAE 连续或恢复训练。

        Args:
            model: 本次待测试模型。
            folder: tmp_path 下的独立输出子目录。
            steps: 本次配置的总更新步数。
            resume: 可选前一段 Checkpoint；分布函数和验证图来自外层 fixture。

        Results:
            JsonDict: 真实极短训练摘要，用于比较状态和验证 RNG。

        副作用:
            执行极短训练，更新传入模型并在临时目录写日志和 Checkpoint。
        """
        return train_vae(
            model,
            stream(),
            VAEConfig(str(tmp_path / folder), max_steps=steps),
            reparameterize_fn=reparam,
            kl_fn=kl,
            validation=images[:3],
            resume_path=resume,
        )

    a = run(copy.deepcopy(original), "full", 6)
    first = run(copy.deepcopy(original), "partial", 2)
    b = run(TinyVAE(2), "resumed", 6, first["checkpoint"])
    saved_a = torch.load(a["checkpoint"], weights_only=True)
    saved_b = torch.load(b["checkpoint"], weights_only=True)
    for key in ("model", "optimizer", "rng", "data_state", "step"):
        compare_nested(saved_a[key], saved_b[key])
    state = torch.get_rng_state().clone()
    evaluate_vae(original, images, reparameterize_fn=reparam, kl_fn=kl)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match="configuration"):
        train_vae(
            TinyVAE(2),
            stream(),
            VAEConfig(str(tmp_path / "bad"), beta=4),
            reparameterize_fn=reparam,
            kl_fn=kl,
            validation=images,
            resume_path=first["checkpoint"],
        )


def test_bce_scale_and_bad_target() -> None:
    """验证重建 BCE 的像素求和尺度，并拒绝范围错误的目标。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    x = torch.zeros(2, 1, 28, 28)
    assert abs(reconstruction_loss(x, x).item() - 784 * torch.log(torch.tensor(2.0)).item()) < 0.001
    with pytest.raises(ValueError):
        reconstruction_loss(x, x - 1)


def test_fixed_batch_stream_keeps_images_and_noise_aligned() -> None:
    """验证固定监督循环的数据次序稳定，并拒绝恢复时改变 shuffle。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    stream = StatefulBatchStream(TensorDataset(torch.arange(32)), 32, data_id="fit", shuffle=False)
    first = next(stream)[0]
    for _ in range(3):
        assert torch.equal(next(stream)[0], first)
    state = stream.state_dict()
    with pytest.raises(ValueError):
        StatefulBatchStream(stream.dataset, 32, data_id="fit", shuffle=True).load_state_dict(state)
