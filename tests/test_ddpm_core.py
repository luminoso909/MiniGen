"""独立验收入口、DDPM 边界、默认长日程、裁剪和缓冲区回归测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minigen.objectives.ddpm_scheduler import DDPMScheduler


def test_independent_core_acceptance() -> None:
    """从真实脚本入口验证个人核心，不提取或调用 Notebook 教学实现。

    Raises:
        AssertionError: 验收进程、来源或数值结果不符合约定。
        OSError: 进程无法启动时透传；JSONDecodeError: 验收输出不是合法 JSON。

    副作用:
        启动当前项目解释器的 CPU 验收子进程，捕获 stdout/stderr。
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_ddpm_core.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "已验证" and report["source"] == "project_core"
    assert report["empirical_samples"] == 32768
    assert report["posterior_mean_max_error"] < 3e-7


@pytest.mark.parametrize("num_steps", [0, -1, 1.5, True])
def test_invalid_step_count(
    num_steps: int | float,
) -> None:
    """验证日程步数不能为非正值、小数或布尔值。

    Args:
        num_steps: 故意违反构造接口的测试值；bool 是 int 的子类但不是合法步数。

    Raises:
        AssertionError: 未明确抛出 ValueError；预期异常由 pytest 捕获。
    """
    with pytest.raises(ValueError, match="整数"):
        DDPMScheduler(num_steps=num_steps)


@pytest.mark.parametrize("start,end", [(0.0, 0.2), (-0.1, 0.2), (0.3, 0.2), (0.1, 1.0)])
def test_invalid_beta_range(
    start: float,
    end: float,
) -> None:
    """验证 beta 起止值严格满足 0 < start <= end < 1。

    Args:
        start: 故意违反日程约束的起点或配对起点。
        end: 故意违反日程约束的终点或配对终点。

    Raises:
        AssertionError: 未拒绝非法 beta；预期 ValueError 由 pytest 捕获。
    """
    with pytest.raises(ValueError, match="beta"):
        DDPMScheduler(beta_start=start, beta_end=end)


@pytest.mark.parametrize("case", ["rank", "batch", "dtype", "negative", "overflow"])
def test_invalid_batch_times(
    case: str,
) -> None:
    """验证逐样本时间索引的维度、数量、dtype 和边界。

    Args:
        case: 非法时间索引的类别，对应一个明确的 ValueError 条件。

    Raises:
        AssertionError: 未拒绝无效索引；预期异常由 pytest 捕获。
    """
    scheduler = DDPMScheduler(num_steps=3)
    image = torch.zeros(2, 1, 2, 2)
    invalid = {
        "rank": torch.zeros(2, 1, dtype=torch.long),
        "batch": torch.zeros(1, dtype=torch.long),
        "dtype": torch.zeros(2),
        "negative": torch.tensor([-1, 0]),
        "overflow": torch.tensor([0, 3]),
    }
    with pytest.raises(ValueError):
        scheduler.add_noise(image, image, invalid[case])


@pytest.mark.parametrize("timestep", [-1, 3, 0.5, 1.5, True])
def test_invalid_reverse_index(
    timestep: int | float,
) -> None:
    """验证反向时间既在范围内，又不会静默截断小数或接受 bool。

    Args:
        timestep: 故意越界或违反整数约定的时间索引。

    Raises:
        AssertionError: 非法索引没有明确失败；预期 ValueError 由 pytest 捕获。
    """
    scheduler = DDPMScheduler(num_steps=3)
    image = torch.zeros(2, 1, 2, 2)
    with pytest.raises(ValueError):
        scheduler.step(image, timestep, image, image)


def test_shape_mismatches_never_broadcast() -> None:
    """验证加噪、损失和反向一步都拒绝错误形状的隐式广播。

    Raises:
        AssertionError: 任一入口接受不一致形状；预期 ValueError 由 pytest 捕获。
    """
    scheduler = DDPMScheduler(num_steps=3)
    image = torch.zeros(2, 1, 2, 2)
    shorter = image[:1]
    with pytest.raises(ValueError):
        scheduler.add_noise(image, shorter, torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        scheduler.loss(image, shorter)
    with pytest.raises(ValueError):
        scheduler.step(shorter, 1, image, image)
    with pytest.raises(ValueError):
        scheduler.step(image, 1, image, shorter)


@pytest.mark.parametrize("clip", [False, True])
def test_one_step_oracle_and_clipping(
    clip: bool,
) -> None:
    """验证单步日程能解析恢复干净值，裁剪只改变估计值且最终步不加随机项。

    Args:
        clip: 是否把估计的干净图像限制到 [-1,1]。

    Raises:
        AssertionError: 恢复值、裁剪、输入保持或最后一步噪声独立性不正确。
    """
    scheduler = DDPMScheduler(num_steps=1, beta_start=0.1, beta_end=0.1, clip_denoised=clip)
    clean = torch.tensor([-2.0, 0.5, 3.0]).reshape(3, 1, 1, 1)
    epsilon = torch.full_like(clean, 0.25)
    noisy = scheduler.add_noise(clean, epsilon, torch.zeros(3, dtype=torch.long))
    before = noisy.clone()
    a = scheduler.step(epsilon, 0, noisy, torch.zeros_like(clean))
    b = scheduler.step(epsilon, 0, noisy, torch.full_like(clean, 1e5))
    torch.testing.assert_close(a, clean.clamp(-1, 1) if clip else clean)
    assert torch.equal(a, b) and torch.equal(noisy, before)
    assert scheduler.post_var.item() == 0


def test_default_schedule_high_noise_oracle() -> None:
    """验证默认 1000 步日程的首尾索引，防止只在短日程上正确。

    Raises:
        AssertionError: 高噪声解析恢复、系数有限性或后验方差不符合约定。
    """
    scheduler = DDPMScheduler()
    clean = torch.tensor([-0.7, 0.3, 0.9]).reshape(3, 1, 1, 1)
    noise = torch.full_like(clean, 0.4)
    times = torch.tensor([0, 500, 999])
    noisy = scheduler.add_noise(clean, noise, times)
    # 已知噪声的代数恢复不代表网络已经具备生成能力。
    recovered = (noisy - scheduler.noise_scales[times, None, None, None] * noise) / (
        scheduler.signals[times, None, None, None]
    )
    torch.testing.assert_close(recovered, clean, rtol=0, atol=1e-5)
    assert all(torch.isfinite(value).all() for value in scheduler.buffers())
    assert scheduler.post_var[0] == 0 and torch.all(scheduler.post_var[1:] > 0)


def test_schedule_buffers_roundtrip() -> None:
    """验证日程随模块迁移 dtype、进入 state_dict，且没有可训练参数。

    Raises:
        AssertionError: 日程表缺失、迁移/重载不一致或误注册为可训练参数。

    副作用:
        将局部测试模块的浮点缓冲区迁移为 float64，并向另一实例载入其状态。
    """
    scheduler = DDPMScheduler(num_steps=3).to(dtype=torch.float64)
    restored = DDPMScheduler(num_steps=3).to(dtype=torch.float64)
    restored.load_state_dict(scheduler.state_dict())
    assert len(list(scheduler.parameters())) == 0
    assert set(scheduler.state_dict()) == set(dict(scheduler.named_buffers()))
    assert all(
        value.dtype == torch.float64 and not value.requires_grad for value in scheduler.buffers()
    )
    for name, value in scheduler.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
