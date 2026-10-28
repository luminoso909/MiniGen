"""L01 独立数学验收：只调用个人 DDPM，不加载 Notebook 或参考调度器。"""

import json
import math

import torch

from minigen.objectives.ddpm_scheduler import DDPMScheduler


def check_core() -> dict[str, str | int | float]:
    """用手算常数、高斯精度与噪声参数化检查 DDPM 数值和梯度。

    Raises:
        AssertionError: 加噪分布、后验、反向更新、梯度或最终步不符合独立对照。
        个人实现的未捕获异常原样透传，不自动调用教学参考。

    Results:
        dict[str, str | int | float]: CPU 数学验收指标。
            status/source/device: 验收状态、实际实现来源和执行设备。
            *_max_error: 实测绝对误差；经验分布误差使用预先给定的统计容差。
            empirical_samples: 每个时间级别用于统计的样本数。
    """
    scheduler = DDPMScheduler(num_steps=3, beta_start=0.1, beta_end=0.3)
    # 手算三步累计系数：0.9、0.9×0.8、0.9×0.8×0.7；不读取实现的累计表作真值。
    cumulative = (0.9, 0.72, 0.504)
    clean = torch.tensor([1.25, -0.5, 2.0]).reshape(3, 1, 1, 1)
    epsilon = torch.tensor([-0.5, 0.75, 0.25]).reshape_as(clean)
    times = torch.tensor([0, 1, 2])
    noisy = scheduler.add_noise(clean, epsilon, times)
    expected = torch.tensor(
        [
            math.sqrt(a) * x + math.sqrt(1 - a) * e
            for a, x, e in zip(
                cumulative, clean.flatten().tolist(), epsilon.flatten().tolist(), strict=True
            )
        ]
    ).reshape_as(clean)
    torch.testing.assert_close(noisy, expected, rtol=0, atol=3e-7)
    forward_error = (noisy - expected).abs().max().item()

    # q(x_{t-1}|x0) 与 q(xt|x_{t-1}) 相乘；用精度相加验证后验，避免照抄系数表。
    mean, variance = scheduler._posterior(clean, noisy, times)
    means = [float(clean[0].item())]
    variances = [0.0]
    for index in (1, 2):
        beta = (0.1, 0.2, 0.3)[index]
        alpha = 1 - beta
        prior_variance = 1 - cumulative[index - 1]
        prior_mean = math.sqrt(cumulative[index - 1]) * clean[index].item()
        precision = 1 / prior_variance + alpha / beta
        means.append(
            (prior_mean / prior_variance + math.sqrt(alpha) * noisy[index].item() / beta)
            / precision
        )
        variances.append(1 / precision)
    expected_mean = torch.tensor(means).reshape_as(clean)
    expected_variance = torch.tensor(variances).reshape_as(clean)
    torch.testing.assert_close(mean, expected_mean, rtol=0, atol=3e-7)
    torch.testing.assert_close(variance, expected_variance, rtol=0, atol=3e-8)

    reverse_error = 0.0
    stochastic_error = 0.0
    for index, abar in enumerate(cumulative):
        beta = (0.1, 0.2, 0.3)[index]
        xt, eps = noisy[index : index + 1], epsilon[index : index + 1]
        # DDPM 式 (11)：直接由 epsilon 参数化均值，不经过实现的 x0/后验组合。
        expected_reverse = (xt - beta / math.sqrt(1 - abar) * eps) / math.sqrt(1 - beta)
        reverse = scheduler.step(eps, index, xt, torch.zeros_like(xt))
        torch.testing.assert_close(reverse, expected_reverse, rtol=0, atol=3e-7)
        reverse_error = max(reverse_error, (reverse - expected_reverse).abs().max().item())
        supplied_noise = torch.full_like(xt, 0.75)
        sampled = scheduler.step(eps, index, xt, supplied_noise)
        expected_sample = expected_reverse + math.sqrt(variances[index]) * supplied_noise
        torch.testing.assert_close(sampled, expected_sample, rtol=0, atol=3e-7)
        stochastic_error = max(stochastic_error, (sampled - expected_sample).abs().max().item())
    final_a = scheduler.step(epsilon[:1], 0, noisy[:1], torch.zeros_like(clean[:1]))
    final_b = scheduler.step(epsilon[:1], 0, noisy[:1], torch.full_like(clean[:1], 1e5))
    assert torch.equal(final_a, final_b)
    torch.testing.assert_close(final_a, clean[:1], rtol=0, atol=3e-7)

    sample_count = 32768
    generator = torch.Generator().manual_seed(20261028)
    draws = torch.randn(3, sample_count, 1, 1, generator=generator)
    population = torch.full_like(draws, 1.25)
    observations = scheduler.add_noise(population, draws, times)
    observed_mean = observations.mean(dim=(1, 2, 3))
    observed_variance = observations.var(dim=(1, 2, 3), unbiased=True)
    distribution_mean = torch.tensor([1.25 * math.sqrt(a) for a in cumulative])
    distribution_variance = torch.tensor([1 - a for a in cumulative])
    mean_error = (observed_mean - distribution_mean).abs()
    variance_error = (observed_variance - distribution_variance).abs()
    # 固定种子只保证复现；容差分别为均值和方差理论标准误的六倍。
    assert torch.all(mean_error < 6 * torch.sqrt(distribution_variance / sample_count))
    assert torch.all(variance_error < 6 * distribution_variance * math.sqrt(2 / (sample_count - 1)))

    prediction = torch.tensor([0.2, -0.7, 1.1, 0.0], dtype=torch.float64).reshape(2, 1, 1, 2)
    prediction.requires_grad_()
    target = torch.zeros_like(prediction)
    loss = scheduler.loss(prediction, target)
    expected_loss = sum(value * value for value in (0.2, -0.7, 1.1, 0.0)) / 4
    assert abs(loss.item() - expected_loss) < 1e-12 and loss.ndim == 0
    gradient = torch.autograd.grad(loss, prediction)[0]
    expected_gradient = 2 * (prediction.detach() - target) / prediction.numel()
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        scheduler.loss(prediction.repeat(3, 1, 1, 1), target.repeat(3, 1, 1, 1)), loss
    )
    assert torch.autograd.gradcheck(scheduler.loss, (prediction, target))
    clean_grad = prediction.detach().clone().requires_grad_()
    noise_grad = torch.ones_like(clean_grad).requires_grad_()
    assert torch.autograd.gradcheck(
        scheduler.add_noise, (clean_grad, noise_grad, torch.tensor([0, 2]))
    )

    return {
        "status": "已验证",
        "source": "project_core",
        "device": "cpu",
        "forward_max_error": forward_error,
        "posterior_mean_max_error": (mean - expected_mean).abs().max().item(),
        "posterior_variance_max_error": (variance - expected_variance).abs().max().item(),
        "reverse_mean_max_error": reverse_error,
        "reverse_sample_max_error": stochastic_error,
        "empirical_samples": sample_count,
        "empirical_mean_max_error": mean_error.max().item(),
        "empirical_variance_max_error": variance_error.max().item(),
        "mse": loss.item(),
        "gradient_max_error": (gradient - expected_gradient).abs().max().item(),
    }


def main() -> int:
    """执行个人 DDPM 的 CPU 数学验收并输出 JSON。

    Raises:
        AssertionError: 数学验收失败；个人实现的其他未捕获异常透传，进程返回非零。

    Results:
        int: 通过返回 0，输出包含来源与实测误差；不表示 MNIST 生成质量达标。

    副作用:
        向 stdout 打印数学验收结果，不写数据、模型或学习笔记。
    """
    print(json.dumps(check_core(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
