"""L02 / S1-06：个人对角高斯重参数化与 KL 接口。"""

import torch
from jaxtyping import Float
from torch import Tensor


def reparameterize(
    mean: Float[Tensor, " b *z"],
    log_variance: Float[Tensor, " b *z"],
    noise: Float[Tensor, " b *z"],
) -> Float[Tensor, " b *z"]:
    """用独立标准高斯噪声构造可微对角高斯样本。

    Args:
        mean: [b,*z] 后验均值。
        log_variance: 与 mean 同形的对数方差。
        noise: 与 mean 同形的独立标准高斯噪声。

    Raises:
        RuntimeError: 张量形状无法广播、设备或 dtype 不兼容时由 Torch 透传。

    Results:
        Float[Tensor, " b *z"]: 同形后验样本，保留均值和对数方差的梯度。
    """

    return mean + torch.exp(0.5 * log_variance) * noise


def kl_standard_normal(
    mean: Float[Tensor, " b *z"],
    log_variance: Float[Tensor, " b *z"],
) -> Float[Tensor, ""]:
    """计算对角高斯后验到标准正态的 KL，特征求和后取 batch 均值。

    Args:
        mean: [b,*z] 后验均值；至少有 batch 和一个潜变量轴。
        log_variance: 同形对数方差；需明确特征求和和 batch 均值。

    Raises:
        RuntimeError: 张量无法广播或计算不兼容时由 Torch 透传。
        IndexError: 输入缺少 batch 后的潜变量轴，flatten(start_dim=1) 无法执行。

    Results:
        Float[Tensor, ""]: 零维平均 KL，用于 VAE 正则项。
    """

    D_kl = 0.5 * (torch.pow(mean, 2) + torch.exp(log_variance) - 1 - log_variance)

    return D_kl.flatten(start_dim=1).sum(dim=1).mean()  # .mean() 用于平均化 Batch 维
