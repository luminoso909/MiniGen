"""L02 / S1-06：个人对角高斯重参数化与 KL 接口。"""

from jaxtyping import Float
from torch import Tensor

import torch
from minigen._learning import pending


def reparameterize(
    mean: Float[Tensor, " b *z"],
    log_variance: Float[Tensor, " b *z"],
    noise: Float[Tensor, " b *z"],
) -> Float[Tensor, " b *z"]:
    """预留对角高斯的可微重参数化。

    设计输出：同形潜变量，并保留分布参数梯度；当前不返回。

    Args:
        mean: [b,*z] 后验均值。
        log_variance: 与 mean 同形的对数方差。
        noise: 与 mean 同形的独立标准高斯噪声。
    """

    return mean + torch.exp(0.5 * log_variance) * noise

def kl_standard_normal(
    mean: Float[Tensor, " b *z"],
    log_variance: Float[Tensor, " b *z"],
) -> Float[Tensor, ""]:
    """预留近似后验到标准高斯的 KL 项。

    设计输出：零维 KL 损失；当前不返回。

    Args:
        mean: [b,*z] 后验均值。
        log_variance: 同形对数方差；需明确特征求和和 batch 均值。
    """

    D_kl = 0.5 * (torch.pow(mean, 2) + torch.exp(log_variance) - 1 - log_variance)

    return D_kl.flatten(start_dim=1).sum(dim=1).mean()      # .mean() 用于平均化 Batch 维
