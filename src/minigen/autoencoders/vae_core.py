"""L02 / S1-06：个人对角高斯重参数化与 KL 接口。"""

from jaxtyping import Float
from torch import Tensor

from minigen._learning import pending


def reparameterize(
    mean: Float[Tensor, " batch *latent"],
    log_variance: Float[Tensor, " batch *latent"],
    noise: Float[Tensor, " batch *latent"],
) -> Float[Tensor, " batch *latent"]:
    """预留对角高斯的可微重参数化。

    1 输入：
        mean：[batch,*latent] 后验均值。
        log_variance：与 mean 同形的对数方差。
        noise：与 mean 同形的独立标准高斯噪声。

    2 输出：
        设计返回同形潜变量，并保留分布参数梯度；当前不返回。

    3 报错（Raises）：
        CoreImplementationRequired：L02 重参数化核心尚待个人实现。
    """
    pending("L02 reparameterize")


def kl_standard_normal(
    mean: Float[Tensor, " batch *latent"], log_variance: Float[Tensor, " batch *latent"]
) -> Float[Tensor, ""]:
    """预留近似后验到标准高斯的 KL 项。

    1 输入：
        mean：[batch,*latent] 后验均值。
        log_variance：同形对数方差；需明确特征求和和 batch 均值。

    2 输出：
        设计返回零维 KL 损失；当前不返回。

    3 报错（Raises）：
        CoreImplementationRequired：L02 KL 核心尚待个人实现。
    """
    pending("L02 kl_standard_normal")
