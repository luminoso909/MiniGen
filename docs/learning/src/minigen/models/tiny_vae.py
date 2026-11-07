"""简化 VAE 编解码外壳；重参数化与 KL 由调用方提供。"""

from jaxtyping import Float
from torch import Tensor, nn

from minigen._typing import ReparameterizeFn


class TinyVAE(nn.Module):
    """连接 MNIST 的 MLP 编解码器，分布核心由调用方显式提供。

    Args:
        latent_dim: 每张图的潜变量维数，必须为正。

    Raises:
        ValueError: latent_dim 非正时主动抛出。

    Results:
        TinyVAE: TinyVAE 实例，支持编码分布参数、解码 logits 和显式重参数化前向。

    副作用:
        注册编码器、分布头与解码器；参数初始化会消耗 Torch 全局 RNG。
    """

    def __init__(
        self,
        latent_dim: int = 16,
    ) -> None:
        """建立 784 维图像与潜变量之间的 MLP 连接。

        Args:
            latent_dim: 均值、log variance 和解码器输入的共同宽度。

        Raises:
            ValueError: latent_dim 小于或等于 0。

        副作用:
            注册编码器、分布头与解码器；参数初始化会消耗 Torch 全局 RNG。
        """
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, latent_dim)
        self.log_variance_head = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def encode(
        self,
        images: Float[Tensor, "b 1 28 28"],
    ) -> tuple[Float[Tensor, "b z"], Float[Tensor, "b z"]]:
        """编码图像，分别预测对角高斯的均值和对数方差。

        Args:
            images: [b,1,28,28] 灰度图；训练工程使用 [0,1] 像素范围。

        Raises:
            ValueError: 输入不是四维或单图形状不是 [1,28,28]。

        Results:
            tuple[Float[Tensor, "b z"], Float[Tensor, "b z"]]: 两个 [b,z] 张量，依次为 mean 和
                log_variance。
        """
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("Expected [B,1,28,28]")
        
        hidden = self.encoder(images)
        return self.mean_head(hidden), self.log_variance_head(hidden)

    def decode(
        self,
        latent: Float[Tensor, "b z"],
    ) -> Float[Tensor, "b 1 28 28"]:
        """从潜变量还原 MNIST 图像的未归一化 logits。

        Args:
            latent: [b,z]，最后一维等于 latent_dim。

        Raises:
            RuntimeError: 潜变量宽度不匹配解码线性层时由 PyTorch 透传。

        Results:
            Float[Tensor, "b 1 28 28"]: [b,1,28,28] logits，展示概率时需显式 sigmoid。
        """
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent of shape [B, {self.latent_dim}], "
                f"got {tuple(latent.shape)}"
            )
        return self.decoder(latent).reshape(-1, 1, 28, 28)

    def forward(
        self,
        images: Float[Tensor, "b 1 28 28"],
        noise: Float[Tensor, "b z"],
        *,
        reparameterize_fn: ReparameterizeFn,
    ) -> tuple[Float[Tensor, "b 1 28 28"], Float[Tensor, "b z"], Float[Tensor, "b z"]]:
        """显式调用传入的分布核心完成编码、采样与解码。

        Args:
            images: [b,1,28,28] 输入图像。
            noise: [b,z] 标准噪声，供重参数化使用。
            reparameterize_fn: 接收均值、log variance 和噪声的函数；不自动选择教学实现。

        Raises:
            ValueError: 图像形状错误，由 encode 抛出；重参数化函数的异常原样透传。

        Results:
            tuple[Float[Tensor, "b 1 28 28"], Float[Tensor, "b z"], Float[Tensor, "b z"]]: 重建 logits、均值、log variance 三个张量。
        """
        mean, log_variance = self.encode(images)
        latent = reparameterize_fn(mean, log_variance, noise)
        return self.decode(latent), mean, log_variance
