"""Small time-conditioned U-Net for 28x28 MNIST; no diffusion math is hidden here.
"""

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        width: int = 128,
    ):
        super().__init__()
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(
        self,
        time: Int[Tensor, "b"],
    ) -> Float[Tensor, "b d"]:
        angles = time.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        # 此处 sin 和 cos 块不是交叉的，实际上变成交叉型只需乘一个矩阵；
        # 但是由于紧接着是 mlp，存在线性层矩阵，所以两个矩阵合并实际上是一个矩阵，
        # 事先变不变交叉对于神经网络而言没有任何影响


class TimeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_width: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.time = nn.Linear(time_width, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        # 1x1 卷积，用于残差连接保证 image 和 hidden 维度形状都一致
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        image: Float[Tensor, "b ci h w"],
        time: Float[Tensor, "b d"],
    ) -> Float[Tensor, "b co h w"]:
        hidden = F.silu(self.norm1(self.conv1(image)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(image))  # 残差连接


class TinyUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28, predicting one noise channel."""

    def __init__(
        self,
        in_channels=1,
        base: int = 32,
        time_width: int = 128,
    ):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_width)
        self.enc1 = TimeBlock(in_channels, base, time_width)
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)
        self.enc2 = TimeBlock(base, base * 2, time_width)
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)

        self.middle = TimeBlock(base * 2, base * 4, time_width)

        self.dec2 = TimeBlock(base * 6, base * 2, time_width)
        self.dec1 = TimeBlock(base * 3, base, time_width)
        self.out = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(
        self,
        image: Float[Tensor, "b 1 h w"],
        timesteps: Int[Tensor, "b"],
    ) -> Float[Tensor, "b 1 h w"]:
        time = self.time_embedding(timesteps)
        skip1 = self.enc1(image, time)
        skip2 = self.enc2(self.down1(skip1), time)
        hidden = self.middle(self.down2(skip2), time)

        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.dec2(torch.cat([hidden, skip2], dim=1), time)

        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.dec1(torch.cat([hidden, skip1], dim=1), time)
        return self.out(hidden)
