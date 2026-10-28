"""小型时间条件 U-Net 工程；DDPM 数学由外部调度器负责。"""

import math

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """将离散噪声时间编码成网络可用的连续条件。

    1 输入：
        width：输出宽度，正弦/余弦各占一半，需为匹配线性层的偶数。

    2 输出：
        TimeEmbedding 实例；forward 将 [batch] 时间变为 [batch,width] 特征。

    3 报错（Raises）：
        无显式 raise；构造或方法调用的下层异常原样透传。
    """

    def __init__(
        self,
        width: int = 128,
    ) -> None:
        """初始化固定频率缓冲和可学习的两层时间 MLP。

        1 输入：
            width：频率拼接与 MLP 的宽度；应为偶数。

        2 输出：
            无返回值；频率随模型迁移设备但不参与优化。

        3 报错（Raises）：
            无显式 raise；下层调用的异常原样透传。
        """
        super().__init__()
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(
        self,
        time: Int[Tensor, " batch"],
    ) -> Float[Tensor, "batch hidden"]:
        """对每张图的时间索引生成时间条件。

        1 输入：
            time：[batch] 整数噪声索引，先转浮点再计算各频率角度。

        2 输出：
            [batch,hidden] 浮点条件，hidden 等于初始化的 width。

        3 报错（Raises）：
            RuntimeError：时间维度或频率拼接与线性层输入不兼容时，由 PyTorch 透传。
        """
        # [batch,1] × [1,width/2] 广播成每张图的一组频率角度。
        angles = time.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        # 此处 sin 和 cos 块不是交叉的，实际上变成交叉型只需乘一个矩阵；
        # 但是由于紧接着是 mlp，存在线性层矩阵，所以两个矩阵合并实际上是一个矩阵，
        # 事先变不变交叉对于神经网络而言没有任何影响


class TimeBlock(nn.Module):
    """通过时间偏置调制卷积特征的残差块。

    1 输入：
        in_channels：输入特征通道数。
        out_channels：输出通道数，应可被 GroupNorm 的 8 组整除。
        time_width：传入时间条件的宽度。

    2 输出：
        TimeBlock 实例；保持空间尺寸并把通道转换为 out_channels。

    3 报错（Raises）：
        ValueError：GroupNorm 通道不能被 8 整除时，由构造过程透传。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_width: int,
    ) -> None:
        """创建卷积、组归一化、时间投影和通道对齐残差。

        1 输入：
            in_channels：图像特征输入通道。
            out_channels：两层卷积和残差的输出通道。
            time_width：时间向量最后一维宽度。

        2 输出：
            无返回值；注册可训练子模块。

        3 报错（Raises）：
            ValueError：输出通道不满足 8 组 GroupNorm 的整除条件。
        """
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
        image: Float[Tensor, "batch in_channels height width"],
        time: Float[Tensor, "batch hidden"],
    ) -> Float[Tensor, "batch out_channels height width"]:
        """将时间条件作为通道偏置注入卷积残差块。

        1 输入：
            image：[batch,in_channels,height,width] 特征。
            time：[batch,hidden] 条件，hidden 等于 time_width，投影后沿空间广播。

        2 输出：
            保持 height、width，通道变为 out_channels 的特征。

        3 报错（Raises）：
            RuntimeError：输入通道、时间宽度或 batch 广播不兼容时由张量算子透传。
        """
        hidden = F.silu(self.norm1(self.conv1(image)))
        # 时间向量先投影到输出通道，再沿 height、width 广播。
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(image))  # 残差连接


class TinyUNet(nn.Module):
    """用于小型去噪实验的时间条件 U-Net，支持配置输入与输出通道。

    1 输入：
        in_channels：图像或潜变量通道数，同时决定噪声输出通道。
        base：基础通道数，应满足各层 8 组归一化要求。
        time_width：时间条件宽度，取偶数。

    2 输出：
        TinyUNet 实例；forward 返回与输入同形的噪声预测。

    3 报错（Raises）：
        无显式 raise；构造或方法调用的下层异常原样透传。
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 32,
        time_width: int = 128,
    ) -> None:
        """组装两次下采样、瓶颈和带跳跃连接的上采样网络。

        1 输入：
            in_channels：输入与输出共享的通道数，默认 1。
            base：第一层宽度，后续按倍数展开。
            time_width：时间特征宽度，供所有残差块使用。

        2 输出：
            无返回值；注册完整去噪网络。

        3 报错（Raises）：
            ValueError：通道不满足 GroupNorm 的分组要求时透传。
        """
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
        image: Float[Tensor, "batch channels height width"],
        timesteps: Int[Tensor, " batch"],
    ) -> Float[Tensor, "batch channels height width"]:
        """结合时间条件，预测输入中与其同形的噪声。

        1 输入：
            image：[batch,channels,height,width]，channels 必须等于 in_channels；MNIST 默认 28×28。
            timesteps：[batch] 整数噪声时间，每张图一个索引。

        2 输出：
            [batch,channels,height,width] 噪声预测，不执行加噪或反向采样。

        3 报错（Raises）：
            RuntimeError：通道、时间 batch、空间尺寸或设备与网络算子不兼容时透传。
        """
        time = self.time_embedding(timesteps)
        skip1 = self.enc1(image, time)
        skip2 = self.enc2(self.down1(skip1), time)
        hidden = self.middle(self.down2(skip2), time)

        # 显式对齐跳跃分支的空间尺寸，避免两次下采样后的奇数尺寸误差。
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
        hidden = self.dec2(torch.cat([hidden, skip2], dim=1), time)

        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
        hidden = self.dec1(torch.cat([hidden, skip1], dim=1), time)
        return self.out(hidden)
