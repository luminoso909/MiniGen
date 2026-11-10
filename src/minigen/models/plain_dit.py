"""P3 B0：时间 adaLN-Zero、图像自注意力与文本交叉注意力。

工程来源：复用个人 S2 切块顺序及 DiT 的固定二维编码和零初始化思想；
本文件由 Agent 实现连续时间/速度输出与文本工程路径，不代表个人 TODO 验收。
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from minigen.contracts import Condition
from minigen.models.attention import attention


@dataclass(frozen=True)
class PlainDiTConfig:
    """定义可序列化的 B0 规格，正式默认保留 256px 对应的 32px 潜空间。

    Args:
        hidden_size: token 宽度 d，必须被 4 和头数整除。
        num_heads: 注意力头数。
        depth: 图像自注意力/文本交叉注意力 Block 数量。
        patch_size: 潜空间 patch 边长，整除 latent_size。
        latent_size: 正方形潜空间边长；正式规格为 32。
        in_channels: 潜变量通道 c；正式冻结 VAE 为 4。
        text_dim: 冻结文本特征宽度；CLIP-B/32 为 512。
        mlp_ratio: MLP 中间宽度与 d 的正整数比值。
    """

    hidden_size: int = 384
    num_heads: int = 6
    depth: int = 8
    patch_size: int = 2
    latent_size: int = 32
    in_channels: int = 4
    text_dim: int = 512
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        """在分配模型前检查规格的整除约束。

        Raises:
            ValueError: 规格不是正整数，或空间/头宽/位置宽度无法整除。
        """
        values = (
            self.hidden_size,
            self.num_heads,
            self.depth,
            self.patch_size,
            self.latent_size,
            self.in_channels,
            self.text_dim,
            self.mlp_ratio,
        )
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or self.hidden_size % self.num_heads
            or self.hidden_size % 4
            or self.latent_size % self.patch_size
        ):
            raise ValueError("Invalid PlainDiT dimensions or divisibility")


def sincos(
    values: Float[Tensor, " n"],
    width: int,
) -> Float[Tensor, "n d"]:
    """将连续坐标编码为 FP32 多频率正弦余弦，顺序为 cos 后 sin。

    Args:
        values: [n] 连续时间或空间坐标。
        width: 正偶数输出维度。

    Raises:
        ValueError: 输出宽度不是正偶数。

    Results:
        Float[Tensor, "n d"]: [n,width] 固定频率特征。
    """
    if width < 2 or width % 2:
        raise ValueError("Sine/cosine width must be positive and even")
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=values.device).float() / (width // 2)
    )
    angles = values.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def patchify(
    latents: Float[Tensor, "b c h w"],
    patch_size: int,
) -> Float[Tensor, "b n d_p"]:
    """按行优先排列空间块，每块内部先高、再宽、最后通道。

    Args:
        latents: [b,c,h,w] 空间潜变量。
        patch_size: 同时整除 h/w 的正整数边长。

    Raises:
        ValueError: 输入维数或空间整除条件不满足。

    Results:
        Float[Tensor, "b n d_p"]: n=(h/p)*(w/p)，d_p=p*p*c。
    """
    if (
        latents.ndim != 4
        or patch_size <= 0
        or latents.shape[2] % patch_size
        or latents.shape[3] % patch_size
    ):
        raise ValueError("Expected divisible [b,c,h,w] latents")
    b, c, h, w = latents.shape
    p = patch_size
    return (
        latents.reshape(b, c, h // p, p, w // p, p)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(b, h // p * (w // p), p * p * c)
    )


def unpatchify(
    tokens: Float[Tensor, "b n d_p"],
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> Float[Tensor, "b c h w"]:
    """按与 patchify 相反的排列恢复潜空间。

    Args:
        tokens: [b,n,d_p] 原始 patch 特征。
        channels: 目标通道 c。
        height: 目标高度 h。
        width: 目标宽度 w。
        patch_size: 切块时的边长 p。

    Raises:
        ValueError: 目标尺寸或 token 形状不一致。

    Results:
        Float[Tensor, "b c h w"]: [b,c,h,w] 空间张量。
    """
    p = patch_size
    if (
        min(channels, height, width, p) <= 0
        or height % p
        or width % p
        or tokens.ndim != 3
        or tokens.shape[1:] != (height // p * (width // p), p * p * channels)
    ):
        raise ValueError("Patch tokens do not match target shape")
    b = tokens.shape[0]
    return (
        tokens.reshape(b, height // p, width // p, p, p, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(b, channels, height, width)
    )


class Attention(nn.Module):
    """使用 SDPA 计算多头注意力，自注意力与交叉注意力共享形状约定。

    Args:
        width: 输入和输出的宽度 d。
        heads: 注意力头数。
    """

    def __init__(
        self,
        width: int,
        heads: int,
    ) -> None:
        """创建独立 Q/K/V 及输出投影。

        Args:
            width: token 宽度。
            heads: 整除 width 的头数。

        Raises:
            ValueError: 头数或宽度不符。

        副作用:
            初始化可训练参数并消耗 PyTorch 随机状态。
        """
        super().__init__()
        if heads <= 0 or width <= 0 or width % heads:
            raise ValueError("Attention width must be divisible by positive heads")
        self.attention_backend = "sdpa"
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(
        self,
        query: Float[Tensor, "b n d"],
        context: Float[Tensor, "b m d"],
        mask: Bool[Tensor, "b m"] | None = None,
    ) -> Float[Tensor, "b n d"]:
        """对可见 key 做非因果注意力；SDPA 布尔 True 表示允许访问。

        Args:
            query: [b,n,d] 图像 token。
            context: [b,m,d] 图像或文本 token。
            mask: 可选 [b,m] 布尔 key 可见性。

        Results:
            Float[Tensor, "b n d"]: 聚合并投影回 d 的图像 token。
        """
        b, n, d = query.shape
        q = self.q(query).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        kv = self.kv(context).reshape(b, context.shape[1], 2, self.heads, d // self.heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        visible = None if mask is None else mask[:, None, None, :]
        values = attention(q, k, v, visible, self.attention_backend)
        return self.out(values.transpose(1, 2).reshape(b, n, d))


class PlainBlock(nn.Module):
    """三个由时间调制的残差分支：图像注意力、文本注意力和 MLP。

    Args:
        config: 统一 B0 规格。
    """

    def __init__(
        self,
        config: PlainDiTConfig,
    ) -> None:
        """分配三个非仿射 LayerNorm 和九组 adaLN 调制参数。

        Args:
            config: 宽度、头数和 MLP 扩张比例。

        副作用:
            创建可训练模块；调制头零初始化，使 Block 初始为恒等映射。
        """
        super().__init__()
        d = config.hidden_size
        self.norms = nn.ModuleList(
            nn.LayerNorm(d, elementwise_affine=False, eps=1e-6) for _ in range(3)
        )
        self.self_attention = Attention(d, config.num_heads)
        self.cross_attention = Attention(d, config.num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * config.mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(d * config.mlp_ratio, d),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        time: Float[Tensor, "b d"],
        condition: Condition,
    ) -> Float[Tensor, "b n d"]:
        """将 shift/scale/gate 广播到 token 轴，再逐分支更新图像流。

        Args:
            x: [b,n,d] 图像 token。
            time: [b,d] 时间 MLP 特征。
            condition: 已投影到 d 的文本和有效掩码。

        Results:
            Float[Tensor, "b n d"]: 保持图像序列长度与宽度的结果。
        """
        shift_a, scale_a, gate_a, shift_c, scale_c, gate_c, shift_m, scale_m, gate_m = (
            self.modulation(time).unsqueeze(1).chunk(9, dim=-1)
        )
        normalized = self.norms[0](x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.self_attention(normalized, normalized)
        normalized = self.norms[1](x) * (1 + scale_c) + shift_c
        x = x + gate_c * self.cross_attention(normalized, condition.features, condition.mask)
        normalized = self.norms[2](x) * (1 + scale_m) + shift_m
        return x + gate_m * self.mlp(normalized)


class PlainConditionalDiT(nn.Module):
    """B0 连续时间条件 DiT，输出速度，方差与 VAE/CLIP 均不在主干内学习。

    Args:
        config: 可序列化的网络规格，默认约 30M 参数。
    """

    def __init__(
        self,
        config: PlainDiTConfig | None = None,
    ) -> None:
        """建立图像、文本与时间投影，并执行 adaLN-Zero 初始化。

        Args:
            config: 网络规格。

        副作用:
            初始化参数和固定二维位置缓存，消耗 PyTorch 随机状态。
        """
        super().__init__()
        config = config or PlainDiTConfig()
        self.config = config
        d, p = config.hidden_size, config.patch_size
        self.patch_embed = nn.Linear(config.in_channels * p * p, d)
        self.text_embed = nn.Linear(config.text_dim, d)
        self.time_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        side = config.latent_size // p
        rows, cols = torch.meshgrid(
            torch.arange(side).float(), torch.arange(side).float(), indexing="ij"
        )
        position = torch.cat(
            (sincos(rows.flatten(), d // 2), sincos(cols.flatten(), d // 2)), dim=-1
        )
        self.register_buffer("position", position[None], persistent=True)
        self.blocks = nn.ModuleList(PlainBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_projection = nn.Linear(d, p * p * config.in_channels)
        self.initialize()

    def initialize(self) -> None:
        """初始化投影；残差调制与输出头置零，建立可审查的初始梯度路径。

        副作用:
            原位重置所有 Linear 参数并消耗 PyTorch 随机状态。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in (self.time_embed[0], self.time_embed[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for layer in (self.final_modulation[-1], self.final_projection):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """从空间潜变量、连续时间与有效文本预测同形速度。

        Args:
            z: [b,c,h,w] 已缩放潜状态，尺寸由 config 固定。
            t: [b] 连续时间 [0,1]；内部乘 1000 后做频率嵌入。
            condition: [b,n,text_dim] 文本和 [b,n] 掩码。

        Raises:
            ValueError: 潜形状、时间范围、条件批/宽度或设备不匹配。

        Results:
            Float[Tensor, "b c h w"]: 同形速度；不学习方差、不裁剪速度。
        """
        cfg = self.config
        if (
            z.ndim != 4
            or z.shape[1:] != (cfg.in_channels, cfg.latent_size, cfg.latent_size)
            or not z.is_floating_point()
            or t.shape != (z.shape[0],)
            or not t.is_floating_point()
            or not torch.isfinite(t).all()
            or (t < 0).any()
            or (t > 1).any()
            or condition.features.shape[0] != z.shape[0]
            or condition.features.shape[-1] != cfg.text_dim
            or t.device != z.device
            or condition.features.device != z.device
        ):
            raise ValueError("B0 expected configured latent shape, text batch and time in [0,1]")
        x = self.patch_embed(patchify(z, cfg.patch_size))
        x = x + self.position.to(dtype=x.dtype)
        time = self.time_embed(sincos(t * 1000, cfg.hidden_size).to(dtype=x.dtype))
        # 先清空 padding 再投影，保证其数值对后续交叉注意力不可见。
        text = self.text_embed(condition.features.masked_fill(~condition.mask[..., None], 0))
        projected = Condition(text, condition.mask)
        for block in self.blocks:
            x = block(x, time, projected)
        shift, scale = self.final_modulation(time).unsqueeze(1).chunk(2, dim=-1)
        prediction = self.final_projection(self.final_norm(x) * (1 + scale) + shift)
        return unpatchify(
            prediction, cfg.in_channels, cfg.latent_size, cfg.latent_size, cfg.patch_size
        )
