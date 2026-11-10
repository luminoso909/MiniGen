"""B1/B2 共享图文接口、二维旋转位置和显式联合注意力，不依赖教学实现。"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

from minigen.contracts import Condition
from minigen.models.plain_dit import patchify, sincos, unpatchify


@dataclass(frozen=True)
class StreamConfig:
    """固定图文联合模型的空间、头宽和位置约定。

    Args:
        hidden_size: token 宽度 d。
        num_heads: 头数 a，每头宽度必须被 4 整除。
        patch_size: 潜空间 patch 边长。
        latent_size: 方形潜空间边长，正式为 32。
        in_channels: 潜空间通道，正式为 4。
        text_dim: 冻结文本宽度，正式为 512。
        mlp_ratio: MLP 的正整数扩张倍数。
        position_encoding: 固定为 image_2d_rope_text_clip，文本不另旋转。
    """

    hidden_size: int = 384
    num_heads: int = 6
    patch_size: int = 2
    latent_size: int = 32
    in_channels: int = 4
    text_dim: int = 512
    mlp_ratio: int = 4
    position_encoding: str = "image_2d_rope_text_clip"

    def __post_init__(self) -> None:
        """在构建模块前拒绝不支持的维度和位置定义。

        Raises:
            ValueError: 非正整数、不能整除或位置方案不支持。
        """
        values = (
            self.hidden_size,
            self.num_heads,
            self.patch_size,
            self.latent_size,
            self.in_channels,
            self.text_dim,
            self.mlp_ratio,
        )
        if (
            any(type(value) is not int or value < 1 for value in values)
            or self.hidden_size % self.num_heads
            or (self.hidden_size // self.num_heads) % 4
            or self.latent_size % self.patch_size
            or self.position_encoding != "image_2d_rope_text_clip"
        ):
            raise ValueError("Invalid stream dimensions or position encoding")


class JointTokens(NamedTuple):
    """保存一次前向共用的已投影流和联合位置，顺序始终文本在前。

    Args:
        image: [b,n_i,d] 图像 token。
        text: [b,n_t,d] 文本 token，padding 清零。
        time: [b,d] 时间调制特征。
        mask: [b,n] 联合 key mask，n=n_t+n_i。
        cosine: [n,d_h/2] 旋转角余弦，文本位置全部为 1。
        sine: [n,d_h/2] 旋转角正弦，文本位置全部为 0。
    """

    image: Float[Tensor, "b n_i d"]
    text: Float[Tensor, "b n_t d"]
    time: Float[Tensor, "b d"]
    mask: Bool[Tensor, "b n"]
    cosine: Float[Tensor, "n d_r"]
    sine: Float[Tensor, "n d_r"]


def image_rotary(
    side: int,
    head_width: int,
) -> tuple[Float[Tensor, "n_i d_r"], Float[Tensor, "n_i d_r"]]:
    """为行优先网格构造二维 RoPE；一半旋转对用于行，另一半用于列。

    Args:
        side: 方形图像 token 网格边长。
        head_width: 每个注意力头的宽度 d_h，须被 4 整除。

    Raises:
        ValueError: 网格或每头维度非法。

    Results:
        tuple: [side²,d_h/2] 的 FP32 cos 与 sin；每个值对应一对相邻特征。
    """
    if side < 1 or head_width < 4 or head_width % 4:
        raise ValueError("2D rotary needs positive grid and head width divisible by four")
    rows, cols = torch.meshgrid(
        torch.arange(side).float(), torch.arange(side).float(), indexing="ij"
    )
    quarter = head_width // 4
    frequencies = torch.exp(-math.log(10000) * torch.arange(quarter).float() / quarter)
    angles = torch.cat(
        (rows.flatten()[:, None] * frequencies, cols.flatten()[:, None] * frequencies), dim=-1
    )
    return angles.cos(), angles.sin()


def apply_rotary(
    values: Float[Tensor, "b a n d_h"],
    cosine: Float[Tensor, "n d_r"],
    sine: Float[Tensor, "n d_r"],
) -> Float[Tensor, "b a n d_h"]:
    """按相邻偶奇维旋转 Q 或 K，不旋转 V。

    Args:
        values: [b,a,n,d_h] 多头特征。
        cosine: [n,d_h/2] 的旋转余弦。
        sine: 与 cosine 同形的正弦。

    Raises:
        ValueError: 位置数量或旋转宽度不匹配。

    Results:
        Float[Tensor, "b a n d_h"]: 形状和 dtype 不变的旋转特征。
    """
    if (
        values.ndim != 4
        or values.shape[-1] % 2
        or cosine.shape != (values.shape[-2], values.shape[-1] // 2)
        or sine.shape != cosine.shape
    ):
        raise ValueError("Rotary tables do not match attention shape")
    paired = values.unflatten(-1, (-1, 2))
    even, odd = paired.unbind(-1)
    cos, sin = cosine.to(values.dtype), sine.to(values.dtype)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def joint_attention(
    qkv: Float[Tensor, "3 b a n d_h"],
    mask: Bool[Tensor, "b n"],
    cosine: Float[Tensor, "n d_r"],
    sine: Float[Tensor, "n d_r"],
) -> Float[Tensor, "b n d"]:
    """执行非因果图文联合注意力，所有有效 query 可读取全部有效 key。

    Args:
        qkv: [3,b,a,n,d_h]，文本 token 在前、图像在后。
        mask: [b,n]，True 为有效 key；图片位置总是有效。
        cosine: 联合 token 的旋转余弦，文本为恒等。
        sine: 联合 token 的旋转正弦，文本为恒等。

    Results:
        Float[Tensor, "b n d"]: 合并各头后的联合特征。
    """
    q, k, v = qkv.unbind(0)
    q, k = apply_rotary(q, cosine, sine), apply_rotary(k, cosine, sine)
    attended = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.0
    )
    b, _, n, _ = q.shape
    return attended.transpose(1, 2).reshape(b, n, -1)


class StreamBranch(nn.Module):
    """一条流的独立 QKV、输出投影、MLP 与六组时间调制。

    Args:
        config: 共享宽度、头数和 MLP 规格。
    """

    def __init__(
        self,
        config: StreamConfig,
    ) -> None:
        """创建可被单流使用或被双流分别实例化的参数集合。

        Args:
            config: 网络规格。

        副作用:
            初始化参数，消耗 Torch 随机状态；调制头初始化为零。
        """
        super().__init__()
        d = config.hidden_size
        self.heads = config.num_heads
        self.norms = nn.ModuleList(
            nn.LayerNorm(d, elementwise_affine=False, eps=1e-6) for _ in range(2)
        )
        self.qkv = nn.Linear(d, 3 * d)
        self.projection = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.mlp_ratio * d),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.mlp_ratio * d, d),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def prepare(
        self,
        x: Float[Tensor, "b n d"],
        time: Float[Tensor, "b d"],
    ) -> tuple[Float[Tensor, "3 b a n d_h"], Float[Tensor, "b 1 d_m"]]:
        """对时间调制后的 token 计算独立 QKV，并保留残差所需调制。

        Args:
            x: [b,n,d] 当前流。
            time: [b,d] 时间特征。

        Results:
            tuple: [3,b,a,n,d_h] QKV 与 [b,1,6d] 调制值。
        """
        modulation = self.modulation(time).unsqueeze(1)
        shift, scale, *_ = modulation.chunk(6, dim=-1)
        normalized = self.norms[0](x) * (1 + scale) + shift
        b, n, d = x.shape
        qkv = self.qkv(normalized).reshape(b, n, 3, self.heads, d // self.heads)
        return qkv.permute(2, 0, 3, 1, 4), modulation

    def finish(
        self,
        x: Float[Tensor, "b n d"],
        attended: Float[Tensor, "b n d"],
        modulation: Float[Tensor, "b 1 d_m"],
    ) -> Float[Tensor, "b n d"]:
        """使用本流独立投影及 MLP 完成注意力和前馈残差。

        Args:
            x: [b,n,d] 更新前的流。
            attended: 联合注意力中对应本流的 [b,n,d] 切片。
            modulation: [b,1,6d]，由同一前向的 prepare 产生。

        Results:
            Float[Tensor, "b n d"]: 两个残差更新后的流。
        """
        _, _, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, dim=-1)
        x = x + gate_a * self.projection(attended)
        return x + gate_m * self.mlp(self.norms[1](x) * (1 + scale_m) + shift_m)


class SingleBlock(StreamBranch):
    """在合并序列上共享一组 QKV/MLP；文本、图像双向交互。

    Args:
        config: 传入继承的 StreamBranch 构造函数的网络规格。
    """

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        time: Float[Tensor, "b d"],
        mask: Bool[Tensor, "b n"],
        cosine: Float[Tensor, "n d_r"],
        sine: Float[Tensor, "n d_r"],
    ) -> Float[Tensor, "b n d"]:
        """联合注意力后清空无效 query，使 padding 不携带后续层状态。

        Args:
            x: 文本在前的 [b,n,d] 联合流。
            time: [b,d] 时间特征。
            mask: [b,n] 有效位置，图像段全部为 True。
            cosine: 联合旋转余弦。
            sine: 联合旋转正弦。

        Results:
            Float[Tensor, "b n d"]: 更新后的联合流，padding 为零。
        """
        qkv, modulation = self.prepare(x, time)
        attended = joint_attention(qkv, mask, cosine, sine)
        return self.finish(x, attended, modulation).masked_fill(~mask[..., None], 0)


class DoubleBlock(nn.Module):
    """图文各自拥有 QKV、MLP 和调制参数，只在注意力矩阵中交互。

    Args:
        config: 共享维度及头数，不代表参数共享。
    """

    def __init__(
        self,
        config: StreamConfig,
    ) -> None:
        """建立两个独立参数分支。

        Args:
            config: 网络规格。

        副作用:
            初始化图文两套参数并消耗 Torch 随机状态。
        """
        super().__init__()
        self.image_branch = StreamBranch(config)
        self.text_branch = StreamBranch(config)

    def forward(
        self,
        image: Float[Tensor, "b n_i d"],
        text: Float[Tensor, "b n_t d"],
        time: Float[Tensor, "b d"],
        mask: Bool[Tensor, "b n"],
        cosine: Float[Tensor, "n d_r"],
        sine: Float[Tensor, "n d_r"],
    ) -> tuple[Float[Tensor, "b n_i d"], Float[Tensor, "b n_t d"]]:
        """拼接独立 QKV 做联合注意力，随后拆回各自分支更新。

        Args:
            image: [b,n_i,d] 图像流。
            text: [b,n_t,d] 文本流。
            time: [b,d] 时间条件。
            mask: [b,n_t+n_i] 的有效位置。
            cosine: 文本在前的联合旋转余弦。
            sine: 文本在前的联合旋转正弦。

        Results:
            tuple: 更新后的图像流和文本流；文本 padding 为零。
        """
        qi, mi = self.image_branch.prepare(image, time)
        qt, mt = self.text_branch.prepare(text, time)
        attended = joint_attention(torch.cat((qt, qi), dim=3), mask, cosine, sine)
        n_t = text.shape[1]
        image = self.image_branch.finish(image, attended[:, n_t:], mi)
        text = self.text_branch.finish(text, attended[:, :n_t], mt)
        return image, text.masked_fill(~mask[:, :n_t, None], 0)


class JointInputOutput(nn.Module):
    """B1/B2 共用输入、模态身份、时间与速度输出，避免复制主干外逻辑。

    Args:
        config: 已验证的模型配置。
    """

    def __init__(
        self,
        config: StreamConfig,
    ) -> None:
        """创建投影和固定旋转表，旋转表随模型迁移设备但不训练。

        Args:
            config: 潜空间与 token 规格。

        副作用:
            注册参数、位置缓存并消耗 Torch 随机状态。
        """
        super().__init__()
        self.config = config
        d, p = config.hidden_size, config.patch_size
        self.patch_embed = nn.Linear(p * p * config.in_channels, d)
        self.text_embed = nn.Linear(config.text_dim, d)
        self.time_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.modality = nn.Parameter(torch.zeros(2, d))
        cos, sin = image_rotary(config.latent_size // p, d // config.num_heads)
        self.register_buffer("image_cosine", cos, persistent=True)
        self.register_buffer("image_sine", sin, persistent=True)
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_projection = nn.Linear(d, p * p * config.in_channels)

    def prepare(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> JointTokens:
        """检查 Flow 契约并建立联合流的图文、时间、mask 与位置。

        Args:
            z: [b,c,h,w] 已缩放潜状态，正式尺寸 [b,4,32,32]。
            t: [b] 连续时间 [0,1]，嵌入前乘 1000。
            condition: [b,n_t,text_dim] 冻结文本特征及掩码。

        Raises:
            ValueError: 形状、时间、设备或文本宽度不符。

        Results:
            JointTokens: 投影后的双流、时间条件和联合可见性/位置。
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
            raise ValueError("Joint DiT requires configured latents, text and time in [0,1]")
        image = self.patch_embed(patchify(z, cfg.patch_size)) + self.modality[0]
        text = self.text_embed(condition.features.masked_fill(~condition.mask[..., None], 0))
        text = (text + self.modality[1]).masked_fill(~condition.mask[..., None], 0)
        time = self.time_embed(sincos(t * 1000, cfg.hidden_size).to(image.dtype))
        n_t = text.shape[1]
        # 文本保留 CLIP 位置，不引入虚构的二维坐标；模态参数与图像旋转共同区分两流。
        cosine = torch.cat(
            (self.image_cosine.new_ones(n_t, self.image_cosine.shape[1]), self.image_cosine)
        )
        sine = torch.cat(
            (self.image_sine.new_zeros(n_t, self.image_sine.shape[1]), self.image_sine)
        )
        mask = torch.cat(
            (condition.mask, condition.mask.new_ones(z.shape[0], image.shape[1])), dim=1
        )
        return JointTokens(image, text, time, mask, cosine, sine)

    def finish(
        self,
        image: Float[Tensor, "b n_i d"],
        time: Float[Tensor, "b d"],
    ) -> Float[Tensor, "b c h w"]:
        """只从图像 token 预测速度，不为文本位置定义图像损失。

        Args:
            image: [b,n_i,d] 主干最后的图像段。
            time: [b,d] 与输入相同的时间特征。

        Results:
            Float[Tensor, "b c h w"]: 与输入潜状态同形的速度。
        """
        shift, scale = self.final_modulation(time).unsqueeze(1).chunk(2, dim=-1)
        values = self.final_projection(self.final_norm(image) * (1 + scale) + shift)
        cfg = self.config
        return unpatchify(values, cfg.in_channels, cfg.latent_size, cfg.latent_size, cfg.patch_size)


def initialize_joint(
    model: nn.Module,
    interface: JointInputOutput,
) -> None:
    """统一初始化所有投影，再将残差门控和速度头置零。

    Args:
        model: 完整 Single/Hybrid 模型，包含待初始化的分支。
        interface: 该模型持有的共享输入输出接口。

    副作用:
        原位重设参数并消耗 Torch 随机状态。
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    for module in model.modules():
        if isinstance(module, StreamBranch):
            nn.init.zeros_(module.modulation[-1].weight)
            nn.init.zeros_(module.modulation[-1].bias)
    for layer in (interface.time_embed[0], interface.time_embed[2]):
        nn.init.normal_(layer.weight, std=0.02)
    nn.init.normal_(interface.modality, std=0.02)
    for layer in (interface.final_modulation[-1], interface.final_projection):
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
