"""L03 / S2-04：个人空间切块与 adaLN-Zero 核心接口。"""
import math
from einops import rearrange

from jaxtyping import Float, Int
import torch
from torch import Tensor, nn


def sincos_1d(
    values: Float[Tensor, "n"] | Int[Tensor, "n"],
    width: int,
) -> Float[Tensor, "n d"]:
    """生成一维正弦余弦编码，可用于时间步或空间坐标。

    编码由多频率 cos 和 sin 特征拼接得到，不包含可学习参数。

    Args:
        values: [n] 待编码的时间步或坐标值。
        width: 输出编码维度 D，必须为正偶数。

    Returns:
        Float[Tensor, "n d"]:
            [n,width] 正弦余弦编码。
            前 width/2 维为 cos，后 width/2 维为 sin。

    Raises:
        ValueError: width 非正数或不是偶数。

    副作用:
        不修改输入，不创建可学习参数。
        输出位于 values 所在设备。
    """
    if width%2 != 0 or width<=0:
        raise ValueError("Sine/cosine width must be positive and even")
    freqs = torch.exp(
        -math.log(10000) * torch.arange(
            0, 
            width//2, 
            device=values.device,
            dtype=torch.float32
        ) / (width//2)
    ).unsqueeze(0)
    angles = values.float().unsqueeze(-1) * freqs
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def position_2d(
    height: int,
    width: int,
    hidden_size: int,
) -> Float[Tensor, "1 n d"]:
    """生成固定二维正弦余弦位置编码。

    对二维 Patch 网格分别编码行坐标和列坐标，
    每个坐标使用 hidden_size/2 维编码，最后拼接。

    Patch 顺序与 patchify 保持一致：
        从上到下、从左到右。

    Args:
        height: Patch 网格高度 H_p。
        width: Patch 网格宽度 W_p。
        hidden_size: Token 隐藏维度 D，必须能被 4 整除。

    Returns:
        Float[Tensor, "1 n d"]:
            [H_p*W_p, hidden_size] 固定位置编码。
            n = height * width。

    Raises:
        ValueError: height 或 width 非正数，或 hidden_size 不能被 4 整除。

    副作用:
        不创建可学习参数。
        DiT 初始化时通过 register_buffer 保存编码。
    """
    if hidden_size % 4:
        raise ValueError("2D position width must be divisible by four")
    if height <= 0 or width <= 0 or hidden_size <= 0:
        raise ValueError("Invalid position dimensions")
    
    rows, cols = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij"
    )
    return torch.cat(
        [sincos_1d(rows.flatten(), hidden_size // 2), sincos_1d(cols.flatten(), hidden_size // 2)],
        dim=-1
    ).unsqueeze(0)


def patchify(
    latents: Float[Tensor, "b c h w"],
    patch_size: int,
) -> Float[Tensor, "b n d_p"]:
    """预留可逆空间切块，固定 patch 内和块间顺序。

    设计输出： [b,n,d_p]，tokens=H/p×W/p，patch_features=C×p²。

    Args:
        latents: [b,c,h,w] 潜变量。
        patch_size: patch 边长 p，设计要求整除 height 和 width。
    """
    if latents.ndim != 4 or patch_size <= 0 or latents.shape[2] % patch_size or latents.shape[3] % patch_size:
        raise ValueError("Expected divisible [B,C,H,W] input")
    tokens = rearrange(
        latents, 
        "b c (p_h h) (p_w w) -> b (p_h p_w) (h w c)", 
        h = patch_size, 
        w = patch_size, 
    )
    return tokens
    


def unpatchify(
    tokens: Float[Tensor, "b n d_p"],
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> Float[Tensor, "b c h w"]:
    """预留 patchify 的严格逆排列。

    设计输出： [b,c,h,w]；当前不返回。

    Args:
        tokens: [b,n,d_p] 原始 patch 数值。
        channels: 恢复的通道数 C。
        height: 恢复的高度 H。
        width: 恢复的宽度 W。
        patch_size: 切块时使用的同一边长 p。
    """
    if patch_size <= 0 or height % patch_size or width % patch_size or tokens.ndim != 3:
        raise ValueError("Invalid target spatial size")
    
    _, n, k = tokens.shape
    if n != (height // patch_size) * (width // patch_size) or k != patch_size * patch_size * channels:
        raise ValueError("Token count/width mismatch")
    
    latents = rearrange(
        tokens, 
        " b (p_h p_w) (h w c) -> b c (p_h h) (p_w w)", 
        c = channels, 
        h = patch_size, 
        w = patch_size, 
        p_h = height // patch_size, 
        p_w = width // patch_size, 
    )
    return latents


class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Hidden dimension must be divisible by heads")
        self.heads, self.head_width = num_heads, hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, 
        x: Float[Tensor, "b n d"], 
    ) -> Float[Tensor, "b n d"]:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_width)).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.projection(attended)


class FeedForward(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 256, 
        expansion: int = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, expansion * hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(expansion * hidden_size, hidden_size),
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
    ) -> Float[Tensor, "b n d"]:
        return self.net(x)

class DiTBlock(nn.Module):
    """保留带 adaLN-Zero 调制与门控的个人 Transformer 核心。

    Args:
        hidden_size: token 与条件的特征宽度。
        num_heads: 注意力头数，应整除 hidden_size。

    Results:
        DiTBlock: DiTBlock 接口实例，尚未创建内部投影和调制参数。

    副作用:
        保存 hidden_size 和 num_heads；尚未创建内部算法参数。
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
    ) -> None:
        """记录学习接口的宽度与头数，不补写内部算法。

        Args:
            hidden_size: [b,n,d] 中的 hidden。
            num_heads: 多头注意力的头数。

        副作用:
            保存 hidden_size 和 num_heads；尚未创建内部算法参数。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp = FeedForward(hidden_size, expansion=4)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6*hidden_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def modulate(
        x: Float[Tensor, "b n d"], 
        gamma: Float[Tensor, "b d"], 
        beta: Float[Tensor, "b d"], 
    ) -> Float[Tensor, "b n d"]:
        return (1 + gamma[:, None, :]) * x + beta[:, None, :]

    def forward(
        self,
        tokens: Float[Tensor, "b n d"],
        condition: Float[Tensor, "b d"],
    ) -> Float[Tensor, "b n d"]:
        """预留时间或其他条件驱动的 Transformer 残差更新。

        设计输出：同形 token；零门控初始化应为恒等映射，当前不返回。

        Args:
            tokens: [b,n,d] 图像 token。
            condition: [b,d] 条件，调制时沿 token 维广播。

        """
        gamma_a, beta_a, gate_a, gamma_m, beta_m, gate_m = self.modulation(condition).chunk(6, dim=-1)
        hidden = tokens + gate_a[:, None, :] * self.attn(self.modulate(self.norm1(tokens), gamma_a, beta_a))
        out = hidden + gate_m[:, None, :] * self.mlp(self.modulate(self.norm2(hidden), gamma_m, beta_m))
        return out


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        depth: int = 6,
        hidden_size: int = 256,
        num_heads: int = 4,
        channels: int = 4,
        height: int = 16,
        width: int = 16,
        patch_size: int = 2,
    ):
        super().__init__()

        assert height % patch_size == 0 and width % patch_size == 0

        self.hidden_size = hidden_size
        self.channels = channels
        self.height = height
        self.width = width
        self.patch_size = patch_size

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_dim = channels * patch_size ** 2

        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)
        pos_2d = position_2d(
            height // patch_size,
            width // patch_size,
            hidden_size
        )
        self.register_buffer("position_2d", pos_2d, persistent=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output = nn.Linear(hidden_size, self.patch_dim)

        # 一般 Linear 使用 Xavier 初始化
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(init_linear)

        # DiTBlock 的 adaLN-Zero 初始化
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)

        # Final adaLN 初始化
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

        # 最终噪声输出头初始化
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        timesteps: Int[Tensor, "b"],
    ) -> Float[Tensor, "b c h w"]:

        condition = self.time_mlp(sincos_1d(timesteps, self.hidden_size))
        tokens = (self.patch_embed(patchify(z, self.patch_size)) + self.position_2d)

        for block in self.blocks:
            tokens = block(tokens, condition)

        gamma_f, beta_f = self.final_modulation(condition).chunk(2, dim=-1)
        tokens = self.output(
            (1 + gamma_f[:, None, :]) * self.final_norm(tokens) + beta_f[:, None, :]
        )
        out = unpatchify(
            tokens,
            self.channels,
            self.height,
            self.width,
            self.patch_size
        )

        return out
