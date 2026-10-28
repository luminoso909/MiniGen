"""L03 / S2-04：个人空间切块与 adaLN-Zero 核心接口。"""

from jaxtyping import Float
from torch import Tensor, nn

from minigen._learning import pending


def patchify(
    latents: Float[Tensor, "b c h w"],
    patch_size: int,
) -> Float[Tensor, "b n d_p"]:
    """预留可逆空间切块，固定 patch 内和块间顺序。

    设计输出： [b,n,d_p]，tokens=H/p×W/p，patch_features=C×p²。

    Args:
        latents: [b,c,h,w] 潜变量。
        patch_size: patch 边长 p，设计要求整除 height 和 width。

    Raises:
        CoreImplementationRequired: L03 patchify 核心尚待个人实现。
    """
    pending("L03 patchify")


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

    Raises:
        CoreImplementationRequired: L03 unpatchify 核心尚待个人实现。
    """
    pending("L03 unpatchify")


class DiTBlock(nn.Module):
    """保留带 adaLN-Zero 调制与门控的个人 Transformer 核心。

    Args:
        hidden_size: token 与条件的特征宽度。
        num_heads: 注意力头数，应整除 hidden_size。

    Raises:
        CoreImplementationRequired: 调用 forward 时主动抛出。

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

        Raises:
            CoreImplementationRequired: L03 DiT Block 与 adaLN-Zero 尚待个人实现。
        """
        pending("L03 DiTBlock / adaLN-Zero")
