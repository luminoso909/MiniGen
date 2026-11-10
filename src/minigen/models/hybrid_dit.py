"""B2 双流 Joint Attention 到单流的简化 Hybrid DiT，无预训练主干。"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from minigen.contracts import Condition
from minigen.models.joint import (
    DoubleBlock,
    JointInputOutput,
    SingleBlock,
    StreamConfig,
    initialize_joint,
)


@dataclass(frozen=True)
class HybridDiTConfig(StreamConfig):
    """定义两段深度，保持双流到单流的融合边界可检查。

    Args:
        double_depth: 独立图文参数的双流块数量，正式为 3。
        single_depth: 融合后的单流块数量，正式为 5；其他字段含义同 StreamConfig。
    """

    double_depth: int = 3
    single_depth: int = 5

    def __post_init__(self) -> None:
        """验证共享规格，并要求两类块都实际存在。

        Raises:
            ValueError: 共享规格无效或任一深度不是正整数。
        """
        super().__post_init__()
        if any(
            type(depth) is not int or depth < 1 for depth in (self.double_depth, self.single_depth)
        ):
            raise ValueError("Hybrid requires positive double and single depths")


class HybridDiT(nn.Module):
    """前段独立图文投影/FFN 共享注意力，后段拼接为统一 token 流。

    Args:
        config: 默认约 30M 的双流/单流模型配置。
    """

    def __init__(
        self,
        config: HybridDiTConfig | None = None,
    ) -> None:
        """建立两段主干并执行共享零门控初始化。

        Args:
            config: HybridDiTConfig，None 使用正式默认规格。

        副作用:
            初始化参数及固定位置缓存，消耗 Torch 随机状态。
        """
        super().__init__()
        self.config = config or HybridDiTConfig()
        self.interface = JointInputOutput(self.config)
        self.double_blocks = nn.ModuleList(
            DoubleBlock(self.config) for _ in range(self.config.double_depth)
        )
        self.single_blocks = nn.ModuleList(
            SingleBlock(self.config) for _ in range(self.config.single_depth)
        )
        initialize_joint(self, self.interface)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """先保持图文双流，随后在明确的拼接边界合流并输出图像速度。

        Args:
            z: [b,c,h,w] 当前 Flow 潜状态。
            t: [b] 位于 [0,1] 的连续时间。
            condition: [b,n_t,text_dim] 文本特征及有效掩码。

        Raises:
            ValueError: 输入未满足共享形状、时间或设备约定。

        Results:
            Float[Tensor, "b c h w"]: 同形图像速度，文本部分不参与输出损失。
        """
        parts = self.interface.prepare(z, t, condition)
        image, text = parts.image, parts.text
        for block in self.double_blocks:
            image, text = block(image, text, parts.time, parts.mask, parts.cosine, parts.sine)
        tokens = torch.cat((text, image), dim=1)
        for block in self.single_blocks:
            tokens = block(tokens, parts.time, parts.mask, parts.cosine, parts.sine)
        return self.interface.finish(tokens[:, text.shape[1] :], parts.time)
