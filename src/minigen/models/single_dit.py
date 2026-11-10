"""B1 简化双向单流 DiT；受联合图文建模启发，不声称复现原模型。"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor, nn

from minigen.contracts import Condition
from minigen.models.joint import JointInputOutput, SingleBlock, StreamConfig, initialize_joint


@dataclass(frozen=True)
class SingleDiTConfig(StreamConfig):
    """定义 B1 的单流深度，其余空间与位置规格继承 StreamConfig。

    Args:
        depth: 单流块数量，正式规格为 11；其他字段含义同 StreamConfig。
    """

    depth: int = 11

    def __post_init__(self) -> None:
        """验证共享规格及单流深度。

        Raises:
            ValueError: 共享尺寸不合法，或深度不是正整数。
        """
        super().__post_init__()
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("Single stream depth must be positive")


class SingleStreamDiT(nn.Module):
    """将图文投影后拼接，通过双向联合注意力共同更新，仅输出图像速度。

    Args:
        config: 默认约 30M 的模型配置。
    """

    def __init__(
        self,
        config: SingleDiTConfig | None = None,
    ) -> None:
        """创建共享输入输出和单流块；生成头与残差门控置零。

        Args:
            config: SingleDiTConfig，None 使用正式默认规格。

        副作用:
            分配并初始化可训练参数，消耗 Torch 随机状态。
        """
        super().__init__()
        self.config = config or SingleDiTConfig()
        self.interface = JointInputOutput(self.config)
        self.blocks = nn.ModuleList(SingleBlock(self.config) for _ in range(self.config.depth))
        initialize_joint(self, self.interface)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """在文本在前的联合流中执行所有块，再还原图像速度。

        Args:
            z: [b,c,h,w] 当前 Flow 潜状态。
            t: [b] 噪声 0 到数据 1 的连续时间。
            condition: [b,n_t,text_dim] 冻结文本和有效掩码。

        Raises:
            ValueError: 输入未满足共享形状、时间或设备约定。

        Results:
            Float[Tensor, "b c h w"]: 与 z 同形的速度，不预测方差。
        """
        parts = self.interface.prepare(z, t, condition)
        tokens = torch.cat((parts.text, parts.image), dim=1)
        for block in self.blocks:
            tokens = block(tokens, parts.time, parts.mask, parts.cosine, parts.sine)
        return self.interface.finish(tokens[:, parts.text.shape[1] :], parts.time)
