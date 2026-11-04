"""L04 / S3-01 个人文本 Cross-Attention 接口；不自动替换为教学参考。"""

from jaxtyping import Bool, Float
from torch import Tensor, nn

from minigen._learning import pending


class TextCrossAttention(nn.Module):
    """预留图像 query 检索文本 key/value 的个人核心。

    Args:
        hidden: 图像特征宽度 d，需能被 heads 整除。
        text_width: 文本特征宽度 d_c，CLIP 默认 512。
        heads: 注意力头数。

    Results:
        TextCrossAttention: 学习接口实例，尚未创建内部投影；forward 明确报待实现。

    副作用:
        保存宽度与头数约定；不复制 Notebook 答案。
    """

    def __init__(
        self,
        hidden: int = 128,
        text_width: int = 512,
        heads: int = 4,
    ) -> None:
        """记录接口尺寸，内部 q/kv/out 投影留给用户建立。

        Args:
            hidden: 图像 token 宽度。
            text_width: 文本 token 宽度。
            heads: 注意力头数。

        副作用:
            保存学习参数；当前没有可训练投影。
        """
        super().__init__()
        self.hidden, self.text_width, self.heads = hidden, text_width, heads

    def forward(
        self,
        image: Float[Tensor, "b n_x d"],
        text: Float[Tensor, "b n_y d_c"],
        mask: Bool[Tensor, "b n_y"],
    ) -> Float[Tensor, "b n_x d"]:
        """预留带有效位 mask 的多头文本检索。

        设计输出与 image 同形，当前只抛出待实现异常；残差和门控由外层负责。

        Args:
            image: 图像 query 来源，n_x 为图像 token 数。
            text: 文本 key/value 来源，n_y 为文本 token 数，不要求等于 n_x。
            mask: True 表示有效文本；每条描述至少一个有效位。

        Raises:
            CoreImplementationRequired: L04 文本注意力主体尚待用户实现。
        """
        pending("L04 TextCrossAttention")
