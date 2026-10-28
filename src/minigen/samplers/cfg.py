"""L04 / S3-02：个人无分类器引导组合接口。"""

from jaxtyping import Float
from torch import Tensor

from minigen._learning import pending


def classifier_free_guidance(
    unconditioned: Float[Tensor, "..."], conditioned: Float[Tensor, "..."], scale: float
) -> Float[Tensor, "..."]:
    """预留无条件预测加条件差值的 CFG 组合。

    1 输入：
        unconditioned：任意形状的浮点无条件预测。
        conditioned：与无条件分支同形的条件预测。
        scale：差值权重，0 对应无条件，1 对应条件预测。

    2 输出：
        设计返回与两个分支同形的组合预测；当前不返回。

    3 报错（Raises）：
        CoreImplementationRequired：L04 CFG 尚待个人实现。
    """
    pending("L04 classifier_free_guidance")
