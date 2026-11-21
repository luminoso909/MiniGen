"""对 Flow 训练时间应用分段线性的逆累积分布。"""

import torch
from jaxtyping import Float
from torch import Tensor


def flow_training_times(
    uniform: Float[Tensor, " b"],
    middle_probability: float | None,
) -> Float[Tensor, " b"]:
    """按配置将均匀随机数映射为 Flow 训练时间。

    Args:
        uniform: 同一训练随机流生成的 [b] 均匀样本。
        middle_probability: [0.3,0.7] 的概率。两端平分余量，三段内各自均匀。

    Results:
        Float[Tensor," b"]: [0,1] 的训练时间。None 或 0.4 为原均匀分布。
    """
    if middle_probability is None or middle_probability == 0.4:
        return uniform
    tail = (1 - middle_probability) / 2
    upper = tail + middle_probability
    return torch.where(
        uniform < tail,
        0.3 * uniform / tail,
        torch.where(
            uniform < upper,
            0.3 + 0.4 * (uniform - tail) / middle_probability,
            0.7 + 0.3 * (uniform - upper) / tail,
        ),
    )
