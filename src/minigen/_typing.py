"""静态接口与异构数据别名；导入本模块不会加载可选模型组件。

维度简称：b 为批量，c 为通道，h/w 为高宽，d 为隐藏宽度，n 为 token 数。
z 表示潜变量轴，d_p 为 patch 内特征数；c_z/h_z/w_z 表示空间潜变量尺寸。
* 表示一组可变维度；一维形状的前导空格仅用于 Ruff 兼容。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from jaxtyping import Float, Int, Shaped
    from torch import Tensor

# JSON 元数据保持递归结构；Checkpoint 还包含张量、优化器状态等异构 Python 对象。
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]
)
type JsonDict = dict[str, JsonValue]
type Checkpoint = dict[str, Any]
type Batch = Shaped[Tensor, "..."] | list[Shaped[Tensor, "..."]] | tuple[Shaped[Tensor, "..."], ...]
type ReparameterizeFn = Callable[
    [
        Float[Tensor, " b *z"],
        Float[Tensor, " b *z"],
        Float[Tensor, " b *z"],
    ],
    Float[Tensor, " b *z"],
]
type KLFn = Callable[[Float[Tensor, " b *z"], Float[Tensor, " b *z"]], Float[Tensor, ""]]
type PatchifyFn = Callable[
    [Float[Tensor, "b c h w"], int],
    Float[Tensor, "b n d_p"],
]
type UnpatchifyFn = Callable[
    [Float[Tensor, "b n d_p"], int, int, int, int],
    Float[Tensor, "b c h w"],
]


class DenoisingObjective(Protocol):
    """描述训练器实际需要的噪声目标接口，不提供算法实现。

    Args:
        实现方提供 num_steps、add_noise 和 loss；不要求继承这个 Protocol。

    Results:
        DenoisingObjective: 供类型标注使用的结构接口，训练时仍调用用户显式传入的对象。
    """

    num_steps: int

    def add_noise(
        self,
        clean: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
        timesteps: Int[Tensor, " b"],
    ) -> Float[Tensor, "b c h w"]:
        """声明按每张图的时间构造带噪输入的契约。

        Args:
            clean: [b, c, h, w] 干净图像或潜变量。
            noise: 与 clean 同形的标准噪声。
            timesteps: [b] 整数时间，每张图一个索引。

        Results:
            Float[Tensor, "b c h w"]: 与 clean 同形的带噪张量，作为预测网络输入。
        """
        ...

    def loss(
        self,
        predicted_noise: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, ""]:
        """声明噪声预测与监督噪声之间的标量训练目标。

        Args:
            predicted_noise: [b, c, h, w] 网络预测。
            noise: 同形的真实加噪值。

        Results:
            Float[Tensor, ""]: 零维浮点张量，用于反向传播；归一化由实现方明确。
        """
        ...


class SamplingSchedule(Protocol):
    """描述离散反向循环需要的时间长度和单步采样接口。

    Args:
        实现方提供 num_steps 和 step；可以是个人实现或显式教学测试替身。

    Results:
        SamplingSchedule: 静态采样契约，不实例化参考调度器，也不补写算法。
    """

    num_steps: int

    def step(
        self,
        predicted_noise: Float[Tensor, "b c h w"],
        timestep: int,
        sample: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b c h w"]:
        """声明给定预测、时间和随机项时更新反向状态的契约。

        Args:
            predicted_noise: [b, c, h, w] 预测噪声。
            timestep: 当前反向更新的整数时间索引。
            sample: 与预测同形的当前状态。
            noise: 同形的本步独立随机项。

        Results:
            Float[Tensor, "b c h w"]: 更新后的同形状态，由反向循环继续使用。
        """
        ...
