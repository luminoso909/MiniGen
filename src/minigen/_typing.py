"""静态接口与异构数据别名；导入本模块不会加载可选模型组件。"""

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
        Float[Tensor, " batch *latent"],
        Float[Tensor, " batch *latent"],
        Float[Tensor, " batch *latent"],
    ],
    Float[Tensor, " batch *latent"],
]
type KLFn = Callable[
    [Float[Tensor, " batch *latent"], Float[Tensor, " batch *latent"]], Float[Tensor, ""]
]
type PatchifyFn = Callable[
    [Float[Tensor, "batch channels height width"], int],
    Float[Tensor, "batch tokens patch_features"],
]
type UnpatchifyFn = Callable[
    [Float[Tensor, "batch tokens patch_features"], int, int, int, int],
    Float[Tensor, "batch channels height width"],
]


class DenoisingObjective(Protocol):
    """描述训练器实际需要的噪声目标接口，不提供算法实现。

    1 输入：
        实现方提供 num_steps、add_noise 和 loss；不要求继承这个 Protocol。
    2 输出：
        供类型标注使用的结构接口，训练时仍调用用户显式传入的对象。
    3 报错（Raises）：
        无显式 raise；具体实现负责数学验证与异常处理。
    """

    num_steps: int

    def add_noise(
        self,
        clean: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
        timesteps: Int[Tensor, " batch"],
    ) -> Float[Tensor, "batch channels height width"]:
        """声明按每张图的时间构造带噪输入的契约。

        1 输入：
            clean：干净图像或潜变量；noise：同形标准噪声；timesteps：每张图的整数时间。
        2 输出：
            与 clean 同形的带噪张量，作为预测网络输入。
        3 报错（Raises）：
            无显式 raise；这是静态协议，异常由实现方决定。
        """
        ...

    def loss(
        self,
        predicted_noise: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """声明噪声预测与监督噪声之间的标量训练目标。

        1 输入：
            predicted_noise：网络预测；noise：同形的真实加噪值。
        2 输出：
            零维浮点张量，用于反向传播；归一化由实现方明确。
        3 报错（Raises）：
            无显式 raise；协议不执行计算。
        """
        ...


class SamplingSchedule(Protocol):
    """描述离散反向循环需要的时间长度和单步采样接口。

    1 输入：
        实现方提供 num_steps 和 step；可以是个人实现或显式教学测试替身。
    2 输出：
        静态采样契约，不实例化参考调度器，也不补写算法。
    3 报错（Raises）：
        无显式 raise；实现方的异常由调用者接收。
    """

    num_steps: int

    def step(
        self,
        predicted_noise: Float[Tensor, "batch channels height width"],
        timestep: int,
        sample: Float[Tensor, "batch channels height width"],
        noise: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        """声明给定预测、时间和随机项时更新反向状态的契约。

        1 输入：
            predicted_noise：预测噪声；timestep：单个离散时间索引。
            sample：当前状态；noise：本步独立随机项，均与预测同形。
        2 输出：
            更新后的同形状态，由反向循环继续使用。
        3 报错（Raises）：
            无显式 raise；这是协议声明，未定义具体反向计算。
        """
        ...
