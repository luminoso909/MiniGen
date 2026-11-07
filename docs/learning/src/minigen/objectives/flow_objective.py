"""L05 / S4-01：个人噪声到数据的线性路径与 Euler 核心。"""

from jaxtyping import Float
from torch import Tensor

from minigen._learning import pending


def linear_path(
    clean: Float[Tensor, "b c h w"],
    noise: Float[Tensor, "b c h w"],
    time: Float[Tensor, " b"],
) -> tuple[Float[Tensor, "b c h w"], Float[Tensor, "b c h w"]]:
    """预留从噪声到数据的线性 Flow 路径与速度目标。

    设计输出：带噪路径点和目标速度，二者均与 clean 同形；当前不返回。

    Args:
        clean: [b,c,h,w] 干净潜变量。
        noise: 同形标准高斯起点。
        time: [b] 浮点时间，设计范围 [0,1]，方向为噪声到数据。

    Raises:
        CoreImplementationRequired: L05 线性路径尚待个人实现。
    """
    pending("L05 linear_path")


def velocity_loss(
    predicted: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
) -> Float[Tensor, ""]:
    """预留速度场回归的均方误差。

    设计输出：对 batch 与特征取均值的零维损失；当前不返回。

    Args:
        predicted: [b,c,h,w] 预测速度。
        target: 同形路径速度目标。

    Raises:
        CoreImplementationRequired: L05 速度损失尚待个人实现。
    """
    pending("L05 velocity_loss")


def euler_step(
    state: Float[Tensor, "b c h w"],
    velocity: Float[Tensor, "b c h w"],
    delta_time: float,
) -> Float[Tensor, "b c h w"]:
    """预留沿时间递增方向的 Euler 积分更新。

    设计输出：同形更新状态；当前不返回。

    Args:
        state: [b,c,h,w] 当前状态。
        velocity: 同形预测速度。
        delta_time: 时间增量，噪声到数据方向使用正步长。

    Raises:
        CoreImplementationRequired: L05 Euler 更新尚待个人实现。
    """
    pending("L05 euler_step")
