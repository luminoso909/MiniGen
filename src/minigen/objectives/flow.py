"""P3 线性 Flow Matching 工程实现，方向固定为噪声 0 到数据 1。"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from minigen.contracts import Condition


def flow_interpolate(
    data: Float[Tensor, "b c h w"],
    noise: Float[Tensor, "b c h w"],
    times: Float[Tensor, " b"],
) -> tuple[Float[Tensor, "b c h w"], Float[Tensor, "b c h w"]]:
    """构造直线条件路径及其时间导数，保留梯度以便数值检查。

    Args:
        data: [b,c,h,w] 已缩放的干净潜变量。
        noise: 同形的标准高斯样本。
        times: [b] 位于 [0,1] 的连续时间。

    Raises:
        ValueError: 输入形状、浮点类型、设备或时间范围不符。

    Results:
        tuple: z_t=(1-t)noise+t*data，以及恒定监督速度 data-noise。
    """
    if (
        data.ndim != 4
        or data.shape != noise.shape
        or times.shape != (data.shape[0],)
        or not all(x.is_floating_point() for x in (data, noise, times))
        or data.device != noise.device
        or data.device != times.device
        or not torch.isfinite(times).all()
        or (times < 0).any()
        or (times > 1).any()
    ):
        raise ValueError("Expected aligned floating latent/noise and times in [0,1]")
    t = times.to(dtype=data.dtype)[:, None, None, None]
    return (1 - t) * noise + t * data, data - noise


def flow_loss(
    prediction: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
) -> Float[Tensor, ""]:
    """在 FP32 下按所有样本与潜元素平均速度平方误差。

    Args:
        prediction: [b,c,h,w] 网络速度。
        target: 同形的真实路径速度。

    Raises:
        ValueError: 形状不一致或不是非空四维张量。

    Results:
        Float[Tensor, ""]: 可反向传播的零维 FP32 MSE。
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.numel() == 0:
        raise ValueError("Velocity MSE requires matching nonempty [b,c,h,w] tensors")
    return (prediction.float() - target.float()).square().mean()


def drop_condition(
    condition: Condition,
    blank: Condition,
    drop_mask: Bool[Tensor, " b"],
) -> Condition:
    """按样本替换为真实空文本编码，同时替换有效 token 掩码。

    Args:
        condition: 原始 [b,n,d] 文本条件。
        blank: 同一编码器产生的空文本，可以只有一行。
        drop_mask: [b] 布尔值；True 对应整个样本的条件丢弃。

    Raises:
        ValueError: 空条件、丢弃掩码或设备不匹配。

    Results:
        Condition: 逐样本选择原文本或空文本后的特征与掩码。
    """
    b = condition.features.shape[0]
    blank = blank.repeat_batch(b)
    if (
        blank.features.shape != condition.features.shape
        or blank.features.device != condition.features.device
        or blank.features.dtype != condition.features.dtype
        or drop_mask.shape != (b,)
        or drop_mask.dtype != torch.bool
        or drop_mask.device != condition.features.device
    ):
        raise ValueError("Condition dropping requires aligned blank and bool batch mask")
    return Condition(
        torch.where(drop_mask[:, None, None], blank.features, condition.features),
        torch.where(drop_mask[:, None], blank.mask, condition.mask),
    )
