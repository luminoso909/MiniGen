"""P3 工程条件与速度模型协议；不依赖历史学习模块。"""

from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Bool, Float
from torch import Tensor


@dataclass(frozen=True)
class Condition:
    """封装冻结文本 token 表示和有效位置，供不同生成主干共用。

    Args:
        features: [b,n,d] 浮点文本特征；正式 CLIP 的 d 为 512。
        mask: [b,n] 布尔有效掩码，True 表示 BOS、内容或 EOS。
    """

    features: Float[Tensor, "b n d"]
    mask: Bool[Tensor, "b n"]

    def __post_init__(self) -> None:
        """检查条件张量的形状、设备及至少一个可见 token。

        Raises:
            ValueError: 特征或掩码的类型、形状、设备或有效位置不符。
        """
        if (
            self.features.ndim != 3
            or not self.features.is_floating_point()
            or self.mask.shape != self.features.shape[:2]
            or self.mask.dtype != torch.bool
            or self.mask.device != self.features.device
            or not self.mask.any(dim=1).all()
            or not torch.isfinite(self.features).all()
        ):
            raise ValueError("Expected finite text [b,n,d] and nonempty bool mask [b,n]")

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "Condition":
        """迁移文本精度及设备，同时保留掩码的布尔语义。

        Args:
            device: 目标 CPU、MPS 或 CUDA 设备。
            dtype: 浮点特征精度。

        Raises:
            ValueError/RuntimeError: 非浮点目标或设备迁移失败。

        Results:
            Condition: 已迁移特征和掩码的条件实例。
        """
        return Condition(self.features.to(device=device, dtype=dtype), self.mask.to(device))

    def repeat_batch(
        self,
        b: int,
    ) -> "Condition":
        """扩展单条空条件；已有匹配批大小时直接返回自身。

        Args:
            b: 所需批大小，必须为正数。

        Raises:
            ValueError: 批大小无效，或原批既非 1 也不等于 b。

        Results:
            Condition: [b,n,d] 特征和 [b,n] 掩码，扩展时只创建视图。
        """
        if b < 1 or self.features.shape[0] not in (1, b):
            raise ValueError("Condition batch must be singleton or match requested batch")
        if self.features.shape[0] == b:
            return self
        return Condition(self.features.expand(b, -1, -1), self.mask.expand(b, -1))


class FlowModel(Protocol):
    """描述所有生成主干共用的连续时间速度场接口。"""

    def __call__(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """预测噪声到数据方向的瞬时速度，不在主干中编解码图文。

        Args:
            z: [b,c,h,w] 当前潜状态。
            t: [b] 连续时间，0 为噪声，1 为数据。
            condition: 与 b 一致的文本特征和掩码。

        Results:
            Float[Tensor, "b c h w"]: 与 z 同形的速度。
        """
        ...
