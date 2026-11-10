"""仅对 MiniGen 主干切换注意力执行方式，不改变冻结组件或权重结构。"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn


def attention(
    query: Float[Tensor, "b a n d_h"],
    key: Float[Tensor, "b a m d_h"],
    value: Float[Tensor, "b a m d_h"],
    mask: Bool[Tensor, "b 1 1 m"] | None = None,
    backend: str = "sdpa",
) -> Float[Tensor, "b a n d_h"]:
    """计算非因果注意力，naive 仅作为数学和性能对照。

    Args:
        query: [b,a,n,d_h] 查询。
        key: [b,a,m,d_h] 键。
        value: [b,a,m,d_h] 值。
        mask: [b,1,1,m] 布尔可见性，True 允许访问。
        backend: sdpa 或显式矩阵 naive；均不施加 dropout。

    Raises:
        ValueError: 后端名称不受支持。

    Results:
        Float[Tensor, "b a n d_h"]: 聚合后的每头特征。
    """
    if backend == "sdpa":
        return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
    if backend != "naive":
        raise ValueError("Attention backend must be naive or sdpa")
    logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.inf)
        # 全屏蔽行与 SDPA 一致地输出零，不把 softmax(-inf) 传播为 NaN。
        valid = mask.any(dim=-1, keepdim=True)
        logits = torch.where(valid, logits, torch.zeros_like(logits))
        probabilities = logits.softmax(-1) * valid
    else:
        probabilities = logits.softmax(-1)
    return probabilities @ value


def set_attention_backend(
    model: nn.Module,
    backend: str,
) -> int:
    """只修改明确声明 attention_backend 的主干模块，不触及参数和 buffer。

    Args:
        model: 已构造的 MiniGen 主干，不能传冻结组件组成的整套 pipeline。
        backend: naive 或 sdpa。

    Raises:
        ValueError: 后端名称不支持或模型没有可切换模块。

    Results:
        int: 被设置的模块数。

    副作用:
        更新实例内非参数属性；不改变 state_dict 或全局 PyTorch 函数。
    """
    if backend not in {"naive", "sdpa"}:
        raise ValueError("Attention backend must be naive or sdpa")
    modules = [module for module in model.modules() if hasattr(module, "attention_backend")]
    if not modules:
        raise ValueError("Model does not expose MiniGen attention modules")
    for module in modules:
        module.attention_backend = backend
    return len(modules)
