"""基于固定特征的分布与图文指标；使用 CPU float64 聚合，保留估计器定义。"""

import math

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


def _features(
    value: Float[Tensor, "n d"],
    minimum: int = 1,
) -> Float[Tensor, "n d"]:
    """校验特征矩阵并转为独立的 CPU 双精度计算视图。

    Args:
        value: [n,d] 浮点特征，样本顺序由调用方负责。
        minimum: 最少样本数。

    Raises:
        ValueError: 形状、数量、dtype 或有限性非法。

    Results:
        Float[Tensor, "n d"]: 不参与梯度计算的 CPU float64 特征。
    """
    if value.ndim != 2 or min(value.shape) < 1 or len(value) < minimum:
        raise ValueError("Expected nonempty feature matrix with enough samples")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("Features must be finite floating point values")
    return value.detach().to(device="cpu", dtype=torch.float64)


def unit_features(
    value: Float[Tensor, "n d"],
) -> Float[Tensor, "n d"]:
    """逐样本归一化投影，不把零向量当作有效特征。

    Args:
        value: [n,d] 模型投影。

    Raises:
        ValueError: 特征非法或存在零范数。

    Results:
        Float[Tensor, "n d"]: CPU float64 单位特征。
    """
    result = _features(value)
    norm = result.norm(dim=1, keepdim=True)
    if (norm == 0).any():
        raise ValueError("Zero-norm feature has no cosine direction")
    return result / norm


def polynomial_mmd(
    left: Float[Tensor, "n d"],
    right: Float[Tensor, "m d"],
) -> float:
    """计算 KID 的无偏三次多项式核 MMD；有限样本允许负值。

    Args:
        left: [n,d] 第一组未归一化 Inception 特征，n≥2。
        right: [m,d] 第二组特征，m≥2。

    Raises:
        ValueError: 特征无效或宽度不一致。

    Results:
        float: 无偏 MMD²，未乘 100 或 1000。
    """
    x, y = _features(left, 2), _features(right, 2)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature widths differ")
    xx, yy, xy = (
        (x @ x.T / x.shape[1] + 1) ** 3,
        (y @ y.T / y.shape[1] + 1) ** 3,
        (x @ y.T / x.shape[1] + 1) ** 3,
    )
    return float(
        (xx.sum() - xx.trace()) / (len(x) * (len(x) - 1))
        + (yy.sum() - yy.trace()) / (len(y) * (len(y) - 1))
        - 2 * xy.mean()
    )


def kid(
    left: Float[Tensor, "n d"],
    right: Float[Tensor, "m d"],
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 20261110,
) -> dict[str, object]:
    """以固定随机状态进行无放回子集 KID，保存每次估计以审查方差。

    Args:
        left: [n,d] 参考特征。
        right: [m,d] 生成特征。
        subsets: 子集估计次数。
        subset_size: 每组每次抽取数量，至少 2。
        seed: NumPy RandomState 种子，与 torch-fidelity 取样约定一致。

    Raises:
        ValueError: 数量、特征或随机种子无效。

    Results:
        dict: mean/std/estimates、数量和子集设置；std 使用 ddof=0。
    """
    x, y = _features(left, 2), _features(right, 2)
    if subsets < 1 or subset_size < 2 or subset_size > min(len(x), len(y)):
        raise ValueError("KID requires positive subsets and 2 <= subset_size <= sample counts")
    rng = np.random.RandomState(seed)
    values = [
        polynomial_mmd(
            x[rng.choice(len(x), subset_size, replace=False)],
            y[rng.choice(len(y), subset_size, replace=False)],
        )
        for _ in range(subsets)
    ]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "estimates": values,
        "subsets": subsets,
        "subset_size": subset_size,
        "seed": seed,
        "reference_count": len(x),
        "generated_count": len(y),
        "kernel": "(x.y/d+1)^3;unbiased",
    }


def _rbf_mean(
    left: Float[Tensor, "n d"],
    right: Float[Tensor, "m d"],
    sigma: float,
    block_size: int,
) -> float:
    """分块累加全部核元素，包含同集合的对角线。

    Args:
        left: [n,d] CPU 双精度特征。
        right: [m,d] 同宽特征。
        sigma: 正有限 RBF 带宽。
        block_size: 每次物化的矩阵边长。

    Results:
        float: 所有 n×m 个核值的平均数。
    """
    total = 0.0
    for i in range(0, len(left), block_size):
        x = left[i : i + block_size]
        for j in range(0, len(right), block_size):
            y = right[j : j + block_size]
            distance = (
                x.square().sum(1)[:, None] + y.square().sum(1)[None] - 2 * x @ y.T
            ).clamp_min(0)
            total += float(torch.exp(-distance / (2 * sigma**2)).sum())
    return total / (len(left) * len(right))


def cmmd(
    left: Float[Tensor, "n d"],
    right: Float[Tensor, "m d"],
    sigma: float = 10.0,
    block_size: int = 256,
) -> float:
    """计算单位 CLIP 投影的 biased RBF MMD²×1000。

    Args:
        left: [n,d] 参考图像投影。
        right: [m,d] 生成图像投影。
        sigma: 参考实现固定为 10。
        block_size: CPU float64 核聚合块大小。

    Raises:
        ValueError: 特征、带宽或块大小非法。

    Results:
        float: CMMD 数值，浮点消减可能产生极小负值，不做隐蔽截断。
    """
    x, y = unit_features(left), unit_features(right)
    if x.shape[1] != y.shape[1] or not math.isfinite(sigma) or sigma <= 0 or block_size < 1:
        raise ValueError("Invalid CMMD dimensions, bandwidth or block size")
    return 1000 * (
        _rbf_mean(x, x, sigma, block_size)
        + _rbf_mean(y, y, sigma, block_size)
        - 2 * _rbf_mean(x, y, sigma, block_size)
    )


def clip_score(
    images: Float[Tensor, "n d"],
    texts: Float[Tensor, "n d"],
) -> dict[str, object]:
    """逐行计算 CLIPScore L/14@336 无前缀变体，保留原始余弦。

    Args:
        images: [n,d] 图像投影。
        texts: [n,d] 严格同序的文本投影。

    Raises:
        ValueError: 特征非法或形状不同。

    Results:
        dict: mean、per_sample 和 cosine；分数为 2.5×max(cosine,0)。
    """
    x, y = unit_features(images), unit_features(texts)
    if x.shape != y.shape:
        raise ValueError("CLIPScore needs one text for each image")
    cosine = (x * y).sum(1).clamp(-1, 1)
    score = 2.5 * cosine.clamp_min(0)
    return {
        "mean": float(score.mean()),
        "per_sample": score.tolist(),
        "cosine": cosine.tolist(),
        "variant": "CLIPScore-L14-336-no-prefix",
        "scale": 2.5,
    }
