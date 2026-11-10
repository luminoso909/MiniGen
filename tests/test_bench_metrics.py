"""用独立 NumPy 与可信库检查评测数学，不依赖模型权重。"""

import numpy as np
import pytest
import torch

from minigen.evaluation.metrics import clip_score, cmmd, kid, polynomial_mmd


def test_kid_independent_formula_and_reference() -> None:
    """验证无偏估计、合法负值及与 torch-fidelity 核结果一致。

    Raises:
        AssertionError: 三种独立计算不一致，或负值被隐蔽截断。
    """
    from torch_fidelity.metric_kid import polynomial_mmd as reference_mmd

    x = np.random.RandomState(4).normal(size=(8, 5))
    y = np.random.RandomState(8).normal(size=(8, 5))
    xx, yy, xy = (x @ x.T / 5 + 1) ** 3, (y @ y.T / 5 + 1) ** 3, (x @ y.T / 5 + 1) ** 3
    expected = (xx.sum() - np.trace(xx) + yy.sum() - np.trace(yy)) / 56 - 2 * xy.mean()
    actual = polynomial_mmd(torch.from_numpy(x), torch.from_numpy(y))
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual == pytest.approx(reference_mmd(x, y, degree=3, gamma=None, coef0=1), abs=1e-12)
    assert polynomial_mmd(torch.eye(4), torch.eye(4)) < 0
    small = kid(torch.from_numpy(x), torch.from_numpy(y), subsets=7, subset_size=4, seed=3)
    assert small == kid(torch.from_numpy(x), torch.from_numpy(y), 7, 4, 3)
    assert small["mean"] == pytest.approx(np.mean(small["estimates"]))


def test_cmmd_blocked_matches_independent_rbf() -> None:
    """分块 CMMD 对称、同输入为零，且等于独立双循环参考。

    Raises:
        AssertionError: 带宽、归一化、尺度或尾块计算不一致。
    """
    x = np.random.RandomState(1).normal(size=(7, 5))
    y = np.random.RandomState(2).normal(size=(4, 5))
    x, y = (
        x / np.linalg.norm(x, axis=1, keepdims=True),
        y / np.linalg.norm(y, axis=1, keepdims=True),
    )
    expected = 1000 * (
        np.mean([np.exp(-np.sum((a - b) ** 2) / 200) for a in x for b in x])
        + np.mean([np.exp(-np.sum((a - b) ** 2) / 200) for a in y for b in y])
        - 2 * np.mean([np.exp(-np.sum((a - b) ** 2) / 200) for a in x for b in y])
    )
    left, right = torch.from_numpy(x), torch.from_numpy(y)
    assert cmmd(left, right, block_size=3) == pytest.approx(expected, abs=1e-11)
    assert cmmd(left, right, block_size=1) == pytest.approx(cmmd(right, left), abs=1e-11)
    assert cmmd(left, left) == pytest.approx(0, abs=1e-12)


def test_clipscore_order_scaling_and_invalid_features() -> None:
    """验证逐行匹配、余弦归一化、负值裁零和非法输入拒绝。

    Raises:
        AssertionError: 分数与手算不符或错误输入未拒绝。
    """
    x = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    y = torch.tensor([[4.0, 0.0], [0.0, -5.0]])
    result = clip_score(x, y)
    assert result["per_sample"] == [2.5, 0.0]
    assert result["cosine"] == [1.0, -1.0]
    assert clip_score(x, y.flip(0))["mean"] == 0
    with pytest.raises(ValueError, match="Zero-norm"):
        cmmd(torch.zeros(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError, match="finite"):
        kid(torch.full((4, 2), float("nan")), torch.ones(4, 2), 2, 2)
    with pytest.raises(ValueError, match="subset_size"):
        kid(torch.ones(3, 2), torch.ones(3, 2), 2, 4)
    with pytest.raises(ValueError, match="enough samples"):
        polynomial_mmd(torch.ones(1, 2), torch.ones(3, 2))
