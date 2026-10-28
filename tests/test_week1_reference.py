"""仅验证已授权 Notebook 中的 DDPM 教学参考。"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch


@pytest.fixture(scope="module")
def reference_class() -> Callable[..., Any]:
    """只提取唯一带 ddpm_reference 标记的教学调度器类。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        可调用的教学类对象；动态 exec 命名空间不能当作普通 JSON。

    3 报错（Raises）：
        AssertionError：标记单元不唯一；文件、JSON、编译或代码装载异常透传。
    """
    path = Path(__file__).resolve().parents[1] / "docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb"
    notebook = json.loads(path.read_text())
    sources = [
        "".join(c["source"])
        for c in notebook["cells"]
        if "ddpm_reference" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(sources) == 1
    namespace = {}
    exec(compile(sources[0], str(path), "exec"), namespace)
    return namespace["TeachingDDPMScheduler"]


def test_reference_matches_independent_gaussian_precision(
    reference_class: Callable[..., Any],
) -> None:
    """用独立高斯精度计算对照教学加噪、后验与最终步随机项。

    1 输入：
        reference_class：动态从已授权 Notebook 提取的教学调度器类；只供数学测试。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    schedule = reference_class(num_steps=2, beta_start=0.1, beta_end=0.2)
    clean = torch.tensor([2.0]).reshape(1, 1, 1, 1)
    noise = torch.full_like(clean, -0.5)
    value = schedule.add_noise(clean, noise, torch.tensor([1]))
    xt = math.sqrt(0.72) * 2 - math.sqrt(0.28) * 0.5
    precision = 0.8 / 0.2 + 1 / 0.1
    mean_expected = (math.sqrt(0.8) / 0.2 * xt + math.sqrt(0.9) / 0.1 * 2) / precision
    mean, variance = schedule.posterior(clean, value, torch.tensor([1]))
    assert value.item() == pytest.approx(xt, abs=1e-6)
    assert mean.item() == pytest.approx(mean_expected, abs=1e-6)
    assert variance.item() == pytest.approx(1 / precision, abs=1e-7)
    a = schedule.step(noise, 0, clean, torch.zeros_like(clean))
    b = schedule.step(noise, 0, clean, torch.full_like(clean, 1e5))
    assert torch.equal(a, b)


def test_reference_rejects_shape_and_time_errors(reference_class: Callable[..., Any]) -> None:
    """验证教学 DDPM 拒绝无效时间、广播和损失形状。

    1 输入：
        reference_class：动态从已授权 Notebook 提取的教学调度器类；只供数学测试。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    schedule = reference_class(num_steps=2)
    x = torch.zeros(2, 1, 4, 4)
    for invalid in [torch.tensor([-1, 0]), torch.tensor([0, 2]), torch.zeros(2)]:
        with pytest.raises(ValueError):
            schedule.add_noise(x, x, invalid)
    with pytest.raises(ValueError):
        schedule.loss(x, torch.zeros(1, 1, 4, 4))
