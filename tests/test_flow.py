"""P3 直线路径、CFG 与积分器的独立解析对照。"""

import math

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from minigen.contracts import Condition
from minigen.objectives.flow import drop_condition, flow_interpolate, flow_loss
from minigen.samplers.flow import guided_velocity, integrate


class AnalyticField:
    """提供已知解析速度，并累计真实函数调用次数。

    Args:
        kind: constant 使用文本首值作为速度，exponential 使用状态自身。
    """

    def __init__(
        self,
        kind: str,
    ) -> None:
        """记录解析场的类型。

        Args:
            kind: constant 或 exponential。

        副作用:
            初始化实例调用计数。
        """
        self.kind = kind
        self.calls = 0

    def __call__(
        self,
        z: Float[Tensor, "b c h w"],
        t: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b c h w"]:
        """返回与 t 无关的常速度，或 dz/dt=z。

        Args:
            z: 当前状态 [b,c,h,w]。
            t: 接口要求的 [b] 时间；解析场不依赖它。
            condition: 文本首值控制常速度。

        Results:
            Float[Tensor, "b c h w"]: 解析速度。

        副作用:
            累加实际求值次数。
        """
        self.calls += 1
        if self.kind == "exponential":
            return z
        return condition.features[:, :1, :1, None].expand_as(z)


def condition(
    value: float,
    b: int = 2,
) -> Condition:
    """生成解析测试使用的单 token 条件。

    Args:
        value: 控制常速度的特征值。
        b: 批大小。

    Results:
        Condition: [b,1,1] 常数特征与全有效掩码。
    """
    return Condition(torch.full((b, 1, 1), value), torch.ones(b, 1, dtype=torch.bool))


def test_flow_endpoints_derivative_and_scalar_mse() -> None:
    """手算端点、广播、时间导数和均方误差归一化。

    Raises:
        AssertionError: 与独立解析公式不符或梯度错误。
    """
    data = torch.tensor([2.0, 4.0, -2.0, 6.0]).reshape(2, 1, 1, 2)
    noise = torch.tensor([0.0, 2.0, 2.0, 2.0]).reshape_as(data)
    times = torch.tensor([0.25, 0.75], requires_grad=True)
    state, target = flow_interpolate(data, noise, times)
    torch.testing.assert_close(state.flatten(), torch.tensor([0.5, 2.5, -1.0, 5.0]))
    torch.testing.assert_close(target.flatten(), torch.tensor([2.0, 2.0, -4.0, 4.0]))
    state.sum().backward()
    torch.testing.assert_close(times.grad, torch.tensor([4.0, 0.0]))
    assert torch.equal(flow_interpolate(data, noise, torch.zeros(2))[0], noise)
    assert torch.equal(flow_interpolate(data, noise, torch.ones(2))[0], data)
    pred = torch.zeros_like(target, requires_grad=True)
    loss = flow_loss(pred, target)
    assert loss.item() == 10.0
    loss.backward()
    torch.testing.assert_close(pred.grad, -target / 2)
    with pytest.raises(ValueError, match="times"):
        flow_interpolate(data, noise, torch.tensor([-0.1, 0.2]))
    with pytest.raises(ValueError, match="matching"):
        flow_loss(pred[:1], target)


def test_condition_drop_and_cfg_extremes() -> None:
    """验证样本级空条件替换及 s=0/1/3 的 CFG 独立计算。

    Raises:
        AssertionError: 特征/掩码丢弃不一致或 CFG 代数与调用数错误。
    """
    cond, blank = condition(5.0), condition(2.0, 1)
    dropped = drop_condition(cond, blank, torch.tensor([False, True]))
    assert dropped.features[:, 0, 0].tolist() == [5.0, 2.0]
    full = Condition(torch.ones(2, 4, 3), torch.ones(2, 4, dtype=torch.bool))
    empty = Condition(torch.zeros(1, 4, 3), torch.tensor([[True, True, False, False]]))
    replaced = drop_condition(full, empty, torch.tensor([False, True]))
    assert replaced.mask.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert torch.equal(replaced.features[1], empty.features[0])
    model = AnalyticField("constant")
    z, t = torch.zeros(2, 1, 1, 2), torch.zeros(2)
    for scale, expected, calls in ((0.0, 2.0, 1), (1.0, 5.0, 1), (3.0, 11.0, 2)):
        model.calls = 0
        result = guided_velocity(model, z, t, cond, blank, scale)
        assert torch.all(result == expected)
        assert model.calls == calls
    with pytest.raises(ValueError):
        guided_velocity(model, z, t, cond, blank, -1)
    with pytest.raises(ValueError):
        Condition(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))


def test_integrators_direction_accuracy_and_real_nfe() -> None:
    """对二维常速度检查终点，等 NFE 比较指数场的 Euler/Heun 误差。

    Raises:
        AssertionError: 积分方向、精度、输入保护或真实求值计数错误。
    """
    initial = torch.tensor([1.0, -2.0]).reshape(1, 1, 1, 2)
    cond, blank = condition(3.0, 1), condition(1.0, 1)
    for method in ("euler", "heun"):
        field = AnalyticField("constant")
        result = integrate(field, initial, cond, blank, 5, 2.0, method)
        torch.testing.assert_close(result.samples, torch.tensor([6.0, 3.0]).reshape_as(initial))
        assert result.solver_nfe == (5 if method == "euler" else 10)
        assert result.model_evaluations == field.calls == 2 * result.solver_nfe
        assert initial.flatten().tolist() == [1.0, -2.0]
    field = AnalyticField("exponential")
    euler = integrate(field, initial, cond, blank, 8)
    heun = integrate(field, initial, cond, blank, 4, method="heun")
    exact = initial * math.e
    assert (heun.samples - exact).abs().max() < (euler.samples - exact).abs().max()
    assert euler.model_evaluations == heun.model_evaluations == 8
    with pytest.raises(ValueError):
        integrate(field, initial, cond, blank, -2)
    with pytest.raises(ValueError):
        integrate(field, initial, cond, blank, 2, method="ddpm")
