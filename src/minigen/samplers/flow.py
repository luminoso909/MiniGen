"""独立的正向时间 Euler/Heun 积分及速度 CFG，不混用离散 DDPM 日程。"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from minigen.contracts import Condition, FlowModel


@dataclass(frozen=True)
class SamplingResult:
    """保存最终潜变量和真实网络求值成本。

    Args:
        samples: [b,c,h,w] 时间 1 的积分结果。
        solver_nfe: 速度场评估次数；Euler 为 steps，Heun 为 2*steps。
        model_evaluations: 含 CFG 分支的实际主干调用次数。
    """

    samples: Float[Tensor, "b c h w"]
    solver_nfe: int
    model_evaluations: int


def guided_velocity(
    model: FlowModel,
    z: Float[Tensor, "b c h w"],
    t: Float[Tensor, " b"],
    condition: Condition,
    blank: Condition,
    scale: float,
) -> Float[Tensor, "b c h w"]:
    """用 v_u+s(v_c-v_u) 组合速度；s 为 0/1 时只执行必要分支。

    Args:
        model: 遵循同形速度输出契约的生成主干。
        z: [b,c,h,w] 当前状态。
        t: [b] 当前连续时间。
        condition: 条件分支的文本。
        blank: 空文本分支，可为单行。
        scale: 有限、非负 CFG 系数，1 对应普通条件预测。

    Raises:
        ValueError: CFG 系数非法，或条件批大小不匹配。

    Results:
        Float[Tensor, "b c h w"]: 组合后的速度，未裁剪或重缩放。
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    if condition.features.shape[0] != z.shape[0]:
        raise ValueError("Condition batch does not match state")
    if scale == 1:
        return model(z, t, condition)
    unconditional = model(z, t, blank.repeat_batch(z.shape[0]))
    if scale == 0:
        return unconditional
    return unconditional + scale * (model(z, t, condition) - unconditional)


@torch.no_grad()
def integrate(
    model: FlowModel,
    initial: Float[Tensor, "b c h w"],
    condition: Condition,
    blank: Condition,
    steps: int,
    guidance: float = 1.0,
    method: str = "euler",
) -> SamplingResult:
    """从 t=0 积分到 t=1，显式统计积分器和 CFG 的求值成本。

    Args:
        model: 调用方已设置为 eval 的速度场，不在这里更改训练状态。
        initial: [b,c,h,w] 固定初始高斯噪声，不修改原张量。
        condition: 与批大小一致的文本条件。
        blank: 同编码器空文本，用于 CFG。
        steps: 正整数积分步数，不能解释为 DDPM 时间索引。
        guidance: 非负 CFG 系数。
        method: euler 或 heun；Heun 包括最终 t=1 的正确器求值。

    Raises:
        ValueError: 方法、步数、CFG 系数或状态非法。
        FloatingPointError: 速度或更新后的状态出现 NaN/Inf。

    Results:
        SamplingResult: 时间 1 潜变量及两种明确的求值计数。
    """
    if type(steps) is not int or steps < 1 or method not in {"euler", "heun"}:
        raise ValueError("Expected positive integer steps and Euler/Heun method")
    if initial.ndim != 4 or not initial.is_floating_point() or not torch.isfinite(initial).all():
        raise ValueError("Initial state must be finite floating [b,c,h,w]")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    state = initial.clone()
    dt = 1.0 / steps
    for i in range(steps):
        times = torch.full((state.shape[0],), i / steps, device=state.device, dtype=torch.float32)
        velocity = guided_velocity(model, state, times, condition, blank, guidance)
        if velocity.shape != state.shape:
            raise ValueError("Velocity shape does not match integration state")
        candidate = state + dt * velocity
        if method == "heun":
            next_times = torch.full_like(times, (i + 1) / steps)
            next_velocity = guided_velocity(
                model, candidate, next_times, condition, blank, guidance
            )
            if next_velocity.shape != state.shape:
                raise ValueError("Velocity shape does not match integration state")
            candidate = state + 0.5 * dt * (velocity + next_velocity)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("Non-finite Flow integration state")
        state = candidate
    nfe = steps * (2 if method == "heun" else 1)
    return SamplingResult(state, nfe, nfe * (1 if guidance in (0, 1) else 2))
