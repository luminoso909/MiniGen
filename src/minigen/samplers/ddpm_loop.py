"""完整离散反向循环；超时状态不冒充最终生成样本。"""

import time
from collections.abc import Iterable

import torch
from jaxtyping import Float
from torch import Tensor, nn

from minigen._typing import Checkpoint, SamplingSchedule


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    scheduler: SamplingSchedule,
    initial_noise: Float[Tensor, "b c h w"],
    *,
    seed: int = 123,
    max_seconds: float = 300,
    capture: Iterable[int] | None = None,
) -> Checkpoint:
    """执行完整离散反向链，并明确区分超时中间状态与最终样本。

    Args:
        model: 无梯度噪声预测网络，正常结束后恢复原训练模式。
        scheduler: 显式反向单步接口，提供 num_steps 与 step。
        initial_noise: [b,c,h,w] 初始噪声，决定采样设备与形状。
        seed: 本次独立 CPU 随机发生器的种子。
        max_seconds: 采样时间上限，必须为正。
        capture: 需要保存的代码时间索引；None 或空集合沿用默认索引。

    Raises:
        ValueError: max_seconds 非正；FloatingPointError: 反向状态出现非有限值。
        模型或调度器异常透传。

    Results:
        Checkpoint: 采样状态、快照与真实完成信息。
            images: CPU 上最后得到的 [b, c, h, w] 状态，需结合 completed 判断是否为最终样本。
            snapshots: 以 initial 或时间索引字符串为键的 CPU 张量快照。
            completed: 是否执行完全部反向步骤。
            completed_steps/model_calls: 实际更新次数与模型求值次数。
            elapsed_seconds/seed: 实测耗时和独立采样种子。

    副作用:
        临时切换模型到 eval，正常返回时恢复原模式；异常路径未做恢复保护。
        使用独立 CPU 随机发生器，不改变传入初始噪声或调度器状态。
    """
    if max_seconds <= 0:
        raise ValueError("采样时间上限必须为正")
    device = initial_noise.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    state = initial_noise.clone()
    snapshots = {"initial": state.cpu().clone()}
    capture = set(capture or [749, 499, 249, 99, 0])
    start = time.monotonic()
    completed_steps = 0
    # 倒序访问所有离散索引；是否到达终点与图像数值是两项不同检查。
    for timestep in reversed(range(scheduler.num_steps)):
        if time.monotonic() - start >= max_seconds:
            break
        times = torch.full((len(state),), timestep, dtype=torch.long, device=device)
        prediction = model(state, times)
        noise = torch.randn(state.shape, generator=generator).to(device)
        state = scheduler.step(prediction, timestep, state, noise)
        if not torch.isfinite(state).all().item():
            raise FloatingPointError(f"采样在索引 {timestep} 出现非有限值")
        completed_steps += 1
        if timestep in capture:
            snapshots[str(timestep)] = state.cpu().clone()
    model.train(was_training)
    return {
        "images": state.cpu(),
        "snapshots": snapshots,
        "completed": completed_steps == scheduler.num_steps,
        "completed_steps": completed_steps,
        "elapsed_seconds": time.monotonic() - start,
        "seed": seed,
        "model_calls": completed_steps,
    }
