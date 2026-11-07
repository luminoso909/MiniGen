"""显式注入 CFG 的条件 DDPM/DDIM 采样与真实计算计数。"""

import math
import time
from collections.abc import Callable

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from minigen._typing import Checkpoint
from minigen.objectives.ddpm_scheduler import DDPMScheduler

# Callable[[无条件预测, 条件预测, scale], 同形引导预测]；核心主体由调用者提供。
Guidance = Callable[
    [Float[Tensor, "b c h w"], Float[Tensor, "b c h w"], float],
    Float[Tensor, "b c h w"],
]


def synchronize(
    device: torch.device,
) -> None:
    """等待实际设备运算完成，以便计时包含异步计算。

    Args:
        device: CPU、MPS 或 CUDA 设备。

    Raises:
        RuntimeError: 请求的加速设备不可用时由 Torch 透传。

    副作用:
        MPS/CUDA 时阻塞到设备队列完成；CPU 直接返回。
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_conditional(
    model: nn.Module,
    scheduler: DDPMScheduler,
    initial_noise: Float[Tensor, "b c h w"],
    condition: Float[Tensor, "b n d"],
    mask: Bool[Tensor, "b n"],
    empty: Float[Tensor, "1 n d"],
    empty_mask: Bool[Tensor, "1 n"],
    guidance: Guidance,
    *,
    scale: float = 1.0,
    method: str = "ddpm",
    steps: int = 1000,
    seed: int = 42,
    max_seconds: float = 120,
) -> Checkpoint:
    """执行相邻 DDPM 或 eta=0 DDIM；少步模式使用正确的跨时间更新。

    Args:
        model: forward(z,t,text,mask) 的外部显式模型。
        scheduler: 个人噪声预测 DDPM；潜空间必须关闭像素裁剪。
        initial_noise: [b,c,h,w] 初始噪声。
        condition: [b,n,d] 条件 token；与初始噪声在同一设备。
        mask: [b,n] 有效位。
        empty: [1,n,d] 空文本 token。
        empty_mask: [1,n] 空文本有效位。
        guidance: 调用方提供的 CFG，不自动调用个人未实现接口。
        scale: 无条件加差值约定；0/1 各只执行一条分支。
        method: ddpm 或 ddim；DDIM 固定 eta=0。
        steps: 正更新步数；DDPM 必须完整覆盖 scheduler.num_steps。
        seed: DDPM 独立 CPU 噪声发生器种子。
        max_seconds: 正时间上限；超时结果不标为最终样本。

    Raises:
        ValueError: 配置/形状不符或启用潜变量像素裁剪。
        FloatingPointError: 反向状态出现非有限值。
        RuntimeError: 网络或设备错误透传。

    Results:
        Checkpoint: CPU 图像状态、完成标记、更新/调用/分支等价 NFE、计时与峰值显存。
            model_calls: Python 模型调用次数；双分支合批时仍是一次调用。
            effective_nfe: 每张输出图对应的分支求值次数；scale 非 0/1 时为更新次数的两倍。
            peak_memory_bytes: CUDA 峰值已分配显存；MPS/CPU 没有同口径峰值，返回 None。

    副作用:
        临时切换模型到 eval，并在异常时也恢复原模式；CUDA 重置峰值统计。
        使用局部噪声发生器，不改调用者初始噪声或全局随机状态。
    """
    if (
        method not in {"ddpm", "ddim"}
        or not 1 <= steps <= scheduler.num_steps
        or max_seconds <= 0
        or not math.isfinite(scale)
    ):
        raise ValueError("Invalid sampler configuration")
    if scheduler.clip_denoised or (method == "ddpm" and steps != scheduler.num_steps):
        raise ValueError("Latents require no pixel clipping; DDPM requires every step")
    b = len(initial_noise)
    if (
        condition.ndim != 3
        or condition.shape[0] != b
        or mask.shape != condition.shape[:2]
        or empty.shape != (1, *condition.shape[1:])
        or empty_mask.shape != (1, condition.shape[1])
        or not mask.any(-1).all()
        or not empty_mask.any(-1).all()
    ):
        raise ValueError("Aligned nonempty token conditions required")
    device = initial_noise.device
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.monotonic()
    state = initial_noise.clone()
    indices = (
        list(reversed(range(scheduler.num_steps)))
        if method == "ddpm"
        else torch.linspace(scheduler.num_steps - 1, 0, steps).round().long().tolist()
    )
    completed = calls = nfe = 0
    try:
        for i, index in enumerate(indices):
            if time.monotonic() - start >= max_seconds:
                break
            times = torch.full((b,), index, device=device, dtype=torch.long)
            if scale == 1:
                prediction = model(state, times, condition, mask)
                branches = 1
            elif scale == 0:
                prediction = model(state, times, empty.expand(b, -1, -1), empty_mask.expand(b, -1))
                branches = 1
            else:
                both = model(
                    torch.cat([state, state]),
                    torch.cat([times, times]),
                    torch.cat([empty.expand(b, -1, -1), condition]),
                    torch.cat([empty_mask.expand(b, -1), mask]),
                )
                unconditioned, conditioned = both.chunk(2)
                prediction = guidance(unconditioned, conditioned, scale)
                branches = 2
            calls += 1
            nfe += branches
            if method == "ddpm":
                noise = torch.randn(state.shape, generator=generator).to(device)
                state = scheduler.step(prediction, index, state, noise)
            else:
                # s=-1 表示干净端点 alpha_bar=1，而不是误用代码索引 0 的系数。
                previous = indices[i + 1] if i + 1 < len(indices) else -1
                a = scheduler.alpha_bars[index].to(device)
                a_previous = (
                    scheduler.alpha_bars[previous].to(device)
                    if previous >= 0
                    else a.new_tensor(1.0)
                )
                clean = (state - (1 - a).sqrt() * prediction) / a.sqrt()
                state = a_previous.sqrt() * clean + (1 - a_previous).sqrt() * prediction
            if not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite conditional sample")
            completed += 1
        synchronize(device)
        elapsed = time.monotonic() - start
    finally:
        model.train(was_training)
    return {
        "images": state.cpu(),
        "completed": completed == len(indices),
        "completed_steps": completed,
        "model_calls": calls,
        "effective_nfe": nfe,
        "branch_examples": nfe * b,
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / b,
        "images_per_second": b / elapsed if completed else None,
        "method": method,
        "eta": 0 if method == "ddim" else None,
        "indices": indices,
        "scale": scale,
        "seed": seed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_memory_limitation": "CUDA allocated peak only; MPS/CPU unavailable",
    }
