"""Full discrete reverse loop; interrupted trajectories are explicitly incomplete."""

import time

import torch


@torch.no_grad()
def sample_ddpm(model, scheduler, initial_noise, *, seed=123, max_seconds=300, capture=None):
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
