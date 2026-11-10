"""对同一个 B0/B1/B2 Checkpoint 比较 CPU FP32 与指定设备的前向和梯度。"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from minigen.commands import open_cache, read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.checkpoint import load_checkpoint
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device


def main() -> int:
    """使用前两条固定训练表示检查跨设备 FP32 数值，不改变检查点。

    Raises:
        SystemExit: 命令参数无效。
        OSError/ValueError/RuntimeError: 配置、缓存或设备不可用。

    Results:
        int: 逐张量数值检查通过为 0，误差超限为 1。

    副作用:
        在 CPU 及目标设备做一次前向反向，写入带容差和实际误差的回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    device = resolve_device(args.device, "fp32")
    config, data = read_experiment(root, args.config)
    torch.set_num_threads(config.cpu_threads)
    source, cache, _, identity = open_cache(root, data)
    state = load_checkpoint(root / args.checkpoint)
    if state["data_identity"] != identity:
        raise ValueError("Checkpoint and data identities differ")
    indices = selected_indices(source, data, "train")[:2]
    batch = default_collate([cache.split("train")[i] for i in indices])
    generator = torch.Generator().manual_seed(20261110)
    z, target = flow_interpolate(
        batch.latent.float(),
        torch.randn(batch.latent.shape, generator=generator),
        torch.tensor([0.2, 0.8]),
    )
    times = torch.tensor([0.2, 0.8])
    condition = Condition(batch.text.float(), batch.mask)
    reference = build_model(config.model).eval()
    reference.load_state_dict(state["model"])
    candidate = copy.deepcopy(reference).to(device)
    expected = reference(z, times, condition)
    reference_loss = flow_loss(expected, target)
    reference_loss.backward()
    actual = candidate(z.to(device), times.to(device), condition.to(device))
    actual_loss = flow_loss(actual, target.to(device))
    actual_loss.backward()
    atol, rtol = 2e-4, 2e-4
    errors = []
    max_gradient_error = 0.0
    try:
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), atol=atol, rtol=rtol)
    except AssertionError as error:
        errors.append(f"forward: {error}")
    for (name, cpu), (_, other) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        if cpu.grad is None or other.grad is None:
            errors.append(f"{name}: missing gradient")
            continue
        actual_gradient = other.grad.cpu()
        max_gradient_error = max(
            max_gradient_error, float((cpu.grad - actual_gradient).abs().max())
        )
        try:
            torch.testing.assert_close(actual_gradient, cpu.grad, atol=atol, rtol=rtol)
        except AssertionError as error:
            errors.append(f"{name}: {error}")
    report = {
        "status": "passed" if not errors else "failed",
        "device": str(device),
        "precision": "fp32",
        "torch": str(torch.__version__),
        "samples": batch.sample_id,
        "atol": atol,
        "rtol": rtol,
        "cpu_loss": float(reference_loss.detach()),
        "device_loss": float(actual_loss.detach()),
        "output_max_abs_error": float((actual.detach().cpu() - expected.detach()).abs().max()),
        "gradient_max_abs_error": max_gradient_error,
        "errors": errors,
        "scope": "2条固定表示前反向；不代表跨设备训练逐位一致或 CUDA AMP 验收",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
