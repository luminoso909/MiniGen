"""用冻结缓存、解析路径及固定采样分离 Flow 拟合与解码链路问题。"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import open_cache, read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.samplers.flow import integrate
from minigen.training.checkpoint import load_checkpoint
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device


@torch.no_grad()
def main() -> int:
    """比较在线/EMA、等 NFE 求解器与更多积分步，保留真实参考图及误差。

    Raises:
        SystemExit: 命令参数错误。
        FileExistsError: 输出目录已经存在。
        ValueError: 检查点和数据或模型身份不符。
        AssertionError: 已知目标的解析路径积分终点误差超限。
        OSError/RuntimeError: 本地资产读取、模型计算或图像写入失败。

    Results:
        int: 完成诊断为 0；数值结果不自动转成视觉质量通过。

    副作用:
        加载本地冻结 VAE 与检查点，保存诊断图、潜变量和 JSON；不更新参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    root = Path.cwd()
    if args.output.exists():
        raise FileExistsError(args.output)
    config, data = read_experiment(root, args.config)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(args.device, "fp32")
    source, cache, blank, identity = open_cache(root, data)
    state = load_checkpoint(args.checkpoint)
    if state["data_identity"] != identity or state["config"]["model"] != config.model:
        raise ValueError("Checkpoint and experiment identities differ")
    indices = selected_indices(source, data, "train")[:4]
    # 单样本使用四个独立初噪，诊断能否恢复同一目标；不改变正式数据清单。
    if len(indices) == 1:
        indices *= 4
    rows = [cache.split("train")[i] for i in indices]
    target = torch.stack([row.latent.float() for row in rows]).to(device)
    condition = Condition(
        torch.stack([row.text.float() for row in rows]), torch.stack([row.mask for row in rows])
    ).to(device)
    blank = blank.to(device)
    rng = torch.Generator().manual_seed(90210)
    noise = torch.randn(target.shape, generator=rng).to(device)
    model = build_model(config.model).to(device).eval().requires_grad_(False)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=device)
    args.output.mkdir(parents=True)
    reference = (vae.decode(target).clamp(-1, 1) + 1) / 2
    save_image(reference.cpu(), args.output / "reference.png", nrow=4)
    # 已知端点时的恒定 oracle 速度；验证积分加号、终点与已缩放 latent 的解码。
    oracle = noise.clone()
    for _ in range(40):
        oracle = oracle + (target - noise) / 40
    oracle_error = float((oracle - target).abs().max())
    assert oracle_error < 1e-4
    values = {}
    for weights in ("model", "ema"):
        model.load_state_dict(state[weights], strict=True)
        slices = []
        for t in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
            times = torch.full((len(rows),), t, device=device)
            z, velocity = flow_interpolate(target, noise, times)
            prediction = model(z, times, condition)
            slices.append(
                {
                    "t": t,
                    "velocity_mse": float(flow_loss(prediction, velocity)),
                    "endpoint_estimate_mse": float(
                        (z + (1 - t) * prediction - target).square().mean()
                    ),
                }
            )
        values[weights] = slices
    results = []
    for name, weights, method, steps, guidance in (
        ("online_euler40", "model", "euler", 40, 1.0),
        ("ema_euler40", "ema", "euler", 40, 1.0),
        ("ema_heun20", "ema", "heun", 20, 1.0),
        ("ema_euler200", "ema", "euler", 200, 1.0),
        ("ema_cfg3", "ema", "euler", 40, 3.0),
    ):
        model.load_state_dict(state[weights], strict=True)
        generated = integrate(model, noise, condition, blank, steps, guidance, method)
        image = (vae.decode(generated.samples).clamp(-1, 1) + 1) / 2
        save_image(image.cpu(), args.output / f"{name}.png", nrow=4)
        torch.save(generated.samples.cpu(), args.output / f"{name}.pt")
        results.append(
            {
                "name": name,
                "latent_target_mse": float((generated.samples - target).square().mean()),
                "decoded_reference_mse": float((image - reference).square().mean()),
                "solver_nfe": generated.solver_nfe,
                "model_evaluations": generated.model_evaluations,
            }
        )
    report = {
        "checkpoint_sha256": hash_file(args.checkpoint),
        "step": state["step"],
        "device": str(device),
        "seed": 90210,
        "sample_ids": [row.sample_id for row in rows],
        "oracle_endpoint_max_abs": oracle_error,
        "latent_scaling_factor": vae.scale,
        "latent_std": float(target.std()),
        "time_slices": values,
        "samples": results,
        "quality_passed": None,
        "scope": "目标误差是固定训练图诊断，不是正式生成质量指标；需独立看图",
    }
    (args.output / "diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
