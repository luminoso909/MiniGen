"""从随机初始化运行六组动漫头像 30k Flow 训练，并保留可配对的随机流。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from run_anime_b1 import open_anime_cache
from torch import nn
from torch.utils.data import Subset, default_collate

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.config import TrainConfig, build_model
from minigen.training.engine import train_flow
from minigen.training.runtime import autocast_context, resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path("configs/training/anime_six_30k")
RUNS = Path("artifacts/runs/anime_faces_256/six_30k_v1")
ARMS = ("b1_30m_p2", "b2_30m_p2", "b1_50m_p2", "b2_50m_p2", "b1_50m_p1", "b2_50m_p1")
STEPS = 30000
CHECKPOINT_STEPS = (3000, 5000, 7000, 10000, 13000, 15000, 17000, 20000, 23000, 25000, 27000, 30000)
EXPECTED_PARAMETERS = {
    "b1_30m_p2": 30061456,
    "b2_30m_p2": 30061456,
    "b1_50m_p2": 48681232,
    "b2_50m_p2": 48681232,
    "b1_50m_p1": 48672004,
    "b2_50m_p1": 48672004,
}


def read_arm_config(
    root: Path,
    arm: str,
) -> tuple[TrainConfig, JsonDict]:
    """读取指定组的完整训练配置。

    Args:
        root: MiniGen 项目根目录。
        arm: ARMS 中的一组，配置及输出目录按同一组名组织。

    Raises:
        ValueError/OSError: 组名、配置文件或已定稿的步数与随机种子不符。

    Results:
        tuple: 训练配置，以及补齐缓存读取字段的数据配置。
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown six-model arm: {arm}")
    config, data = read_experiment(root, CONFIG_DIR / f"{arm}.json")
    if (
        config.seed != 42
        or config.max_steps != STEPS
        or config.schedule_steps != STEPS
        or tuple(config.checkpoint_steps) != CHECKPOINT_STEPS
        or Path(config.output_dir) != root / RUNS / arm
    ):
        raise ValueError(f"Six-model configuration changed: {arm}")
    return config, data


def initial_model(
    config: TrainConfig,
    arm: str,
) -> tuple[nn.Module, str]:
    """在模型构造前设置统一种子，并对随机初始化状态做内容摘要。

    Args:
        config: 所选模型的训练和架构配置。
        arm: 用于核对可训练参数量的组名。

    Raises:
        ValueError: 实际可训练参数量不符合冻结规格。

    Results:
        tuple: CPU 上的主干和包含全部参数、位置缓存的初始状态 SHA-256。

    副作用:
        设置 Torch 全局随机种子，构造模型消耗随机状态。
    """
    torch.manual_seed(config.seed)
    model = build_model(config.model)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != EXPECTED_PARAMETERS[arm]:
        raise ValueError(f"Unexpected trainable parameter count: {arm}={count}")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return model, digest.hexdigest()


def main() -> None:
    """做零更新真实缓存前向检查，或执行一组可精确恢复的正式训练。

    Raises:
        OSError/RuntimeError/ValueError: 缓存、模型、设备或恢复契约校验失败。
        SystemExit: 命令行错误或训练未完成 30,000 次更新。

    副作用:
        设置线程与随机状态，输出 JSON。train 更新网络和优化器，写日志和检查点。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "train"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    config, data = read_arm_config(root, args.arm)
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, identity = open_anime_cache(root, data)
    train = Subset(cache.split("train"), selected_indices(source, data, "train"))
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    model, initial_sha = initial_model(config, args.arm)
    # 模型在 train_flow 外构造。引擎在取得模型后重设训练 RNG，参数量差异不会移动噪声流。
    initialization = {
        "kind": "random",
        "seed": config.seed,
        "initial_state_sha256": initial_sha,
        "runner_sha256": hash_file(Path(__file__)),
        "config_sha256": hash_file(root / CONFIG_DIR / f"{args.arm}.json"),
    }
    if args.operation == "train":
        resolve_device(config.device, config.precision)
        result = train_flow(
            config,
            train,
            blank,
            identity,
            model=model,
            validation=validation,
            initialization=initialization,
            resume=(root / args.resume) if args.resume else None,
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "completed" or result["step"] != STEPS:
            raise SystemExit(1)
        return
    if args.resume:
        parser.error("--resume applies only to train")
    precision = config.precision if args.device == "cuda" else "fp32"
    device = resolve_device(args.device, precision)
    batch = default_collate([train[index] for index in range(config.batch_size)])
    latent = batch.latent.float().to(device)
    generator = torch.Generator().manual_seed(config.seed)
    noise = torch.randn(latent.shape, generator=generator).to(device)
    times = torch.rand(len(latent), generator=generator).to(device)
    noisy, target = flow_interpolate(latent, noise, times)
    condition = Condition(batch.text.float(), batch.mask).to(device)
    model = model.eval().to(device)
    with torch.inference_mode(), autocast_context(device, precision):
        prediction = model(noisy, times, condition)
        loss = flow_loss(prediction, target)
    if prediction.shape != target.shape or not torch.isfinite(loss):
        raise ValueError("Six-model cache forward returned invalid shape or values")
    print(
        json.dumps(
            {
                "status": "passed",
                "arm": args.arm,
                "optimizer_updates": 0,
                "device": str(device),
                "precision": precision,
                "trainable_parameters": EXPECTED_PARAMETERS[args.arm],
                "image_tokens": (config.model["latent_size"] // config.model["patch_size"]) ** 2,
                "batch_size": config.batch_size,
                "latent_shape": list(latent.shape),
                "prediction_shape": list(prediction.shape),
                "flow_loss": float(loss),
                "train_samples": len(train),
                "validation_samples": len(validation),
                "data_identity": identity,
                "initialization": initialization,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
