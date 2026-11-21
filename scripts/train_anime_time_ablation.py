"""从相同的 B1 10k EMA 初始化时间采样配方，复用 MiniGen 正式训练循环。"""

import argparse
import json
import math
from pathlib import Path

import torch
from run_anime_b1 import open_anime_cache
from torch.utils.data import Subset

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.contracts import Condition
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.checkpoint import run_identity
from minigen.training.config import TrainConfig, build_model
from minigen.training.engine import train_flow
from minigen.training.runtime import resolve_device
from minigen.training.time_sampling import flow_training_times

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("artifacts/runs/anime_faces_256")
SOURCE = RUNS / "b1_v1_10000" / "step_010000.pt"
SOURCE_SHA = "1325beb724b55917f64575a24da20cc8e4af1bb4f6fc6989541de4a59dbd8814"
PARENT_RUN_ID = "d9be223b5abe251e9b2e00dd2d1cc08332714ea144f229d7424ca96d4acad007"
PROBABILITIES = {"uniform": 0.4, "mid60": 0.6}
INITIALIZATION = {"checkpoint_sha256": SOURCE_SHA, "checkpoint_step": 10000, "weights": "ema"}


def paired_configs(
    root: Path,
) -> dict[str, tuple[TrainConfig, JsonDict]]:
    """读取两组配置并核对共享训练配方和数据来源。

    Args:
        root: MiniGen 项目根目录。

    Raises:
        OSError: 实验或模型配置文件不可读。
        ValueError: 两组模型、初始化预算、采样或共享配置不一致。

    Results:
        dict: 从组名到训练及数据配置的映射。
    """
    configs = {
        arm: read_experiment(root, Path(f"configs/training/anime_b1_time_{arm}.json"))
        for arm in PROBABILITIES
    }
    recipes = []
    for arm, (config, _) in configs.items():
        expected_path = root / RUNS / "b1_time_ablation_v1" / arm
        if (
            config.model["architecture"] != "b1"
            or config.time_mid_probability != PROBABILITIES[arm]
            or config.max_steps != 3000
            or config.device != "cuda"
            or config.precision != "bf16"
            or Path(config.output_dir) != expected_path
        ):
            raise ValueError(f"Unexpected B1 ablation configuration: {arm}")
        recipes.append(
            {
                k: v
                for k, v in config.to_dict().items()
                if k not in {"output_dir", "time_mid_probability"}
            }
        )
    if recipes[0] != recipes[1] or configs["uniform"][1] != configs["mid60"][1]:
        raise ValueError("The arms must use identical training and data settings")
    return configs


def load_initial_model(
    root: Path,
    config: TrainConfig,
    data_identity: JsonDict,
) -> torch.nn.Module:
    """从已验收快照读取 EMA 权重并复制到独立的 B1 参数。

    Args:
        root: MiniGen 项目根目录。
        config: 消融组的训练与架构配置。
        data_identity: 当前训练和验证缓存的真实身份。

    Raises:
        OSError/RuntimeError: 检查点读取或严格权重加载失败。
        ValueError: 文件、训练来源、模型架构或数据不一致。

    Results:
        nn.Module: CPU 上持有独立可训练参数的 B1。
    """
    path = root / SOURCE
    if hash_file(path) != SOURCE_SHA:
        raise ValueError("Starting checkpoint SHA-256 changed")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        state["step"] != 10000
        or state["objective"] != "flow_velocity"
        or state["data_identity"] != data_identity
        or state["config"]["model"] != config.model
        or run_identity(state["contract"]) != PARENT_RUN_ID
    ):
        raise ValueError("Starting EMA checkpoint does not match B1 10k run")
    # assign=False 把 mmap 权重复制到独立参数，后续优化不会共享映射。
    model = build_model(config.model)
    model.load_state_dict(state["ema"], strict=True)
    return model


def main() -> None:
    """检查配方或显式启动一组训练，写入独立运行目录。

    Raises:
        OSError/RuntimeError: 缓存、权重、运行设备或训练失败。
        ValueError: 两组配置、来源或数值预检不一致。

    副作用:
        输出 JSON。仅 train 子命令写入消融训练产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "train"))
    parser.add_argument("--arm", choices=PROBABILITIES, default="uniform")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    configs = paired_configs(root)
    config, data = configs[args.arm]
    torch.set_num_threads(1)
    if args.operation == "train":
        resolve_device(config.device, config.precision)
    source, cache, blank, identity = open_anime_cache(root, data)
    if args.operation == "check":
        if args.resume:
            parser.error("--resume applies only to train")
        model = load_initial_model(root, config, identity).eval()
        valid = cache.split("validation")
        row = valid[selected_indices(source, data, "validation")[0]]
        latent = row.latent[None].float()
        noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(1234))
        condition = Condition(row.text[None].float(), row.mask[None])
        uniform = (torch.arange(100_000, dtype=torch.float64) + 0.5) / 100_000
        observed = {}
        losses = {}
        for arm, probability in PROBABILITIES.items():
            mapped = flow_training_times(uniform, probability)
            observed[arm] = float(((mapped >= 0.3) & (mapped < 0.7)).double().mean())
            if not math.isclose(observed[arm], probability, abs_tol=1e-5):
                raise ValueError(f"Unexpected distribution for {arm}")
            if not torch.all(mapped[1:] > mapped[:-1]) or not (0 < mapped[0] < mapped[-1] < 1):
                raise ValueError(f"Time mapping lost ordering or full support: {arm}")
            times = flow_training_times(torch.tensor([0.25]), probability)
            noisy, target = flow_interpolate(latent, noise, times)
            with torch.inference_mode():
                prediction = model(noisy, times, condition)
                losses[arm] = float(flow_loss(prediction, target))
            if not math.isfinite(losses[arm]):
                raise ValueError(f"Nonfinite starting EMA forward: {arm}")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "optimizer_updates": 0,
                    "device": "cpu",
                    "initialization": INITIALIZATION,
                    "cache_index_sha256": identity["cache_index_sha256"],
                    "sample_id": row.sample_id,
                    "mid_probabilities": observed,
                    "forward_losses": losses,
                    "configs_sha256": {
                        arm: hash_file(root / f"configs/training/anime_b1_time_{arm}.json")
                        for arm in PROBABILITIES
                    },
                },
                indent=2,
            )
        )
        return
    model = None if args.resume else load_initial_model(root, config, identity)
    train = Subset(cache.split("train"), selected_indices(source, data, "train"))
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    result = train_flow(
        config,
        train,
        blank,
        identity,
        model=model,
        validation=validation,
        initialization=INITIALIZATION,
        resume=(root / args.resume) if args.resume else None,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "completed" or result["step"] != config.max_steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
