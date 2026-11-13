"""用独立动漫缓存驱动 MiniGen 的 B1 前向验收、训练和采样。"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from cache_anime_faces import anime_source

from minigen import commands
from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.contracts import Condition
from minigen.data.representation_cache import SPLITS, CacheSource, RepresentationCache
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.config import TrainConfig, build_model
from minigen.training.runtime import resolve_device

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/training/anime_b1_10000.json")
VALIDATION_SEED = 20261113
DEVICES = ("auto", "cpu", "mps", "cuda")


def open_anime_cache(
    root: Path,
    data: JsonDict,
    smoke_rows: int | None = None,
) -> tuple[CacheSource, RepresentationCache, Condition, JsonDict]:
    """以动漫清单与固定模型锁验收缓存，绑定训练及采样使用的固定子集。

    Args:
        root: 项目及缓存所在目录；允许没有原始图像，仅消费已验证的缓存。
        data: 项目数据配置；freeze 指向动漫划分回执，清单与回执位于同一目录。
        smoke_rows: 验收 smoke 快照时的固定前缀数量；训练和采样始终使用全量。

    Raises:
        OSError: 清单、模型锁文件或缓存索引及数组不可读。
        ValueError: 身份、哈希、行顺序或显式子集不符。

    Results:
        tuple: 有序来源、缓存、FP32 空条件，以及与绝对路径无关的实验数据身份。

    副作用:
        若没有显式验证下标，以样本 ID 和固定种子确定子集并写入 data。
    """
    receipt = root / data["freeze"]
    source = anime_source(
        receipt.with_name("manifest.jsonl"),
        receipt,
        root / data["data_root"],
        root / data["vae_lock"],
        root / data["clip_lock"],
        smoke_rows,
    )
    cache = RepresentationCache(root / data["cache"], source)
    if data["validation_indices"] is None:
        rows = source.records["validation"]
        order = sorted(
            range(len(rows)),
            key=lambda i: hashlib.sha256(
                f"{VALIDATION_SEED}:{rows[i]['sample_id']}".encode()
            ).digest(),
        )
        data["validation_indices"] = order[: data["validation_limit"]]
    subsets = {}
    for split in SPLITS:
        ids = [
            source.records[split][i]["sample_id"]
            for i in commands.selected_indices(source, data, split)
        ]
        subsets[split] = {
            "count": len(ids),
            "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
    identity = {
        "cache_index_sha256": hash_file(root / data["cache"] / "index.json"),
        "source": source.identity,
        "subsets": subsets,
    }
    features, mask = cache.blank()
    return source, cache, Condition(features[None].float(), mask[None]), identity


def check_forward(
    args: argparse.Namespace,
    config: TrainConfig,
    data: JsonDict,
) -> JsonDict:
    """逐行回读缓存后，用真实验证表示执行无梯度的单样本 B1 前向。

    Args:
        args: 项目根、缓存覆盖路径、设备、线程数和可选的 smoke 前缀。
        config: 已验证架构为 B1 的训练配方。
        data: 与本次模型配方配套的缓存及子集配置。

    Raises:
        OSError: 缓存或输入文件不可读。
        ValueError: 设备、缓存身份、张量形状或前向数值不合法。
        RuntimeError: 底层设备或模型计算失败。

    Results:
        JsonDict: 真实缓存行数、固定子集身份和 FP32 前向摘要。

    副作用:
        设置当前进程的 CPU 线程数与随机种子。前向不更新任何模型参数。
    """
    if args.cache is not None:
        data["cache"] = str(args.cache)
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device, "fp32")
    source, cache, blank, identity = open_anime_cache(args.root, data, args.smoke_rows)
    counts = cache.verify_rows(source)
    index = commands.selected_indices(source, data, "validation")[0]
    row = cache.split("validation")[index]
    target = row.latent[None].to(device, dtype=torch.float32)
    condition = Condition(row.text[None].float(), row.mask[None]).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    noise = torch.randn(target.shape, generator=generator).to(device)
    times = torch.rand(1, generator=generator).to(device)
    noisy, velocity = flow_interpolate(target, noise, times)
    torch.manual_seed(config.seed)
    model = build_model(config.model).to(device).eval()
    with torch.inference_mode():
        prediction = model(noisy, times, condition)
        blank_prediction = model(noisy, times, blank.to(device))
        loss = flow_loss(prediction, velocity)
    if (
        prediction.shape != velocity.shape
        or blank_prediction.shape != velocity.shape
        or not torch.isfinite(prediction).all()
        or not torch.isfinite(blank_prediction).all()
        or not torch.isfinite(loss)
    ):
        raise ValueError("Anime B1 forward produced invalid shape or nonfinite values")
    return {
        "status": "passed",
        "cache_mode": source.identity["kind"],
        "verified_rows": counts,
        "cache_index_sha256": identity["cache_index_sha256"],
        "manifest_sha256": source.identity["manifest_sha256"],
        "split_receipt_sha256": source.identity["split_receipt_sha256"],
        "subsets": identity["subsets"],
        "blank_valid_tokens": int(blank.mask.sum()),
        "forward": {
            "device": str(device),
            "precision": "fp32",
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "sample_id": row.sample_id,
            "latent_shape": list(target.shape),
            "condition_shape": list(condition.features.shape),
            "condition_valid_tokens": int(condition.mask.sum()),
            "velocity_shape": list(prediction.shape),
            "flow_mse": float(loss),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_updates": 0,
        },
        "training_recipe": config.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    """声明独立 B1 的验收、训练和采样入口。

    Raises:
        SystemExit: 命令行参数不合法或仅请求帮助。

    Results:
        argparse.Namespace: 校验后的命令选项。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "逐行回读缓存并执行一次无更新 B1 前向"),
        ("train", "从完整动漫缓存训练或恢复 B1"),
        ("sample", "使用动漫 B1 检查点和已有 Flow 采样入口"),
    ):
        child = subcommands.add_parser(name, help=help_text)
        child.add_argument("--root", type=Path, default=ROOT)
        child.add_argument("--config", type=Path, default=CONFIG)
        if name == "check":
            child.add_argument("--cache", type=Path)
            child.add_argument("--smoke-rows", type=int)
            child.add_argument("--device", choices=DEVICES, default="cpu")
            child.add_argument("--cpu-threads", type=int, default=1)
        elif name == "train":
            child.add_argument("--resume", type=Path)
            child.set_defaults(device=None)
        else:
            child.add_argument("--checkpoint", type=Path, required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--device", choices=DEVICES, default="auto")
            child.add_argument("--weights", choices=("ema", "model"), default="ema")
            child.add_argument("--split", choices=SPLITS, default="validation")
            child.add_argument(
                "--condition-mode", choices=("correct", "blank", "shuffled"), default="correct"
            )
            child.add_argument("--count", type=int, default=16)
            child.add_argument("--start-index", type=int, default=0)
            child.add_argument("--seed", type=int, default=1234)
            child.add_argument("--prompts", type=Path)
            child.add_argument("--steps", type=int, default=40)
            child.add_argument("--method", choices=("euler", "heun"), default="euler")
            child.add_argument("--guidance", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """检查独立配置，再将动漫缓存显式交给共享终端工作流。

    Raises:
        OSError: 配置、缓存或输出无法访问。
        ValueError: 配置不是 B1，或运行设备、缓存或张量不符合契约。
        RuntimeError: 底层前向、训练或采样失败。

    副作用:
        输出 JSON。仅在显式 train 或 sample 子命令中写训练或采样产物。
    """
    args = parse_args()
    args.root = args.root.resolve()
    config, data = commands.read_experiment(args.root, args.config)
    if config.model.get("architecture") != "b1":
        raise ValueError("Anime B1 runner requires a B1 architecture")
    if args.command == "check":
        if args.cpu_threads < 1:
            raise ValueError("CPU threads must be positive")
        report = check_forward(args, config, data)
    else:
        if args.command == "train":
            resolve_device(config.device, config.precision)
        # 仅替换当前 CLI 调用的加载器，沿用训练、采样、输出及恢复契约。
        with patch.object(commands, "open_cache", new=open_anime_cache):
            report = (
                commands.train_command(args)
                if args.command == "train"
                else commands.sample_command(args)
            )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
