"""P3 终端工作流：固定缓存上的训练、条件采样及有限诊断。"""

import hashlib
import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import Subset, default_collate
from torchvision.utils import save_image

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition
from minigen.data.representation_cache import (
    CacheSource,
    RepresentationCache,
    cache_source,
    verify_cache_acceptance,
)
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.samplers.flow import integrate
from minigen.training.checkpoint import load_checkpoint
from minigen.training.config import TrainConfig, build_model
from minigen.training.engine import train_flow
from minigen.training.runtime import resolve_device, synchronize

DEFAULT_DATA = {
    "cache": "artifacts/cache/artbench_pd_256/v1_formal",
    "data_root": "data/processed/artbench_pd_256/v1",
    "freeze": "reports/v2/data/artbench_freeze.json",
    "vae_lock": "configs/models/sdxl_vae.json",
    "clip_lock": "configs/models/clip_text.json",
    "approval": "reports/v2/components/vae_user_approval.json",
    "evidence": "reports/v2/components/vae_reconstruction_200_summary.json",
    "local_acceptance": "reports/v2/components/artbench_local_fp32_20.json",
    "train_limit": 16,
    "validation_limit": 16,
    "train_indices": None,
    "validation_indices": None,
}


def read_experiment(
    root: Path,
    config_path: Path,
) -> tuple[TrainConfig, JsonDict]:
    """解析相对项目根的实验配置；模型规格和训练资源分开保存。

    Args:
        root: 数据与模型路径的共同根，可由终端 --root 指定。
        config_path: JSON 实验文件，含 model、training 和 data。

    Raises:
        ValueError: 未知配置字段或无效子集规模。
        OSError/KeyError: 配置无法读取或缺少必要字段。

    Results:
        tuple: 训练配置以及补齐默认值的数据配置。
    """
    raw = json.loads((root / config_path).read_text())
    if set(raw) != {"model", "training", "data"}:
        raise ValueError("Experiment requires exactly model, training and data sections")
    if set(raw["data"]) - DEFAULT_DATA.keys():
        raise ValueError("Unknown data configuration field")
    model = json.loads((root / raw["model"]).read_text())
    training = dict(raw["training"])
    if "model" in training:
        raise ValueError("Model specification belongs in the separate model file")
    training["model"] = model
    training["output_dir"] = str((root / training["output_dir"]).resolve())
    data = DEFAULT_DATA | raw["data"]
    for name in ("train_limit", "validation_limit"):
        if data[name] is not None and (type(data[name]) is not int or data[name] < 1):
            raise ValueError(f"{name} must be a positive integer or null")
    return TrainConfig.from_dict(training), data


def selected_indices(
    source: CacheSource,
    data: JsonDict,
    split: str,
) -> list[int]:
    """从已冻结划分选择显式子集，或按上限取前缀。

    Args:
        source: 已核验的来源记录。
        data: 可含唯一整数索引列表的配置；显式列表优先于上限。
        split: train 或 validation。

    Raises:
        ValueError: 显式子集为空、重复或越界。
        KeyError: 未声明的划分。

    Results:
        list[int]: 相对于该划分的固定有序行下标。
    """
    count = len(source.records[split])
    indices = data.get(f"{split}_indices")
    if indices is None:
        return list(range(count))[: data[f"{split}_limit"]]
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or not 0 <= index < count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(f"Invalid unique {split} indices")
    return list(indices)


def open_cache(
    root: Path,
    data: JsonDict,
) -> tuple[CacheSource, RepresentationCache, Condition, JsonDict]:
    """核验 P2 正式缓存及两项审批，形成与路径无关的实验身份。

    Args:
        root: 本地资产共同根目录。
        data: 已解析的路径、训练和验证子集上限。

    Raises:
        OSError/ValueError: P2 资产、审核或缓存身份不符。

    Results:
        tuple: 来源、只读缓存、单行空条件和固定实验数据身份。
    """
    source = cache_source(
        root / data["data_root"],
        root / data["freeze"],
        root / data["vae_lock"],
        root / data["clip_lock"],
        None,
    )
    acceptance = verify_cache_acceptance(
        source, root / data["approval"], root / data["evidence"], root / data["local_acceptance"]
    )
    cache = RepresentationCache(
        root / data["cache"],
        source,
        approval_sha256=acceptance.approval_sha256,
        local_acceptance_sha256=acceptance.local_acceptance_sha256,
    )
    selected = {}
    for split in ("train", "validation"):
        ids = [source.records[split][i]["sample_id"] for i in selected_indices(source, data, split)]
        selected[split] = {
            "count": len(ids),
            "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
    identity = {
        "cache_index_sha256": hash_file(root / data["cache"] / "index.json"),
        "source": source.identity,
        "subsets": selected,
    }
    features, mask = cache.blank()
    return source, cache, Condition(features[None].float(), mask[None]), identity


def train_command(
    args: Namespace,
) -> JsonDict:
    """把冻结正式缓存的确定子集交给通用训练器。

    Args:
        args: train 命令解析结果，含配置、根目录、设备和恢复路径。

    Raises:
        OSError/ValueError/RuntimeError: 配置、资产、恢复或训练失败。

    Results:
        JsonDict: 训练器记录的真实步数、停止原因和产物。

    副作用:
        执行有限训练，写日志及 Checkpoint。
    """
    root = args.root.resolve()
    config, data = read_experiment(root, args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    source, cache, blank, identity = open_cache(root, data)
    dataset = cache.split("train")
    valid = cache.split("validation")
    return train_flow(
        config,
        Subset(dataset, selected_indices(source, data, "train")),
        blank,
        identity,
        resume=(root / args.resume) if args.resume else None,
        validation=Subset(valid, selected_indices(source, data, "validation")),
    )


def _loaded_run(
    args: Namespace,
) -> tuple[TrainConfig, JsonDict, CacheSource, RepresentationCache, Condition, torch.nn.Module]:
    """加载 Checkpoint 并核对训练时的数据身份，便于采样及独立评估共用。

    Args:
        args: 含根目录、配置、Checkpoint、设备与权重选择的命令参数。

    Raises:
        ValueError/OSError/RuntimeError: 身份不符、资产或权重无法读取。

    Results:
        tuple: 配置、数据设置、来源、缓存、空条件及 eval 主干。
    """
    root = args.root.resolve()
    config, data = read_experiment(root, args.config)
    source, cache, blank, identity = open_cache(root, data)
    state = load_checkpoint(root / args.checkpoint)
    if state["data_identity"] != identity or state["config"]["model"] != config.model:
        raise ValueError("Checkpoint data or architecture identity differs from experiment")
    device = resolve_device(args.device, "fp32")
    torch.set_num_threads(config.cpu_threads)
    model = build_model(config.model).to(device)
    model.load_state_dict(state[args.weights], strict=True)
    model.eval().requires_grad_(False)
    return config, data, source, cache, blank.to(device), model


def _conditions(
    args: Namespace,
    source: CacheSource,
    data: JsonDict,
    cache: RepresentationCache,
    blank: Condition,
    device: torch.device,
) -> tuple[Condition, list[JsonDict]]:
    """读取固定缓存条件并施加明确的正确、空或循环打乱对照。

    Args:
        args: 含划分、起点、数量和条件模式的命令参数。
        source: 与缓存匹配的冻结清单。
        data: 含固定子集索引的数据配置。
        cache: 已校验的正式只读缓存。
        blank: 编码器的空文本特征。
        device: 主干计算设备。

    Raises:
        ValueError: 请求越界，或仅一项时请求打乱条件。

    Results:
        tuple: 实际输入条件及逐行的样本与 Caption 来源。
    """
    rows = source.records[args.split]
    selected = selected_indices(source, data, args.split)
    if args.count < 1 or args.start_index < 0 or args.start_index + args.count > len(selected):
        raise ValueError("Requested sample range exceeds split")
    if args.condition_mode == "shuffled" and args.count < 2:
        raise ValueError("Shuffled comparison requires at least two conditions")
    indices = selected[args.start_index : args.start_index + args.count]
    order = indices[1:] + indices[:1] if args.condition_mode == "shuffled" else indices
    batch = default_collate([cache.split(args.split)[i] for i in order])
    condition = Condition(batch.text.float(), batch.mask).to(device)
    if args.condition_mode == "blank":
        condition = blank.repeat_batch(args.count)
    records = [
        {
            "sample_id": rows[i]["sample_id"],
            "condition_sample_id": rows[j]["sample_id"],
            "caption": rows[j]["caption"] if args.condition_mode != "blank" else "",
        }
        for i, j in zip(indices, order, strict=True)
    ]
    return condition, records


def sample_command(
    args: Namespace,
) -> JsonDict:
    """执行完整 Flow ODE 积分和冻结 VAE 解码，保存每张图片及真实求值次数。

    Args:
        args: 配置、Checkpoint、采样参数和新输出目录。

    Raises:
        FileExistsError: 输出目录已存在。
        ValueError/OSError/RuntimeError: 数据、条件、采样或解码失败。

    Results:
        JsonDict: 完整生成记录，含种子、条件、求值次数、延迟与文件路径。

    副作用:
        加载冻结模型，保存 PNG、潜变量和 JSON；不更新模型参数。
    """
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    config, data, source, cache, blank, model = _loaded_run(args)
    device = next(model.parameters()).device
    if args.prompts is None:
        condition, records = _conditions(args, source, data, cache, blank, device)
    else:
        if args.condition_mode != "correct":
            raise ValueError("Prompt files currently require correct condition mode")
        prompts = json.loads((root / args.prompts).read_text())
        text = FrozenCLIPText.from_lock(root / data["clip_lock"], root=root, device=device)
        encoded = text.encode(prompts)
        condition = Condition(encoded.features, encoded.mask)
        records = [{"caption": prompt} for prompt in prompts]
        del text
    generator = torch.Generator().manual_seed(args.seed)
    shape = (
        len(records),
        config.model["in_channels"],
        config.model["latent_size"],
        config.model["latent_size"],
    )
    initial = torch.randn(shape, generator=generator).to(device)
    synchronize(device)
    denoise_started = perf_counter()
    result = integrate(model, initial, condition, blank, args.steps, args.guidance, args.method)
    synchronize(device)
    denoise_seconds = perf_counter() - denoise_started
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=device)
    decoded = vae.decode(result.samples)
    images = (decoded.cpu().clamp(-1, 1) + 1) / 2
    output.mkdir(parents=True)
    save_image(images, output / "grid.png", nrow=min(4, len(images)))
    for index, image in enumerate(images):
        save_image(image, output / f"sample_{index:03d}.png")
    torch.save(result.samples.cpu(), output / "latents.pt")
    report = {
        "checkpoint_sha256": hash_file(root / args.checkpoint),
        "weights": args.weights,
        "device": str(device),
        "precision": "fp32",
        "seed": args.seed,
        "method": args.method,
        "steps": args.steps,
        "guidance": args.guidance,
        "solver_nfe": result.solver_nfe,
        "model_evaluations": result.model_evaluations,
        "condition_mode": args.condition_mode,
        "split": args.split if args.prompts is None else "prompts",
        "records": records,
        "denoise_seconds": denoise_seconds,
        "timing_warmup": False,
        "total_seconds_including_load_decode_save": perf_counter() - started,
        "output": str(output),
        "source": source.identity,
    }
    (output / "samples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def evaluate_command(
    args: Namespace,
) -> JsonDict:
    """在固定时间和噪声上评估 Flow MSE，区分训练子集与验证集诊断。

    Args:
        args: 配置、权重、划分、范围、条件对照和随机种子。

    Raises:
        FileExistsError: 目标结果文件已经存在。
        ValueError/OSError/RuntimeError: 配置、数据或前向评估失败。

    Results:
        JsonDict: 固定配对的原始/零速度基线 MSE；不代表生成质量评分。

    副作用:
        保存真实条件对照及逐项损失 JSON；不更新参数。
    """
    output = args.root.resolve() / args.output
    if output.exists():
        raise FileExistsError(output)
    _, data, source, cache, blank, model = _loaded_run(args)
    device = next(model.parameters()).device
    condition, records = _conditions(args, source, data, cache, blank, device)
    dataset = cache.split(args.split)
    indices = selected_indices(source, data, args.split)[
        args.start_index : args.start_index + args.count
    ]
    batch = default_collate([dataset[i] for i in indices])
    target = batch.latent.to(device, dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(target.shape, generator=generator).to(device)
    times = torch.rand(args.count, generator=generator).to(device)
    noisy, velocity = flow_interpolate(target, noise, times)
    with torch.no_grad():
        predicted = model(noisy, times, condition)
        loss = flow_loss(predicted, velocity).item()
        per_sample = (
            (predicted.float() - velocity.float()).square().flatten(1).mean(1).cpu().tolist()
        )
    report = {
        "metric": "fixed_pair_flow_velocity_mse",
        "generation_quality_evaluated": False,
        "split": args.split,
        "condition_mode": args.condition_mode,
        "seed": args.seed,
        "samples": len(records),
        "loss": loss,
        "zero_velocity_baseline": velocity.square().mean().item(),
        "per_sample_mse": per_sample,
        "times": times.cpu().tolist(),
        "records": records,
        "checkpoint_sha256": hash_file(args.root.resolve() / args.checkpoint),
        "weights": args.weights,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
