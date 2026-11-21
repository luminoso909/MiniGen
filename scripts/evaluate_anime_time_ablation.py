"""用固定验证图和眼色换词对照评测 B1 时间采样消融的两个训练分支。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from run_anime_b1 import CONFIG, ROOT, open_anime_cache
from torchvision.utils import save_image

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import read_experiment, selected_indices
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition, FlowModel
from minigen.samplers.flow import integrate
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, FLOW_OBJECTIVE, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device, synchronize

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor, nn

CHECKPOINT_ROOT = Path("artifacts/runs/anime_faces_256/b1_time_ablation_v1")
OUTPUT = Path("artifacts/exports/anime-faces-b1/time_ablation_3000")
PLAN = Path("configs/evaluation/anime_b1_time_eyes.json")
INITIALIZATION = {
    "checkpoint_sha256": "1325beb724b55917f64575a24da20cc8e4af1bb4f6fc6989541de4a59dbd8814",
    "checkpoint_step": 10000,
    "weights": "ema",
}
ARMS = {"uniform": 0.4, "mid60": 0.6}
STAGE_STEPS = 3000
COUNT = 16
EYE_PAIRS = 6
SEED = 1234
STEPS = 40
GUIDANCE = 3.0


def tensor_digest(
    value: Float[Tensor, "b c_z h_z w_z"],  # noqa: F722
) -> str:
    """计算固定初噪的连续 CPU FP32 字节摘要，逐行摘要保留单元素批量轴。

    Args:
        value: [b,c_z,h_z,w_z] 的 CPU FP32 噪声。

    Results:
        str: 初噪原始字节的 SHA-256。
    """
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def eye_records(
    plan: list[JsonDict],
    correct: list[JsonDict],
) -> list[JsonDict]:
    """验证眼色计划对应同一批验证样本，再构造原标签和单标签替换对照。

    Args:
        plan: 固定的 6 条换词行，或完整的 16 条含空操作行，按原验证次序排列。
        correct: 由固定缓存取得的 16 条正确条件记录，含有序索引和噪声身份。

    Raises:
        ValueError: 行顺序、样本身份、标签替换位置或有效配对数不符。

    Results:
        list[JsonDict]: 成对交替排列的原标签和换词记录，两条沿用同一初噪。
    """
    positions = [row["index"] for row in plan]
    if (
        len(plan) not in {EYE_PAIRS, COUNT}
        or any(type(i) is not int or not 0 <= i < COUNT for i in positions)
        or positions != sorted(set(positions))
    ):
        raise ValueError("Invalid eye plan row order or count")
    records = []
    for row in plan:
        base = correct[row["index"]]
        if any(row[key] != base[key] for key in ("sample_id", "caption")):
            raise ValueError(f"Eye plan differs from locked validation row {row['index']}")
        before, after = row["eye_from"], row["eye_to"]
        if before is None and after is None:
            continue
        tags = base["caption"].split(", ")
        if (
            not isinstance(before, str)
            or not isinstance(after, str)
            or not after
            or "," in after
            or not before.endswith(" eyes")
            or not after.endswith(" eyes")
            or tags.count(before) != 1
            or after in tags
        ):
            raise ValueError(f"Invalid eye tag substitution at validation row {row['index']}")
        tags[tags.index(before)] = after
        pair = base | {"pair_index": len(records) // 2, "eye_from": before, "eye_to": after}
        records.append(pair | {"variant": "base"})
        records.append(pair | {"variant": "swap", "caption": ", ".join(tags)})
    if len(records) != 2 * EYE_PAIRS:
        raise ValueError("Eye plan must contain six single-tag substitutions")
    return records


def read_arm(
    path: Path,
    arm: str,
    model_config: JsonDict,
    data_identity: JsonDict,
) -> tuple[nn.Module, JsonDict, JsonDict]:
    """核对已完成分支的训练元数据并只装载用于采样的 CPU EMA 权重。

    Args:
        path: 分支最终检查点，与 run.json 及 result.json 同级。
        arm: 预先登记的 uniform 或 mid60 分支名。
        model_config: 基础动漫 B1 配方指定的模型结构。
        data_identity: 已回读的动漫缓存、划分与固定子集身份。

    Raises:
        OSError/RuntimeError: 运行产物不可读或张量无法反序列化、装载。
        ValueError: 分支未完成或检查点的训练、结构、数据身份不符。

    Results:
        tuple: 只读 EMA 模型、采样回执的训练身份字段，以及剔除处理变量的公共训练契约。
    """
    run = json.loads((path.parent / "run.json").read_text())
    result_path = path.parent / "result.json"
    result = json.loads(result_path.read_text())
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    contract = state["contract"]
    recipe = contract["recipe"]
    config = json.loads(json.dumps(state["config"]))
    run_id = run_identity(contract)
    metadata = {
        "objective": FLOW_OBJECTIVE,
        "architecture": "b1",
        "run_id": run_id,
        "source_sha256": contract["source_sha256"],
        "model_config": model_config,
        "data_identity": data_identity,
    }
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["step"] != STAGE_STEPS
        or state["objective"] != FLOW_OBJECTIVE
        or config["model"] != model_config
        or config["max_steps"] != STAGE_STEPS
        or state["data_identity"] != data_identity
        or contract["data"] != data_identity
        or contract["initialization"] != INITIALIZATION
        or recipe["time_mid_probability"] != ARMS[arm]
        or config["time_mid_probability"] != ARMS[arm]
        or run["config"] != config
        or run["contract"] != contract
        or run["run_id"] != run_id
        or result["attempt_id"] != run["attempt_id"]
        or result["status"] != "completed"
        or result["step"] != STAGE_STEPS
        or result["device"] != contract["device"]
        or not result["device"].startswith("cuda")
        or result["precision"] != "bf16"
        or result["performance"]["optimizer_steps"] != STAGE_STEPS
        or any(result[key] != expected for key, expected in metadata.items())
    ):
        raise ValueError(f"Checkpoint and completed training receipts do not match arm {arm}")
    metrics_path = path.parent / "metrics.jsonl"
    metrics_bytes = metrics_path.read_bytes()
    metrics = [json.loads(line) for line in metrics_bytes.splitlines()]
    batch = config["batch_size"] * config["grad_accum"]
    train_rows = data_identity["subsets"]["train"]["count"]
    tail = train_rows % batch or batch
    epoch_steps = (train_rows + batch - 1) // batch
    expected_total = STAGE_STEPS * batch - (STAGE_STEPS // epoch_steps) * (batch - tail)
    if (
        len(metrics) != STAGE_STEPS
        or metrics != state["history"]
        or any(
            row["step"] != step
            or row["samples"] != (tail if step % epoch_steps == 0 else batch)
            or type(row["mid_time_samples"]) is not int
            or not 0 <= row["mid_time_samples"] <= row["samples"]
            for step, row in enumerate(metrics, start=1)
        )
        or result["performance"]["samples_seen"] != expected_total
    ):
        raise ValueError(f"Invalid optimizer update or time-sampling records for arm {arm}")
    middle_count = sum(row["mid_time_samples"] for row in metrics)
    middle_fraction = middle_count / expected_total
    if abs(middle_fraction - ARMS[arm]) > 0.01:
        raise ValueError(f"Observed mid-time share differs for arm {arm}: {middle_fraction:.4f}")
    weights = state["ema"]
    if {key: [list(t.shape), str(t.dtype)] for key, t in weights.items()} != contract["tensors"]:
        raise ValueError(f"EMA tensor layout does not match arm {arm}")
    with torch.device("meta"):
        model = build_model(config["model"])
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    receipt = metadata | {
        "arm": arm,
        "objective_source": "explicit",
        "checkpoint_step": state["step"],
        "checkpoint_sha256": hash_file(path),
        "training_result_sha256": hash_file(result_path),
        "training_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "trainable_parameters": result["trainable_parameters"],
        "initialization": INITIALIZATION,
        "time_mid_probability": ARMS[arm],
        "observed_time_mid_fraction": middle_fraction,
        "observed_mid_time_samples": middle_count,
        "observed_total_time_samples": expected_total,
    }
    shared_recipe = {key: value for key, value in recipe.items() if key != "time_mid_probability"}
    return model, receipt, contract | {"recipe": shared_recipe}


def sample_variant(
    target: Path,
    model: FlowModel,
    vae: FrozenVAE,
    noise: Float[Tensor, "b c_z h_z w_z"],  # noqa: F722
    condition: Condition,
    blank: Condition,
    records: list[JsonDict],
    metadata: JsonDict,
    *,
    seed: int = SEED,
) -> None:
    """复用标准 Euler 和 VAE 入口生成一组图片及其采样回执。

    Args:
        target: 本变体的全新输出目录。
        model: 当前分支 EMA 权重，处于 FP32 eval 模式。
        vae: 与缓存同锁的冻结 VAE。
        noise: [b,c_z,h_z,w_z]，已经按本变体顺序挑选的完全相同初噪。
        condition: [b,n,d] 条件特征及其掩码。
        blank: [1,n,d] 空条件特征及其掩码。
        records: 与本变体一一对应的原验证样本及输入标签。
        metadata: 两个分支共用的协议及本分支检查点身份。
        seed: 实际产生当前初噪张量的随机种子。

    Raises:
        OSError: 目标目录或产物写入失败。
        ValueError/FloatingPointError/RuntimeError: 积分、解码或设备计算失败。

    副作用:
        写入样图、网格及 JSON 采样回执。
    """
    started = perf_counter()
    synchronize(noise.device)
    denoise_started = perf_counter()
    result = integrate(model, noise, condition, blank, STEPS, GUIDANCE, "euler")
    synchronize(noise.device)
    denoise_seconds = perf_counter() - denoise_started
    # 分批解码避免 16 张 256px 图的激活同时驻留；积分仍使用完整变体批量。
    images = torch.cat([vae.decode(batch).cpu() for batch in result.samples.split(4)])
    images = (images.clamp(-1, 1) + 1) / 2
    target.mkdir(parents=True)
    save_image(images, target / "grid.png", nrow=4)
    saved = []
    for i, (image, record) in enumerate(zip(images, records, strict=True)):
        name = f"sample_{i:03d}.png"
        save_image(image, target / name)
        saved.append(record | {"image_path": name})
    report = metadata | {
        "schema": "minigen.samples.v1",
        "weights": "ema",
        "device": str(noise.device),
        "precision": "fp32",
        "seed": seed,
        "method": "euler",
        "steps": STEPS,
        "guidance": GUIDANCE,
        "solver_nfe": result.solver_nfe,
        "model_evaluations": result.model_evaluations,
        "condition_mode": target.name,
        "split": "validation",
        "condition_encoding": "cached" if target.name == "correct" else "cpu_clip_fp16_roundtrip",
        "records": saved,
        "denoise_seconds": denoise_seconds,
        "samples_per_denoise_second": len(records) / denoise_seconds,
        "total_seconds_including_decode_save": perf_counter() - started,
        "output": str(target),
    }
    (target / "samples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{target}: {len(records)} images", flush=True)


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
) -> None:
    """读取同一缓存和眼色计划，对两个已完成分支执行固定采样协议。

    Args:
        args: 项目、训练目录、眼色计划和采样输出路径。

    Raises:
        OSError: 输入或输出不可用，或目标目录已存在。
        ValueError/RuntimeError: 缓存、计划、模型或 CUDA 校验及计算失败。

    副作用:
        配置当前进程的计算精度和线程数，写入各分支独立评测产物。
    """
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    device = resolve_device("cuda", "fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, data = read_experiment(root, args.config)
    torch.set_num_threads(config.cpu_threads)
    source, cache, blank, data_identity = open_anime_cache(root, data)
    indices = selected_indices(source, data, "validation")[:COUNT]
    if len(indices) != COUNT:
        raise ValueError("Expected sixteen locked validation samples")
    dataset = cache.split("validation")
    entries = [dataset[i] for i in indices]
    latent_size = config.model["latent_size"]
    shape = (COUNT, config.model["in_channels"], latent_size, latent_size)
    # 与既有采样入口一致，一次性在 CPU 上生成完整 [16,4,32,32] 初噪。
    noise = torch.randn(
        shape, generator=torch.Generator(device="cpu").manual_seed(SEED), dtype=torch.float32
    )
    correct = []
    for i, index in enumerate(indices):
        row = source.records["validation"][index]
        correct.append(
            {
                "index": i,
                "sample_id": row["sample_id"],
                "condition_sample_id": row["sample_id"],
                "source_image_path": row["image_path"],
                "condition_image_path": row["image_path"],
                "caption": row["caption"],
                "noise_sha256": tensor_digest(noise[i : i + 1]),
            }
        )
    plan_bytes = (root / args.plan).read_bytes()
    control = eye_records(json.loads(plan_bytes)["rows"], correct)
    arms = {}
    for arm in ARMS:
        path = root / args.checkpoint_root / arm / "last.pt"
        arms[arm] = read_arm(path, arm, config.model, data_identity)
    if arms["uniform"][2] != arms["mid60"][2]:
        raise ValueError("Training arm contracts differ beyond time sampling")
    common = {
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "noise_sha256": tensor_digest(noise),
        "noise_shape": list(noise.shape),
        "noise_device": "cpu",
        "source_image_root": data["data_root"],
        "source": source.identity,
        "evaluation_torch": str(torch.__version__),
        "tf32": False,
    }
    cached = Condition(
        torch.stack([entry.text for entry in entries]).float(),
        torch.stack([entry.mask for entry in entries]),
    ).to(device)
    clip = FrozenCLIPText.from_lock(root / data["clip_lock"], root=root, device="cpu")
    encoded = clip.encode([row["caption"] for row in control])
    # 两种标签共用编码批次和缓存同款精度，排除编码硬件与量化途径的差异。
    eye = Condition(encoded.features.half().float(), encoded.mask).to(device)
    del clip
    initial = noise.to(device)
    eye_initial = initial[[row["index"] for row in control]]
    blank = blank.to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    output.mkdir(parents=True)
    for arm in ARMS:
        model, meta, _ = arms.pop(arm)
        model = model.to(device)
        sample_variant(
            output / arm / "correct", model, vae, initial, cached, blank, correct, meta | common
        )
        sample_variant(
            output / arm / "eye_control",
            model,
            vae,
            eye_initial,
            eye,
            blank,
            control,
            meta | common,
        )
        del model


def main() -> None:
    """解析命令参数并评测第 3,000 步的双分支 EMA。

    副作用:
        读取终端参数，写入采样产物并打印各组输出位置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
