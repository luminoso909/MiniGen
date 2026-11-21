"""使用已批准的动漫头像题单，在 CUDA 上生成一组 B1 消融检查点的盲评素材。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from evaluate_anime_time_ablation import ARMS, CHECKPOINT_ROOT, read_arm, tensor_digest
from evaluate_anime_uniform_continuation import FINAL_STEP, read_uniform_milestone
from jaxtyping import Float
from run_anime_b1 import CONFIG, ROOT, open_anime_cache
from torch import Tensor
from torchvision.utils import save_image

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import read_experiment
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition, FlowModel
from minigen.data.representation_cache import CacheSource
from minigen.samplers.flow import integrate
from minigen.training.runtime import resolve_device, synchronize

BENCHMARK = Path("configs/benchmark/anime_faces_v1.json")
LOCK = Path("configs/benchmark/anime_faces_v1_lock.json")
APPROVAL = Path("configs/benchmark/anime_faces_v1_approval.json")
RUBRIC = Path("reports/v2/anime/benchmark_v1_rubric.md")
OUTPUT = Path("artifacts/exports/anime-faces-b1/benchmark_v1")
SEEDS = [2026111401, 2026111402]
NOISE_SHAPE = (104, 4, 32, 32)
STEPS = 40
GUIDANCE = 3.0
BATCH_SIZE = 16
DECODE_BATCH_SIZE = 4
UNIFORM_CONTINUATION = "uniform20k"


def read_protocol(
    root: Path,
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """读取题单与批准记录，验算内容锁及固定采样参数。

    Args:
        root: 项目根目录，包含正式题单、规则、排除清单及批准回执。

    Raises:
        OSError: 任一锁定资料缺失或不可读。
        ValueError: 内容摘要、批准、题单顺序或采样协议不符。

    Results:
        tuple: 已锁题单、应排除的已使用组及写入生成回执的协议摘要。
    """
    paths = {"benchmark": root / BENCHMARK, "rubric": root / RUBRIC}
    benchmark = json.loads(paths["benchmark"].read_text())
    lock_path, approval_path = root / LOCK, root / APPROVAL
    lock = json.loads(lock_path.read_text())
    approval = json.loads(approval_path.read_text())
    provenance = benchmark["provenance"]
    exclusion_path = root / provenance["exclusions_path"]
    hashes = {f"{name}_sha256": hash_file(path) for name, path in paths.items()}
    if (
        benchmark["schema"] != "minigen.anime_faces.benchmark.v1"
        or lock["schema"] != "minigen.anime_faces.benchmark_lock.v1"
        or approval.get("approved") is not True
        or any(approval.get(key) != sha or lock[key] != sha for key, sha in hashes.items())
        or hash_file(exclusion_path) != lock["group_exclusions_sha256"]
        or provenance["exclusions_sha256"] != lock["group_exclusions_sha256"]
        or provenance["manifest_sha256"] != lock["source_manifest_sha256"]
    ):
        raise ValueError("Locked anime benchmark and user approval do not match")
    sampling = {
        "resolution": 256,
        "weights": "ema",
        "solver": "euler",
        "steps": STEPS,
        "cfg": GUIDANCE,
        "precision": "fp32",
    }
    regular, pairs = benchmark["regular"], benchmark["pairs"]
    if (
        benchmark["seeds"] != SEEDS
        or any(benchmark["sampling"][key] != value for key, value in sampling.items())
        or [row["id"] for row in regular] != [f"R{i:03d}" for i in range(1, 81)]
        or [row["id"] for row in pairs] != [f"P{i:03d}" for i in range(1, 25)]
        or any(row["seeds"] != SEEDS for row in regular + pairs)
        or lock["counts"]
        != {
            "regular": 80,
            "pairs": 24,
            "source_groups": 104,
            "endpoint_prompts": 128,
            "seeds": 2,
            "outputs_per_model": 256,
            "outputs_two_models": 512,
        }
    ):
        raise ValueError("Anime benchmark sampling, order or frozen counts changed")
    return (
        benchmark,
        json.loads(exclusion_path.read_text()),
        hashes
        | {
            "approval_sha256": hash_file(approval_path),
            "benchmark_lock_sha256": hash_file(lock_path),
            "group_exclusions_sha256": lock["group_exclusions_sha256"],
            "source_manifest_sha256": lock["source_manifest_sha256"],
        },
    )


def validate_sources(
    benchmark: JsonDict,
    exclusions: JsonDict,
    source: CacheSource,
    identity: JsonDict,
) -> None:
    """核对题单参照真实验证清单并确认训练组与既用组均未重用。

    Args:
        benchmark: 实际批准并锁定的题单。
        exclusions: 按来源列出旧使用组、含训练固定验证子集的排除清单。
        source: 与训练缓存身份匹配的动漫数据来源。
        identity: 缓存和实际训练/验证子集验收摘要。

    Raises:
        ValueError: 已冻结的清单有变更，或参照行错误、越界、重复。
    """
    if (
        source.identity["manifest_sha256"] != benchmark["provenance"]["manifest_sha256"]
        or identity["subsets"]["validation"]["ordered_ids_sha256"]
        != exclusions["ordered_fixed_validation_ids_sha256"]
    ):
        raise ValueError("Benchmark was not prepared for the loaded anime cache")
    by_id = {row["sample_id"]: row for row in source.records["validation"]}
    groups = {row["group_id"] for row in source.records["train"]}
    groups.update(group for entry in exclusions["groups"].values() for group in entry["group_ids"])
    for case in benchmark["regular"] + benchmark["pairs"]:
        reference = case["source"]
        actual = by_id.get(reference["sample_id"])
        if (
            actual is None
            or reference["group_id"] in groups
            or any(actual[key] != reference[key] for key in reference)
        ):
            raise ValueError(f"Unexpected or already-used validation source for {case['id']}")
        groups.add(reference["group_id"])


def plan_endpoints(
    benchmark: JsonDict,
) -> list[JsonDict]:
    """将 104 道题展开为 128 个端点，干预对继承同一个原始噪声下标。

    Args:
        benchmark: 先列 80 条常规题、再列 24 对干预题的锁定记录。

    Results:
        list[JsonDict]: 每个端点的题号、文本、端点标签与噪声索引。
    """
    regular = [
        {"id": row["id"], "endpoint": None, "prompt": row["prompt"], "noise_index": i}
        for i, row in enumerate(benchmark["regular"])
    ]
    pairs = [
        {
            "id": row["id"],
            "endpoint": endpoint,
            "prompt": row[f"prompt_{endpoint}"],
            "noise_index": i,
        }
        for i, row in enumerate(benchmark["pairs"], start=len(regular))
        for endpoint in ("a", "b")
    ]
    return regular + pairs


def benchmark_noise(
    seed: int,
) -> Float[Tensor, "b c_z h_z w_z"]:  # noqa: F722
    """在 CPU 上按冻结的题目顺序一次生成本初噪种子的完整张量。

    Args:
        seed: 题目顺序共用的一个固定初噪种子。

    Results:
        Float[Tensor, "b c_z h_z w_z"]: [104,4,32,32] 的 CPU FP32 初噪。
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(NOISE_SHAPE, dtype=torch.float32, generator=generator)


def encode_benchmark_conditions(
    root: Path,
    clip_lock: Path,
    endpoints: list[JsonDict],
) -> Condition:
    """用固定 CPU CLIP 批量编码端点，并复现基线的 FP16 往返。

    Args:
        root: 冻结 CLIP 资产的项目根目录。
        clip_lock: CLIP 组件身份文件。
        endpoints: 按题库顺序展开的所有端点。

    Results:
        Condition: CPU 上的 [b,n,d] FP32 文本特征和 [b,n] 布尔掩码。
    """
    text = FrozenCLIPText.from_lock(clip_lock, root=root, device="cpu")
    prompts = [row["prompt"] for row in endpoints]
    encoded = [text.encode(prompts[i : i + BATCH_SIZE]) for i in range(0, len(prompts), BATCH_SIZE)]
    return Condition(
        torch.cat([entry.features for entry in encoded]).half().float(),
        torch.cat([entry.mask for entry in encoded]),
    )


def generate_seed(
    output: Path,
    model: FlowModel,
    vae: FrozenVAE,
    condition: Condition,
    blank: Condition,
    endpoints: list[JsonDict],
    seed: int,
) -> tuple[str, list[JsonDict]]:
    """为一个种子一次生成完整初噪，并按端点顺序完成采样和 PNG 输出。

    Args:
        output: 当前实验臂临时输出目录。
        model: FP32 设备上的已验收 EMA 速度模型。
        vae: 同训练缓存锁定的冻结 FP32 解码器。
        condition: [b,n,d] 的端点条件及其布尔掩码。
        blank: [1,n,d] 的空条件及其布尔掩码。
        endpoints: 与条件批顺序一致的 128 个端点。
        seed: 用于一次生成 104 道题初始噪声的固定种子。

    Raises:
        OSError: PNG 无法写入或回读。
        ValueError/RuntimeError/FloatingPointError: 积分、解码或设备计算失败。

    Results:
        tuple: 完整初噪的 SHA-256 及按端点排列的图像回执。

    副作用:
        写入此种子的 PNG 文件并报告进度。
    """
    noise = benchmark_noise(seed)
    batch_hash = tensor_digest(noise)
    row_hashes = [tensor_digest(noise[i : i + 1]) for i in range(len(noise))]
    noise = noise.to(condition.features.device)
    records = []
    for start in range(0, len(endpoints), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(endpoints))
        subset = endpoints[start:stop]
        selected = [row["noise_index"] for row in subset]
        text = Condition(condition.features[start:stop], condition.mask[start:stop])
        result = integrate(model, noise[selected], text, blank, STEPS, GUIDANCE, "euler")
        images = torch.cat([vae.decode(x).cpu() for x in result.samples.split(DECODE_BATCH_SIZE)])
        if images.shape != (len(subset), 3, 256, 256) or not torch.isfinite(images).all():
            raise ValueError(f"Invalid decoded anime benchmark images for seed {seed}")
        images = (images.clamp(-1, 1) + 1) / 2
        for image, row in zip(images, subset, strict=True):
            endpoint = row["endpoint"]
            suffix = f"_{endpoint}" if endpoint else ""
            filename = f"{row['id']}_{seed}{suffix}.png"
            path = output / filename
            save_image(image, path)
            records.append(
                row
                | {
                    "seed": seed,
                    "image_path": filename,
                    "image_sha256": hash_file(path),
                    "noise_sha256": row_hashes[row["noise_index"]],
                }
            )
        print(f"seed {seed}: {stop}/{len(endpoints)}", flush=True)
    return batch_hash, records


def publish_benchmark_arm(
    target: Path,
    model: FlowModel,
    vae: FrozenVAE,
    condition: Condition,
    blank: Condition,
    endpoints: list[JsonDict],
    sampling: JsonDict,
    metadata: JsonDict,
    started: float,
) -> None:
    """对单个 EMA 模型生成所有固定端点，并在结束后发布图片与生成回执。

    Args:
        target: 此实验臂的新输出目录。
        model: 待评测的 FP32 速度模型。
        vae: 同固定资产锁的冻结 FP32 解码器。
        condition: [b,n,d] 端点文本特征及 [b,n] 掩码。
        blank: [1,n,d] 空条件及 [1,n] 掩码。
        endpoints: 与条件顺序一致的所有固定端点。
        sampling: 已核验的冻结题库采样配置。
        metadata: 通过核验的检查点、题库和范围身份。
        started: 当前运行的单调计时起点。

    Raises:
        OSError: 输出目录或产物无法发布。
        ValueError/RuntimeError: 数值、设备或生成记录校验失败。

    副作用:
        在暂存目录生成 256 张 PNG 及回执，完成后将目录重命名为正式产物。
    """
    device = condition.features.device
    target.parent.mkdir(parents=True, exist_ok=True)
    records, noise_hashes = [], {}
    with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temp:
        pending = Path(temp) / target.name
        pending.mkdir()
        for seed in SEEDS:
            noise_hashes[str(seed)], generated = generate_seed(
                pending, model, vae, condition, blank, endpoints, seed
            )
            records.extend(generated)
        synchronize(device)
        receipt = metadata | {
            "schema": "minigen.anime_faces.benchmark_generation.v1",
            "status": "completed",
            "seeds": SEEDS,
            "sampling": sampling,
            "seed_noise_sha256": noise_hashes,
            "noise_shape": list(NOISE_SHAPE),
            "condition_encoding": "cpu_clip_fp16_roundtrip",
            "device": str(device),
            "evaluation_torch": str(torch.__version__),
            "sampling_batch_size": BATCH_SIZE,
            "decode_batch_size": DECODE_BATCH_SIZE,
            "tf32": False,
            "seconds": perf_counter() - started,
            "records": records,
        }
        (pending / "receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        pending.rename(target)
    print(f"{target.name}: {len(records)} images in {target}", flush=True)


@torch.inference_mode()
def generate(
    args: argparse.Namespace,
) -> None:
    """验证正式协议和 EMA 身份，完成后原子发布指定实验臂的图像回执。

    Args:
        args: 项目根路径、检查点根路径、输出根路径和需要采样的实验臂。

    Raises:
        OSError: 已有同名实验臂，或输入/输出文件无法访问。
        ValueError/RuntimeError: 资产、批准、检查点、设备或生成失败。

    副作用:
        配置本进程推理精度，在指定目录写入 256 张图片与生成回执。
    """
    started = perf_counter()
    root = args.root.resolve()
    target = root / args.output / args.arm
    if target.exists():
        raise FileExistsError(target)
    benchmark, exclusions, protocol = read_protocol(root)
    endpoints = plan_endpoints(benchmark)
    device = resolve_device("cuda", "fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, data = read_experiment(root, CONFIG)
    torch.set_num_threads(config.cpu_threads)
    source, _, blank, identity = open_anime_cache(root, data)
    validate_sources(benchmark, exclusions, source, identity)
    checkpoint_root = root / args.checkpoint_root
    if args.arm == UNIFORM_CONTINUATION:
        path = checkpoint_root / "uniform_to_20000" / "step_010000.pt"
        model, checkpoint = read_uniform_milestone(path, FINAL_STEP, config.model, identity)
    else:
        model, checkpoint, _ = read_arm(
            checkpoint_root / args.arm / "last.pt", args.arm, config.model, identity
        )
    condition = encode_benchmark_conditions(root, root / data["clip_lock"], endpoints).to(device)
    blank = blank.to(device)
    model = model.to(device)
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    publish_benchmark_arm(
        target,
        model,
        vae,
        condition,
        blank,
        endpoints,
        benchmark["sampling"],
        checkpoint | protocol,
        started,
    )


def main() -> None:
    """解析命令行并执行显式选中的一个模型臂的基准生成。

    副作用:
        读取终端参数，生成并登记盲评所需文件。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS, UNIFORM_CONTINUATION), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
