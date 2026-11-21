"""只读 B2 50M patch1 的 10k/20k EMA，复用 30k 原图补齐同噪对照。"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from evaluate_anime_time_ablation import tensor_digest
from run_anime_b1 import ROOT, open_anime_cache
from torchvision.utils import save_image
from train_anime_six import read_arm_config

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.commands import selected_indices
from minigen.contracts import Condition
from minigen.samplers.flow import integrate
from minigen.training.checkpoint import CHECKPOINT_SCHEMA, run_identity
from minigen.training.config import build_model
from minigen.training.runtime import resolve_device, synchronize

if TYPE_CHECKING:
    from torch import nn

ARM = "b2_50m_p1"
SCOPE = Path("configs/evaluation/anime_b2_p1_milestones.json")
OUTPUT = Path("artifacts/exports/anime-faces-b1/b2_p1_milestones_10k_20k_30k")


def load_checkpoint(
    root: Path,
    scope: JsonDict,
    step: int,
    identity: JsonDict,
) -> nn.Module:
    """从同一完成训练的里程碑中加载准确的只读 FP32 EMA。

    Args:
        root: 远端项目根目录。
        scope: 本轮配置、run_id、模型及三个编号快照的身份。
        step: 10,000 或 20,000 的真实优化器累计更新数。
        identity: 实际读取缓存及验证子集身份。

    Raises:
        OSError/ValueError/RuntimeError: 权重、步数、模型或数据契约不一致。

    Results:
        nn.Module: CPU 上已冻结梯度、仅供推理的 EMA 速度网络。
    """
    config, _ = read_arm_config(root, ARM)
    path = root / scope["checkpoints"][str(step)]["checkpoint"]
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        state["schema"] != CHECKPOINT_SCHEMA
        or state["objective"] != "flow_velocity"
        or state["step"] != step
        or state["scheduler"]["step"] != step
        or run_identity(state["contract"]) != scope["run_id"]
        or state["contract"]["source_sha256"] != scope["source_sha256"]
        or state["config"]["model"] != config.model
        or state["data_identity"] != identity
    ):
        raise ValueError(f"Milestone differs from the accepted B2 run: {step}")
    with torch.device("meta"):
        model = build_model(config.model)
    model.load_state_dict(state["ema"], strict=True, assign=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def generate(
    root: Path,
    output: Path,
) -> None:
    """生成两组新图，并将经过哈希核对的 30k 原图复制到第三列。

    Args:
        root: 原权重、缓存及冻结代码所在项目根目录。
        output: 本次独立输出根目录。

    Raises:
        OSError/ValueError/RuntimeError: 资产、配对条件、噪声、设备或输出不符。

    副作用:
        设置 CPU 线程及 CUDA 精度，写 32 张新图、16 张复用图和每步回执；不更新模型。
    """
    scope = json.loads((root / SCOPE).read_text())
    ref_path = root / scope["reference_receipt"]
    reference = json.loads(ref_path.read_text())
    if hash_file(ref_path) != scope["reference_receipt_sha256"]:
        raise ValueError("Original 30k receipt changed")
    source = Path(inspect.getfile(build_model)).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(source.rglob("*.py")):
        digest.update(str(file.relative_to(source)).encode() + b"\0" + file.read_bytes())
    if digest.hexdigest() != scope["source_sha256"]:
        raise ValueError("Use the frozen six-model training package")
    for spec in scope["checkpoints"].values():
        if hash_file(root / spec["checkpoint"]) != spec["sha256"]:
            raise ValueError(f"Checkpoint differs from accepted snapshot: {spec['checkpoint']}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device("cuda", "fp32")
    _, data = read_arm_config(root, ARM)
    source_data, cache, blank, identity = open_anime_cache(root, data)
    indices = selected_indices(source_data, data, "validation")[:16]
    entries = [cache.split("validation")[index] for index in indices]
    condition = Condition(
        torch.stack([row.text for row in entries]).float(),
        torch.stack([row.mask for row in entries]),
    ).to(device)
    blank = blank.to(device)
    noise = torch.randn((16, 4, 32, 32), generator=torch.Generator().manual_seed(5001))
    labels = [
        {
            "index": i,
            "sample_id": source_data.records["validation"][index]["sample_id"],
            "caption": source_data.records["validation"][index]["caption"],
            "noise_sha256": tensor_digest(noise[i : i + 1]),
        }
        for i, index in enumerate(indices)
    ]
    original_identity = reference["identity"]
    if (
        original_identity["checkpoint_sha256"] != scope["checkpoints"]["30000"]["sha256"]
        or original_identity["run_id"] != scope["run_id"]
        or original_identity["data_identity"] != identity
        or original_identity["guidance"] != 3
        or original_identity["method"] != "euler"
        or original_identity["steps"] != 40
        or original_identity["seed"] != 5001
        or original_identity["sampling_batch_size"] != 4
        or original_identity["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("Original 30k generation uses another inference recipe")
    for label, row in zip(labels, reference["records"], strict=True):
        if any(row[key] != value for key, value in label.items()):
            raise ValueError("Development tags or initial noise differ from original 30k")
    vae = FrozenVAE.load(root, root / data["vae_lock"], device=str(device))
    for step in (10000, 20000, 30000):
        folder = output / "generation" / f"step_{step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        if step == 30000:
            for label, original in zip(labels, reference["records"], strict=True):
                old = ref_path.parent.parent / original["image_path"]
                if hash_file(old) != original["image_sha256"]:
                    raise ValueError("Original 30k PNG changed")
                path = folder / f"sample_{label['index']:02d}.png"
                shutil.copyfile(old, path)
                rows.append(label | {"image_path": path.name, "image_sha256": hash_file(path)})
        else:
            model = load_checkpoint(root, scope, step, identity).to(device)
            for first in range(0, 16, 4):
                text = Condition(
                    condition.features[first : first + 4], condition.mask[first : first + 4]
                )
                result = integrate(
                    model,
                    noise[first : first + 4].to(device),
                    text,
                    blank,
                    steps=40,
                    guidance=3.0,
                    method="euler",
                )
                images = (vae.decode(result.samples).clamp(-1, 1).cpu() + 1) / 2
                if not torch.isfinite(images).all():
                    raise ValueError(f"Nonfinite milestone images: {step}")
                for offset, image in enumerate(images):
                    label = labels[first + offset]
                    path = folder / f"sample_{label['index']:02d}.png"
                    save_image(image, path)
                    rows.append(label | {"image_path": path.name, "image_sha256": hash_file(path)})
            del model
            torch.cuda.empty_cache()
        synchronize(device)
        spec = scope["checkpoints"][str(step)]
        receipt = {
            "schema": "minigen.anime_b2_milestones.generation.v1",
            "status": "completed",
            "arm": ARM,
            "checkpoint_step": step,
            "checkpoint_sha256": spec["sha256"],
            "run_id": scope["run_id"],
            "model_config": scope["model_config"],
            "source_sha256": scope["source_sha256"],
            "data_identity": identity,
            "seed": 5001,
            "guidance": 3,
            "method": "euler",
            "steps": 40,
            "precision": "fp32",
            "sampling_batch_size": 4,
            "weights": "ema",
            "torch_version": str(torch.__version__),
            "sampler_sha256": hash_file(Path(__file__)),
            "scope_sha256": hash_file(root / SCOPE),
            "metadata": {
                key: spec[key]
                for key in (
                    "validation_flow_mse",
                    "equivalent_epochs",
                    "samples_seen",
                )
            },
            "reused_original_30k": step == 30000,
            "training_updates": 0,
            "records": rows,
        }
        (folder / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"{step}: {len(rows)} images; original reuse={step == 30000}", flush=True)


def main() -> None:
    """提供本轮独立只采样入口。

    副作用:
        读取命令行并执行 GPU 采样及原图复用。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    generate(root, root / args.output)


if __name__ == "__main__":
    main()
