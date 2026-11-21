"""从 B1/B2 50M patch1 的 25k 完整状态恢复，添加独立 REPA 对齐头。"""

# ruff: noqa: F722

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from accept_anime_six import check_checkpoint, same_stream_and_rng
from cache_anime_repa_teacher import FEATURE_SPEC, PROBE_RECEIPT, verified_probe
from jaxtyping import Float
from run_anime_b1 import open_anime_cache
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Subset, default_collate
from train_anime_b2_p1_mid60 import same_state, write_json
from train_anime_six import ROOT, read_arm_config
from train_anime_six import RUNS as SIX_RUNS

import minigen
from minigen._typing import Checkpoint, JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.commands import selected_indices
from minigen.contracts import Condition
from minigen.models.hybrid_dit import HybridDiT
from minigen.models.single_dit import SingleStreamDiT
from minigen.objectives.flow import flow_interpolate, flow_loss
from minigen.training.checkpoint import run_identity, save_checkpoint
from minigen.training.config import build_model
from minigen.training.engine import _contract, train_flow
from minigen.training.runtime import autocast_context, hardware_description, resolve_device

ARMS = {"b1_50m_p1": 9, "b2_50m_p1": 5}
START = 25000
FINAL = 30000
WEIGHT = 0.1
RUNS = SIX_RUNS.parent
TEACHER = Path("artifacts/cache/anime_faces_256/p1_repa_clip_v1")
ORIGINAL_SOURCE = Path("artifacts/exports/anime-faces-b1/six_30k_code/src/minigen")
BASELINES = Path("configs/benchmark/anime_faces_v1_four_50m_30k.json")
ENGINE = Path("training/engine.py")


class Parent(NamedTuple):
    """已验收的 Uniform 恢复目录、契约、状态及两个快照摘要。"""

    directory: Path
    run: JsonDict
    state: Checkpoint
    sha256: str
    uniform_sha256: str


class PatchAlignment:
    """将带噪 B1/B2 图像网格池化并对齐对应干净图像的 CLIP 特征。"""

    def __init__(
        self,
        root: Path,
        sample_ids: list[str],
        manifest_sha256: str,
        projector: nn.Module,
    ) -> None:
        """核对教师数组的来源、实际摘要和训练样本映射。

        Args:
            root: 含真实缓存和教师资产锁的项目根目录。
            sample_ids: 冻结训练表示表的有序样本 ID。
            manifest_sha256: 冻结训练划分清单的内容摘要。
            projector: 独立 AdamW 更新且仅训练时使用的投影头。

        Raises:
            ValueError/OSError: 教师数组、映射、探针或来源身份不符。
        """
        directory = root / TEACHER
        path = directory / "train.npy"
        receipt = json.loads((directory / "index.json").read_text())
        teacher = np.load(path, mmap_mode="r", allow_pickle=False)
        assets = json.loads((root / "configs/models/evaluation.json").read_text())
        clip = assets["models"]["clip"]["files"]["pytorch_model.bin"]
        expected_ids = hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest()
        expected_shape = (len(sample_ids), 256, 1024)
        if (
            receipt["schema"] != "minigen.anime_repa_clip.v1"
            or receipt["feature_spec"] != FEATURE_SPEC
            or receipt["train_ids_sha256"] != expected_ids
            or receipt["source_manifest_sha256"] != manifest_sha256
            or receipt["weights_sha256"] != clip["sha256"]
            or receipt["probe_sha256"] != verified_probe(root, PROBE_RECEIPT)
            or receipt["array_sha256"] != hash_file(path)
            or teacher.shape != expected_shape
            or tuple(receipt["shape"]) != expected_shape
            or teacher.dtype != np.float16
        ):
            raise ValueError("REPA teacher does not match the frozen training ledger")
        self.teacher = teacher
        self.rows = {sample_id: row for row, sample_id in enumerate(sample_ids)}
        self.identity = receipt
        self.projector = projector
        self.tokens: Float[Tensor, "b n d"] | None = None

    def capture(
        self,
        block: nn.Module,
        inputs: tuple[Tensor, ...],
        tokens: Float[Tensor, "b n d"],
    ) -> None:
        """在 B1 第 9 层只保存末尾的 1,024 个图像 token。

        Args:
            block: 正在运行的第 9 个单流块。
            inputs: 单流块的输入，PyTorch 钩子约定参数。
            tokens: 沿序列轴按文本、图像顺序连接的块输出。

        副作用:
            暂存仍带梯度的图像特征，随后由微批损失读取并清空。
        """
        self.tokens = tokens[:, -1024:]

    def __call__(
        self,
        model: nn.Module,
        noisy: Float[Tensor, "b c_z h_z w_z"],
        times: Float[Tensor, " b"],
        condition: Condition,
        target: Float[Tensor, "b c_z h_z w_z"],
        sample_ids: Sequence[str],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        """执行原 Flow 前向，另对中间图像 token 计算加权余弦距离。

        Args:
            model: B1 或 B2 patch1 主干。
            noisy: 同一训练图像的带噪潜变量。
            times: Uniform Flow 时间。
            condition: 已按原配方应用 CFG 丢弃的文本条件。
            target: 原配方速度目标。
            sample_ids: 微批逐图 ID，绑定同一图的无噪声教师表征。

        Raises:
            RuntimeError: 主干未提供对应中间图像 token。
            KeyError: 样本没有绑定教师特征。

        Results:
            tuple: 原 Flow MSE 和乘以 0.1 的 REPA 余弦损失。

        副作用:
            B1 暂时安装并移除第 9 层钩子，清空暂存图像特征。
        """
        if isinstance(model, HybridDiT):
            prediction, tokens = model.forward_with_image_tokens(noisy, times, condition)
        elif isinstance(model, SingleStreamDiT):
            # 钩子仅在此微批前向存活，避免复制到 EMA 或影响其验证前向。
            with model.blocks[ARMS["b1_50m_p1"] - 1].register_forward_hook(self.capture):
                prediction = model(noisy, times, condition)
            tokens, self.tokens = self.tokens, None
        else:
            raise RuntimeError("REPA requires B1 or B2 patch1")
        if tokens is None or tokens.shape[1] != 1024:
            raise RuntimeError("REPA expected the original 32 by 32 image-token grid")
        grid = tokens.float().transpose(1, 2).reshape(len(noisy), -1, 32, 32)
        pooled = F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2)
        projected = self.projector(pooled).float()
        rows = [self.rows[sample_id] for sample_id in sample_ids]
        features = torch.from_numpy(self.teacher[rows]).to(projected.device, torch.float32)
        distance = 1 - F.cosine_similarity(projected, features, dim=-1).mean()
        return flow_loss(prediction, target), WEIGHT * distance


def read_parent(
    root: Path,
    arm: str,
) -> Parent:
    """读取已验收的 Uniform 父状态及原 30k 对照身份。

    Args:
        root: 六组 30k 已验收产物所在的根目录。
        arm: B1/B2 patch1 启动组名。

    Raises:
        ValueError/AssertionError: 父训练或检查点与原验收不符。
        OSError/RuntimeError: 原始文件不能读取。

    Results:
        Parent: 已核验的 25k 恢复状态和 30k 原对照摘要。
    """
    directory = root / SIX_RUNS / arm
    run = json.loads((directory / "run.json").read_text())
    accepted = json.loads((root / SIX_RUNS / "acceptance.json").read_text())["arms"][arm]
    uniform = json.loads((root / BASELINES).read_text())["candidates"][arm]
    metadata = {item["step"]: item for item in accepted["snapshots"]}
    state, source = check_checkpoint(directory / f"step_{START:06d}.pt", START, run)
    uniform_path = directory / f"step_{FINAL:06d}.pt"
    uniform_sha = hash_file(uniform_path)
    if (
        accepted["result"]["run_id"] != run["run_id"]
        or run_identity(run["contract"]) != run["run_id"]
        or uniform["run_id"] != run["run_id"]
        or uniform_path != root / uniform["checkpoint"]
        or run["contract"]["recipe"].get("time_mid_probability") is not None
        or state["config"]["precision"] != "bf16"
        or state["config"]["model"]["patch_size"] != 1
        or source["sha256"] != metadata[START]["sha256"]
        or uniform_sha != uniform["checkpoint_sha256"]
    ):
        raise ValueError("Parent differs from the accepted Uniform six-run checkpoints")
    return Parent(directory, run, state, source["sha256"], uniform_sha)


def verify_frozen_source(
    root: Path,
    parent_sha256: str,
) -> None:
    """验证训练包仅替换了用于独立辅助头的引擎文件。

    Args:
        root: 冻结原始源码目录所属项目根。
        parent_sha256: 原检查点的完整包源码摘要。

    Raises:
        ValueError/OSError: 任何其他源码变化，或原源码与检查点身份不一致。
    """
    source = Path(minigen.__file__).resolve().parent
    original = root / ORIGINAL_SOURCE
    digest = hashlib.sha256()
    originals = sorted(path.relative_to(original) for path in original.rglob("*.py"))
    if originals != sorted(path.relative_to(source) for path in source.rglob("*.py")):
        raise ValueError("REPA package file list differs from the original")
    for path in originals:
        content = (original / path).read_bytes()
        digest.update(str(path).encode() + b"\0" + content)
        if path != ENGINE and (source / path).read_bytes() != content:
            raise ValueError(f"REPA altered original model/data code: {path}")
    if digest.hexdigest() != parent_sha256:
        raise ValueError("The original source snapshot does not match the parent checkpoint")


def run_branch(
    root: Path,
    arm: str,
    operation: str,
) -> JsonDict:
    """验证真实 25k 数据和设备，训练至原余弦日程的 30k 终点。

    Args:
        root: 冻结源码、配置和训练资产的根目录。
        arm: B1/B2 patch1 运行分支。
        operation: check 做真实 CUDA 前向或 train 继续优化更新。

    Raises:
        ValueError/RuntimeError/OSError: 恢复环境、缓存、前向或训练状态不符合原配方。

    Results:
        JsonDict: 预检或训练回执。

    副作用:
        设置线程，分配显存，写预检或训练产物。train 更新模型、EMA 及两套优化器状态。
    """
    base, data = read_arm_config(root, arm)
    torch.set_num_threads(base.cpu_threads)
    parent = read_parent(root, arm)
    original = parent.run["contract"]
    verify_frozen_source(root, original["source_sha256"])
    device = resolve_device(base.device, base.precision)
    source, cache, blank, identity = open_anime_cache(root, data)
    train = Subset(cache.split("train"), selected_indices(source, data, "train"))
    validation = Subset(cache.split("validation"), selected_indices(source, data, "validation"))
    model = build_model(base.model)
    expected = _contract(base, identity, model, device, original["initialization"])
    if (
        expected | {"source_sha256": original["source_sha256"]} != original
        or hardware_description(device) != parent.state["hardware"]
    ):
        raise ValueError("Original recipe, data, architecture, Torch or GPU differs")
    hidden = base.model["hidden_size"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(base.seed)
        projector = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1024))
    alignment = PatchAlignment(
        root,
        [row["sample_id"] for row in source.records["train"]],
        source.identity["manifest_sha256"],
        projector,
    )
    directory = root / RUNS / f"{arm}_repa_25k_30k"
    config = replace(base, output_dir=str(directory))
    if operation == "check":
        batch = default_collate([train[index] for index in range(config.batch_size)])
        latent = batch.latent.float().to(device)
        generator = torch.Generator().manual_seed(config.seed)
        times = torch.rand(len(latent), generator=generator).to(device)
        noise = torch.randn(latent.shape, generator=generator).to(device)
        noisy, target = flow_interpolate(latent, noise, times)
        condition = Condition(batch.text.float(), batch.mask).to(device)
        model.load_state_dict(parent.state["model"], strict=True)
        model.to(device)
        projector.to(device)
        with autocast_context(device, config.precision):
            flow, aux = alignment(model, noisy, times, condition, target, batch.sample_id)
        (flow + aux).backward()
        if not torch.isfinite(flow + aux) or any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in projector.parameters()
        ):
            raise ValueError("REPA CUDA preflight failed forward/gradient checks")
        report = {
            "status": "passed",
            "arm": arm,
            "step": START,
            "optimizer_updates": 0,
            "parent_sha256": parent.sha256,
            "uniform_sha256": parent.uniform_sha256,
            "torch": str(torch.__version__),
            "hardware": hardware_description(device),
            "precision": base.precision,
            "flow_loss": float(flow.detach()),
            "aux_loss": float(aux.detach()),
            "teacher_sha256": alignment.identity["array_sha256"],
        }
        write_json(directory / "preflight.json", report)
        return report

    initialization = {
        "kind": "repa_fork",
        "parent_step": START,
        "parent_run_id": parent.run["run_id"],
        "parent_checkpoint_sha256": parent.sha256,
        "parent_source_sha256": original["source_sha256"],
        "fork_runner_sha256": hash_file(Path(__file__)),
        "repa": {
            "weight": WEIGHT,
            "layer": ARMS[arm],
            "pool": "avg_2x2_32_to_16",
            "teacher": alignment.identity,
            "projector_optimizer": "independent_adamw",
            "projector_seed": config.seed,
        },
        "restored_state": ["model", "ema", "optimizer", "scaler", "stream", "rng", "scheduler"],
    }
    contract = _contract(config, identity, model, device, initialization, projector)
    resume = directory / "last.pt"
    if not resume.exists():
        resume = directory / f"fork_{START:06d}.pt"
        auxiliary_optimizer = torch.optim.AdamW(
            projector.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        fork = parent.state | {
            "config": config.to_dict(),
            "contract": contract,
            "stream": parent.state["stream"] | {"data_id": run_identity(contract)},
            "auxiliary": {
                "model": projector.state_dict(),
                "optimizer": auxiliary_optimizer.state_dict(),
            },
        }
        save_checkpoint(resume, fork)
        del fork
    del parent
    result = train_flow(
        config,
        train,
        blank,
        identity,
        model=model,
        resume=resume,
        validation=validation,
        initialization=initialization,
        microbatch_loss=alignment,
        auxiliary_module=projector,
    )
    if result["status"] != "completed" or result["step"] != FINAL:
        raise RuntimeError(f"{arm} REPA stopped at {result['step']}: {result['status']}")
    return result


def accept(
    root: Path,
    arm: str,
) -> JsonDict:
    """CPU 回读 REPA 的 25k 和 30k，各自与 Uniform 核对配对条件。

    Args:
        root: 已发布训练检查点和验收回执的根目录。
        arm: B1/B2 patch1 训练分支。

    Raises:
        ValueError/AssertionError: 父状态、配对条件、实际更新数或回执身份不符。
        OSError/RuntimeError: 快照和回执读取失败。

    Results:
        JsonDict: 可供冻结评测入口读取的正式训练验收。

    副作用:
        设置 CPU 线程并写入分支 acceptance.json。
    """
    torch.set_num_threads(2)
    parent = read_parent(root, arm)
    directory = root / RUNS / f"{arm}_repa_25k_30k"
    run = json.loads((directory / "run.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    restored, _ = check_checkpoint(directory / f"step_{START:06d}.pt", START, run)
    for key in ("model", "ema", "optimizer", "scaler", "scheduler"):
        if not same_state(parent.state[key], restored[key]):
            raise ValueError(f"Restored REPA 25k {key} differs from Uniform")
    same_stream_and_rng(parent.state, restored, f"{arm} REPA 25k restore")
    if restored["auxiliary"]["optimizer"]["state"]:
        raise ValueError("New REPA projector optimizer already has updates at 25k")
    del restored
    path = directory / f"step_{FINAL:06d}.pt"
    final, metadata = check_checkpoint(path, FINAL, run)
    baseline, baseline_metadata = check_checkpoint(
        parent.directory / f"step_{FINAL:06d}.pt", FINAL, parent.run
    )
    same_stream_and_rng(baseline, final, f"{arm} Uniform vs REPA 30k")
    recipe = final["contract"]["recipe"]
    initialization = final["contract"]["initialization"]
    tail = final["history"][START:]
    uniform_tail = baseline["history"][START:]
    if (
        result["status"] != "completed"
        or result["step"] != FINAL
        or result["attempt_id"] != run["attempt_id"]
        or result["run_id"] != run["run_id"]
        or run["run_id"] != run_identity(run["contract"])
        or initialization["parent_checkpoint_sha256"] != parent.sha256
        or initialization["parent_step"] != START
        or initialization["fork_runner_sha256"] != hash_file(Path(__file__))
        or recipe != parent.run["contract"]["recipe"]
        or final["scheduler"] != baseline["scheduler"]
        or final["history"][:START] != parent.state["history"]
        or baseline["history"][:START] != parent.state["history"]
        or not all("aux_loss" in row and "flow_loss" in row for row in tail)
        or {float(item["step"]) for item in final["auxiliary"]["optimizer"]["state"].values()}
        != {FINAL - START}
        or any(
            any(a[key] != b[key] for key in ("step", "samples", "lr"))
            for a, b in zip(tail, uniform_tail, strict=True)
        )
    ):
        raise ValueError("REPA does not match the frozen Uniform comparison protocol")
    report = {
        "schema": "minigen.anime_p1_repa.acceptance.v1",
        "status": "verified",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "arm": arm,
        "parent_step": START,
        "step": FINAL,
        "training_updates": FINAL - START,
        "parent_sha256": parent.sha256,
        "parent_run_id": parent.run["run_id"],
        "baseline_sha256": baseline_metadata["sha256"],
        "checkpoint": str(path.relative_to(root)),
        "checkpoint_sha256": metadata["sha256"],
        "run_id": run["run_id"],
        "source_sha256": final["contract"]["source_sha256"],
        "data_identity": final["data_identity"],
        "paired_stream_and_rng": True,
        "parent_online_ema_adamw_scheduler_verified": True,
        "matched_learning_rates_and_batch_lengths": True,
        "shared_history_prefix_steps": START,
        "projector_optimizer_updates": FINAL - START,
        "branch_samples": sum(row["samples"] for row in tail),
        "initial_learning_rate": tail[0]["lr"],
        "final_learning_rate": tail[-1]["lr"],
        "branch_seconds": final["elapsed_seconds"] - parent.state["elapsed_seconds"],
        "uniform_validation_flow_mse": baseline_metadata["validation_flow_mse"],
        "repa_validation_flow_mse": metadata["validation_flow_mse"],
        "aux_loss_first_100": sum(row["aux_loss"] for row in tail[:100]) / 100,
        "aux_loss_last_100": sum(row["aux_loss"] for row in tail[-100:]) / 100,
        "repa": initialization["repa"],
    }
    write_json(directory / "acceptance.json", report)
    return report


def main() -> None:
    """执行 REPA 预检、训练或最终配对验收。

    副作用:
        读取命令行参数，执行指定动作并输出回执。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "train", "accept"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "accept":
        report = accept(root, args.arm)
    else:
        report = run_branch(root, args.arm, args.operation)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
