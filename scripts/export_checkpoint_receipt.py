"""从真实中间检查点导出资源回执；缺失硬件不猜测，FLOPs 仅为显式近似。"""

import argparse
import json
import math
from pathlib import Path

import torch

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.training.checkpoint import load_checkpoint, run_identity
from minigen.training.config import build_model


def export_receipt(
    checkpoint: Path,
    output: Path,
    flops_inspection: Path | None = None,
) -> JsonDict:
    """读取已完成步的权威历史，将资源绑定到该检查点而非训练末次结果。

    Args:
        checkpoint: 现有 Flow 检查点，始终在 CPU 读取。
        output: 新 JSON 路径，不覆盖旧证据。
        flops_inspection: 可选 inspect_flow_models.py 报告，必须匹配模型与输入规格。

    Raises:
        FileExistsError: 输出已经存在。
        ValueError: 检查点历史、模型或 FLOPs 身份不匹配。
        OSError/KeyError/RuntimeError: 读取、模型规格检查或写入失败。

    Results:
        JsonDict: 可交给 bench report --training 的该步回执；未知硬件/显存保持 null。

    副作用:
        在 CPU 加载检查点，在 meta 设备构造形状模型，写小型 JSON；不训练或推理。
    """
    if output.exists():
        raise FileExistsError(output)
    state = load_checkpoint(checkpoint)
    config, contract, history = state["config"], state["contract"], state["history"]
    step = state["step"]
    if (
        type(step) is not int
        or step < 0
        or [r["step"] for r in history] != list(range(1, step + 1))
    ):
        raise ValueError("Checkpoint history must contain every completed update exactly once")
    elapsed = float(state["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Checkpoint elapsed time must be finite and nonnegative")
    if any(type(r["samples"]) is not int or r["samples"] < 1 for r in history):
        raise ValueError("History must record actual positive sample counts")
    with torch.device("meta"):
        model = build_model(config["model"])
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if expected != {key: tuple(value.shape) for key, value in state["model"].items()}:
        raise ValueError("Checkpoint model tensors disagree with its specification")
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    samples = sum(row["samples"] for row in history)
    update_seconds = sum(row["step_seconds"] for row in history)
    if not math.isfinite(update_seconds) or update_seconds < 0:
        raise ValueError("Recorded update time is invalid")
    estimate, method, inspection_hash = None, None, None
    if flops_inspection is not None:
        inspection = json.loads(flops_inspection.read_text())
        matches = [row for row in inspection["models"] if row["specification"] == config["model"]]
        if len(matches) != 1:
            raise ValueError("FLOPs inspection requires one exact model specification match")
        row = matches[0]
        batch = row["latent_shape"][0]
        latent_shape = [
            batch,
            model.config.in_channels,
            model.config.latent_size,
            model.config.latent_size,
        ]
        if (
            type(batch) is not int
            or batch < 1
            or row["latent_shape"] != latent_shape
            or row["text_shape"] != [batch, 77, model.config.text_dim]
            or row["trainable_parameters"] != parameters
            or not math.isfinite(row["dominant_forward_flops"])
            or row["dominant_forward_flops"] <= 0
        ):
            raise ValueError("FLOPs inspection shapes, parameters or forward count differ")
        estimate = row["dominant_forward_flops"] / batch * 3 * samples
        method = (
            "3 * measured-shape dominant forward matrix FLOPs per image * actual history samples; "
            "forward+backward approximation, not hardware measurement; excludes nonmatrix ops, "
            "optimizer, EMA, validation, sampling and frozen encoders"
        )
        inspection_hash = hash_file(flops_inspection)
    result = {
        "schema": "minigen.training-result.v1",
        "receipt_source": "checkpoint_export",
        "run_id": run_identity(contract),
        "objective": state["objective"],
        "architecture": config["model"].get("architecture", "b0"),
        "model_config": config["model"],
        "source_sha256": contract.get("source_sha256"),
        "data_identity": state.get("data_identity", contract.get("data")),
        "status": "checkpoint_snapshot",
        "step": step,
        "elapsed_seconds": elapsed,
        "device": contract.get("device"),
        "precision": config.get("precision"),
        "hardware": state.get("hardware"),
        "hardware_reason": None
        if state.get("hardware")
        else "not_recorded_in_historical_checkpoint",
        "trainable_parameters": parameters,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hash_file(checkpoint),
        "metrics": str(checkpoint.resolve().with_name("metrics.jsonl")),
        "performance": {
            "optimizer_steps": step,
            "samples_seen": samples,
            "optimizer_step_seconds": update_seconds,
            "samples_per_optimizer_second": samples / update_seconds
            if update_seconds > 0
            else None,
            "scope": "recorded optimization updates only; not a warmup-excluded performance pilot",
        },
        "memory": {"cuda_peak_allocated": None, "cuda_peak_reserved": None},
        "memory_reason": "not_stored_per_checkpoint",
        "training_flops_estimate": estimate,
        "training_flops_method": method,
        "flops_inspection_sha256": inspection_hash,
        "cost_actual_cny": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return result


def main() -> None:
    """解析独立检查点与可选 FLOPs 依据，打印实际导出结果。

    Raises:
        SystemExit: 参数不合法。
        ValueError/OSError/RuntimeError: 身份或资源回执不能导出。

    副作用:
        写一份新的 JSON 回执并向终端输出；不执行训练。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flops-inspection", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_receipt(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
