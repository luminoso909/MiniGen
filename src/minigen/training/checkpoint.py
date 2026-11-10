"""原子检查点、可移植张量和同设备随机状态恢复。"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minigen._typing import Checkpoint, JsonDict
from minigen.training.runtime import device_rng

FLOW_OBJECTIVE = "flow_velocity"
CHECKPOINT_SCHEMA = "minigen.flow-training.v2"


def run_identity(
    contract: JsonDict,
) -> str:
    """把完整恢复契约转为训练、采样与评估共用的实验标识。

    Args:
        contract: 包含源码、配方、数据和设备身份的 JSON 契约。

    Raises:
        ValueError/TypeError: 契约不是合法有限 JSON。

    Results:
        str: 规范排序后契约内容的 SHA-256。
    """
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def cpu_tree(
    value: Any,
) -> Any:
    """把异构检查点中的张量复制到 CPU，保留容器结构。

    Args:
        value: 检查点中的张量、容器或标量。

    Results:
        Any: 可在 CPU 上读取的独立检查点值。
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state(
    device: torch.device,
) -> Checkpoint:
    """保存 Python、NumPy、CPU 和当前加速设备的随机状态。

    Args:
        device: 需要恢复的实际训练设备。

    Results:
        Checkpoint: 仅含安全加载支持的基础对象和张量的 RNG 状态。
    """
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cpu": torch.get_rng_state().clone(),
        "accelerator": device_rng(device),
    }


def restore_rng(
    state: Checkpoint,
    device: torch.device,
) -> None:
    """在模型、优化器和数据流构造后恢复随机状态。

    Args:
        state: 同设备执行保存的完整随机状态。
        device: 当前已经核验兼容的设备。

    Raises:
        KeyError/RuntimeError/ValueError: 状态缺失或底层 RNG 恢复失败。

    副作用:
        修改进程全局随机数状态和当前加速设备随机数状态。
    """
    random.setstate(state["python"])
    raw = state["numpy"]
    np.random.set_state((raw[0], np.asarray(raw[1], dtype=np.uint32), *raw[2:]))
    torch.set_rng_state(state["cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"].cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"].cpu())


def save_checkpoint(
    path: Path,
    payload: Checkpoint,
) -> None:
    """先写完整临时文件再原子替换检查点，失败不损坏旧版本。

    Args:
        path: 目标检查点路径。
        payload: 模型、优化器和恢复状态等异构数据。

    Raises:
        OSError/RuntimeError: 序列化、写盘或原子替换失败。

    副作用:
        写入并同步临时文件，再替换目标。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(cpu_tree(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
) -> Checkpoint:
    """安全读取 Flow 检查点并核对目标，默认映射到 CPU。

    Args:
        path: v2 或明确历史 Flow v1 的本地检查点。

    Raises:
        ValueError: schema、训练目标或目标字段之间的约定不符。
        OSError/RuntimeError: 文件无法读取或反序列化。

    Results:
        Checkpoint: CPU 对象；旧 v1 仅在内存中补充已知 Flow 目标，并标记来源。
            objective_source: explicit 或 legacy_flow_schema；不会改写磁盘旧文件或恢复 RNG。
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema") not in {
        CHECKPOINT_SCHEMA,
        "minigen.flow-training.v1",
    }:
        raise ValueError("Unsupported Flow checkpoint schema")
    legacy = state["schema"] == "minigen.flow-training.v1"
    objective = state.get("objective", FLOW_OBJECTIVE if legacy else None)
    if (
        objective != FLOW_OBJECTIVE
        or state.get("config", {}).get("objective", objective) != objective
        or state.get("contract", {}).get("recipe", {}).get("objective", objective) != objective
    ):
        raise ValueError("Checkpoint requires objective=flow_velocity")
    state["objective_source"] = "legacy_flow_schema" if "objective" not in state else "explicit"
    state["objective"] = objective
    return state
