"""原子检查点、可移植张量和同设备随机状态恢复。"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minigen._typing import Checkpoint
from minigen.training.runtime import device_rng


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
    """只用安全权重模式读取本项目检查点，默认映射到 CPU。

    Args:
        path: 当前 schema 的本地检查点。

    Raises:
        ValueError: 检查点不是本项目 Flow 训练格式。
        OSError/RuntimeError: 文件无法读取或反序列化。

    Results:
        Checkpoint: 含模型和恢复状态的 CPU 对象；加载本身不恢复 RNG。
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema") != "minigen.flow-training.v1":
        raise ValueError("Unsupported Flow checkpoint schema")
    return state
