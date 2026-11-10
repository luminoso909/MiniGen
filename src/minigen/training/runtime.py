"""显式选择单设备运行时，准备 CUDA AMP 并保留本机 FP32 路径。"""

import platform
import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from functools import cache

import torch
from jaxtyping import UInt8
from torch import Tensor


def resolve_device(
    requested: str,
    precision: str,
) -> torch.device:
    """在当前进程验证实际设备和训练精度，不静默回退。

    Args:
        requested: auto、cpu、mps 或 cuda。
        precision: fp32、fp16 或 bf16；混合精度当前仅开放 CUDA。

    Raises:
        ValueError: 设备不可用、精度不受支持或 CUDA 不支持 BF16。

    Results:
        torch.device: 明确的一张计算设备。
    """
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unsupported device")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process")
    if precision not in {"fp32", "fp16", "bf16"} or (requested != "cuda" and precision != "fp32"):
        raise ValueError("CPU/MPS training requires fp32; CUDA supports explicit AMP")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 is unsupported on this device")
    return torch.device(requested)


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager:
    """构造显式 AMP 上下文，FP32 不改变底层计算精度。

    Args:
        device: 已通过设备验证的计算设备。
        precision: 训练配置中的浮点精度。

    Results:
        AbstractContextManager: CUDA 自动混合精度或空上下文。
    """
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type=device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    )


def synchronize(
    device: torch.device,
) -> None:
    """计时前等待所选加速设备完成已提交的工作。

    Args:
        device: 当前训练或采样设备。

    副作用:
        等待当前设备的异步操作完成。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_rng(
    device: torch.device,
) -> UInt8[Tensor, " n"] | None:
    """取得所选加速设备的随机状态，CPU 状态另行保存。

    Args:
        device: 当前实际计算设备。

    Results:
        Tensor | None: GPU RNG 字节状态；CPU 返回 None。
    """
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).cpu()
    if device.type == "mps":
        return torch.mps.get_rng_state().cpu()
    return None


@cache
def hardware_description(
    device: torch.device,
) -> str:
    """记录实际处理器标识，无法读取型号时保留操作系统与架构。

    Args:
        device: 本次实际运行设备。

    Results:
        str: CUDA 型号、Mac 芯片标识或可获得的 CPU 型号。
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    fallback = f"{platform.system()} {platform.machine()} {platform.processor()}".strip()
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            return f"{fallback}; {chip}" if chip else fallback
        except (OSError, subprocess.SubprocessError):
            return fallback
    return fallback
