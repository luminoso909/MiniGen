"""真实环境与小张量诊断，不下载资源、不启动训练。"""

import importlib
import os
import platform
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

from minigen._typing import JsonDict

PACKAGES = {
    "minigen": "minigen",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}


def tensor_check(torch: ModuleType, device: str) -> JsonDict:
    """在目标设备真实执行矩阵前向、反向与同步，检查数值有限性。

    1 输入：
        torch：由诊断入口动态导入的 Torch 模块，便于先报告导入失败。
        device：cpu、mps 或 cuda；不下载数据或权重。

    2 输出：
        JSON 结果，包含 passed 状态、Python loss 值和输入形状 [3,4]。

    3 报错（Raises）：
        RuntimeError：梯度缺失、前向或梯度非有限时主动抛出；设备不可用等异常透传。
    """
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    x.requires_grad_(True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all().item():
        raise RuntimeError("梯度缺失或包含非有限值")
    if not torch.isfinite(loss).item():
        raise RuntimeError("前向结果包含非有限值")
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    return {"status": "passed", "loss": loss.detach().cpu().item(), "shape": list(x.shape)}


def diagnose(device: str = "auto") -> JsonDict:
    """检查实际包导入、设备可用性与小张量自动求导。

    1 输入：
        device：auto 检查可用设备，或指定 cpu/mps/cuda；CPU 始终参与可用路径。

    2 输出：
        JSON 报告：Python、各包版本、各设备状态及总体 ok；普通检查异常转换为失败记录。

    3 报错（Raises）：
        ValueError：device 不属于支持列表；包导入和设备检查的常见异常被捕获并写入报告。
    """
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"不支持的设备: {device}")
    # 尊重调用者的缓存设置；未指定时只写项目本地 Matplotlib 缓存。
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
    report = {"python": platform.python_version(), "packages": {}, "devices": {}, "ok": True}
    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            report["packages"][distribution] = {
                "version": version(distribution),
                "status": "passed",
            }
        except Exception as exc:
            report["packages"][distribution] = {"status": "failed", "error": str(exc)}
            report["ok"] = False

    if report["packages"]["torch"]["status"] != "passed":
        report["devices"]["cpu"] = {"status": "not_tested", "reason": "torch import failed"}
        return report

    import torch

    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    for name, enabled in available.items():
        if not enabled:
            report["devices"][name] = {"status": "unavailable"}
            if device == name:
                report["ok"] = False
        elif device != "auto" and name not in {"cpu", device}:
            report["devices"][name] = {"status": "not_tested", "reason": "not requested"}
        else:
            try:
                report["devices"][name] = tensor_check(torch, name)
            except Exception as exc:
                report["devices"][name] = {"status": "failed", "error": str(exc)}
                report["ok"] = False
    return report
