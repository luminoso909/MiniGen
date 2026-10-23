"""Environment diagnostics; no downloads, model training, or global mutations."""

import importlib
import os
import platform
from importlib.metadata import version
from pathlib import Path

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


def tensor_check(torch, device: str) -> dict:
    """Exercise allocation, matmul, reduction, autograd and device synchronization."""
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


def diagnose(device: str = "auto") -> dict:
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"不支持的设备: {device}")
    # Respect an explicit setting; otherwise keep Matplotlib's process-local cache in the project.
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
