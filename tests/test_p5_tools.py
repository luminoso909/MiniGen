"""评测资产、显式 CLI 和云配置准备的离线契约测试，不发起训练或下载。"""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

from minigen import bench_command
from minigen._typing import JsonDict
from minigen.evaluation import generation
from minigen.evaluation.assets import asset_spec, download_assets, verify_asset
from minigen.training.checkpoint import CHECKPOINT_SCHEMA
from minigen.training.config import TrainConfig, build_model
from minigen.training.runtime import hardware_description


def load_cloud_tools(
    name: str = "prepare_cloud_run",
) -> ModuleType:
    """按绝对路径加载配置脚本，不依赖 pytest 的 sys.path 或 scripts 包。

    Args:
        name: 不含扩展名的本仓库脚本名。

    Raises:
        ImportError: 脚本没有可用模块加载器。

    Results:
        ModuleType: 只提供配置准备函数的脚本命名空间。
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"minigen_{name}_fixture", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load cloud config script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_config = load_cloud_tools().prepare_config


def pilot_fixture(
    root: Path,
) -> Path:
    """构造明确属于测试的 CUDA 格式回执，验证配置转换而非 CUDA 运行。

    Args:
        root: 隔离临时目录。

    Results:
        Path: 具有相邻 run.json 的测试 pilot.json。

    副作用:
        写小型模型规格与模拟运行回执，不分配 CUDA。
    """
    model = {"architecture": "b0", "hidden_size": 32, "depth": 1, "num_heads": 4}
    folder = root / "configs/models"
    folder.mkdir(parents=True)
    (folder / "b0_30m.json").write_text(json.dumps(model))
    run_id = "simulated-pilot-fixture"
    report = {
        "status": "measured",
        "cuda_device": "fixture-device-not-real-hardware",
        "precision": "bf16",
        "measured_steps": 2,
        "step_seconds_mean": 2.0,
        "training_result": {
            "device": "cuda",
            "architecture": "b0",
            "precision": "bf16",
            "run_id": run_id,
        },
    }
    path = root / "pilot.json"
    path.write_text(json.dumps(report))
    config = TrainConfig(model=model, device="cuda", precision="bf16").to_dict()
    (root / "run.json").write_text(json.dumps({"run_id": run_id, "config": config}))
    return path


def test_asset_verification_reuse_and_path_escape(
    tmp_path: Path,
) -> None:
    """锁定资产复用不访问网络，拒绝改写文件与逃出根目录的路径。

    Args:
        tmp_path: 临时资产根。

    Raises:
        AssertionError: 错误资产被接受或已有正确资产重复下载。

    副作用:
        写微小测试文件与锁，仅调用已有文件复用途径。
    """
    folder = tmp_path / "weights"
    folder.mkdir()
    data = b"fixture model bytes"
    file = folder / "model.bin"
    file.write_bytes(data)
    spec = {
        "models": {
            "fixture": {
                "local_path": "weights",
                "files": {
                    "model.bin": {
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "url": "https://invalid.example/not-requested",
                    }
                },
            }
        }
    }
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(spec))
    assert verify_asset(tmp_path, lock, "fixture") == folder
    result = download_assets(tmp_path, lock, ["fixture"])
    assert result["models"]["fixture"]["downloaded"] == 0
    file.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="mismatch"):
        verify_asset(tmp_path, lock, "fixture")
    spec["models"]["fixture"]["local_path"] = "../outside"
    lock.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        asset_spec(tmp_path, lock, "fixture")
    spec["models"]["fixture"]["local_path"] = "weights"
    spec["models"]["fixture"]["files"]["../../escape"] = spec["models"]["fixture"]["files"].pop(
        "model.bin"
    )
    lock.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        asset_spec(tmp_path, lock, "fixture")


def test_cloud_config_requires_actual_matching_pilot_and_budget(
    tmp_path: Path,
) -> None:
    """核对 CUDA 身份、运行配方和预算，准备文件本身绝不启动训练。

    Args:
        tmp_path: 模拟云资产根。

    Raises:
        AssertionError: 非 CUDA、超预算或错误相邻运行身份被接受。

    副作用:
        写测试配置与预算回执，不连接远程或分配 GPU。
    """
    pilot = pilot_fixture(tmp_path)
    args = {
        "root": tmp_path,
        "architecture": "b0",
        "pilot": pilot,
        "output": Path("formal.json"),
        "run_dir": "outputs/formal",
        "steps": 100,
        "max_seconds": 1000.0,
        "hourly_cny": 2.0,
        "budget_cny": 1.0,
    }
    with pytest.raises(ValueError, match="budget"):
        prepare_config(**{**args, "budget_cny": 0.01})
    assert not (tmp_path / "formal.json").exists()
    report = json.loads(pilot.read_text())
    report["training_result"]["device"] = "mps"
    pilot.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="CUDA"):
        prepare_config(**args)
    report["training_result"]["device"] = "cuda"
    pilot.write_text(json.dumps(report))
    run_path = tmp_path / "run.json"
    original = json.loads(run_path.read_text())
    for field, value in [("device", "cpu"), ("precision", "fp32")]:
        changed = json.loads(json.dumps(original))
        changed["config"][field] = value
        run_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            prepare_config(**args)
    changed = json.loads(json.dumps(original))
    changed["run_id"] = "unrelated-run"
    run_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        prepare_config(**args)
    run_path.write_text(json.dumps(original))
    receipt = prepare_config(**args)
    config = json.loads((tmp_path / "formal.json").read_text())
    assert config["data"]["train_limit"] is None
    assert config["training"]["device"] == "cuda"
    assert config["training"]["schedule_steps"] == 100
    assert receipt["actual_cost_cny"] is None
    assert receipt["estimated_seconds_with_reserve"] == 250
    assert not (tmp_path / "outputs/formal").exists()
    with pytest.raises(FileExistsError):
        prepare_config(**args)
    single = {**args, "output": Path("one.json"), "run_dir": "outputs/one", "steps": 1}
    prepare_config(**single)
    assert json.loads((tmp_path / "one.json").read_text())["training"]["warmup_steps"] == 0


def test_bench_cli_forwards_explicit_generation_and_partial_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI 只将明确参数交给生成器，部分完成返回非零且不调用训练。

    Args:
        tmp_path: CLI 显式根目录。
        monkeypatch: 替换生成器为测试回调。
        capsys: 捕获 JSON 回执。

    Raises:
        AssertionError: 参数丢失、隐式默认不符或部分失败返回成功。

    副作用:
        临时替换生成器并输出模拟 JSON，不加载权重或生成实际图片。
    """
    received: JsonDict = {}

    def fake_generation(
        **options: object,
    ) -> JsonDict:
        """记录 CLI 调用参数并返回明确未完成的模拟结果。

        Args:
            options: CLI 传入的路径、恢复和采样设置。

        Results:
            JsonDict: 模拟的 partial 回执。

        副作用:
            更新测试内参数记录。
        """
        received.update(options)
        return {"scope": "simulated", "complete": False, "status": "partial"}

    monkeypatch.setattr(generation, "run_generation", fake_generation)
    result = bench_command.main(
        [
            "generate",
            "--root",
            str(tmp_path),
            "--config",
            "run.json",
            "--checkpoint",
            "model.pt",
            "--benchmark",
            "benchmark.json",
            "--output",
            "output",
            "--resume",
            "--retry-failed",
            "--count",
            "4",
            "--method",
            "heun",
            "--solver-nfe",
            "20",
            "--device",
            "cpu",
        ]
    )
    assert result == 1 and received["resume"] and received["retry_failed"]
    assert received["count"] == 4 and received["solver_nfe"] == 20
    assert received["method"] == "heun" and received["scope"] == "real_local"
    assert received["root"] == tmp_path and received["config"] == Path("run.json")
    assert json.loads(capsys.readouterr().out)["scope"] == "simulated"


def test_checkpoint_receipt_binds_exact_step_and_preserves_unknown_hardware(
    tmp_path: Path,
) -> None:
    """检查点回执只使用实际历史；FLOPs 需精确模型依据，不反推旧硬件。

    Args:
        tmp_path: 测试权重与回执目录。

    Raises:
        AssertionError: 步数、累计样本、FLOPs 或未知硬件边界错误。

    副作用:
        初始化微型 CPU 模型，写测试检查点并导出少量 JSON，不训练。
    """
    export_receipt = load_cloud_tools("export_checkpoint_receipt").export_receipt
    specification = {
        "architecture": "b0",
        "hidden_size": 16,
        "depth": 1,
        "num_heads": 2,
        "in_channels": 4,
        "latent_size": 4,
        "patch_size": 2,
        "text_dim": 8,
    }
    model = build_model(specification)
    config = TrainConfig(model=specification, device="cpu").to_dict()
    state = {
        "schema": CHECKPOINT_SCHEMA,
        "objective": "flow_velocity",
        "model": model.state_dict(),
        "config": config,
        "contract": {"recipe": config, "device": "cpu", "source_sha256": "fixture"},
        "step": 2,
        "elapsed_seconds": 5.0,
        "history": [
            {"step": 1, "samples": 2, "step_seconds": 1.0},
            {"step": 2, "samples": 3, "step_seconds": 2.0},
        ],
    }
    checkpoint = tmp_path / "fixture.pt"
    torch.save(state, checkpoint)
    output = tmp_path / "receipt.json"
    result = export_receipt(checkpoint, output)
    assert result["step"] == 2 and result["elapsed_seconds"] == 5.0
    assert result["performance"]["samples_seen"] == 5
    assert result["hardware"] is None and result["training_flops_estimate"] is None
    assert result["memory"]["cuda_peak_allocated"] is None
    assert hardware_description(torch.device("cpu"))
    inspection = {
        "models": [
            {
                "specification": specification,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "latent_shape": [1, 4, 4, 4],
                "text_shape": [1, 77, 8],
                "dominant_forward_flops": 100,
            }
        ]
    }
    path = tmp_path / "inspection.json"
    path.write_text(json.dumps(inspection))
    estimated = export_receipt(checkpoint, tmp_path / "estimated.json", path)
    assert estimated["training_flops_estimate"] == 1500
    assert "not hardware measurement" in estimated["training_flops_method"]
    inspection["models"][0]["trainable_parameters"] += 1
    path.write_text(json.dumps(inspection))
    with pytest.raises(ValueError, match="parameters"):
        export_receipt(checkpoint, tmp_path / "invalid.json", path)
    with pytest.raises(FileExistsError):
        export_receipt(checkpoint, output)
    state["history"].pop()
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="history"):
        export_receipt(checkpoint, tmp_path / "incomplete.json")
