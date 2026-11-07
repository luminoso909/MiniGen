"""核对正式缓存对本机通过回执和显式 CPU 回退的要求。"""

import json
from pathlib import Path

import pytest

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import hash_file
from minigen.data.representation_cache import CacheSource
from minigen.evaluation.local_components import (
    CPU_REFERENCE_SHA256,
    _limits,
    validate_local_receipt,
)


@pytest.fixture
def local_receipt(
    tmp_path: Path,
) -> tuple[CacheSource, Path, JsonDict]:
    """构造 VAE 用 MPS、CLIP 显式回退 CPU 的离线最小回执。

    Args:
        tmp_path: pytest 分配的临时路径。

    Results:
        tuple: 独立的来源身份、原始审计路径和可变更的本机回执字典。

    副作用:
        写入模拟原始审计文件；不加载资产或使用 GPU。
    """
    ids = [f"sample-{i}" for i in range(200)]
    vae = {"revision": "revision-vae", "files": {}, "scaling_factor": 0.13025}
    evidence = tmp_path / "audit.json"
    evidence.write_text(
        json.dumps(
            {
                "manifest_sha256": "manifest",
                "random_sample_ids": ids,
                "reconstruction": {
                    "count_random": 200,
                    "scale": 0.13025,
                    "latent_shape": [4, 32, 32],
                    "model": vae,
                },
            }
        )
    )
    source = CacheSource(
        tmp_path,
        {"train": [{"sample_id": sid} for sid in ids]},
        {
            "manifest_sha256": "manifest",
            "vae": {**vae, "lock_sha256": "vae-lock"},
            "clip": {"lock_sha256": "clip-lock"},
        },
    )
    limits = _limits()
    report = {
        "schema_version": 1,
        "status": "passed",
        "sample_ids": ids[:20],
        "reference_sha256": CPU_REFERENCE_SHA256,
        "limits": limits,
        "identities": {
            "manifest_sha256": "manifest",
            "reconstruction_sha256": hash_file(evidence),
            "vae_lock_sha256": "vae-lock",
            "clip_lock_sha256": "clip-lock",
        },
        "devices": {"vae": "mps", "clip": "cpu"},
        "checks": {
            "vae": {
                "status": "passed",
                "cpu": {
                    "invariants": dict.fromkeys(limits["vae_invariants"], 0.0),
                    "historical_parity": dict.fromkeys(limits["vae_cpu_audit"], 0.0),
                },
                "mps": {
                    "status": "passed",
                    "invariants": dict.fromkeys(limits["vae_invariants"], 0.0),
                    "parity": dict.fromkeys(limits["vae_parity"], 0.0),
                },
            },
            "clip": {
                "status": "passed",
                "cpu": {"invariants": dict.fromkeys(limits["clip_invariants"], 0.0)},
                "mps": {"status": "failed", "error": "fixture: unsupported operator"},
            },
        },
        "runtime": {
            "dtype": "float32",
            "torch": "fixture",
            "mps_fallback": "0",
            "mps_fast_math": "0",
            "cuda_run": False,
            "mps": {"built": True, "available": True, "tensor_passed": True},
        },
    }
    return source, evidence, report


def test_accepted_devices_require_auditable_fallback(
    local_receipt: tuple[CacheSource, Path, JsonDict],
) -> None:
    """允许两个组件分别选择设备；CPU 回退缺少原因时拒绝。

    Args:
        local_receipt: 模拟混合设备验收回执及原证据。

    Raises:
        AssertionError: 接受结果、文件摘要或回退门槛不符合约定。

    副作用:
        在 pytest 临时目录写入两次模拟回执。
    """
    source, evidence, report = local_receipt
    path = evidence.with_name("receipt.json")
    path.write_text(json.dumps(report))
    assert validate_local_receipt(source, path, evidence) == (hash_file(path), "mps", "cpu")

    del report["checks"]["clip"]["mps"]["error"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not document MPS failure"):
        validate_local_receipt(source, path, evidence)


def test_receipt_rejects_out_of_bounds_measurement(
    local_receipt: tuple[CacheSource, Path, JsonDict],
) -> None:
    """组件状态写为通过时，仍拒绝超出固定容差的具体测量。

    Args:
        local_receipt: 模拟通过回执及原证据。

    Raises:
        AssertionError: 被测回执未因数值越界而拒绝。

    副作用:
        在 pytest 临时目录写入超界回执。
    """
    source, evidence, report = local_receipt
    report["checks"]["vae"]["mps"]["parity"]["latent_mean_abs"] = 0.1
    path = evidence.with_name("receipt.json")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="latent_mean_abs.*exceeds"):
        validate_local_receipt(source, path, evidence)
