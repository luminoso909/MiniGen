"""固定验证图检查评测特征的设备与批处理一致性，不读取正式测试图。"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from minigen._typing import JsonDict
from minigen.evaluation.benchmark import load_benchmark
from minigen.evaluation.features import FeatureSuite


def check_backends(
    root: Path,
    benchmark: Path,
    output: Path,
    device: str,
) -> JsonDict:
    """以 CPU FP32 为参考检查两张固定验证图，超过预设误差时明确选择 CPU。

    Args:
        root: 项目根目录。
        benchmark: 已锁定参考清单；只读取 validation_references 前两条。
        output: 新的数值回执文件。
        device: 待验设备 mps 或 cuda。

    Raises:
        FileExistsError: 输出已存在。
        OSError/ValueError: 参考、CPU 特征或权重不可读取。

    Results:
        JsonDict: 三路特征的绝对误差、固定阈值和建议设备；失败不放宽阈值。

    副作用:
        顺序加载冻结权重、写特征缓存与诊断回执，设置 CPU 线程数为 5。
    """
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(5)
    definition = load_benchmark(root / benchmark)
    rows = definition["validation_references"][:2]
    paths = [root / definition["data_root"] / row["image_path"] for row in rows]
    texts = [row["caption"] for row in rows]
    results = []
    started = perf_counter()
    for name in ("inception", "clip_images", "clip_text"):
        inputs = texts if name == "clip_text" else paths
        cpu = FeatureSuite(root, Path("configs/models/evaluation.json"), "cpu", 2)
        baseline = getattr(cpu, name)(inputs)
        # 独立缓存位置避免命中首批输出，从而实际验证 batch=1 路径。
        cpu.batch_size = 1
        cpu.cache = root / "artifacts/cache/evaluation_batch_check"
        single = getattr(cpu, name)(inputs)
        batch_error = float((baseline - single).abs().max())
        cpu.close()
        del cpu
        gc.collect()
        report = {
            "component": name,
            "shape": list(baseline.shape),
            "dtype": str(baseline.dtype),
            "batch_max_abs": batch_error,
            "batch_passed": batch_error <= 5e-4,
            "tolerance_max_abs": 5e-4,
            "requested_device": device,
            "device_passed": False,
            "selected_device": "cpu",
        }
        candidate = None
        try:
            candidate = FeatureSuite(root, Path("configs/models/evaluation.json"), device, 2)
            actual = getattr(candidate, name)(inputs)
            error = float((baseline - actual).abs().max())
            report.update(device_max_abs=error, device_passed=error <= 5e-4)
            if report["device_passed"] and report["batch_passed"]:
                report["selected_device"] = device
        except (RuntimeError, ValueError, OSError, NotImplementedError) as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            if candidate is not None:
                candidate.close()
            del candidate
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        results.append(report)
        print(json.dumps(report), flush=True)
    receipt = {
        "scope": "real_local",
        "reference": "cpu_fp32",
        "torch": str(torch.__version__),
        "sample_ids": [row["sample_id"] for row in rows],
        "results": results,
        "elapsed_seconds": perf_counter() - started,
        "cuda_executed": device == "cuda" and any(r["device_passed"] for r in results),
        "complete": all(r["batch_passed"] for r in results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    """解析本机或 CUDA 的有限评测器数值验收参数。

    Raises:
        SystemExit: 参数缺失。
        OSError/ValueError: 配置或资产不符合要求。

    副作用:
        执行固定图像特征对照并写回执，不训练。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    args = parser.parse_args()
    check_backends(**vars(args))


if __name__ == "__main__":
    main()
