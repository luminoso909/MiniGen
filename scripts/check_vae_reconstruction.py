"""运行 P2 ArtBench 的离线 VAE 重建审计；没有模型下载或训练。"""

import argparse
import json
import math
from pathlib import Path

import torch

from minigen.autoencoders.sdxl import FrozenVAE
from minigen.evaluation.reconstruction import reconstruct, select_samples
from minigen.evaluation.reconstruction_metrics import LPIPSAlex
from minigen.evaluation.reconstruction_review import score_reconstructions

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/artbench_pd_256/v1"
DATA_LOCK = ROOT / "reports/v2/data/artbench_freeze.json"
VAE_LOCK = ROOT / "configs/models/sdxl_vae.json"
LPIPS_LOCK = ROOT / "reports/v2/components/p2_assets.json"


def _reconstruct(
    output: Path,
    device: str,
    seed: int,
    batch_size: int,
    cpu_threads: int,
) -> None:
    """从训练切分重建固定样本，并持久化评分使用的全部身份。

    Args:
        output: 全新的本地审计目录。
        device: VAE 的设备名。
        seed: 主抽样及诊断候选池的种子。
        batch_size: VAE 图像批量。
        cpu_threads: 已设置的 Torch CPU 线程数，写入元数据。

    Raises:
        ValueError: 数据或模型身份不符；OSError: 本地资源读写失败。

    副作用:
        写入审计图像清单、FP32 重建张量和运行元数据；打印进度。
    """
    samples, manifest_hash = select_samples(DATA_ROOT, DATA_LOCK, seed=seed)
    adapter = FrozenVAE.load(ROOT, VAE_LOCK, device=device)
    report = reconstruct(adapter, DATA_ROOT, samples, output, batch_size=batch_size)
    report.update(
        manifest_sha256=manifest_hash,
        seed=seed,
        cpu_threads=cpu_threads,
        batch_size=batch_size,
        count_random=200,
        count_diagnostic=len(samples) - 200,
        sample_policy="random.Random(seed).sample over TRAIN sorted by sample_id",
        risk_policy="another 256 TRAIN candidates; 3 per dense edges, strong chroma, sparse detail",
    )
    (output / "reconstruction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _score(
    output: Path,
    batch_size: int,
) -> None:
    """确认评分输入与固定的 P1/P2 身份一致后，写入指标及人工复核页面。

    Args:
        output: 含完整重建结果的审计目录。
        batch_size: CPU 评分批量。

    Raises:
        ValueError: 数据、VAE 身份或 LPIPS 固定数值探针不符。
        OSError: 读取资源或写入报告失败。

    副作用:
        写入 metrics.json、联系表、逐图图像对及 review.html。
    """
    run = json.loads((output / "reconstruction.json").read_text())
    samples, manifest_hash = select_samples(DATA_ROOT, DATA_LOCK, seed=run["seed"])
    lock = json.loads(VAE_LOCK.read_text())
    if (
        any(run["model"][key] != lock[key] for key in ("repo_id", "revision", "files"))
        or run["scale"] != lock["scaling_factor"]
        or run["count_random"] != 200
    ):
        raise ValueError("Expected the frozen ArtBench-200 and SDXL VAE run identity")
    lpips = LPIPSAlex.load(ROOT, LPIPS_LOCK)
    probe = json.loads(LPIPS_LOCK.read_text())["lpips_alex"]["cpu_smoke"]
    rng = torch.Generator().manual_seed(probe["seed"])
    a = torch.rand((1, 3, 64, 64), generator=rng)
    b = torch.rand((1, 3, 64, 64), generator=rng)
    if lpips(a, a).item() != 0 or not math.isclose(
        lpips(a, b).item(), probe["different_fixture_score"], rel_tol=1e-5, abs_tol=1e-7
    ):
        raise ValueError("Fixed offline LPIPS probe does not match the asset acceptance")
    score_reconstructions(
        DATA_ROOT,
        output,
        lpips,
        manifest_sha256=manifest_hash,
        expected_samples=samples,
        batch_size=batch_size,
    )


def main() -> None:
    """完整重建及评分，或将两步拆开执行以单独复算指标。

    Raises:
        SystemExit: 命令参数解析失败；ValueError: 资源或模型验证失败。
        OSError: 本地资源缺失、读写失败或要求的输出目录已存在。

    副作用:
        设置 Torch CPU 线程数，写入本地审计产物，打印重建进度。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="本地审计目录")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--reconstruct-only", action="store_true", help="只执行 VAE 重建")
    stages.add_argument("--score-existing", action="store_true", help="重新评分已完成的重建")
    args = parser.parse_args()
    if args.cpu_threads <= 0 or args.batch_size <= 0:
        parser.error("CPU thread and batch counts must be positive")

    torch.set_num_threads(args.cpu_threads)
    if not args.score_existing:
        _reconstruct(args.output, args.device, args.seed, args.batch_size, args.cpu_threads)
    if not args.reconstruct_only:
        _score(args.output, args.batch_size)


if __name__ == "__main__":
    main()
