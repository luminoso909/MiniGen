"""构建、回读和加载 AutoDL 独立全量 CUDA 表示缓存。"""

import argparse
import hashlib
import json
import runpy
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter

import torch

from minigen import commands
from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file
from minigen.cli import main as minigen_main
from minigen.conditioning.clip import FrozenCLIPText
from minigen.contracts import Condition
from minigen.data.representation_cache import (
    CacheSource,
    RepresentationCache,
    build_cache,
    cache_source,
)

EXPECTED_ROWS = {"train": 37_042, "validation": 4_637}
CACHE_SCOPE = "autodl_independent_full_cuda"


def full_source(
    root: Path,
    data: JsonDict,
) -> CacheSource:
    """按冻结清单加载完整训练和验证记录，登记独立 CUDA 编码身份。

    Args:
        root: MiniGen 项目根目录。
        data: 包含数据、冻结清单和冻结组件路径的配置。

    Raises:
        OSError/ValueError: 输入不可读取、身份或划分记录不匹配。

    Results:
        CacheSource: 保持原划分顺序的全量记录和本次独立编码身份。
    """
    source = cache_source(
        root / data["data_root"],
        root / data["freeze"],
        root / data["vae_lock"],
        root / data["clip_lock"],
        None,
    )
    if {split: len(rows) for split, rows in source.records.items()} != EXPECTED_ROWS:
        raise ValueError("Expected all 37,042 training and 4,637 validation rows")
    # 既有写入器的非 Mac 批准缓存入口以 pilot 标记兼容，本次保留全部行。
    identity = {
        **source.identity,
        "kind": "pilot",
        "cache_scope": CACHE_SCOPE,
        "covered_splits": EXPECTED_ROWS,
        "encoding_device": "cuda",
        "encoding_math": "fp32_no_tf32",
    }
    return replace(source, identity=identity)


def open_full_cuda_cache(
    root: Path,
    data: JsonDict,
    *,
    directory: Path,
) -> tuple[CacheSource, RepresentationCache, Condition, JsonDict]:
    """打开已独立回读通过的缓存，并为训练和采样绑定行与缓存身份。

    Args:
        root: MiniGen 项目根目录。
        data: 包含数据路径、缓存路径和所需划分子集的配置。
        directory: 本轮全量缓存目录。

    Raises:
        OSError/ValueError: 配置、完整回读回执、来源或载荷不一致。

    Results:
        tuple: 缓存来源、只读表示、空文本条件及检查点数据身份。
    """
    if (root / data["cache"]).resolve() != directory.resolve():
        raise ValueError(f"Full CUDA experiment must set data.cache to {directory}")
    index_hash = hash_file(directory / "index.json")
    verified = json.loads((directory.parent / "full_cache_readback.json").read_text())
    if (
        verified["status"] != "passed"
        or verified["cache_index_sha256"] != index_hash
        or verified["verified_rows"] != EXPECTED_ROWS
    ):
        raise ValueError("Independent full-cache readback does not match this index")
    source = full_source(root, data)
    cache = RepresentationCache(directory, source)
    selected = {}
    for split, records in source.records.items():
        indices = commands.selected_indices(source, data, split)
        ids = [records[index]["sample_id"] for index in indices]
        selected[split] = {
            "count": len(ids),
            "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
    features, mask = cache.blank()
    identity = {
        "cache_index_sha256": index_hash,
        "source": source.identity,
        "subsets": selected,
    }
    return source, cache, Condition(features[None].float(), mask[None]), identity


def prepare(
    run_root: Path,
    directory: Path,
    *,
    build: bool,
    batch_size: int,
) -> None:
    """独立构建或完整回读 AutoDL CUDA 缓存并保存回执。

    Args:
        run_root: 本轮证据目录。
        directory: 已发布或待发布的全量缓存路径。
        build: 为真时编码并发布，为假时独立回读。
        batch_size: 编码微批大小，回读时不使用。

    Raises:
        OSError/RuntimeError/ValueError: 缓存、冻结组件、CUDA 或输出异常。

    副作用:
        编码模式在 CUDA 运行冻结组件并写入缓存，回读模式扫描所有行。
    """
    root = Path.cwd()
    data = commands.DEFAULT_DATA
    started = perf_counter()
    torch.set_num_threads(5)
    source = full_source(root, data)
    if build:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        build_cache(
            source,
            FrozenVAE.load(root, root / data["vae_lock"], device="cuda"),
            FrozenCLIPText.from_lock(root / data["clip_lock"], root=root, device="cuda"),
            directory,
            batch_size=batch_size,
            report_progress=True,
        )
        torch.cuda.synchronize()
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    report = {
        "status": "built" if build else "passed",
        "cache_scope": CACHE_SCOPE,
        "cache_path": str(directory),
        "cache_index_sha256": hash_file(index_path),
        "counts": {split: len(rows) for split, rows in source.records.items()},
        "source": source.identity,
        "execution": index["execution"],
        "payload_sha256": index["payload_sha256"],
    }
    if build:
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    else:
        report["verified_rows"] = RepresentationCache(directory, source).verify_rows(source)
    report["elapsed_seconds"] = round(perf_counter() - started, 3)
    filename = "full_cache_build.json" if build else "full_cache_readback.json"
    (run_root / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> int:
    """派发缓存构建、独立回读和使用同一缓存的 MiniGen 命令。

    Raises:
        SystemExit: 参数无效或委派脚本主动退出。
        OSError/RuntimeError/ValueError: 冻结资产、缓存或运行失败。

    Results:
        int: 缓存操作成功返回零，MiniGen 命令返回原退出码。

    副作用:
        按所选命令创建缓存和回执，或向当前进程注入独立缓存读取器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("action", choices=("build", "verify", "minigen", "script"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    run_root = opts.run_root.resolve()
    directory = run_root / "full_cache"
    if opts.action in {"build", "verify"}:
        options = argparse.ArgumentParser()
        options.add_argument("--batch-size", type=int, default=64)
        parsed = options.parse_args(opts.args)
        prepare(run_root, directory, build=opts.action == "build", batch_size=parsed.batch_size)
        return 0
    commands.open_cache = partial(open_full_cuda_cache, directory=directory)
    if opts.action == "minigen":
        return minigen_main(opts.args)
    if not opts.args:
        parser.error("script requires a script path")
    sys.argv[:] = opts.args
    runpy.run_path(opts.args[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
