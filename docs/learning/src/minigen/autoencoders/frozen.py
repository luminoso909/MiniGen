"""历史固定 128 像素缓存接口及共用 VAE 适配器的兼容导出。"""

import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE as FrozenVAE
from minigen.autoencoders.sdxl import hash_file as hash_file
from minigen.autoencoders.sdxl import image_batch as image_batch
from minigen.autoencoders.sdxl import to_image as to_image


def cache_identity(
    adapter: FrozenVAE,
    records: list[JsonDict],
    manifest_hash: str,
) -> JsonDict:
    """汇集足以判定潜变量缓存能否复用的身份。

    Args:
        adapter: 冻结 VAE，提供权重身份和缩放。
        records: 按缓存排列的样本，包含 ID 与图像哈希。
        manifest_hash: 完整数据清单的 SHA-256，同时约束描述及数据版本。

    Raises:
        KeyError: 模型身份或样本记录缺少必要字段时透传。

    Results:
        JsonDict: JSON 身份，包含组件、缩放、后验策略、预处理、dtype 与有序样本哈希。
    """
    return {
        "schema_version": 1,
        "model_revision": adapter.identity["revision"],
        "model_files": adapter.identity["files"],
        "scale": adapter.scale,
        "encoder_mode": "posterior_mean",
        "preprocessing": "RGB/128/no-resize/[-1,1]",
        "dtype": "float32",
        "manifest_sha256": manifest_hash,
        "samples": [{"id": r["sample_id"], "sha256": r["sha256"]} for r in records],
    }


def load_cache(
    path: str | Path,
    expected: JsonDict,
) -> Float[Tensor, "b c h w"]:
    """验证缓存身份和文件哈希后，以禁用 pickle 的方式加载数组。

    Args:
        path: 本地 .npy 路径，旁边需存在同名 .json 元数据。
        expected: 调用方本次期待的完整缓存身份，需逐项完全一致。

    Raises:
        ValueError: 身份、载荷哈希、形状或 dtype 不符。
        OSError/JSONDecodeError: 文件或元数据读取失败。

    Results:
        Float[Tensor, "b c h w"]: CPU FP32 [b,c,h,w] 张量，次序与身份中的样本表一致。
    """
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["identity"] != expected:
        raise ValueError("Incompatible latent cache identity")
    if hash_file(path) != meta["payload_sha256"]:
        raise ValueError("Latent payload hash mismatch")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != meta["shape"] or str(array.dtype) != "float32":
        raise ValueError("Invalid latent shape/dtype")
    return torch.from_numpy(array)


def write_cache(
    path: str | Path,
    latents: Float[Tensor, "b c h w"],
    identity: JsonDict,
) -> None:
    """先保存 CPU FP32 潜变量，再以临时文件替换元数据。

    Args:
        path: 目标 .npy 文件，父目录会创建。
        latents: [b,c,h,w]，保存时 detach、转 CPU FP32。
        identity: 本次数据和编码约定的 JSON 身份。

    Raises:
        OSError/TypeError: 文件写入或身份无法 JSON 序列化时透传；张量迁移错误透传。

    副作用:
        创建父目录并替换目标数组及 JSON 身份文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = latents.detach().cpu().to(torch.float32).numpy()
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)
    meta = {"identity": identity, "payload_sha256": hash_file(path), "shape": list(array.shape)}
    temp_meta = path.with_suffix(".json.partial")
    temp_meta.write_text(json.dumps(meta, indent=2))
    temp_meta.replace(path.with_suffix(".json"))
