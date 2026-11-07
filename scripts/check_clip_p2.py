"""用正式 ArtBench 清单和锁定文本塔验收 P2 文本条件；不会训练或联网。"""

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from minigen._typing import JsonDict
from minigen.conditioning.clip import CLIP_CONTENT_TOKENS, CLIP_TEXT_WIDTH, FrozenCLIPText

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/processed/artbench_pd_256/v1/manifest.jsonl")
SAMPLE_IDS = ("ab-00000", "ab-07820")
PROBE_SEED = 20261107


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    """记录清单中长度核对与真实图像测试使用的最少字段。

    Args:
        sample_id: 稳定的 ArtBench 样本标识。
        image_path: 相对于清单所在目录的图像路径。
        split: 项目训练、验证或测试划分。
        caption: 清洗后的英文条件。
        caption_tokens: 数据准备时记录的 CLIP 内容 token 数。
        sha256: 冻结清单记录的实际图像文件摘要。
    """

    sample_id: str
    image_path: str
    split: str
    caption: str
    caption_tokens: int
    sha256: str


def _rows(
    path: Path,
) -> Iterator[CaptionRecord]:
    """顺序读取已经核对哈希的正式清单。

    Args:
        path: ArtBench JSONL 清单路径。

    Raises:
        OSError: 文件不存在或无法读取。
        ValueError: JSON 行无法解析。
        KeyError: 记录缺少验收所用字段。

    Results:
        Iterator[CaptionRecord]: 保持清单顺序的 Caption 记录。
    """
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield CaptionRecord(*(row[key] for key in CaptionRecord.__dataclass_fields__))


def _audit_lengths(
    encoder: FrozenCLIPText,
    manifest: Path,
) -> tuple[JsonDict, dict[str, CaptionRecord]]:
    """重新分词全量 Caption，逐项核对清单长度，不运行文本塔。

    Args:
        encoder: 已加载、与冻结清单配套的文本编码器。
        manifest: 冻结数据的 JSONL 清单。

    Raises:
        OSError: 清单无法读取。
        ValueError: 分词记录超限、空白、含特殊 token，或与清单内容长度不符。

    Results:
        tuple[JsonDict, dict[str, CaptionRecord]]: 全量长度统计与验收使用的两条记录。
    """
    rows = list(_rows(manifest))
    examples = {row.sample_id: row for row in rows if row.sample_id in SAMPLE_IDS}
    histogram = [0] * (CLIP_CONTENT_TOKENS + 1)
    split_counts: dict[str, int] = {}
    special = set(encoder.tokenizer.all_special_ids)
    for offset in range(0, len(rows), 256):
        chunk = rows[offset : offset + 256]
        raw = encoder.tokenizer(
            [row.caption for row in chunk],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        for row, tokens in zip(chunk, raw, strict=True):
            size = len(tokens)
            if size != row.caption_tokens or not 1 <= size <= CLIP_CONTENT_TOKENS:
                raise ValueError(f"Invalid CLIP length for {row.sample_id}: {size}")
            if special.intersection(tokens):
                raise ValueError(f"Embedded special token in {row.sample_id}")
            histogram[size] += 1
            split_counts[row.split] = split_counts.get(row.split, 0) + 1
    return {
        "total": len(rows),
        "split_counts": split_counts,
        "content_tokens_min": next(i for i, count in enumerate(histogram) if count),
        "content_tokens_max": max(i for i, count in enumerate(histogram) if count),
        "content_tokens_histogram": {str(i): count for i, count in enumerate(histogram) if count},
        "zero_tokens": 0,
        "over_capacity": 0,
        "embedded_special_tokens": 0,
        "manifest_count_mismatches": 0,
    }, examples


def _condition_probe(
    encoder: FrozenCLIPText,
    manifest: Path,
    examples: dict[str, CaptionRecord],
) -> JsonDict:
    """用同一真实图像的 query 比较不同 Caption，检查 mask、梯度和存储误差。

    Args:
        encoder: 已核验的冻结 CLIP 文本塔。
        manifest: 冻结清单路径，用于定位测试图像。
        examples: 图像源与替换 Caption 的两条训练记录。

    Raises:
        AssertionError: 边界、特征、mask、梯度或 FP16 存储检查失败。
        OSError: 样本图像无法读取。
        RuntimeError: 编码、注意力计算或设备失败。

    Results:
        JsonDict: 可序列化的实际编码与消费者测试数字。

    副作用:
        固定当前脚本进程的 PyTorch 随机种子；仅更新临时消费者的梯度。
    """
    first, second = (examples[sample_id] for sample_id in SAMPLE_IDS)
    assert first.split == second.split == "train"
    captions = [first.caption, second.caption, "", " " * 3, "one", "one " * 75]
    condition = encoder.encode(captions)
    features, mask, ids = condition.features, condition.mask, condition.input_ids
    tokenizer = encoder.tokenizer
    assert list(features.shape) == [6, 77, 512] and features.dtype == torch.float32
    assert mask.dtype == torch.bool and ids.dtype == torch.long
    assert mask.sum(-1).tolist() == [
        first.caption_tokens + 2,
        second.caption_tokens + 2,
        2,
        2,
        3,
        77,
    ]
    assert torch.all(mask.any(-1)) and torch.isfinite(features).all()
    assert torch.all(ids[:, 0] == tokenizer.bos_token_id)
    assert tokenizer.eos_token_id == tokenizer.pad_token_id
    assert torch.all(ids[2, 1:] == tokenizer.eos_token_id) and mask[2, 1] and not mask[2, 2]
    assert torch.equal(mask[2], mask[3]) and torch.equal(features[2], features[3])
    assert not features.requires_grad
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())
    try:
        encoder.encode(["one " * 76])
    except ValueError as exc:
        assert "76 content tokens" in str(exc)
    else:
        raise AssertionError("A 76-token caption was not rejected")

    # 仅用很小的可训练投影验证文本条件到图像 query 的链路，避免实现任何 P3 主干。
    torch.manual_seed(PROBE_SEED)
    device = encoder.device
    query_projection = nn.Linear(3, 16, bias=False, device=device, dtype=torch.float32)
    text_projection = nn.Linear(CLIP_TEXT_WIDTH, 32, bias=False, device=device, dtype=torch.float32)
    image_path = manifest.parent / first.image_path
    with image_path.open("rb") as image_bytes:
        assert hashlib.file_digest(image_bytes, "sha256").hexdigest() == first.sha256
        image_bytes.seek(0)
        with Image.open(image_bytes) as image:
            assert image.mode == "RGB" and image.size == (256, 256)
            pixels = pil_to_tensor(image).to(device, dtype=torch.float32) / 255
    patches = F.adaptive_avg_pool2d(pixels, (4, 4)).flatten(1).transpose(0, 1)
    query = query_projection(patches).unsqueeze(0).expand(2, -1, -1)
    key, value = text_projection(features[:2]).chunk(2, dim=-1)
    visible = mask[:2, None, :]
    conditioned = F.scaled_dot_product_attention(query, key, value, attn_mask=visible)
    # 只修改填充 K/V；正确的 mask 必须让输出对该扰动不变。
    masked_delta = (~visible).transpose(1, 2).to(key.dtype) * 1000.0
    padded = F.scaled_dot_product_attention(
        query, key + masked_delta, value - masked_delta, attn_mask=visible
    )
    torch.testing.assert_close(conditioned, padded, rtol=0.0, atol=0.0)
    conditioned.square().mean().backward()
    assert text_projection.weight.grad is not None and query_projection.weight.grad is not None
    text_grad = text_projection.weight.grad.norm().item()
    query_grad = query_projection.weight.grad.norm().item()
    assert 0 < text_grad < float("inf") and 0 < query_grad < float("inf")
    assert all(p.grad is None for p in encoder.model.parameters())

    shared_visible = mask[0] & mask[1]
    feature_delta = (features[0, shared_visible] - features[1, shared_visible]).abs().mean().item()
    condition_delta = (conditioned[0] - conditioned[1]).abs().mean().item()
    assert feature_delta > 0 and condition_delta > 0
    # 在 CPU 上模拟按 IEEE FP16 存盘再恢复为 FP32，仅统计下游可见位置。
    visible_features = features[mask].cpu()
    storage_error = (visible_features.half().float() - visible_features).abs()
    assert torch.isfinite(storage_error).all()
    return {
        "same_image_id": first.sample_id,
        "same_image_sha256": first.sha256,
        "caption_sample_ids": [first.sample_id, second.sample_id],
        "caption_content_tokens": [first.caption_tokens, second.caption_tokens],
        "feature_shape": list(features.shape),
        "mask_valid_counts": mask.sum(-1).tolist(),
        "fp16_visible_values": storage_error.numel(),
        "fp16_abs_mean": storage_error.mean().item(),
        "fp16_abs_max": storage_error.max().item(),
        "fp16_rmse": storage_error.square().mean().sqrt().item(),
        "valid_feature_abs_diff_mean": feature_delta,
        "conditioned_output_abs_diff_mean": condition_delta,
        "text_projection_gradient_norm": text_grad,
        "query_projection_gradient_norm": query_grad,
        "masked_padding_perturbation_max_output_diff": (conditioned - padded).abs().max().item(),
        "clip_parameters": sum(p.numel() for p in encoder.model.parameters()),
        "clip_trainable_parameters": sum(
            p.numel() for p in encoder.model.parameters() if p.requires_grad
        ),
        "clip_parameters_with_grad": sum(p.grad is not None for p in encoder.model.parameters()),
    }


def main() -> None:
    """执行 P2 CLIP 正式验收，并写入可复核的 JSON 回执。

    Raises:
        AssertionError: 资产来源、边界输入或梯度等验收不通过。
        OSError: 缺少已冻结的数据、组件文件或输出目录不可写。
        ValueError: 锁、清单或 Caption 不符合正式约定。
        RuntimeError: 模型或设备不可用。

    副作用:
        读取本地数据和权重，写入 JSON、打印回执，并设置当前脚本进程的 PyTorch 线程数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="MiniGen 项目根目录")
    parser.add_argument("--device", default="cpu", help="FP32 编码设备，默认 CPU")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2/p2/clip_acceptance.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    manifest = args.root / MANIFEST
    lock = args.root / "configs/models/clip_text.json"
    freeze = json.loads(
        (args.root / "reports/v2/data/artbench_freeze.json").read_text(encoding="utf-8")
    )
    with manifest.open("rb") as stream:
        manifest_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert manifest_sha == freeze["manifest_sha256"]
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert lock_sha == freeze["identity"]["token_lock_sha256"]
    encoder = FrozenCLIPText.from_lock(lock, root=args.root, device=args.device)
    lengths, examples = _audit_lengths(encoder, manifest)
    assert lengths["total"] == freeze["manifest_rows"] == 46298
    assert lengths["split_counts"] == freeze["split_counts"]
    receipt = {
        "status": "passed",
        "device": args.device,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "manifest_sha256": manifest_sha,
        "clip_lock_sha256": lock_sha,
        "encoder": encoder.identity,
        "length_audit": lengths,
        "probe_seed": PROBE_SEED,
        "condition_probe": _condition_probe(encoder, manifest, examples),
        "seconds": round(time.perf_counter() - started, 3),
    }
    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"P2 CLIP checks passed: {output}")


if __name__ == "__main__":
    main()
