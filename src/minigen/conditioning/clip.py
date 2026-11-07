"""正式 ArtBench 条件：离线锁定的单段 CLIP-B/32 token 特征。"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from minigen._typing import JsonDict

if TYPE_CHECKING:
    from transformers import CLIPTextModel, CLIPTokenizer

CLIP_TEXT_CONTEXT = 77
CLIP_TEXT_WIDTH = 512
CLIP_CONTENT_TOKENS = CLIP_TEXT_CONTEXT - 2
_CLIP_REPO = "openai/clip-vit-base-patch32"
_CLIP_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
)


@dataclass(frozen=True, slots=True)
class CLIPCondition:
    """保存下游可直接使用的冻结文本条件及其 token 证据。

    Args:
        features: [b, n, d] FP32 最后一层 token 特征，n=77、d=512；不做池化或投影。
        mask: [b, n] 布尔有效位，True 表示 BOS、内容或 EOS；False 表示填充。
        input_ids: [b, n] int64 token ID，用于定位文本长度与特殊符号问题。
    """

    features: Float32[Tensor, "b n d"]
    mask: Bool[Tensor, "b n"]
    input_ids: Int64[Tensor, "b n"]


def _verified_directory(
    root: Path,
    lock_path: Path,
) -> tuple[Path, JsonDict]:
    """验证锁的来源和本地文件完整性，再允许组件加载文件。

    Args:
        root: 解析权重目录相对路径的项目根目录。
        lock_path: 固定 CLIP 来源、修订号和文件摘要的 JSON 路径。

    Raises:
        OSError: 锁或组件文件缺失、不可读。
        ValueError: 锁格式、来源、路径、文件大小或 SHA-256 不匹配。

    Results:
        tuple[Path, JsonDict]: 已核验目录与设备无关的文件身份。
    """
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("repo_id") != _CLIP_REPO:
        raise ValueError("Expected the locked openai/clip-vit-base-patch32 text tower")
    revision, relative, files = (lock.get(key) for key in ("revision", "local_path", "files"))
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CLIP lock requires a full commit revision")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("CLIP lock requires a project-relative model directory")
    if not isinstance(files, dict) or files.keys() != _CLIP_FILES:
        raise ValueError("CLIP lock does not list the expected tokenizer and weights files")

    directory = (root / relative).resolve()
    if directory == root.resolve() or not directory.is_relative_to(root.resolve()):
        raise ValueError("CLIP model directory must be inside the project root")
    unexpected = {item.name for item in directory.iterdir() if not item.name.startswith(".")}
    if unexpected - _CLIP_FILES:
        raise ValueError(f"Untracked CLIP component files: {sorted(unexpected - _CLIP_FILES)}")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid CLIP file metadata: {name}")
        size, sha256 = record.get("bytes"), record.get("sha256")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            raise ValueError(f"Invalid CLIP file metadata: {name}")
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.stat().st_size != size:
            raise ValueError(f"CLIP file path or size mismatch: {name}")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != sha256:
            raise ValueError(f"CLIP file SHA-256 mismatch: {name}")

    return directory, {"repo_id": _CLIP_REPO, "revision": revision, "files": files}


class FrozenCLIPText:
    """将原文或空条件编码为固定 77 个 token 的 CLIP-B/32 特征。

    正式入口使用 from_lock 校验本地来源；本类不持有生成模型的可训练文本投影。

    Args:
        model: 已加载的 CLIP-B/32 文本塔。
        tokenizer: 与模型匹配的 CLIP tokenizer。
        identity: 权重和 tokenizer 的锁定身份。
        device: FP32 编码设备。

    副作用:
        构造时迁移传入模型，并将其设为冻结的 eval 模式。
    """

    def __init__(
        self,
        model: "CLIPTextModel",
        tokenizer: "CLIPTokenizer",
        identity: JsonDict,
        device: torch.device | str = "cpu",
    ) -> None:
        """校验模型和 tokenizer 的窗口与词表，再固定推理模式。

        Args:
            model: 77 位置、512 维的 CLIP 文本网络。
            tokenizer: 具有 BOS、EOS 与 PAD 的同源 tokenizer。
            identity: 模型与词表文件身份。
            device: FP32 编码使用的设备。

        Raises:
            ValueError: 位置数、宽度、词表或特殊 token 不兼容。
            RuntimeError: PyTorch 无法把模型迁移到指定设备。

        副作用:
            切换传入模型的训练模式、梯度开关、精度与设备。
        """
        if (model.config.max_position_embeddings, model.config.hidden_size) != (
            CLIP_TEXT_CONTEXT,
            CLIP_TEXT_WIDTH,
        ):
            raise ValueError("CLIP text tower must have 77 positions and width 512")
        bos, eos, pad = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        vocab_size = model.config.vocab_size
        if (
            tokenizer.model_max_length != CLIP_TEXT_CONTEXT
            or len(tokenizer) != vocab_size
            or not all(
                isinstance(token, int) and 0 <= token < vocab_size for token in (bos, eos, pad)
            )
            or bos == eos
            or bos == pad
        ):
            raise ValueError(
                "CLIP tokenizer context, vocabulary or special tokens are incompatible"
            )

        self.device = torch.device(device)
        self.model = model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.identity: JsonDict = {
            **identity,
            "encoding": {
                "version": "clip-text-single-v1",
                "context_tokens": CLIP_TEXT_CONTEXT,
                "max_content_tokens": CLIP_CONTENT_TOKENS,
                "hidden_width": CLIP_TEXT_WIDTH,
                "features": "last_hidden_state",
                "compute_dtype": "float32",
                "mask": "true for BOS/content/EOS; false for right padding",
                "bos_token_id": bos,
                "eos_token_id": eos,
                "pad_token_id": pad,
            },
        }

    @classmethod
    def from_lock(
        cls,
        lock_path: Path,
        *,
        root: Path,
        device: torch.device | str = "cpu",
    ) -> "FrozenCLIPText":
        """严格加载本地 Safetensors 文本塔与 tokenizer，无网络回退。

        Args:
            lock_path: 声明固定文本塔与词表文件身份的 JSON 路径。
            root: 模型目录相对于此路径解析。
            device: FP32 编码使用的设备。

        Raises:
            ImportError: 未安装 conditioning 可选依赖。
            OSError: 锁或组件文件缺失、不可读取。
            ValueError: 锁、文件摘要、模型参数集合或 tokenizer 不符合契约。
            RuntimeError: PyTorch 无法加载参数或迁移模型到指定设备。

        Results:
            FrozenCLIPText: 已校验、冻结且切到 eval 模式的 FP32 编码器。
        """
        from transformers import CLIPTextModel, CLIPTokenizer

        directory, identity = _verified_directory(root, lock_path)
        model, loading = CLIPTextModel.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        issues = {key: value for key, value in loading.items() if value}
        if issues:
            raise ValueError(f"CLIP text tower state does not match exactly: {issues}")
        tokenizer = CLIPTokenizer.from_pretrained(directory, local_files_only=True)
        return cls(model, tokenizer, identity, device)

    @torch.no_grad()
    def encode(
        self,
        captions: Sequence[str],
    ) -> CLIPCondition:
        """无截断地编码单段文本；空字符串保留有效 BOS 和 EOS。

        Args:
            captions: 非空字符串序列；每条最多 75 个内容 token，不接受嵌入的 CLIP 特殊符号。

        Raises:
            ValueError: 没有文本、输入包含非字符串、超出单段容量或内容含特殊 token。
            RuntimeError: 文本塔返回不符合契约的形状或精度，或设备计算失败。

        Results:
            CLIPCondition: 编码设备上的 [b, 77, 512] FP32 特征、[b, 77] 有效位与 token ID。
        """
        if (
            not captions
            or isinstance(captions, str)
            or not all(isinstance(c, str) for c in captions)
        ):
            raise ValueError("Expected a nonempty sequence of caption strings")
        tokenizer = self.tokenizer
        raw = tokenizer(list(captions), add_special_tokens=False, truncation=False, verbose=False)[
            "input_ids"
        ]
        special = set(tokenizer.all_special_ids)
        ids = torch.full(
            (len(captions), CLIP_TEXT_CONTEXT), tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, tokens in enumerate(raw):
            if len(tokens) > CLIP_CONTENT_TOKENS:
                raise ValueError(
                    f"Caption {row} has {len(tokens)} content tokens; maximum is "
                    f"{CLIP_CONTENT_TOKENS} without truncation"
                )
            if any(token in special for token in tokens):
                raise ValueError(f"Caption {row} contains an embedded CLIP special token")
            sequence = [tokenizer.bos_token_id, *tokens, tokenizer.eos_token_id]
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            # PAD 与 EOS 可能同 ID，因此根据位置而非 token ID 构造有效位。
            mask[row, : len(sequence)] = True

        ids, mask = ids.to(self.device), mask.to(self.device)
        # CLIP 缓存的源值固定为 FP32，外层训练器的自动混合精度不改变该约定。
        with torch.autocast(device_type=self.device.type, enabled=False):
            features = self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        if features.shape != (len(captions), CLIP_TEXT_CONTEXT, CLIP_TEXT_WIDTH) or (
            features.dtype != torch.float32
        ):
            raise RuntimeError("CLIP text tower must produce FP32 [b, 77, 512] token features")
        return CLIPCondition(features, mask, ids)
