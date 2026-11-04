"""固定 CLIP 文本网络、无截断分段与可核验的 token 特征缓存。"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from minigen._typing import JsonDict
from minigen.autoencoders.frozen import hash_file
from minigen.data.latent import identity_digest

if TYPE_CHECKING:
    from transformers import CLIPTextModel, CLIPTokenizer


class FrozenTextEncoder:
    """冻结 CLIP 文本网络，把完整描述分为独立的 77-token 上下文。

    Args:
        model: 已严格加载的 CLIPTextModel；输出最后层 token 特征，不用 pooled projection。
        tokenizer: 同一固定版本的 CLIPTokenizer。
        identity: 权重与 tokenizer 文件身份。
        device: 实际执行设备。
        max_chunks: 每条文本的最大分段数；每段保留 BOS/EOS，不静默截断。

    Results:
        FrozenTextEncoder: 提供 encode 接口及完整分段约定。

    副作用:
        冻结传入模型参数，迁移到指定设备并切到 eval。
    """

    def __init__(
        self,
        model: "CLIPTextModel",
        tokenizer: "CLIPTokenizer",
        identity: JsonDict,
        device: str = "cpu",
        max_chunks: int = 2,
    ) -> None:
        """记录分段约定，冻结模型；不会访问网络。

        Args:
            model: CLIP 文本网络。
            tokenizer: 同源 tokenizer。
            identity: 模型与词表身份。
            device: cpu/mps/cuda。
            max_chunks: 正整数分段上限。

        Raises:
            ValueError: 分段上限非正。

        副作用:
            修改模型模式、梯度开关与设备。
        """
        if max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        self.tokenizer, self.device = tokenizer, device
        self.max_chunks = max_chunks
        self.context = int(model.config.max_position_embeddings)
        self.identity = {
            **identity,
            "segmentation": "contiguous tokens; BOS/EOS each chunk; reset positions",
            "max_chunks": max_chunks,
            "context": self.context,
            "features": "last_hidden_state; no pooling; FP32 -> FP16 cache",
        }

    @classmethod
    def load(
        cls,
        root: Path,
        lock_path: Path,
        device: str = "cpu",
        max_chunks: int = 2,
    ) -> "FrozenTextEncoder":
        """校验本地身份后加载组件；禁止隐式联网或视觉模型加载。

        Args:
            root: 项目根目录。
            lock_path: 固定文本组件 JSON。
            device: 执行设备。
            max_chunks: 最大分段数。

        Raises:
            ValueError: 文件哈希不匹配。
            ImportError/OSError: 缺少 conditioning extra 或本地文件。

        Results:
            FrozenTextEncoder: 冻结 FP32 文本编码器。
        """
        from transformers import CLIPTextModel, CLIPTokenizer

        identity = json.loads(lock_path.read_text())
        directory = root / identity["local_path"]
        for name, info in identity["files"].items():
            if hash_file(directory / name) != info["sha256"]:
                raise ValueError(f"Text component hash mismatch: {name}")
        model = CLIPTextModel.from_pretrained(directory, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(directory, local_files_only=True)
        return cls(model, tokenizer, identity, device, max_chunks)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
    ) -> tuple[Float[Tensor, "b n d"], Bool[Tensor, "b n"], list[int]]:
        """无截断编码完整文本；mask 中 True 表示可被图像 query 访问的 token。

        Args:
            texts: 非空文本列表；空字符串仍有有效 BOS/EOS。

        Raises:
            ValueError: 输入为空或任一描述超过显式分段容量。
            模型/tokenizer 异常: 底层调用失败时透传。

        Results:
            tuple: CPU FP32 特征 [b,n,d]、有效位 mask [b,n]、无特殊符号的长度。
            n: context × max_chunks；缺失段全被 mask，不影响注意力。
        """
        if not texts:
            raise ValueError("Expected at least one text")
        raw = self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        capacity = self.context - 2
        if any(len(ids) > capacity * self.max_chunks for ids in raw):
            raise ValueError("Caption exceeds explicit chunk capacity; no silent truncation")
        ids = torch.full(
            (len(texts), self.max_chunks, self.context),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, tokens in enumerate(raw):
            chunks = [tokens[i : i + capacity] for i in range(0, len(tokens), capacity)] or [[]]
            for column, chunk in enumerate(chunks):
                sequence = [self.tokenizer.bos_token_id, *chunk, self.tokenizer.eos_token_id]
                ids[row, column, : len(sequence)] = torch.tensor(sequence)
                mask[row, column, : len(sequence)] = True
        # 缺失段也给编码器 BOS/EOS，避免全 mask 的内部计算；输出 mask 仍保持全 False。
        missing = ~mask.any(-1)
        ids[missing] = ids[0, 0].new_tensor(
            [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
            + [self.tokenizer.pad_token_id] * (self.context - 2)
        )
        encoder_mask = mask.clone()
        encoder_mask[missing, :2] = True
        hidden = self.model(
            input_ids=ids.flatten(0, 1).to(self.device),
            attention_mask=encoder_mask.flatten(0, 1).long().to(self.device),
        ).last_hidden_state
        features = hidden.reshape(len(texts), -1, hidden.shape[-1]).cpu()
        return features, mask.flatten(1), [len(tokens) for tokens in raw]


def text_cache(
    encoder: FrozenTextEncoder,
    texts: list[str],
    directory: Path,
    *,
    batch_size: int = 32,
) -> tuple[Float[Tensor, "n_s n d"], Bool[Tensor, "n_s n"], dict[str, int], JsonDict]:
    """去重描述后缓存 FP16 特征与 mask，身份变化拒绝复用。

    Args:
        encoder: 显式冻结的本地文本编码器。
        texts: 所需描述；自动加入唯一空条件。
        directory: 独立缓存目录。
        batch_size: 编码批量，必须为正。

    Raises:
        ValueError: 批量非正、现有缓存身份/哈希/形状或 dtype 不兼容。
        OSError: 缓存文件访问失败时透传。

    Results:
        tuple: CPU FP16 特征、Bool mask、文本到行索引、完整身份与统计。

    副作用:
        首次运行时创建特征数组及元数据；不会下载模型或更改描述。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vocabulary = sorted(set(["", *texts]))
    identity = {
        "encoder": encoder.identity,
        "texts_sha256": identity_digest({"texts": vocabulary}),
        "count": len(vocabulary),
    }
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, payload = directory / "index.json", directory / "tokens.npz"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity or hash_file(payload) != meta["payload_sha256"]:
            raise ValueError("Incompatible or corrupted text cache")
        with np.load(payload, allow_pickle=False) as arrays:
            features, masks = (
                torch.from_numpy(arrays["features"].copy()),
                torch.from_numpy(arrays["mask"].copy()),
            )
        if (
            features.dtype != torch.float16
            or masks.dtype != torch.bool
            or features.shape[:2] != masks.shape
            or len(features) != len(vocabulary)
            or not masks.any(-1).all()
            or not torch.isfinite(features).all()
            or list(features.shape) != meta["shape"]
            or features.shape[1:]
            != (
                encoder.context * encoder.max_chunks,
                encoder.model.config.hidden_size,
            )
        ):
            raise ValueError("Invalid text cache shape, dtype, mask or values")
    else:
        outputs, valid, lengths = [], [], []
        for start in range(0, len(vocabulary), batch_size):
            hidden, mask, sizes = encoder.encode(vocabulary[start : start + batch_size])
            outputs.append(hidden.half())
            valid.append(mask)
            lengths.extend(sizes)
        features, masks = torch.cat(outputs), torch.cat(valid)
        temporary = payload.with_suffix(".partial")
        with temporary.open("wb") as handle:
            np.savez(handle, features=features.numpy(), mask=masks.numpy())
        temporary.replace(payload)
        meta = {
            "identity": identity,
            "payload_sha256": hash_file(payload),
            "shape": list(features.shape),
            "max_content_tokens": max(lengths),
            "over_one_context": sum(n > encoder.context - 2 for n in lengths),
            "bytes": payload.stat().st_size,
        }
        temporary_meta = meta_path.with_suffix(".partial")
        temporary_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        temporary_meta.replace(meta_path)
    return features, masks, {text: i for i, text in enumerate(vocabulary)}, meta
