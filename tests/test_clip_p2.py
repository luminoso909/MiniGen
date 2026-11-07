"""不读取模型或数据的 P2 CLIP 文件锁、形状与特殊 token 测试。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from jaxtyping import Int64
from torch import Tensor, nn

from minigen.conditioning.clip import FrozenCLIPText, _verified_directory

if TYPE_CHECKING:
    from transformers import CLIPTextModel, CLIPTokenizer


class _Tokenizer:
    """用空格切词的测试替身；复现实际词表的 EOS 与 PAD 共用行为。"""

    bos_token_id = 4
    eos_token_id = pad_token_id = 5
    all_special_ids = [bos_token_id, eos_token_id]
    model_max_length = 77

    def __len__(self) -> int:
        """报告替身的词表大小。

        Results:
            int: 词表中的 ID 数，与替身文本塔配置相同。
        """
        return 6

    def __call__(
        self,
        captions: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        verbose: bool,
    ) -> dict[str, list[list[int]]]:
        """生成长度可预测的内容 token，并确认生产调用没有截断。

        Args:
            captions: 待编码的测试字符串列表。
            add_special_tokens: 是否由 tokenizer 注入特殊符号；必须关闭。
            truncation: 是否由 tokenizer 截断超限文本；必须关闭。
            verbose: 是否输出底层长度警告；必须关闭，由上层明确拒绝。

        Raises:
            AssertionError: 上述任一开关没有明确关闭。

        Results:
            dict[str, list[list[int]]]: 每条文本的内容 ID；标记 eos 时模拟输入内嵌特殊符号。
        """
        assert not add_special_tokens and not truncation and not verbose
        return {
            "input_ids": [[5 if word == "eos" else 1 for word in text.split()] for text in captions]
        }


class _TextTower(nn.Module):
    """只记录入参、返回重复特征的模型替身，用于检测遮罩和冻结行为。"""

    config = SimpleNamespace(max_position_embeddings=77, hidden_size=512, vocab_size=6)

    def __init__(self) -> None:
        """提供一个原本可训练的参数，检查编码器是否显式冻结它。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.received_mask: Int64[Tensor, "b n"] | None = None

    def forward(
        self,
        input_ids: Int64[Tensor, "b n"],
        attention_mask: Int64[Tensor, "b n"],
    ) -> SimpleNamespace:
        """记录实际送入网络的遮罩，并返回与真实塔同形的隐藏状态。

        Args:
            input_ids: [b, n] token ID。
            attention_mask: [b, n] 由编码器转换成整数的有效位置。

        Results:
            SimpleNamespace: last_hidden_state 为 [b, n, 512] 的 FP32 特征。

        副作用:
            把收到的有效位记录到 received_mask，供断言检查。
        """
        self.received_mask = attention_mask
        features = input_ids.float().unsqueeze(-1).expand(-1, -1, 512) * self.weight
        return SimpleNamespace(last_hidden_state=features)


@pytest.fixture
def clip_encoder() -> FrozenCLIPText:
    """构造无可选依赖、无真实权重的测试编码器。

    Results:
        FrozenCLIPText: 接入模拟 tokenizer 与文本塔的正式编码逻辑。
    """
    return FrozenCLIPText(
        cast("CLIPTextModel", _TextTower()), cast("CLIPTokenizer", _Tokenizer()), {}
    )


def test_clip_mask_separates_eos_from_equal_pad_and_keeps_fp32(
    clip_encoder: FrozenCLIPText,
) -> None:
    """校验空条件、整窗口、实际模型遮罩和训练可用的冻结输入。

    Args:
        clip_encoder: 用替身实现验证输入输出契约的编码器。

    Raises:
        AssertionError: 张量形状、有效位、冻结状态或下游梯度不符合契约。
    """
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = clip_encoder.encode(["", "one", "one " * 75])
    assert output.features.dtype == torch.float32
    assert output.features.shape == (3, 77, 512)
    assert output.mask.dtype == torch.bool and output.input_ids.dtype == torch.long
    assert output.mask.sum(1).tolist() == [2, 3, 77]
    assert output.input_ids[0, 1:3].tolist() == [5, 5]
    assert output.mask[0, 1:3].tolist() == [True, False]
    model = cast(_TextTower, clip_encoder.model)
    assert model.received_mask is not None and torch.equal(model.received_mask, output.mask.long())
    assert not model.training and not model.weight.requires_grad
    projection = nn.Linear(512, 2)
    projection(output.features[output.mask]).sum().backward()
    assert projection.weight.grad is not None and torch.count_nonzero(projection.weight.grad)
    assert model.weight.grad is None


@pytest.mark.parametrize(
    ("captions", "message"),
    [([], "nonempty"), (["one " * 76], "76 content tokens"), (["one eos"], "embedded")],
)
def test_clip_rejects_invalid_content_without_calling_model(
    clip_encoder: FrozenCLIPText,
    captions: list[str],
    message: str,
) -> None:
    """确保没有有效批量、超限或内嵌特殊标记时不会运行模型。

    Args:
        clip_encoder: 能检查是否收到模型调用的测试替身。
        captions: 应拒绝的 Caption 列表。
        message: 错误中应可见的诊断文本。

    Raises:
        AssertionError: 模型被调用，或异常与预期不符。
    """
    with pytest.raises(ValueError, match=message):
        clip_encoder.encode(captions)
    assert cast(_TextTower, clip_encoder.model).received_mask is None


def test_clip_lock_rejects_untracked_or_modified_assets(
    tmp_path: Path,
) -> None:
    """核对合法锁后，对同尺寸篡改和未登记文件要求显式失败。

    Args:
        tmp_path: 可写的 pytest 临时根目录。

    Raises:
        AssertionError: 锁验证错误接受或返回不一致。
        OSError: 测试文件无法读取或写入。

    副作用:
        在临时根目录中创建并修改模拟模型文件和锁。
    """
    model_dir = tmp_path / "clip"
    model_dir.mkdir()
    locked_files = (
        "config.json",
        "model.safetensors",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    files = {}
    for name in locked_files:
        payload = name.encode()
        (model_dir / name).write_bytes(payload)
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "repo_id": "openai/clip-vit-base-patch32",
                "revision": "1" * 40,
                "local_path": "clip",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    verified, identity = _verified_directory(tmp_path, lock)
    assert verified == model_dir.resolve() and identity["files"] == files
    extra = model_dir / "added_tokens.json"
    extra.write_bytes(b"{}")
    with pytest.raises(ValueError, match="Untracked CLIP component files"):
        _verified_directory(tmp_path, lock)
    extra.unlink()
    (model_dir / "vocab.json").write_bytes(b"x" * len("vocab.json"))
    with pytest.raises(ValueError, match="SHA-256 mismatch: vocab.json"):
        _verified_directory(tmp_path, lock)
