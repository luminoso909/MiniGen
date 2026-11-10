"""P3 B0 的空间排列、零初始化、梯度、文本可见性与正式参数量验证。"""

import pytest
import torch

from minigen.contracts import Condition
from minigen.models.plain_dit import (
    Attention,
    PlainBlock,
    PlainConditionalDiT,
    PlainDiTConfig,
    patchify,
    unpatchify,
)
from minigen.objectives.flow import flow_loss


def tiny_config() -> PlainDiTConfig:
    """提供只用于快速数学回归的微型规格，不替代正式 30M 验收。

    Results:
        PlainDiTConfig: 4px 潜空间、32 维、2 层、8 维文本的测试规格。
    """
    return PlainDiTConfig(hidden_size=32, num_heads=4, depth=2, latent_size=4, text_dim=8)


def test_patch_order_and_shape_rejection() -> None:
    """用手工列举验证行/列/通道排列，防止错误排列自洽地往返通过。

    Raises:
        AssertionError: 手工空间顺序、逆变换或非法规格检查不正确。
    """
    image = torch.arange(16.0).reshape(1, 2, 2, 4)
    tokens = patchify(image, 2)
    assert tokens.tolist() == [[[0, 8, 1, 9, 4, 12, 5, 13], [2, 10, 3, 11, 6, 14, 7, 15]]]
    assert torch.equal(unpatchify(tokens, 2, 2, 4, 2), image)
    with pytest.raises(ValueError):
        patchify(image, 3)
    with pytest.raises(ValueError):
        unpatchify(tokens, 2, 3, 4, 2)
    with pytest.raises(ValueError):
        PlainDiTConfig(hidden_size=30)


def test_block_identity_and_delayed_branch_gradient() -> None:
    """确认零门控的恒等映射、首步门控梯度及更新后分支梯度。

    Raises:
        AssertionError: 恒等初始化或 adaLN 的预期梯度路径不成立。

    副作用:
        固定并消耗测试 RNG，更新独立测试 Block 参数。
    """
    torch.manual_seed(19)
    block = PlainBlock(tiny_config())
    x = torch.randn(2, 4, 32)
    time = torch.randn(2, 32)
    cond = Condition(torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(block(x, time, cond), x)
    loss = block(x, time, cond).square().mean()
    loss.backward()
    assert block.modulation[-1].weight.grad.abs().sum() > 0
    assert block.cross_attention.q.weight.grad.abs().sum() == 0
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    block(x, time, cond).square().mean().backward()
    assert block.cross_attention.q.weight.grad.abs().sum() > 0
    assert block.self_attention.q.weight.grad.abs().sum() > 0


def test_attention_matches_explicit_masked_softmax() -> None:
    """将 SDPA 输出对照独立的矩阵乘法与 masked softmax。

    Raises:
        AssertionError: SDPA 布尔掩码方向、缩放或头重排不符。

    副作用:
        固定并消耗测试 RNG，初始化独立注意力模块。
    """
    torch.manual_seed(8)
    attention = Attention(8, 2)
    query, context = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    q = attention.q(query).reshape(2, 3, 2, 4).permute(0, 2, 1, 3)
    kv = attention.kv(context).reshape(2, 4, 2, 2, 4)
    k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
    logits = (q @ k.transpose(-1, -2)) / 2
    probabilities = logits.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1)
    expected = attention.out((probabilities @ v).transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(attention(query, context, mask), expected, rtol=1e-5, atol=1e-6)


def test_model_text_mask_and_gradient_path() -> None:
    """更新小模型后独立检查可见文本作用、padding 不变性及全路径梯度。

    Raises:
        AssertionError: 条件链路、mask 或合法输入检查失败。

    副作用:
        固定并消耗测试 RNG，并优化独立小模型。
    """
    torch.manual_seed(47)
    model = PlainConditionalDiT(tiny_config())
    x = torch.randn(2, 4, 4, 4)
    t = torch.tensor([0.1, 0.8])
    features = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    cond = Condition(features, mask)
    assert torch.count_nonzero(model(x, t, cond)) == 0
    target = torch.randn_like(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        flow_loss(model(x, t, cond), target).backward()
        optimizer.step()
    assert model.blocks[0].cross_attention.q.weight.grad.abs().sum() > 0
    assert model.text_embed.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        base = model(x, t, cond)
        altered_padding = features.clone()
        altered_padding[~mask] += 100
        torch.testing.assert_close(
            model(x, t, Condition(altered_padding, mask)), base, rtol=0, atol=0
        )
        altered_valid = features.clone()
        altered_valid[mask] += 2
        assert (model(x, t, Condition(altered_valid, mask)) - base).abs().max() > 1e-5
    with pytest.raises(ValueError, match="time"):
        model(x, torch.tensor([-0.1, 0.2]), cond)
    with pytest.raises(ValueError):
        model(x[:, :, :2], t, cond)


def test_formal_shape_and_parameter_count() -> None:
    """核对完整 30M B0 的真实参数数目和 batch=1/2 的 256px 潜空间接口。

    Raises:
        AssertionError: 参数预算或正式形状不符合约定。

    副作用:
        初始化正式网络并消耗测试 RNG；不训练或加载冻结权重。
    """
    model = PlainConditionalDiT().eval()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 30_359_824
    assert 27_000_000 <= count <= 33_000_000
    for b in (1, 2):
        z = torch.zeros(b, 4, 32, 32)
        cond = Condition(torch.zeros(b, 77, 512), torch.ones(b, 77, dtype=torch.bool))
        with torch.no_grad():
            result = model(z, torch.full((b,), 0.5), cond)
        assert result.shape == z.shape and result.dtype == z.dtype
        assert torch.count_nonzero(result) == 0
