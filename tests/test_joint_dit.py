"""B1/B2 的独立旋转、联合可见性、双流参数与完整模型回归。"""

import math

import pytest
import torch
from torch import nn

from minigen.contracts import Condition
from minigen.models.hybrid_dit import HybridDiT, HybridDiTConfig
from minigen.models.joint import (
    DoubleBlock,
    JointInputOutput,
    SingleBlock,
    StreamConfig,
    apply_rotary,
    image_rotary,
    joint_attention,
)
from minigen.models.single_dit import SingleDiTConfig, SingleStreamDiT
from minigen.objectives.flow import flow_loss


def tiny_model(
    architecture: str,
) -> nn.Module:
    """构造共享输入尺寸的微型模型，仅供快速数学测试。

    Args:
        architecture: b1 或 b2。

    Results:
        nn.Module: 32 维、4 头、4px 潜空间的 Single 或 Hybrid。
    """
    dimensions = {"hidden_size": 32, "num_heads": 4, "latent_size": 4, "text_dim": 8}
    if architecture == "b1":
        return SingleStreamDiT(SingleDiTConfig(**dimensions, depth=2))
    return HybridDiT(HybridDiTConfig(**dimensions, double_depth=1, single_depth=1))


def test_rotary_manual_coordinates_and_norm() -> None:
    """手算不同网格坐标的旋转，检查相邻配对、广播和长度保持。

    Raises:
        AssertionError: 行列分配、旋转方向或范数不符。
    """
    cosine, sine = image_rotary(2, 8)
    values = torch.arange(1.0, 9.0).expand(2, 3, 4, 8).clone()
    result = apply_rotary(values, cosine, sine)
    expected = values.clone()
    coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for token, (row, col) in enumerate(coordinates):
        # 独立标量三角式，不使用被测 apply_rotary 的分块或广播。
        for pair, angle in enumerate((row, row * 0.01, col, col * 0.01)):
            even, odd = 2 * pair + 1, 2 * pair + 2
            expected[:, :, token, 2 * pair] = even * math.cos(angle) - odd * math.sin(angle)
            expected[:, :, token, 2 * pair + 1] = even * math.sin(angle) + odd * math.cos(angle)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(result.square().sum(-1), values.square().sum(-1))
    assert not torch.equal(result[:, :, 1], result[:, :, 2])
    assert torch.equal(result[:, :, 0], values[:, :, 0])
    with pytest.raises(ValueError):
        image_rotary(2, 6)
    with pytest.raises(ValueError):
        apply_rotary(values, cosine[:2], sine[:2])
    with pytest.raises(ValueError):
        StreamConfig(hidden_size=18, num_heads=3)


def test_joint_attention_explicit_softmax_and_text_visibility() -> None:
    """独立矩阵 softmax 对照联合注意力，同时验证文本也能读取图像。

    Raises:
        AssertionError: 头重排、mask 方向、文本恒等位置或双向性不符。

    副作用:
        固定测试随机种子，初始化独立输入投影。
    """
    torch.manual_seed(41)
    config = StreamConfig(hidden_size=16, num_heads=2, latent_size=4, text_dim=8)
    interface = JointInputOutput(config)
    condition = Condition(torch.randn(2, 2, 8), torch.tensor([[True, False], [True, True]]))
    parts = interface.prepare(torch.randn(2, 4, 4, 4), torch.tensor([0.1, 0.9]), condition)
    assert torch.equal(parts.cosine[:2], torch.ones(2, 4))
    assert torch.count_nonzero(parts.sine[:2]) == 0
    assert torch.all(parts.mask[:, 2:])
    qkv = torch.randn(3, 2, 2, 6, 8)
    q, k, v = qkv.unbind(0)
    qr, kr = apply_rotary(q, parts.cosine, parts.sine), apply_rotary(k, parts.cosine, parts.sine)
    logits = (qr @ kr.transpose(-2, -1)) / math.sqrt(8)
    probabilities = logits.masked_fill(~parts.mask[:, None, None, :], -torch.inf).softmax(-1)
    expected = (probabilities @ v).transpose(1, 2).reshape(2, 6, 16)
    actual = joint_attention(qkv, parts.mask, parts.cosine, parts.sine)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    corrupted = qkv.clone()
    corrupted[:, 0, :, 1] += 100
    changed = joint_attention(corrupted, parts.mask, parts.cosine, parts.sine)
    torch.testing.assert_close(changed[0, parts.mask[0]], actual[0, parts.mask[0]], rtol=0, atol=0)
    changed_values = qkv.clone()
    changed_values[2, :, :, 2:] += 1
    text_output = joint_attention(changed_values, parts.mask, parts.cosine, parts.sine)[:, 0]
    assert (text_output - actual[:, 0]).abs().max() > 0.01


def test_double_block_independence_identity_and_cross_stream_effect() -> None:
    """检查独立 QKV/FFN、恒等初始化和图文双向影响，而不只检查拼接形状。

    Raises:
        AssertionError: 参数共享、初始化、掩码或跨流路径不符。

    副作用:
        初始化测试模块，显式打开测试残差门控。
    """
    torch.manual_seed(29)
    cfg = StreamConfig(hidden_size=16, num_heads=2, latent_size=4, text_dim=8)
    block = DoubleBlock(cfg)
    interface = JointInputOutput(cfg)
    condition = Condition(
        torch.randn(2, 3, 8), torch.tensor([[True, False, False], [True, True, False]])
    )
    parts = interface.prepare(torch.randn(2, 4, 4, 4), torch.tensor([0.2, 0.8]), condition)
    assert block.image_branch.qkv.weight.data_ptr() != block.text_branch.qkv.weight.data_ptr()
    assert block.image_branch.mlp[0].weight.data_ptr() != block.text_branch.mlp[0].weight.data_ptr()
    image, text = block(parts.image, parts.text, parts.time, parts.mask, parts.cosine, parts.sine)
    assert torch.equal(image, parts.image) and torch.equal(text, parts.text)
    with torch.no_grad():
        for branch in (block.image_branch, block.text_branch):
            branch.modulation[-1].bias[2 * cfg.hidden_size : 3 * cfg.hidden_size] = 1
    image, text = block(parts.image, parts.text, parts.time, parts.mask, parts.cosine, parts.sine)
    changed_image = parts.image.clone()
    changed_image[:, :, 0] += 3
    _, new_text = block(changed_image, parts.text, parts.time, parts.mask, parts.cosine, parts.sine)
    assert (new_text[:, 0] - text[:, 0]).abs().max() > 1e-4
    changed_text = parts.text.clone()
    changed_text[:, 0, 0] += 3
    new_image, _ = block(
        parts.image, changed_text, parts.time, parts.mask, parts.cosine, parts.sine
    )
    assert (new_image - image).abs().max() > 1e-4
    assert torch.count_nonzero(text[~condition.mask]) == 0
    single = SingleBlock(cfg)
    combined = torch.cat((parts.text, parts.image), dim=1)
    assert torch.equal(single(combined, parts.time, parts.mask, parts.cosine, parts.sine), combined)


@pytest.mark.parametrize("architecture", ["b1", "b2"])
def test_joint_models_gradient_mask_and_shape(
    architecture: str,
) -> None:
    """以小模型数次更新打开零门控，检验全路径梯度、padding 和条件作用。

    Args:
        architecture: 当前测试的单流或混合模型标识。

    Raises:
        AssertionError: 梯度、文本可见性、融合或错误输入拒绝失败。

    副作用:
        固定 RNG、优化独立模型并消耗随机数。
    """
    torch.manual_seed(73)
    model = tiny_model(architecture)
    z, target = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    times = torch.tensor([0.2, 0.7])
    features = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    condition = Condition(features, mask)
    assert torch.count_nonzero(model(z, times, condition)) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(6):
        optimizer.zero_grad()
        flow_loss(model(z, times, condition), target).backward()
        optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.interface.text_embed.weight.grad.abs().sum() > 0
    if architecture == "b2":
        assert model.double_blocks[0].text_branch.qkv.weight.grad.abs().sum() > 0
        assert model.double_blocks[0].image_branch.qkv.weight.grad.abs().sum() > 0
        assert model.single_blocks[0].qkv.weight.grad.abs().sum() > 0
    with torch.no_grad():
        reference = model(z, times, condition)
        corrupted = features.clone()
        corrupted[~mask] += 1e3
        torch.testing.assert_close(
            model(z, times, Condition(corrupted, mask)), reference, atol=0, rtol=0
        )
        corrupted = features.clone()
        corrupted[mask] += 2
        assert (model(z, times, Condition(corrupted, mask)) - reference).abs().max() > 1e-5
        padded = Condition(
            torch.cat((features, torch.randn(2, 2, 8)), dim=1),
            torch.cat((mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1),
        )
        torch.testing.assert_close(model(z, times, padded), reference, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        model(z[:, :, :2], times, condition)
    with pytest.raises(ValueError):
        model(z, torch.tensor([-0.1, 0.1]), condition)
    with pytest.raises(ValueError):
        Condition(features, torch.zeros_like(mask))


@pytest.mark.parametrize("architecture", ["b1", "b2"])
def test_joint_formal_parameter_budget_and_interface(
    architecture: str,
) -> None:
    """实例化完整约 30M 规格，检查参数和真实 256px 潜空间接口。

    Args:
        architecture: b1 或 b2。

    Raises:
        AssertionError: 实际参数预算、空间接口或输出数值不符。

    副作用:
        初始化正式模型并执行前向；不训练、不加载编码器。
    """
    model = (SingleStreamDiT() if architecture == "b1" else HybridDiT()).eval()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 30_061_456 and 27_000_000 <= count <= 33_000_000
    for b in (1, 2):
        z = torch.zeros(b, 4, 32, 32)
        condition = Condition(torch.zeros(b, 77, 512), torch.ones(b, 77, dtype=torch.bool))
        with torch.no_grad():
            result = model(z, torch.full((b,), 0.5), condition)
        assert result.shape == z.shape and result.dtype == torch.float32
        assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0
