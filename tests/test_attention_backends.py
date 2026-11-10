"""显式注意力与 SDPA 的数值、梯度及三模型权重兼容性。"""

import copy

import pytest
import torch

from minigen.contracts import Condition
from minigen.models.attention import attention, set_attention_backend
from minigen.training.config import build_model


def test_attention_forward_gradient_and_all_masked() -> None:
    """比较自/交叉形状、padding 和全屏蔽行的前向及梯度。

    Raises:
        AssertionError: 与独立 SDPA 路径不一致或全屏蔽行非零。

    副作用:
        使用固定 RNG 创建张量并计算局部梯度。
    """
    rng = torch.Generator().manual_seed(192)
    tensors = [
        torch.randn(shape, generator=rng, dtype=torch.float64)
        for shape in [(2, 3, 4, 8), (2, 3, 6, 8), (2, 3, 6, 8)]
    ]
    mask = torch.tensor([[[[True, False, True, False, True, False]]], [[[False] * 6]]])
    results, gradients = [], []
    for backend in ["sdpa", "naive"]:
        args = [x.clone().requires_grad_() for x in tensors]
        out = attention(*args, mask, backend)
        out.square().sum().backward()
        results.append(out.detach())
        gradients.append([x.grad for x in args])
    torch.testing.assert_close(results[0], results[1], atol=1e-12, rtol=1e-12)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(results[1][1]) == 0
    with pytest.raises(ValueError, match="backend"):
        attention(*tensors, mask, "unknown")


@pytest.mark.parametrize("architecture", ["b0", "b1", "b2"])
def test_backbone_runtime_backend_preserves_weights_and_padding(
    architecture: str,
) -> None:
    """非零生成头下验证三架构后端等价，保持 state_dict 和 padding 隔离。

    Args:
        architecture: B0/B1/B2 中一个主干。

    Raises:
        AssertionError: 状态键、数值、梯度或 padding 不一致。

    副作用:
        初始化小模型并进行两套后端的前向反向。
    """
    torch.manual_seed(123)
    specification = {
        "architecture": architecture,
        "hidden_size": 16,
        "num_heads": 2,
        "patch_size": 1,
        "latent_size": 2,
        "in_channels": 4,
        "text_dim": 3,
    }
    if architecture == "b2":
        specification.update(double_depth=1, single_depth=1)
    else:
        specification["depth"] = 2
    model = build_model(specification)
    # 激活输出与门控，避免在零初始化输出上得到没有意义的一致性。
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(std=0.03)
    other = copy.deepcopy(model)
    original_keys = list(other.state_dict())
    set_attention_backend(other, "naive")
    assert original_keys == list(other.state_dict())
    for a, b in zip(model.state_dict().values(), other.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    z = torch.randn(2, 4, 2, 2)
    t = torch.tensor([0.2, 0.8])
    text = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    condition = Condition(text, mask)
    expected = model(z, t, condition)
    actual = other(z, t, condition)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)
    changed = text.clone()
    changed[~mask] = 1e4
    torch.testing.assert_close(other(z, t, Condition(changed, mask)), actual, rtol=0, atol=0)
