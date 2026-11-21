"""验证 B2 对齐特征取自双流末端并可回传梯度。"""

import torch

from minigen.contracts import Condition
from minigen.models.hybrid_dit import HybridDiT, HybridDiTConfig


def test_image_token_forward_preserves_velocity_and_stops_at_stream_boundary() -> None:
    """检查额外取出图像 token 时，速度保持一致且对齐梯度只经过双流。

    Raises:
        AssertionError: 速度、token 数量或双流与单流之间的梯度边界不符。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        model = HybridDiT(
            HybridDiTConfig(
                hidden_size=32,
                num_heads=4,
                latent_size=4,
                text_dim=8,
                double_depth=2,
                single_depth=1,
            )
        )
        image = torch.randn(2, 4, 4, 4)
        times = torch.tensor([0.2, 0.7])
        condition = Condition(torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
    with torch.no_grad():
        model.interface.final_projection.weight.fill_(0.125)
        model.interface.final_projection.weight[0, 0] = 0.25

    velocity, tokens = model.forward_with_image_tokens(image, times, condition)
    assert torch.count_nonzero(velocity) > 0
    torch.testing.assert_close(velocity, model(image, times, condition), rtol=0, atol=0)
    assert tokens.shape == (2, 4, 32)

    last_double_gate = model.double_blocks[-1].image_branch.modulation[-1].bias
    first_single_qkv = model.single_blocks[0].qkv.weight
    double_grad, single_grad = torch.autograd.grad(
        tokens.square().mean(), (last_double_gate, first_single_qkv), allow_unused=True
    )
    assert double_grad is not None and torch.count_nonzero(double_grad) > 0
    assert single_grad is None
