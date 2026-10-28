"""个人 DiT 数学验收，不提供教学 fallback。"""

import json
from collections.abc import Callable

import torch
from torch import nn

from minigen._learning import CoreImplementationRequired
from minigen._typing import JsonDict, PatchifyFn, UnpatchifyFn
from minigen.models.dit_block import DiTBlock, patchify, unpatchify


def verify(
    patchify_fn: PatchifyFn, unpatchify_fn: UnpatchifyFn, block_factory: Callable[..., nn.Module]
) -> JsonDict:
    """以手排 patch 和门控梯度验证显式传入的 DiT 实现。

    1 输入：
        patchify_fn：[B,C,H,W] 与 p 到 [B,N,p²C] 的切块函数。
        unpatchify_fn：接收 token、C、H、W、p 并恢复空间的函数。
        block_factory：接受 hidden_size、num_heads 的构造器；forward 接收 [B,N,D] 与 [B,D]。

    2 输出：
        JSON 验证摘要，包含 patch 顺序、恒等初始化及更新后的条件差异。

    3 报错（Raises）：
        AssertionError：顺序、边界报错、恒等/梯度或条件作用不符合契约；传入实现的异常透传。
    """
    image = torch.arange(16).float().reshape(1, 2, 2, 4)
    expected = torch.tensor([[[0, 8, 1, 9, 4, 12, 5, 13], [2, 10, 3, 11, 6, 14, 7, 15]]]).float()
    actual = patchify_fn(image, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(unpatchify_fn(actual, 2, 2, 4, 2), image)
    try:
        patchify_fn(torch.zeros(1, 2, 3, 4), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-divisible dimensions must raise ValueError")
    torch.manual_seed(19)
    block = block_factory(hidden_size=16, num_heads=4)
    inputs = torch.randn(2, 5, 16, requires_grad=True)
    condition = torch.randn(2, 16)
    output = block(inputs, condition)
    torch.testing.assert_close(output, inputs, rtol=0, atol=0)
    output.square().mean().backward()
    torch.testing.assert_close(inputs.grad, 2 * inputs / inputs.numel())
    norms = [p.grad.norm().item() for p in block.parameters() if p.grad is not None]
    assert any(n > 0 for n in norms), "Zero initialization must still allow a gate gradient"
    assert any(n == 0 for n in norms), "Expected gated branch gradients initially zero"
    optimizer = torch.optim.SGD(block.parameters(), lr=0.02)
    target = torch.randn_like(inputs)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (block(inputs.detach(), condition) - target).square().mean().backward()
        optimizer.step()
    with torch.no_grad():
        difference = (block(inputs, condition) - block(inputs, -condition)).abs().max().item()
    assert difference > 1e-7, "Condition should affect the trained modulation"
    return {"patch_order": "passed", "identity": "exact", "condition_difference": difference}


def main() -> int:
    """只连接项目自己的 L03 接口与独立验收，不执行参考 fallback。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        退出码：待实现为 2，全部验收通过为 0，控制台打印 JSON。

    3 报错（Raises）：
        CoreImplementationRequired 被捕获为待做；AssertionError 或其他实现错误保留以便定位。
    """
    try:
        result = verify(patchify, unpatchify, DiTBlock)
    except CoreImplementationRequired as error:
        print(
            json.dumps(
                {"status": "待做", "source": "project_core", "reason": str(error)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps({"status": "已验证", "source": "project_core", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
