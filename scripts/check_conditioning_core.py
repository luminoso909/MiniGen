"""L04 个人 Cross-Attention 独立数值、mask 与梯度验收。"""

import json

import torch
from _learning_overlay import enable_learning_imports
from torch.nn import functional as F

# 历史脚本先启用显式课件路径，再加载旧模块名。
enable_learning_imports()

from minigen._learning import CoreImplementationRequired  # noqa: E402
from minigen.conditioning.cross_attention import TextCrossAttention  # noqa: E402


def main() -> int:
    """只检查个人实现，与 Torch SDPA 独立对照，不读取 Notebook。

    Raises:
        AssertionError: 数值、padding 不变性或梯度不符合约定。
        AttributeError/其他异常: 个人实现缺少规定 q/kv/out 投影或下层失败时透传。

    Results:
        int: 全部通过为 0；未实现为 2。

    副作用:
        设置测试种子、执行小型前反向并打印结果；待实现异常转为待做状态。
    """
    torch.manual_seed(5)
    model = TextCrossAttention(hidden=8, text_width=6, heads=2).double()
    image = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    text = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    try:
        actual = model(image, text, mask)
    except CoreImplementationRequired as error:
        print(
            json.dumps(
                {"status": "待做", "source": "project_core", "reason": str(error)},
                ensure_ascii=False,
            )
        )
        return 2
    q = model.q(image).reshape(2, 3, 2, 4).transpose(1, 2)
    k, v = model.kv(text).reshape(2, 5, 2, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    expected = model.out(
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        .transpose(1, 2)
        .reshape(2, 3, 8)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    changed = text.detach().clone()
    changed[~mask] += 1000
    torch.testing.assert_close(model(image, changed, mask), actual)
    actual.square().sum().backward()
    assert text.grad[~mask].abs().sum() == 0
    assert image.grad.abs().sum() > 0 and text.grad[mask].abs().sum() > 0
    print(json.dumps({"status": "已验证", "source": "project_core"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
