"""L04 个人 CFG 数值与梯度验收；不会加载 Notebook 参考。"""

import json

import torch

from minigen._learning import CoreImplementationRequired
from minigen.samplers.cfg import classifier_free_guidance


def main() -> int:
    """仅调用个人 CFG，检查两种边界、外推与梯度。

    Raises:
        AssertionError: 输出或梯度不符；其他个人实现异常透传。

    Results:
        int: 验证成功返回 0；个人核心待做返回 2，不自动使用参考。

    副作用:
        打印真实状态；待实现异常转换为 JSON 和退出码 2。
    """
    u = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([3.0, 1.0], dtype=torch.float64, requires_grad=True)
    try:
        zero = classifier_free_guidance(u, c, 0)
    except CoreImplementationRequired as error:
        print(
            json.dumps(
                {"status": "待做", "source": "project_core", "reason": str(error)},
                ensure_ascii=False,
            )
        )
        return 2
    torch.testing.assert_close(zero, u)
    torch.testing.assert_close(classifier_free_guidance(u, c, 1), c)
    torch.testing.assert_close(classifier_free_guidance(u, c, 3), u.new_tensor([7, -1]))
    grad_u, grad_c = torch.autograd.grad(classifier_free_guidance(u, c, 3).sum(), (u, c))
    torch.testing.assert_close(grad_u, torch.full_like(u, -2))
    torch.testing.assert_close(grad_c, torch.full_like(c, 3))
    print(json.dumps({"status": "已验证", "source": "project_core"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
