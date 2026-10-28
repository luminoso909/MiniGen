"""个人 VAE 数学验收，不导入 Notebook 教学参考。"""

import json

import torch

from minigen._learning import CoreImplementationRequired
from minigen.autoencoders.vae_core import kl_standard_normal, reparameterize


def main() -> int:
    """仅验收个人 VAE 核心的数值、归一化和梯度，不读取教学答案。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        退出码：未实现输出待做并返回 2；数学和梯度通过返回 0。

    3 报错（Raises）：
        AssertionError：数值或梯度不符；CoreImplementationRequired 被捕获为待做，其他核心异常透传。
    """
    mean = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([[0.5, -0.5], [0.0, 1.0]], dtype=torch.float64)
    try:
        actual = reparameterize(mean, logvar, noise)
        actual_kl = kl_standard_normal(mean, logvar)
    except CoreImplementationRequired as error:
        print(
            json.dumps(
                {"status": "待做", "source": "project_core", "reason": str(error)},
                ensure_ascii=False,
            )
        )
        return 2
    distribution = torch.distributions.Normal(mean, (logvar / 2).exp())
    prior = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    expected = torch.distributions.kl_divergence(distribution, prior).sum(1).mean()
    torch.testing.assert_close(actual_kl, expected)
    torch.testing.assert_close(actual, mean + distribution.scale * noise)
    torch.testing.assert_close(kl_standard_normal(mean.repeat(3, 1), logvar.repeat(3, 1)), expected)
    assert kl_standard_normal(torch.zeros_like(mean), torch.zeros_like(logvar)) == 0
    assert torch.autograd.gradcheck(reparameterize, (mean, logvar, noise))
    assert torch.autograd.gradcheck(kl_standard_normal, (mean, logvar))
    print(
        json.dumps(
            {"status": "已验证", "source": "project_core", "kl": actual_kl.item()},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
