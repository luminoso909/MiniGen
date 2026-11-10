"""主导 FLOPs 计数的手算对照与作用域恢复测试。"""

import pytest
import torch
import torch.nn.functional as F
from inspect_flow_models import inspect_model
from jaxtyping import Float
from torch import Tensor, nn

from minigen.contracts import Condition


class InspectionFixture(nn.Module):
    """两个 Linear 和一次 SDPA 的可手算速度模型。

    Args:
        fail: 是否在完成 SDPA 后主动抛异常，用于验证清理路径。
    """

    def __init__(
        self,
        fail: bool = False,
    ) -> None:
        """建立明确的一维输入、四维注意力及一维输出。

        Args:
            fail: 异常路径开关。

        副作用:
            注册参数和异常标记，初始化 Linear 时消耗 RNG。
        """
        super().__init__()
        self.input = nn.Linear(1, 4)
        self.output = nn.Linear(4, 1)
        self.fail = fail

    def forward(
        self,
        latents: Float[Tensor, "b 1 h w"],
        times: Float[Tensor, " b"],
        condition: Condition,
    ) -> Float[Tensor, "b 1 h w"]:
        """切换到 token 布局后执行可精确计数的前向。

        Args:
            latents: [b,1,h,w] 测试图像。
            times: [b] 接口占位，计算不使用。
            condition: [b,m,4] 文本及掩码。

        Raises:
            RuntimeError: fail=True 时人为触发异常。

        Results:
            Float[Tensor, "b 1 h w"]: 经过注意力和投影的同形测试输出。
        """
        query = self.input(latents.flatten(2).transpose(1, 2))[:, None]
        context = condition.features[:, None]
        attended = F.scaled_dot_product_attention(query, context, context)
        if self.fail:
            raise RuntimeError("intentional fixture failure")
        return self.output(attended[:, 0]).transpose(1, 2).reshape_as(latents)


def test_dominant_flops_match_hand_calculation() -> None:
    """两批四个查询、三个键、四维特征的主导 FLOPs 应恰好为 512。

    Raises:
        AssertionError: 参数、Linear 或 QK/PV 计数与手算不一致。

    副作用:
        初始化测试模型及临时测量 hooks。
    """
    model = InspectionFixture()
    condition = Condition(torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    report = inspect_model(model, torch.ones(2, 1, 2, 2), torch.zeros(2), condition)
    # 输入 Linear: 2*2*4*4*1=64；输出 Linear: 2*2*4*1*4=64。
    assert report["trainable_parameters"] == 13
    assert report["linear_call_count"] == 2
    assert report["linear_flops"] == 128
    # 两批单头各有 4*3=12 个注意力元素；QK/PV 各 2*24*4=192。
    assert report["attention_call_count"] == 1
    assert report["dense_attention_matrix_elements"] == 24
    assert report["attention_flops"] == 384
    assert report["dominant_forward_flops"] == 512
    assert report["output_shape"] == [2, 1, 2, 2]


def test_inspection_restores_hooks_sdpa_and_mixed_modes_on_failure() -> None:
    """前向失败后仍还原 SDPA、hooks 及父子模块不同的 train/eval 状态。

    Raises:
        AssertionError: 作用域退出后残留 patch、hook 或模式改变。

    副作用:
        初始化故意失败的模型，调用测量并验证清理。
    """
    original = F.scaled_dot_product_attention
    model = InspectionFixture(fail=True).train()
    model.input.eval()
    condition = Condition(torch.ones(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="intentional"):
        inspect_model(model, torch.ones(2, 1, 2, 2), torch.zeros(2), condition)
    assert F.scaled_dot_product_attention is original
    assert model.training and not model.input.training and model.output.training
    assert all(not module._forward_hooks for module in model.modules())
