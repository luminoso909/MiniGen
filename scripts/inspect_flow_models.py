"""统计真实模型前向中的参数与主导矩阵乘法成本，不替代硬件性能测量。"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from minigen._typing import JsonDict
from minigen.contracts import Condition
from minigen.training.config import build_model


def inspect_model(
    model: nn.Module,
    latents: Float[Tensor, "b c h w"],
    times: Float[Tensor, " b"],
    condition: Condition,
) -> JsonDict:
    """观察一次 eval 前向，统计实际执行的 Linear 与 SDPA 主导 FLOPs。

    Args:
        model: 遵循统一速度接口的模型；不修改参数或参数梯度。
        latents: [b,c,h,w] 输入潜变量。
        times: [b] 连续时间。
        condition: [b,n,d] 文本特征及掩码。

    Raises:
        RuntimeError/ValueError: 模型前向或 SDPA 输入不合法时透传。

    Results:
        JsonDict: 参数数量、调用明细与主导矩阵乘法 FLOPs。
            linear_flops: 每次 Linear 按 2*输出元素数*输入宽度计数，不含 bias 加法。
            attention_flops: 每次 SDPA 的 QK 和 PV 密集矩阵乘法，不含 softmax 等操作。
            dominant_forward_flops: 两项之和，不是完整模型 FLOPs 或硬件实测吞吐。

    副作用:
        暂时安装前向 hooks、替换本进程 SDPA 并切换 eval；包括异常路径均还原。
    """
    names = {id(module): name for name, module in model.named_modules()}
    modes = [(module, module.training) for module in model.modules()]
    linear_calls: list[JsonDict] = []
    attention_calls: list[JsonDict] = []
    original_sdpa = F.scaled_dot_product_attention

    def count_linear(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Float[Tensor, "..."],
    ) -> None:
        """按实际 Linear 输出元素数记录一次矩阵乘法。

        Args:
            module: 仅由 nn.Linear 注册的 hook 调用方。
            inputs: 原始调用参数，不影响计数。
            output: 任意前缀形状的 Linear 浮点输出。

        副作用:
            向本次测量的 Linear 明细列表追加记录。
        """
        linear_calls.append(
            {
                "module": names[id(module)],
                "input_features": module.in_features,
                "output_shape": list(output.shape),
                "flops": 2 * output.numel() * module.in_features,
            }
        )

    def count_sdpa(
        query: Float[Tensor, "..."],
        key: Float[Tensor, "..."],
        value: Float[Tensor, "..."],
        *args: object,
        **kwargs: object,
    ) -> Float[Tensor, "..."]:
        """透传 SDPA，并按成功前向的真实批/头前缀计算 QK 与 PV 成本。

        Args:
            query: 任意合法 SDPA 前缀和 [n,d] 后缀的查询。
            key: 任意合法前缀和 [m,d] 后缀的键。
            value: 任意合法前缀和 [m,d_v] 后缀的值。
            args: 原样透传的位置参数，例如掩码和 dropout。
            kwargs: 原样透传的关键字参数，例如 is_causal、scale、enable_gqa。

        Raises:
            RuntimeError/ValueError/TypeError: 底层 SDPA 拒绝非法输入时透传。

        Results:
            Float[Tensor, "..."]: 原始 SDPA 输出，没有插入额外计算变换。

        副作用:
            向本次测量的注意力明细追加记录；底层 dropout 按调用约定消耗 RNG。
        """
        output = original_sdpa(query, key, value, *args, **kwargs)
        # 采用真实输出前缀，兼容广播和 GQA；理论密集矩阵不扣除 padding/因果掩码。
        matrices = math.prod(output.shape[:-2])
        elements = matrices * query.shape[-2] * key.shape[-2]
        qk_flops = 2 * elements * query.shape[-1]
        pv_flops = 2 * elements * value.shape[-1]
        attention_calls.append(
            {
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "output_shape": list(output.shape),
                "query_tokens": query.shape[-2],
                "key_tokens": key.shape[-2],
                "dense_attention_matrix_elements": elements,
                "qk_flops": qk_flops,
                "pv_flops": pv_flops,
                "flops": qk_flops + pv_flops,
            }
        )
        return output

    handles = []
    try:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(count_linear))
        model.eval()
        with torch.no_grad(), patch.object(F, "scaled_dot_product_attention", count_sdpa):
            output = model(latents, times, condition)
    finally:
        for handle in handles:
            handle.remove()
        # 逐模块恢复，保留父模块 train、个别子模块 eval 的混合状态。
        for module, training in modes:
            module.training = training
    linear_flops = sum(call["flops"] for call in linear_calls)
    attention_flops = sum(call["flops"] for call in attention_calls)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(latents.device),
        "dtype": str(latents.dtype),
        "latent_shape": list(latents.shape),
        "text_shape": list(condition.features.shape),
        "output_shape": list(output.shape),
        "linear_calls": linear_calls,
        "attention_calls": attention_calls,
        "linear_call_count": len(linear_calls),
        "attention_call_count": len(attention_calls),
        "dense_attention_matrix_elements": sum(
            call["dense_attention_matrix_elements"] for call in attention_calls
        ),
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "dominant_forward_flops": linear_flops + attention_flops,
        "scope": (
            "一次前向的 Linear 与 SDPA QK/PV 密集矩阵乘法，乘加计 2 FLOPs。"
            "不含 bias、Norm、激活、softmax、RoPE、VAE/CLIP、反向、优化器或 CFG 额外分支。"
            "不是完整硬件 FLOPs/吞吐，不代表同参数即同计算预算。"
        ),
    }


def main() -> int:
    """通过真实模型工厂测量多个配置，统一使用 CPU 的正式输入形状。

    Raises:
        SystemExit: 命令参数无效。
        OSError/ValueError/TypeError/RuntimeError: 配置读取、模型构建、前向或结果写入失败。

    Results:
        int: 全部模型测量并写入完成时为 0。

    副作用:
        设置线程数和随机种子，构造模型并执行 CPU 前向，写入 JSON 报告。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=5)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    reports = []
    for config_path in args.configs:
        torch.manual_seed(0)
        content = config_path.read_bytes()
        specification = json.loads(content)
        model = build_model(specification)
        condition = Condition(torch.zeros(1, 77, 512), torch.ones(1, 77, dtype=torch.bool))
        result = inspect_model(model, torch.zeros(1, 4, 32, 32), torch.zeros(1), condition)
        result.update(
            {
                "config_file": str(config_path),
                "config_sha256": hashlib.sha256(content).hexdigest(),
                "specification": specification,
                "resolved_model_config": asdict(model.config),
            }
        )
        reports.append(result)
    report = {"torch": str(torch.__version__), "batch_size": 1, "models": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
