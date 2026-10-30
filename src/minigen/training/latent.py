"""显式组合个人 DDPM 与外部 DiT，冻结数据、监督和恢复身份。"""

import hashlib
import inspect
import json
import marshal
from pathlib import Path

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from torch.utils.data import TensorDataset

from minigen._typing import JsonDict
from minigen.data.latent import identity_digest
from minigen.data.stateful import StatefulBatchStream
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.training.denoising import TrainConfig, save_checkpoint, train_denoiser


def tensor_digest(
    tensor: Shaped[Tensor, "..."],
) -> str:
    """把张量的精确 dtype、形状和连续 CPU 字节绑定为一个摘要。

    Args:
        tensor: 浮点或整数张量，支持 NumPy 表示的 dtype；不修改输入。

    Raises:
        TypeError/RuntimeError: dtype 不支持 NumPy，或张量不能复制到 CPU 时透传。

    Results:
        str: 形状、dtype 和数据字节共同构成的 SHA-256。
    """
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_identity(
    model: nn.Module,
    declared: JsonDict,
) -> JsonDict:
    """同时冻结显式源码来源和实例实际执行代码、结构，排除初始权重值。

    Args:
        model: 调用者显式传入的噪声预测模块，不读取或替换个人 DiT 核心。
        declared: 必须包含 source、source_sha256、architecture 的 JSON 来源说明。

    Raises:
        ValueError: 缺失来源、源码 SHA-256 或架构说明。
        TypeError: 元数据不能 JSON 序列化时透传。

    Results:
        JsonDict: 原声明、实际模块类、repr、状态形状/dtype 和 Python 方法代码摘要。
            source_sha256: 调用方核验的完整源码单元摘要。
            executed_methods: 实例实际方法代码对象的摘要，弥补动态 exec 无法 inspect 源码的限制。
    """
    source_hash = declared.get("source_sha256", "")
    if (
        not declared.get("source")
        or not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(c not in "0123456789abcdef" for c in source_hash)
        or not isinstance(declared.get("architecture"), dict)
    ):
        raise ValueError("Model source, source_sha256 and architecture must be explicit")
    methods = {}
    for module in model.modules():
        cls = type(module)
        for name, method in vars(cls).items():
            if inspect.isfunction(method):
                key = f"{cls.__module__}.{cls.__qualname__}.{name}"
                methods[key] = hashlib.sha256(marshal.dumps(method.__code__)).hexdigest()
    return {
        "declared": json.loads(json.dumps(declared)),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "structure": repr(model),
        "states": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        },
        "executed_methods": methods,
    }


def _supervision_identity(
    values: tuple[Float[Tensor, "b c h w"], Float[Tensor, "b c h w"], Int[Tensor, " b"]] | None,
) -> JsonDict:
    """区分重新抽样的训练与固定图像、噪声、时间监督。

    Args:
        values: None 表示随机时间/噪声；否则为明确的 clean、noise、times 三元组。

    Results:
        JsonDict: 随机模式标记或三个实际张量的摘要，不能把固定拟合的恢复用于随机训练。
    """
    if values is None:
        return {"mode": "random_time_and_noise"}
    return {
        "mode": "fixed_supervision",
        "clean": tensor_digest(values[0]),
        "noise": tensor_digest(values[1]),
        "times": tensor_digest(values[2]),
    }


def train_latent_denoiser(
    model: nn.Module,
    scheduler: DDPMScheduler,
    latents: Float[Tensor, "n_s c h w"],
    config: TrainConfig,
    *,
    data_identity: JsonDict,
    model_identity: JsonDict,
    batch_size: int = 16,
    data_seed: int = 0,
    probe: tuple[Float[Tensor, "b c h w"], Float[Tensor, "b c h w"], Int[Tensor, " b"]]
    | None = None,
    fixed_batch: tuple[Float[Tensor, "b c h w"], Float[Tensor, "b c h w"], Int[Tensor, " b"]]
    | None = None,
    resume_path: str | None = None,
) -> JsonDict:
    """冻结恢复契约后调用既有训练器，显式保留模型与个人 DDPM 来源。

    Args:
        model: 已显式构造的噪声预测网络，不在工程中实现或自动补全 DiT。
        scheduler: 用户已完成的 DDPMScheduler，完整日程与裁剪开关进入契约。
        latents: 非空 CPU FP32 [n_s,c,h,w] 缓存或明确标识的训练子集。
        config: 既有训练配置，scheduler_source 必须为 project_core。
        data_identity: 原缓存身份及选取子集说明，另绑定实际输入张量摘要。
        model_identity: source、source_sha256、architecture 及额外来源信息。
        batch_size: StatefulBatchStream 的正批量大小。
        data_seed: 数据排列专属种子，与训练器随机噪声种子分别记录。
        probe: 可选固定探针，三元组及实际数值进入恢复契约。
        fixed_batch: 可选固定监督；与随机训练分开登记和拒绝交叉恢复。
        resume_path: 已有 Checkpoint，恢复前校验源码、日程、数据和监督身份。

    Raises:
        ValueError: 输入不是有限 CPU FP32 四维张量、来源不明确、现有目录或恢复契约不符。
        OSError/KeyError: 文件缺失、不可读或 Checkpoint 缺必要状态；训练异常透传。

    Results:
        JsonDict: 原训练摘要，以及 contract_path、contract_sha256、model_source、data_id。
            resume_mode: stateful_exact，仅保证同一确定数据、设备与软件环境。
            contract: 明确冻结的日程、实际数据、模型来源/结构和固定或随机监督。

    副作用:
        写恢复契约、训练日志与 Checkpoint，更新模型、优化器、EMA 及进程随机状态。
        使用既有训练器执行受限训练；不下载数据或模型，不执行反向生成。
    """
    if (
        latents.ndim != 4
        or len(latents) == 0
        or latents.dtype != torch.float32
        or latents.device.type != "cpu"
        or not torch.isfinite(latents).all()
    ):
        raise ValueError("Expected nonempty finite CPU FP32 latent data [n_s,c,h,w]")
    if config.scheduler_source != "project_core" or not isinstance(scheduler, DDPMScheduler):
        raise ValueError("Personal DDPM must be explicitly recorded as project_core")
    contract = {
        "schema_version": 1,
        "data_identity": json.loads(json.dumps(data_identity)),
        "data_tensor_sha256": tensor_digest(latents),
        "model": _model_identity(model, model_identity),
        "scheduler": {
            "source": "project_core",
            "num_steps": scheduler.num_steps,
            "betas": scheduler.betas.detach().cpu().tolist(),
            "clip_denoised": scheduler.clip_denoised,
            "executed_methods": {
                name: hashlib.sha256(marshal.dumps(getattr(scheduler, name).__code__)).hexdigest()
                for name in ("__init__", "_extract", "add_noise", "loss", "_posterior", "step")
            },
            "buffers": {
                name: tensor_digest(value) for name, value in scheduler.state_dict().items()
            },
        },
        "batch_size": batch_size,
        "data_seed": data_seed,
        "supervision": _supervision_identity(fixed_batch),
        "probe": _supervision_identity(probe),
        "torch_version": str(torch.__version__),
    }
    contract_hash = identity_digest(contract)
    data_id = f"latent_training:{contract_hash}"
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        previous = saved.get("latent_contract")
        if previous is None:
            sidecar = Path(resume_path).parent / "latent_contract.json"
            if not sidecar.exists():
                raise ValueError("Missing latent training contract")
            previous = json.loads(sidecar.read_text())
        if previous != contract or (saved.get("data_state") or {}).get("data_id") != data_id:
            raise ValueError("Latent resume contract changed: model, schedule, data or supervision")
    directory = Path(config.run_dir)
    path = directory / "latent_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Existing run directory has a different latent training contract")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    loader = StatefulBatchStream(
        TensorDataset(latents), batch_size, data_id=data_id, seed=data_seed, shuffle=True
    )
    result = train_denoiser(
        model,
        scheduler,
        loader,
        config,
        probe=probe,
        fixed_batch=fixed_batch,
        resume_path=resume_path,
    )
    # 周期性 Checkpoint 已通过 data_state.data_id 绑定完整契约；最终文件再嵌入可移植元数据。
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    saved["latent_contract"] = contract
    save_checkpoint(Path(result["checkpoint"]), saved)
    result.update(
        {
            "contract_path": str(path),
            "contract_sha256": contract_hash,
            "contract": contract,
            "data_id": data_id,
            "model_source": model_identity["source"],
        }
    )
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
