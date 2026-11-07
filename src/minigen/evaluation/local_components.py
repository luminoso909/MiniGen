"""在 Mac 上验证 ArtBench 冻结编码；以 CPU FP32 为数值参考。"""

import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import torch
from jaxtyping import Float
from torch import Tensor, nn

from minigen._typing import JsonDict
from minigen.autoencoders.sdxl import FrozenVAE, hash_file, image_batch
from minigen.conditioning.clip import CLIPCondition, FrozenCLIPText
from minigen.data.representation_cache import CacheSource
from minigen.evaluation.reconstruction import ReconstructionSample, load_image

SAMPLE_COUNT = 20
BATCH_SIZE = 4
CPU_REFERENCE_SHA256 = "7be7d345137ec0ab303c73a0d6d1268551365152bb136d56093a2bd7e37617b2"
# 预先固定最差单样本误差上界；都在量化前 FP32 下测量。
VAE_INVARIANTS = {
    "normalization_max_abs": 2e-7,
    "mode_mean_max_abs": 0.0,
    "forward_scale_max_abs": 0.0,
    "inverse_scale_max_abs": 2e-5,
    "independent_decode_max_abs": 2e-4,
    "repeat_latent_max_abs": 1e-6,
    "repeat_decode_max_abs": 1e-6,
}
VAE_CPU_AUDIT_LIMITS = {"audit_decode_max_abs": 2e-3, "audit_decode_mean_abs": 1e-4}
VAE_PARITY_LIMITS = {
    "latent_max_abs": 0.035,
    "latent_mean_abs": 0.002,
    "decode_max_abs": 0.10,
    "decode_mean_abs": 0.003,
}
CLIP_INVARIANTS = {"repeat_features_max_abs": 1e-6}
CLIP_PARITY_LIMITS = {"features_max_abs": 0.08, "features_mean_abs": 0.005}


def _identity(
    source: CacheSource,
    evidence_path: Path,
) -> tuple[JsonDict, JsonDict]:
    """读取已固定的 200 张证据并派生 20 张回执绑定的身份。

    Args:
        source: 正式缓存的数据与组件版本。
        evidence_path: 被用户批准的原始 200 张结构化证据。

    Raises:
        OSError/ValueError/KeyError: 证据无法读取、解析失败或身份和随机队列无效。

    Results:
        tuple[JsonDict, JsonDict]: 身份字段及原始证据。
    """
    data = evidence_path.read_bytes()
    evidence = json.loads(data)
    ids = evidence["random_sample_ids"]
    model = source.identity["vae"]
    audit = evidence["reconstruction"]
    if (
        evidence["manifest_sha256"] != source.identity["manifest_sha256"]
        or audit["count_random"] != 200
        or len(ids) != 200
        or len(set(ids)) != 200
        or audit["scale"] != model["scaling_factor"]
        or audit["latent_shape"] != [4, 32, 32]
        or any(audit["model"][key] != model[key] for key in ("revision", "files"))
    ):
        raise ValueError("Local parity requires the same frozen data and VAE as the review")
    return {
        "manifest_sha256": source.identity["manifest_sha256"],
        "reconstruction_sha256": hashlib.sha256(data).hexdigest(),
        "vae_lock_sha256": model["lock_sha256"],
        "clip_lock_sha256": source.identity["clip"]["lock_sha256"],
    }, evidence


def _audit_reference(
    directory: Path,
    evidence: JsonDict,
) -> Float[Tensor, "b c h w"]:
    """核对原始 CPU FP32 重建审计文件，提取固定顺序的前 20 个原始输出。

    Args:
        directory: 原 200 张及压力样本实验产物目录。
        evidence: 已绑定固定数据的 200 张结构化证据。

    Raises:
        OSError/ValueError/RuntimeError: CPU 参考不可读、身份或数值不符。

    Results:
        Float[Tensor, "b c h w"]: CPU FP32 原始解码 [20,3,256,256]，未裁剪。
    """
    manifest, decoded = directory / "samples.json", directory / "decoded.pt"
    audit = evidence["reconstruction"]
    if (
        audit["device"] != "cpu"
        or audit["dtype"] != "torch.float32"
        or hash_file(manifest) != audit["samples_sha256"]
        or hash_file(decoded) != audit["decoded_sha256"]
    ):
        raise ValueError("CPU reconstruction audit files do not match the review evidence")
    selected = json.loads(manifest.read_text())[:SAMPLE_COUNT]
    if [row["sample_id"] for row in selected] != evidence["random_sample_ids"][:SAMPLE_COUNT]:
        raise ValueError("CPU reconstruction audit sample order changed")
    restored = torch.load(decoded, map_location="cpu", weights_only=True, mmap=True)
    if not torch.is_tensor(restored) or restored.dtype != torch.float32:
        raise ValueError("Expected CPU FP32 raw reconstruction tensor")
    reference = restored[:SAMPLE_COUNT].contiguous()
    if (
        reference.shape != (SAMPLE_COUNT, 3, 256, 256)
        or not torch.isfinite(reference).all()
        or hashlib.sha256(reference.numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        != CPU_REFERENCE_SHA256
    ):
        raise ValueError("Pinned CPU FP32 reconstruction reference does not match")
    return reference


def _maxima(
    measurements: JsonDict,
    limits: dict[str, float],
) -> None:
    """校验测量集合完整且全部未超过预先确定的上界。

    Args:
        measurements: 每项的最差单样本绝对误差汇总。
        limits: 与 measurements 同键的上界。

    Raises:
        ValueError: 缺失、非数值、非有限或超过对应上界。
    """
    if not isinstance(measurements, dict) or measurements.keys() != limits.keys():
        raise ValueError("Unexpected local acceptance measurement set")
    for name, limit in limits.items():
        value = measurements[name]
        if type(value) not in (int, float) or not 0 <= value <= limit:
            raise ValueError(f"{name}={value!r} exceeds fixed bound {limit}")


def _delta(
    left: Float[Tensor, " b *z"],
    right: Float[Tensor, " b *z"],
) -> tuple[float, float]:
    """按样本归约绝对误差，再提取最大的逐样本峰值和均值。

    Args:
        left: [b,*z] 被测结果。
        right: 同形、同设备的对照结果。

    Results:
        tuple[float, float]: 最大元素绝对差、最大的单样本平均绝对差。
    """
    differences = (left - right).abs().flatten(1)
    return differences.max().item(), differences.mean(1).max().item()


def _frozen(
    model: nn.Module,
    device: str,
) -> None:
    """核对实际参数所在设备、推理精度和冻结状态。

    Args:
        model: 已加载的 PyTorch 编码组件。
        device: 预期设备类型 cpu 或 mps。

    Raises:
        ValueError: 模型仍处于训练态、有可训练参数、存在异设备或非 FP32 参数。
    """
    if model.training or any(
        parameter.requires_grad
        or parameter.dtype != torch.float32
        or parameter.device.type != device
        for parameter in model.parameters()
    ):
        raise ValueError(f"Expected a frozen FP32 model on {device}")


def _pixel_batches(
    source: CacheSource,
    records: list[JsonDict],
) -> tuple[list[Float[Tensor, "b c h w"]], list[Float[Tensor, "b c h w"]], float]:
    """按固定顺序加载图片，并对照独立 [0,1] 转换检查归一化。

    Args:
        source: 冻结训练图像所在目录。
        records: 固定 20 条清单记录。

    Raises:
        ValueError/OSError: 图像哈希、原始 RGB 尺寸、范围不符或无法解码。

    Results:
        tuple: 正式与原审计的 [b,3,256,256] CPU FP32 小批次列表、最差归一化误差。
    """
    batches, audit_batches, error = [], [], 0.0
    for start in range(0, len(records), BATCH_SIZE):
        rows = records[start : start + BATCH_SIZE]
        images = image_batch(rows, source.root)
        independent = (
            torch.stack(
                [
                    load_image(
                        source.root,
                        ReconstructionSample(
                            row["sample_id"],
                            row["image_path"],
                            row["sha256"],
                            "",
                            "random",
                            "local",
                        ),
                    )
                    for row in rows
                ]
            )
            .mul(2)
            .sub(1)
        )
        error = max(error, _delta(images, independent)[0])
        if (
            images.shape != (len(rows), 3, 256, 256)
            or images.dtype != torch.float32
            or not torch.isfinite(images).all()
            or images.min() < -1
            or images.max() > 1
        ):
            raise ValueError("Local parity requires finite 256x256 RGB FP32 in [-1,1]")
        batches.append(images)
        audit_batches.append(independent)
    return batches, audit_batches, error


@torch.inference_mode()
def _replay_cpu_audit(
    model: FrozenVAE,
    batches: list[Float[Tensor, "b c h w"]],
) -> Float[Tensor, "b c h w"]:
    """使用原 CPU 审计的转换表达式和内存布局复算，以核对档案。

    Args:
        model: 已冻结在 CPU 的 VAE。
        batches: 原审计表达式产生的连续 [b,3,256,256] 图像批。

    Raises:
        ValueError/RuntimeError: 输入约定或编码、解码失败。

    Results:
        Float[Tensor, "b c h w"]: 未裁剪的 CPU FP32 解码，b 为固定样本总数。
    """
    return torch.cat([model.decode(model.encode(images)) for images in batches])


@torch.inference_mode()
def _vae_run(
    model: FrozenVAE,
    batches: list[Float[Tensor, "b c h w"]],
    normal_error: float,
) -> tuple[JsonDict, Float[Tensor, "b c_z h_z w_z"], Float[Tensor, "b c h w"]]:
    """实测一个设备上的后验、正逆缩放、FP32 和重复推理。

    Args:
        model: 已冻结、绑定指定设备的 SDXL VAE。
        batches: 固定顺序的 [b,3,256,256] CPU 输入批。
        normal_error: 正式归一化与独立像素路径的最大绝对差。

    Raises:
        ValueError/RuntimeError: 实际设备、精度、形状或数值异常，或模型运行失败。

    Results:
        tuple: 误差汇总、CPU 上的 [20,4,32,32] 潜变量和 [20,3,256,256] 原始解码。
    """
    device = torch.device(model.device).type
    _frozen(model.model, device)
    measurements = dict.fromkeys(VAE_INVARIANTS, 0.0)
    measurements["normalization_max_abs"] = normal_error
    zs, decoded = [], []
    with torch.autocast(device_type=device, enabled=False):
        for pixels in batches:
            images = pixels.to(device)
            posterior = model.model.encode(images).latent_dist
            mean, mode = posterior.mean, posterior.mode()
            z = model.encode(images)
            output = model.decode(z)
            reference = model.model.decode(mean).sample
            repeated_z = model.encode(images)
            repeated_output = model.decode(repeated_z)
            if (
                any(t.shape != (len(images), 4, 32, 32) for t in (mean, mode, z, repeated_z))
                or any(t.shape != pixels.shape for t in (output, reference, repeated_output))
                or any(
                    t.dtype != torch.float32
                    or t.device.type != device
                    or not torch.isfinite(t).all()
                    for t in (mean, mode, z, repeated_z, output, reference, repeated_output)
                )
            ):
                raise ValueError(f"Unexpected VAE shape, precision or nonfinite output on {device}")
            errors = {
                "mode_mean_max_abs": _delta(mode, mean)[0],
                "forward_scale_max_abs": _delta(z, mean * model.scale)[0],
                "inverse_scale_max_abs": _delta(z / model.scale, mean)[0],
                "independent_decode_max_abs": _delta(output, reference)[0],
                "repeat_latent_max_abs": _delta(z, repeated_z)[0],
                "repeat_decode_max_abs": _delta(output, repeated_output)[0],
            }
            for key, value in errors.items():
                measurements[key] = max(measurements[key], value)
            zs.append(z.cpu())
            decoded.append(output.cpu())
    _maxima(measurements, VAE_INVARIANTS)
    return measurements, torch.cat(zs), torch.cat(decoded)


@torch.inference_mode()
def _clip_run(
    model: FrozenCLIPText,
    records: list[JsonDict],
) -> tuple[JsonDict, CLIPCondition]:
    """实测固定 20 条 Caption 与空条件的文本塔输出、位置掩码和重复性。

    Args:
        model: 已冻结、绑定指定设备的 CLIP。
        records: 与 20 张验证图片同序的 Caption 记录。

    Raises:
        ValueError/RuntimeError: 设备、形状、精度、掩码、token 约定或数值不符。

    Results:
        tuple[JsonDict, CLIPCondition]: 重复误差及 CPU 上的 [21,77,512] 特征、掩码和 ID。
    """
    device = model.device.type
    _frozen(model.model, device)
    captions = [row["caption"] for row in records] + [""]
    features, masks, ids = [], [], []
    error = 0.0
    for start in range(0, len(captions), BATCH_SIZE):
        chunk = captions[start : start + BATCH_SIZE]
        output = model.encode(chunk)
        repeat = model.encode(chunk)
        f, m, i = output.features, output.mask, output.input_ids
        if (
            f.device.type != device
            or m.device.type != device
            or i.device.type != device
            or f.shape != (len(chunk), 77, 512)
            or m.shape != i.shape
            or m.shape != (len(chunk), 77)
            or f.dtype != torch.float32
            or m.dtype != torch.bool
            or i.dtype != torch.long
            or not torch.isfinite(f).all()
            or not torch.isfinite(repeat.features).all()
            or not torch.equal(m, repeat.mask)
            or not torch.equal(i, repeat.input_ids)
        ):
            raise ValueError(
                f"Unexpected CLIP device, precision, shape or token output on {device}"
            )
        error = max(error, _delta(f, repeat.features)[0])
        features.append(f.cpu())
        masks.append(m.cpu())
        ids.append(i.cpu())
    output = CLIPCondition(torch.cat(features), torch.cat(masks), torch.cat(ids))
    counts = torch.tensor([row["caption_tokens"] + 2 for row in records] + [2])
    mask, tokens = output.mask, output.input_ids
    if (
        not torch.equal(mask, torch.arange(77) < counts[:, None])
        or not (tokens[:, 0] == 49406).all()
        or not (tokens.gather(1, counts[:, None] - 1) == 49407).all()
        or not (tokens[~mask] == 49407).all()
    ):
        raise ValueError("CLIP tokens/masks differ from frozen captions or the blank BOS/EOS")
    measurements = {"repeat_features_max_abs": error}
    _maxima(measurements, CLIP_INVARIANTS)
    return measurements, output


def _mps_probe() -> JsonDict:
    """检查当前进程能否执行真实 MPS 运算，记录运行时错误。

    Results:
        JsonDict: MPS 构建状态、可用状态及实际小张量检查。
    """
    built, available = torch.backends.mps.is_built(), torch.backends.mps.is_available()
    if not available:
        return {
            "built": built,
            "available": available,
            "tensor_passed": False,
            "error": "MPS runtime unavailable",
        }
    try:
        result = (torch.ones(2, device="mps") * 2).cpu().tolist()
    except (RuntimeError, NotImplementedError) as error:
        return {"built": built, "available": available, "tensor_passed": False, "error": str(error)}
    if result != [2.0, 2.0]:
        return {
            "built": built,
            "available": available,
            "tensor_passed": False,
            "error": "MPS probe returned unexpected values",
        }
    return {"built": built, "available": available, "tensor_passed": True}


def run_local_acceptance(
    root: Path,
    source: CacheSource,
    evidence_path: Path,
    reference_directory: Path,
    vae_lock: Path,
    clip_lock: Path,
) -> JsonDict:
    """分别决定 VAE 和 CLIP 的本机缓存设备，始终先验收 CPU FP32。

    Args:
        root: 项目根，用于定位两个冻结组件的本地资产。
        source: 正式 ArtBench 缓存数据及固定身份。
        evidence_path: 被用户审核的 200 张摘要。
        reference_directory: 200 张的原始 CPU FP32 实验产物目录。
        vae_lock: 固定 VAE 版本和权重哈希的锁。
        clip_lock: 固定 CLIP 权重与 tokenizer 的锁。

    Raises:
        ValueError/RuntimeError/OSError: CPU 基准、输入身份、环境或文件错误。
            MPS 单组件失败记入回执。

    Results:
        JsonDict: 固定样本、运行环境、原始测量以及可用于完整缓存的两项设备。

    副作用:
        实际运行 CPU/MPS 推理并在组件之间清理闲置 MPS 分配。
    """
    if any(
        os.getenv(key) != "0" for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")
    ):
        raise ValueError(
            "Start Python with PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0"
        )
    started = perf_counter()
    identities, evidence = _identity(source, evidence_path)
    ids = evidence["random_sample_ids"][:SAMPLE_COUNT]
    fixed = {row["sample_id"]: row for row in source.records["train"] if row["sample_id"] in ids}
    if set(fixed) != set(ids):
        raise ValueError("Pinned sample is absent from the ArtBench training split")
    records = [fixed[sid] for sid in ids]
    historical = _audit_reference(reference_directory, evidence)
    batches, audit_batches, normalization = _pixel_batches(source, records)
    runtime = _mps_probe()
    checks, devices = {}, {}

    cpu_started = perf_counter()
    cpu_vae = FrozenVAE.load(root, vae_lock, device="cpu")
    # 原审计逐图堆叠为连续 NCHW，正式输入为 channels-last。
    # 跨设备比较统一用正式输入；核对档案时复用原运算顺序与布局，避免混入 CPU 内核差异。
    replay = _replay_cpu_audit(cpu_vae, audit_batches)
    cpu_invariants, cpu_latent, cpu_decoded = _vae_run(cpu_vae, batches, normalization)
    del cpu_vae
    max_abs, mean_abs = _delta(replay, historical)
    audit_error = {"audit_decode_max_abs": max_abs, "audit_decode_mean_abs": mean_abs}
    _maxima(audit_error, VAE_CPU_AUDIT_LIMITS)
    vae_check = {
        "status": "passed",
        "cpu": {
            "invariants": cpu_invariants,
            "historical_parity": audit_error,
            "elapsed_seconds": round(perf_counter() - cpu_started, 3),
        },
    }
    devices["vae"] = "cpu"
    if runtime["tensor_passed"]:
        try:
            mps_started = perf_counter()
            invariants, latent, decoded = _vae_run(
                FrozenVAE.load(root, vae_lock, device="mps"), batches, normalization
            )
            l_max, l_mean = _delta(latent, cpu_latent)
            d_max, d_mean = _delta(decoded, cpu_decoded)
            parity = {
                "latent_max_abs": l_max,
                "latent_mean_abs": l_mean,
                "decode_max_abs": d_max,
                "decode_mean_abs": d_mean,
            }
            _maxima(parity, VAE_PARITY_LIMITS)
        except (RuntimeError, NotImplementedError, ValueError) as error:
            vae_check["mps"] = {"status": "failed", "error": str(error)}
        else:
            vae_check["mps"] = {
                "status": "passed",
                "invariants": invariants,
                "parity": parity,
                "elapsed_seconds": round(perf_counter() - mps_started, 3),
            }
            devices["vae"] = "mps"
        finally:
            gc.collect()
            torch.mps.empty_cache()
    else:
        vae_check["mps"] = {"status": "unavailable", "error": runtime["error"]}
    checks["vae"] = vae_check

    cpu_started = perf_counter()
    cpu_invariants, cpu = _clip_run(FrozenCLIPText.from_lock(clip_lock, root=root), records)
    clip_check = {
        "status": "passed",
        "cpu": {
            "invariants": cpu_invariants,
            "elapsed_seconds": round(perf_counter() - cpu_started, 3),
        },
    }
    devices["clip"] = "cpu"
    if runtime["tensor_passed"]:
        try:
            mps_started = perf_counter()
            invariants, mps = _clip_run(
                FrozenCLIPText.from_lock(clip_lock, root=root, device="mps"), records
            )
            if not torch.equal(cpu.mask, mps.mask) or not torch.equal(cpu.input_ids, mps.input_ids):
                raise ValueError("CLIP token IDs or mask differ across devices")
            # 只比较被下游使用的有效 token；两边仍验证全部输出有限。
            m = cpu.mask.unsqueeze(-1)
            absolute = (cpu.features - mps.features).abs().masked_fill(~m, 0)
            parity = {
                "features_max_abs": absolute.max().item(),
                "features_mean_abs": (absolute.sum((1, 2)) / (cpu.mask.sum(1) * 512)).max().item(),
            }
            _maxima(parity, CLIP_PARITY_LIMITS)
        except (RuntimeError, NotImplementedError, ValueError) as error:
            clip_check["mps"] = {"status": "failed", "error": str(error)}
        else:
            clip_check["mps"] = {
                "status": "passed",
                "invariants": invariants,
                "parity": parity,
                "token_ids_and_masks_equal": True,
                "elapsed_seconds": round(perf_counter() - mps_started, 3),
            }
            devices["clip"] = "mps"
        finally:
            gc.collect()
            torch.mps.empty_cache()
    else:
        clip_check["mps"] = {"status": "unavailable", "error": runtime["error"]}
    checks["clip"] = clip_check
    return {
        "schema_version": 1,
        "status": "passed",
        "identities": identities,
        "sample_ids": ids,
        "devices": devices,
        "checks": checks,
        "limits": _limits(),
        "runtime": {
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "dtype": "float32",
            "cpu_threads": torch.get_num_threads(),
            "mps": runtime,
            "mps_fallback": "0",
            "mps_fast_math": "0",
            "cuda_run": False,
        },
        "reference_sha256": CPU_REFERENCE_SHA256,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def _limits() -> JsonDict:
    """汇总固定容差，作为执行及离线验证共享的契约。

    Results:
        JsonDict: 各组件的 CPU、同设备以及跨设备误差上界。
    """
    return {
        "vae_invariants": VAE_INVARIANTS,
        "vae_cpu_audit": VAE_CPU_AUDIT_LIMITS,
        "vae_parity": VAE_PARITY_LIMITS,
        "clip_invariants": CLIP_INVARIANTS,
        "clip_parity": CLIP_PARITY_LIMITS,
    }


def validate_local_receipt(
    source: CacheSource,
    path: Path,
    evidence_path: Path,
) -> tuple[str, str, str]:
    """离线校验正式缓存的本机设备验收回执，不要求消费者访问 MPS。

    Args:
        source: 当前冻结数据及组件的固定身份。
        path: 20 张 CPU FP32/MPS 验收回执。
        evidence_path: 当前用户批准的原始 200 张结构化结果。

    Raises:
        ValueError/KeyError/OSError: 回执不全、样本及资产身份不符、阈值未通过或文件不可读。

    Results:
        tuple[str, str, str]: 本机回执 SHA-256、VAE 设备、CLIP 设备。
    """
    identities, evidence = _identity(source, evidence_path)
    data = path.read_bytes()
    report = json.loads(data)
    runtime = report.get("runtime", {})
    if (
        type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
        or report.get("status") != "passed"
        or report.get("sample_ids") != evidence["random_sample_ids"][:SAMPLE_COUNT]
        or report.get("identities") != identities
        or report.get("reference_sha256") != CPU_REFERENCE_SHA256
        or report.get("limits") != _limits()
        or runtime.get("dtype") != "float32"
        or not isinstance(runtime.get("torch"), str)
        or runtime.get("mps_fallback") != "0"
        or runtime.get("mps_fast_math") != "0"
        or runtime.get("cuda_run") is not False
    ):
        raise ValueError("Local acceptance does not match the frozen protocol or inputs")
    devices = report["devices"]
    checks = report["checks"]
    if devices.keys() != {"vae", "clip"} or checks.keys() != devices.keys():
        raise ValueError("Local acceptance is missing component results")
    mps_runtime = runtime["mps"]
    probe_passed = mps_runtime.get("tensor_passed") is True
    for component, invariants, parity in (
        ("vae", VAE_INVARIANTS, VAE_PARITY_LIMITS),
        ("clip", CLIP_INVARIANTS, CLIP_PARITY_LIMITS),
    ):
        check = checks[component]
        cpu, mps = check["cpu"], check["mps"]
        if check.get("status") != "passed":
            raise ValueError(f"Local acceptance failed for {component}")
        _maxima(cpu["invariants"], invariants)
        if component == "vae":
            _maxima(cpu["historical_parity"], VAE_CPU_AUDIT_LIMITS)
        if devices[component] == "mps":
            if (
                mps_runtime.get("built") is not True
                or mps_runtime.get("available") is not True
                or not probe_passed
                or mps.get("status") != "passed"
                or (component == "clip" and mps.get("token_ids_and_masks_equal") is not True)
            ):
                raise ValueError(f"MPS acceptance not demonstrated for {component}")
            _maxima(mps["invariants"], invariants)
            _maxima(mps["parity"], parity)
        elif (
            devices[component] != "cpu"
            or mps.get("status") != ("failed" if probe_passed else "unavailable")
            or not isinstance(mps.get("error"), str)
            or not mps["error"]
        ):
            raise ValueError(f"CPU fallback does not document MPS failure for {component}")
    return hashlib.sha256(data).hexdigest(), devices["vae"], devices["clip"]
