"""冻结 SDXL VAE 及共享图像转换；加载只访问本地固定资产。"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn

from minigen._typing import JsonDict

SDXL_FILES = {"config.json", "diffusion_pytorch_model.safetensors"}


def hash_file(
    path: str | Path,
) -> str:
    """分块读取文件并计算 SHA-256，避免整体载入权重。

    Args:
        path: 本地文件路径，用于校验模型或缓存。

    Raises:
        OSError: 文件缺失、不可读或读取失败时透传。

    Results:
        str: 十六进制文件摘要。
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FrozenVAE:
    """冻结模型专属的空间 VAE，统一后验均值编码与缩放解码。

    Args:
        model: 已以 FP32 加载的模块，需提供 config.scaling_factor、encode 和 decode。
        identity: 模型 revision、文件哈希及本地路径等 JSON 身份。
        device: cpu、mps 或 cuda 等执行设备。

    Raises:
        ValueError: 模型浮点参数不是 FP32，或 scaling_factor 非正；设备错误透传。

    Results:
        FrozenVAE: 适配器实例；encode/decode 显式关闭梯度，不下载额外模型。

    副作用:
        把传入模块改为 eval、关闭参数梯度并迁移设备；调用者持有的同一模型也受影响。
    """

    def __init__(
        self,
        model: nn.Module,
        identity: JsonDict,
        device: str = "cpu",
    ) -> None:
        """冻结参数、设为评估模式并记录模型自身缩放。

        Args:
            model: 具有空间 VAE 接口的模块，测试中可显式传入假模型。
            identity: 可追溯的 JSON 组件身份。
            device: 仅迁移设备，不在此改变模型浮点精度。

        Raises:
            ValueError: 浮点参数不是 FP32 或缩放系数非正；AttributeError: 缺少模型专属接口。

        副作用:
            把传入模块改为 eval、关闭参数梯度并迁移设备；调用者持有的同一模型也受影响。
        """
        self.model = model.eval().requires_grad_(False).to(device=device)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Frozen VAE must be loaded in FP32")
        self.identity = identity
        self.device = device
        self.scale = float(model.config.scaling_factor)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Scaling factor must be finite and positive")

    @classmethod
    def load(
        cls,
        root: str | Path,
        lock_path: str | Path,
        device: str = "cpu",
    ) -> "FrozenVAE":
        """校验固定文件哈希并从本地 Safetensors 加载冻结 VAE。

        Args:
            root: 项目根目录，用于解析身份中的 local_path。
            lock_path: 固定模型身份 JSON 的路径。
            device: 模型目标设备，默认 CPU。

        Raises:
            ValueError: 模型身份、文件、权重集合或配置不符；ImportError: 缺少 latent 可选依赖。
            OSError/模型加载错误原样透传。

        Results:
            FrozenVAE: 已冻结的 FP32 FrozenVAE 实例；local_files_only=True 禁止隐式下载。
        """
        identity = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        root = Path(root).resolve()
        relative = Path(identity["local_path"])
        directory = (root / relative).resolve()
        files = identity["files"]
        if (
            identity.get("repo_id") != "madebyollin/sdxl-vae-fp16-fix"
            or not re.fullmatch(r"[0-9a-f]{40}", identity["revision"])
            or relative.is_absolute()
            or directory == root
            or not directory.is_relative_to(root)
            or files.keys() != SDXL_FILES
        ):
            raise ValueError("Expected a locked local SDXL VAE with two exact component files")
        for name, info in files.items():
            file = directory / name
            if (
                not file.resolve().is_relative_to(directory)
                or file.stat().st_size != info["bytes"]
                or hash_file(file) != info["sha256"]
            ):
                raise ValueError(f"SDXL VAE file identity mismatch: {name}")

        from diffusers import AutoencoderKL

        model, load_info = AutoencoderKL.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            output_loading_info=True,
        )
        if any(load_info.values()):
            raise ValueError(f"SDXL VAE weights did not strictly match the model: {load_info}")
        if (
            model.config.latent_channels != 4
            or len(model.config.block_out_channels) != 4
            or model.config.scaling_factor != identity["scaling_factor"]
        ):
            raise ValueError("SDXL VAE channels, compression or scaling do not match the lock")
        return cls(model, identity, device)

    @torch.no_grad()
    def encode(
        self,
        images: Float[Tensor, "b 3 h w"],
    ) -> Float[Tensor, "b c_z h_z w_z"]:
        """编码 RGB 图像并对固定后验均值应用模型缩放。

        Args:
            images: [b,3,h,w] 有限浮点像素，范围 [-1,1]；自动迁移到适配器设备。

        Raises:
            ValueError: 不是四维 RGB、含非有限值或像素超出范围；底层编码错误透传。

        Results:
            Float[Tensor, "b c_z h_z w_z"]: 缩放后的 [b,c_z,h_z,w_z]。
            本组件在 256 像素输入时为 [b,4,32,32]。
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected [B,3,H,W] image batch")
        if not torch.isfinite(images).all() or images.min() < -1.001 or images.max() > 1.001:
            raise ValueError("Expected finite pixels in [-1,1]")
        # 缓存固定后验均值，先应用模型系数；不引入额外采样噪声。
        return self.model.encode(images.to(self.device)).latent_dist.mode() * self.scale

    @torch.no_grad()
    def decode(
        self,
        scaled_latents: Float[Tensor, "b c_z h_z w_z"],
    ) -> Float[Tensor, "b 3 h w"]:
        """逆缩放后解码，不把可显示裁剪混入模型原始输出。

        Args:
            scaled_latents: 与本组件缩放约定匹配的空间潜变量，自动迁移设备。

        Raises:
            RuntimeError: 通道、空间或设备与解码器不兼容时透传。

        Results:
            Float[Tensor, "b 3 h w"]: 原始解码浮点图像 [b,3,h,w]，数值可能略超出 [-1,1]。
        """
        # 解码器需要未缩放的表示，与 encode 的乘法严格配对。
        return self.model.decode(scaled_latents.to(self.device) / self.scale).sample


def image_batch(
    records: list[JsonDict],
    data_root: str | Path,
) -> Float[Tensor, "b 3 h w"]:
    """按记录顺序加载 RGB 图像并转换为 VAE 像素范围。

    Args:
        records: 非空图像清单，含相对 image_path。
        data_root: 图像路径根目录。

    Raises:
        OSError: 图像缺失或损坏；ValueError: 空输入或各图尺寸不一致导致 stack 失败。

    Results:
        Float[Tensor, "b 3 h w"]: CPU FP32 [b,3,h,w]，由 0–255 映射到 [-1,1]，不额外 resize。
    """
    arrays = []
    for record in records:
        with Image.open(Path(data_root) / record["image_path"]) as image:
            arrays.append(np.array(image.convert("RGB"), dtype=np.float32))
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2) / 127.5 - 1


def to_image(
    tensor: Float[Tensor, "3 h w"],
) -> Image.Image:
    """把单张解码结果裁剪、量化为用于展示和规则评测的 RGB 图像。

    Args:
        tensor: [3,h,w] 浮点图；函数转 CPU 并裁剪到 [-1,1]。

    Raises:
        RuntimeError/TypeError: 输入维数、通道或 PIL 支持格式不符时透传。

    Results:
        Image.Image: Pillow RGB 图像，像素 uint8；不能替代原始解码值的 MSE 计算。
    """
    pixels = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())
