"""以固定动态范围测量 RGB 重建；LPIPS 权重只从校验后的本地文件加载。"""

import json
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from minigen.autoencoders.sdxl import hash_file

_ALEX_TAPS = (1, 4, 7, 9, 11)
_ALEX_CHANNELS = (64, 192, 384, 256, 256)


def _validate_pair(
    reference: Float[Tensor, "b 3 h w"],
    estimated: Float[Tensor, "b 3 h w"],
    min_size: int,
) -> None:
    """拒绝会使重建指标的形状或动态范围失去含义的输入。

    Args:
        reference: [b,3,h,w] 浮点 RGB 原图，范围 [0,1]。
        estimated: 同设备、同类型、同形状的重建图，范围 [0,1]。
        min_size: 指标窗口要求的最小高宽。

    Raises:
        ValueError: 批为空、形状或设备不一致、像素范围错误，或图像尺寸不足。
    """
    if (
        reference.ndim != 4
        or reference.shape[0] == 0
        or reference.shape[1] != 3
        or reference.shape != estimated.shape
        or min(reference.shape[-2:]) < min_size
        or reference.device != estimated.device
        or reference.dtype != estimated.dtype
        or not reference.is_floating_point()
    ):
        raise ValueError("Expected nonempty same-size floating point RGB image pairs")
    for batch in (reference, estimated):
        if not torch.isfinite(batch).all() or batch.amin() < 0 or batch.amax() > 1:
            raise ValueError("Reconstruction metric inputs must be finite in [0,1]")


def psnr(
    reference: Float[Tensor, "b 3 h w"],
    estimated: Float[Tensor, "b 3 h w"],
) -> Float[Tensor, " b"]:
    """用单位峰值和所有 RGB 像素的 MSE 计算每图 PSNR。

    Args:
        reference: [b,3,h,w]、[0,1] 原图。
        estimated: 同形、同设备、同范围重建图。

    Raises:
        ValueError: 不是合法的浮点 RGB 配对。

    Results:
        Float[Tensor, " b"]: 各图的 dB 值；完全一致时为正无穷。
    """
    _validate_pair(reference, estimated, min_size=1)
    return -10 * (reference - estimated).square().mean(dim=(1, 2, 3)).log10()


def ssim(
    reference: Float[Tensor, "b 3 h w"],
    estimated: Float[Tensor, "b 3 h w"],
) -> Float[Tensor, " b"]:
    """在每个 RGB 通道计算局部 SSIM，再平均通道与有效窗口。

    Args:
        reference: [b,3,h,w]、[0,1] 原图，h 和 w 至少 11。
        estimated: 同形、同设备、同范围重建图。

    Raises:
        ValueError: 不是合法的浮点 RGB 配对，或图像小于高斯窗口。

    Results:
        Float[Tensor, " b"]: 各图的 SSIM；完全一致时约为 1。
    """
    _validate_pair(reference, estimated, min_size=11)
    grid = torch.arange(-5, 6, device=reference.device, dtype=reference.dtype)
    kernel = (-grid.square() / (2 * 1.5**2)).exp()
    kernel = kernel / kernel.sum()
    # 11×11 高斯可分离；不填充边界，避免引入人工像素。
    stacked = torch.cat(
        (reference, estimated, reference.square(), estimated.square(), reference * estimated), dim=1
    )
    horizontal = F.conv2d(stacked, kernel.view(1, 1, 1, 11).expand(15, -1, -1, -1), groups=15)
    windows = F.conv2d(horizontal, kernel.view(1, 1, 11, 1).expand(15, -1, -1, -1), groups=15)
    mx, my, ex2, ey2, exy = windows.chunk(5, dim=1)
    vx, vy = (ex2 - mx.square()).clamp_min(0), (ey2 - my.square()).clamp_min(0)
    covariance = exy - mx * my
    c1, c2 = 0.01**2, 0.03**2
    local = ((2 * mx * my + c1) * (2 * covariance + c2)) / (
        (mx.square() + my.square() + c1) * (vx + vy + c2)
    )
    return local.mean(dim=(1, 2, 3))


class LPIPSAlex(nn.Module):
    """使用 LPIPS v0.1 非空间 AlexNet 配方计算重建的感知距离。

    Args:
        features: torchvision AlexNet 预训练卷积特征主干。
        calibration: 五层 v0.1 校准权重，每项为 [1,c,1,1]。
    """

    def __init__(
        self,
        features: nn.Sequential,
        calibration: list[Float[Tensor, "1 c 1 1"]],
    ) -> None:
        """冻结 AlexNet 特征层并注册各通道校准权重。

        Args:
            features: torchvision AlexNet 的卷积主干，保留五个 ReLU 输出。
            calibration: 五组 1×1 线性校准权重。

        Raises:
            ValueError: 主干不含五个特征输出或校准权重形状错误。

        副作用:
            将传入 features 设置为评估模式并禁用其参数梯度。
        """
        super().__init__()
        if len(features) <= _ALEX_TAPS[-1] or [tuple(w.shape) for w in calibration] != [
            (1, c, 1, 1) for c in _ALEX_CHANNELS
        ]:
            raise ValueError("LPIPS AlexNet taps or calibration shapes do not match v0.1")
        self.features = features.eval().requires_grad_(False)
        self.register_buffer("input_shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer("input_scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))
        for tap, weight in zip(_ALEX_TAPS, calibration, strict=True):
            self.register_buffer(f"calibration_{tap}", weight.detach())

    @classmethod
    def load(
        cls,
        root: Path,
        lock: Path,
    ) -> "LPIPSAlex":
        """校验两个本地权重文件并严格加载 LPIPS 的固定实现。

        Args:
            root: 项目根目录，用于解析模型身份中的权重路径。
            lock: 包含 lpips_alex 权重身份的 P2 资产回执。

        Raises:
            ValueError: 字节数、哈希、权重路径、版本或校准形状错误。
            OSError: 文件缺失或不能读取；底层权重反序列化和加载错误透传。

        Results:
            LPIPSAlex: 本地 CPU FP32 的冻结距离模型；不请求网络。
        """
        from torchvision.models import alexnet

        identity = json.loads(lock.read_text())["lpips_alex"]
        if identity["calibration"]["version"] != "0.1":
            raise ValueError("Expected LPIPS AlexNet calibration v0.1")
        paths = []
        for entry in (identity["backbone"], identity["calibration"]):
            path = (root / entry["path"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or path.stat().st_size != entry["bytes"]
                or hash_file(path) != entry["sha256"]
            ):
                raise ValueError(f"LPIPS weight lock mismatch: {path.name}")
            paths.append(path)
        # 构建待加载网络不应改变调用方的 CPU RNG 序列。
        with torch.random.fork_rng(devices=[]):
            backbone = alexnet(weights=None)
        backbone.load_state_dict(
            torch.load(paths[0], map_location="cpu", weights_only=True), strict=True
        )
        state = torch.load(paths[1], map_location="cpu", weights_only=True)
        keys = [f"lin{i}.model.1.weight" for i in range(len(_ALEX_TAPS))]
        if set(state) != set(keys):
            raise ValueError("Unexpected LPIPS v0.1 calibration keys")
        return cls(backbone.features, [state[key] for key in keys]).eval()

    @torch.inference_mode()
    def forward(
        self,
        reference: Float[Tensor, "b 3 h w"],
        estimated: Float[Tensor, "b 3 h w"],
    ) -> Float[Tensor, " b"]:
        """比较五层 AlexNet 单位通道特征并累加加权空间均值。

        Args:
            reference: [b,3,h,w]、[0,1] 原图，h 和 w 至少 32。
            estimated: 同形、同设备、同范围重建图。

        Raises:
            ValueError: 不是合法 RGB 配对；底层特征模型的设备或类型错误透传。

        Results:
            Float[Tensor, " b"]: 各图的非空间 LPIPS 距离，越小越相近。
        """
        _validate_pair(reference, estimated, min_size=32)
        # v0.1 先用 [-1,1] 像素再做通道偏移缩放；不是 AlexNet 分类预处理。
        paired = torch.cat((reference, estimated))
        features = (paired.mul(2).sub(1) - self.input_shift) / self.input_scale
        distance = reference.new_zeros(len(reference))
        for idx, layer in enumerate(self.features):
            if idx > _ALEX_TAPS[-1]:
                break
            features = layer(features)
            if idx not in _ALEX_TAPS:
                continue
            unit = features / (features.square().sum(dim=1, keepdim=True).sqrt() + 1e-10)
            left, right = unit.chunk(2, dim=0)
            weight = getattr(self, f"calibration_{idx}")
            distance += F.conv2d((left - right).square(), weight).mean(dim=(1, 2, 3))
        return distance
