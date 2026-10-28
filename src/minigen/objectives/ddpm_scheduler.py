"""L01 / S1-03：个人离散加噪、噪声目标与反向更新接口。

维度简称：b = 批量，c = 通道，h/w = 高/宽，t = 噪声级别总数。
形状中的每个项代表一个轴；空字符串表示零维标量，1 表示广播轴。
一维形状字符串的前导空格仅用于兼容 Ruff，不代表额外维度。
类型和形状标注不自动执行运行时检查，实际检查以方法体为准。
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


class DDPMScheduler(nn.Module):
    """管理离散噪声日程，并提供加噪、噪声损失和反向一步更新。

    Args:
        num_steps: 噪声级别总数 t，默认 1000，索引范围为 [0, t - 1]。
        beta_start: 线性 beta 日程的起点，默认 0.0001。
        beta_end: 线性 beta 日程的终点，默认 0.02。
        clip_denoised: 是否将估计的干净图像裁剪到 [-1, 1]。

    Raises:
        ValueError: num_steps 不是正整数（不接受 bool），或 beta 不满足规定范围。

    Results:
        DDPMScheduler: 保存噪声日程的模块实例。
            num_steps: 时间级别总数。
            clip_denoised: 反向采样时的裁剪开关。
            buffers: 长度为 t 的浮点系数表，随模块迁移设备并保存在 state_dict 中。

    副作用:
        设置 num_steps、clip_denoised 并注册长度为 t 的 float32 缓冲区。
        系数先用 float64 计算再保存；缓冲区随模块迁移，不是可训练参数。
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_denoised: bool = False,
    ) -> None:
        """构建线性 beta 日程与反向后验系数表。

        Args:
            num_steps: 时间级别总数 t，必须为 >= 1 的整数，不接受 bool。
            beta_start: beta 日程起点，必须 > 0。
            beta_end: beta 日程终点，必须满足 beta_start <= beta_end < 1。
            clip_denoised: 为 True 时，step 将估计的干净图像裁剪到 [-1, 1]。

        Raises:
            ValueError: num_steps 不是正整数（不接受 bool），或 beta 的起止值不满足上述范围。

        副作用:
            设置 num_steps、clip_denoised 并注册长度为 t 的 float32 缓冲区。
            系数先用 float64 计算再保存；缓冲区随模块迁移，不是可训练参数。
        """
        super().__init__()
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps 必须为 >= 1 的整数，不接受 bool")
        if not (0 < beta_start <= beta_end < 1):
            raise ValueError("beta 必须满足 0 < start <= end < 1")

        self.num_steps = num_steps
        self.clip_denoised = clip_denoised

        beta = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        alpha = 1.0 - beta
        abar = torch.cumprod(alpha, dim=-1)
        prev_abar = torch.cat((torch.ones_like(abar[:1]), abar[:-1]), dim=-1)

        post_c0 = (prev_abar.sqrt() * beta) / (1.0 - abar)
        post_ct = alpha.sqrt() * (1.0 - prev_abar) / (1.0 - abar)
        post_var = beta * (1.0 - prev_abar) / (1.0 - abar)

        params = {
            "betas": beta,
            "alphas": alpha,
            "alpha_bars": abar,
            "signals": abar.sqrt(),
            "noise_scales": (1 - abar).sqrt(),
            "post_c0": post_c0,
            "post_ct": post_ct,
            "post_var": post_var,
        }
        for name, values in params.items():
            self.register_buffer(name, values.float())

    def _extract(
        self,
        table: Float[Tensor, " t"],
        time_steps: Int[Tensor, " b"],
        image: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b 1 1 1"]:
        """按每个样本的时间索引取系数，并调整为可沿图像维度广播的形状。

        Args:
            table: [t] 浮点系数表，长度应等于 num_steps。
            time_steps: [b] 的 torch.long 时间索引，每个样本一个值。
            image: [b, c, h, w] 参考图像，用于确定批量大小和广播维数。

        Raises:
            ValueError: time_steps 不是一维、长度不等于 b、类型不是 torch.long，或索引越界。
            IndexError: table 长度不足导致索引失败时，由 PyTorch 透传。
            RuntimeError: 索引与系数表设备不兼容等张量操作错误，由 PyTorch 透传。

        Results:
            Float[Tensor, "b 1 1 1"]: 每个样本一个系数，可广播到 [b, c, h, w]。
        """
        if time_steps.ndim != 1:
            raise ValueError("时间索引必须是一维 Tensor")
        if len(time_steps) != len(image):
            raise ValueError("时间索引数量必须等于 batch_size")
        if time_steps.dtype != torch.long:
            raise ValueError("时间索引必须是 torch.long")
        if bool((time_steps < 0).any()) or bool((time_steps >= self.num_steps).any()):
            raise ValueError("时间索引越界")

        return table[time_steps].reshape(-1, *([1] * (image.ndim - 1)))

    def add_noise(
        self,
        clean: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
        timesteps: Int[Tensor, " b"],
    ) -> Float[Tensor, "b c h w"]:
        """按样本时间将干净图像与给定噪声组合成带噪图像。

        Args:
            clean: [b, c, h, w] 干净图像或潜变量。
            noise: [b, c, h, w] 标准高斯噪声，由调用者生成，必须与 clean 同形。
            timesteps: [b] 的 torch.long 时间索引，范围为 [0, num_steps - 1]。

        Raises:
            ValueError: clean 与 noise 形状不同，或 _extract 检查发现时间索引无效。
            RuntimeError: 张量设备等条件不兼容时，由 PyTorch 透传。

        Results:
            Float[Tensor, "b c h w"]: 与 clean 同形的带噪图像。
        """
        if clean.shape != noise.shape:
            raise ValueError("图片形状与噪声形状不同")

        signal = self._extract(self.signals, timesteps, clean)
        noise_scale = self._extract(self.noise_scales, timesteps, noise)
        return signal * clean + noise_scale * noise

    def loss(
        self,
        predicted_noise: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, ""]:
        """计算预测噪声与目标噪声在所有元素上的平均平方误差。

        Args:
            predicted_noise: [b, c, h, w] 网络预测的噪声。
            noise: [b, c, h, w] 真实加噪值，必须与 predicted_noise 同形。

        Raises:
            ValueError: 预测噪声与真实噪声形状不同，禁止隐式广播。
            RuntimeError: 张量设备等条件不兼容时，由 PyTorch 透传。

        Results:
            Float[Tensor, ""]: 零维浮点损失，对 b、c、h、w 的全部元素取均值。
                输入参与自动求导时，返回值保留计算图，可用于反向传播。
        """
        if predicted_noise.shape != noise.shape:
            raise ValueError("预测噪声与真实噪声形状不一致")

        return F.mse_loss(predicted_noise, noise, reduction="mean")

    def _posterior(
        self,
        clean: Float[Tensor, "b c h w"],
        sample: Float[Tensor, "b c h w"],
        timesteps: Int[Tensor, " b"],
    ) -> tuple[Float[Tensor, "b c h w"], Float[Tensor, "b 1 1 1"]]:
        """根据干净图像与当前带噪图像计算反向条件高斯分布的均值和方差。

        Args:
            clean: [b, c, h, w] 干净图像；采样时传入其估计值。
            sample: [b, c, h, w] 当前带噪图像，应与 clean 同形。
            timesteps: [b] 的 torch.long 时间索引，范围为 [0, num_steps - 1]。

        Raises:
            ValueError: _extract 检查发现时间索引无效时透传。
            RuntimeError: 张量形状无法广播或设备不兼容时，由 PyTorch 透传。
                本方法不主动检查 clean 与 sample 是否同形，调用者应保证这一条件。

        Results:
            tuple[Float[Tensor, "b c h w"], Float[Tensor, "b 1 1 1"]]: 均值和方差。
                mean: [b, c, h, w]，由干净图像和当前带噪图像线性组合得到。
                var: [b, 1, 1, 1]，每个样本一个方差，沿通道和空间维度共享。
        """
        mean = (
            self._extract(self.post_c0, timesteps, sample) * clean
            + self._extract(self.post_ct, timesteps, sample) * sample
        )
        var = self._extract(self.post_var, timesteps, sample)
        return mean, var

    def step(
        self,
        pred_noise: Float[Tensor, "b c h w"],
        timestep: int,
        sample: Float[Tensor, "b c h w"],
        noise: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b c h w"]:
        """执行一次反向去噪更新，在最终索引 0 处只返回均值。

        Args:
            pred_noise: [b, c, h, w] 网络预测的噪声，与 sample 同形。
            timestep: 整个批量共享的整数时间索引，范围为 [0, num_steps - 1]，不接受 bool。
            sample: [b, c, h, w] 当前带噪图像。
            noise: [b, c, h, w] 调用者提供的标准高斯噪声，与 sample 同形。
                timestep 为 0 时不加入此噪声，但仍检查其形状。

        Raises:
            ValueError: timestep 不是整数（不接受 bool）或越界，或三个图像张量形状不一致。
            RuntimeError: 输入与系数表设备等条件不兼容时，由 PyTorch 透传。

        Results:
            Float[Tensor, "b c h w"]: 前一噪声级别的图像，形状与 sample 相同。
                timestep >= 1: 返回后验均值加标准差乘 noise。
                timestep == 0: 返回最终去噪均值，不再添加随机项。
                clip_denoised: 为 True 时裁剪估计的干净图像，影响后验均值；不原地修改 sample。
        """
        # torch.full(..., dtype=torch.long) 会截断小数；先拒绝非整数，避免错用噪声级别。
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise ValueError("反向索引必须为整数，不接受 bool")
        if not 0 <= timestep < self.num_steps:
            raise ValueError("反向索引越界")
        if sample.shape != noise.shape or sample.shape != pred_noise.shape:
            raise ValueError("反向采样形状不一致")

        timesteps = torch.full((len(sample),), timestep, dtype=torch.long, device=sample.device)
        pred_x0 = (
            sample - self._extract(self.noise_scales, timesteps, sample) * pred_noise
        ) / self._extract(self.signals, timesteps, sample)

        if self.clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)

        mean, var = self._posterior(pred_x0, sample, timesteps)

        return mean + var.sqrt() * noise if timestep >= 1 else mean
