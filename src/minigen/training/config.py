"""单卡 Flow 训练配置及显式模型工厂。"""

import math
from dataclasses import asdict, dataclass, field, fields

from torch import nn

from minigen._typing import JsonDict


@dataclass(frozen=True)
class TrainConfig:
    """冻结优化配方，把停止上限与学习率日程分开记录。

    Args:
        output_dir: 独立运行目录；所有日志和检查点的根。
        model: 模型构造参数，architecture 为 b0、b1 或 b2。
        objective: 固定为 flow_velocity，禁止混用 DDPM 噪声预测目标。
        device: auto、cpu、mps 或 cuda。
        precision: fp32，或 CUDA 上的 fp16/bf16 自动混合精度。
        seed: 模型、噪声和数据排序的种子。
        batch_size: 每个微批的最大样本数。
        grad_accum: 一次优化更新积累的微批数量。
        learning_rate: warmup 后的峰值学习率。
        weight_decay: AdamW 权重衰减。
        grad_clip: 全模型梯度范数上限。
        ema_decay: EMA 保留旧权重的比例。
        condition_dropout: 整条文本替换为空条件的概率。
        max_steps: 优化更新总上限，可在兼容恢复时延长。
        max_seconds: 当前调用训练循环的墙钟上限。
        checkpoint_every: 周期检查点间隔；同时保留 last.pt。
        validation_every: 固定噪声验证的更新间隔。
        checkpoint_steps: 除 last.pt 外保留的指定更新快照。
        cpu_threads: 当前进程的 CPU 线程数。
        warmup_steps: 固定日程的线性预热步数。
        schedule_steps: 固定余弦日程终点，独立于停止上限。
        min_lr_ratio: 日程末端学习率相对峰值的比例。
    """

    output_dir: str = "artifacts/runs/p3_local"
    model: JsonDict = field(default_factory=dict)
    objective: str = "flow_velocity"
    device: str = "auto"
    precision: str = "fp32"
    seed: int = 0
    batch_size: int = 4
    grad_accum: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    condition_dropout: float = 0.1
    max_steps: int = 500
    max_seconds: float = 900.0
    checkpoint_every: int = 100
    validation_every: int = 50
    checkpoint_steps: tuple[int, ...] = (0, 100)
    cpu_threads: int = 5
    warmup_steps: int = 20
    schedule_steps: int = 1000
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        """拒绝拼写错误之外的非法取值与不明确的精度组合。

        Raises:
            ValueError: 数值、设备、精度或日程不合法。
        """
        positive = (
            self.batch_size,
            self.grad_accum,
            self.max_steps,
            self.checkpoint_every,
            self.validation_every,
            self.cpu_threads,
            self.schedule_steps,
        )
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError("Positive integer training limits required")
        if any(type(step) is not int or step < 0 for step in self.checkpoint_steps):
            raise ValueError("checkpoint_steps must contain nonnegative integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("warmup_steps must precede schedule_steps")
        for value in (self.learning_rate, self.grad_clip, self.max_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Learning rate, clipping and time limit must be finite and positive"
                )
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.condition_dropout <= 1:
            raise ValueError("Invalid EMA or condition dropout")
        if not 0 <= self.min_lr_ratio <= 1 or not self.output_dir:
            raise ValueError("Invalid minimum learning rate or output path")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto/cpu/mps/cuda")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32/fp16/bf16")
        if self.objective != "flow_velocity":
            raise ValueError("This trainer requires objective=flow_velocity")

    @classmethod
    def from_dict(
        cls,
        value: JsonDict,
    ) -> "TrainConfig":
        """从显式配置读取参数并拒绝静默忽略未知键。

        Args:
            value: 与数据路径配置分离的训练配置映射。

        Raises:
            ValueError: 出现未定义配置键或配置值不合法。
            TypeError: 参数类型不符合构造要求。

        Results:
            TrainConfig: 可序列化的训练配方。
        """
        unknown = value.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> JsonDict:
        """导出检查点及运行身份使用的完整参数。

        Results:
            JsonDict: 包括模型配置和停止上限的普通映射。
        """
        return asdict(self)


def build_model(
    specification: JsonDict,
) -> nn.Module:
    """按显式架构名构造三种共享速度接口的主干。

    Args:
        specification: architecture 和对应模型配置字段；省略架构时为 b0。

    Raises:
        ValueError: 请求未定义的架构。
        TypeError: 对应模型的配置字段不存在或类型不符。

    Results:
        nn.Module: 从随机初始化构建的速度预测模型。
    """
    options = dict(specification)
    architecture = options.pop("architecture", "b0")
    if architecture == "b0":
        from minigen.models.plain_dit import PlainConditionalDiT, PlainDiTConfig

        return PlainConditionalDiT(PlainDiTConfig(**options))
    if architecture == "b1":
        from minigen.models.single_dit import SingleDiTConfig, SingleStreamDiT

        return SingleStreamDiT(SingleDiTConfig(**options))
    if architecture == "b2":
        from minigen.models.hybrid_dit import HybridDiT, HybridDiTConfig

        return HybridDiT(HybridDiTConfig(**options))
    raise ValueError(f"Architecture {architecture!r} is not implemented")


def learning_rate_for_step(
    config: TrainConfig,
    step: int,
) -> float:
    """按零起始更新编号计算线性预热与余弦衰减。

    Args:
        config: 固定日程；恢复后不能改 schedule_steps。
        step: 本次将要执行的零起始优化更新编号。

    Raises:
        ValueError: 更新编号为负数。

    Results:
        float: 当前优化更新的学习率，超过日程后保持最小值。
    """
    if step < 0:
        raise ValueError("step must be nonnegative")
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = min((step - config.warmup_steps) / (config.schedule_steps - config.warmup_steps), 1)
    factor = (
        config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )
    return config.learning_rate * factor
