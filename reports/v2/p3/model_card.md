## **一、B0 工程规格**

| 项目 | 本轮实际规格 |
| --- | --- |
| 类型 | `PlainConditionalDiT`，随机初始化，连续时间速度输出 |
| 输入 / 输出 | 潜变量 `[b,4,32,32]`，保持 ArtBench 256×256 |
| 文本 | 冻结 CLIP-B/32 的 `[b,77,512]` 与布尔 mask |
| Patch | 潜空间 2×2，256 个图像 token；手工顺序与逆变换均测试 |
| 宽度 / 层数 / 头数 | 384 / 8 / 6，MLP 扩张 4 |
| 参数量 | **30,359,824** 可训练参数；冻结 VAE/CLIP 不计入 |
| 位置 | 固定二维 sin/cos，按行优先空间排列 |
| 时间 | `[0,1]` 的连续时间；乘 1000 后进入 sin/cos 和两层 MLP |
| Block | 图像 Self-Attention → 文本 Cross-Attention → MLP，各有时间 shift/scale/gate |
| 初始化 | 九组调制与最终输出头置零，初始速度为零；分支梯度延迟已验证 |
| 注意力 | PyTorch SDPA；布尔 True 允许读取，文本 padding 在投影前清零并在 key 侧屏蔽 |
| 训练目标 | $z_t=(1-t)\epsilon+tz$，回归 $z-\epsilon$，FP32 MSE |
| CFG | $v_u+s(v_c-v_u)$；s=0/1 只计算必要分支，其他值计算两次主干 |
| 采样 | Euler / Heun，均积分噪声 0 到数据 1；分别记录速度求值与实际主干调用 |

本文件由 Agent 完成工程实现，参考已有学习成果与 [DiT 官方实现](https://github.com/facebookresearch/DiT/blob/main/models.py) 的结构思路。它不替代个人 `Flow`、`CFG` 或文本注意力 TODO 的验收，也没有复制完整教学 Scheduler。

## **二、工程边界与后续接口**

- `contracts.py` 定义 `Condition`、`FlowModel`；主干不负责 VAE、文本编码、损失或 ODE。
- `objectives/flow.py`、`samplers/flow.py` 独立；换主干不改变目标与积分器。
- `training/config.py` 的显式模型工厂目前只接受 b0，B1/B2 返回未实现错误；后续按同一输入输出协议接入。
- `training/engine.py` 使用确定数据流、AdamW、梯度累积/裁剪、EMA、独立验证 RNG。CPU/MPS 为 FP32，CUDA AMP 路径待实机验收。
- Checkpoint 绑定源文件摘要、优化配方、模型张量规格、缓存及子集身份、设备和 PyTorch；原子保存模型/EMA/优化器/scaler/日程/RNG/数据游标/历史。严格恢复要求相同设备与源码，只能延长停止上限；权重采样支持跨设备加载。
- 日志及采样保存到独立运行目录。模型可替换和梯度累积可供后续共享；当前只有单进程单设备，DDP、分布式采样器和多卡精确恢复 **尚未实现**。
- `evaluate` 当前只输出固定配对 Flow MSE 和零速度基线，**不是** P5 的生成质量评测器。

## **三、使用限制**

本轮仅用 16 张训练图诊断，不代表 37,042 张正式训练，也不证明开放域能力。验证使用前 16 张，仅作有限诊断。测试集封存。VAE 的细密线条、文字和小物体限制，以及现有 Caption 噪声继续保留。实际生成质量与阶段门槛见 [P3 交付](../03_p3_delivery.md)。
