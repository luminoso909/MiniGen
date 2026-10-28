## **一、五个核心学习任务**

这些项目接口尚未实现，调用会显式失败。接口检查只能证明没有静默 fallback；数学正确性测试将在对应学习任务中准备。Agent 不替你补完 src 主体。

知识笔记统一放 [docs/learning](../docs/learning/README.md)。第一周 [DDPM Notebook](../docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb) 已按用户授权提供完整教学参考实现，教学训练显式使用它；这不表示项目核心或用户理解已通过验收。

| 学习任务 | 源码 | 你要解决的问题 | 后续数学验收 |
| --- | --- | --- | --- |
| L01 / S1-03 | `src/minigen/objectives/ddpm_scheduler.py` | 累计信号和噪声如何计算？反向均值、方差和最后一步如何处理？ | 经验均值方差、索引边界、独立反向数值、损失梯度 |
| L02 / S1-06 | `src/minigen/autoencoders/vae_core.py` | 从均值与 log variance 如何得到可求导样本？KL 的方向和归一化是什么？ | 标准正态 KL 为零、非负性、参数梯度和维度归一化 |
| L03 / S2-04 | `src/minigen/models/dit_block.py` | 空间怎样变 token？adaLN-Zero 如何调制、门控并初始化？ | patch 往返、调制维度、预期门控与梯度 |
| L04 / S3-02 | `src/minigen/samplers/cfg.py` | 条件和无条件预测如何组合？为什么需要条件丢弃？ | 权重 0/1、相同预测、形状与求值次数 |
| L05 / S4-01 | `src/minigen/objectives/flow_objective.py` | 噪声到数据的路径、速度目标和 Euler 更新方向是什么？ | 端点、广播、解析速度场与积分误差 |

## **二、现在从 L01 开始**

1. 阅读 DDPM 的前向过程与采样算法，画出训练和采样分别使用哪些量。
2. 定义单步噪声方差和累计信号系数，推导直接加噪形式；自己算一个标量例子。
3. 让 Agent 为 L01 准备独立数值和边界测试，先说明每项测试证明什么。
4. 你完成接口，Agent 检查梯度、边界和广播，再连接训练器。

下一阶段的数学目标与公式见根目录任务清单。学习接口不额外维护参考版本，避免出现两套不同算法。

## **三、L02 第二周入口**

[VAE Notebook](../docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb) 已提供完整教学参考与真实输出。按 [L02 核心任务包](L02_VAE核心任务.md) 完成自己的两个函数；验收不会自动使用参考。S1-05 的单进程随机顺序恢复已验证，教学实验不代替 S1-06 的个人实现。

## **四、L03 第三周入口**

阅读 [第三周 Notebook](../docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb)，再合上参考完成 [L03 个人任务包](L03_DiT核心任务.md)。运行 `uv run --locked python scripts/check_dit_core.py`；当前输出待做且返回 2，禁止 fallback。

教学固定监督拟合、项目核心验收和你的理解分别登记。S1 的个人 DDPM/VAE 前置没有因为第三周工程完成而自动通过。
