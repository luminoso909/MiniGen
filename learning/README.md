## **一、五个核心学习任务**

DDPM 个人核心已完成，独立验收入口为 `uv run --locked python scripts/check_ddpm_core.py`，证据见 [验收报告](../reports/tasks/ddpm_review.md)。其余四个个人核心尚未实现，调用会显式失败；Agent 不替你补完 src 主体。

知识笔记统一放 [docs/learning](../docs/learning/README.md)。第一周 [DDPM Notebook](../docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb) 已按用户授权提供完整教学参考实现，教学训练显式使用它；这不表示项目核心或用户理解已通过验收。

| 学习任务 | 源码 | 你要解决的问题 | 后续数学验收 |
| --- | --- | --- | --- |
| L01 / S1-03 | `src/minigen/objectives/ddpm_scheduler.py` | 累计信号和噪声如何计算？反向均值、方差和最后一步如何处理？ | 经验均值方差、索引边界、独立反向数值、损失梯度 |
| L02 / S1-06 | `src/minigen/autoencoders/vae_core.py` | 从均值与 log variance 如何得到可求导样本？KL 的方向和归一化是什么？ | 标准正态 KL 为零、非负性、参数梯度和维度归一化 |
| L03 / S2-04 | `src/minigen/models/dit_block.py` | 空间怎样变 token？adaLN-Zero 如何调制、门控并初始化？ | patch 往返、调制维度、预期门控与梯度 |
| L04 / S3-02 | `src/minigen/samplers/cfg.py` | 条件和无条件预测如何组合？为什么需要条件丢弃？ | 权重 0/1、相同预测、形状与求值次数 |
| L05 / S4-01 | `src/minigen/objectives/flow_objective.py` | 噪声到数据的路径、速度目标和 Euler 更新方向是什么？ | 端点、广播、解析速度场与积分误差 |

## **二、L01 代码验收后复盘**

1. 阅读 DDPM 的前向过程与采样算法，画出训练和采样分别使用哪些量。
2. 定义单步噪声方差和累计信号系数，推导直接加噪形式；自己算一个标量例子。
3. 运行个人验收脚本，解释独立高斯精度对照、经验分布、均值误差和梯度分别证明什么。
4. 代码已通过验收；接下来不看实现解释零起始时间、最终步和裁剪，个人理解另行登记。

下一阶段的数学目标与公式见根目录任务清单。学习接口不额外维护参考版本，避免出现两套不同算法。

## **三、L02 第二周入口**

[VAE Notebook](../docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb) 已提供完整教学参考与真实输出。按 [L02 核心任务包](L02_VAE核心任务.md) 完成自己的两个函数；验收不会自动使用参考。S1-05 的单进程随机顺序恢复已验证，教学实验不代替 S1-06 的个人实现。

## **四、L03 第三周入口**

阅读 [第三周 Notebook](../docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb)，再合上参考完成 [L03 个人任务包](L03_DiT核心任务.md)。运行 `uv run --locked python scripts/check_dit_core.py`；当前输出待做且返回 2，禁止 fallback。

教学固定监督拟合、项目核心验收和你的理解分别登记。DDPM 代码由后续独立验收确认通过；VAE 个人核心与两项理解验收仍待做。

## **五、L03 第四周连接入口**

[第四周连接任务](L03_第四周训练连接任务.md) 说明个人 DiT 验收后怎样接入缓存、噪声目标、训练器、恢复与完整采样。教学版本先在 [第四周 Notebook](../docs/learning/S2_第4周_潜空间扩散训练与采样.ipynb) 显式复用第三周教学类，不改写 src 个人核心。

先读懂“缓存与缩放 → 随机时间训练 → 分桶验证 → EMA/恢复 → 完整反向采样”的数据流，再自行实现 L03。你当前的第二周 VAE 作业继续按 L02 完成；本周不把教学分支的成功记作 VAE、DiT 或个人理解已通过。

建议 Git 提交名为 `feat(s2): 添加第四周潜空间扩散训练与采样实验`；本轮不执行暂存、提交或推送，待你完成作业后自行整理范围。

## **六、S3 接手任务**

先通过 L03，再打开 [L04 文本条件与 CFG](L04_文本条件与CFG核心任务.md)。完整教材与实验统一放 [S3 Notebook](../docs/learning/S3_文本条件CFG与可信评测.ipynb)。Agent 已准备冻结文本缓存、工程与独立测试；你的重点是注意力、mask、CFG、整体数据流和研究结论。

个人命令：`uv run --locked python scripts/check_conditioning_core.py`、`uv run --locked python scripts/check_cfg_core.py`。未实现返回待做和退出码 2，不会走教学参考。S1 的 VAE 已通过代码验收，理解与材料收尾分别记录。
