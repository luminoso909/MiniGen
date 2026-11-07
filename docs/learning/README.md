## **一、MiniGen v2 学习入口**

学习课程 S4–S7 与工程阶段 P0–P7 分开。先完成 Flow Matching 与 ODE，再学习 Plain、Joint Attention/2D RoPE、Single-Stream、Hybrid 和研究评测。课程内容与对应工程进度见 [v2 学习目录](../MiniGen2学习目录.md)。用户启动学习回合后先交付 S4-01，后续按学习反馈逐课推进。课程参考、个人 TODO、工程集成与正式实验分别验收。

## **二、S1–S3 只读笔记**

以下五份 Notebook 保留现有的 Markdown、代码、输出、执行计数和元数据。可直接查看保存的输出。保护范围及 SHA-256 见 [P0 审计](../../reports/v2/00_repository_audit.md)。

| 阶段 | 笔记 | 学习重点与证据 |
| --- | --- | --- |
| S1 第一周 | [DDPM 原理与 MNIST 实验](S1_第1周_DDPM原理与MNIST实验.ipynb) | 加噪与后验推导、训练目标、反向采样。见 [实验](../../reports/stage1/week1.md) |
| S1 第二周 | [VAE 原理与潜空间实验](S1_第2周_VAE原理与潜空间实验.ipynb) | ELBO、KL、重参数化、先验采样和恢复。见 [实验](../../reports/stage1/week2.md) 与 [个人数学验收](../../reports/stage1/s1_acceptance_2026-10-30.md) |
| S2 第三周 | [图标数据与潜空间 DiT 基础](S2_第3周_图标数据与潜空间DiT基础.ipynb) | 数据、冻结 VAE、patch、位置与时间、adaLN-Zero。见 [几何数据、表示与拟合](../../reports/stage2/week3.md) |
| S2 第四周 | [潜空间扩散训练与采样](S2_第4周_潜空间扩散训练与采样.ipynb) | 缓存、随机时间训练、EMA、恢复、反向链和生成局限。见 [实验](../../reports/stage2/week4.md) |
| S3 | [文本条件、CFG 与可信评测](S3_文本条件CFG与可信评测.ipynb) | 文本 token、条件丢弃、CFG、条件对照与二维 Flow。见 [实验](../../reports/stage3/stage3.md) |

相关论文见 [固定版本导航](../papers/README.md)。个人复验与任务从 [learning/](../../learning/README.md) 进入。

独立运行旧课程代码或 Notebook 前，参照 [学习源码与显式加载方式](src/README.md)。
