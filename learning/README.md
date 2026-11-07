## **一、v2 课程与个人核心**

个人课程按 S4–S7 编号。下一课 S4-01 由用户开启学习回合后交付，从线性 Flow、Euler/Heun 和匹配 NFE 开始。课程顺序、个人 TODO 和对应工程阶段见 [v2 学习目录](../docs/MiniGen2学习目录.md)。带真实输出的学习 Notebook 位于 [docs/learning/](../docs/learning/README.md)。

## **二、原有个人任务与当前证据**

| 任务 | 学习与复验 | 当前证据 |
| --- | --- | --- |
| L01 DDPM | `uv run --locked python scripts/check_ddpm_core.py` | 个人数学及 CPU 小案例通过。见 [独立验收](../reports/tasks/ddpm_review.md) |
| L02 VAE | [个人核心任务](L02_VAE核心任务.md)，`uv run --locked python scripts/check_vae_core.py` | 重参数化与 KL 已通过个人数学验收。见 [S1 验收](../reports/stage1/s1_acceptance_2026-10-30.md) |
| L03 DiT | [核心任务](L03_DiT核心任务.md)、[第四周连接任务](L03_第四周训练连接任务.md)，`uv run --locked python scripts/check_dit_core.py` | 当前工作树的 patch 顺序、初始恒等和门控梯度小案例通过。完整模型与条件 Flow 适配仍待 P3。见 [P0 审计](../reports/v2/00_repository_audit.md) |
| L04 文本条件与 CFG | [个人核心任务](L04_文本条件与CFG核心任务.md)，`uv run --locked python scripts/check_conditioning_core.py`、`uv run --locked python scripts/check_cfg_core.py` | 个人接口仍显式待实现。S3 参考与历史输出可在只读笔记查看 |
| Flow 核心 | `docs/learning/src/minigen/objectives/flow_objective.py` 与 v2 S4-01 计划 | 个人插值、速度目标和积分待实现，S3 已保存二维教学实验 |

个人核心数值检查、脱离参考的理解与主项目验收各自登记。旧任务包中的历史状态以对应实际报告和 [当前交接](../reports/v2/00_handoff.md) 为准。
