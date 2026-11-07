## **一、v2 报告**

| 文档 | 内容 |
| --- | --- |
| [P0 仓库审计](v2/00_repository_audit.md) | 当前代码、配置、环境、资产和测试基线 |
| [P0 迁移差异](v2/00_migration_diff.md) | 新旧正式范围、文件导航和后续迁移顺序 |
| [P1 ArtBench 审计](v2/data/artbench_audit.md) | 全量解码、数据划分、Caption 清洗及配对抽查 |
| [P0/P1 交付](v2/01_p0_p1_delivery.md) | 数据切换、旧产物清理、版本冻结与 P2 门槛 |
| [P2 工程验收](v2/02_p2_delivery.md) | VAE 用户批准、CLIP、固定 20 张本机检查与正式缓存状态 |
| [当前交接](v2/00_handoff.md) | 阶段证据与下一项工作 |
| [OpenClipart 历史记录](v2/archive/openclipart/p0_p1_delivery_openclipart.md) | 已下线的数据试跑摘要 |

后续正式数据、组件、实验和报告在 `reports/v2/` 分阶段记录。大型运行输出放 `artifacts/`。未运行项保留待做状态。

## **二、历史实验**

- 初始化见 [bootstrap](bootstrap.md)。
- S1 见 [第一周](stage1/week1.md)、[第二周](stage1/week2.md) 与 [S1 个人核心验收](stage1/s1_acceptance_2026-10-30.md)。
- S2 见 [第三周重建与固定拟合](stage2/week3.md) 与 [第四周缓存、训练、恢复及采样](stage2/week4.md)。
- S3 见 [文本条件与评测](stage3/stage3.md)、[CLIP 恢复证据](stage3/resource_recovery.json) 与 [研究卡](research/README.md)。
- 阶段交接与历史学习任务见 [tasks](tasks/README.md)。

[最新 P2 工程验收](v2/02_p2_delivery.md) · [当前交接](v2/00_handoff.md)
