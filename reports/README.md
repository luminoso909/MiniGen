## **一、v2 报告**

| 文档 | 内容 |
| --- | --- |
| [P0 仓库审计](v2/00_repository_audit.md) | 当前代码、配置、环境、资产和测试基线 |
| [P0 迁移差异](v2/00_migration_diff.md) | 新旧正式范围、文件导航和后续迁移顺序 |
| [P1 ArtBench 审计](v2/data/artbench_audit.md) | 全量解码、数据划分、Caption 清洗及配对抽查 |
| [P0/P1 交付](v2/01_p0_p1_delivery.md) | 数据切换、旧产物清理、版本冻结与 P2 门槛 |
| [P2 工程验收](v2/02_p2_delivery.md) | VAE 用户批准、CLIP、固定 20 张本机检查与正式缓存状态 |
| [P3 本地交付](v2/03_p3_delivery.md) | B0 Flow、CLI/恢复、164 项测试与真实失败采样；CUDA 待验 |
| [P3 质量收尾](v2/03_p3_closeout.md) | 单图可辨、16 图有限对照仍未通过；区分 EMA、采样配置与训练不足 |
| [P4 工程交付](v2/04_p4_delivery.md) | 三架构主体、统一训练/采样/恢复与 180 项测试；正式训练未执行 |
| [P4 本地实验](v2/p4/local_experiments.md) | 三个约 30M 模型 MPS 8+4 步、采样与诊断的配置、曲线及限制 |
| [云端就绪清单](v2/p4/cloud_readiness.md) | 已实现接口、待 CUDA/AMP 验收、预算与正式运行前置条件 |
| [P5 工程交付](v2/05_p5_delivery.md) | 独立指标、Benchmark 候选、真实小样本、人工审核工具与完整云端手册 |
| [P5 本地实验](v2/experiments/p5_local.md) | 真实生成和评分、数值对照、性能与失败边界 |
| [P5 本地工程交接](v2/05_local_handoff.md) | 下一阶段评测实现任务、可复用产物契约及停止边界 |
| [当前交接](v2/00_handoff.md) | 阶段证据与下一项工作 |
| [OpenClipart 历史记录](v2/archive/openclipart/p0_p1_delivery_openclipart.md) | 已下线的数据试跑摘要 |

后续正式数据、组件、实验和报告在 `reports/v2/` 分阶段记录。大型运行输出放 `artifacts/`。未运行项保留待做状态。

## **二、历史实验**

- 初始化见 [bootstrap](bootstrap.md)。
- S1 见 [第一周](stage1/week1.md)、[第二周](stage1/week2.md) 与 [S1 个人核心验收](stage1/s1_acceptance_2026-10-30.md)。
- S2 见 [第三周重建与固定拟合](stage2/week3.md) 与 [第四周缓存、训练、恢复及采样](stage2/week4.md)。
- S3 见 [文本条件与评测](stage3/stage3.md)、[CLIP 恢复证据](stage3/resource_recovery.json) 与 [研究卡](research/README.md)。
- 阶段交接与历史学习任务见 [tasks](tasks/README.md)。

[最新 P5 本地交付](v2/05_p5_delivery.md) · [云端交接](v2/05_p5_handoff.md) · [当前交接](v2/00_handoff.md)
