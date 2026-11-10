## **一、当前交接**

**停在用户审核 Benchmark 与云端手册、准备亲自执行 CUDA Pilot 的入口。** 2026-11-10，P5 本地主体工程与真实小样本验收已交付，完整 pytest 217 项、Ruff/格式/注释、CPU doctor、评测器 CPU/MPS 数值对照通过。未执行 CUDA、正式训练、真实 BLIP 或人工校准。[P5 阶段报告](05_p5_delivery.md) · [云端入口交接](05_p5_handoff.md) · [完整手册](../../docs/cloud/MiniGen_云端训练与评测完整操作手册.md)

ArtBench Public Domain 256x256 冻结 v1 全量 46,298 张：train 37,042、validation 4,637、test 4,619。P2 的用户 VAE 审批、固定 20 张 CPU/MPS 检查、两套正式缓存独立回读继续有效，本轮没有重建。VAE 高频线条/文字/小物体损失是已知局限，不据此删样本。[P2 验收](02_p2_delivery.md) · [正式缓存](components/artbench_cache_formal.json)

B0 为 30,359,824 参数，B1/B2 各 30,061,456；共享 Flow 和冻结组件。本轮复用 P4 step=12 检查点，生成 3×8 张验证图并实际计算 KID、CMMD、CLIPScore、LPIPS；24 张仍是不可辨噪声纹理，只证明评测工程闭环。P3-05 原固定 16 图质量未通过，单图记忆不是替代验收，P3-06 CUDA 待验。[本地实验](experiments/p5_local.md) · [P3 收尾](03_p3_closeout.md) · [P4 历史交付](04_p4_delivery.md)

候选 Benchmark 已冻结 2,000 Test 参考索引、80 ID/40 组合压力/40 配对改写、80 验证题；固定种子、组和文件隔离及资产协议哈希都有记录。**内容冻结不等于用户批准**。无可信绘画类别真值、不宣称真实 OOD；Test 只索引和完整性核查，未用于本轮指标或调参。[候选与审核重点](benchmark/README.md)

已执行：独立离线评测资产、真实 24 图评分、固定 2 图数值对照、九组 MPS 性能、人评/校准工具、8+8 待审材料、真实生成重入及报告。真实 BLIP 只实现适配器；大权重未下载。本轮收费 GPU/API 费用 0 元。保护回合前 698 个文件哈希，用户现有笔记和问答改动未纳入本次提交。

**唯一下一任务**：用户先审阅候选题目和手册，再按 [依赖清单](../../docs/cloud/云端任务清单.md) 自行上云短跑；质量、CUDA/AMP、预算及正式批准分别过关后才训练。当前没有授权启动 P6/P7 或自动长训。P4 正式训练和 P5 正式架构比较继续未完成。[待办](../../待办清单.md)

## **二、P0 历史交接**

2026-11-06（Asia/Shanghai）。P0 交付时，工作树基于 `main` 的 `7c2a3d0`，本轮未生成新提交。主 Agent 负责集成与验收，源码盘点和历史证据对照采用独立只读复核。

| 任务 | 验收与证据 |
| --- | --- |
| P0-01 Git、环境、文件与 Notebook 快照 | [仓库审计](00_repository_audit.md) 记录起点、用户改动、五个 SHA-256、当前命令和测试 |
| P0-02 v1 到 v2 代码与任务映射 | [迁移差异](00_migration_diff.md) 对照当前职责，列出 P1→P3 的实际复用模块及候选变更路径 |
| P0-03 README 主入口 | [README](../../README.md) 使用 v2 规划和任务索引，列出真实诊断、命令占位及历史课件 |
| P0-04 任务与数据更新 | [任务索引](../../MiniGen项目任务清单与学习路径.md) 归档 v1，[数据规范](../../docs/data.md) 记录当时的数据计划与 200 张重建门槛 |
| P0-05 指令、导航和交接 | [AGENTS](../../AGENTS.md)、[工作流](../../docs/ai-workflow.md)、[导航及文档变更表](00_migration_diff.md) |
| P0-06 验证与停止点 | 离线基础检查、相对链接、改动范围及五份 Notebook 的字节校验完成。唯一下一项是 P1-01 |

CPU doctor、锁文件一致性和个人 DiT 的 L03 小案例通过。完整 pytest 为 106 通过、1 失败，原因是旧 bootstrap 测试仍预期 `patchify` 抛出待实现异常。另有既有 Ruff、格式和注释问题，详见 [审计命令与结果](00_repository_audit.md)。目前的 L03 证据只覆盖小案例，正式 B0 的连续时间、文本条件和速度输出仍待验收。

五份 Notebook、历史报告和结果、两份用户规划原稿、源码、配置、旧预登记及其他已有改动保留原字节或原删除状态。复核通过 229 个相对文件链接和 P0 文档差异检查。期间出现的未跟踪 S4-01 Notebook 作为独立改动记录，课程验收另行登记。数据与模型状态以 [资产盘点](00_repository_audit.md) 为准。OpenClipart revision、有效样本数量、正式 Manifest、缓存与 run_id 待后续阶段。GPU 训练时间 0，云和 API 实际费用 0 元。

早期数据试跑归入 [OpenClipart 历史交付](archive/openclipart/p0_p1_delivery_openclipart.md)。
