## **一、当前交接**

ArtBench Public Domain 256x256 冻结版 `v1` 含 46,298 张，train 37,042、validation 4,637、test 4,619。P2 已完成 SDXL VAE 随机 200 张重建与 9 张诊断、CLIP 单段条件验收、256px 配对缓存试点。用户批准保留当前冻结 VAE；细密线条、文字和小物体的高频损失记为局限，不据此剔除样本。[用户批准](components/vae_user_approval.json) · [含 LPIPS 最差 20 张的审核证据](components/vae_review_evidence.json)

**P2 已验收。** 固定 20 张的本机 CPU FP32 对照通过；VAE 和 CLIP 均使用 MPS FP32 构建正式缓存。train 37,042 行、validation 4,637 行全部通过独立回读；test 的 4,619 张保持封存。本地缓存位于 `artifacts/cache/artbench_pd_256/v1_formal`，缓存数组未入 Git。[P2 工程验收](02_p2_delivery.md) · [正式缓存回执](components/artbench_cache_formal.json) · [本机固定 20 张回执](components/artbench_local_fp32_20.json) · [P0/P1 交付](01_p0_p1_delivery.md)

**停在 P4 入口，等待用户下一轮指令。** 2026-11-10 已完成 P3-01～04 本地工程：统一 Flow/CFG/Euler/Heun、30,359,824 参数 B0、真实 CLI、Checkpoint 和恢复。164 项测试、CPU/MPS FP32 对照通过；16 张训练子集完成 500→1000 步真实恢复，循环累计 151.465 秒。固定采样仍是不可辨色块和重复纹理，**P3-05 质量未通过**；P3-06 入口/配置已准备，**CUDA/AMP/性能待云端验收**，不能称整个 P3 全部通过。[P3 阶段报告](03_p3_delivery.md) · [模型卡](p3/model_card.md) · [CUDA Pilot](p3/cuda_pilot.md)

已执行：固定 8 张正确/空/打乱条件、500 步在线与 EMA、1000 步等 NFE Euler/Heun 对照。失败：可辨生成门槛；验证子集正确文本 MSE 不优于空条件。未执行：正式全量长训、B1/B2、P5 指标和 CUDA。实际收费 GPU/API 费用 0 元。权重与大型缓存未入 Git，历史课件及用户已有改动保留。

**唯一下一任务**：用户启动 P4 工程回合后，先确认 B0 失败诊断的修复范围，再按同一接口接入 B1/B2 的本地实现与测试。按最新授权，P3–P5 主体代码先于统一云端训练推进，但质量与预算审批仍独立保留。本轮不自行进入 P4/P5。正式条件训练前仍需决定如何处理 Caption 噪声（P1 的 240 对中有 71 条明确事实错误）。[重构规划](../../docs/MiniGen2重构规划.md) · [待办清单](../../待办清单.md)

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
