## **一、当前交接**

2026-11-13 独立 B1 动漫头像实验已完成本机数据准备和 AutoDL 无卡验收。

- [x] 36,547 张去重头像按近重复组作 90/10 划分，训练 32,892 张、验证 3,655 张。35,259 个分组与 1,454 对命中近重复判定的样本均无跨集重叠。[划分回执](anime/split_receipt.json)
- [x] 本机用真实头像通过 8+8 条编码与回读 smoke，再用 MPS FP32 生成全量 SDXL VAE latent 和 CLIP Tag 缓存。FP16 latent、FP16 Tag 特征及布尔掩码共 3,183,944,948 字节，27,833 条唯一 Tag 文本。全量逐行回读通过。本机回执位于 `artifacts/cache/anime_faces_256/b1_v1/readback.json`。
- [x] AutoDL `/root/autodl-tmp/minigen` 已部署划分清单、派生缓存、三份 Python 脚本和启动器。[云端回读](anime/autodl_cache_readback.json) 与本机索引和数组 SHA-256 一致。镜像 PyTorch 2.8.0+cu128 用真实验证样本和空条件跑通 B1 CPU 前向，参数 30,061,456，优化器更新 0 次。[前向回执](anime/autodl_cpu_forward.json)
- [ ] CUDA 实机验收及正式训练。切换 GPU 模式后先确认设备与 BF16，再从远端项目根目录运行 `bash scripts/run_anime_b1_autodl.sh check --device cuda`。正式训练配方为 CUDA BF16、有效批量 64、10,000 步、每 1,000 步存档。[训练配方](../../configs/training/anime_b1_10000.json) · [AutoDL 启动器](../../scripts/run_anime_b1_autodl.sh)

本次沿用已开机的 AutoDL 无卡实例。GPU 训练时间为零，未使用收费 API，实例费用待账单。原始头像保留在本机，云端已有冻结权重和依赖继续使用。

2026-11-12 用户批准 AutoDL 上 B1、B2 同卡并行训练。两组从头各跑完 10,000 步，每 1,000 步各存一个编号快照，连同 `last.pt` 均通过回读。三架构在第 10,000 步用同 16 条验证描述、同初噪对照。B0/B1/B2 的验证 Flow MSE 分别为 0.7852、0.7698、0.7712，固定验证主体仍难稳定辨认。P4-04 与 P4-06 完成。**唯一下一步是用户审阅三模型固定验证图并决定正式评测口径。** [B1/B2 验收及三模型对照](p4/autodl_b1_b2_full_10000_2026-11-12.md)

同日使用 AutoDL 无卡实例补齐 B0 第 3,000 步与三模型第 7,000 步网格。CPU FP32 补采使用同一组固定 16 条验证描述、初噪种子 1234、EMA 权重、Euler 40 步与 CFG 1。四组输出与编号检查点、共享缓存和提示词顺序核验通过，未启动训练。报告包含 12 张网格及三模型架构、训练配方和 loss。无卡实例实际费用待账单。[B0–B2 训练对比 HTML](p4/B0-B2_训练对比.html)

同日较早完成 AutoDL 独立全量缓存与 B0 从头 10,000 步训练。缓存回读 37,042 条训练、4,637 条验证，20 个编号快照与 `last.pt` 均通过回读。2,500、5,000、10,000 步共 48 张固定验证图已生成核对。P4-02 已完成。[全量 B0 运行及验收](p4/autodl_b0_full_10000_2026-11-12.md) · [历史短训审批与诊断](p4/P4-01_B0长训审批.md)

### **1. AutoDL CUDA Pilot 和 B0 短训**

2026-11-12 用户授权使用已开启的 RTX 4080 SUPER vGPU。基于 `f0ea839`，CUDA doctor、39 项定向测试、隔离缓存 272/16 回读、B0 CPU/CUDA FP32 前反向和 BF16 逐位续训通过。BF16 50+50 Pilot 实测 137.43 图/秒、峰值 allocated 1.12 GiB。固定 16 图短训从第 500 步续到第 1,000 步，完整生成 96 张对照图。[AutoDL 短训证据](p3/autodl_pretrain_2026-11-12.md) · [云端任务清单](../../docs/cloud/云端任务清单.md)

这次短训结束时，P3-05 的质量验收未通过。训练 loss 下降，验证 loss 上升，多数样图呈块状伪影，验证图未见稳定条件收益。后续获批的全量运行和独立 CUDA 缓存见上文。远端保留 `artifacts/runs/autodl-b0-pretrain-20261112/`。短训实际费用待账单。

### **2. 本轮 CUDA 工作前的记录**

2026-11-12 AutoDL 无卡准备已验收：仓库已部署到 `autodl:/root/autodl-tmp/minigen`（`f0ea839`）。远端 CPU doctor、锁定 SDXL VAE/CLIP 文本塔核验通过；ArtBench 46,298 张已复现冻结 Manifest，并逐张回读校验 SHA-256；Benchmark 候选内容锁一致，人工审批仍待定；16 项 CPU 定向测试通过。[AutoDL 逐项状态和日志路径](../../docs/cloud/MiniGen_AutoDL无卡准备.md)。Mac 负责改代码并用 rsync 同步，Python 测试、训练和评测走 SSH；不从 Mac 上传大权重、数据集或虚拟环境。此轮未开启 GPU、训练或正式评测，正式缓存未迁移或重建。

2026-11-12 较早的 Mac 复核：基于 `f0ea839`，46,298 张图哈希、37,042/4,637 条正式缓存回读、27 项定向测试和 Benchmark 候选锁通过；Benchmark 仍待人工批准，B0 固定 16 图质量仍未通过。[复核与逐项命令](../../docs/cloud/MiniGen_云端逐项验收.md) · [云端验收脚本](../../scripts/check_cloud_readiness.sh)。这些结果不替代 AutoDL 的实测或迁移。

2026-11-10 完成基于 `77b7506` 的最终本地审查与工程交接，当时停在等待用户审核 Benchmark、实验配置和云端操作手册。[最终工程摘要](final/P3_P4_P5工程最终验收摘要.md) · [执行前清单](../../docs/cloud/MiniGen_云端执行前检查清单.md)

P3/P4 主干、Flow、采样、训练与恢复代码未改，B0 30,359,824、B1/B2 各 30,061,456 参数。本轮复用此前完整 217 项测试和真实 P4/P5 运行，只补 12 项 CPU 训练/恢复、14 项 VQA/校准/评分和 1 项配置防护定向测试，以及三模型旧评分的实际报告回放。唯一源码修复是数量/对象 VQA 问句语法，已补四个类别契约案例；不是模型质量修复。旧回执保持原源码身份。

ArtBench 46,298 张（train 37,042/validation 4,637/test 4,619）、256×256、已批准 SDXL VAE/CLIP 与 P2 正式缓存不变、不重建。[P2 证据](02_p2_delivery.md)。B0 单图可辨，但固定 16 图质量仍未过；P5 24 张烟测仍为噪声，不能解除长训门槛。[P3 收尾](03_p3_closeout.md) · [P4 交付](04_p4_delivery.md) · [P5 交付](05_p5_delivery.md)

[Benchmark 最终审核包](benchmark/MiniGen_Benchmark_最终审核包.md) 检查全部 240 题。候选数量/哈希/跨集合隔离通过，但存在题文数量/场景未评分、颜色问句重复、同色绑定和计数边界；2,000 条参考只有 1,999 唯一图片/组。**旧候选仍 pending，不建议直接批准为全文遵循或真实 OOD 基准。** 用户决定发布修订候选或明确限定评分范围后再批准，不按最终效果改题。

[正式实验预登记](final/MiniGen_正式实验预登记.md) 绑定现有资产与模型，硬件、最终步数/精度/LR/EMA/预算继续待实测；[完整手册](../../docs/cloud/MiniGen_云端训练与评测完整操作手册.md) 已复核真实命令、全16图条件对照、三模型恢复和单卡边界。CUDA/AMP、Linux安装、真实 BLIP、100+100人评与正式训练均未执行，DDP未实现。源码变化会影响精确恢复；云端须固定本轮提交新建Pilot，不绕过检查。

上述本地审查未使用收费 GPU/API。Git 当时未包含大资产或用户已有笔记改动，未 Push，远端只读核实为 `54345ca`。[Git 同步说明](final/Git提交与云端迁移说明.md)

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
