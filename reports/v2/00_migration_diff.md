## **一、v1 到 v2 的范围变化**

以下为 P0 时的迁移记录。数据路线之后切换到 ArtBench，当前入口与验收见 [P0/P1 交付](01_p0_p1_delivery.md)。

正式方案唯一依据为 [用户的 v2 重构规划](../../docs/MiniGen2重构规划.md)。[任务索引](../../MiniGen项目任务清单与学习路径.md) 跟踪实际进度，折叠保留 v1 原文和原编号。[课程目录](../../docs/MiniGen2学习目录.md) 单独定义个人学习。

| 领域 | v1 历史路线 | v2 正式路线 |
| --- | --- | --- |
| 研究变量 | 合成几何域上比较 DDPM 与 Flow 配方，另有 Sana LoRA 适配计划 | 固定 Flow 速度目标，比较从随机初始化训练的 B0 Plain、B1 Single-Stream、B2 Hybrid |
| 数据 | MNIST 与 128px 几何教学，OpenMoji 为 LoRA 候选 | 仅使用经审计的 OpenClipart，统一 256px 卡通插画；先分组切分，再完成离线标注与人工审核 |
| 表示 | 旧几何域冻结 SDXL VAE 与最多双段的 CLIP 文本缓存 | 同一候选冻结组件在新域重验。随机 200 张 VAE 重建含 PSNR、SSIM、LPIPS 和人审，CLIP 采用单段 token 特征 |
| 参数与优化 | 小型教学主干，记录固定拟合和有限短训 | 每个主干目标约 30M，可训练层数以实际计数冻结。先 B0 小案例和单卡先导，再逐个加入 B1/B2 |
| 采样与控制变量 | DDPM/DDIM 与噪声预测 CFG | 连续时间从噪声到数据，速度 CFG，按分支等价 NFE 匹配 Euler/Heun。同更新步数与同训练算力分开报告 |
| 评测 | 几何解析器、24 条验证提示词与人工待审 | MiniGen-Bench 正式提示词提前冻结，ID/组合外/改写各自统计，分布参考图单独切分，自动 VQA 与人工盲审校准 |
| 扩展 | Sana/OpenMoji、旧四组训练、原日期与预算 | 旧目标归档。额外种子、10M/50M、Chimera 由 P5 后的结果决定，费用按实测吞吐与用户批准冻结 |

最小成功是 OpenClipart 上的 256px B0+Flow 可训练、可从噪声采样、可恢复、可评测并记录成本。计划成功是三主干共享协议完成可比训练和评测。S1–S3 的 Notebook、实验结果、代码与旧预登记保持其历史身份。

## **二、执行顺序与解释口径**

- 工程阶段按重构规划第 9 章 P0–P7。第 7 章 A/D 组的「P0 必做」、B/C 的「P1 可选」与 E 的「P2」按实验优先级理解。A 组正式训练在工程 P4，D 组对照在 P5，B/C/E 扩展在 P6 另行批准。
- P0 本轮到仓库审计、文档迁移、基础检查及 P1-01 计划为止。用户提供的规格留在原 `docs/` 路径，README 指向唯一正式规划。
- S4-01 是首个获准学习回合的产物。S4–S7 的课程序号与工程阶段无继承关系，个人 TODO 与来源明确的工程代码分别验证，长训前分别记录个人理解/测试、工程短跑、数据与组件、预算批准。
- Benchmark 在正式训练前冻结，验证提示词用于选型，正式测试待可比检查点与评分门槛满足后执行。数据规模、最终层数、训练步数、GPU 型号和费用在相应验收完成后冻结。

## **三、P1 到 P3 的最小源码迁移顺序**

下表列出候选路径和已存在的接口。所有候选路径将在对应阶段按实际代码评审后创建或修改。现状详见 [仓库审计](00_repository_audit.md)。

| 顺序 | 候选新增文件或适配位置 | 最小变更与门槛 |
| --- | --- | --- |
| P1-01 来源和样本审计 | 新 `scripts/audit_openclipart.py`、`configs/datasets/openclipart_v2.yaml`、`src/minigen/data/openclipart.py`、`tests/test_openclipart_data.py` | 脚本和配置明确 dry-run、revision、来源、限制与安全预览。先有界验证访问方式并抽样，输出证据后才决定批量路线 |
| P1-02 至 P1-05 风格与标注 | 扩展 `src/minigen/data/openclipart.py`，按需要新增 `src/minigen/data/captioning.py`，随后更新 `configs/datasets/sources.yaml`，新建 `configs/examples/openclipart_v2_manifest.example.jsonl` | 复用 `data/icons.py::load_manifest` 中的校验原则，旧 loader 保留几何字段与行为。冻结新 schema、来源/近重分组、划分和 Caption |
| P2-01 至 P2-02 表示验收 | 新 `src/minigen/evaluation/vae_reconstruction.py`、`tests/test_v2_components.py`；复用 `src/minigen/autoencoders/frozen.py` 的编解码接口 | 200 张随机重建独立报告，先检查渲染、缩放与精度，再提交人工验收。确需更换组件时统一评估 |
| P2-03 至 P2-04 文本与缓存 | 适配 `src/minigen/autoencoders/frozen.py`、`src/minigen/data/latent.py`、`src/minigen/conditioning/text.py`；新 `configs/cache/openclipart_v2.yaml` | 显式 256px 处理版本，使用 CLIP `max_chunks=1`，冻结新 manifest、scale、mask、dtype、预处理与文件身份。旧默认值和历史缓存保持各自版本 |
| P3-01 至 P3-03 Flow 与 B0 | 新 `src/minigen/objectives/flow.py`、`src/minigen/samplers/flow.py`、`src/minigen/models/plain_dit.py`、`tests/test_v2_flow.py`、`tests/test_v2_dit.py` | 明确个人接口与工程实现的来源。验收连续时间、端点、Euler/Heun、速度 CFG、图文 mask、梯度、参数量和模型尺寸。评审后复用 `dit_block.py` 的已验证部件 |
| P3-04 至 P3-06 训练闭环 | 新 `src/minigen/training/flow.py`、`configs/experiments/v2_b0_plain_flow.yaml`、`tests/test_v2_training.py`；增量修改 `src/minigen/cli.py` 与必要的 bootstrap 断言 | 复用 EMA、RNG、`StatefulBatchStream` 和检查点通用部件。绑定 Flow、缓存与模型身份，先 CPU 冒烟、过拟合与采样恢复，再做经批准的单卡短跑及成本先导 |

P4 才依次加入候选 `src/minigen/models/single_stream_dit.py` 和 `src/minigen/models/hybrid_dit.py`。共同数据、恢复、训练及评测链路只维护一套。P5 新建插画评分与报告入口，历史 `scripts/evaluate_stage3_holdout.py` 仍属于旧几何任务。

## **四、本轮文档调整**

| 位置 | 变更 |
| --- | --- |
| [README](../../README.md)、[根任务索引](../../MiniGen项目任务清单与学习路径.md) | 切到 v2，列出现有命令与计划命令，旧路线归档 |
| [数据规范](../../docs/data.md)、[配置导航](../../configs/README.md) | 补齐新域标注、分组、硬门槛与缓存规则，保留旧配置身份 |
| [AGENTS](../../AGENTS.md)、[工作流](../../docs/ai-workflow.md) | 明确启动权限、历史保护、验收与费用，保留 Python 类型、形状、注释和 Git 约定 |
| [文档](../../docs/README.md)、[笔记](../../docs/learning/README.md)、[个人任务](../../learning/README.md)、[论文](../../docs/papers/README.md)、[报告](../README.md)、[历史交接](../tasks/README.md) | 导向同一 v2 入口，五份历史笔记保留只读访问和相应证据 |
| [审计](00_repository_audit.md)、[交接](00_handoff.md) | 登记工作树、实际检查、未通过项和唯一下一任务 |
