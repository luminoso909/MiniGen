## **一、MiniGen v2.0**

MiniGen 在同一套 ArtBench Public Domain 256x256 数据的训练切分上，从随机初始化训练三个约 30M 可训练参数的文本条件生成主干。三者共享冻结的 SDXL VAE、CLIP-B/32 文本塔和 Flow Matching 速度目标，比较画质、提示词遵循与计算成本。

| 主干 | 图文条件路径 | 顺序 |
| --- | --- | --- |
| B0 PlainConditionalDiT | 图像自注意力读取图像 token，Cross-Attention 读取文本 token | 先验证数据、训练、采样和恢复，再正式训练 |
| B1 SingleStreamDiT | 图文 token 拼接，共用 Transformer 块 | B0 稳定后接入相同训练接口 |
| B2 HybridDiT | 前段保留图文双流，通过 Joint Attention 交互，后段合流 | B1 后接入相同训练接口 |

正式范围、阶段门槛和实验矩阵以 [MiniGen v2 重构规划](docs/MiniGen2重构规划.md) 为准。阶段状态见 [任务索引](MiniGen项目任务清单与学习路径.md)，阶段证据和下一项工作见 [当前交接](reports/v2/00_handoff.md)。

## **二、当前可用功能与验证**

| 功能 | 当前状态 |
| --- | --- |
| `minigen doctor` | 可用。支持 `--device` 和 `--json`，检查依赖与小张量前向、反向 |
| `minigen train`、`sample`、`evaluate`、`cache` | 已可用：B0 Flow 训练/恢复、Euler/Heun 采样、固定配对 MSE 诊断、正式缓存构建/回读；P5 图像质量评测尚未实现 |
| `minigen data`、`report` | 计划中的入口，尚未注册 |
| 数据、冻结组件与训练工具 | ArtBench 46,298 张已接入，离线数据准备使用 `scripts/prepare_artbench.py`。冻结 VAE、CLIP 和 train/validation 正式缓存已完成 [P2 验收](reports/v2/02_p2_delivery.md)。P3 本地工程已交付，停在 P4 入口；1000 步短训可辨质量未通过，CUDA 待验 |
| 个人学习核心 | DDPM、VAE 数学已验收。当前 DiT 的 L03 独立小案例通过。文本 Cross-Attention、CFG、Flow 仍待实现 |

项目使用 Python 3.12.14 和 uv。初始化或恢复依赖时使用 `uv sync --locked`。已有环境可以执行以下离线检查：

```bash
uv run --locked --offline --no-sync minigen --help
uv run --locked --offline --no-sync minigen doctor --device cpu --json
uv run --locked --offline --no-sync pytest -p no:cacheprovider
```

P0 的 CPU 诊断和独立 L03 检查通过。旧 DiT 启动测试已修复。历史环境基线和最新验收分别见 [仓库审计](reports/v2/00_repository_audit.md)、[P2 工程交付](reports/v2/02_p2_delivery.md) 与 [P3 工程交付](reports/v2/03_p3_delivery.md)。当前完整 pytest 164 通过；CPU/MPS 数值和本地训练验收与将来的 CUDA/AMP 分开记录。

B0 实际参数量 **30,359,824**。本地 16 图、1000 步损失下降，但固定采样主体仍不可辨；当前属于可靠工程入口与真实失败实验，尚不是合格艺术图像生成模型。[训练曲线与全部对照](reports/v2/03_p3_delivery.md) · [模型规格](reports/v2/p3/model_card.md)。

以下命令需要已验收的 P2 缓存；环境已有可选组件时使用 `--no-sync` 保留它们。新实验先复制配置并设置新的 `output_dir`，避免覆盖既有结果：

```bash
uv run --locked --offline --no-sync minigen train --config configs/training/p3_tiny.json --device cpu
uv run --locked --offline --no-sync minigen train --config configs/training/p3_local.json --device mps
uv run --locked --offline --no-sync minigen sample --config configs/training/p3_local.json --checkpoint artifacts/runs/p3_local/last.pt --output artifacts/runs/example_samples --device mps --count 8 --steps 40
```

恢复、条件对照和 CUDA Pilot 见 [P3 命令与边界](reports/v2/03_p3_delivery.md)。使用 CPU 可显式改 `--device cpu`；CUDA 代码分支已预留但本轮没有实机验收，多卡尚未实现。

## **三、数据、实验与协作**

- 正式数据采用 ArtBench Public Domain 256x256。三份固定 Parquet 共 46,298 张，按组分为训练、验证和测试，保留原生 256×256 RGB。使用清洗后的英文 Caption；图文语义噪声见实际审计。[P0/P1 交付](reports/v2/01_p0_p1_delivery.md) · [数据管理](docs/data.md) · [P0–P7 下载清单](docs/MiniGen2下载清单.md)
- P2 保留随机 200 张重建统计、9 张诊断与 LPIPS 最差 20 张；用户已接受 SDXL VAE 用于普通绘画。正式缓存依赖本机固定 20 张数值检查，可用通过检查的 MPS 或 CPU，CUDA 训练运行另行验收。P3 先做 B0 小样本验证与单卡成本试验，训练预算由你审批。
- MiniGen-Bench 在正式训练前冻结提示词与对照协议，分别报告图像质量、文本遵循、质量与算力曲线。完整矩阵见 [重构规划第 7–8 章](docs/MiniGen2重构规划.md)。
- 工程阶段用 P0–P7 编号，个人课程用 S4–S7 编号。课件从 S4-01 开始，按反馈逐课交付。来源明确的工程实现、课程参考与个人 TODO 分别验收。见 [学习目录](docs/MiniGen2学习目录.md)、[个人任务入口](learning/README.md) 和 [AI 工作流](docs/ai-workflow.md)。

## **四、S1–S3 历史学习与实验**

| 阶段 | 只读 Notebook | 实际证据 |
| --- | --- | --- |
| S1 第一周 | [DDPM 原理与 MNIST 实验](docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb) | [短训与采样](reports/stage1/week1.md) |
| S1 第二周 | [VAE 原理与潜空间实验](docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb) | [教学实验](reports/stage1/week2.md)、[个人数学验收](reports/stage1/s1_acceptance_2026-10-30.md) |
| S2 第三周 | [图标数据与潜空间 DiT 基础](docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb) | [几何数据、重建与固定监督拟合](reports/stage2/week3.md) |
| S2 第四周 | [潜空间扩散训练与采样](docs/learning/S2_第4周_潜空间扩散训练与采样.ipynb) | [训练与恢复](reports/stage2/week4.md)，8 张样本中 6 张解析失败 |
| S3 | [文本条件、CFG 与可信评测](docs/learning/S3_文本条件CFG与可信评测.ipynb) | [两种子实验](reports/stage3/stage3.md)，正确条件成功 2/24、1/24，人工评测待验且正式测试封存 |

旧 DDPM 对 Flow 的正式比较和 Sana/OpenMoji/LoRA 方案已归档。相关计划与预研分别见 [旧预登记](configs/experiments/s4_preregistered.yaml) 和 [Sana 预研](reports/research/s3_sana_feasibility.md)。

正式包位于 `src/minigen/`：`data/` 管数据与表示缓存，`autoencoders/` 和 `conditioning/` 管冻结编码器，`evaluation/` 管独立审计，`models/`、`objectives/`、`samplers/`、`training/` 分别管理速度主干、目标、积分与恢复训练。旧课程实现已移至 [学习源码](docs/learning/src/README.md)；正式包不加载它们。测试位于 `tests/`。查看 [配置导航](configs/README.md)、[学习导航](docs/learning/README.md)、[论文导航](docs/papers/README.md) 与 [报告导航](reports/README.md)。实际数据和缓存分别放 `data/` 与 `artifacts/`，由 Git 忽略规则管理。
