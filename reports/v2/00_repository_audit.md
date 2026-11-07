## **一、仓库与保护基线**

审计日期为 2026-11-06（Asia/Shanghai）。工作树位于 MiniGen，分支 `main`，起点 `7c2a3d006c5da615115638d3a5852d845757c985`，暂存区为空。审计以本地工作树为准，历史状态取自原报告。P0 的改动是工作树中的文档，阶段、命令与当前交接见 [迁移差异](00_migration_diff.md) 和 [交接](00_handoff.md)。

开始前快照覆盖 Git 已跟踪文件（含已删除路径）和未忽略的未跟踪文件，共 198 个路径。已有 19 个跟踪状态和 8 个未跟踪文件。

| 已有用户改动 | 保护方式 |
| --- | --- |
| S1 第二周、S2 第三和第四周 Notebook | 按当前工作树原始字节取哈希，包含用户改过的输出和元数据 |
| `src/minigen/models/dit_block.py`、`tiny_vae.py`、`pyproject.toml`、`uv.lock` | 保留源码、依赖与锁文件。当前依赖新增 `einops>=0.8.2`，锁定 0.8.2 |
| `reports/stage2/week3_results.json` | 保留已有工作树内容。S2 历史实验结论从当时的报告读取 |
| `docs/learning/week1QA/` 的 11 个删除、`docs/QA/` 的 6 个未跟踪文件 | 保留删除及新目录状态 |
| `docs/MiniGen2重构规划.md`、`docs/MiniGen2学习目录.md` | 保留两份用户原稿及未跟踪状态，新增导航引用原路径 |

复核时工作树另外出现了未跟踪的 `docs/learning/S4_01_FlowMatching原理与ODE采样.ipynb`。这是独立工作树变更，P0 仅记录路径和复核时的 SHA-256 `187e6c98ee0259e8fb2d4f3c2e05267a779acd2f9b487dda63f5c9f71dd69633`，其课程内容与执行状态未纳入本次验收。

### **1. 五份只读 Notebook 的 SHA-256**

这五份文件以 P0 启动时的字节为基线。后续每回合复核这组哈希。

| `docs/learning/` 下的文件 | SHA-256 |
| --- | --- |
| `S1_第1周_DDPM原理与MNIST实验.ipynb` | `b9d21a00094821dbe62966269a119c23857290fa555e09bcdb912692141f9aad` |
| `S1_第2周_VAE原理与潜空间实验.ipynb` | `847b1616649db3921e27078a6e380be6ef97be7d74cf1a01854b01857bbc1313` |
| `S2_第3周_图标数据与潜空间DiT基础.ipynb` | `e3d16df5756fbcde34eec5018b16ebec57f2794e198e0563c97dcc160c7ef46d` |
| `S2_第4周_潜空间扩散训练与采样.ipynb` | `a8bc8a6aa6056c998b31ecb35712689d82c75b14bcd6f3af95a3e75b0cea6812` |
| `S3_文本条件CFG与可信评测.ipynb` | `8e68b87abb910eebbf995490524d69427af4a97749a048bbb75e5f89f2af3993` |

## **二、可复用代码与迁移任务**

表中的现有路径相对于仓库根目录。候选新增和适配路径见 [迁移顺序](00_migration_diff.md)。

| 现有文件与入口 | 实际用途和状态 | v2 处理与任务 |
| --- | --- | --- |
| `src/minigen/cli.py::main`、`doctor.py::diagnose`、`__main__.py` | `doctor` 真实可用，`train/sample/evaluate/cache` 均返回占位退出码 2，`data/report` 未注册。P0 实测确认 | 保留入口和诊断。P1-01 先用显式审计脚本，P3 逐项接入真实训练、采样和缓存，P5/P7 补评测与报告 |
| `src/minigen/data/mnist.py::prepare_mnist`、`data/icons.py::build_icons/audit_icons/select_validation` | MNIST 与几何图标的数据和可控校验 | 保留教学与回归。P1 新建 OpenClipart 审计、渲染、Caption 和分组实现 |
| `src/minigen/data/icons.py::load_manifest`、`configs/datasets/sources.yaml`、`configs/examples/manifest.example.jsonl` | 几何清单依赖 `caption/sha256/mode` 等旧字段，已有路径、样本 ID、示例记录检查 | P1 复用验证原则并定义版本化 schema。OpenClipart 的 `caption_en/rendered_sha256` 通过明确适配传给共享层 |
| `src/minigen/autoencoders/frozen.py::FrozenVAE.load/encode/decode`、`image_batch/to_image/cache_identity`、`configs/models/sdxl_vae.json` | 固定本地组件身份、后验均值和乘除 scale。`cache_identity` 仍绑定 128px 预处理 | P2 先用现有组件做 200 张新域重建，随后显式加入 256px 预处理和新身份 |
| `src/minigen/data/latent.py::latent_cache_identity/build_latent_cache/CachedLatentDataset` | 分片写入、恢复、载荷/身份验证，旧入口按 train/validation 使用 | P2 适配版本、尺寸、渲染哈希和 split 契约。保留旧 4×16×16 缓存，正式目标按实测 4×32×32 |
| `src/minigen/conditioning/text.py::FrozenTextEncoder.load/encode/text_cache`、`configs/models/clip_text.json` | 冻结文本塔、token 特征、有效位 mask 与空文本，支持 `max_chunks=1`，旧默认值 2 | P2 显式选择单段，重验 75+2 边界、拒绝截断、mask 与空条件，缓存身份绑定新 Caption |
| `src/minigen/models/dit_block.py::sincos_1d/position_2d/patchify/unpatchify/DiTBlock/DiffusionTransformer`、`scripts/check_dit_core.py::verify` | 工作树已有个人实现及固定网格的无条件骨干，完整模型接收离散时间且使用加法位置编码。P0 的 L03 小案例通过 | P3 单独验收可复用 patch、调制与梯度，再接连续时间、文本与速度头。检查同 token 数而网格不同的尺寸拒绝、mask 和 SDPA。完整骨干尚未验收为 B0 |
| `src/minigen/conditioning/cross_attention.py::TextCrossAttention`、`samplers/cfg.py::classifier_free_guidance`、`objectives/flow_objective.py::linear_path/velocity_loss/euler_step` | 个人接口显式报待实现，S3 教学参考及二维演示另存 Notebook | P3 定义带来源的工程契约与独立测试。学习 TODO 保持独立。Flow 端点、速度、CFG、Euler/Heun 需新验收 |
| `src/minigen/objectives/ddpm_scheduler.py::DDPMScheduler`、`samplers/ddpm_loop.py::sample_ddpm`、`autoencoders/vae_core.py`、`models/tiny_unet.py` 与 `tiny_vae.py` | S1 个人数学核心与教学训练所用部件 | 保留旧算法与验证，不接入 v2 正式速度目标 |
| `src/minigen/training/denoising.py::TrainConfig/train_denoiser/save_checkpoint`、`data/stateful.py::StatefulBatchStream`、`training/latent.py::train_latent_denoiser` | 已有优化、EMA、种子、数据顺序与恢复组件。基础训练器日程校验较弱，latent 包装层加强身份校验 | P3 复用通用恢复逻辑，显式绑定 Flow objective、架构、配置、清单、缓存与缩放身份 |
| `src/minigen/training/conditional.py::train_conditional`、`samplers/conditional.py::sample_conditional/synchronize` | 显式接收模型、回调与 CFG，现有路径绑定离散 DDPM 训练和 DDPM/DDIM 采样 | P3 复用训练记账、同步与条件分支计数，Flow 路径独立核对连续时间和 NFE |
| `src/minigen/evaluation/latent.py::reconstruct`、`evaluation/icons.py::parse_icons/score_image`、`evaluation/conditional.py::score_request/summarize/prepare_blind_review/merge_human_review` | 几何解析、请求计分、分母与盲审工具。旧封存测试在 `scripts/evaluate_stage3_holdout.py` | P2 新增重建指标与人工细节检查。P5 复用匿名与分母原则，OpenClipart 评分单独校准 |
| `scripts/prepare_week3_assets.py`、`prepare_stage3_assets.py`、`execute_learning.py` | 旧固定资源准备、CLIP 缺失恢复与教学 Notebook 运行工具 | 保留已有行为。P2 可复用严格校验与离线加载，P0 只核对历史证据与组件文件元数据 |
| `tests/test_bootstrap.py`、`test_ddpm_core.py`、`test_week2.py` 至 `test_week4_engineering.py`、`test_stage3.py`、`.github/workflows/ci.yml` | 现有数学、教学和工程回归。旧 bootstrap 假设与当前工作树 DiT 实现有冲突 | P0 保留基线并报告。后续分别维护已实现 DiT 的正向契约与仍待实现接口的失败契约，添加 v2 单测 |

## **三、历史实验和本地资源**

| 阶段 | 记录中的真实结果 | 对 v2 的作用 |
| --- | --- | --- |
| S1 | DDPM 数学与 CPU 工程小案例、VAE 重参数化与 KL 通过个人代码验收。教学训练、理解与整阶段状态独立记录。见 [S1 验收](../stage1/s1_acceptance_2026-10-30.md) | 复用数值验证及恢复方法 |
| S2 第三周 | 14,000 张几何图，96/96 压力样本重建属性通过，256 张试点缓存。663,952 参数模型固定拟合 100 步，7,372,560 参数规格仅前向。见 [报告](../stage2/week3.md) | 复用验证流程，表示和数据需在 256px 插画域重验 |
| S2 第四周 | 10,000 训练和 1,000 验证缓存，663,952 参数教学模型实际恢复至 1,000 步，完整采样 8 张中 6 张解析失败。见 [报告](../stage2/week4.md) | 复用分片、恢复与证据记录，保留失败样本 |
| S3 | 两训练种子各 3,000 步，正确条件综合成功 2/24、1/24，区间含零。人工盲审待完成，正式测试封存。Flow 仅二维演示，Sana 仅预研。见 [报告](../stage3/stage3.md)、[交接](../tasks/stage3.md) 与 [Sana 预研](../research/s3_sana_feasibility.md) | 保留条件、匿名与失败分析方法。正式插画评测按新域校准 |

本地元数据可见 `data/raw/MNIST`、`data/processed/synthetic_icons_week3_v1`、`artifacts/cache/s2_week3`、`artifacts/cache/s2_week4` 与 `artifacts/runs/s3_conditional_v3`。冻结组件对应 [SDXL VAE 身份](../../configs/models/sdxl_vae.json) 中的 `207b116dae70ace3637169f1ddd2434b91b3a8cd` 和 [CLIP 文本塔身份](../../configs/models/clip_text.json) 中的 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。核对了 VAE 两个和 CLIP 七个文件的存在与锁定字节数。P0 未加载模型或重新哈希权重。CLIP 七文件重建有 [历史恢复证据](../stage3/resource_recovery.json)。本轮对本地数据与输出目录的有限盘点未发现 OpenClipart 路径或正式 v2 缓存。

## **四、本地基础检查**

已安装环境为 macOS arm64、Python 3.12.14、uv 0.12.14、PyTorch 2.14.1、torchvision 0.29.1、einops 0.8.2、pytest 9.1.1、Ruff 0.17.2、Diffusers 0.42.0、Transformers 5.20.2、nbformat 5.11.1。使用已有环境的 `uv run --locked --offline --no-sync`，uv 缓存在本轮临时目录。P0 GPU 训练时间 0，云和 API 支出 0 元。

| 执行项 | 实测结果 |
| --- | --- |
| `uv lock --check --offline` | 退出 0，解析 170 个锁定包 |
| `uv run --locked --offline --no-sync minigen doctor --device cpu --json` | 退出 0，CPU 小张量前向、反向通过，形状 `[3,4]`。该次 CPU 检查将 MPS、CUDA 标为 unavailable，未作加速设备验收 |
| `.venv/bin/minigen --help`，及六个现状命令 | 帮助退出 0。四个占位明确退出 2，`data/report` 因未注册退出 2 |
| `uv run --locked --offline --no-sync pytest -p no:cacheprovider` | 107 项，106 通过、1 失败。失败为 `tests/test_bootstrap.py::test_remaining_core_interfaces_never_silently_fall_back`，首个 `patchify` 已实现，旧测试仍断言抛待实现异常 |
| `uv run --locked --offline --no-sync python scripts/check_dit_core.py` | 退出 0，手排 patch、恒等初始化、梯度和三次合成小张量 SGD 更新后的条件响应通过。最大变化约 0.00819248。验证范围为 L03 独立 Block 小案例 |
| `uv run --locked --offline --no-sync ruff check --no-cache .` | 17 个已有诊断。DiT 10、TinyVAE 1、旧 Notebook 6 |
| `uv run --locked --offline --no-sync ruff format --check --no-cache .` | 9 个已有文件会被重排，含 DiT、TinyVAE、两份旧 Notebook、四份 QA/论文笔记 Markdown 及用户重构规划的 Python 代码块 |
| `uv run --locked --offline --no-sync python scripts/check_code_annotations.py` | 57 文件、253 函数、31 类，37 个已有问题均在 `dit_block.py` |

需要区分 L03 小案例通过与全仓基线失败。P3 应针对完整 B0 做独立测试，再处理 bootstrap 中对实现状态的旧假设。详见 [交接](00_handoff.md)。

## **五、P0 交付复核**

15 份 P0 Markdown 的 229 个相对文件链接均可解析到本地目标。12 份既有文档的范围内 `git diff --check` 通过，3 份新增报告通过末尾空白检查。全局 `git diff --check` 仍报告已有 `dit_block.py` 的行尾空白。复验前后全局 Ruff 为相同的 17 项，格式检查仍为同 9 个文件需重排。

198 个基线路径中，186 个在 P0 编辑范围之外，逐项校验原始字节或删除状态一致。原有 27 条 Git 状态均保留，五份 Notebook 的哈希全匹配。v1 折叠正文与原提交内容一致，Git HEAD 和暂存区与开始时一致。
