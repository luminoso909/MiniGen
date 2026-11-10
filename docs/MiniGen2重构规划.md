# MiniGen v2.0｜最终项目重构、开发与研究实验执行规格书

> **用途**：把本文件放入现有 MiniGen 仓库根目录，作为 Codex 的**当前总执行依据**。它取代旧版任务清单中尚未执行的 S4/S5、Sana LoRA 和旧正式实验路线，但**不改写 S1–S3 的教学史实**。\
> **方案版本**：v2.0，正式数据源已更新为 ArtBench。数据产物、组件、训练预算与模型精确层数按对应阶段验收。状态以当前交接为准。\
> **项目一句话**：以经过审计的 ArtBench 绘画为共同数据域，**从随机初始化训练**三种约 30M 参数的文本条件 Latent DiT，统一使用 Flow Matching，生成 256×256 绘画图像；以可复现的架构对照、生成质量与算力成本评测形成研究型项目成果。

---

## 0. 给 Codex 的最优先执行指令（第一次接手必须先读）

### 0.1 不允许误解的十条规则

1. **这是“现有仓库渐进式重构”，不是新建仓库。** 先盘点代码、依赖、Git 状态与本地资产，再决定最小改动路径。不要按本文件中的建议路径假定文件已存在；以实际仓库为准。
2. **S1–S3 五个学习 Notebook 完全只读。** 禁止修改其 Markdown、代码、metadata、cell outputs、execution counts，禁止重跑后保存覆盖；不得为修复 lint 或适配新工程去“顺手”改它们：
   - `docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb`
   - `docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb`
   - `docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb`
   - `docs/learning/S2_第4周_潜空间扩散训练与采样.ipynb`
   - `docs/learning/S3_文本条件CFG与可信评测.ipynb`
   若实际文件名不同，按内容及 S1/S2/S3 标识匹配并列入只读清单。已有历史报告和结果也不可覆盖；只能新建 v2.0 文档解释变化。
3. **不要删掉旧 DDPM/VAE、MNIST、几何图标、S3 CFG、Sana 预研的历史代码和证据。** Sana/OpenMoji/LoRA 路线不再是**本期正式目标**，只标记 legacy/archived，不启动训练、不删除资料。
4. **正式实验只有三个生成主干**：B0 `PlainConditionalDiT`、B1 `SingleStreamDiT`、B2 `HybridDiT`。它们都是**从零初始化训练的模型**，约 30M 可训练参数/模型（以实际统计为准）；不是三种预训练模型微调。
5. **Flow Matching 是三个主干共用的损失和速度预测目标，不是第四种模型。** 禁止沿用旧 S4 “DDPM 噪声预测 vs Flow”作为正式主对比。旧 DDPM 仅保留为学习、数学测试及非正式诊断。
6. **预训练权重冻结**：`madebyollin/sdxl-vae-fp16-fix` 的 VAE Encoder/Decoder；`openai/clip-vit-base-patch32` 的 CLIP 文本塔。文本投影、图像 Patch Embedding、时间调制、注意力/MLP、输出头都由本项目随机初始化训练。
7. **正式数据只采用 ArtBench Public Domain 256x256**，全量 **46,298 张**纳入项目数据版本并分配训练、验证和测试集合。保持 **256×256 RGB**。OpenClipart 路线下线，PixelArt-308K 只作备选参考。
8. **复用 ArtBench 英文 Caption，在数据准备阶段离线清洗。** 逐项核对抽样图文的可见事实，记录失败。需要纠正个例时先保存原文与修订依据。训练和推理仍使用同一冻结 CLIP 文本塔。
9. **先只实现并短训 B0 + Flow**，证明训练、采样、恢复、条件利用及数据/组件正确，再接 B1、B2。先 P0～P4，再 P5 对照；**10M/50M Scaling、Chimera、DDP、手写 FlashAttention/Triton 都是后续可选项**，不能抢在 Baseline 前面。
10. **状态必须真实。** “Notebook 执行通过”“短训能跑”“生成图质量达标”“用户亲手完成核心”“正式实验已完成”是不同的状态。必须以代码、命令、配置、真实日志、图像和验收记录分别证明；不得制造指标、预填结果、宣称已完成或擅自降低阈值。

### 0.2 Codex 接到文件后立刻执行的第一个工作回合

**本回合目标是 P0：完成仓库审计、迁移规划与文档适配；不要直接启动昂贵训练或批量下载。**

1. 在仓库根目录执行 `git status --short`、`git branch --show-current`、`git log -1 --oneline`，列出已有用户改动及未追踪文件；任何现有改动不得丢失，不得执行破坏性 `reset/clean/checkout`。
2. 读取根目录 `README.md`、`AGENTS.md`、旧 `MiniGen项目任务清单与学习路径.md`、`docs/data.md`、`docs/ai-workflow.md`，再读取 `src/minigen/`、`configs/`、`tests/`、`learning/`、`scripts/` 的**真实文件树**。定位 `reports/stage1/`、`stage2/`、`stage3/` 与 `reports/tasks/stage3.md`，用报告核实状态。
3. 为五个 Notebook 记录 SHA-256；所有后续提交/工作回合复核 Hash 不变（文件未找到则说明路径，不要擅自创建占位替身）。
4. 运行现有**非破坏、非下载、非训练**的基础检查。推荐：`uv run --locked minigen doctor --device cpu --json` 和 `uv run --locked pytest`，但依赖未安装、网络不可用或本地有旧 lint 问题时，记录原因，不改历史文件掩盖问题。
5. 生成 `reports/v2/00_repository_audit.md`，内容必须包括：**当前文件/入口/类/函数** → **当前用途和真实状态** → **保留/复用/适配/新增/归档** → **对应新任务**；同时写入 `reports/v2/00_migration_diff.md`（新旧范围差异）。
6. 在不修改 Notebook 的前提下，更新 README、后续任务主清单、`docs/data.md`、相关导航、`AGENTS.md` 和 v2 任务交接（见第 10 章）。**旧 S1–S3 的历史状态必须保留，只把未来目标改成新路线。**
7. 运行小规模测试并给用户提交 P0 交付摘要：改了哪些文件、验证命令与结果、遗留失败、紧接着要做的 **唯一下一任务 P1-01**；不自动 `git push`，不自动购买算力和 API。

> 本文件中的相对文件路径、YAML 示意和 CLI 指令是**接口目标与示例**；旧仓库尚未实现的命令不得假装现在可运行。Codex 在 P0 审计中应给出实际精确路径、受影响模块和最小迁移实现顺序。

---

## 1. 现有基础：已验证、已有但不足、未实现

以下根据上传的 v1.0 规划、README、`data.md` 和五份教学 Notebook 整理。**它们说明上传时的状态，Codex 必须结合真实仓库再次核实**；不能把本文件当作实时源码快照。

### 1.1 已有可复用的工程与学习资产

| 领域 | 已有证据/工程 | v2.0 中怎样处理 |
|---|---|---|
| S1 DDPM | 数学与独立测试、MNIST 教学训练/采样、部分恢复能力 | **保留学习资产**；其噪声预测、调度器不再参与正式主对比。 |
| S1 VAE | 变分推导、简化 VAE 教学、个人数学核心验收 | 保留；正式生成使用**冻结的预训练 SDXL VAE**，不可把教学 TinyVAE 误接为正式 VAE。 |
| S2 几何数据 | 已生成带准确属性的受控几何图标集，旧训练 10,000 / 验证 1,000，测试有隔离 | 继续保留作**单元/回归/小过拟合调试集**；与 ArtBench 正式结果分开。 |
| S2 VAE 与缓存 | 旧项目已做 128px 图标的 VAE 表示测试、潜变量试点及完整训练/验证缓存、身份验证 | **复用缓存框架**，但 ArtBench 256px 数据、不同 Manifest/预处理须**新缓存版本**；旧 128px Latent 不得混用。 |
| S2 DiT | 教学 DiT Block、Patch Embedding、adaLN-Zero、固定监督拟合、短训练 | 可复用**数学原理和通过测试的模块**；把教学单元复制为生产源码前必须核对来源、接口、梯度、数值与权属。 |
| S2 第四周 | 约 0.664M 教学模型随机训练 1,000 步；约 7.37M 主规格当时仅前向验证；8 张生成中 6 张解析失败 | 历史事实保留，**不能声称 7.37M 正式训练成功或 30M 训练已完成**。 |
| S3 条件 | CLIP 文本特征缓存、Cross-Attention、CFG、条件丢弃与独立评测器实现/实验 | 复用可验证部件与测试；新任务需重新验证与 256px、Flow、三架构的适配。 |
| S3 效果 | 两个种子各 3,000 步的教学实验，条件综合成功约 2/24、1/24；人工审核/个人核心仍未全部验收 | **结果不达标**，保留失败记录；必须在新数据及新的主任务上重建质量门槛，不得假装已经解决。 |
| 环境 | uv / Python 3.12、`minigen doctor`、pytest、ruff、目录与报告约定 | 最大化复用；Cloud CUDA 单独诊断，Mac MPS 诊断不能替代云 CUDA。 |
| CLI | 上传时 `train`、`sample`、`evaluate`、`cache` 是报错的占位入口 | P3/P4 **补齐真实可执行 CLI**；README 不得在实现前给出“可运行”的描述。 |
| LoRA / Sana | 旧文档和 Notebook 里只有计划/可行性预研 | 归档，不进入本期工作队列，不下载 Sana 完整权重。 |

### 1.2 旧路线与新路线的映射

| 旧版 v1.0 | 新版 v2.0 决议 | 动作 |
|---|---|---|
| A 线：合成几何图标上的 DDPM / 条件 DiT / Flow 配方比较 | ArtBench 上统一 **Flow Matching**，比较 **Plain/Stream/Hybrid 三种架构** | 保留旧教学，替换后续主实验定义。 |
| B 线：Sana 600M + OpenMoji + LoRA | **取消本期正式目标** | 文档标记 superseded/legacy，存档既有素材，不再安排训练。 |
| 128×128、几何图标 | **256×256、ArtBench 绘画** | 新数据清洗、Caption、表示审计与缓存；旧缓存不共享。 |
| 文本条件 CLIP-B/32 | **继续沿用冻结 CLIP-B/32** | 验证 Token hidden states、mask、77 Token 窗口和新数据描述。 |
| 冻结 SDXL VAE | **继续沿用，但需新域验证** | 200 张随机 ArtBench 图像 的 PSNR/SSIM/LPIPS 与人工检查。 |
| S3 旧受控几何解析器 | ArtBench 的 VQA + CLIPScore + 人工盲审、KID/CMMD 等 | 旧解析器留作几何回归；绘画域评分单独校准。 |
| 分布式、Triton 等 | 单 GPU + BF16 + SDPA 优先 | 仅在实际瓶颈出现后追加优化；其他能力由其他项目展示。 |

### 1.3 本期完成的定义

**最低成功（必须）**：ArtBench 绘画 数据与 Caption 可追溯；冻结组件验收通过；Plain DiT + Flow 在 256px 上真正训练并从噪声生成绘画图像；可恢复、可评测、成本有记录。

**计划成功（主目标）**：B0/B1/B2 三个约 30M 主干在**同一数据、同一条件编码、同一 Flow 目标**下各完成一轮正式训练；输出固定对照图和多维度评测；形成质量—计算量/成本比较、复现配置和报告。

**增强成功（可选）**：额外随机种子、10M/50M Scaling、Chimera-Lite。不能用可选项拖住主目标交付。

**不属于本期**：LoRA、SFT、GRPO、生成模型 RL、Sana 适配、像素空间生成系统、RAE 全量替换、训练自己的 CLIP/VAE、必需的多卡集群及复杂的模型服务平台。

---

## 2. 技术总体架构和训练边界

### 2.1 图像、文本与时间三路输入

```text
离线数据准备（仅一次或按数据版本更新）
ArtBench 本地 Parquet --[全量解码校验/保留尺寸]--> RGB 256×256
          |                                             |
          +--[英文 caption 与来源元数据]--+              |
                                       v               v
                              离线清洗与审计     🔒 SDXL VAE Encoder
                                       |              |
                                英文 Caption         Latent: [B,4,32,32]
                              |                       |
                  🔒 CLIP Text Encoder                |  乘模型 scaling_factor
                              |                       |
                    Token hidden states               +--> 缓存干净 z [B,4,32,32]
                       [B,L,512]                                |
                              |                            ε ~ N(0,I), t~U[0,1]
                              |                                 |
                              |                z_t = (1-t)ε + tz;  target = z-ε
                              |                                 |
                              +--------> 🔥 Text Projection     |
                                              |                  |
                                  +-----------+------------------+----- t
                                  |
                   🔥 B0 Plain / B1 Single / B2 Hybrid DiT
                                  |
                         v_theta(z_t,t,c)  [B,4,32,32]
                                  |
                   Flow Matching velocity MSE → 更新 DiT 及投影

推理：Prompt → 🔒 CLIP → ε → 已训练的某一个 DiT → Euler/Heun
      → 预测干净 z → 除 scaling_factor → 🔒 VAE Decoder → 256px PNG
```

“🔒 冻结”意味着权重从公开预训练模型加载、不更新梯度；“🔥 从零训练”只指生成主干和其训练适配层。VAE/Text Encoder 不计入 `trainable_params`，但需在**端到端运行内存和时间**中记录其开销。

### 2.2 固定预训练组件（先不做消融）

| 模块 | 建议模型 ID | 角色 | 训练状态 | 必测内容 |
|---|---|---|---|---|
| VAE | `madebyollin/sdxl-vae-fp16-fix` | 图片编码为 Latent、最后解码 | 冻结 | 权重/Revision；像素范围；后验均值；`scaling_factor`；256px 形状；200 张重建。 |
| Text Encoder | `openai/clip-vit-base-patch32` 的 **Text Tower** | Caption/Prompt → Token 级上下文特征 | 冻结 | tokenizer、最大 77 位置、hidden_size=512、padding mask、空文本、特征哈希。 |
| 源英文 Caption | 数据卡记录为 Florence-2-large 生成 | 离线清洗的原始文本输入 | 固定源版本 | 200 张以上配对抽查、文本清洗追溯、CLIP 长度校验。 |

**重要实现细节**：

- 不要使用 `CLIPModel.get_text_features()` 的**单个 pooled 向量**替代 Token 特征；要取文本塔的 `last_hidden_state`，通常为 `[B,L,512]`；条件投影 `Linear(512, hidden_size)` **随机初始化且可训练**。
- 正式 Caption 建议一段英文简洁、客观、可视觉验证的短描述；目标控制在 CLIP 的 **75 个内容 token（+ BOS/EOS 合计 77）**内。超长标注须退回重写或显式拒绝，不静默截断。**旧 S3 的双段 77-token 教学实现保留不变**，但其跨段语境不属于主项目基线。
- 训练/采样统一使用相同的 VAE 编码策略（初版可沿用 `posterior.mode()`/均值作为固定编码，需在报告写明）；把 VAE `scaling_factor` 乘到 Encoder 输出、在 Decoder 前除掉，不能重复缩放。
- VAE 输入前按其配置统一到 `[-1,1]`、RGB；推理后将 Decoder 的图像范围准确还原到 `[0,1]` 再保存。
- 256px 下若 SDXL VAE 空间压缩 8 倍，则 Latent `[B,4,32,32]`；Patch Size=2 则图像 Token 数 `(32/2)^2=256`，每个 patch 展平 16 个数。上述是**形状验收预期**，实际以模型输出验证。
- Manifest/Cache Identity 必须能追溯 ArtBench revision、文本清洗、CLIP、VAE 和预处理配置。预训练模型冻结并不等于它们不占推理显存；使用离线缓存时要区分“核心 DiT 训练峰值”和“端到端占用”。

### 2.3 三个**需要从零训练**的生成主干

| ID | 主干与参考 | Text → Image 的条件路径 | 重要区别 | 本期约束 |
|---|---|---|---|---|
| **B0** | `PlainConditionalDiT`；传统 DiT | 图像 Self-Attention + Token 级文本 Cross-Attention | 清晰、稳定、适合作为基线；保留时间调制 | **先实现、先打通**。 |
| **B1** | `SingleStreamDiT`；参考 Qwen-Image-2.1 的单流思想 | 文本/图像各自投影后拼接，同一组 Transformer Block 联合注意力 | 图文 Token 共用主干；需要模态位置与 mask 管理 | **采用简化双向联合注意力**；原版特定 Block-Causal 等复杂设计先不强制复刻。 |
| **B2** | `HybridDiT`；参考 HunyuanImage-2.1 | 前段双流 Block：图文独立投影/FFN + Joint Attention；后段合并为 Single Stream | 文本和图像前期各有参数，再在联合流中处理 | 双流与单流块数由实测参数量决定；先实现最小必要机制。 |

**不是复现声明**：这些是受现代公开架构启发的**MiniGen 简化模型**，不是 Qwen-Image-2.1/HunyuanImage-2.1 的原版权重、参数、训练策略或指标；不得宣传为“复现 7B/17B 原模型”。

**关于 Joint Attention、位置编码**：B1/B2 实现时对图像 Token 支持可测试的二维位置表示（优先 2D RoPE，亦可在 B0 保留传统 2D 位置编码）；文本位置由 CLIP 编码本身提供，联合流的文本和图像需要有一致、明确的模态位置方案。切换位置编码会影响公平性，必须在实验配置中记录，不把多项架构差异归因于某一个 Attention 模块。若 RoPE 接入导致工程风险，优先确保数学与生成闭环再扩展；不必声称现代模型所有结构都被完整移植。

### 2.4 模型参数预算与可训练边界

**正式对比目标：每个生成器约 30M 个可训练参数**，不包含 VAE、CLIP 和 Caption VLM；建议三者可训练参数量差异控制在目标值的约 **±10%**，并报告真实 `trainable_params` 与实际 FLOPs。若同规模与同 FLOPs 不可兼得，**分别报告“同参数”“同计算预算”结果**，不能假装都完美匹配。

初始候选（**不是不可修改的精确网络定义**）：hidden_size≈384、heads=6、head_dim=64、MLP ratio≈4、latent patch=2。B0/B1/B2 的实际 Block 成本不同，**Codex 必须先做参数计数脚本**，再选择各自层数及双流/单流比例；主干所有投影、时间 AdaLN、输出预测头均计入 30M。正式训练前冻结带参数统计的模型卡。

10M、50M 只保留为**之后**可能进行的同架构 Scaling，不要提前建立 9 个训练任务或为规模实验修改三模型主线。Chimera 在条件成熟后作为第四架构参考卡，不进入第一轮实现清单。

### 2.5 Flow Matching 的唯一数学习惯

设预处理后干净 Latent 为 `z`，标准高斯噪声 `epsilon ~ N(0,I)`，`t∈[0,1]` 从噪声流向数据：

```text
z_t = (1 - t) * epsilon + t * z
velocity_target = z - epsilon
pred = dit(z_t, t, text_hidden, text_mask)
loss = mean((pred - velocity_target)^2)
```

- 模型输出 shape 与 `z_t` **相同**，`[B,4,32,32]`，不是 DDPM 的别种预测头，更不是速度标量。
- 采样从 `t=0` 的噪声开始，向 `t=1` 正向积分：`z_next = z + Δt * pred`（Euler）。Heun 是可选二阶数值积分，先做 Euler 正确性。
- `CFG`: `v_guided = v_uncond + s * (v_cond - v_uncond)`；`s=1` 为纯条件分支，通常可以跳过无条件计算。训练使用概率 `p=0.1` 的空文本条件替换，空条件须来自**相同的 CLIP 文本塔**。
- `t` 在接近端点的采样、浮点精度、广播及数值稳定性必须测试；训练可先 `Uniform(0,1)`，不额外加入 logit-normal 或 OT 路径。
- DDPM `epsilon` 预测 checkpoint 不得直接使用 Flow Euler/Heun 解码；必须强制检查 checkpoint 的 `objective=flow_velocity`。
- 采样 Euler/Heun 比较必须按**网络前向评估次数 NFE**匹配：Heun 的每个积分步通常要两次主干求值，CFG 还会额外影响实际主干次数；分别报告积分步数、预测评估次数、实际网络调用数和延迟。

### 2.6 推荐统一的最低代码契约

仅作为结构目标，需结合原源码做最小适配，不要求逐字复制：

```python
@dataclass
class TextCondition:
    hidden_states: Tensor  # [B, L, 512] 或已投影后的 [B,L,D]，须在类型中明确
    attention_mask: Tensor  # [B, L], bool, True 表示有效文本位置


# 统一三架构调用：
# 不在 model.forward 中加载 VAE、CLIP 或执行 tokenizer。
def forward(
    self,
    noisy_latents: Tensor,  # [B, 4, 32, 32]
    timestep: Tensor,  # [B]，范围 [0, 1]
    text_hidden: Tensor,  # [B, L, 512]，接口处做内部可训练投影
    text_mask: Tensor,  # [B, L], bool
) -> Tensor:  # [B, 4, 32, 32] 的 velocity
    ...
```

建议模块边界：`models/` 只处理已给定的张量和时间；`conditioning/` 负责编码与缓存；`autoencoders/` 负责冻结 VAE；`objectives/flow.py` 负责生成 `(z_t,target)`；`samplers/flow.py` 负责 ODE 积分；`training/` 负责批数据、优化器与恢复；`evaluation/` 负责独立质量测量。复用已有 `src/minigen/` 子目录，不新增第二套并行核心实现。

---

## 3. ArtBench 正式数据接入（P1）

### 3.1 数据范围与来源

- 正式数据为 **ArtBench Public Domain 256x256**，来源 `alfredplpl/artbench-pd-256x256`。全量 **46,298 张**纳入项目数据版本，训练、验证和测试合计覆盖全部样本。输入目标为原生 **256×256 RGB**，保留画面构图和绘画题材。OpenClipart 下线，PixelArt-308K 保留为备选参考。[发布者数据卡](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256)
- 上游仅发布一个 `train` 集合。项目另按可复现的分组规则切分。以来源 URL、原图与解码像素身份识别同一作品，在分组后划分，另记近重复排查方法和覆盖范围。作者字段用于溯源和样本分析。训练与验证分开，测试集在选型时封存。
- 数据卡描述为原 ArtBench 的公版图像子集，发布页标注 MIT。项目保留每行 `url` 和 `artist`，将数据卡声明、作品来源和可能的独立再分发核验分开。[发布者数据卡](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256) · [原 ArtBench](https://github.com/liaopeiyuan/artbench)

### 3.2 数据审计与 Caption 清洗

1. **全量结构校验。** 固定 revision、Parquet 分片、文件长度和哈希；离线读取 `image`、`caption`、`caption_jp`、`artist`、`url`。解码并校验尺寸、模式、行数和失败分母，发现异常直接记录并排查。
2. **至少 200 张配对抽查。** 固定样本索引与抽样种子。检查损坏、严重技术模糊、拉伸、畸变和异常裁切，区分绘画笔触与技术缺陷。对照原图检查 Caption 的主体、场景、颜色、数量和空间关系。年份或作者讹误单独归类；语义结果保留具体错误实例和各项适用分母。
3. **保守清洗。** 数据卡记录英文 `caption` 来自 Florence-2-large，日文 `caption_jp` 来自 Qwen2-VL-2B-Instruct。训练只使用清洗后的英文 Caption。保留可见主体、场景、颜色、数量、关系和基础题材风格词；移除可安全分离的署名、年份、标题归属、背景说明和重复文本。视觉事实错误须靠图文对照发现，正则清洗只处理文字表达。[Caption 生成说明](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256)
4. **长度与身份。** 按冻结的 CLIP tokenizer 校验最多 75 个内容 token，超限保留完整可用句，仍无法安全缩短的样本单列。保留原文、清洗后文本和清洗版本，汇总空文本、超长和需复核项。分组规则、划分种子、清单与来源哈希一起记录。正式 VAE/CLIP 缓存验收在 P2。

### 3.3 产物与冻结口径

项目使用固定版 Parquet 和本地可复查的图片与 JSONL 清单。清单记录样本 ID、行来源与 revision、来源 URL、作者、图像身份、256×256 尺寸、组与划分、原英文和清洗后 Caption、文本状态及 schema 版本。以实际实现和固定配置为准。测试样本的图片和描述均按正式训练隔离协议处理。

任何示例条目必须标明 `example_only=true`，正式 Loader 拒绝示例和越出数据根目录的路径。预处理、清洗规则或划分变更要产生新的清单和缓存版本。真实素材、权重和私密审核记录保存在本地忽略目录。[固定来源配置](../configs/datasets/artbench_pd_256.yaml) · [数据规范](data.md) · [P1 审计](../reports/v2/data/artbench_audit.md)

---

## 4. 冻结组件的训练前验收（P2）

### 4.1 必做：200 张 ArtBench 图像的 VAE 重建测试【硬门槛】

**请原样保留此任务，不因以后修改项目而删掉：**

> **随机抽取 200 张 ArtBench 绘画，对比原图与 VAE 重建图。记录 PSNR、SSIM、LPIPS，并人工检查边缘、颜色与小物体保留情况。如果重建质量足够好，就直接使用；如果明显损失轮廓细节，再考虑其他 Autoencoder。**

执行规范：

1. 从**冻结清单的 ArtBench 训练切分** 中以固定 seed 随机抽取 200 张，记录 sample_id 和随机生成方式；再按高风险情况附加若干边缘案例，但这批附加样本不计入原定 200 的随机统计。
2. 保存原图/重建图/局部细节的对照网格。输入严格按 SDXL VAE 规定的图像范围与颜色通道，明确 Encoder 用后验均值还是采样；缩放要可逆：`z_scaled = z_raw * factor; z_raw = z_scaled / factor`。
3. 计算每图与汇总的 PSNR ↑、SSIM ↑、LPIPS ↓，固定图像动态范围、LPIPS 网络和版本；提供均值/中位数、分位数及最差样本索引。不得把 PSNR/SSIM/LPIPS 当作完整的绘画语义指标。
4. 人工检查**轮廓锐度、细线、小物体、颜色偏移、背景伪影**，记录失败等级（轻微/明显/无法使用）与代表性案例。**不在看结果之前伪造通过阈值**；先以可见主体、重要颜色、局部细节是否保留为 go/no-go 条件，再据真实分布拟定阈值并在报告说明。
5. 若存在系统性丢线/糊边：优先排查解码、归一化、缩放或 FP16 问题，再比较一个明确可用的替代 VAE/Autoencoder；替换将导致**所有模型共享 VAE 并重新计算全量缓存**，禁止只给其中一个主干换 VAE。
6. 输出 `reports/v2/components/vae_reconstruction_200.md`、`metrics.json`、200 样本清单与图表；用户审核通过且本机数值验收通过后，才允许构建正式 Latent 缓存。正式长训仍有独立的验收和授权门槛。

**注意**：旧版 S2 的“96 张几何图标重建通过”不能代替新数据域的 200 张验收。

**当前判定**：用户已批准保留冻结的 `madebyollin/sdxl-vae-fp16-fix`，用于 256×256 的一般艺术绘画生成；细密线条、文字和小物体的高频信息损失记录为已知局限，不因重建损失剔除 ArtBench 样本。模型不更换、不微调。保留 [200 张统计、9 张诊断和 LPIPS 最差 20 张证据](../reports/v2/components/vae_review_evidence.json) 与 [用户批准](../reports/v2/components/vae_user_approval.json)。

### 4.2 CLIP 文本编码验收

- 加载 `CLIPTextModel`/等价文本塔，检查输出 `[B,L,512]`、Token mask、空 Caption 编码，严格保存 revision/权重哈希；冻结参数并通过 `requires_grad=False` 检查。
- 新正式 Caption **不允许 silent truncation**；统计文本 token 长度，超限回到 Caption 清洗阶段。Padding Token 对 Cross/Joint Attention 不可见；检测 mask 方向、极端短/空文本、无有效 Token 的边界。
- 固定相同图片、两个不同 Caption，证明生成器收到的条件特征不同，并检查 Image Token 对条件是否存在非零可学习梯度；“特征不同”不是模型已会遵循条件。
- 允许 CLIP 特征**一次性预缓存**，缓存包含 tokenizer/模型/拼接方式、caption 版本、mask、dtype，FP16/FP32 差异记录在报告，不隐瞒缓存量化误差。

### 4.3 冻结缓存策略

- 首轮采用和旧项目一致的**确定性 VAE 后验均值**缓存以减小实验方差；所有 B0/B1/B2 使用同一份 `z` 语义定义。将“固定后验均值可能不等于预训练 VAE 训练分布”写入限制，并在未来有证据需要时改策略（改策略必须统一重训比较模型）。
- 新 Latent 形状应为 `4×32×32`；旧几何图标缓存 `4×16×16` 不可复用。
- VAE Encoder 在缓存预计算阶段用；VAE Decoder 在验证图、推理、重建审计阶段用。训练主循环可只读 Latent Cache，不反复加载 Encoder。
- CLIP 文本特征可缓存训练集、验证集的已冻结 Caption；模型文本投影属于**可训练 DiT**，不能被缓存为冻结特征。Blank/unconditional 条件单独缓存并保存 mask。
- 正式编码前从原随机重建批固定 20 张，核验 RGB 归一化、后验 mode/均值、模型缩放和输出稳定性；以 CPU FP32 为参考，分别验证 VAE 和 CLIP 的 MPS 数值一致性。对无法在 MPS 正常运行或超出误差门槛的组件，显式选用已验证的 CPU，逐组件记录原因。
- P2 冻结组件只做前向编码，没有 CUDA 依赖。正式缓存按批准的后端在本机分批写入并支持断点恢复；发布后验证 train/validation 的完整性、顺序、样本对应、形状和 dtype，测试切分封存。CUDA 训练运行在后续阶段另验。

---

## 5. 代码迁移方案：保留旧骨架，统一模型与训练链（P0–P3）

### 5.1 必须先形成“真实文件—新职责”映射

本文件不能看到私有仓库实时 `src/` 文件，因此下表是**目标职责**，不是声称真实目录当前已经实现；Codex 在 P0 输出中用仓库实际路径逐项替换。

| 现有/目标目录 | 保留/调整 | 目标职责与测试 |
|---|---|---|
| `src/minigen/models/` | 复用旧 DiT、添加三个主干工厂/注册 | `plain_dit.py`、`single_stream_dit.py`、`hybrid_dit.py`（建议名），共同协议、参数统计与模型配置。 |
| `src/minigen/conditioning/` | 复用 S3 CLIP 和 mask | 冻结文本塔、空文本、缓存、稳定的 `TextCondition` 契约。 |
| `src/minigen/autoencoders/` | 复用 S2 SDXL VAE 包装 | 统一 encode/decode、缩放与混合精度策略，不在 DiT forward 中调用。 |
| `src/minigen/objectives/` | 保留旧 DDPM 目标，新加 Flow | 独立 `sample_t_eps`、`make_flow_pair`、`velocity_mse`；无隐藏下载。 |
| `src/minigen/samplers/` | 保留 DDPM/DDIM；增加 Flow Euler/Heun | CFG 使用 velocity 预测，时间方向与 NFE 清晰。 |
| `src/minigen/data/` | 沿用 Manifest/Cache/校验基类 | 接入 ArtBench 本地 Parquet、图片解码与审计、Caption 清洗、256px Manifest 和缓存身份。 |
| `src/minigen/training/` | 优先复用 trainer/EMA/恢复 | 支持模型配置、随机种子、BF16、日志、checkpoint 完整身份、真实 CUDA 训练。 |
| `src/minigen/evaluation/` | 旧几何评分器保留 | 新增 ArtBench 绘画域质量、语义、人工审核、性能指标；评测数据独立。 |
| `src/minigen/cli.py` | 从占位逐步实现 | `doctor`, `data`, `cache`, `train`, `sample`, `evaluate`, `report`（按需）。 |
| `configs/` | 新建 v2 数据/模型/训练/评测/实验配置 | 绑定模型版本/源代码版本和参数数目；不覆盖 v1 正式登记。 |
| `tests/` | 旧数学测试保持稳定，新增 v2 测试 | Flow 端点/梯度、mask、CFG、RoPE、3 架构 shape/参数、缓存拒绝、resume/CPU 冒烟。 |
| `reports/`、`artifacts/` | 创建 v2 名空间 | 旧报告不可覆盖，新报告有阶段证据，新生成权重默认不进 Git。 |

### 5.2 新模块最小正确性测试清单

**Flow**：`t=0` 必为 `eps`，`t=1` 必为 `z`，`target=z-eps`；常速度二维轨迹 Euler 能到达目标；Heun 在非线性已知速度场下提升精度；负向时间和 DDPM checkpoint 混用应失败。

**Condition**：相同 `z_t,t` 配两个不同 Caption 产生可检测的预测差异（初始化时差异可以很小，但应该有梯度通路）；Padding Mask 改变无效位置不能改变输出（在容差内）；CFG `s=0,1` 满足代数定义；训练条件丢弃概率及空编码路径可测试。

**DiT**：三个模型都接受 `[B,4,32,32]` + `[B]` + `[B,L,512]` + `[B,L]`，返回同形浮点速度；支持 batch=1 和 >1；不同尺寸必须明确报错或按配置处理；参数计数自动报告。2D RoPE 验证行列坐标不同、频率广播正确、注意力计算不出 NaN；Hybrid 的双流融合输出维度正确、文本 mask 贯通。

**Optimization**：有非零、有限的可训练梯度，冻结 VAE/CLIP 没有梯度；尝试在 16–32 张真实绘画 Latent 上过拟合，观察损失下降、条件变化、采样可辨；如果短跑失败，先定位张量/损失/scale/mask，再增加计算预算。

**Checkpoint**：保存模型、优化器、调度器、EMA、GradScaler（适用时）、训练 step、样本/采样 RNG 状态、配置、源码 commit、模型身份、manifest hash、文本/图像缓存 hash；不同 objective/模型类/数据/缩放禁止静默 resume。比较短连续训练和中断恢复后的 loss/权重在设备允许的容差内一致。

**CLI**：开始时所有新命令只需 `--help`、`--dry-run`、小数据 CPU 测试；未实现的入口明确非零退出，不伪装成功。正式 CUDA 环境的全量训练必须**用户另行授权**。

### 5.3 参数与计算差异的适配顺序

1. 先实现 `PlainConditionalDiT` 和统一 Flow + Euler + 组件/缓存适配。
2. 在 CPU/MPS 或小显存环境进行 tensor、梯度、过拟合和短采样。
3. 参数统计并调到约 30M，检查 loss finite、样本可辨、条件确实生效，再做正式 B0 长训。
4. 在同一训练框架下添加 Single-Stream，保持模型的 `forward` 输入输出不变；只替换 backbone。
5. 在前两者稳定后实现 Hybrid 双流/单流，输出相同 velocity；不要复制三套 Trainer、Scheduler、Cache 或 Eval。
6. 每加一个架构先跑全套通用测试+短跑，再允许进入正式实验表格。

### 5.4 谁负责核心代码？

Codex 可以承担**环境、数据、训练器、缓存、测试、脚本、评测、调试和架构工程搭建**；对 DiT/Joint Attention/Flow 核心实现，可以先提供一个明确标记来源的正确实现和数学/数值测试，供用户阅读、复盘、修改或亲自替换。**不要把 Codex 生成或从 Notebook 迁移的代码写成“用户独立手写”。** 用户学习任务 `L03/L04/Flow` 若尚未亲手验收，保持待完成状态，不阻塞工程的独立正确性验证，但简历与报告中明确实现和验证事实。

---

## 6. 单卡训练配置、成本试验与资源约束（P3–P4）

### 6.1 首轮共同默认值（可由 P3 短跑校准）

| 项目 | 基线提案 | 说明 |
|---|---|---|
| 图片尺寸 | **256×256 RGB** | 正式分辨率。128px 只允许短跑/诊断；不能冒充正式结果。 |
| Latent | **4×32×32** | 按冻结 SDXL VAE 实测。 |
| Latent Patch | **2×2** | 图像 token=256。 |
| 主干大小 | **目标 30M trainable** | 深度/双流层数按统计选择，允许约 ±10%。 |
| Text Encoder | CLIP-B/32 token hidden | 冻结，输出 512 维。 |
| Text Dropout | `p=0.1` | 无条件分支用于 CFG。 |
| Flow Path | 线性噪声→数据，velocity MSE | 所有三模型一致。 |
| Optimizer | AdamW，试点 LR=1e-4 | 可比较 2e-4，但必须在**验证集**调，主对比统一。 |
| Effective Batch | 目标 64 | micro batch 由实测显存决定，gradient accumulation 补足；如资源不足则共同调整。 |
| Precision | **BF16 autocast**（设备支持时） | loss/归一化敏感路径根据数值表现提升精度。 |
| Attention | `torch.nn.functional.scaled_dot_product_attention` | 按实际后端可用性自动选择；显式记录内核。 |
| EMA | 启用；系数先经短跑选定并三模型共用 | 不将未来收敛速率当成既定事实。 |
| Scheduler | 简单 warmup + 衰减（按短跑冻结） | 时间单位与真实 optimizer steps 对齐。 |
| Checkpoint | 定期保存，完整恢复 | 保存 latest + 指定 best/step，注明选取规则。 |
| 正式训练步数 | **暂不锁死** | 依据 1k/数千步短跑、模型质量变化、预算审批冻结。 |
| 硬件 | **单卡 NVIDIA CUDA 优先** | 本地 Mac 用于开发、清洗/标注、测试；DDP 非必要。 |

### 6.2 训练前先做资源 Pilot，而不是凭参数量猜成本

在**同一 GPU 型号与环境**对 B0 做 `dry-run` + 50–200 次 warmup/计时步：记录每 step 秒数、samples/sec、GPU peak allocated/reserved、CPU RAM、optimizer+EMA 内存、DataLoader 等待占比、单样本文本预处理与缓存成本。正式训练按**实测 sec/step × 目标步数 × 实际时价**预估，并另加 checkpoint、验证与生成评测开销。不要把单位 step 价格当作已发生费用。

建议先做 `16–32 样本过拟合 → 500–1000 步可辨生成短跑 → 第一轮较长 B0` 的关卡。未通过时停止而非直接扩大到 50M/购买更多 GPU。历史规划的“100–300 元云资源预算”是当时的预算估计，不保证 256px/30M 三模型全部足够；P3 应产生**新的、真实、可审批的预算表**，任何收费操作等用户明确授权。

### 6.3 是否用 FlashAttention、DDP、ZeRO？

- **P0 默认**：BF16、SDPA、缓存、合理 DataLoader、按需梯度累积。
- **若显存不足**：依次考虑 microbatch↓+accumulation、gradient checkpointing、冻结组件卸载、简化验证并发；不能为了保住 batch 值悄悄改变计算目标。
- **若注意力占显存/耗时明显**：检查 SDPA 是否已命中 Flash/Memory-Efficient 后端，再决定独立引入 FlashAttention-2；只在收益通过测量后保留。不要先手写 Triton。
- **DDP/ZeRO**：单卡无法在合理预算内运行、且已证明多卡会带来有效收益时才作为增量方案，**不属于本期成功条件**。
- 256px 比旧版 128px 图像 token 从 64→256（patch=2），注意力显存/算力变化取决于文本 token、注意力类型、GPU 内核与 batch；不要简单断言实际训练成本一定×16。**模型参数数目不会因输入分辨率自动变化**，但可在首轮结果后决定是否扩容。

---

## 7. 正式实验矩阵【必须保留到最终报告】

### 7.1 实验研究问题

- **RQ1 架构**：在相同数据、表示、Flow、训练协议下，Plain/Single/Hybrid 谁的图像质量、文本遵循和成本更好？
- **RQ2 采样**：同一 checkpoint，Euler/Heun 在匹配 NFE 时质量与延迟如何变化？CFG 和 NFE 的边际收益是多少？
- **RQ3 计算效率**：固定训练计算预算时，架构排序和参数规模是否变化？**借鉴 Abra 的实验方法，不声称复现大型模型 Scaling Law。**
- **RQ4 可用性**：对颜色、对象数、属性绑定及空间关系，三种架构在哪些任务失败，自动评测有多可靠？

### 7.2 A 组：三架构主实验（P0 必做；先三次）

| 实验 ID | 架构 | Trainable 参数目标 | 训练目标 | 图片 | 首轮 seed | 首轮要求 |
|---|---|---:|---|---|---:|---|
| **B0** | PlainConditionalDiT | ~30M | Flow Velocity | 256px | 0 | **优先训练完成**。 |
| **B1** | SingleStreamDiT | ~30M | Flow Velocity | 256px | 0 | 使用 B0 同一数据/预算。 |
| **B2** | HybridDiT | ~30M | Flow Velocity | 256px | 0 | 使用 B0 同一数据/预算。 |

共享：VAE revision、CLIP revision、数据 manifest、Caption 规则、训练/验证/测试划分、优化器/学习率策略、目标有效 batch、训练步或计算预算、增强原则、固定评测 Prompt/Noise seeds、采样方案。架构差异及实测参数、每步 FLOPs、每秒吞吐单独报告。

**主比较必须给出两种视角**：

1. **同更新步数 / 同见过的样本数**：看训练收敛与优化行为；不同架构算力可能不同。
2. **同计算预算（或尽可能相近的可实测 GPU 时间/FLOPs）**：看单位算力下的质量与速度效率。不可把“同训练步数”自动叫作“公平计算预算”。

若无法在有限云资源下满足两种对照，优先按同训练配置跑三主干，同时**诚实报告资源不等同**；后续用中间 checkpoint 建相同预算曲线，不强制为了绝对严格匹配而重训一遍。所有差异与置信度在报告交代。

### 7.3 B 组：额外随机种子（P1，可选）

B0/B1/B2 各补 seed=1 → 共 **6 次正式训练**。每个模型不仅汇报最高分，至少给各自独立 run 的结果；2 seeds 只能给初步波动参考，不足以做强统计显著性结论。**不允许现在就把 6 次全部标为必须花钱完成的任务。**

### 7.4 C 组：Abra 式模型规模实验（P1，可选）

只选择前三者**最稳定或最有代表性**的一种架构（预先写明选择准则），再在 **10M / 30M / 50M** 下比较；30M 复用 B0/B1/B2 的对应结果，额外通常只需 10M、50M 各一个 seed（若资源允许再重复）。

- 主分析：在**近似相同训练 FLOPs / GPU 时间预算**下比较质量，研究“小模型多训 vs 大模型少训”。
- 辅助分析：相同更新步数下比较质量，但注明总计算量不同。
- 参数、架构深度和超参数随规模变化需记录，若从 30M 配置变到 50M 时改变了别的重要因素，不得把全部收益归因于参数量。
- **不拟合夸大的普适 Scaling Law**；它只是 MiniGen 小规模受控的计算资源分配实验。

### 7.5 D 组：同 Checkpoint 的采样/推理对照（P0 必做，通常无需重训）

| 对照 | 建议档位 | 评价 | 附注 |
|---|---|---|---|
| Sampler | Euler vs Heun | 质量/文本遵循/延迟 | 尽量以**总 NFE**匹配而非同积分步数。 |
| NFE | 10 / 20 / 40 | 质量—调用次数—耗时曲线 | 记录 CFG 条件/无条件实际调用数。 |
| CFG | 1 / 2 / 3 | 文本遵循—质量—多样性 | CFG=1 是条件分支；需在验证集选定最终值。 |
| Attention 工程 | naive attention vs SDPA（可实现时） | speed、VRAM、数值差异 | 不同精度/后端说明数值容差；主要是**性能对照**。 |

不要一次做 2×3×3×3 的无意义全笛卡尔积。先固定基准 CFG/NFE 逐变量扫描，再针对可能更优的 1–2 个组合验证；实际执行次数在实验预登记中列明。

### 7.6 E 组：未来扩展（P2，不纳入一期交付）

- Chimera-Lite（混合注意力、NoPE 等只择一到两个可测因素，不把完整 Chimera/MoE 当成本期必须复刻的项目）。
- 2D RoPE 位置编码消融、Hybrid 双流层数比例、时间采样分布等，仅在出现明确问题且预算允许时考虑。
- 更大分辨率、额外文本编码器、VAE/RAE 替换均会改变共同控制变量，必须单列新实验版本。

### 7.7 每次实验都必须可复现的记录字段

`experiment_id / git_commit / date / architecture / model_config / trainable_params / optimizer / lr_schedule / batch_micro / grad_accum / effective_batch / objective / vae_revision / vae_scale / clip_revision / caption_manifest_hash / train_manifest_hash / seed / hardware / precision / attention_backend / step / tokens_or_samples_seen / training_wallclock / FLOPs_estimate_method / GPU_hours / cost_actual / checkpoint_hash / sampler / NFE / CFG / metric_versions / eval_prompt_version / failure_notes`。

只允许真实字段，未知写 `null + 原因`，不要补假值。对比训练过程中最好报告**质量-步数、质量-GPU 小时、质量-NFE**曲线。

---

## 8. MiniGen-Bench：独立测试与多维度评测（P1–P5）

### 8.1 借鉴 DiffusionBench 的是什么

借鉴的是**统一设置、多指标、多维质量—效率比较、公开配置、失败分析**的研究思想，不是直接用其大模型绝对分数评定 MiniGen，也不假装完整复现公开 Benchmark。若引用 GenEval 2 或 T2I-CompBench 的题型，需要标记“借鉴题型/抽样子集”，**不得声称跑过完整官方基准**。

### 8.2 两套评测数据不可混淆

**集合 A：独立真实图片集合（用于 KID/CMMD 等分布指标）**

- 从 ArtBench 的冻结 Test split 抽取**覆盖绘画风格、与训练组隔离**的真实图片，目标 1,000–2,000 张（若数据总量不足，按实际数量报告）。
- 用统一 prompt 分布/条件采样规则生成匹配数量的图片；KID/CMMD 需写明特征提取器版本、样本数量、子集次数/随机种子与抽样方差。只有 160×2=320 张时，指标可能不稳，应明确标记探索性而非直接当作稳健 KID/CMMD 结论。
- ArtBench 的 Inception 特征距离受特征域影响，搭配人评与语义指标解读。

**集合 B：MiniGen-Bench v1.0 Prompt Benchmark（用于条件遵循和组合泛化）**

| 子集 | 计划 Prompt 数 | 测试目的 | 数据约束 |
|---|---:|---|---|
| ID | **80** | 常见动物/物体/植物/简单单场景，验证学习到的域内视觉概念 | 不复用训练描述原句，已见概念但不是训练图片。 |
| Composition-OOD | **40** | 已见对象与颜色/数量/空间关系的保留组合 | 需要**在训练前定义并封存“未见组合”**；若训练数据根本缺相关结构，须报告能力边界，不能伪造 OOD 设计。 |
| Paraphrase | **40** | 同概念不同英文表达，验证语义改写稳健性 | 保持对象/属性不变，仅改变句式；与原始 ID 对照成对。 |
| **合计** | **160** | 各 prompt 固定 2 个噪声 seeds → 每个模型 320 张主评测图 | **先冻结 v1.0 Prompt 清单，再正式训练**。 |

此外单设 **Validation Prompt**（建议 80–120 条）用于选择 checkpoint、CFG/NFE/采样器；主测试集不能用于选模型。注意训练图片的 Test Split 与 Test Prompts 不完全是一回事，需要关联图像来源、目标属性和 prompt family，并防止近重复或文本泄漏。

### 8.3 评价指标、解释范围与验收条件

| 类别 | 指标 | 使用/限制 |
|---|---|---|
| 真实分布质量 | KID↓、CMMD↓ | 用独立参考集及足够样本；注明域和特征提取器；数量不足降为探索性。 |
| 图文语义 | CLIPScore↑ | 辅助图文匹配，**会偏向与冻结 CLIP 相关的表示**；不能单独说明文字都符合。 |
| 对象与属性 | VLM/VQA Accuracy、对象数量、颜色、属性绑定、关系准确率 | 对每个 Prompt 的可验证子问题评分；VLM 在绘画域须先做人工校准。 |
| 组合泛化 | OOD 组合成功率 | 训练中封存明确组合；分母为包含相关组合要求的 prompts；不可偷看测试调整训练。 |
| 多样性 | LPIPS↑/↓ 随定义解释 | 多 seeds 比较前先排除无效图片；不能奖励“完全不遵守 Prompt 的变化”。 |
| 人工 | 盲评质量/文本遵循/明显缺陷 | 匿名、打乱架构/seed，量化记录并公布评分标准；样本足够且覆盖失败。 |
| 性能与成本 | GPU Hours、samples/s、train FLOPs（测量方法）、NFE、端到端延迟、peak VRAM | 同设备/精度/生成张数/条件编码缓存策略；至少报告端到端与 DiT 本体两种开销。 |

**人工校准流程**：先拿至少约 100 张真实 ArtBench 绘画及约 100 张生成或人为退化图像做匿名判断，比较自动 VQA/检测输出与人工的一致性、误判率；如果自动评分在某个类别可靠性不足，修正评分器或将该类别降为探索性，不允许用 VLM 自评分充作人工真值。来自旧几何任务的规则解析器只用于几何 sanity，绘画域另行校准。

### 8.4 分母与失败判定

- 无法辨识、缺主体、颜色错误、数量错误、关系错误、生成失败都必须保留；**模型崩图不能在成功率分母里被删除**。
- 单独记录每种要求适用的样本数。无空间关系要求的 Prompt 不计入关系准确率分母，但仍计入总体可用性分母。
- 使用固定提示词、固定初始噪声集合和匿名架构标签生成全量图；**禁止只展示最漂亮的样本**，技术报告同时给成功/失败例图和全量索引。
- 如自建 Benchmark 引入人工或 VLM 标注，存储版本与生成时使用的 VLM 必须区分；同一个 VLM 既生成 Caption 又充当评测器可能存在共同系统偏差，应做独立人工审核和必要时不同评测器交叉核查。

---

## 9. 分阶段执行任务：完成一阶段，再进入下一阶段

当前勾选与未完成事项均以实测证据为准。[P0/P1 交付](../reports/v2/01_p0_p1_delivery.md) · [全量资源待办](../待办清单.md)

### P0｜仓库盘点与文档迁移（**Codex 接到本文件后立即开始**）

**依赖**：现有仓库可读。\
**允许修改**：README、旧总计划“未来任务部分”、docs/data、AGENTS、docs/learning 纯导航、阶段交接/新报告；可新建配置模板/测试但不大改核心源码。\
**不得修改**：五份 S1–S3 Notebook、历史报告/结果/旧权重、当前用户其它未提交改动。

- [x] `P0-01` Git/环境/目录/测试快照；五 Notebook SHA-256。
- [x] `P0-02` v1→v2 代码与任务映射、复用/归档清单，明确旧状态证据。
- [x] `P0-03` 给 README 设 v2.0 入口、保留教学记录链接、区分现有可运行 CLI 与计划命令。
- [x] `P0-04` 更新主任务总清单与 docs/data：移除**未来执行**的 Sana LoRA、OpenMoji、旧扩散/Flow 正式四组比较；保留历史研究设计注释和历史执行事实。
- [x] `P0-05` 更新 AGENTS 和工作交接规范，新增 `reports/v2/00_repository_audit.md`、`00_migration_diff.md`、`00_handoff.md`。
- [x] `P0-06` 现有基础测试及文档链接检查，给出新代码工作第一步 `P1-01`，**停止本回合**。

**验收**：新旧身份清楚；旧教学资产 Hash 不变；文档不再给出相互冲突的正式任务；没有大规模算力、API、权重下载；每个修改可用 git diff 审查。

### P1｜ArtBench 接入、图文审计与冻结

- [x] `P1-01` 决定采用 ArtBench Public Domain 256x256 全量 46,298 张，固定正式图像目标为 256×256。
- [x] `P1-02` 核验固定 Parquet 来源、全量行数、字段、文件哈希、解码与尺寸。
- [x] `P1-03` 验证可复现的组级划分、精确重复隔离和近重风险记录。
- [x] `P1-04` 应用轻量 Caption 清洗与 CLIP 长度校验，输出原文、清洗结果和错误记录。
- [x] `P1-05` 完成 240 张配对代理审计，记录图像技术缺陷、语义错误和待人工确认项。正式条件训练前待确认是否接受当前 Caption 噪声。
- [x] `P1-06` 冻结清单、版本化数据卡和划分证据，完成进入 P2 的交接。

**交付**：`reports/v2/data/` 的图文审计、分组证明、数据卡和冻结清单身份。实际状态以 [P1 交付](../reports/v2/01_p0_p1_delivery.md) 为准。

### P2｜冻结 VAE/CLIP 的数据域验收与缓存

- [x] **资产准备。** 核验 SDXL VAE 与 CLIP 文本塔，并下载 AlexNet 和 LPIPS 校准权重。[资产回执](../reports/v2/components/p2_assets.json)

- [x] `P2-01` **随机 200 张** VAE 重建与 PSNR/SSIM/LPIPS 已运行，附可复查材料与代理抽查。用户视觉审核单列为 P2-02 的门槛（第 4 章不可删）。
- [x] `P2-02` 用户视觉审核通过，冻结保留 SDXL VAE；登记细线、文字及小物体的信息损失，不删样本。[用户批准](../reports/v2/components/vae_user_approval.json)
- [x] `P2-03` CLIP token/mask/空条件/维度/长度验收，全量 Caption 重新分词，另做真实输出与投影梯度对照。
- [x] `P2-04` [固定 20 张本机数值验收](../reports/v2/components/artbench_local_fp32_20.json) 通过；正式 256px Latent 与 Text Cache 已构建，train 37,042、validation 4,637 全部通过 [独立回读](../reports/v2/components/artbench_cache_formal.json)，test 未编码。

**交付**：[P2 工程验收](../reports/v2/02_p2_delivery.md) 与 `artifacts/cache/artbench_pd_256/`。旧 128px 可重建缓存已清理，相关源码、配置与结果保持历史身份。

### P3｜统一 Flow + Plain DiT Baseline 工程

- [x] `P3-01` 统一 `Condition`、VAE 包装、`FlowObjective`、Euler Sampler、CFG、可训练模块契约。
- [x] `P3-02` 从已验证的旧 DiT 代码/学习成果迁移或实现 B0；统计约30M参数并提供模型规格表。
- [x] `P3-03` 运行 Flow 二维解析测试、patch/shape/mask/梯度/时间方向/输出数值测试。
- [x] `P3-04` 补齐 `minigen train/sample/cache` 的真正可用入口（逐个实现）、配置和 resume；CPU tiny 与 MPS 本地验收通过，CUDA 冒烟按本轮授权移入 P3-06 待云端执行。
- [ ] `P3-05` 16–32 张数据小样本过拟合，500–1000 步有限预算可辨结果、统一日志和短训质量诊断。已完成 1000 步及全部固定采样；主体仍不可辨，质量门槛未通过，不能勾选。
- [ ] `P3-06` GPU pilot：吞吐、显存、step cost、成本估算和训练审批配置。入口与配置已准备，真实 CUDA/AMP、性能和预算审批待云端验收。

**交付**：B0 在 256px 的可靠训练/采样基础、可运行命令、日志、模型参数统计、恢复验证、预算建议；**不因 loss 下降就称生成成功**。

2026-11-10 本地成果见 [P3 交付](../reports/v2/03_p3_delivery.md)：B0 **30,359,824** 参数，P3-01～04 本地工程通过；P3-05 质量未通过、P3-06 CUDA 待验。停在 P4 入口，不自动启动训练。

### P4｜第一轮 B0 训练、再添加 B1/B2

**本轮用户更新的执行顺序**：先完成 P3–P5 主体代码与本地小规模端到端验证，再由用户统一安排云端正式训练与评测。P4-03/05 的工程可在下一轮获准后先做；P4-01/02/04/06 的质量、预算审批和正式运行仍须独立验收，任务编号保持不变。B0 短训未达标的诊断不能跳过。

- [ ] `P4-01` 用户审核短训质量、单卡成本、数据/组件门槛后批准 B0 长训。
- [ ] `P4-02` B0 正式训练、多个中间 checkpoint、完整 sampling/validation/失败案例。
- [ ] `P4-03` 以相同输入输出协议实现 B1 Single Stream，完成 mask/位置编码/参数/梯度/恢复测试及短跑。
- [ ] `P4-04` 在与 B0 相近的冻结训练预算下，正式训练 B1。
- [ ] `P4-05` 实现 B2 Hybrid（Double Stream Joint Attention → Single Stream），测试独立 QKV 与融合、mask、梯度及短跑。
- [ ] `P4-06` 在与 B0/B1 相近预算下，正式训练 B2。
- [ ] `P4-07` 汇总 B0/B1/B2 同条件训练日志、Checkpoint、生成图和成本，冻结用于正式评测的选型标准。

**交付**：三组可复现训练结果。若 B0 质量明显不达标，先按第 12 章诊断，再决定是否继续 B1/B2；不能为了完成表格盲目花费预算。

### P5｜MiniGen-Bench、主实验矩阵与报告

- [ ] `P5-01` 在正式训练前就已构建/冻结的 Benchmark 上执行 ID/OOD/Paraphrase（不能事后改题）。
- [ ] `P5-02` 计算 KID/CMMD、CLIPScore、VQA/属性与关系准确率、人工盲评及评测器可靠性。
- [ ] `P5-03` Euler/Heun、NFE、CFG 实验与 SDPA 工程性能对照。
- [ ] `P5-04` 相同训练 step、GPU time/FLOPs 的质量—效率图，明确每个模型实际参数量差异。
- [ ] `P5-05` 生成真实实验矩阵、表格、图像画廊、失败案例、完整结果清单、可重复脚本。
- [ ] `P5-06` 用户决定是否再补 seed=1 或 10M/50M/Chimera，不把扩展自动计入已承诺任务。

**交付**：`reports/v2/experiments/`、`reports/v2/benchmark/`、`reports/v2/final/`，含真实 CSV/JSON、配置、图表、指标可信性说明。

### P6｜可选研究扩展（默认跳过，需明确新批准）

- [ ] 再跑 B0/B1/B2 seed=1（三个新增训练）。
- [ ] 选择一个架构做 10M/30M/50M 的固定计算预算对照（30M 复用，新增10M与50M）。
- [ ] Chimera-Lite：先做研究卡和一组短序列/长序列性能假设，再决定值得借鉴的最小机制。
- [ ] VAE 替换、位置编码消融、更大分辨率或 DDP 只在实测瓶颈成立时开新研究分支。

### P7｜工程交付、README、简历与复现

- [ ] 稳定的 `train` / `sample` / `evaluate` / `cache` 入口；并给出无需大规模训练即可运行的 toy/CPU smoke 命令。
- [ ] 项目 README：最终效果画廊、三架构图、真实来源与训练状态、最小复现路径、硬件预算、评测与局限。
- [ ] 数据卡、模型卡、源码来源表、公开许可/权重再分发限制、实际可分享的配置/少量样本。
- [ ] `reports/v2/final/MiniGen_v2_research_report.md`：研究问题、数据、训练协议、**实验矩阵**、质量/效率、失败与总结。
- [ ] 5 分钟项目介绍、深入技术问答、简历项目说明（**仅写已验证的成果与真实数字**）。
- [ ] 干净环境安装、最低烟测、权重恢复和固定示例生成复现；有云资源时补真实 CUDA 复验。

---

## 10. 对旧文档的明确修改清单（Codex 需要真正执行）

> 目标：用户下一次打开仓库，只看到一条当前有效的 v2.0 主线，同时仍能追溯 S1–S3 的知识和历史实验。

| 文件 | 动作 | 关键要求 |
|---|---|---|
| `MiniGen项目任务清单与学习路径.md` | **优先更新**，升级标题/版本/路线及未来任务 | 保留 v1 原始文件历史的 Git 可追溯性；前部改为 v2 总任务索引，S1–S3 作为已执行/待个人验收的历史段落；旧 S4/S5 “Sana/LoRA/DDPM vs Flow”标记 superseded，链接本规格书；不要把历史记录改成没发生过。 |
| `README.md` | **必须更新** | 明确 v2 目标、数据与架构、当前真实状态、快速入门；分隔“可执行”“计划中”的 CLI；链接五份教学 Notebook（只读）和新版路线、实验报告。 |
| `docs/data.md` | **必须更新** | ArtBench 为正式数据，记录原生 256×256、图文审计与 Caption 清洗、200 张 VAE 重建、划分、缓存和来源许可；MNIST/几何数据移至 legacy/教育用途。 |
| `AGENTS.md` | **必须更新** | 添加第 0 章硬约束、阶段优先级、试验签核、历史保护、付费前询问和交接规范；不覆盖其它非相关用户自定义规则。 |
| `docs/ai-workflow.md` | 按需调整 | 从“每周教学任务优先”切换到“P0–P7 可验证迭代”，保留对用户个人核心验收的说明。 |
| `docs/learning/README.md` 和 `learning/README.md` | **仅更新导航/状态** | 五份教学 Notebook 不变；S4/S5 新课件若需要则**新建独立文件**，不覆写旧 Notebook；旧 L03/L04 个人未验收的状态如实保留。 |
| `configs/datasets/sources.yaml` 等数据配置 | 增加新来源，不破坏原项目 | 锁定 ArtBench revision 与 Parquet 分片；旧 MNIST/几何/OpenMoji 来源保留标记 legacy；新增正式配置、manifest schema/处理版本。 |
| `configs/experiments/s4_preregistered.yaml` 等旧 S4 预登记 | **不可原地改写历史预登记** | 保留旧版并加 obsolete 说明/导航；新建 `configs/experiments/v2_*.yaml` 与 `reports/v2/experiments/preregistered.md`。 |
| `configs/models/`、`configs/training/`（如存在） | 新建 v2 规格 | 三模型、CLIP/VAE、30M 参数统计、训练与缓存身份；不要假设旧文件结构。 |
| `reports/stage1/`, `stage2/`, `stage3/` | **只读历史** | 不覆盖真实输出和失败记录。新增 `reports/v2/` 阶段报告。 |
| `docs/papers/README.md`、研究卡 | 更新前沿导航 | 保留已有论文阅读痕迹；增补现代参考的代码链接和“简化借鉴范围”；不虚构读过或实现。 |
| `.gitignore`、依赖与 scripts | 在审计后按需增量修改 | 保护 `data/`, `artifacts/`, 权重、私密标注/凭据；新依赖明确 optional extras；不得在 import 时自动下载。 |
| `docs/learning/S1_*.ipynb`, `S2_*.ipynb`, `S3_*.ipynb` | **严禁修改** | Hash 必须一致；即使旧报告说需要补改也另开新文件。 |

### 10.1 不冲突的“唯一当前入口”

建议根目录放 `MiniGen_v2.0_最终重构与执行规划.md`（本文件），README 顶部明确指向此文件。旧规划如已经被其他文件链接，不要断链；可保留原标题与历史导航，顶部加“未来任务已由 v2 计划取代”的显著提示。不要仓促同时维护多个内容冲突的“最终规划”；**以本文件的核心决策和 `reports/v2/00_handoff.md` 的实际进度为准**。

### 10.2 Codex 任务交接格式（每一阶段结束强制填写）

`reports/v2/00_handoff.md` 每轮至少包含：

- 当前阶段、已完成任务 ID、**证据路径/运行命令**、实际日期与 commit（若未提交写 working tree）。
- 修改/新增/未修改文件；五个 Notebook SHA 对照；已有用户改动保护状态。
- 通过的单元测试、失败测试和原因、短训结果与代表性输出。
- 当前数据版本、VAE/CLIP/VLM revision、样本数、缓存身份、实验 run_id、真实 GPU 开销/预算。
- 当前最大风险、阻塞原因、**下次唯一优先任务**，以及需要用户做决定的选择点。

每阶段另写一份能看懂的**成果说明**，既能回答“做了哪些代码/实验/评测、结果如何”，也明确没有完成和下一步任务。禁止单纯列改动文件名而不说明验证内容。

### 10.3 Git 操作边界

- 保留当前工作树用户改动；先检查再改文件。建议单独 feature branch，但分支创建不得丢改动。
- 禁止修改五个 Notebook 的任何一个字节；不强行 `ruff format` 原有 Notebook 或历史用户代码。
- 不自动提交大权重、大数据、私密日志/秘钥；除非用户已授权，不自动推送远程。
- 任何 cloud GPU、API、批量数据抓取等可能产生实际成本的动作先提供 dry-run 和预计费用、获得批准后执行；本文件是开发许可，不是付款授权。
- P0 文档迁移不应强制触发新实验；代码改动遵循“最小影响、真实测试和可回滚”的规则。

---

## 11. 最终研究报告结构（与实验矩阵一一对应）

1. **摘要**：MiniGen 做了什么、使用多少数据和 GPU、哪些结果是真实测得的。
2. **研究问题**：比较三个现代轻量级条件生成主干的**质量、文本遵循、计算效率**。
3. **已有基础与迭代**：v1 教学链路为新工程提供的实现与验证资产；v2 改进动机与研究边界。
4. **数据与标注**：ArtBench 来源、绘画题材、图文清洗和划分、源 Caption 质量与局限。
5. **冻结组件**：SDXL VAE 重建 200 张及 CLIP 特征验证、参数冻结清单。
6. **架构与公式**：B0 Plain、B1 Single、B2 Hybrid 的模块图、Token shapes、Flow Matching、CFG 和采样。
7. **训练系统**：硬件、吞吐、缓存、BF16/SDPA、恢复验证、真实计算量。
8. **预登记实验矩阵**：A 架构 / B Seed / C Scaling / D Sampler&CFG / E 可选扩展（实际完成程度明确打勾）。
9. **MiniGen-Bench**：测试切分/提示词、指标定义、人工评分规则、测试封存与误差校准。
10. **定量结果**：KID/CMMD、CLIPScore、VQA/组合泛化、成本、延迟、显存、置信信息和质量—算力图。
11. **定性结果与失败**：固定种子全量画廊、好坏案例、训练近邻、局限与后续方向。
12. **结论与复现**：能支持的结论、不能支持的结论、配置/模型卡/运行命令和简历可使用的真实描述。

必须把实验矩阵原封不动保留为报告表格骨架；没有跑到的实验记 **NOT_RUN**，不得直接填猜测的 KID 或性能提升百分比。

---

## 12. 风险、应急决策与阶段验收

| 问题 | 先验证什么 | 默认处理 | 是否改变既定路线 |
|---|---|---|---|
| 绘画风格和扫描质量差异 | 固定样本的图像网格、技术缺陷和题材分布 | 区分柔和笔触与扫描、压缩和裁切问题；按实际缺陷分层报告 | 保持全量数据范围和 256×256。 |
| 数据数量或解码不符 | 分片哈希、行数、图像格式和尺寸 | 修复下载并重新审计；仍异常则保留清单与证据 | 维持 46,298 张冻结目标。 |
| VAE 轮廓损坏 | 200 张原图/重建图、数值与人工分类 | 先排查预处理、scale、精度，再评估备选 VAE | 只有失败证据才可统一替换。 |
| 源 Caption 含视觉事实错误 | 对图抽查主体、场景、颜色、数量与关系 | 按严重程度修订、标记待审或扩大抽查 | 固定 CLIP 文本塔。 |
| CLIP 文本条件没效果 | 特征变化、投影梯度、mask、CFG、caption 准确度 | 修条件通路和数据，不盲目增加 DiT 参数 | 不直接换文本编码器。 |
| 30M B0 不收敛 | 16–32 张过拟合、Flow 时间方向、VAE scale、optimizer、EMA、CFG | 先小规模排查，再考虑小范围 LR/步数调整 | 保留30M为初始目标，但允许实证修订。 |
| 256px 显存不足 | microbatch、attention 后端、optimizer 内存 | BF16+SDPA+累积、按需 checkpointing | 单卡优先；不直接加 DDP。 |
| B1/B2 工程复杂 | shared interface/JointAttention mask/位置编码 | 先完成 B0，逐架构加入；失败写明 | 最低成功可先 B0，计划成功需三架构。 |
| 评测器误判绘画 | 100 真图+100 生成/退化图人工校准 | 改评测器/降级指标，不美化结果 | Benchmark 题目冻结不变。 |
| 云预算超出 | 真实 step/s、GPU 小时、评测成本 | 缩短先导实验、减少可选项/seed，提报审批 | 不强制增加资源。 |
| Seed 波动大 | 独立重复、置信区间、数据/代码身份 | 报告方差，不夸大架构优劣 | 可选补种子。 |

**各阶段 Go/No-go**：

- P0→P1：Notebook Hash 不变、旧进度不伪造、当前入口一致。
- P1→P2：46,298 张图像身份和 256×256 解码验收、图文抽查和 Caption 清洗有记录，分组隔离可复现，正式清单冻结。语义或近重问题按审计证据报告。
- P2→P3：VAE 200 张用户审核、CLIP 验收、固定 20 张本机数值检查和完整 train/validation 缓存验收通过。CUDA 运行验收另记；是否进入 P3 由用户决定。
- P3→P4：Flow 数学/梯度、B0 小样本过拟合、采样、恢复和预算审批通过。
- P4→P5：至少有真实 B0/B1/B2 的可比检查点与冻结评测配置；若某模型失败，可做失败分析但必须标记“不构成有效架构对比”。
- P5→P7：评分校准、完整实验配置、失败案例和复现链都可追溯。

时间里程碑可以弹性调整，**不以牺牲正确性来迎合旧版 12 月 15 日日期**。任何变更需写清理由、影响和需重训哪些组。

---

## 13. 参考源码与方法入口（只借鉴需要的部分）

| 来源 | 入口 | 本期具体借鉴 |
|---|---|---|
| 官方 DiT | https://github.com/facebookresearch/DiT | patchify、adaLN-Zero、基础 Transformer 与输出头。 |
| Qwen-Image-2.1 | https://github.com/QwenLM/Qwen-Image-2.1 | 了解 Single-Stream 图文联合注意力思想；**不是把 7B 模型训练脚本直接照搬**。 |
| HunyuanImage-2.1 | https://github.com/Tencent-Hunyuan/HunyuanImage-2.1 | 读 `hyimage/models/hunyuan/modules/hunyuanimage_dit.py`、`models.py`、`posemb_layers.py` 的 Double/Single Block、Joint Attention、RoPE。 |
| Diffusers | https://github.com/huggingface/diffusers | 冻结 VAE/CLIP 组件使用、模型接口与可复现配置；不是让 Diffusers 替代自研 DiT 核心。 |
| SDXL VAE | https://huggingface.co/madebyollin/sdxl-vae-fp16-fix | 权重、配置与 scaling_factor；固定版本。 |
| CLIP-B/32 | https://huggingface.co/openai/clip-vit-base-patch32 | 预训练 Text Tower、77 长度与 512 hidden；固定版本。 |
| ArtBench Public Domain 256x256 | https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256 | 46,298 张图像及英文 Caption，固定 Parquet 身份、保留来源信息。 |
| Flow Matching | https://arxiv.org/abs/2210.02747 | 线性概率路径、velocity 回归和 ODE 推导。 |
| Abra（研究方法） | https://arxiv.org/abs/2608.17286 | **借鉴固定计算预算下模型大小与训练时间的对照设计**，不复现大规模论文配置。 |
| Chimera（可选） | https://arxiv.org/abs/2607.28611 | 仅未来候选；本期不强制实现混合注意力/MoE。 |
| 公开评测思想 | DiffusionBench / GenEval 2 等 | 借鉴多维评测、固定协议和人工误差审计；引用前须由 Codex 核实具体仓库版本与数据协议。 |

当任何链接、仓库路径或第三方方法发生变更，**在研究卡记录“原先假设、实际核验、修正”**，不要为了与旧记忆一致而使用不存在的源码或夸大方法来源。

---

## 14. 启动命令规范与第一回合“完成”的样子

以下是建议命令的**类别**。真实语法由 P0 审计及后续实现确定；不要现在直接假设已可执行：

```bash
# P0 中可以尝试的已有基础检查（网络/依赖不足则记录）
git status --short
uv run --locked minigen doctor --device cpu --json
uv run --locked pytest

# P1～P3 新增 CLI 后应支持的示意入口；此时尚不能保证可运行
uv run --locked minigen data audit --config configs/datasets/artbench_pd_256.yaml
uv run --locked minigen cache --config configs/cache/artbench_pd_256.yaml --dry-run
uv run --locked minigen train --config configs/experiments/v2_b0_plain_flow.yaml --dry-run
uv run --locked minigen sample --config configs/experiments/v2_b0_plain_flow.yaml --prompt 'An oil painting of a harbor under a cloudy sky.'
uv run --locked minigen evaluate --config configs/evaluation/minigen_bench_v1.yaml
```

### 第一回合完成标准（Codex 必须按此交卷）

1. 清楚告诉用户**当前仓库有哪些可用模块和已有实验，哪些只是 Notebook 教学/CLI 占位**。
2. 更新 `README.md`、旧任务主清单、`docs/data.md`、`AGENTS.md` 与可安全修改的导航，使它们的未来工作全部指向本 v2.0 方案。
3. 输出 `reports/v2/00_repository_audit.md`、`00_migration_diff.md`、`00_handoff.md`，列明**实际会修改的 Python 文件路径和按 P1→P3 的先后次序**。
4. **五份 S1–S3 Notebook SHA-256 保持不变**；历史报告和旧实验不被覆盖；没有擅自启动付费操作或不必要大下载。
5. 给出**下一阶段 P1-01 的具体数据审计脚本/命令计划**，但不抢跑未来的大规模训练。

> **最终执行原则**：先理解现有仓库 → 保护学习资产 → 更新唯一主规划与文件导航 → 冻结 ArtBench 数据与清洗后的 Caption → 验收 VAE/CLIP → 打通 B0+Flow → 再训练 B1/B2 → 运行冻结实验矩阵和 Benchmark → 最终形成真正可信的技术报告。**不增加本期强制架构、不把可选 Scaling 变成负担、不为进度表伪造成果。**
