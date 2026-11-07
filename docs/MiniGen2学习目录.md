# MiniGen v2.0｜S4–S7 学习 Notebook 目录与 Codex 生成规范

> **用途**：将本文交给 Codex，与《MiniGen v2.0｜最终项目重构、开发与研究实验执行规格书》配合使用。本文专管**用户学习与个人核心实现**，不替代工程任务清单。版本：v1.0，2026-11-06。
>
> **严格保护**：`docs/learning/` 中原有 **5 份 S1–S3 `.ipynb` 全部只读**，包括 Markdown、代码、执行结果、输出、元数据；不要在它们内部补充 v2 内容，不要重跑覆盖。历史报告和历史实验结果也不能改写为新架构成果。

## 0. 学习边界与现有知识衔接

已上传的 S1–S3 课件涵盖：DDPM 与 MNIST、VAE、潜空间与 DiT、扩散训练/采样、CLIP 文本条件、Cross-Attention、CFG、条件评测，以及**二维 Flow 的初步预研**。它们是前置课程；接下来的课程应**引用和链接**旧内容，不大篇幅重复，也不把“课件跑过”误判为“个人核心实现通过”。

新的正式研究方案统一为：**Plain DiT + Flow / Single-Stream DiT + Flow / Hybrid DiT + Flow**；图像 256×256，生成主干目标约 30M；SDXL VAE 和 CLIP-B/32 **预训练冻结**；ArtBench Public Domain 256x256 为正式数据集。本期不做生成后训练、Sana LoRA；10M/50M 与 Chimera **等首轮 Baseline 后再决定**。

> **区分标识**：本文的 S4–S7 表示**学习课程**；工程计划的 P0–P7 表示**Codex 实施阶段**。不得混用。原 v1.0 中“S4 = Sana LoRA”的未来任务不再沿用，但历史文档须可追溯。

## 1. 学习目录总表（7 份必修 + 1 份选修）

| 学习序号 | 建议新 Notebook 文件名 | 核心问题 | 对应工程进度 | 优先级 |
|---|---|---|---|---|
| S4-01 | `S4_01_FlowMatching原理与ODE采样.ipynb` | 为什么预测速度？怎样从噪声积分成图像？ | P3：训练目标/采样器 | **必修** |
| S4-02 | `S4_02_LatentFlow与PlainDiT训练闭环.ipynb` | 怎样把 Flow 真正接入已学的条件 DiT？ | P2–P4：Plain Baseline | **必修** |
| S5-01 | `S5_01_JointAttention与2DRoPE.ipynb` | 文本和图像 Token 怎样在同一注意力层交互？ | P3：条件与位置编码 | **必修** |
| S5-02 | `S5_02_SingleStreamDiT架构与实现.ipynb` | 单流 DiT 与 Cross-Attention DiT 有什么实际差异？ | P3–P4：Stream Baseline | **必修** |
| S6-01 | `S6_01_HybridDiT双流单流架构.ipynb` | Dual-Stream 和 Joint Attention 如何协作，再转入 Single-Stream？ | P3–P4：Hybrid Baseline | **必修** |
| S7-01 | `S7_01_训练系统性能与公平架构实验.ipynb` | 如何让三模型比较公平？如何分析显存和速度？ | P4–P5：训练与对照 | **必修** |
| S7-02 | `S7_02_MiniGenBench评测与消融实验.ipynb` | 如何客观判断生成质量、文本遵循和计算成本？ | P1/P5：评测预登记与报告 | **必修** |
| S8-01 | `S8_01_Chimera混合注意力选修.ipynb` | Chimera 的混合注意力有哪些设计，何时值得缩小复现？ | P6：仅在可选扩展获批后 | **选修，先不生成** |

## 2. S4｜Flow Matching：从原理到 Plain Baseline

### S4-01：Flow Matching 原理与 ODE 采样
1. **从已知 DDPM 出发**：噪声预测、速度预测、Flow Matching、Probability Flow ODE 的区别；“速度”是与 Latent 同形状的向量场，不是标量。
2. **直觉与数学**：从噪声 `ε` 到数据 `z` 的线性路径 `z_t=(1-t)ε+t z`；条件目标速度 `u_t=z-ε`；网络学习的边际速度场为何不同于单个配对轨迹。
3. **训练目标**：`L=E||v_θ(z_t,t,c)-(z-ε)||²`；时间方向 0→1；t 采样、mask、损失归一化、数值稳定性。
4. **ODE 与采样器**：Euler、Heun 的推导与代码；**时间步数 ≠ NFE**，比较时按网络前向次数对齐；和 DDPM/DDIM 的联系与区别。
5. **可执行练习**：二维点分布→轨迹可视化→实现线性场的 Euler/Heun→解析解误差测试→时间方向/端点测试。
6. **个人核心 TODO**：亲手写 `flow_interpolate`、`flow_target_velocity`、`euler_step`、`heun_step`；Codex 准备可运行的参考数值演示、独立单元测试，不代写用户 TODO。

### S4-02：Latent Flow 与 Plain DiT 训练闭环
1. **图文双输入链路**：ArtBench 图像 → 冻结 VAE Encoder → `z`；Caption → 冻结 CLIP Text Encoder → `c`；`DiT(z_t,t,c)`；训练 vs 推理对比。
2. **256×256 张量追踪**：SDXL VAE 假定 8 倍压缩得到 `[B,4,32,32]`；patch size 2 → 256 image tokens；通过实际组件配置与运行验证，不能硬编码为永远不变。
3. **Plain DiT 模型结构**：Patch Embedding、时间调制、Self-Attention、Cross-Attention、Velocity Head；冻结组件与训练组件的参数/梯度边界。
4. **完整训练与采样**：构造 `z_t`、速度标签、masked conditional features；CFG 对**速度预测**的组合；Euler/Heun；Latent scaling 与 VAE 解码。
5. **数值与集成测试**：固定监督拟合、noise-to-latent 方向、形状/梯度、resume、同 prompt 不同 seed、空条件和 CFG；区分可运行性与生成质量。
6. **个人核心 TODO**：`flow_loss` 与最小 `sample_ode`；实现与原 `src/minigen/` 接口对照，不能只在 Notebook 中写孤立的一套。

## 3. S5｜联合注意力、空间位置与单流生成主干

### S5-01：Joint Attention 与 2D RoPE
1. **复习入口**：先引用 S3 的 Cross-Attention，不重复讲完 CLIP/CFG；比较 Self/Cross/Joint Attention 的 Q/K/V、信息流与计算量。
2. **两种融合设计**：`[text_tokens; image_tokens]` 的共享序列注意力；以及模态分别投影 QKV 后合并注意力再分离的 Joint Attention。明确二者**有关联但不是完全相同的实现**。
3. **图像 2D RoPE**：行列坐标 `(y,x)`，维度拆分、频率、Q/K 旋转与点积性质；Text Token 的位置处理要与图像位置明确区分。
4. **Attention Mask**：文本 padding、图像有效位置、双向与 block-causal 可见性矩阵；**不要把参考论文具体 mask 当作所有单流模型必须采用的规则**。
5. **可执行练习**：手算 2×2 patch 的位置编号；实现 2D RoPE；构造 3 个文本 token + 4 个图像 token 的 Joint Attention；对照显式矩阵与 SDPA；验证 padding 不影响有效输出。
6. **个人核心 TODO**：最小 `apply_2d_rope(q,k,coords)` 与 `joint_attention(...)`；Q/K shape、RoPE 维度、梯度、mask、NaN 测试。

### S5-02：Single-Stream DiT 架构与实现
1. **从 S5-01 组装主干**：Text Projection、Image Patch Embedding、Timestep Conditioning、序列融合、输出仅恢复 image tokens。
2. **与 Plain DiT 的比较**：Cross-Attention 的条件读取 vs 联合/单流 token 自注意力；共享与独立投影的权衡；文本 token 的 padding 与图像 token 的输出索引。
3. **前沿参考的边界**：介绍 Qwen-Image-2.1 的设计思路与参考源码入口，但本项目是**受其启发的小模型**，不能宣传成原模型等比例复现。
4. **一次完整前向**：逐层手工写 Shape 表；速度输出 `[B,4,32,32]`；统计真实可训练参数（目标约 30M，非强制精确值）。
5. **可执行练习**：同一 noise、t、caption 对比 Plain/Stream 输入输出；padding、时间变化、训练梯度、推理输出；32×32 或 64×64 教学短跑后再接 256×256。
6. **个人核心 TODO**：`SingleStreamBlock`、`SingleStreamDiT.forward` 关键流程；独立测试与实际工程适配。

## 4. S6｜Hybrid DiT：双流联合处理，再合流

### S6-01：Hybrid DiT 双流与单流架构
1. **模型直觉**：为什么先保留 text/image 两条流，再进入联合序列；何时联合注意力已经让双流交换信息。
2. **Double-Stream Block**：text/image 各自归一化、QKV/MLP/调制；将 QKV 合并进行 Joint Attention，再按原 token 数拆回两条流；残差与梯度路径。
3. **Single-Stream Block**：文本与图像合并后继续计算；如何只从 image tokens 预测 latent velocity。
4. **参考论文与代码映射**：HunyuanImage-2.1 源码中的 `MMDoubleStreamBlock`、`MMSingleStreamBlock`；不要误将 Chimera 的 hybrid attention 视为同一个概念。
5. **模型尺寸与公平性**：双流块每层参数量不同，不能以“层数相同”假装参数量相同；通过代码统计调整 Block 比例、隐藏维度和计算量。
6. **可执行练习与核心 TODO**：实现 `DoubleStreamBlock`、合流/拆流逻辑、Hybrid forward；验证 shape、mask、梯度、可视化注意力、与 Plain/Stream 共用训练接口。

## 5. S7｜训练、评测与研究实验（不再扩展模型架构）

### S7-01：训练系统性能与公平架构实验
1. **为什么要控制变量**：三架构共享 ArtBench 划分、VAE、CLIP、Flow 目标、训练条件和验证策略；记录参数量、Token、实际 FLOPs、GPU 小时。
2. **核心架构实验矩阵**：B0/B1/B2 各约30M、同数据与预算；先 seed=0 跑通；资源允许再补 seed=1。
3. **Abra 方法的正确借鉴**：固定**总训练计算量（或有注明的 GPU 时间）**，比较不同模型规模/架构性能；10M/50M 是选做的规模实验，**绝不预先启动 3×3 全组合训练**。
4. **性能优化**：BF16、PyTorch SDPA、梯度累积、可选 activation checkpoint；解释标准注意力、SDPA 和 FlashAttention 的关系；剖析显存、吞吐与注意力 FLOPs。
5. **分布式的适用边界**：默认单 GPU，何时才考虑 DDP/ZeRO；理解梯度累积与全局 Batch、同步与可复现性。
6. **可执行练习**：测 micro-batch→峰值显存→吞吐、训练步耗时、GPU 成本预测；比较普通 attention vs SDPA 误差与速度；检查 checkpoint 恢复。

### S7-02：MiniGen-Bench 评测与消融实验
1. **评价对象与数据隔离**：ID、Compositional OOD、Paraphrase Prompt；验证集用于选 checkpoint 和 CFG，测试集只在冻结方案后使用。
2. **图片分布质量**：KID、CMMD 的特征空间、真实图像参考集、样本量和区间；为什么不能靠少数好看生成图证明模型能力。
3. **文本遵循**：CLIPScore、VQA/分解式问题、颜色绑定、数量、空间关系；在 2D 插画上自动判分可能失败，必须有人审和标注校准。
4. **生成与推理消融**：Euler/Heun 用相同 NFE 对照，CFG 1/2/3、NFE 10/20/40，记录 Prompt、noise seed、checkpoint 身份。
5. **研究统计与图表**：逐 seed 原始结果、均值/区间、质量—成本图、失败样本分组、训练中途 checkpoint 曲线；避免根据测试集反复调参。
6. **可执行练习**：为最小示例集生成 JSONL Prompt 与真值 schema；计算一组 toy 计数/绑定分数；从 run log 自动生成带证据来源的报告表。

## 6. S8｜选修（当前不要生成）

### S8-01：Chimera 混合注意力
- 阅读论文/官方代码，厘清混合注意力的算子、容量、效率与适用序列长度；与 Hunyuan 的双流/单流区别。
- 先做 256×256 的吞吐/显存先验检查，再决定是否开发 `Chimera-Lite`。
- 必须等 B0/B1/B2 第一轮正式结果和成本明确后，由用户单独批准；**不得抢占 P0–P5 的开发预算**。

## 7. Codex 生成 Notebook 的统一要求（必须执行）

### 7.1 单份 Notebook 的固定教学结构
1. 本课学什么、与 S1–S3 哪个部分衔接（附引用路径）。
2. **先跑一个能看到数值/图形的最小实验**，再讲直觉、动机、数学。
3. 完整张量 Shape 表 + 必要的公式推导（符号、端点和损失归一化必须写对）。
4. 参考源码定位：项目仓库已有类/函数 + 外部参考仓库/论文；区分“引用”与“自主实现”。
5. **用户亲手完成的核心 TODO**：补函数/类、通过事先编写的单元测试；不要由 Codex 自动替用户写满后标记已掌握。
6. 参考可执行最小案例、逐步运行指令、CPU/MPS 和 CUDA 差异、缓存与算力上限。
7. 独立数值测试：shape、有限值、梯度、边界、mask、恢复/一致性；区分“代码正确”和“训练质量好”。
8. 自测问题、关键易错点、可选择展开的答案、与下一章的连接。
9. 产出标记：**Notebook 参考跑通 / 用户 TODO 通过 / 实际主仓集成 / 正式训练实验**分别记录，不混为一个“已完成”。

### 7.2 文件保护及代码规范
- **不得修改、重命名、重新执行、清空输出**原有五份 `S1–S3 .ipynb`。仅新建 `docs/learning/S4_...` 之后的文件，以及相应新测试、导航和学习记录。
- 首轮**只生成 S4-01**并完成可执行检查；后续经用户学习反馈，依次生成，不一次性堆出所有代码/Notebooks。
- 用户核心算法优先由用户自己实现；Codex 负责说明、示例、测试、数据与接口适配，若用户之后明确授权，才代写核心 TODO。
- Notebook 不要新造一套与 `src/minigen/` 互不兼容的模型。参考逻辑放在工程模块中，以模块导入为主；为了数学直觉可写短小独立演示。
- 源码/权重/数据下载必须显式、可选择、可复现；默认离线/CPU 小案例可执行，不能自动触发收费 GPU 或多 GB 下载。
- 新 Notebook 自动执行只验证参考示例；个人 TODO 未完成应显示“待完成”，**不能悄悄 fallback 到参考答案**。
- 若旧依赖、旧模型接口与 v2 规格不符，记录差异并在**新文件**修正；绝不回头修改教学历史。

### 7.3 两条路线的配合方式
- **工程路线**：按正式重构文档 P0–P7 推进；数据建设 P1 / VAE 200 张审计 P2 可以与个人课程并行。
- **学习路线**：S4-01 → S4-02（Plain Flow）→ S5-01 → S5-02（Stream）→ S6-01（Hybrid）→ S7-01/S7-02（实验设计与评测）。
- **闸门**：进入某正式模型的长训前，对应个人核心理解/测试、工程短跑、数据与组件验收各自记录通过；不要求人手写完所有外围工程。
- **实验矩阵**：长期放在项目 v2 正式执行规格书的“第 7 章”，本课 S7-01 讲它为什么这样设计，S7-02 教怎样分析；不要让实验定义分叉。

## 8. Codex 第一回合指令（直接复制）

> 阅读本文件、《MiniGen v2.0｜最终项目重构、开发与研究实验执行规格书》以及现有 S1–S3 五份 Notebook；先列出学习前置、要复用的现有 `src/minigen/` 接口、拟新增的 Notebook 和测试文件。严格保护旧 Notebook。**第一回合只生成 `S4_01_FlowMatching原理与ODE采样.ipynb` 及对应必要的测试/学习导航更新**：内容必须有二维 Flow、速度场与条件速度区别、Euler/Heun、NFE 对齐、手算、Shape、梯度和数值测试、用户自行填写的核心 TODO。先用低成本演示验证 Notebook 的参考部分可执行，报告尚待用户完成的 TODO 和实际验证证据；不要先写完后续六份课件，不启动云训练，不改旧文件。
