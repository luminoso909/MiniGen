# MiniGen · 第三周学习 Q&A｜分类整合·最终版

> **专题**：RGB 空间 VAE · Latent DDPM · Diffusion Transformer（DiT） · adaLN-Zero · 工程验证  
> **学习资料**：`S2_第3周_图标数据与潜空间DiT基础.ipynb` 与第三周 Q1–Q34 学习问答  
> **整合日期**：2026-11-05  
> **整理原则**：保留每道题的 **Q 编号与提问背景**，按知识依赖重排；将重复结论统一到本页速览与复习清单；讲义证据、代码测试与未来目标严格区分。

<a id="top"></a>

## 📌 待补学｜当前还没有真正学会

> **以下两项请一直保留在开头，直到完成自己推导＋手写＋测试后再勾选。不要因为已经看过解释就标记为掌握。**
>
> - [ ] **`position_2d()` 原理**：`meshgrid(indexing="ij")` 的行/列坐标是什么？为什么分别使用 `hidden_size//2` 宽度的 `sincos_1d()`？`flatten()` 后坐标与 Patch 排列如何严格对齐？请使用 `2×3` 网格手算一遍，再写代码比对。（对应 **Q20 / Q33**）
> - [ ] **在 `DiffusionTransformer` 中创建和注册 `position_2d`**：为什么必须用 `height//patch_size` 和 `width//patch_size`，而不是潜变量原高宽？为什么位置编码是 `[1,N,D]`？为什么用 `register_buffer(..., persistent=False)`？请完成 `__init__()` 并手动检查位置和 token shape。（对应 **Q28 / Q33 / Q34**）
>
> **待验证而非“不会”**：个人手写 DiT 的所有参考测试、三阶段梯度以及固定目标拟合；目前不能仅凭代码结构相似判定完全通过。

## 本周一页总览

### 1. 三种技术各自做什么

| 组件 | 核心问题 | 本项目训练时输入 → 输出 | 是否在本周训练 |
|---|---|---|---|
| **VAE Encoder / Decoder** | 图像如何压缩、重建？ | `[B,3,128,128] ↔ [B,4,16,16]` | 使用冻结预训练权重 |
| **DDPM** | 如何按时间加噪，如何利用预测噪声采样？ | `z₀,t,ε → zₜ` | 噪声调度本身无须学习 |
| **DiT** | 给定带噪潜变量与时间，预测什么噪声？ | `(zₜ,t) → ε̂`，均为潜空间 | 需要训练，第三周主要验证基础结构 |
| **VAE Decoder** | 去噪后的潜变量怎样回到 RGB？ | `ẑ₀/s → x̂` | 冻结，生成阶段使用 |

**训练：** `RGB x → VAE Encoder → h → ×s → z₀ → 随机 t 与 ε → zₜ → DiT(zₜ,t) → ε̂ → MSE(ε̂,ε) → 更新 DiT`。

**生成：** `z_T~N(0,I) → [DiT + Sampler 反复迭代] → ẑ₀ → ÷s → VAE Decoder → RGB 图像`。

> **最重要的边界**：DiT 是一个**噪声预测网络**，不是单独调用一次就能生成 RGB 图像的完整系统。

### 2. 关键公式卡片

**VAE 固定缩放：**

$$z_0=s\,h,\qquad h=\operatorname{mode}(q_\phi(h\mid x)),\qquad \text{解码前}\ h=\hat z_0/s.$$

**潜空间前向加噪：**

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$

**Noise MSE：**

$$\mathcal L=\mathbb E_{z_0,t,\epsilon}\left[\frac{1}{CHW}\left\|\epsilon_\theta(z_t,t)-\epsilon\right\|_2^2\right].$$

**空间 Patch：**

$$N=(H/p)(W/p),\quad K=Cp^2,\quad [B,C,H,W]\to[B,N,K]\to[B,N,D].$$

**时间条件与 DiT Block：**

$$c(t)=\mathrm{TimeMLP}(\operatorname{SinCos1D}(t));\quad [\gamma_a,\beta_a,g_a,\gamma_m,\beta_m,g_m]=\mathrm{Linear}_{D\to6D}(\operatorname{SiLU}(c)).$$

$$X'=X+g_a\odot\operatorname{Attention}((1+\gamma_a)\odot\operatorname{LN}(X)+\beta_a),$$

$$X''=X'+g_m\odot\operatorname{MLP}((1+\gamma_m)\odot\operatorname{LN}(X')+\beta_m).$$

### 3. 最重要的形状表（默认 `C=4,H=W=16,p=2,D=256,depth=6`）

| 环节 | Tensor Shape | 解释 |
|---|---|---|
| RGB 图 | `[B,3,128,128]` | 原始数据 |
| VAE latent / 扩散 `zₜ` | `[B,4,16,16]` | 噪声预测的工作空间 |
| Patchify | `[B,64,16]` | 64 个 Patch，每个 16 个元素 |
| Patch Projection | `[B,64,256]` | 变为 DiT Token |
| 2D Position Buffer | `[1,64,256]` | 广播加到每个 Batch 的 Token 上 |
| 时间步 `t` | `[B]` | 每个样本一个时间 |
| Time SinCos / MLP | `[B,256]` | 共用条件 `c(t)` |
| 每层 `6D` 调制输出 | `[B,1536]` | 六组 `[B,256]` 向量 |
| 六层 DiT Block | `[B,64,256]` | 维度保持不变 |
| Final Modulation | `[B,512]` | 最后只需 `γ_final,β_final` 两组 |
| Output Projection | `[B,64,16]` | 每 Patch 预测 16 个噪声值 |
| Unpatchify / `ε̂` | `[B,4,16,16]` | 和真实噪声 `ε` 同形 |

### 4. 已完成学习 ≠ 已完成正式生成

- **讲义层面**：受控图标数据、冻结 VAE 重建、缩放和缓存、Patch/Attention/adaLN 单元验证、固定目标短拟合均有教学过程。
- **个人实现层面**：已经写出 SinCos、Patchify/Unpatchify、Attention、FFN、DiTBlock 与 DiT 主类的主要代码；仍应在最终修正版上独立跑完测试，不能以“照着结构写完”替代验证。
- **未完成的能力边界**：本周还没有证明模型经过**随机时间的大规模训练**后能从噪声生成新 RGB 样本；也未做独立生成质量评测或文本条件泛化评测。

<a id="toc"></a>
## 分类目录与问题索引

- [第 1 章：全局路线｜从 DDPM 与 VAE 到潜空间 DiT](#chapter-01)：[Q1](#q1)、[Q2](#q2)、[Q3](#q3)、[Q4](#q4)、[Q14](#q14)、[Q15](#q15)
- [第 2 章：图像数据与 VAE｜后验、缩放、重建](#chapter-02)：[Q5](#q5)、[Q6](#q6)、[Q7](#q7)、[Q8](#q8)、[Q9](#q9)、[Q10](#q10)、[Q13](#q13)
- [第 3 章：复现与工程边界｜冻结、缓存、内存](#chapter-03)：[Q11](#q11)、[Q12](#q12)、[Q16](#q16)、[Q17](#q17)、[Q18](#q18)
- [第 4 章：潜空间 DDPM｜公式、SNR 与 Noise MSE](#chapter-04)：[Q19](#q19)、[Q22](#q22)、[Q24](#q24)
- [第 5 章：Patch 与 Embedding｜二维空间信息如何进入 Transformer](#chapter-05)：[Q20](#q20)、[Q21](#q21)、[Q23](#q23)、[Q32](#q32)、[Q33](#q33)
- [第 6 章：DiT 主干｜Attention、adaLN-Zero、梯度与完整架构](#chapter-06)：[Q25](#q25)、[Q26](#q26)、[Q27](#q27)、[Q28](#q28)、[Q29](#q29)、[Q30](#q30)、[Q34](#q34)
- [第 7 章：实验与证据｜固定四样本过拟合说明了什么](#chapter-07)：[Q31](#q31)

> **检索提示**：无论本次重排到哪一章，原始问题编号 `Q1`—`Q34` 不变；可用 Markdown 搜索 `Q28` 等迅速定位。

---

<a id="chapter-01"></a>
## 第 1 章｜全局路线｜从 DDPM 与 VAE 到潜空间 DiT

**本章学习目标：** 先弄清「表示空间、扩散规则、预测网络」三个不同层次，再区分训练与采样。

**整合后的三条主线：**

- VAE 决定工作空间；DDPM 决定加噪与采样规则；DiT 学习噪声估计。
- 训练图像先编码成潜变量，推理则从随机潜噪声出发；两条路径不要混淆。
- 重建指标与生成指标回答不同问题。

<a id="q1"></a>

### Q1｜第三周到底在学什么？前三周怎样连成一条线？

**原始问题：**「我现在开始 MiniGen 的第三周的任务了。为我先介绍这一周的主要知识以及概念。」

**一句话：** 把第一周的 DDPM（扩散/去噪）、第二周的 VAE（潜变量压缩与重建）和 Transformer（噪声预测网络）接成潜空间图像生成系统。

- **VAE / 空间 Autoencoder：** 将图像 $x$ 编码为低维潜变量 $z$，再将 $z$ 解码为图像；本周主要使用冻结的预训练组件。
- **DDPM / 扩散：** 定义加噪过程与去噪学习目标；可以在像素空间运行，也可以在潜空间运行。
- **DiT：** 将潜变量切分为 patch token，利用 Transformer 及时间条件预测噪声（或在其他建模框架中预测其他目标）。
- **CFG：** 条件生成时的引导方法；**Flow Matching：** 与噪声预测不同的一类生成训练目标和概率路径视角。

训练：$x\to E(x)=z_0\to z_t\to\mathrm{DiT}(z_t,t)\to\mathcal L_{\mathrm{noise}}$。  
生成：$z_T\sim\mathcal N(0,I)\to\text{多步采样}\to\hat z_0\to D(\hat z_0)\to\hat x$。

<a id="q2"></a>

### Q2｜「其实就是 U-Net + DDPM 的架构，只不过把 U-Net 换成 DiT；然后组合 VAE、DDPM、DiT」——对吗？

**结论：** **作为整体架构直觉，基本正确**；要补充“在哪个空间扩散”与“网络架构不等于训练目标”的区分。

| 系统 | 数据空间 | 常见去噪网络 |
| --- | --- | --- |
| 经典 DDPM | 像素空间 | U-Net |
| Latent Diffusion（典型配置） | VAE 潜空间 | U-Net |
| 原始 Latent DiT | VAE 潜空间 | Transformer（DiT） |

**易错点：** DDPM 是扩散建模方法，DiT 是神经网络骨干。换成 DiT 不意味着自动改变 DDPM 的加噪公式；DiT 也能用于 Flow Matching 等其他训练目标。

<a id="q3"></a>

### Q3｜这些东西是不是已经很老了？现在学还值不值？

**结论：** 是几年前的基础技术，**不是最新论文，却仍是理解后续生成研究的重要起点**。

- VAE：2013；DDPM：2020；Latent Diffusion：2022；DiT：2022 年预印本 / ICCV 2023；Flow Matching：2022 年预印本 / ICLR 2023。
- 对 MiniGen 而言，先能解释 **表示压缩、噪声预测、条件注入、采样** 各自负责什么，比急着堆叠新架构更重要。
- 本周的目标是理解、独立实现并验证局部模块，**不是**声称已经训练出完整高质量潜空间生成器。

<a id="q4"></a>

### Q4｜为什么正式训练 DiT 前要先检查 VAE 重建？

**直觉：** 如果压缩时图标的颜色、形状、数量或位置已经丢失，DiT 再强也无法可靠恢复未被保存的信息。

**诊断顺序：** 原图 $x$ → VAE 重建 $D(E(x))$（先检查信息保留）→ 扩散训练（再检查噪声预测）→ 生成质量评估（最后检查分布是否学到）。

**易错点：** 重建 MSE 低，不代表对象数量、颜色关系等任务属性一定正确；噪声 MSE 下降，也不等于生成质量一定提高。

<a id="q14"></a>

### Q14｜为什么第三周用空间潜变量？VAE、DDPM、DiT 的输入输出形状分别是什么？

**原始问题：**「讲解为什么这一周的 VAE 变成的是空间潜变量，DDPM 中的输入输出是什么样子？我忘了。然后具体 VAE 是怎么变的（b, 1, 28, 28）到（b, 3,128,128）怎么变的？」

**首先纠正关键误解：** `[B,1,28,28] → [B,3,128,128]` **不是 VAE 编码操作**，而是两周的数据与模型配置不同：第二周用 MNIST 灰度图和 MLP-VAE，第三周用 128×128 RGB 合成图标和预训练的卷积式空间 Autoencoder。若真的需要把灰度图改为 RGB 尺寸，可以插值放大再复制三个通道，但**不会凭空产生彩色语义**，也不是本周训练流程。

**第二周：** `[B,1,28,28]`（原始灰度图）`→ [B,784]`（展平）`→ [B,d_z]`（后验均值/采样，如 `d_z=16` 或 2），每张图是一条潜向量。

**第三周：** `[B,3,128,128]`（原始 RGB 图）`→` 卷积与多级下采样 `→ [B,4,16,16]`（后验均值/采样）`→ ×s`（固定数值缩放，形状不变）。例如三个 2 倍下采样阶段令空间边长 `128→64→32→16`，而卷积输出通道数由网络层决定；后验分布一般分别预测 `[B,4,16,16]` 的 `μ` 和 `logvar`（概念上的中间层示意，不代表实际预训练模型的精确层表）。空间特征图保留二维网格，便于后续切分 Patch Token。

**DDPM 在像素空间：** $x_0,x_t,\epsilon,\hat\epsilon$ 可均为 `[B,1,28,28]`（也可以是 RGB 图），用 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$；U-Net 通常输出**预测噪声** $\hat\epsilon$，而非直接输出最终成图。

**DDPM + DiT 在潜空间：** $z_0,z_t,\epsilon,\hat\epsilon$ 均为 `[B,4,16,16]`；DiT 内部示例 `patch_size=2` 时切为 `[B,64,16]`，映射到 `[B,64,D]`，最后投影并重排到 `[B,4,16,16]`，供噪声 MSE 与采样器使用。

**训练：** `RGB x → VAE Encoder → 原始 h → ×s → z0 → 加噪 zt → DiT → ε̂`。  
**生成：** `随机 zT → DiT + Scheduler 多步去噪 → z0估计 → ÷s → VAE Decoder → RGB 图片`。

**易错点：** `z` 是潜变量，不是维度数字；`d_z` 才是一维潜向量的长度。`[B,4,16,16]` 单样本有 1024 个数，但不是 1024 个通道。空间下采样会改变形状；乘 Scaling Factor 不会改变形状。VAE 的重建通过门槛不代表 DiT 已学会生成。

<a id="q15"></a>

### Q15｜把 VAE、DDPM、DiT 完整连接起来：训练与生成的数据流、形状及各模块职责

**原始问题：**「你把上面的『四、把 VAE、DDPM、DiT 完整连接起来』的知识都整理到 QA 中。」

**一句话总览：** **VAE 决定在哪个表示空间工作；DDPM 规定如何加噪及学习去噪；DiT 是负责预测噪声的 Transformer 网络。** 三者分别是*表示、生成过程、预测网络*，不是三个顺序训练的生成器。

#### 训练阶段：真实 RGB 图像 → 扩散噪声预测

```text
RGB 图像 x                    [B, 3, 128, 128]
    ↓ 冻结的 VAE Encoder（本项目使用后验均值 mode）
原始潜变量 h                   [B, 4, 16, 16]
    ↓ 乘固定 Scaling Factor s（此模型 s=0.13025）
潜变量 z₀                      [B, 4, 16, 16]
    ↓ 采样 t、ε ~ N(0,I)，按 DDPM 公式加噪
带噪潜变量 zₜ                   [B, 4, 16, 16]
    ↓ DiT 内部 Patchify + Embedding（如 patch_size=2）
Patch tokens                   [B, 64, D]（示例隐藏维 D）
    ↓ 时间条件 DiT Blocks + 输出头 + Unpatchify
预测噪声 ε̂                     [B, 4, 16, 16]
    ↓ 与真实噪声 ε 计算 MSE
只更新 DiT 参数；VAE 保持冻结
```

DDPM 加噪公式：

$$z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$

典型噪声预测训练损失：

$$\mathcal L_{\mathrm{noise}}=\mathbb E_{z_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(z_t,t)\|_2^2\right]$$

（实际实现通常对所有样本及张量元素求平均。）**不是**直接用 DiT 输出一张最终 RGB 图像。

#### 生成阶段：随机潜噪声 → RGB 图像

```text
标准高斯噪声 zT                [B, 4, 16, 16]
    ↓ DiT + Scheduler 重复多步预测/更新
去噪后的潜变量 ẑ₀              [B, 4, 16, 16]
    ↓ 除以同一个缩放系数 s
VAE 原始尺度潜变量 ĥ           [B, 4, 16, 16]
    ↓ 冻结的 VAE Decoder
生成 RGB 图像 x̂               [B, 3, 128, 128]
```

**三个角色：**

| 模块 | 输入 | 输出 | 职责 |
| --- | --- | --- | --- |
| 冻结 VAE Encoder | RGB 图像 `[B,3,128,128]` | 空间潜变量 `[B,4,16,16]` | 压缩表示、保留重要结构 |
| DDPM 加噪/采样器 | 潜变量与噪声/时间 | 同形状的带噪或更新潜变量 | 定义噪声水平与去噪更新规律 |
| DiT | 带噪潜变量 `z_t` 和时间 `t` | 同形状噪声估计 `ε̂` | 学习潜空间去噪规律 |
| 冻结 VAE Decoder | 逆缩放后的潜变量 | RGB 图像 | 还原到像素空间 |

**形状与易错点：**

1. `[B,4,16,16]` 中 `4` 是潜通道数，空间网格是 `16×16`；单样本共 `4×16×16=1024` 个元素。`D` 表示 DiT 内部 token 的隐藏宽度，不等于潜通道数。
2. 以 `patch_size=2` 为例：`[B,4,16,16] → [B,64,16] → [B,64,D]`；最终须投影、unpatchify 回 `[B,4,16,16]`，才能同 `ε` 比较。
3. 乘 `s` / 除 `s` 改变**数值尺度**，不改变张量形状；一般编码后缩放，最终解码前逆缩放，并非每次 DiT 迭代都逆缩放。
4. 训练中 VAE/DiT 分工明确：本周冻结 VAE；只有以后专门选择端到端训练方案时，才可能改变这一约定。
5. **重建通过 ≠ 生成通过。** VAE 的原图重建表明真实图像信息可以保留，但 DiT 仍需学习从噪声到合理潜变量的分布。

[返回目录](#toc)

---

<a id="chapter-02"></a>
## 第 2 章｜图像数据与 VAE｜后验、缩放、重建

**本章学习目标：** 第二周的 MNIST/MLP-VAE 与第三周的 RGB/空间 VAE 不是“同一批图像被拉伸”，而是不同数据与表示模型。

**整合后的三条主线：**

- RGB → 潜变量是学习的卷积编码，不是 reshape；潜变量 4 通道不等于 RGBA。
- 后验均值（mode）与随机采样（sample）不同；缩放系数改变数值尺度，不改变形状。
- VAE 重建先检查语义属性，再讨论扩散模型；像素误差只是其中一个视角。

<a id="q5"></a>

### Q5｜VAE 也可以应用到 RGB 图片吗？应该怎么做？

**原始问题：**「VAE也可以应用到rgb图片吗？应该怎么做？」

**可以。** 第二周将灰度图展平后送入 MLP VAE，只是教学实现；RGB 图像通常用卷积式 VAE / 空间 Autoencoder，保留高宽维度。数据先从 RGB 整数像素变为浮点，并按模型要求规范化（本周预训练组件使用 $[-1,1]$）。编码器通过卷积、下采样和通道投影得到后验均值与方差；可采样 $h=\mu+\sigma\odot\eta$，本周为保证诊断可复现，使用后验均值 $h=\mu$。解码器通过上采样/卷积重建图像。本周使用**冻结的预训练 VAE**，并不重新训练它。

**注意：** 潜变量的 4 通道是学习到的特征，不是 RGBA；使用前后的缩放系数必须配对：$z=sh$，解码前 $h=z/s$。

<a id="q6"></a>

### Q6｜图一的 Autoencoder 是什么？RGB 为什么会变成 [B,4,16,16]？

**原始问题：**「图一autoencoder是什么？为什么把rgb文件这样变形，以及怎么变的形？」

Autoencoder（自编码器）由 **Encoder 压缩 → latent 表示 → Decoder 重建**构成。普通 AE 输出确定的潜表示；VAE 还学习后验分布，并通常配有 KL 正则。本周“空间 Autoencoder”是预训练的图像表示组件。

$$[B,3,128,128]\xrightarrow{\mathrm{Encoder}}[B,4,16,16]$$

这里不是 `reshape`，而是经过学习的卷积和逐级下采样（空间边长 $128\to64\to32\to16$，例如三次 2 倍下采样），再投影到 4 个潜通道。其元素数 $3\cdot128^2=49152\to4\cdot16^2=1024$，约少 **48 倍**；高宽各缩小 **8 倍**。空间网格仍保留区域对应关系，适合 DiT 的 patch token 化。解码器学习近似逆变换；**这是有损压缩，不保证对象属性完整保存**。

<a id="q7"></a>

### Q7｜图二的数据是什么？是否类似 VLM？各数据集划分是什么意思？

**原始问题：**「图二在干什么？没看懂讲的是什么。也就是图片数据+标签描述的文本是吗？类似vlm吗（只不过是输出类生成模型）。图二的表格各行都说的什么？」

**是“图像 + 结构化真值 + 文本描述 + 元数据”的受控合成数据集。** 每张图是白底的彩色圆形/方形/三角形（1–3 个对象）；程序先决定对象、颜色、数量和相对关系，再渲染 PNG 并由真值生成准确的描述，而非让 AI 猜标签。文件还记录 ID、路径、哈希、场景组和数据划分，方便复现并防止近似样本跨集合泄漏。

| 划分 | 数量 | 用途 |
|---|---:|---|
| `train` | 10,000 | 以后用于训练生成模型 |
| `validation` | 1,000 | 本周诊断 VAE 表示质量，后续可选超参数 |
| `test_id` | 1,000 | 选型冻结后，测试同分布新样本 |
| `test_combo` | 1,000 | 测试训练未见的**颜色+形状组合**（如红三角、蓝方形） |
| `test_template` | 1,000 | 视觉组合已见，但描述措辞/模板未见；留待后续文本条件泛化测试 |

**与 VLM 的异同：** 形式上都可包含图文对；VLM 常学习“读图→理解/回答”，未来的文本条件生成模型则学习“读描述→生成图”。但**本周尚未训练文本模型或文本条件 DiT**；文本描述主要作为可靠标注与后续实验准备，当前核心任务是 VAE 重建和 DiT 的潜空间噪声预测。

<a id="q8"></a>

### Q8｜VAE 的 φ、qφ(h|x)、μφ(x)、σφ(x) 是什么？posterior.sample() / mode() 有什么区别？

**原始问题：**「图一这里我没看懂，phi是什么？这个公式表达什么，各个符号都代表什么？mu_phi(x)和sigma_phi(x)和正常的mu、sigma的区别？posterior是个什么变量、函数，为啥会有.sample()和.mode()方法？以及什么是sample？是重参数化的采样还是直接采样？mode模式就是直接取得均值项，不采样是吗？」

**直觉：** Encoder 不是直接给一个确定坐标，而是根据输入图像 $x$ 预测一个潜变量分布的参数。

$$q_\phi(h\mid x)=\mathcal N\left(h;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right)$$

- $\phi$：**编码器的全部可学习权重**；$x$：图像；$h$：原始潜变量；$q_\phi(h\mid x)$：编码器构造的**近似后验分布**，不是真实后验的解析解。
- $\mu_\phi(x)$ / $\sigma_\phi(x)$：由网络根据 $x$ 算出来的均值/标准差（通常是多维张量）；与一般概率论的 $\mu,\sigma$ 在意义上相同，下标 $\phi$ 和自变量 $x$ 只是说明来源。
- $\operatorname{diag}(\sigma^2)$：用各维方差组成对角协方差矩阵，假设这个近似后验在给定 $x$ 时各维条件独立。
- `posterior`：某种封装了均值、方差和采样方法的**概率分布对象**（具体 API 取决于 VAE 实现），不是一个标量，也不是名称固定的数学函数。
- `posterior.sample()`：从该分布产生随机 $h$；本项目这类 VAE 可内部使用**重参数化** $h=\mu+\sigma\odot\eta,\ \eta\sim\mathcal N(0,I)$。注意 PyTorch 原生 `Normal.sample()` 与 `.rsample()` 不同，不能仅由函数名推断可微性。
- `posterior.mode()`：取众数；**高斯分布的众数 = 均值**，所以 $h=\mu_\phi(x)$，不注入额外的采样噪声。第三周为稳定缓存和对照选择 `.mode()`。

**易错点：** 固定后验均值不等于关闭 DDPM 加噪；DiT 训练仍会在 $z_0$ 上采样时间 $t$ 和高斯噪声 $\epsilon$。

<a id="q9"></a>

### Q9｜为什么 $z=sh$？s 从哪来、是否训练？何处缩放/逆缩放？与 downsampling 有何区别？

**原始问题：**「z=sh中h为什么要缩放？缩放的系数怎么确定？这个系数需要训练吗？缩放最后要达成的效果是什么？在ddpm这一块的前后（也就是encoder后面紧接着，和decoder前面紧接着）都要用s进行缩放/逆缩放是吗？什么是downsampling，为什么你要和scaling进行对比？」

**核心：** 预训练 VAE 的原始潜变量 $h$ 的总体数值方差，不一定与 DDPM 加入的单位高斯噪声相匹配；使用固定缩放系数 $s$，令 $z_0=sh$，通常希望训练数据的潜变量**整体标准差接近 1**，使预定噪声调度的 SNR 尺度更可控：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I)$$

$$\operatorname{SNR}(t)\approx\frac{\bar\alpha_t}{1-\bar\alpha_t}\quad\text{（当 $\operatorname{Var}(z_0)\approx1$ 时）}$$

- 一种确定 $s$ 的方式：在代表性图像上统计编码后潜变量的总体标准差 $\sigma_h$，取 $s\approx1/\sigma_h$；实际项目**直接读取预训练模型的配置**，本周为 `0.13025`。一般不是额外训练的参数，也不能任意替换为别的 VAE 系数。
- **编码/训练入口：** $x\to E(x)=h\to z_0=sh\to\text{DDPM 加噪与 DiT}$。
- **生成/解码出口：** $\hat z_0\to \hat h=\hat z_0/s\to D(\hat h)$。中途每一次去噪步骤不需要反复逆缩放。
- **Scaling** 乘常数、只改变数值，不改变张量形状；**Downsampling** 下采样、降低空间分辨率，如 $128\times128\to16\times16$。本周空间编码器两种变化都有，但不是一回事。

**易错点：** 整体标准差接近 1 **不代表**所有维度都严格满足 $\mathcal N(0,1)$，也不代表均值恰好为 0；缩放约定必须随权重配置一致地贯穿训练、缓存与生成。

<a id="q10"></a>

### Q10｜为什么本周 VAE 输入用 [-1,1]，第二周却使用 [0,1]？

**原始问题：**「VAE的输入约定为什么是[-1, 1]？在第二周的课程中哪里有说到VAE的输入是这样确定的？」

**结论：** **不是 VAE 数学定义要求必须使用某个范围**，而是具体 VAE 的训练和预处理约定。

| 数据范围 | 典型变换 | 常见约定 |
|---|---|---|
| $[0,1]$ | `RGB/255` | 第二周教学版使用 Sigmoid 输出 / BCE 重建目标，要求目标像素在 $[0,1]$ |
| $[-1,1]$ | `RGB/127.5 - 1` | 第三周加载的预训练 SDXL VAE 所要求的输入预处理 |

**如何在第二周定位？** 检查 `ToTensor()` / `.float()/255` 等数据预处理、Decoder 的 `sigmoid` 输出及 BCE 损失，而非寻找“VAE 必须输入 [0,1]”的普遍定理。当前对话没有第二周原讲义的确切行号，所以不能指定具体页码或代码行。

<a id="q13"></a>

### Q13｜Pixel Difference 怎么计算？为什么误差往往集中在图形边缘？

**原始问题：**「为我整合图中的信息这一块的知识（独立于讲义为我整理，需要的背景和知识简述就好）」

**直觉：** 比较原图和 VAE 重建图在**同一个像素位置**的 RGB 数值。大片平坦背景较易重建；锐利边缘、细线等高频细节易产生轻微平滑或位置偏移，所以误差图往往呈现轮廓亮线。

设原图与重建图为 $x,\hat x\in\mathbb R^{H\times W\times3}$，逐位置像素差异为：

$$E_{ij}=\frac13\sum_{c=1}^{3}|x_{ijc}-\hat x_{ijc}|.$$

实现：先把像素转为浮点数，再逐元素取绝对值，最后 `mean(-1)` 在 RGB 通道维求平均。`E` 的形状是 `[H,W]`，可用 `imshow(E,cmap="magma")` 绘制热力图。此处是**每像素 RGB 平均绝对误差**，不是平方误差 MSE。使用固定的 `vmin/vmax` 才更便于跨图片比较。

**质量要分开：** 像素准确（MAE/MSE/PSNR）与结构属性正确（对象数量、颜色、形状、空间关系）是两回事。边缘误差不必然表示 VAE 失败；小对象消失或关系错误则属于不可忽略的结构损失。热力图能定位误差，规则解析器能检查受控图标属性。

#### 本章补充｜VAE 表示门槛与“是否生成成功”不可混为一谈

讲义在正式潜变量试点前设定了**重建表示质量**压力检查：96 张固定验证图（1/2/3 个对象各 32 张）；预设门槛包括原图解析全部正确、重建颜色与形状准确率至少 98%、数量／关系／完整属性同时正确至少 95%。这些是**事先设定的检查标准**，不是“已实现生成的准确率”；验证子集带压力取样性质，不代表总体分布的无偏评估。

[返回目录](#toc)

---

<a id="chapter-03"></a>
## 第 3 章｜复现与工程边界｜冻结、缓存、内存

**本章学习目标：** 这一章回答一个工程问题：为什么“能运行一次”不等于“可复现、可比较”。

**整合后的三条主线：**

- train/eval、requires_grad 和 no_grad 是三组不同控制开关。
- 冻结的 VAE 可离线缓存潜变量，必须记录模型、scale、数据身份等信息。
- 缓存加快反复训练，但不能承诺占用磁盘一定比 PNG 小。

<a id="q11"></a>

### Q11｜train/eval、requires_grad_(True/False)、enable_grad/no_grad 如何区别与组合？

**原始问题：**「eval()与train，requires_grad_(False)和true，torch.no_grad()和enable_grad，为我用简单的表格对比这三个维度，并告诉我什么时候用这三者或者其组合」

| 独立维度 | 开启 | 另一状态 | 控制对象 |
|---|---|---|---|
| 模型运行模式 | `model.train()` | `model.eval()` | Dropout、BatchNorm 等层的行为；**不**直接决定能否反传 |
| 参数梯度需求 | `model.requires_grad_(True)` | `False` | 是否要求为模型参数累计梯度；冻结参数不一定阻止梯度传回输入 |
| 自动求导记录 | `torch.enable_grad()` | `torch.no_grad()` | 当前运算是否记录计算图；`enable_grad()` 不会自动解冻参数 |

**常见组合：**

| 场景 | 模式 | 参数 | 求导记录 |
|---|---|---|---|
| 正常训练 DiT | `train()` | DiT True | 开启 |
| 验证 | `eval()` | 可保持 True | `no_grad()` |
| 冻结 VAE、预计算潜变量 | `eval()` | VAE False | `no_grad()` |
| 冻结 VAE + 训练 DiT | VAE eval、DiT train | VAE False、DiT True | 编码时 `no_grad()`、DiT 前反传时开启 |
| LoRA | 常用 train（具体层需核对） | 基座 False、适配器 True | 开启 |
| 冻结解码器但优化其输入 | eval | 解码器 False，输入 True | **开启**，以便梯度流向输入 |

**易错点：** `eval()` ≠ `no_grad()` ≠ `requires_grad_(False)`；`no_grad()` 内即便参数 `requires_grad=True` 也通常不记录运算图。对纯推理也可考虑 `torch.inference_mode()`，但它与 `no_grad()` 不是完全可互换的。

<a id="q12"></a>

### Q12｜1B 是否等于十亿个 FP32 参数、约 4GB？

**原始问题：**「1B参数的意思就是有十亿个参数是吗，也就是有10亿个4byte参数（float32的情况）」

**是。** $1B=10^9$ 个参数；FP32 每参数 $32\ \mathrm{bit}=4\ \mathrm{bytes}$，权重本体约 $4\times10^9$ bytes = **4GB（十进制）约 3.73GiB（二进制）**。常见权重估算：FP16/BF16 约 2GB，INT8 理想约 1GB，INT4 理想约 0.5GB（量化常有元数据开销）。

**不要与训练显存混淆：** FP32 AdamW 的权重、梯度、一阶动量、二阶动量通常合计至少 $4+4+4+4=16$ bytes/参数，1B 约 **16GB**，还没有算激活、暂存、通信等额外开销；混合精度和不同优化器实现还可能引入其他常驻状态。

<a id="q16"></a>

### Q16｜图 3：为什么冻结 VAE 后需要缓存潜变量？

**原始问题：**「图 3 的缓存这里的知识点内容保存到 QA。」

**核心直觉：同一张图像的确定性编码只需要计算一次。** 本项目冻结 VAE 参数，并使用近似后验的众数（高斯时为均值）作为编码结果，因此在图像、权重、输入预处理、后验策略与缩放系数不变的情况下：

$$z_i=s\mu_\phi(x_i)$$

对同一张图像，每个 Epoch 得到的干净潜变量 $z_i$ 相同。若每轮 DiT 训练都执行 VAE Encoder，会白白重复昂贵的编码计算。故先**离线编码并保存到磁盘**，训练时直接读取。

- **省下的是**：反复执行冻结 Encoder 的时间和算力；不意味着模型训练不再需要新的噪声。
- **仍需随机抽样的是**：每次 DDPM 训练的时间步 $t$ 与高斯噪声 $\epsilon$，并据此生成不同的 $z_t$。
- **适用前提**：输入和编码行为稳定；若 Encoder 继续训练、每次图像做随机增强、或编码采用新的随机后验采样，则需改变缓存策略，不宜机械复用固定 $z_i$。

<a id="q17"></a>

### Q17｜图 4：离线缓存如何改变训练流程？为什么还要验证缓存身份？

**原始问题：**「图 4 的缓存流程与工程注意点单独保存到 QA。」

**不缓存时（每个 Epoch 重复）：**

```text
读取 RGB 图片 x_i → VAE Encoder → ×s 得 z_i → 采样 t、ε 构造 z_t → DiT 预测 ε̂ → MSE / 更新 DiT
```

**缓存后（分为两个阶段）：**

```text
预计算一次：RGB 图片 x_i → 冻结 VAE Encoder → ×s 得 z_i → 写入 .npy 缓存
每轮训练：  读取缓存 z_i → 重新采样 t、ε 构造 z_t → DiT 预测 ε̂ → MSE / 更新 DiT
```

缓存并非单纯依靠文件名，因为“同名图像被替换、同名模型权重升级、Scaling Factor 更换、预处理变化、`mode()` 改为 `sample()`”都会导致旧缓存可能不再代表当前实验。

**缓存身份 / 失效管理**可记录：数据清单指纹与样本哈希、VAE 权重/版本、预处理规则、后验编码策略、Scaling Factor、dtype 等；身份不匹配应拒绝直接复用旧缓存。

**本周测试**仅对 256 张训练图像建立 pilot cache：编码、保存、重载一致性比较、比较单张与批量编码、故意修改身份字段确认拒绝不匹配缓存。**256 张试验缓存不等于 14,000 张已全量缓存。**

<a id="q18"></a>

### Q18｜图 5：为什么潜变量元素更少，磁盘文件反而可能比 PNG 更大？

**原始问题：**「图 5 的 PNG 与潜变量文件大小及缓存成本单独保存到 QA。」

每张空间潜变量 `[4,16,16]` 共 $4\times16\times16=1024$ 个元素，使用 FP32（每个 4 bytes）时：

$$1024\times4=4096\text{ bytes}=4\text{ KiB / 张}.$$

| 数据规模 | FP32 潜变量原始数值负载 |
|---|---:|
| 1 张 | 4 KiB |
| 256 张 | 1 MiB |
| 14,000 张 | 54.69 MiB |

讲义实际 14,000 张 PNG 图像合计约 **6.91 MiB**，反而比 FP32 潜变量数据小得多。

**原因：** 白底、少量纯色形状有大量重复像素，PNG 的无损压缩非常高效；`.npy` 保存的潜变量则是通常未额外压缩的浮点数组，还会有少量文件头等开销。

**必须区分两件事：**

- “压缩表示”是 `[3,128,128] → [4,16,16]`，**元素数量减少约 48 倍**；
- “压缩文件”是磁盘字节数更少，取决于数据类型、压缩格式、内容重复性等，**前者不保证后者**。

因此缓存的核心是**以存储空间换取计算时间**，而不是保证减少磁盘占用。改用 FP16/压缩文件会改变容量，但也需要核验精度和可复现性。

[返回目录](#toc)

---

<a id="chapter-04"></a>
## 第 4 章｜潜空间 DDPM｜公式、SNR 与 Noise MSE

**本章学习目标：** 扩散计算只换了工作空间，前向噪声方程和 MSE 训练目标依然成立。

**整合后的三条主线：**

- 噪声张量与潜变量同形，不是 RGB 图像。
- 按通道方差定义的有效 SNR 受潜变量实际统计影响。
- Noise MSE 对 B、C、H、W 的元素求平均；最优预测是给定 (z_t,t) 的条件均值噪声。

<a id="q19"></a>

### Q19｜DDPM 从像素空间迁移到潜空间后，什么变了、什么没变？为什么 MSE 预测的是条件平均噪声？

**原始问题（本轮讲义导读）：**「把 DDPM 迁移到潜空间，哪些变了、哪些没变？训练目标为什么依然是噪声预测？同一个带噪输入不可能知道每次随机噪声的唯一来源，模型到底学到什么？」

- **不变**：扩散时间步、$\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{i=1}^t\alpha_i$、高斯加噪公式、噪声预测 MSE 的训练逻辑。DDPM 的概率过程不依赖于输入一定是像素图。
- **改变**：先把 RGB 图像 `[B,3,128,128]` 经冻结 VAE 编码并缩放为 $z_0$ `[B,4,16,16]`；此后对潜变量而非 RGB 像素加噪。网络输出是**4 通道潜噪声**，不是 RGB、干净图像或 Flow 速度。

$$z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

$$\mathcal L(\theta)=\mathbb E_{z_0,t,\epsilon}\left[\frac{1}{CHW}\left\|\epsilon_\theta(z_t,t)-\epsilon\right\|_2^2\right],\quad C=4,\ H=W=16.$$

- 每次训练会重新抽样 $t$ 和 $\epsilon$。给定同一个 $(z_t,t)$，可能存在多组 $(z_0,\epsilon)$，不能唯一“读出”本次真实噪声。
- 对固定输入 $v=(z_t,t)$，平方误差的最优预测满足：

$$\mathbb E[\|a-\epsilon\|^2\mid v]=\|a\|^2-2a^\top\mathbb E[\epsilon\mid v]+\mathbb E[\|\epsilon\|^2\mid v].$$

对 $a$ 求导并令零，得到 $\boxed{a^*=\mathbb E[\epsilon\mid z_t,t]}$，即**条件平均噪声**，不是记忆随机数。

- **数值尺度**：若某潜通道干净信号方差 $v_z$，且信号与单位高斯噪声独立：

$$\operatorname{Var}(z_t)=\bar\alpha_t v_z+(1-\bar\alpha_t),\qquad \mathbb E[z_t]=\sqrt{\bar\alpha_t}\,\mathbb E[z_0].$$

通道方差影响有效 SNR；均值非零时也会随扩散衰减。Scaling 并不保证每通道恰好 $N(0,1)$。

- **手算核验**：$z_0=2,\epsilon=-1,\bar\alpha_t=0.64$，则 $z_t=0.8\times2+0.6\times(-1)=1$；**只有事先知道真实噪声**时，才可用 $(z_t-0.6\epsilon)/0.8=2$ 精确恢复。这只是代数验证，不是已训练的生成模型。

<a id="q22"></a>

### Q22｜潜空间 DDPM 的有效信噪比 SNR 如何从加噪公式推导？

**原始问题：**「图1的有效信噪比怎么推导出来的？」

对潜变量通道 $c$，记干净潜变量的**中心化方差**为 $\operatorname{Var}(z_{0,c})=\sigma_c^2$。DDPM 前向过程：

$$z_{t,c}=\underbrace{\sqrt{\bar\alpha_t}z_{0,c}}_{\text{信号}}+\underbrace{\sqrt{1-\bar\alpha_t}\epsilon_c}_{\text{噪声}},\qquad \epsilon_c\sim\mathcal N(0,1).$$

依据 $\operatorname{Var}(aX)=a^2\operatorname{Var}(X)$，并假设 $z_0$ 与 $\epsilon$ 独立：

$$\operatorname{Var}(\text{信号})=\bar\alpha_t\sigma_c^2,\qquad \operatorname{Var}(\text{噪声})=1-\bar\alpha_t.$$

以**中心化信号方差 / 噪声方差**定义 SNR，得到

$$\boxed{\mathrm{SNR}_c(t)=\frac{\bar\alpha_t\sigma_c^2}{1-\bar\alpha_t}}.$$

**手算**：若 $\bar\alpha_t=0.64$，$\sigma_1=1$、$\sigma_2=0.5$，则 $\mathrm{SNR}_1=0.64/0.36\approx1.78$，$\mathrm{SNR}_2=0.64\times0.25/0.36\approx0.44$。同一个时间步，不同潜通道可能有不同有效信噪比。

**易错点**：此式比较的是**中心化方差**；如果信号均值非零且使用总功率 $\mathbb E[z_0^2]$ 定义 SNR，分子还涉及均值平方。对各通道应用同一个 Scaling Factor，并不能自动保证每个通道都均值为零、方差为一。

<a id="q24"></a>

### Q24｜为什么 Noise MSE 要除以 CHW？张量怎么计算 L2 范数？期望下标的 $z_0,t,\epsilon$ 又是什么意思？

**原始问题：**「图2为什么要对CHW都进行平均呢（我真的完全不理解这里）？eps_theta得到的既然是张量矩阵，又怎么进行的L2范数？E右下角的z0，t，eps又代表什么呢？」

本周的噪声预测目标是：

$$\boxed{\mathcal L(\theta)=\mathbb E_{z_0,t,\epsilon}\left[\frac1{CHW}\|\epsilon_\theta(z_t,t)-\epsilon\|_2^2\right]}.$$

**第一层：L2 范数平方。** 对单张样本，$\hat\epsilon,\epsilon\in\mathbb R^{C\times H\times W}$，把张量展平为长向量，则

$$\|\hat\epsilon-\epsilon\|_2^2=\sum_{c=1}^C\sum_{h=1}^H\sum_{w=1}^W(\hat\epsilon_{c,h,w}-\epsilon_{c,h,w})^2.$$

**第二层：除以 CHW。** 上式是误差平方的 **Sum**，$CHW$ 是单个样本的元素数，除以它就得到**逐元素平均平方误差**。本项目 $CHW=4\times16\times16=1024$。

**手算**：若一张 $1\times2\times2$ 的样本误差为 $[1,-2,0,1]$，平方和为 $1+4+0+1=6$，所以 MSE 为 $6/4=1.5$。**不除也能构造目标，但 Sum 会随元素数变大并改变梯度尺度**。

**第三层：期望下标** $\mathbb E_{z_0,t,\epsilon}$ 指训练同时对三种随机来源取平均：$z_0$ 来自编码后的训练数据分布，$t$ 来自随机抽取的时间步，$\epsilon\sim\mathcal N(0,I)$ 是每次重新抽的独立高斯噪声。该期望是理论总体目标，实际用 Mini-batch 随机抽样近似。

因此 Batch 维度 $B$ 没写在单样本的 $1/CHW$ 中，实际批量实现一般为

$$\mathcal L_{\mathrm{batch}}=\frac1{BCHW}\sum_{b,c,h,w}(\hat\epsilon_{b,c,h,w}-\epsilon_{b,c,h,w})^2.$$

```python
# eps_pred, eps: [B,4,16,16]
loss = torch.nn.functional.mse_loss(eps_pred, eps)  # 默认 reduction='mean'
# 等价于 ((eps_pred - eps) ** 2).mean()
```

#### 对应代码示例：潜空间 DDPM 的单次训练前向过程

下面这张图对应本题的 MSE 公式，把 $z_0$、$t$、$\epsilon$ 的三种随机来源和网络预测连接起来：

![潜空间 DDPM 单次训练的 6 个步骤](MiniGen_QA_assets/DDPM_训练流程_NoiseMSE.png)

**图中的 6 步**：从数据加载 $z_0$ → 为 Batch 中每个样本随机抽取 $t$ → 生成相同形状的高斯噪声 $\epsilon$ → 按 $\bar\alpha_t$ 加噪获得 $z_t$ → DiT 预测 $\hat\epsilon$ → 对预测与真实噪声的平方差取平均。

```python
# 假设已定义 T、dit、sqrt_alpha_bar、sqrt_one_minus_alpha_bar
z0 = batch_latents                          # [B, 4, 16, 16]
B = z0.shape[0]
t = torch.randint(0, T, (B,), device=z0.device)  # [B]
eps = torch.randn_like(z0)                   # [B, 4, 16, 16]

# 截图是概念写法；真正运行需把每个样本的时间系数广播到 CHW 维度
a = sqrt_alpha_bar[t].view(B, 1, 1, 1)
b = sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
zt = a * z0 + b * eps                       # [B, 4, 16, 16]
eps_pred = dit(zt, t)                        # [B, 4, 16, 16]
loss = ((eps_pred - eps) ** 2).mean()        # 标量，平均 B*C*H*W 个误差平方
```

**形状与易错点：** 时间 $t$ 只有 `[B]`，而 $z_0$ 是 `[B,C,H,W]`；`sqrt_alpha_bar[t]` 需经 `view(B,1,1,1)` 才能按样本正确广播。截图中的 `sqrt_alpha_bar[t] * z0` 没有展示这一步，直接复制通常会遇到维度不匹配。`torch.randn_like(z0)` 生成形状与 $z_0$ 相同的噪声；`mean()` 默认对 Batch、通道、高、宽的所有元素求平均。这里是**训练时的一次加噪、预测与 Loss 计算**，不是生成阶段的完整多步反向采样。

**易错点**：$\|\cdot\|_2^2$ 对单样本**所有非 Batch 维度**求平方和，$1/CHW$ 将其转为均值；$\mathbb E_{z_0,t,\epsilon}$ 指对样本/时间/噪声的随机性求平均，而不是额外对“矩阵三个轴”再求一次期望。

[返回目录](#toc)

---

<a id="chapter-05"></a>
## 第 5 章｜Patch 与 Embedding｜二维空间信息如何进入 Transformer

**本章学习目标：** 先锁定 Patch 的索引，再理解空间位置与扩散时间这两条完全不同的信息通道。

**整合后的三条主线：**

- Patchify 是无参数且可逆的重排；Patch Projection 是可学习的映射。
- patch_size 决定 token 数及 Attention 代价；行列位置编码对应 Patch 网格而不是原图像素网格。
- position_2d 的推导仍被明确标记为待补学，已有讲解不等于学会。

<a id="q20"></a>

### Q20｜空间潜变量怎么 Patchify 成 Transformer Token？索引顺序和 Unpatchify 怎么验证？

**原始问题（本轮讲义导读）：**「从二维潜变量到 token 怎么转？索引公式是什么意思？切分为何可逆？线性投影也可逆吗？」

输入 $z\in\mathbb R^{B\times C\times H\times W}$，若 Patch 边长为 $p$ 且整除高宽：

$$H_p=H/p,\quad W_p=W/p,\quad N=H_pW_p,\quad K=p^2C.$$

对本项目 `[B,4,16,16]`、$p=2$：$N=8\times8=64$，$K=2\times2\times4=16$，因此：

```text
[B,4,16,16] --patchify--> [B,64,16] --Linear(16,256)--> [B,64,256]
```

- **Patchify 是无参数的数据重排**；线性投影 $W_e\in\mathbb R^{K\times D}$ 是可学习特征变换。后者一般不要求可逆。
- 讲义的顺序：Patch 从上到下、从左到右；Patch 内按行、列、通道顺序展平。以 Patch 网格行列 $(r,s)$、内部行列 $(i,j)$、通道 $c$ 表示：

$$n=rW_p+s,\qquad k=(ip+j)C+c,\qquad \mathrm{tokens}[b,n,k]=z[b,c,rp+i,sp+j].$$

例如 $p=2,C=4,W_p=8,r=2,s=3,i=1,j=0,c=2$，则 $n=19,k=10$，对应 $\mathrm{tokens}[b,19,10]=z[b,2,5,6]$。此处符号 $s$ 是 Patch 列号，**不是**潜变量 Scaling Factor。
- **Unpatchify** 将预测出的 `[B,64,16]` 噪声 Patch 按同一映射放回 `[B,4,16,16]`；检验 $\mathrm{unpatchify}(\mathrm{patchify}(z))=z$。
- **工程易错点**：两个函数用同一套错误排列也可能相互抵消、往返测试仍通过；必须独立人工检查某些具体索引的对应关系。

<a id="q21"></a>

### Q21｜Patch Size 怎样影响 DiT 计算量？位置编码与时间编码为什么不能互相替代？

**原始问题（本轮讲义导读）：**「Patch 大小怎么影响 token 和注意力矩阵？近似计算量公式从哪来？二维位置和扩散时间分别表示什么？」

对于 `[B,4,16,16]` 且 DiT 隐藏宽度 $D=256$，记 $N=(16/p)^2$、原始 Patch 宽度 $K=p^2C$：

| Patch Size $p$ | $N$ (tokens) | $K$ (原始 Patch 宽度) | 每个头注意力矩阵元素 $N^2$ |
|---|---:|---:|---:|
| 1 | 256 | 4 | 65,536 |
| 2 | 64 | 16 | 4,096 |
| 4 | 16 | 64 | 256 |

把 $p$ 从 2 改为 1，$N$ 增至 **4 倍**，$N^2$ 增至 **16 倍**；但**整块 Transformer** 的计算不能简单说成 16 倍。

令 MLP 扩张比为 $r$，忽略 LayerNorm、激活、Embedding 等较小算子，一个 Transformer Block 的乘加次数（MAC）近似：

$$\boxed{\mathrm{MAC}\approx (4+2r)ND^2+2N^2D}.$$

- $(4+2r)ND^2$：Q/K/V 和输出投影约 $4ND^2$，两层 MLP 约 $2rND^2$。
- $2N^2D$：$QK^\top$ 与注意力加权 $AV$。
- 若 $D=256,r=4$，$p=1,2,4$ 对应近似 **234.88、52.43、12.71 MMAC / Block**，是数量级估算，不等于硬件真实耗时，且不包含 Patch 投影、卷积等完整模型开销。

**位置编码（Position Encoding）**：让模型识别 Token 在二维潜网格的行、列位置，否则仅凭无位置标记的自注意力，无法区分空间排列。讲义用**行/列正弦余弦特征拼接**并与 Token Embedding 相加：通常形状 `[B,N,D]`（可广播）。

**时间编码（Timestep Embedding）**：表示当前扩散噪声级别 $t$，决定相同 $z_t$ 在不同时间应该如何解释。讲义将离散 $t\in\{0,\dots,999\}$ 先转换为多频率正余弦特征，再通过 MLP 得到 $c\in\mathbb R^{B\times D}$，供 DiT Block 的条件调制使用。

- 两者都是正余弦特征，但**位置表示空间坐标，时间表示噪声步数**，意义与注入路径不同，不能互相替代。
- 代码中标记的 $t=0$ 是论文约定的首次加噪状态；实际离散索引起点应与 Scheduler 和训练实现保持一致。

<a id="q23"></a>

### Q23｜DiT 的 Patch Embedding 是否像 ViT？用 T 个卷积核映射 T 个 Token，还是用 Linear？

**原始问题：**「DiT的切patch、做映射，其实就类似于vision encoder的切分patch然后用T个卷积核分别映射成T个视觉token是吗？感觉很类似。dit这里用的是卷积进行映射还是别的呢？」

**结论**：DiT 与 ViT 的 **Patch → Token Embedding** 思路相同，但“**几个卷积核 = 几个 Token**”不准确。Token 数由空间 Patch **数量**决定，映射后的维度 $D$ 由 Linear 输出宽度或 Conv2d 的 `out_channels` 决定。

对 `[B,4,16,16]` 且 Patch Size $p=2$：

$$N=(16/2)^2=64,\qquad K=p^2C=2^2\times4=16.$$

本周 Notebook 的 **TeachingDiT 实际采用显式 `patchify_reference` 加 `nn.Linear(K,D)`**：

```python
patches = patchify_reference(z_t, 2)  # [B,64,16]
tokens = nn.Linear(16, 256)(patches)   # [B,64,256]
```

还可以等价实现成特定步幅卷积：`nn.Conv2d(4, 256, kernel_size=2, stride=2)` 先输出 `[B,256,8,8]`，再 `flatten(2).transpose(1,2)` 为 `[B,64,256]`。当 `kernel_size=stride=p` 且不重叠、权重顺序与偏置匹配时，二者可表示相同的逐 Patch 仿射映射。

| 维度 | ViT / 视觉编码器 | 本周 DiT |
|---|---|---|
| 输入 | RGB 图像 Patch | 带噪潜变量 Patch |
| Token 化 | Patch Embedding | Patch Embedding |
| 训练目标 | 视觉特征/理解任务 | 预测潜空间噪声 $\epsilon$ |
| 额外条件 | 视架构而定 | 扩散时间步 $t$（后续 adaLN 等） |

**易错点**：Patchify 本身只是重排，无参数；后续 Linear / Conv2d 投影才可学习。一个 Token 由一个 Patch 映射而成，并非一个卷积核对应一个 Token。

<a id="q32"></a>

### Q32｜为什么要修改 `patchify` 的元素顺序？

**原始问题：**「我写的 `einops.rearrange` 已经把 Patch 展平了，为什么还要修改最后一维的元素顺序？」

你原来的写法是：

```python
"b c (p_h h) (p_w w) -> b (p_h p_w) (c h w)"
```

表示每个 Patch 内部按 **通道 → 行 → 列** 展平。

而本周讲义的 `patchify_reference()` 约定是 **行 → 列 → 通道**，所以改成：

```python
"b c (p_h h) (p_w w) -> b (p_h p_w) (h w c)"
```

例如，一个双通道、$2\times2$ 的 Patch：

```text
Channel 0:      Channel 1:
 0   1           8   9
 4   5          12  13
```

- 你的顺序 `(c h w)`：`[0, 1, 4, 5, 8, 9, 12, 13]`
- 讲义顺序 `(h w c)`：`[0, 8, 1, 9, 4, 12, 5, 13]`

**结论：** 两种顺序都能可逆重排；要修改不是因为原写法在数学上错误，而是为了与本周教学参考实现的索引约定一致。

<a id="q33"></a>

### Q33｜`sincos_1d()` 和 `position_2d()` 在 DiT 中分别做什么？数学原理是什么？

**原始问题：**「`position_2d()` 和 `sincos_1d()` 到底是在干什么？我不太懂这个的原理，为我详细拆解介绍这个架构。」

**核心区别：** `sincos_1d(values, width)` 是**通用的数值→多频率向量编码器**，可以编码一维时间步，也可以编码一维空间坐标；`position_2d(height,width,hidden)` 用它分别编码二维 Patch 网格的**行、列**，再拼接成固定的位置向量。两者都不是可学习的神经网络层。

#### 1. 一维 SinCos 的公式、频率和张量形状

设输出宽度 $D=2m$，对输入标量 $v$ 定义 $m$ 个几何级数频率：

$$\omega_k=\exp\left(-\frac{\log(10000)}m k\right)=10000^{-k/m},\quad k=0,\ldots,m-1.$$

定义编码（本讲义排列：先所有 cos，再所有 sin）：

$$\operatorname{PE}_{1D}(v)=[\cos(v\omega_0),\ldots,\cos(v\omega_{m-1}),\sin(v\omega_0),\ldots,\sin(v\omega_{m-1})]\in\mathbb R^D.$$

数值例：$D=8,m=4$，频率 $[1,0.1,0.01,0.001]$；输入 $v=100$，角度 $[100,10,1,0.1]$，对这些角度分别计算余弦、正弦并拼成 8 维向量。不同频率让编码同时包含变化快、变化慢的周期模式，不保证任意范围绝对唯一。

**形状追踪：** `values: [B]` → `unsqueeze(-1): [B,1]`；`frequencies: [D/2]`；广播乘得 `angles: [B,D/2]`；`cat(cos, sin): [B,D]`。此函数**没有可学习参数**；当输入时间步时，后面再接可学习 `Time MLP`，得到条件向量 $c(t)\in\mathbb R^{B\times D}$。

#### 2. 二维位置编码与 Patch Token 对齐

对于 Patch 网格 $H_p\times W_p$ 的位置 $(r,s)$，令隐藏维度 $D$ 可被 4 整除，分别计算半宽的一维编码：

$$P_{r,s}=[\operatorname{PE}_{1D}(r;D/2);\operatorname{PE}_{1D}(s;D/2)]\in\mathbb R^D.$$

实现顺序：`meshgrid(arange(H_p), arange(W_p), indexing="ij")` → 行、列矩阵分别 `flatten()`（**先行后列**）→ 各调用 `sincos_1d(..., hidden//2)` → `cat(dim=-1)` → `unsqueeze(0)`。

**本项目形状：** VAE 潜变量 `[B,4,16,16]`，`patch=2` → `64` 个 Patch，经线性投影得 `[B,64,256]`；位置编码 `position_2d(8,8,256)` 得 `[1,64,256]`，通过广播相加：$X'=X+P$。每个 Patch 的行列坐标需与 `patchify` 的 Token 排序一致。

一个小网格 $2\times3$ 的展平坐标是：行 `[0,0,0,1,1,1]`、列 `[0,1,2,0,1,2]`。第 $(1,2)$ 个位置的编码包含行 1 和列 2 的不同 SinCos 特征。

#### 3. 为什么 SinCos 可以承载相对位置信息？

对同一频率的两个位置 $a,b$：

$$\cos(a\omega)\cos(b\omega)+\sin(a\omega)\sin(b\omega)=\cos((a-b)\omega).$$

因此编码之间的内积自然包含**位置差**的信息；多个频率提供多尺度差异信号，但模型仍须通过训练学习空间关系。

**最容易混淆的地方：**

- **空间位置**：固定二维 SinCos，形状 `[1,N,D]`，加到 Patch Token 上；可用 `register_buffer` 保存，不参与梯度更新。
- **扩散时间**：时间步 `[B]` 经 `sincos_1d` → `[B,D]`，再经**可学习 Time MLP** 得到 $c(t)$；它不作为额外 Patch Token，而进入各 DiT Block 自己的条件投影 `SiLU + Linear(D,6D)`，生成各自的 scale / shift / gate。
- `sincos_1d` 并不等于可学习的位置 Embedding；**可学习的是后面的 Time MLP 和 adaLN 调制层**。

**完整 DiT 数据流：** 已在 **Q28** 保存，不在此重复架构图。

#### 本章补充｜`position_2d()` 的最小自测（**留作下次自己做，不提前勾选**）

```python
# z: [2,4,16,16], patch_size=2, hidden_size=256
patches = patchify(z, 2)               # [2,64,16]
tokens = patch_embed(patches)           # [2,64,256]
pe = position_2d(16 // 2, 16 // 2, 256)  # [1,64,256]
assert pe.shape == (1, 64, 256)
assert (tokens + pe).shape == (2, 64, 256)
```

这段代码提供**验收目标**，不等于已经学会 `meshgrid` 的数学原理。还需用 `2×3` 网格手写行、列索引，验证 `flatten()` 顺序与 `patchify()` 顺序一致。

[返回目录](#toc)

---

<a id="chapter-06"></a>
## 第 6 章｜DiT 主干｜Attention、adaLN-Zero、梯度与完整架构

**本章学习目标：** 这一章是第三周的核心：为什么普通 Transformer 通过时间条件调制就可以变成噪声预测器。

**整合后的三条主线：**

- 六组条件向量来自当前 t 的可学习投影，但每一层各自拥有调制参数。
- 零初始化的是调制末层与输出头，不是把整个网络全部置零。
- 输出头 → Gate → Attention/MLP 的分阶段梯度，是一个应当测试的预期现象。

<a id="q25"></a>

### Q25｜DiT Block 与普通 Transformer Block 有什么不同？时间条件如何通过 adaLN 注入？

**原始问题：**「为我讲解这部分 DiT。」本题对应第三周讲义的第九节：普通 Transformer、LayerNorm 和 adaLN。

**直觉：** 普通 Pre-LN Transformer 包含 Attention 和 MLP 两条残差分支。DiT 没有推翻它，而是在每个 Block 内用扩散时间的条件向量调制隐藏特征。

- 输入 Token：$X\in\mathbb R^{B\times N\times D}$，本项目典型形状为 `[B,64,256]`。
- 时间步：$t\in\mathbb Z^{B}$，经多频率 Sin/Cos 和时间 MLP 得到 $c\in\mathbb R^{B\times D}$。
- 常规 Pre-LN：$X'=X+\mathrm{Attention}(\mathrm{LN}(X))$，$X''=X'+\mathrm{MLP}(\mathrm{LN}(X'))$。

**LayerNorm 对每个 Token 的 D 个特征独立归一化**，不对 Batch 或 Token 序列求均值：

$$\mu=\frac1D\sum_d X_d,\quad v=\frac1D\sum_d(X_d-\mu)^2,\quad \mathrm{LN}(X)=\frac{X-\mu}{\sqrt{v+\varepsilon_{\mathrm{LN}}}}.$$

**adaLN（Adaptive LayerNorm）：** 让当前条件 $c$ 产生缩放 $\gamma(c)$ 与平移 $\beta(c)$：

$$\boxed{\mathrm{adaLN}(X,c)=(1+\gamma(c))\odot\mathrm{LN}(X)+\beta(c)}.$$

$\gamma,\beta$ 的形状都是 `[B,D]`，扩展为 `[B,1,D]` 后广播到每个 Patch Token；同一图像的不同 Token 共享该时间条件，但有不同的内容特征。之所以是 $1+\gamma$，是为了在 $\gamma=\beta=0$ 时退化为普通 LN，而不是将特征全部抹成 0。

**数值例：** 对 $x=(1,2,3,4)$，均值 $2.5$、方差 $1.25$，LN 结果约 $(-1.342,-0.447,0.447,1.342)$。如果 $\gamma=0.2,\beta=-0.1$（演示用标量），首个值变为 $1.2\times(-1.342)-0.1\approx-1.710$。

**和 U-Net 的区别：** 典型扩散 U-Net 先对 $t$ 做时间嵌入，再在卷积 ResBlock 内将 `[B,C,1,1]` 时间特征广播相加，或用 scale-shift 归一化；本周 DiT 使用 adaLN-Zero 对 `[B,N,D]` Token 的归一化和残差分支进行条件调制。不是说全部 U-Net 只能相加，也不是说全部 DiT 只能用 adaLN。

<a id="q26"></a>

### Q26｜adaLN-Zero 为什么有 6 组参数？为什么零门控不等于网络无法训练？

**原始问题：**「为我讲解这部分 DiT。」本题对应 adaLN-Zero 与分阶段梯度。

每个 Block 有 Attention 和 MLP 两条残差分支；每条分支分别需要 shift、scale、gate 三组 `[B,D]` 向量，共 **$6D$ 个条件数值**：

$$[\beta_a,\gamma_a,g_a,\beta_m,\gamma_m,g_m]=\mathrm{Linear}_{D\to6D}(\mathrm{SiLU}(c)).$$

Attention 与 MLP 的条件计算为：

$$\boxed{X'=X+g_a\odot\mathrm{Attention}((1+\gamma_a)\odot\mathrm{LN}_1(X)+\beta_a)}$$

$$\boxed{X''=X'+g_m\odot\mathrm{MLP}((1+\gamma_m)\odot\mathrm{LN}_2(X')+\beta_m)}.$$

注意 `$g$` 是残差分支输出的**门控向量**，不是 MoE 的专家选择 Router。

**Zero 具体零初始化了什么？** 教学实现把条件投影的末层 `nn.Linear(D, 6*D)` 权重和偏置初始化为零；Attention/MLP 内部并没有全部设零。初始化时 $\beta=\gamma=g=0$，所以 $X'=X,X''=X$，**Block 初始为恒等映射**。

**门控为什么还可以学？** 设简化分支 $y=x+g f_w(x)$，链式法则给出：

$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial y}\,g\,\frac{\partial f_w}{\partial w},\quad \frac{\partial L}{\partial g}=\frac{\partial L}{\partial y}\,f_w(x).$$

在 $g=0$ 时，分支内部参数梯度为零，但 Gate 的梯度**可能非零**。如 $x=2,f_w(x)=3,g=0,L=\frac12(y-5)^2$，初始 $y=2$、$\partial L/\partial y=-3$，则 $\partial L/\partial g=-9$，$\partial L/\partial f_w=0$；学习率 0.01 更新后 $g=0.09$，后续分支便可能获得梯度。

**完整 TeachingDiT 的区别：** 输出预测头 `output` 也零初始化；初始模型预测全零，首次反向传播往往先更新输出头，随后门控、再后续 Attention/MLP 获得梯度。不能要求第一步所有参数梯度均非零。

<a id="q27"></a>

### Q27｜从潜变量到噪声预测，TeachingDiT 的前向流程、张量形状和验收测试是什么？

**原始问题：**「为我讲解这部分 DiT。」本题对应完整教学代码与运行测试。

讲义的 `TeachingDiT` 以 `[B,4,16,16]` 带噪潜变量和 `[B]` 时间步为输入，经过：

```text
z_t                  [B,4,16,16]
  -> patchify(p=2)   [B,64,16]        # 只重排，不学习
  -> Linear(16,256)  [B,64,256]       # Patch Embedding
  -> + 2D SinCos PE  [B,64,256]       # 位置条件
  -> DiT Blocks      [B,64,256]       # 时间 c=[B,256] 经 adaLN-Zero 调制
  -> final LN + modulation
  -> Output Linear   [B,64,16]        # 每个 Patch 预测 16 个噪声值
  -> unpatchify      [B,4,16,16]      # 恢复噪声张量的空间位置
```

时间路径独立于 Token 路径：`timesteps [B] -> sincos_1d -> time_mlp -> c [B,D]`，**c 不等同于第 65 个图像 Token**；各个 Block 用它生成自己的六组调制参数。

实现上：`TeachingAttention` 显式使用 QKV、$QK^\top/\sqrt{d_h}$、Softmax、$AV$；`TeachingDiTBlock` 中两次 `LayerNorm(elementwise_affine=False)`、一次自注意力、一次 `Linear(D,4D) -> GELU -> Linear(4D,D)`、一层 `SiLU -> Linear(D,6D)` 条件调制；`TeachingDiT` 把这些模块连接起来，并零初始化条件调制末层和输出头。该实现是教学代码，**不能据此认为兼容官方模型权重**。

**重要验证：**

1. 手工排列一个微型张量，独立核对 Patchify 各元素索引；不能只测 `unpatchify(patchify(x)) == x`，因为两个错误索引可能相互抵消。
2. 用逐 Token、逐 Head 的显式循环计算 Attention，核对向量化实现的结果。
3. 验证调制广播正确：`[B,D] -> [B,1,D]`，同一 Batch 样本内不同 Token 接收相同的 shift/scale。
4. 初始化后单独 Block 应严格输出输入；内部 Attention 权重第一步梯度可以为 0，条件末层的 Gate 梯度可以非零。
5. 完整模型输出头零初始化时，检查最初输出全零，以及 **输出头 -> Gate -> 内部分支** 的逐步梯度传播。训练时再接 $\mathcal L=\mathrm{MSE}(\hat\epsilon,\epsilon)$。

**易错点：** `LayerNorm(elementwise_affine=False)` 由时间条件负责后续仿射调制；输出头零初始化是为了初始噪声预测为零，不意味着模型不训练；身份 Block 和零输出头是两个不同层次的初始化性质。

<a id="q28"></a>

### Q28｜时间步 t 如何逐层成为 adaLN-Zero 的 γ、β、gate？完整 DiT 前向架构与张量形状

**原始问题：**「所以时间t对于DiT的影响，其实就体现在c是吗，c=c(t)，然后adaLN中adaLN=adaLN（X，c(t)）,所以gamma(c)=gamma(c(t)),beta(c)=beta(c(t))。在这里引入了时间步t是吗？以及在后面的门控g也是一样。gamma，beta，g其实都是c的函数，或者说间接是t的函数是吗？」；「六组条件参数其实是经过对时间的位置编码后经过MLP得到的c形状是(b, d)，再经过SiLU和Linear才变成(b, 6d)是吗？给我一个完整的DiT的text的架构图吧……」

**直接结论：** 对。时间步 $t$ 经时间正弦余弦嵌入和 MLP 得到 $c=c_\theta(t)$；每个 Block 用**各自独立的**调制层输出 $\beta_a(c),\gamma_a(c),g_a(c),\beta_m(c),\gamma_m(c),g_m(c)$。它们都是 $c$ 的函数，因此也是 $t$ 的（经网络参数决定的）间接函数。这里的正弦余弦是**时间嵌入**，与二维空间位置编码虽形式相近，但含义不同。

**本周 `TeachingDiT` 的完整架构（已替换为更详细的双分支文本图）**（$D=256$、$p=2$、Block 深度 6，默认输入为 `[B,4,16,16]`）：

```text
                         【DiT 前向传播：B=任意批量、D=256、p=2】

       【潜变量主干】                            【时间条件分支】
  带噪潜变量 z_t [B,4,16,16]                 时间 t [B]
             │                                  │
        Patchify(p=2)                     sincos_1d(t,256)
             ▼                                  ▼
         [B,64,16]                           [B,256]
             │                                  │
      Linear(16,256)                         Time MLP
             │                         Linear(256,256) → SiLU
             ▼                         → Linear(256,256)
         [B,64,256]                            │
             │                                  ▼
      + 2D Position PE                     c(t) [B,256]
         [1,64,256]                            │
             │                        ┌─────────┴─────────────┐
             ▼                        │                       │
       Tokens X [B,64,256]             │ 每个 Block 自己的       │ Final 层另用
             │                        │ SiLU → Linear(D,6D)   │ Linear(D,2D)
             ▼                        ▼                       │
  ┌──────────────────── DiT Block 1 ────────────────────────┐ │
  │                          │                             │ │
  │                  SiLU → Linear(D,6D)                    │ │
  │                          │                             │ │
  │                       chunk(6)                         │ │
  │                          │                             │ │
  │       ┌──────────────────┴──────────────────┐          │ │
  │       ▼                                     ▼          │ │
  │   γ_attn, β_attn, g_attn               γ_mlp, β_mlp, g_mlp│ │
  │       │                                     │          │ │
  │ X ───┼───────────────┐                     │          │ │
  │ │     ▼               │                     │          │ │
  │ └→ LN1 → ×(1+γ_attn) + β_attn              │          │ │
  │               │        │                    │          │ │
  │               ▼        │                    │          │ │
  │              MHA       │                    │          │ │
  │               │        │                    │          │ │
  │            × g_attn    │                    │          │ │
  │               │        │                    │          │ │
  │               └── + X ─┘                    │          │ │
  │                    ▼                        │          │ │
  │                   X' ───────────┐           │          │ │
  │                    │            │           │          │ │
  │                    ▼            │           ▼          │ │
  │                   LN2 → ×(1+γ_mlp) + β_mlp             │ │
  │                            │                           │ │
  │                            ▼                           │ │
  │                           MLP                          │ │
  │                            │                           │ │
  │                          × g_mlp                       │ │
  │                            │                           │ │
  │                            └──── + X' ◄───────────────┘ │
  │                                      │                   │
  └──────────────────────────────────────┼───────────────────┘
                                         ▼
                                 X'' [B,64,256]
                                         │
                              DiT Block 2  ◄──── c(t)
                                         │
                                        ...
                                         │
                              DiT Block 6  ◄──── c(t)
                                         │
                                         ▼
                                      Final LN
                                         │
                     c(t) ─→ SiLU → Linear(D,2D)
                                         │
                                   γ_final, β_final
                                         │
                                         ▼
                            ×(1+γ_final) + β_final
                                         │
                                 Output Linear(256,16)
                                         │
                                      [B,64,16]
                                         │
                                     Unpatchify
                                         │
                                         ▼
                                ε_pred [B,4,16,16]

注：所有 γ/β/g 的形状均为 [B,D]，在 Token 维广播到 [B,N,D]；
    时间条件 c(t) 为所有 Block 共用，但各 Block 的调制 Linear 参数不同。
```

**更严格的两条残差式：** 下式中 `shift/scale/gate` 的 `[B,D]` 都通过 `[:,None,:]` 广播到 `[B,N,D]`：

$$X'=X+g_a(c)\odot\operatorname{MHA}((1+\gamma_a(c))\odot\operatorname{LN}_1(X)+\beta_a(c))$$

$$X''=X'+g_m(c)\odot\operatorname{MLP}((1+\gamma_m(c))\odot\operatorname{LN}_2(X')+\beta_m(c)).$$

**两个常见误区：** (1) `c(t)` 不是附加的第 65 个 Patch Token；它独立进入 Block 的调制层。(2) 所有 Block **共享当前时间向量** $c$，但**不共享**各自的 `Linear(D,6D)` 权重；而模型最后的调制层输出只有 $2D$，再由噪声输出投影和 `Unpatchify` 恢复 `[B,4,16,16]`。

<a id="q29"></a>

### Q29｜既然 γ、β、g 是 t 的函数，为什么还要零初始化？和普通 LayerNorm 的 γ=1 冲突吗？

**原始问题：**「为什么初始化要求全0？attention中gamma初始化就是ones啊，而且既然gamma、beta、g都是c乃至t的函数，不应该是从t一步步得到的吗，为什么还要初始化？」

**先分清两种 γ：** 普通 `LayerNorm(elementwise_affine=True)` 的可学习 **gain** 通常初始化为 **1**；本项目 `TeachingDiTBlock` 使用 `LayerNorm(elementwise_affine=False)`，没有这组固定 affine 参数。这里写作 $\gamma(c)$ 的是 **adaLN 的额外尺度增量**，真实有效系数是 $1+\gamma(c)$：如果 $\gamma(c)=0$，有效缩放正好是 1，**没有矛盾**。

**零初始化的是什么？** `self.modulation = SiLU() -> Linear(D,6D)` 的**最后一层 Linear 权重 $W_c$ 和偏置 $b_c$** 初始化为零，而非将整个 DiT 的 Attention、MLP 或 Time MLP 全部置零：

$$m(c)=W_c\operatorname{SiLU}(c)+b_c,\quad W_c^{(0)}=0,\ b_c^{(0)}=0.$$

因此初始无论输入时间 $t$ 是多少，六组参数 $\beta,\gamma,g$ 都为零，每条 Block 为恒等映射 $X''=X$。**由 $t$ 计算输出**和**用哪些初始权重计算输出**是不同的两件事：训练后 $W_c,b_c$ 更新为非零，$m(c(t))$ 就可以随 $t$ 改变。

**为什么这么做？** 希望深层残差分支在初始化时对主干的扰动为零，让训练逐渐开启 Attention/MLP 分支。注意完整 `TeachingDiT` 的**最终噪声输出 Linear 也被零初始化**，因此初始化时模型预测 $\hat\epsilon=0$；这与“每个 Block 初始是 Identity”是两个不同层次的性质。

<a id="q30"></a>

### Q30｜零 Gate 为什么还能训练？一行梯度推导与小型数值例子

**原始问题：**「再在qa中给一个简单的例子（篇幅不要太大）简述其实零初始化可以训练网络和参数。」

设一条简化残差分支：

$$y=x+g a,\qquad L=\tfrac12(y-y^*)^2.$$

初始取 $x=2,a=3,g=0,y^*=5$，则 $y=2$，所以

$$\frac{\partial L}{\partial g}=(y-y^*)a=(-3)\times3=-9,\qquad
\frac{\partial L}{\partial a}=(y-y^*)g=(-3)\times0=0.$$

可见 **Gate 虽然为 0，梯度却非零**；学习率为 $0.01$ 时，$g\leftarrow0-0.01(-9)=0.09$，下一步分支输出 $a$ 就可以得到梯度。

**补充限制：** 若完整 DiT 的输出头也初始化为零，第一步可能只有输出头得到梯度；输出头更新后，再轮到门控及 Attention/MLP。故“首次 backward 不是全部参数都有梯度”**不等于**训练失败。

<a id="q34"></a>

### Q34｜`elementwise_affine=False` 与 `register_buffer(..., persistent=False)` 分别控制什么？

**原始问题：**「`nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)` 的 `elementwise_affine=False` 是什么意思？`self.register_buffer("position_2d", pos_2d, persistent=False)` 的 `persistent=False` 是什么意思？」

**① `elementwise_affine=False`：关闭 LayerNorm 自带的可学习仿射参数。**

标准化部分仍然执行：

$$\operatorname{LN}_0(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}.$$

- `True`（默认）：有 `weight`（初始为 1）和 `bias`（初始为 0），输出 $\gamma_{\mathrm{LN}}\odot\operatorname{LN}_0(x)+\beta_{\mathrm{LN}}$，两者均可学习。
- `False`：不创建这些可学习的 `weight`、`bias`，只输出 $\operatorname{LN}_0(x)$；**仍可以沿归一化运算对输入反向传播**，不等于 `no_grad()`。
- **在本周 DiT 中：** adaLN-Zero 已从时间条件 $c(t)$ 生成 $\gamma(c),\beta(c)$，再计算 $(1+\gamma(c))\odot\operatorname{LN}_0(x)+\beta(c)$；因此参考实现不再使用 LN 自带的仿射参数。

**② `register_buffer(..., persistent=False)`：注册非参数状态，但不放入 `state_dict()`。**

- `register_buffer("position_2d", pos_2d, ...)`：把固定的二维位置编码保存为模型 Buffer，可通过 `self.position_2d` 访问；它**不是**可训练的 `nn.Parameter`。
- `persistent=True`（默认）：这个 Buffer 会作为模型状态出现在 `state_dict()`，随通常的权重保存流程一起保存。
- `persistent=False`：仍是 Buffer，仍会随 `model.to(device/dtype)` 移动/转换，**但不会出现在 `state_dict()`**；加载模型时需要按同样的配置重新生成它。
- **在本周 DiT 中：** 固定 SinCos 位置编码可由 Patch 网格大小和隐藏维度重新计算，故可设为 `persistent=False`，减少不必要的检查点内容。位置编码数值是否一致仍取决于构造配置和算法。

| 参数 | 控制什么 | 不代表什么 |
|---|---|---|
| `elementwise_affine=False` | LayerNorm 不创建自己的可学习 `weight`/`bias` | 不是停止标准化，也不是关闭自动求导 |
| `persistent=False` | Buffer 不保存到 `state_dict()` | 不是不注册 Buffer，也不是不随模型移动设备 |

**后续复习关联：** `position_2d()` 的原理及它在 DiT 中的构造与注册**仍未学会**，已在本笔记开头标记待补学；本题只解释这两个 PyTorch 参数。

#### 本章补充｜自己写 `DiffusionTransformer` 时最容易踩的五个坑

```python
# 1. 二维位置编码必须基于 Patch 网格，不是潜变量原高宽
pos = position_2d(height // patch_size, width // patch_size, hidden_size)
self.register_buffer("position_2d", pos, persistent=False)

# 2. adaLN 是增量缩放，不能把 1 + gamma 写成 1 * gamma
modulated = (1 + gamma[:, None, :]) * x + beta[:, None, :]

# 3. 最终条件层只切出两组；两个 [B,D] 要沿 token 维广播
shift_or_gamma, beta = self.final_modulation(condition).chunk(2, dim=-1)

# 4. 最终噪声输出头要 zero-init，初始 eps_pred 应该是全零
nn.init.zeros_(self.output.weight)
nn.init.zeros_(self.output.bias)

# 5. 如果使用 self.apply(Xavier)，必须在它之后重新 zero-init
#    DiTBlock.modulation[-1]、final_modulation[-1]、output。
```

**补充工程检查：** 确认 `unpatchify` 的 Token 数 `N=(H/p)(W/p)`、特征宽度 `Cp²`；`nn.LayerNorm(elementwise_affine=False)` 没有内置 affine，但前向仍可自动求导；普通 Attention 不使用 Causal Mask。

[返回目录](#toc)

---

<a id="chapter-07"></a>
## 第 7 章｜实验与证据｜固定四样本过拟合说明了什么

**本章学习目标：** 把“实现可训练”和“已经学会真实生成”严格区分。

**整合后的三条主线：**

- 小模型在固定 4 样本、固定 t=499 上降低 MSE，只验证端到端连接和拟合能力。
- 真实生成需要随机训练时刻与噪声、完整反向采样、VAE 解码、独立评测。
- Notebook 的参考实现通过检查，不代表个人手写实现已经自动通过相同测试。

<a id="q31"></a>

### Q31｜第三周的固定目标小数据拟合实验究竟证明了什么？

**原始问题：**「这部分在干什么？」（十一、小数据拟合：证明连接正确，不夸大能力）

**目标：** 在正式训练 DiT 之前，验证从缓存潜变量、DDPM 加噪、DiT 前向、Noise MSE、反向传播到优化器更新的**端到端优化链路**能否工作。它不是完整生成训练，也不是泛化能力评估。

**实验一：基准规格与形状检查。** 构造 `TeachingDiT(depth=6, hidden=256, heads=4)`，参数约 7,372,560；用 `[1,4,16,16]` 零输入及时间步 `t=0` 运行前向，断言输出仍为 `[1,4,16,16]`。由于噪声输出头零初始化，初始零预测是预期行为。这个基准模型**没有参与后面的拟合训练**。

**实验二：四个固定样本的小模型拟合。** 从试验潜变量缓存取 `pilot_latents[:4]`，固定随机噪声 $\epsilon$，固定时间 $t=499$，用线性 $\beta$ 调度（1000 步）计算 $\bar\alpha_{499}$，构造

$$z_{499}=\sqrt{\bar\alpha_{499}}z_0+\sqrt{1-\bar\alpha_{499}}\epsilon.$$

训练更小的 `TeachingDiT(depth=2, hidden=128, heads=4)`，不断最小化

$$L=\operatorname{mean}[(\epsilon_\theta(z_{499},499)-\epsilon)^2].$$

四组 `(z₄₉₉, t, ε)` **在所有优化步骤中保持不变**，因此特别容易被记忆；这样设计是为了诊断优化器、梯度流、条件通道与输出头能否正确联通，而不是为了训练通用去噪器。

**结果与边界：** 讲义展示固定目标 MSE 约从 `1.0015` 降至 `0.00181`（约 100 次优化）。这说明模型能够拟合这四组固定数据，并通过了基本训练连通性检查；**不说明**它能在未见图像、其他时间步或新噪声上泛化，也不说明已经学会生成图像分布。

**正式生成训练仍需：** 在大量训练潜变量上随机重采样 $z_0,t,\epsilon$；完成多步反向采样链，得到 $\hat z_0$；再经逆缩放与 VAE Decoder 输出图像，并做独立评估。不能把当前固定目标低 MSE 当作 FID、样本质量或生成成功率。

#### 本章补充｜从 sanity check 走向真实训练的阶梯

| 阶段 | 最小验收 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| Patch 索引测试 | 手工索引 + 双向往返 | 空间排列正确 | 输出表示质量 |
| Attention / adaLN 数值测试 | 向量化与循环结果一致 | 局部计算正确 | 最终生成能力 |
| Zero-init / 梯度测试 | Block identity、分阶段梯度 | 训练路径未被永久阻断 | 已学会去噪 |
| 固定四样本拟合 | 固定输入、时间、噪声，MSE 明显降低 | 优化闭环可运行 | 泛化、采样、图像生成 |
| **后续正式训练** | 新样本 + 随机时间/噪声 + 多步采样 | 才能检验潜空间去噪泛化 | 单凭 MSE 也不代表视觉质量好 |
| **后续独立评测** | 原图/采样图属性、分布与条件一致性 | 才能支持生成质量主张 | 不能用 VAE 重建代替生成 |

[返回目录](#toc)

---

<a id="review"></a>
## 收官复习｜最短知识链、易错点与下一步

### 1. 合上笔记，必须可以独立解释的 10 个问题

1. 为什么 VAE 的 4 个潜通道不代表 RGBA？
2. `posterior.sample()`、`.mode()`、`z=sh` 分别干什么？
3. 为什么先测试重建质量，再训练潜空间 DiT？
4. 为什么 DDPM 的噪声预测公式从像素空间迁到潜空间不必改变？
5. `CHW` 与 Batch 维度分别如何进入 Noise MSE？
6. Patchify 的 `N`、`K` 如何由 `C,H,W,p` 算出？
7. **（待补学）** 为什么 `position_2d(8,8,256)` 生成 `[1,64,256]`？
8. `sincos_1d` 的不同频率提供了什么信息？它为什么不是可学习 MLP？
9. adaLN-Zero 的 6 组向量分别作用在哪里？为什么零 Gate 还能训练？
10. 固定 4 样本拟合降到很低，为什么仍不能声称“成功生成了图像”？

### 2. 容易混淆的概念速查

| 易混淆对象 | 区别 |
|---|---|
| VAE 重建 vs DiT 生成 | 前者有真实图像作为编码输入，后者要从随机噪声经多步采样得到新图像 |
| Downsampling vs Scaling Factor | 前者改变空间尺寸/表示，后者只乘常数、改变数值尺度 |
| VAE posterior sample vs DDPM 噪声采样 | 前者决定 `h` 的表示随机性，后者在训练中对 `z₀` 额外加噪 |
| Patchify vs Patch Projection | 前者只是 reshape/permute；后者有可学习权重 |
| Position Encoding vs Timestep Embedding | 前者编码二维空间位置并与 tokens 相加；后者编码时间并驱动 adaLN |
| 标准 LayerNorm γ vs adaLN γ | 前者是可学习 affine gain（此处关闭）；后者是条件生成的尺度增量，实际系数为 `1+γ` |
| `register_buffer(...,persistent=False)` vs `nn.Parameter` | 前者不是训练参数且不进入 `state_dict`；后者是可学习参数 |
| Block identity vs DiT 零输出 | 前者由残差 Gate=0 实现；后者由最终输出头权重全零实现 |
| 单元测试 vs 固定拟合 vs 正式生成 | 分别检查局部数值、优化链路和真实生成/泛化能力 |

### 3. 继续推进前的行动列表（不要提前标记完成）

- [ ] **补学 `position_2d`**：自己手算 `2×3` 的行列坐标；写函数；用索引例子检查。
- [ ] **补学 DiT 中位置 Buffer 的设置**：手动解释 `H//p`、`W//p`、`[1,N,D]`、`persistent=False`。
- [ ] 本人运行 Patchify 独立索引、Attention 循环、adaLN 广播、Block 恒等性测试。
- [ ] 本人运行完整模型的 Zero Output 与 **输出头 → Gate → Attention/MLP** 分阶段梯度测试。
- [ ] 本人运行固定四样本拟合，记录初末 Loss、设备、随机种子、步数与代码版本。
- [ ] 后续再设计随机时间潜空间训练、多步采样、生成质量评估；避免混淆本周教学已覆盖内容和未来研究目标。

### 4. 参考资料与证据边界

- **本周主要教学来源**：`S2_第3周_图标数据与潜空间DiT基础.ipynb`，共十二节（数据与系统→VAE→缓存→Latent DDPM→Patch→DiT→教学参考→小样本拟合→交接）。
- **本周问答来源**：原始 Q1–Q34；原始版本另存于完整 ZIP，供逐字回看。
- **参考论文**：Auto-Encoding Variational Bayes（VAE）；Denoising Diffusion Probabilistic Models（DDPM）；High-Resolution Image Synthesis with Latent Diffusion Models（LDM）；Scalable Diffusion Models with Transformers（DiT）。
- **结论边界**：文中的实验数字来自教学 Notebook／既有学习记录；这份整理本身不构成一次新的训练或评测。

---

> **收官句**：本周最重要的收获不是“我训练出了生成器”，而是明白 **VAE 怎样决定表示、DDPM 怎样定义噪声、DiT 怎样通过位置与时间条件预测噪声，以及哪些测试只能证明代码正确而不能证明生成能力。**
