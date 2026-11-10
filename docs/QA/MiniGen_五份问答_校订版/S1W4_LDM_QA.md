## **一、LDM 总览与导航**

> MiniGen 第四周：潜空间扩散的教学训练、采样、恢复与诊断。全篇按主题排列 38 道题，最后八道为综合自测。

### **1. 潜空间扩散流程与实验结果**

**本周任务：** 把冻结 VAE、潜变量缓存、DDPM 噪声目标、教学 DiT、EMA、训练恢复和 1000 步采样连接起来，建立 **LDM 的端到端教学闭环**。

```text
训练：RGB 图像 x → Frozen VAE Encoder → h=μϕ(x) → z₀=s·h → 随机选 t、ε
                   → zₜ=√āₜz₀+√(1−āₜ)ε → DiT(zₜ,t)≈ε → MSE → AdamW(θ) → EMA(θ_EMA)
采样：z_T~N(0,I) → [DiT 预测 ε̂ → DDPM 反向更新]×1000 → 生成 z₀ → z₀/s → Frozen VAE Decoder → RGB
评估：固定验证 Probe（按时间分桶）＋ Zero/Moment/DiT/EMA 对照 ＋ 重建控制 ＋ 最近邻 ＋ 生成图/轨迹
```

| 本周具体设置 / 结果 | 实际情况 | 阅读边界 |
|---|---|---|
| 图像 / 潜变量 | `[B,3,128,128]` → `[B,4,16,16]`；`s=0.13025` | 该值为本项目约定，不要自行改标定 |
| 数据 | 10,000 训练 + 1,000 验证；冻结 VAE 后验均值缓存 | 测试集未使用 |
| DiT | 教学模型 663,952 参数，2 层/128 维/4 头 | 7,372,560 参数主规格仅做形状与参数验收，未训练 |
| 训练 | 固定拟合 100 步；随机训练 500 步＋恢复到 1000 步 | `global_step` 与扩散时间索引不是一回事 |
| 验证 | 固定验证 Probe 4 桶 × 16 张；EMA 在本次四桶的 MSE 均低于原模型 | 不代表生成质量已经合格 |
| 生成 | DDPM 完整 1000 次反向更新；8 张无条件图 | **程序完成，但图标生成质量未达标** |
| 定位 | 真实图像 VAE 重建良好，生成潜变量存在离群点（第 5 张显著） | 尚不能单独归因于 DiT 或某个通道 |

#### **i. 最需要分清的六组概念**

| 容易混淆的一对概念 | 应当怎样区分 |
|---|---|
| `μϕ(x)` vs 数据集均值 | 前者是 **单张图像的 VAE 后验参数**，后者是整个样本集合的统计量 |
| VAE 后验噪声 `η` vs DDPM 噪声 `ε` | 前者决定怎样从 VAE 近似后验抽 latent；后者决定如何给已得 `z₀` 加噪 |
| 训练 `global_step` vs 扩散时间 `t` | 一个是优化器更新计数，一个是输入的噪声等级 |
| 训练日程 `T` vs 推理步数 `K` | `T` 指噪声日程/训练时间范围，`K` 是生成时选多少步进行更新 |
| 真后验 $q(z_{t-1}\mid z_t,z_0)$ vs 模型反向分布 $p_\theta(z_{t-1}\mid z_t)$ | 前者已知真 `z₀` 时可精确推导；后者在未知真 `z₀` 时使用神经网络构造近似 |
| 训练完成 vs 采样完成 vs 生成合格 | 三项是三个独立判断；本周前两项达到计划，最后一项 **尚未达到** |

#### **ii. 全文目录（点击直达）**

- [第一篇｜LDM、VAE 与去噪网络结构](#part-a)：[Q1](#q1)、[Q2](#q2)、[Q3](#q3)、[Q4](#q4)、[Q5](#q5)、[Q6](#q6)、[Q7](#q7)
- [第二篇｜潜变量缓存、数据身份与可复现性](#part-b)：[Q8](#q8)、[Q9](#q9)
- [第三篇｜随机训练、监督目标与诊断基线](#part-c)：[Q10](#q10)、[Q11](#q11)、[Q12](#q12)、[Q13](#q13)、[Q14](#q14)、[Q15](#q15)、[Q16](#q16)
- [第四篇｜EMA、Checkpoint 与实验归档](#part-d)：[Q17](#q17)、[Q18](#q18)、[Q19](#q19)、[Q20](#q20)
- [第五篇｜DDPM 反向采样、DDIM 跳步与推理](#part-e)：[Q21](#q21)、[Q22](#q22)、[Q23](#q23)、[Q24](#q24)、[Q25](#q25)、[Q26](#q26)、[Q27](#q27)
- [第六篇｜生成质量评估与失败定位](#part-f)：[Q28](#q28)、[Q29](#q29)、[Q30](#q30)
- [第七篇｜第四周八道综合自测（独立练习）](#part-g)：[Q31](#q31)、[Q32](#q32)、[Q33](#q33)、[Q34](#q34)、[Q35](#q35)、[Q36](#q36)、[Q37](#q37)、[Q38](#q38)

#### **iii. 重要数学公式速查**

| 主题 | 公式 | 对应问题 |
|---|---|---|
| VAE 后验采样 / 固定均值 | $h=\mu_\phi(x)+\sigma_\phi(x)\odot\eta$；本项目取 $h=\mu_\phi(x)$ | Q2 |
| 潜变量缩放 | $z_0=s\,h$；解码前 $h=z_0/s$ | Q3、Q4 |
| 前向加噪 | $z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon$ | Q21、Q10 |
| DiT 噪声目标 | $\mathcal L=\mathbb E[\lVert\epsilon_\theta(z_t,t)-\epsilon\rVert^2/(CHW)]$ | Q10、Q11 |
| 矩基线 | $\hat\epsilon=\frac{\sqrt{1-a}}{av+1-a}(z_t-\sqrt a\mu)$ | Q14、Q15 |
| 反解干净潜变量 | $\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}}$ | Q22 |
| EMA | $\theta^\mathrm{EMA}_k=d\theta^\mathrm{EMA}_{k-1}+(1-d)\theta_k$ | Q17、Q18 |
| DDIM 确定性跳步 | $z_s=\sqrt{\bar\alpha_s}\hat z_0+\sqrt{1-\bar\alpha_s}\hat\epsilon$ | Q25、Q26 |

---

<a id="part-a"></a>

## **二、LDM、VAE 与去噪网络结构**

> **本篇主线：** 先弄清哪些模块压缩图像、哪些模块预测噪声，再掌握潜变量尺度、卷积多尺度与 DiT 条件调制。

<a id="q1"></a>

### **1. 第四周如何把 VAE、DDPM 和 DiT 组成完整系统？**

**答：算法主干基本是在整合旧知识，但整合本身引入了新的系统要求。**

- DDPM：加噪公式、噪声预测损失、反向采样。
- VAE：把 `[B,3,128,128]` 压缩成 `[B,4,16,16]`，训练时冻结并事先编码。
- DiT：在 latent patch token 上预测噪声。
- 新的工程闭环：潜变量尺度协议、确定性缓存、缓存身份校验、EMA 参数采样、模型恢复训练、完整反向采样与 VAE 解码、评测与复现实验。

训练链：`图片 x → VAE Encoder → z₀ → 随机采样 t,ε → zₜ → DiT(时间条件) → ε̂ → MSE → 更新 DiT`。

推理链：`z_T~N(0,I) → DiT+DDPM 反向更新 → z₀ → z₀/s → VAE Decoder → RGB`。

**新价值不在发明新的扩散公式，而在验证完整生成系统真的能训练、恢复、推理和量化评估。**

---

<a id="q2"></a>

### **2. 什么是 VAE 后验均值、后验采样随机性？固定均值为什么不消除 DDPM 的随机性？**

#### **i. VAE Encoder 输出什么？**

普通确定性 AE 可以直接产生特征 `h = encoder(x)`。KL-VAE 则通常产生潜变量高斯 **近似后验分布** 的参数：

$$q_\phi(h\mid x)=\mathcal N\big(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\big).$$

- `μϕ(x)` 是这张图像对应的 **潜变量分布的均值**，张量形状为 `[B,4,16,16]`；它不是整批训练集的统计均值。
- `log σ²ϕ(x)` 是另一组同形状的对数方差参数。
- 「后验」在这里指 **已知输入图片 x 时，对潜变量 h 的概率估计**；`qϕ` 是对真正后验的近似，而不是声称它是精确后验。

#### **ii. 后验采样随机性是什么？**

标准重参数化采样：

$$h=\mu_\phi(x)+\sigma_\phi(x)\odot\eta,\qquad\eta\sim\mathcal N(0,I)$$

若单维例子 `μ=2`、`σ=0.5`，三次抽取 `η=1,-0.6,0.2`，会得到 `h=2.5,1.7,2.1`。**同一张图片、同一套 VAE 权重也可能采样出不同的 h**；随机性来自 `η`。

#### **iii. MiniGen 为什么固定后验均值？**

本周不用 `η` 采样，直接取 `h=μϕ(x)`，再用项目约定 `z₀=s·h`，其中 `s=0.13025`。给定固定输入、固定权重、固定预处理与确定性计算，同一图片对应相同的 `z₀`；方便预编码并缓存。**固定均值是项目选择，并非所有 LDM 都必须如此。**

#### **iv. 为什么 DDPM 仍然随机？**

即使 `z₀` 固定，训练时仍随机抽取图片 Batch、扩散时间步 `t`，并重新生成独立高斯噪声 `ε`：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon, \quad \epsilon\sim\mathcal N(0,I)$$

例：固定 `z₀=0.5`、`ᾱₜ=0.64`，若 `ε=-1` 则 `zₜ=-0.2`；若 `ε=1` 则 `zₜ=1.0`。固定的是 **VAE 后验的 `η` 不再参与**，不是把扩散噪声 `ε` 也固定了。

#### **v. 与先前自测的联系**

- 问：固定 `z₀`，重采 `ε`，同一时刻 `zₜ` 是否相同？**不一定**。
- 问：预测噪声的真实监督标签是什么？**`ε`**。
- 问：为什么不需要连续加噪 750 次？**可由前向高斯链推出 `q(zₜ|z₀)` 的闭式直接采样式**。

**易错点：** `VAE 的后验采样噪声 η ≠ DDPM 扩散噪声 ε`；`每张图像对应的 VAE 后验均值 μϕ(x) ≠ 整个潜变量数据集的通道均值`；缓存固定的 `z₀` 不意味着训练随机性消失。

**数值例子：** 当 `μ=2, σ=0.5, η=-1` 时，后验采样 `h=2+0.5×(-1)=1.5`；固定后验均值时 `h=2`；固定 `h` 不代表 DDPM 的 `zₜ` 必然相同，因为 `t` 和 `ε` 仍可重新采样。

---

<a id="q3"></a>

### **3. 潜空间缩放系数 `s` 如何确定？方差到底对哪些元素计算？**

#### **i. 直觉**

VAE 编码后的 `h` 包含很多数，需要先选定 **统计的集合**，才能谈方差。常见约定是在代表性训练样本上，把 `[N,C,H,W]` **所有样本、所有通道、所有空间位置** 合并为一个总体，计算一个全局标准差，再使用一个固定的乘法系数；也可以按通道统计，但那会得到 C 个不同的数值，不是同一个操作。

#### **ii. 全局方差的计算**

对 `h.shape=[10000,4,16,16]`，元素总数为 `N=10,240,000`：

1. 展平为 `h_1,…,h_N`。
2. 计算总体均值：$\mu_h=\frac1N\sum_{i=1}^{N}h_i$。
3. 计算总体方差：$\operatorname{Var}(h)=\frac1N\sum_{i=1}^{N}(h_i-\mu_h)^2=\mathbb E[h^2]-(\mathbb E[h])^2$。
4. 计算标准差：$\sigma_h=\sqrt{\operatorname{Var}(h)}$。
5. 若期望 $\operatorname{Var}(z)=1$ 且 `z=sh`，利用 $\operatorname{Var}(sh)=s^2\operatorname{Var}(h)$，取 $s=1/\sigma_h$。

```python
# h 是缩放前的 VAE 后验均值，形状 [N, 4, 16, 16]
mu_h = h.float().mean()
std_h = h.float().std(unbiased=False)   # 所有维度合并统计
s_example = 1.0 / std_h
z = s_example * h
```

#### **iii. 与逐通道方差的区别**

```python
channel_mean = z.mean(dim=(0, 2, 3))
channel_std = z.std(dim=(0, 2, 3), unbiased=False)
```

保留通道轴 `dim=1`，所以得到四个不同的通道统计值。**四个通道分别标准化** 是另一种编码约定，不能未经验证就用于已经按固定缩放系数训练的 LDM。

#### **iv. 当前运行结果的数值解释**

Notebook 报告（这里 `z` 已经缩放）：

- 均值约 `[2.2052, 0.1513, 0.9324, -0.8046]`
- 标准差约 `[0.7872, 0.5906, 0.6721, 0.5221]`

等规模通道混合总体时：

$\mu_{\mathrm{global}}=\frac1C\sum_c\mu_c\approx0.6211$

$\operatorname{Var}_{\mathrm{global}}(z)=\frac1C\sum_c(\sigma_c^2+\mu_c^2)-\mu_{\mathrm{global}}^2\approx1.6381$

$\operatorname{Std}_{\mathrm{global}}(z)\approx1.2799$。

所以 **0.13025 并未使这一批图标的方差严格等于 1**。这是正常且需要记录的现象：`0.13025` 是实验规定的固定尺度（也与公开的 SDXL VAE 配置相符），并不是针对这里 10,000 张图重新估算的。不要为了本批数据正好方差为 1 而任意修改已有编码—训练—解码约定。

#### **v. 易错点：方差 vs 二阶原点矩**

$\operatorname{Var}(h)=\mathbb E[h^2]-(\mathbb E[h])^2$，但 $\mathbb E[h^2]$ 不扣去均值；$s=1/\sqrt{\mathbb E[h^2]}$ 只能使 $\mathbb E[z^2]=1$，**不能保证 $\operatorname{Var}(z)=1$**，除非均值为零。缩放只改变尺度，不等于均值归零的标准化。

---

<a id="q4"></a>

### **4. VAE Encoder 具体架构是什么？128×128 图像如何一步步变形？**

#### **i. 重要前提**

本周报告记录的冻结组件为 `madebyollin/sdxl-vae-fp16-fix`，revision 为 `207b116dae70ace3637169f1ddd2434b91b3a8cd`。下表按本地 `config.json` 和 Diffusers `AutoencoderKL` 的编码器结构说明；权重身份另见历史运行记录。

SDXL VAE 公开 config 的常见核心参数：`block_out_channels=[128,256,512,512]`、`down_block_types=4×DownEncoderBlock2D`、`layers_per_block=2`、`latent_channels=4`、`norm_num_groups=32`、激活 `SiLU`、`scaling_factor=0.13025`。

#### **ii. 逐层形状（B=1，三次空间下采样，分辨率每次减半）**

```text
原始 RGB 图像（通常先归一化到 [-1,1]）
  x                          [1,   3,128,128]
  │
  ├─ Conv2d 3 → 128          [1, 128,128,128]
  │
  ├─ DownEncoderBlock #1
  │    ResNetBlock ×2        [1, 128,128,128]
  │    Downsample（边长减半）         [1, 128, 64, 64]
  │
  ├─ DownEncoderBlock #2
  │    ResNetBlock ×2        [1, 256, 64, 64]
  │    Downsample（边长减半）         [1, 256, 32, 32]
  │
  ├─ DownEncoderBlock #3
  │    ResNetBlock ×2        [1, 512, 32, 32]
  │    Downsample（边长减半）         [1, 512, 16, 16]
  │
  ├─ DownEncoderBlock #4
  │    ResNetBlock ×2        [1, 512, 16, 16]
  │    无最后下采样
  │
  ├─ MidBlock
  │    ResNetBlock → Attention → ResNetBlock
  │                             [1, 512, 16, 16]
  │
  ├─ GroupNorm → SiLU → Conv2d 512 → 8
  │                             [1,   8, 16, 16]
  │
  ├─ quant_conv (1×1 Conv，8→8)
  │                             [1,   8, 16, 16]
  │
  ├─ 沿通道维度拆成两半
  │    μ(x)                     [1, 4,16,16]
  │    log σ²(x)                [1, 4,16,16]
  │
  ├─ 当前项目选 μ(x)，不从后验采样 η
  │    h=μ(x)                   [1, 4,16,16]
  │
  └─ z₀=0.13025 × h            [1, 4,16,16]
        │
        └─ DiT patch_size=2
           Patchify          [1,64,16]
           Linear Projection [1,64,d]
```

这里 `DownEncoderBlock` 中两个 ResNetBlock 主要提取/变换特征，`Downsample` 才使图像尺寸减半；最后一级通常没有继续缩小空间分辨率。中间的 Attention 让编码器可以聚合较长距离的信息。`Conv2d 512→8` 输出 8 个中间通道；经过 `quant_conv` 后，才沿通道维拆成 4 通道均值与 4 通道对数方差。`quant_conv` 属于后验参数变换，不是量化为离散 token。

**编码器为什么输出分布而不直接输出四通道？** 因为这是 KL-VAE：训练阶段要学习近似后验 $q_\phi(h\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x)))$，并可用重参数化 $h=\mu_\phi(x)+\sigma_\phi(x)\odot\eta$ 采样；MiniGen 当前仅取后验均值以便确定性缓存。

**解码过程** 大致反向：`z₀/s → [1,4,16,16] → (post_quant_conv) → Decoder 中间块/上采样 → [1,3,128,128]`；不是直接把数值 `z` 重新排列成像素。

参考：[SDXL VAE 官方配置](https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json)、[Diffusers AutoencoderKL 文档](https://huggingface.co/docs/diffusers/main/api/models/autoencoderkl)。

---

<a id="q5"></a>

### **5. Downsample 与 Upsample 分别是什么、解决什么问题？为什么 U-Net 一般先 Down 再 Up？**

#### **i. 直觉**

- **Downsample（下采样）**：降低空间分辨率 `H×W`，得到更紧凑的特征图，减少后续运算量，让深层特征以更低成本覆盖更大的原图范围；高分辨率细节可能因此丢失。
- **Upsample（上采样）**：提高空间分辨率，重新建立高分辨率特征网格，让网络能输出像素/潜变量位置级别的结果；它不等于信息无损逆变换，无法单凭插值恢复已经丢掉的细节。
- U-Net 是 **先下采样（编码全局与语义）再上采样（恢复空间布局）**，利用对应分辨率的 **skip connections（跳跃连接）** 保留浅层局部细节。

#### **ii. 具体维度例子（U-Net 内部，示意）**

```text
输入                         [B,  64,128,128]
  ↓ stride-2 Conv            [B, 128, 64, 64]
  ↓ stride-2 Conv            [B, 256, 32, 32]
  ↓ stride-2 Conv            [B, 512, 16, 16]  ← Bottleneck：较大感受野
  ↑ Upsample + Conv          [B, 256, 32, 32]  ← 结合 32×32 skip
  ↑ Upsample + Conv          [B, 128, 64, 64]  ← 结合 64×64 skip
  ↑ Upsample + Conv          [B,  64,128,128]  ← 结合 128×128 skip
输出头                       [B,   4,128,128]  （通道仅供示意）
```

注意上面的通道数/输出头是示意，**不是 MiniGen 当前 DiT 或 VAE 的准确实现**。VAE Encoder 可以只有 Down 路径；VAE Decoder 可以只有 Up 路径；DDPM 的 U-Net 通常组合成 Down → Bottleneck → Up 的对称架构。

#### **iii. 常见操作**

| 方向 | 方式 | 举例 |
|---|---|---|
| Down | `Conv2d(stride=2)` | `[B,C,128,128]→[B,C',64,64]` |
| Down | AvgPool / MaxPool | `[B,C,128,128]→[B,C,64,64]` |
| Down | Patch Merge | 邻近 patch 合并，空间 token 减少、通道增大 |
| Up | `interpolate(scale_factor=2)` + Conv | `[B,C,32,32]→[B,C',64,64]` |
| Up | `ConvTranspose2d(stride=2)` | 可学习的上采样 |
| Up | PixelShuffle | 把通道信息重新排列为更高空间分辨率 |

**为什么不是先 Up 再 Down？** 对原本高分辨率输入的典型分割/去噪任务，先 Up 会把特征网格进一步放大，计算更贵且尚无充分的高层语义特征；Down → Up 更符合“先抽象、再恢复输出空间”目标。但这不是所有网络必须遵循的普遍定律（超分辨率网络可能直接或主要上采样）。

**易错点**：`Upsample` 不保证恢复被 Down 丢失的全部信息；U-Net 中同层跳连用于提供高分辨率细节；VAE 的 Encoder/Decoder 分开看并不各自都需先 Down 后 Up；DiT 的 `patchify/unpatchify` 是 token 重排及投影流程，与一般卷积 U-Net 的 Down/Up 不完全相同。

---

<a id="q6"></a>

### **6. 从 U-Net 到 DiT，究竟改进了什么？为什么选 DiT？解释性如何？**

#### **i. 它们预测的是同一类目标，不是两种扩散原理**

无论 U-Net 还是 DiT，都可以学习 `ε_θ(z_t,t)`（或其它目标参数化，如 v-pred）；它们最主要的区别是 **去噪模型 Backbone**。

**卷积 U-Net（示意）**：

```text
z_t [B,4,16,16] ─┬─ Conv/ResBlock ────────────────┐ skip
                  ▼                                 │
               Downsample → 更低分辨率特征          │
                  ▼                                 │
               Bottleneck（可含 Attention）         │
                  ▼                                 ▼
               Upsample + Skip Features → Conv → ε̂ [B,4,16,16]
```

**DiT（示意）**：

```text
z_t [B,4,16,16]
   → Patchify(p=2) [B,64,16]
   → Linear Embedding [B,64,d]
   → 位置编码 + t 条件调制
   → 多层 Transformer (Self-Attention + MLP)
   → 投影到每个 patch 的噪声 [B,64,16]
   → Unpatchify ε̂ [B,4,16,16]
```

#### **ii. 架构与效果对比**

| 维度 | 卷积 U-Net | DiT |
|---|---|---|
| 基本单元 | 局部卷积、残差块、编码器/解码器、跳接 | patch token、Self-Attention、MLP、条件调制 |
| 全局信息 | 依靠多层下采样、感受野增大与额外 Attention | 标准自注意力可直接在 token 间交换信息 |
| 归纳偏置 | 强局部性、多尺度结构，对较少数据常有利 | 偏置较弱，统一 token 结构，擅长扩展 |
| 大规模扩展 | 可扩展，但结构较特化 | 在原始 DiT 论文中，随模型和算力提升展现良好 scaling |
| 训练和推理成本 | 通常对小图/小数据较有竞争力 | Attention 代价随 token 数平方增长，未必更省算 |
| 条件、多模态 | 可以通过 cross-attention 等加入 | token 接口较统一，更方便与文本/视觉 token 体系协同 |

**为什么 MiniGen 选 DiT**：学习 Transformer 系生成架构，复用已有 Attention/训练系统知识，便于扩展到条件生成、Flow/多模态；但不能推断它在当前 10k 图标、小模型、有限训练时间下必然优于同预算 U-Net。可将一个轻量 U-Net 设置为对照实验。

**可解释性**：注意力图、token 相似度、条件调制量可以可视化，有助于诊断“哪些位置相互作用”；但 **attention 不是严格的因果解释**，DiT 也没有天然比 U-Net 更透明。可靠解释需要结合遮挡/干预、归因或消融分析。

参考：[原始 DiT 论文](https://arxiv.org/abs/2212.09748)；[官方实现](https://github.com/facebookresearch/DiT)。论文中的较强 FID 结果来源于其特定训练和评测设置，不能直接外推到 MiniGen。

---

<a id="q7"></a>

### **7. adaLN-Zero 的六组向量是否都是 `[B,1,D]`，并对每个 Token 广播？**

#### **i. 结论**

**正确。** 对你当前用 **全局时间条件** 的教学 DiT，一个 Block 的调制网络把时间向量 `[B,D]` 投影成 `[B,6D]`，拆成六组 `[B,D]`，再 `[:,None,:]` → `[B,1,D]`，沿 token 维 `N` 广播到 `[B,N,D]`。六组不是 **六个标量**，而是 **六个长度为 D 的向量**，每个 Block 通常有独立的调制层。

#### **ii. 六组向量**

```text
时间条件 c(t) : [B,D]
         ↓ Linear(D,6D)
         [B,6D]
         ↓ chunk(6,dim=-1)
 shift_msa  [B,D] → [B,1,D]
 scale_msa  [B,D] → [B,1,D]
 gate_msa   [B,D] → [B,1,D]
 shift_mlp  [B,D] → [B,1,D]
 scale_mlp  [B,D] → [B,1,D]
 gate_mlp   [B,D] → [B,1,D]

Token features x: [B,N,D] （MiniGen N=64，D=128 或 256）
广播后：每个样本的 N 个 Token 共享同一组条件向量
```

**典型结构公式**：

`modulate(x, shift, scale) = LN(x) * (1 + scale) + shift`

`x ← x + gate_msa * MSA(modulate(x, shift_msa, scale_msa))`

`x ← x + gate_mlp * MLP(modulate(x, shift_mlp, scale_mlp))`

其中 `LN(x)` 是 `[B,N,D]`；`gate` 也是在残差支路输出上做逐通道的广播乘法。实际代码要保持先后顺序和 `LayerNorm` 的配置一致。上述公式使用的是典型 `adaLN-Zero` 的表示。

#### **iii. 很容易混淆的两件事**

1. 广播的是 **同一样本的条件调制向量**，不是把每个 Token 的特征变得相同：每个 Token 的 `x_i` 不同，Attention 交互仍可不同。
2. 调制向量由 **可学习权重** 和条件 `t` 共同生成，并非六组固定可训练的通用向量。如果条件扩展为 token-specific 局部信息，形状也可能不再是 `[B,1,D]`；此处只讨论当前全局时间条件实现。

#### **iv. 小数值例子**

对于某个 batch 元素，`N=64`、`D=128`，设一个特征通道的 `scale=0.2, shift=-0.1, gate=0.5`：该通道每个 Token 的归一化值 `u_i` 都执行 `u'_i = 1.2 u_i - 0.1`；Attention/MLP 支路输出还会乘 `0.5` 再进入残差。这说明 **调制规则相同，实际 Token 输出仍不同**。

**可验证代码**：

```python
B, N, D = 2, 64, 128
x = torch.randn(B, N, D)
cond = torch.randn(B, D)
modulation = torch.nn.Linear(D, 6 * D)
shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
    p[:, None, :] for p in modulation(cond).chunk(6, dim=-1)
)
assert shift_a.shape == (B, 1, D)
assert (x * (1 + scale_a) + shift_a).shape == (B, N, D)
```

---

<a id="part-b"></a>

## **三、潜变量缓存、数据身份与可复现性**

> **本篇主线：** 区分数据实体、缓存元信息、身份比对和完整性校验；路径存在不代表缓存可复用。

<a id="q8"></a>

### **8. 缓存中除了潜变量，还要保存什么？数据身份校验是怎么执行的？**

#### **i. 缓存实际保存的内容和其必要性**

**A. 实体数据**：每张图的 `z₀`，例如 `[4,16,16]`；样本顺序与对应的文件/样本 ID；必要时分片（shards）而不是一个巨大文件。

**B. 输入数据身份**：图片文件内容哈希（不仅是文件名）；manifest 文件内容哈希；每个样本的 ID、顺序、数据集版本与 train/validation split；需要时记录 `group_id`。

**C. 编码器身份**：VAE 模型名字、revision/commit、**权重文件内容哈希**、配置；如果改过自定义 VAE 代码，最好再有实现版本。

**D. 编码约定**：后验均值还是随机采样；`s=0.13025`、RGB/范围 `[-1,1]`、resize/crop/归一化、图像大小；保存的张量 dtype，例如 fp32。

**E. 缓存完整性与恢复状态**：分片清单、单片样本数、完成标志、总条数、索引、可能的分片校验和。`complete` 必须对应真实可用的完整分片；不能仅因存在目录就当作缓存已完成。

> 不是说每次要把全部输入图片或整份 VAE 权重复制进缓存！通常是保存 **身份指纹、必要元数据以及潜变量实体**，原始文件仍在各自目录。

#### **ii. 什么叫“数据身份校验”？**

对当前训练要求计算一份预期身份 `expected_identity`，读取旧缓存之前，和缓存索引里记录的 `stored_identity` 进行比较：

```text
当前任务期望：
  manifest_hash=M2, vae_weights_hash=W1, scale=0.13025,
  posterior=mean, preprocessing=P1, split=train
                │
                ▼
        calculate expected_identity
                │
                ▼
缓存目录 index.json：stored_identity
                │
       ┌────────┴────────┐
       ▼                 ▼
   完全匹配              不匹配
       │                 │
   样本数/分片          拒绝复用，要求
   完整性再检查         重新生成或报错
       │
   读取 z₀
```

#### **iii. 结合 Notebook 的具体代码**

```python
cache_reports[split] = build_latent_cache(
    adapter, rows, ROOT / "data", hash_file(manifest), directory,
    split=split, batch_size=..., shard_size=..., max_seconds=...,
)
expected_identity = latent_cache_identity(
    adapter, rows, hash_file(manifest), split
)
cache_datasets[split] = CachedLatentDataset(directory, expected_identity)
```

- `hash_file(manifest)`：对 manifest **文件内容** 计算 SHA256，变更内容时通常得到不同哈希值。
- `latent_cache_identity(...)`：结合当前 VAE、样本记录和 split 构造期望身份；具体字段须以项目函数的真实实现为准。
- `CachedLatentDataset(directory, expected_identity)`：读取缓存时传入期望身份，使加载逻辑能够检查与既有缓存是否匹配；对比了哪些字段、是否逐分片重算内容校验和，要读具体实现确定。
- `load_manifest(..., verify_hashes=True)`：在源图片阶段核对记录的文件内容哈希；这与读缓存时的身份比对是 **不同层次的验证**。

**特别区别**：SHA256 使数据变化可被发现的概率极高，然而如果系统只检查一个字符串、并未真的验证图片及分片内容，它不能自动保证所有数据无损；必须看校验位置和覆盖范围。

---

<a id="q9"></a>

### **9. 缓存身份信息能防止哪些错误复用？**

**缓存的身份和载荷需要一起校验。**

1. 缓存就是 **保存在本地磁盘目录的预计算潜变量**，以后按需要加载；在分布式或云端项目里也可能放共享存储，并非必须本地。
2. 保存缓存时，需要同步保存其 **来源、生成方法、完整性** 的身份信息（一般写在索引/元数据文件中，不必嵌进每条 latent）。再次读取时用 **当前实验期望的身份** 去比对，匹配才复用；不匹配则重新编码或拒绝加载。**并不只是“看路径/文件名对不对”。**

#### **i. 错误复用的具体例子**

```text
周一：图像 A + VAE-v1 + 均值 + scale=0.13025
       → 缓存目录 week4/train → [latent A_v1]

周三：换成 VAE-v2（权重变化），或把图像 A 内容改了
       → 你仍然访问 week4/train

若只判断“目录存在”：
       → 错误读取旧 A_v1
       → 以为正在使用新 A_v2 训练
       → 实验数据和报告声明不一致

若有身份校验：
       → 新权重哈希/输入哈希不同
       → 拒绝使用旧缓存 → 重建新缓存
```

**它能够防止**：静默的数据陈旧、训练和评测条件不一致、用错 VAE 版本或处理方式、混用 train/validation、被不完整分片干扰的结果、错误地把实验提升归因于模型改进。它提高复现可信度，**但不等于凭一个哈希就保证所有随机性、硬件算子和训练轨迹逐位一致**。

#### **ii. 三个词分别是什么意思？**

- **可重复使用 (reuse)**：省去重复 VAE 编码时间。
- **完整性 (integrity)**：缓存未缺片、未被错误覆盖或损坏。
- **可复现性 (reproducibility)**：未来能够说清“这次训练究竟用了哪些输入及哪种生成规则”，并在条件允许时重复实验。

#### **iii. 一句话记忆**

> **缓存数据负责“拿来训练”，缓存身份负责“证明拿来的是谁、怎么来的、是否完整”。**

---

<a id="part-c"></a>

## **四、随机训练、监督目标与诊断基线**

> **本篇主线：** 从三种随机采样理解 Monte Carlo，再读懂固定拟合、验证 Probe、条件期望和四组噪声 MSE。

<a id="q10"></a>

### **10. 训练时随机抽取时间步，是否与 SGD 随机抽样类似？模型究竟拟合什么？**

#### **i. 结论与直觉**

随机时间步 `t` 是对训练目标中 **时间维度** 做 Monte Carlo 采样，与小批量 SGD 随机抽取数据样本非常类似；但 **不是单独拟合 `t` 的值**，而是学习统一的条件去噪映射 `ε_θ(z_t,t) ≈ ε`，其中 `t` 是条件输入。

对 MiniGen（离散 1000 步）而言，常见的均匀时间步采样方式是：

```python
# B 是 batch size，T=1000；每个样本独立抽 t，可以重复
idx = torch.randint(0, 1000, (B,), device=device)   # [B]
eps = torch.randn_like(z0)                          # [B,4,16,16]
zt = scheduler.add_noise(z0, eps, idx)               # 本项目的 DDPMScheduler 接口
pred = model(zt, idx)                               # [B,4,16,16]
loss = (pred - eps).square().mean()
```

这里的 `scheduler.add_noise(z0, eps, idx)` 与历史教学 Scheduler 的参数顺序一致；系数表与输入张量应位于同一设备。

#### **ii. 目标函数与随机梯度估计**

记 $p_{\mathrm{latent}}$ 为训练图像经冻结 VAE 编码并缩放后的潜变量分布。当前简化噪声预测目标为：

$$\mathcal L(\theta) = \mathbb E_{z_0\sim p_{\mathrm{latent}},\;t\sim\mathrm{Unif}\{0,\ldots,T-1\},\;\epsilon\sim\mathcal N(0,I)}\bigg[\frac1{CHW}\lVert\epsilon_\theta(z_t,t)-\epsilon\rVert_2^2\bigg],$$

$$z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

对 batch 大小为 `B` 的一次更新，随机抽取 `B` 个 `(z0_i,t_i,eps_i)`，用 `B` 个损失的均值近似整体期望。采样机制符合目标分布且具备必要条件时，批损失/梯度是目标期望/梯度的无偏 Monte Carlo 估计；有限 batch 仍有方差。

**为什么不穷举？** 10,000 张图 × 1,000 个离散时间步已经有一千万个 `(图, t)` 组合；对每一组还能连续采样无数个 `ε`。随机采样能把计算控制在每次一个小批量。

#### **iii. 与固定监督实验的差异**

| 项目 | 正常训练 | `fixed_batch=fixed_probe` 的固定拟合 |
|---|---|---|
| 干净潜变量 `z0` | 批次抽取、随训练变化 | 首四张始终不变 |
| 时间 `t` | 每个样本随机抽取 | 四个固定时间 `99,399,699,999` |
| 高斯噪声 `ε` | 每步新采样 | 每个样本固定不变 |
| 目的 | 估计整体噪声预测目标，学习泛化去噪 | 验证小任务能否过拟合，排查训练链路 |
| 固定 Probe loss 接近零 | 不能单独证明泛化 | 只能证明拟合固定监督集合 |

**关键易错点：** 时间步是模型的条件，不是一个要拟合的标签；模型共享参数以处理多个噪声等级。随机时间采样也不要求每次 batch 覆盖所有时间步，时间标签允许重复。均匀抽样是本案例的简化目标；若采样分布改变而未做权重修正，通常相当于改变了时间步的优化权重。训练优化器的 `step=100` 与扩散时间步 `t=999` 完全是不同概念。

---

<a id="q11"></a>

### **11. 图像、时间步、噪声是否相互独立采样？是两个 Batch 吗？**

**结论：** 通常是一个图像 Batch，以及与之 **逐样本对齐** 的时间步向量和独立噪声张量，不必理解为两个数据 Batch。这里将 $z_0$ 视为来自训练图像经 VAE 编码与缩放后的分布，代码索引 $t$ 从 $\{0,\ldots,T-1\}$ 均匀采样，$\epsilon\sim\mathcal N(0,I)$ 与图像和时间独立；在非均匀时间步采样、分层抽样或特殊训练方案中，采样设定可以不同。

```text
z0 : [B,4,16,16]  ← 从训练缓存抽 B 张
 t : [B]          ← B 个时间步，每个样本一个，可重复
 ε : [B,4,16,16]  ← 每个样本一份独立的标准高斯噪声
 zt = sqrt(abar[t]) * z0 + sqrt(1-abar[t]) * ε
 pred = DiT(zt,t) → [B,4,16,16]
 loss = mean((pred - ε)**2)
```

数学上优化三重期望：

$$\mathcal L(\theta)=\mathbb E_{z_0,t,\epsilon}\left[\frac1{CHW}\lVert\epsilon_\theta(z_t,t)-\epsilon\rVert_2^2\right].$$

同一迭代的三种随机量共同构造 **一批有标签的去噪监督样本**，而不是分别训练两个网络。要区分训练优化步数 `global_step` 和扩散时间步 `t`。

---

<a id="q12"></a>

### **12. `train_latent_denoiser(...)` 的输入参数逐项说明**

#### **i. 函数调用（来自本周固定监督 Notebook）**

```python
fixed_result = train_latent_denoiser(
    fixed_model,
    scheduler,
    train_dataset.latents[:4],
    fixed_config,
    data_identity={"parent": train_dataset.data_id, "subset": "first_4"},
    model_identity=model_identity,
    batch_size=4,
    probe=fixed_probe,
    fixed_batch=fixed_probe,
)
```

#### **ii. 每个输入究竟是什么？**

| 输入 | 类型/示例 | 含义与在本实验的作用 |
|---|---|---|
| `fixed_model` | `TeachingDiT(depth=2, hidden=128, heads=4)` | **被更新参数的去噪网络**；输入 `(z_t,t)`，输出与 `ε` 同形状的预测噪声 `[B,4,16,16]`。与用于规格检查的 6 层主模型不是同一个对象。 |
| `scheduler` | `DDPMScheduler(num_steps=1000, ...)` | **噪声时间表与加噪所需系数**，提供 `α_t`、`ᾱ_t` 等，构造 `z_t`；不是学习率调度器（LR Scheduler）。本项目实际传入的是 project_core 的 DDPM scheduler。 |
| `train_dataset.latents[:4]` | Tensor `[4,4,16,16]` | **可供训练的干净潜变量 `z0` 数据**，已用冻结 VAE 编码并乘缩放系数。此实验只取前四张。 |
| `fixed_config` | `TrainConfig(...)` | **训练控制参数**，例如输出目录、学习率、`max_steps=100`、最大运行时长、设备、随机种子、checkpoint 间隔、scheduler 来源等。 |
| `data_identity=...` | `{"parent": data_id, "subset": "first_4"}` | **数据来源/身份元信息**，表明这次训练使用哪个缓存及其前四个样本；便于实验追溯、必要时的恢复/一致性检验。它不是输入 DiT 的张量。 |
| `model_identity=model_identity` | 字典，如 source、源码 SHA、architecture | **模型实现与配置的身份元信息**，用于确认实验依赖何种 `TeachingDiT` 源码和超参数；不是可训练权重，也不是去噪条件。 |
| `batch_size=4` | 整数 `4` | **每次优化的样本数量**；固定拟合中恰好等于训练集合的四个潜变量。和扩散时间步个数 `T` 无关。 |
| `probe=fixed_probe` | 三元组 `(z0, ε, t)` | **固定诊断样本**，用于训练前/后在完全相同输入上测量 `probe_loss`，方便排除不同随机噪声带来的损失波动。 |
| `fixed_batch=fixed_probe` | 同一三元组 `(z0, ε, t)` | **指定训练使用的固定监督数据**，使每次优化都看到相同图像、时间、噪声；用于“能否过拟合”的链路检查，而非正常随机训练。 |

#### **iii. 对 `fixed_probe` 的形状逐个追踪**

```python
fixed_noise = torch.randn((4, 4, 16, 16), generator=torch.Generator().manual_seed(SEED + 1))
fixed_times = torch.tensor([99, 399, 699, 999])  # Notebook 的配置示例
fixed_probe = (
    train_dataset.latents[:4].clone(),  # z0: [4,4,16,16]
    fixed_noise,                       # ε:  [4,4,16,16]
    fixed_times,                       # t:  [4]
)
```

```text
(z0, ε, t)  ──> scheduler 加噪 ──> z_t: [4,4,16,16]
                                       │
                     DiT(z_t,t) ──> ε_hat: [4,4,16,16]
                                       │
                    mean((ε_hat-ε)^2) ──> scalar loss
                                       │
                        backward + optimizer.step()
```

#### **iv. `probe` vs `fixed_batch`：最关键的区分**

- `probe` 的职责是 **测量**：在固定的一组三元组上评估模型拟合程度，获得 `initial_probe_loss`、`final_probe_loss`。
- `fixed_batch` 的职责是 **改变训练数据的选择方式**：强制训练重复使用固定三元组，不再重新采样 `t/ε`。
- 本实验把它们设为 **同一对象**，所以 `final_probe_loss≈0.000596` 是 **训练集内的固定样本拟合结果**，不是独立验证集成绩；不能据此证明随机噪声去噪泛化，更不能证明图像生成质量。
- 到常规随机训练阶段，原则上 **不提供 `fixed_batch`**，让数据、时间步、噪声按正常训练策略采样；`probe` 仍可保留作为固定诊断对照。两项默认都为 `None`；提供 `probe` 不会让它参与优化。

#### **v. 本次输出怎样解读？**

```text
final_step          = 100         # 进行 100 次优化更新，不是 t=100
initial_probe_loss  = 1.02349
final_probe_loss    = 0.00059556  # 固定诊断集合上拟合很好
elapsed_seconds     ≈ 1.027
stop_reason         = max_steps   # 达到配置的上限，正常结束
```

**易错点：** `scheduler` 不是 optimizer 的学习率 scheduler；`data_identity` 和 `model_identity` 是可追溯性元数据，不是模型输入；`probe` 不等于验证集；`fixed_batch` 不是正常随机 DDPM 训练的默认模式。历史训练包装层还会绑定实际张量摘要、日程和代码身份，恢复时拒绝契约不匹配的运行。

---

<a id="q13"></a>

### **13. 独立 Validation Probe 是验证集吗？为什么按四个时间区间诊断？**

需要区分完整 `validation dataset`（1000 张）与 **固定诊断子集** `validation probe`（当前 64 张；每桶 16 张）。固定 `z0、t、ε` 可在不同训练时刻比较同一套题，减轻验证噪声自身变化造成的扰动；它不用于梯度更新。

| 时间桶（零基索引） | 固定验证数 | 用途 |
|---|---:|---|
| `[0,100)` | 16 | 低噪声去噪能力 |
| `[100,400)` | 16 | 中低噪声 |
| `[400,700)` | 16 | 中高噪声 |
| `[700,1000)` | 16 | 高噪声 |

**边界：** 四桶都有样本 ≠ 覆盖所有 1000 个时间点；每桶样本很少，估计不稳定。桶宽不同（100/300/300/300），四桶 MSE 的算术平均不等于均匀时间抽样下的全时间轴 MSE（除非调整权重）。频繁根据同一 Probe 挑超参也可能对验证子集产生间接过拟合；最终还要用更大且独立的评估及生成质量指标。对于扩散模型，验证噪声 MSE 不等同于图片 FID/视觉质量。

---

<a id="q14"></a>

### **14. 高斯矩基线如何推导？协方差为何等于 √(1-a)？条件均值和 ε_moment 又是什么？**

#### **i. 前提与记号**

在一个固定时间步，写 `a = ᾱ_t`，考虑一个潜变量通道中的一个元素（估计器在该通道的所有样本、空间位置共用统计）：

$$Z_t=\sqrt a\,Z_0+\sqrt{1-a}\,\varepsilon,\quad \mathbb E[Z_0]=\mu,\quad \operatorname{Var}(Z_0)=v, \quad\varepsilon\sim\mathcal N(0,1),\quad Z_0\perp\varepsilon.$$

#### **ii. 均值**

利用期望线性：

$$\mathbb E[Z_t]=\sqrt a\mathbb E[Z_0]+\sqrt{1-a}\mathbb E[\varepsilon]=\sqrt a\mu.$$

#### **iii. 方差**

利用独立性 $\operatorname{Cov}(Z_0,\epsilon)=0$ 和 $\operatorname{Var}(cX)=c^2\operatorname{Var}(X)$：

$$\operatorname{Var}(Z_t)=a\operatorname{Var}(Z_0)+(1-a)\operatorname{Var}(\varepsilon)=av+1-a.$$

#### **iv. 协方差（关键手算）**

定义 $\operatorname{Cov}(X,Y)=\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])]$。用协方差的线性性：

$$\begin{aligned} \operatorname{Cov}(\varepsilon,Z_t) &=\operatorname{Cov}\left(\varepsilon,\sqrt a Z_0+\sqrt{1-a}\,\varepsilon\right)\\ &=\sqrt a\,\operatorname{Cov}(\varepsilon,Z_0) +\sqrt{1-a}\,\operatorname{Cov}(\varepsilon,\varepsilon)\\ &=\sqrt a\cdot0+\sqrt{1-a}\cdot\operatorname{Var}(\varepsilon)\\ &=\sqrt{1-a}. \end{aligned}$$

**直觉：** `Zt` 含有一部分 `ε` 本身，因此二者必然相关；噪声在 `Zt` 中的系数正是 `√(1-a)`。**Cov(ε,Z0)=0 的前提是噪声与干净数据独立**。

#### **v. 从最优线性回归到 ε_moment**

用简单仿射模型预测噪声：$\hat\epsilon=c(Z_t-\mathbb E[Z_t])$。最小化 `E[(ε-ε̂)²]` 对 `c` 的导数，得到：

$$\frac{\partial}{\partial c}\mathbb E[(\varepsilon-c(Z_t-\mathbb E Z_t))^2]=0 \Rightarrow c^*=\frac{\operatorname{Cov}(\varepsilon,Z_t)}{\operatorname{Var}(Z_t)} =\frac{\sqrt{1-a}}{av+1-a}.$$

因此，**ε_moment（Gaussian Moment Baseline）** 就是这个不训练的、只依赖训练集一阶/二阶统计量的预测：

$$\boxed{\hat\varepsilon_{\text{moment}}(Z_t,t)=\frac{\sqrt{1-a}}{av+1-a}(Z_t-\sqrt a\mu)}.$$

MiniGen 的 `train_mu/train_var` 为 `[1,4,1,1]`，四个通道分别计算均值和方差，时间步对应的 `a` 为 `[B,1,1,1]`，因此广播到 `[B,4,16,16]`。这只是 **逐通道、逐位置的标量仿射预测**，忽略跨位置/通道的图像关联；不是对全图最强的线性预测器。

#### **vi. 为何出现条件均值 E[ε|Zt]？**

因为对给定输入 `Zt=z`，所有可测函数中最小化 **平方误差** 的最优预测器是条件均值：

$$f^*(z,t)=\mathbb E[\varepsilon\mid Z_t=z,t].$$

`E[ε|Zt]` 的意思是“已经观察到某个带噪潜变量后，对它背后的真实加噪噪声求平均/最佳 MSE 估计”，**预测器不能把真实噪声当输入；训练代码生成它作为监督标签**。如果 **这里讨论的单个标量** `Z0` 为高斯，且与 `ε` 独立，那么 `(ε,Zt)` 联合高斯，**这个标量条件均值** 恰好等于上面的线性表达式。**向量情况下还需注意协方差结构**：即便整个 `z0` 服从多元高斯，若通道/空间位置之间有相关性，真正的条件均值一般涉及 **完整协方差矩阵**，不一定等于 Notebook 的逐通道标量公式。实际图标潜变量往往非高斯、有复杂空间依赖，所以真实最优条件均值可能是非线性的，DiT 有机会超过 `ε_moment`。

**例子：** `μ=0, v=1, a=0.36`，有 `Var(Zt)=1`、`Cov(ε,Zt)=0.8`，于是 `ε̂_moment=0.8Zt`。这是“根据观察到的混合信号估计噪声”，不是用标签替代预测。

**极端情况：** `a→0` 时 `Zt≈ε`，矩估计器 `ε̂_moment≈Zt`，预测 MSE 可以非常小，但它不一定理解图像结构。

---

<a id="q15"></a>

### **15. 如何解读零预测、矩基线、DiT 与 EMA 的分桶 MSE？**

#### **i. 统一评价标准**

全部方法都在 **同一批固定验证数据** `(Z0,ε,t)` 上测试，预测同一个 **真实噪声 ε**，按时间桶计算：

$$\mathrm{noise\ MSE}=\operatorname{mean}_{b,c,h,w}(\hat\varepsilon-\varepsilon)^2.$$

**MSE 越小，表示对这一组给定带噪潜变量的噪声预测越准确。** 不是分类准确率、图片重建 MSE、FID，也不是生成质量保证。

| 列名 | 具体输出 `ε̂` | 用途/可以证明什么 |
|---|---|---|
| `zero` 零预测 | 全部输出 `0` | **最低信息基准**，因为 `Var(ε)=1`，理论 MSE 约 1；打败它不代表学到复杂结构。 |
| `moment` 高斯矩基线 | `√(1-a)/(av+1-a)·(zt-√a μ)` | 不训练模型，仅用训练集逐通道均值/方差的估计；对照网络是否利用了更多关系/结构。 |
| `model` / `DiT` | 普通模型参数 `θ` 输出 | 当前训练后模型的噪声预测能力。 |
| `EMA` | 指数移动平均参数 `θ_EMA` 组成的同构 DiT 输出 | 比较参数平滑后是否在 **这一套验证 Probe** 上更好；不是另一种标签或另一个损失。 |

#### **ii. 本次实际结果（越小越好）**

| 时间段 | zero | moment | DiT | EMA | 这组最低者 |
|---|---:|---:|---:|---:|---|
| `0–100` | 1.0126 | 0.9268 | 0.4357 | **0.4156** | EMA |
| `100–400` | 1.0021 | 0.3525 | 0.1384 | **0.1343** | EMA |
| `400–700` | 0.9901 | 0.0154 | 0.0161 | **0.0138** | EMA |
| `700–1000` | 1.0044 | **0.000463** | 0.005458 | 0.002903 | moment |

**具体怎么读一行：** 在 `0–100` 这 16 个验证样本中，直接猜 `ε=0` 的 MSE ≈1.0126；统计基线 ≈0.9268；原始 DiT ≈0.4357；EMA DiT ≈0.4156，因此 **这组探针上** EMA 最准，且 MSE 低于简单统计估计。

**怎么读高噪声一行：** `700–1000` 时 `Zt≈ε`，所以 moment 依靠一个很简单的缩放/平移修正就非常准确，MSE 仅 0.000463；EMA =0.002903 更差（约为 moment 的 6.27 倍），说明 **在该 Probe 上，高噪声预测尚不及简单矩基线**，不代表 DiT 完全没学到去噪规律，也不意味着 moment 能替代一个完整的多时间步生成模型。

**结论边界：** 每桶 16 个固定样本可以提供哪些噪声阶段表现较差的线索；不能用这一张表证明真实图像生成质量、总体泛化的统计显著性。模型选择最好结合独立的大规模评估与真实反向采样。MSE 在不同时间段的“绝对低”不代表该时间段生成重建更容易。

#### **iii. 矩基线的作用**

`moment` 是不需要训练、从训练潜变量统计矩得到的 **数值基线（baseline）**，而不是随机对照组，也不是验证集的真实标签。`zero` 是更弱的零输出基线；`DiT` 与 `EMA` 是两套（当前参数/滑动平均参数）模型预测。四者都在 **相同的验证 `(z0, t, ε)` 样本** 上预测真实噪声 `ε`，统一计算 MSE。比较 `DiT/EMA` vs `moment`，检验学习到的网络是否在当前时间桶中超出简单逐通道线性统计估计；比较 `EMA` vs `DiT`，检验参数滑动平均的效果。注意 **Moment 特别强的高噪声端并不意味着它能独立替代 DiT 生成图像**。

---

<a id="q16"></a>

### **16. Monte Carlo、MMSE、EMA 和 Checkpoint 各解决什么问题？**

这四个概念对应训练估计、最优预测、参数平滑和恢复训练。

| 概念 | 解决什么 | 关键公式 / 对应 Q&A |
|---|---|---|
| Monte Carlo | 用抽样图像、时间步、噪声近似整体 DDPM 期望与梯度 | $L=\mathbb E_{z_0,t,\epsilon}[\lVert\epsilon_\theta(z_t,t)-\epsilon\rVert^2/(CHW)]$；**Q10/Q11** |
| MMSE | 固定输入 `(zt,t)`，平方损失下的理论最优预测是什么 | $\epsilon^*(z_t,t)=\mathbb E[\epsilon\mid z_t,t]$；**Q14/Q22** |
| EMA | 让模型权重沿训练轨迹指数平滑，供评估/推理选择 | $\theta^{\mathrm{EMA}}_k=d\theta^{\mathrm{EMA}}_{k-1}+(1-d)\theta_k$；**Q17/Q18** |
| Checkpoint | 将可持续训练所需状态保存并恢复 | model、optimizer、EMA、RNG、data cursor、噪声日程等；**Q19** |

**易错点：** Monte Carlo 指目标期望的抽样估计；MMSE 指平方损失下最优估计的性质；EMA 不替代优化器；Checkpoint 不只是保存模型权重。

---

<a id="part-d"></a>

## **五、EMA、Checkpoint 与实验归档**

> **本篇主线：** 两套模型参数如何并行维护？怎样做到可恢复训练？哪些文件负责恢复、哪些文件负责解释结果？

<a id="q17"></a>

### **17. EMA 是对哪个 θ 平滑？为什么 1-d 很小？是权重还是更新增量？**

#### **i. 参数层面的定义**

$$\boxed{\theta^{\mathrm{EMA}}_k=d\theta^{\mathrm{EMA}}_{k-1}+(1-d)\theta_k},\quad 0\le d<1.$$

- `θ_k`：**第 k 次优化器更新之后，模型当前的一整套参数值**（不是梯度、不是参数更新增量 `Δθ`）。
- `θ^EMA_{k-1}`：前一次维护的 EMA 参数值。
- 对模型中所选择的每个浮点参数张量，**逐元素执行** 此公式；不需要把所有参数混合成一个数字。
- 冻结 VAE 不参加 DiT EMA：本项目 EMA 应针对待训练的 DiT。具体实现可能复制整个 DiT 参数和某些 buffer；不训练参数也可能被原样保留，是否处理 buffer 要看实现。

#### **ii. 一次手算**

`θ_EMA_old = 2.0, θ_current = 3.0, d = 0.9`：

$$\theta_{EMA,new}=0.9\times 2+0.1\times 3=2.1.$$

EMA 不会立刻跳到当前权重 3，而是 **缓慢跟随**。把公式展开：

$$\theta^{EMA}_k=d^k\theta^{EMA}_0+(1-d)\sum_{j=1}^k d^{k-j}\theta_j.$$

历史越近的参数权重越大，越久以前权重指数衰减。`d=0.99` 约对应 100 次更新的有效平滑尺度，`d=0.999` 约对应 1000 次更新（仅作时间尺度直觉，不是严格同权窗口）。

#### **iii. 为什么 1-d 通常很小？**

SGD/AdamW 每次受到不同小批量和随机时间步/噪声影响，参数轨迹有噪声。让 `d` 接近 1，EMA 不会完全追随当前一次偶然更新，相当于降低参数波动；很多生成模型的采样结果会更稳定。但 **过大 d 在训练短、分布快速变化时会落后模型**，EMA 并不必然优于原始参数。初始化通常从 `θ_EMA=θ` 开始，并在 optimizer.step 之后更新。

**区别于 AdamW 的一阶动量：** AdamW 平滑的是 **梯度/梯度平方** 来确定参数更新；EMA 平滑的是 **更新完成后的参数值本身**，常用于推理评估，不负责正常参数的反向传播更新。

---

<a id="q18"></a>

### **18. EMA 和优化器究竟更新哪套参数？原参数和 EMA 是两条并行状态吗？**

通常 `optimizer.step()` 只更新原模型的可训练参数 `θ_k`；EMA 参数 `θ^EMA_k` 是另一套状态，直接执行指数平滑，不参加通常的梯度优化。EMA 的衰减系数是超参数。

$$\theta_k=\mathrm{AdamWStep}(\theta_{k-1},g_k,\text{optimizer state}),\qquad \theta^{\mathrm{EMA}}_k=d\theta^{\mathrm{EMA}}_{k-1}+(1-d)\theta_k.$$

这里 `θ_k` 是 **optimizer.step 结束后的完整参数值**，不是权重增量 `Δθ`；对于每个可训练参数张量逐元素应用。冻结 VAE 不需要参加训练 DiT 的 EMA。常以 `θ^EMA_0=θ_0` 初始化，且可设计 warm-up、不同 EMA 更新频率，具体以实现为准。

**数字例子：** `d=0.9`，`θ_0=θ^EMA_0=10`，原参数更新到 12 时，EMA 为 `0.9×10+0.1×12=10.2`；下一次原参数变为 9，EMA 为 `0.9×10.2+0.1×9=10.08`。EMA 会滞后但减少短期波动；`d` 过于接近 1 且总训练步少时可能更新不足。

**使用与存储：** 训练通常 `model(θ)` 做 forward/backward；验证/生成可分别比较原始参数和 EMA，当前 Week 4 的 `sample_ddpm(ema_model,...)` 用 **EMA** 推理。完整恢复时保存原模型参数、优化器状态、EMA 状态（以及更新计数等相关配置），因为两套数值不同。参见 Q17（EMA 公式）与 Q19（完整 Checkpoint）。

---

<a id="q19"></a>

### **19. 完整 Checkpoint 要保存什么？需要保存 Grad 吗？**

#### **i. Checkpoint 的目标**

**普通权重保存只支持加载模型；精确恢复训练需要保存“下一步继续计算所需的状态”。**

| 类别 | 必要/条件 | 具体保存什么 | 不保存可能造成的影响 |
|---|---|---|---|
| 模型参数与缓冲区 | 必要 | `model.state_dict()` | 模型不一致 |
| 优化器状态 | 必要 | `optimizer.state_dict()`（AdamW 一阶矩、二阶矩、step 等） | 后续更新轨迹变化 |
| EMA 状态 | 使用 EMA 时必要 | EMA 参数及相关更新计数/设置 | 推理权重或后续 EMA 轨迹变化 |
| 全局训练步/epoch | 必要 | `global_step`、`epoch` | 训练位置、调度、日志不一致 |
| 学习率调度器 | 有则必要 | `lr_scheduler.state_dict()` | 学习率接续不一致 |
| RNG 随机状态 | 精确恢复必要 | Python `random`、NumPy、PyTorch CPU、以及相关 CUDA/MPS generator 状态 | 噪声、时间步、dropout 等变化 |
| 数据顺序与位置 | 精确恢复必要 | sampler/shuffle RNG、epoch、data cursor、当前顺序/足以重建的状态 | 恢复后读错 batch |
| 扩散噪声日程 | 必要 | `T`、beta/alpha_bar 数值表或可验证的完整配置、预测类型 | 同一时间步噪声强度变化 |
| 模型/数据身份 | 必要（实验可信） | VAE 来源/权重哈希、潜变量缓存身份、DiT 架构与实现哈希、数据划分 | 错模型、错数据被静默复用 |
| 训练超参/配置 | 必要（实验可信） | LR、batch、accumulation、损失权重、种子、精度等 | 不知道自己恢复的是什么实验 |
| AMP GradScaler | 使用 fp16 AMP 时必要 | scaler 状态 | 动态损失缩放接续不一致 |
| DDP/ZeRO 分片状态 | 使用时必要 | 完整或分片模型、优化器状态和映射元信息 | 多卡恢复错误 |
| 已累计的 `.grad` | **通常不需要** | 见下方例外 | 取决于保存边界 |

#### **ii. 为什么通常不用保存 grad？**

正常一个优化步骤一般是：

```text
zero_grad() → forward → loss.backward() → optimizer.step()
                                    ↓
                            EMA 更新（如有）
                                    ↓
                          checkpoint（在完整 step 边界）
```

当 checkpoint 在 **一个完整优化更新结束、下一次梯度累积开始之前** 保存时，上一轮梯度已经完成用途（下一步会清空或覆盖），**下次前向/反向会重新计算新梯度**。因此常规 `state_dict()` 不保存 `.grad`，也不妨碍正常恢复。

**必须记住的例外：** 若使用 `gradient_accumulation_steps>1` 且在 **累积中途** 保存（例如已经反传了微批 1、2，还没凑够全部微批进行 optimizer.step），这时 `.grad` 中含有尚未完成的累计梯度。若要 **严格从该微批继续**，必须保存这些累计梯度、累积计数以及 AMP/scaler 等相关状态；否则应该只在完整优化器更新边界保存，或恢复时重放微批以重构梯度。

**边界：** CPU 同设备同实现下可做逐位一致性实验；MPS/CUDA、异步计算、算子不确定性可能导致即使状态恢复正确也非逐位一致。`resume_mode=stateful_exact` 是训练器宣称的模式，不能替代连续训练 vs 中断续训的实测比较。

---

<a id="q20"></a>

### **20. 实验归档需要保存哪些类别？为什么这里不另存模型权重和 Optimizer State？**

#### **i. 区分两种不同工件**

- **Checkpoint（二进制，如 `latest.pt`）**：是为了 **恢复训练/加载权重**，真正包含 Model State、AdamW Optimizer State、EMA、Step、RNG、Data Cursor、LR Scheduler/GradScaler（如使用）、配置和身份契约等。见 **Q19**。
- **实验归档（结构化数据，如 `week4_results.json` + `figures/*.png`）**：是为了 **追溯、比较、解释和汇报实验**，通常记录身份、配置、指标、统计结果、文件路径和局限；**不应把巨量权重和 Optimizer Tensor 全部塞进 JSON**。但要留下对应 Checkpoint 的路径/哈希/版本以便回溯。

| 归档类别 | 本次典型字段/文件 | 目的 |
|---|---|---|
| 实验身份与环境 | 日期、`device`、`precision`、Python/PyTorch、随机种子、Git commit（若存在） | 追溯运行条件 |
| 模型实现身份 | `model_source`、`reference_sha256`、模型架构、真实训练参数量、仅验证的主规格参数量 | 防止把形状验收误写为模型训练 |
| 数据与缓存身份 | `manifest_sha256`、数据划分/数量、缓存身份/完整性、VAE 权重/缩放约定 | 确认训练输入来源 |
| 训练与优化配置 | Batch、LR、优化器、EMA decay、max_steps、噪声 Schedule、Checkpoint 间隔 | 复现实验契约 |
| 训练结果与恢复 | Fixed/Random Loss、最终 Step、耗时、stop reason、500→1000 Resume、严格一致性测试位置 | 验证训练闭环 |
| 验证与对照组 | 固定 Probe 的分桶 Zero/Moment/DiT/EMA MSE | 定位噪声区间差异 |
| 生成及可视化 | 完成情况、模型调用数、采样种子、生成范围、图像网格、轨迹、重建对照 | 看最终生成链和质量 |
| 失败/记忆诊断 | Parser 失败数、对象数、潜变量最近邻、生成与训练统计差别 | 提供可追踪的问题线索 |
| 资源与预算 | 缓存/训练/采样/解码耗时、设备、云成本 | 比较成本效率 |
| 结论边界 | 无文本条件/CFG、小教学模型、单种子、主模型未训练、规则解析不等于生成质量等 | 不过度宣称成果 |
| 产物索引 | 报告 JSON、图表路径、Checkpoint 路径和文件哈希（建议补充） | 让结果可复查 |

**易错：** 写了 `resume_mode='stateful_exact'` 并不等于已经证明 CPU 连续/恢复逐位一致，仍需要独立对照测试。归档用于解释结果；Checkpoint 用于恢复实际训练状态。

---

<a id="part-e"></a>

## **六、DDPM 反向采样、DDIM 跳步与推理**

> **本篇主线：** 区分训练时间轴 T 与采样步数 K；从真实后验到近似反向分布，再理解 DDIM 复用的条件。

<a id="q21"></a>

### **21. 扩散终点为什么要接近标准正态？训练时间步 T 与 ComfyUI 的生成步数 K 有什么区别？**

#### **i. 直觉与准确结论**

- **训练终点与推理起点要匹配。** 对以标准高斯为推理起点的 DDPM，关键是噪声日程使训练终点残余信号足够小；不能只看日程长度。
- `T`：训练过程中定义的噪声等级/时间轴（如 `T=1000`）；`K`：推理时调用采样更新规则的次数（如 `K=20`、`50`）。**训练可以 T=1000，推理 K=20**。
- 简单把 `DDPMScheduler(num_steps=1000)` 改成 `num_steps=20`，且保持相同 beta 起止值，也会 **重新定义噪声调度**，不等同于“在原来的 1000 步时间轴上做 20 步跳步采样”。
- 对一个离散 DDPM 来说，训练的时间索引和噪声调度配置是固定约定；对连续时间扩散/Flow Matching，时间可连续表示，不存在普遍的固定 `T=1000` 上限。

#### **ii. 从分布看为什么要匹配终点**

令 `β_t` 为第 `t` 步的噪声方差，`α_t=1-β_t`，$\bar\alpha_t=\prod_{i=1}^t\alpha_i$（这里使用数学记号 `\bar\alpha_t`）。

$q(z_t\mid z_0)=\mathcal N(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)I)$，等价于：

$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon$，其中 $\epsilon\sim\mathcal N(0,I)$。

当 $\bar\alpha_T\to0$，`z_T` 中残余原信号趋于 0，边缘分布 `q(z_T)` 趋近 $\mathcal N(0,I)$；推理时从 $p(z_T)=\mathcal N(0,I)$ 抽样才与训练终点近似一致。**准确说，是让信号被足够淹没，而不是单纯追求更多离散步数。**

`T` 与 `β_t`（及整套 schedule）一起决定 `ᾱ_T`。`β_start`、`β_end` 通常是训练前选定的超参数，但不是所有扩散方法都采用线性 β，也没有普适固定的数值。`T=20` 可以用另一套 schedule 达到很低终点 SNR，但每步变化更大、优化与逆过程会不同。

#### **iii. 对照 MiniGen 的运行数字**

固定线性 `β` 从 `0.0001` 至 `0.02`：

| T | 终点 `ᾱ_T` | `sqrt(ᾱ_T)` 残余信号幅度系数 | 若 `E[z_0]=2.2`，则终点均值 |
|---|---:|---:|---:|
| 20 | 0.8167771 | 0.9037572 | 1.9882659 |
| 1000 | 0.0000403583 | 0.00635282 | 0.0139762 |

终点条件均值 $\mathbb E[z_T\mid z_0]=\sqrt{\bar\alpha_T}z_0$，边缘均值 $\mathbb E[z_T]=\sqrt{\bar\alpha_T}\mathbb E[z_0]$。T=20 时原始数据仍很显著，不能直接把其终点当纯高斯噪声。有限步下，如果训练数据并非标准高斯且终点还保留信号，边缘分布通常与标准高斯有偏差；偏差多大由日程与数据分布共同决定。

#### **iv. 为什么 ComfyUI 只需几十步？**

- 模型（如离散 DDPM 的训练）学到的是多个噪声等级下的预测函数 `ε_θ(z_t,t)`。
- 生成时由 **采样器**（DDIM、Euler、DPM++ 等）决定如何跨越噪声等级，并由 **调度器**（Normal、Karras 等）决定采样位置；与 `K` 共同决定生成路径。
- `T=1000, K=20`：仍然从原训练噪声时间轴的高噪声端出发，但只选约 20 个时间点，并使用能进行非相邻时间点更新的公式；不是把训练 T 改成 20，也不能直接将单步 DDPM 公式拿来跨越任意多个时间点。
- `K` 小通常更快，误差可能更大；`K` 大也不保证单调提高质量。某些高阶采样器的 **一次采样步** 可能调用模型多次，所以 `K` 不一定等于网络前向次数。
- 对离散时间索引 `0..T-1`、只选 **互不重复离散索引** 的方案，最多选 T 个时间点；这 **不是推理模型评估次数或数值积分步数的普适硬上限**。连续时间或可插值积分器也不适用此简单上界。

```text
[训练：时间轴 T=1000]
        t = 0, 1, ..., 999
           ↓ 训练 εθ(zt,t)
[完整 DDPM 推理：K≈1000]
  999 → 998 → 997 → ... → 0
[跳步 DDIM 等推理：K≈20]
  999 → (中间少量采样点) → 0
  使用相应跳步更新规则，而非重新训练20步模型
```

**易错点**：DDPM 推理步数 1000 ≠ 所有生成模型必需 1000 步；终点信号幅度系数 `sqrt(ᾱ_T)` ≠ “剩余信息百分比”；数值离散误差、时间步映射与初始分布匹配是不同问题。

**参考**：Song et al., *Denoising Diffusion Implicit Models* (2020/ICLR 2021)，https://arxiv.org/abs/2010.02502 ；Hugging Face Diffusers scheduler 文档 https://huggingface.co/docs/diffusers/using-diffusers/schedulers ；ComfyUI 官方采样器知识库 https://github.com/Comfy-Org/workflow_templates/blob/main/site/knowledge/concepts/samplers.md 。

---

<a id="q22"></a>

### **22. 为什么真实后验的公式能换成预测噪声？预测值明明不是真实噪声**

**核心区分：真实后验 ≠ 模型定义的反向分布。**

已知前向过程 $z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon$，真实后验 **在给定真实 `z0` 时**：

$$q(z_{t-1}\mid z_t,z_0)=\mathcal N(\tilde\mu_t(z_t,z_0),\tilde\beta_t I),\qquad \tilde\mu_t=A_tz_0+B_tz_t,\quad A_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t},\quad B_t=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}.$$

**生成时没有真实 `z0`，因此不能直接执行这个真实后验。** DDPM **定义** 一个近似反向分布：

$$p_\theta(z_{t-1}\mid z_t)=\mathcal N(\mu_\theta(z_t,t),\sigma_t^2I), \qquad \hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t)}{\sqrt{\bar\alpha_t}}, \quad \mu_\theta=A_t\hat z_0+B_tz_t.$$

通过代数化简可写成常见的直接 `ε` 参数化：

$$\mu_\theta(z_t,t)=\frac{1}{\sqrt{\alpha_t}} \left(z_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(z_t,t)\right).$$

**为什么合理？** 模型通过 MSE 拟合噪声，理论最优 $\epsilon^*(z_t,t)=\mathbb E[\epsilon\mid z_t,t]$。由线性恒等式得到 $\hat z_0^*=\mathbb E[z_0\mid z_t,t]$，从而 `μθ` 可以匹配真实边缘反向条件分布 $q(z_{t-1}\mid z_t)$ 的 **条件均值**（同一噪声日程，且没有额外裁剪干净潜变量估计时）。真实反向分布对未知的 `z0` 积分后一般可能非高斯；固定方差的模型高斯分布只是近似，不保证全分布相等。

**不准确时怎样？** 只要给一个 `εθ`，公式作为 **模型构造的计算规则** 始终能执行；但 `εθ≠ε` 时通常 `ẑ0≠z0`，$\mu_\theta\ne\tilde\mu_t$（对于该样本给定真实 `z0` 的后验均值）。网络训练提高近似质量，采样质量还取决于方差设定和误差传播。

**一句话：** 真后验告诉我们可利用的均值结构，预测噪声是对未知部分的学习性参数化；**定义一个近似反向转移**，不等于证明它与真实转移严格相同。

#### **i. 噪声误差如何影响干净潜变量估计？**

从前向式 $z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon$ 出发，真实噪声代入时（固定 `zt`）可恢复真实 `z0`：

$$z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\varepsilon}{\sqrt{\bar\alpha_t}}.$$

模型把 `ε` 换成 `ε̂`，得到 `ẑ0`。设 **噪声预测误差** `Δε=ε̂-ε`，两式相减：

$$\boxed{\Delta\hat z_0\equiv\hat z_0-z_0=-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\Delta\varepsilon.}$$

因此，对 **同一个固定的带噪状态 `zt`**，噪声误差对应的解析干净潜变量估计误差被系数 `c_t=√((1-ā_t)/ā_t)` 放大。在本周末端 `ā_T≈4.03583×10^-5`，`c_T≈157.4`；若某元素 `Δε=0.01`，则该元素的 `ẑ0` 局部误差约 `-1.574`。这是 **敏感度示例**，不是整条 1000 步采样链的误差预测。

非常重要的限制：DDPM 不是每一步都把 `ẑ0` 当作最终结果。把它代入相邻反向均值，经过代数整理得到：

$$\mu_\theta(z_t,t)=\frac1{\sqrt{\alpha_t}}\left(z_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\varepsilon_\theta(z_t,t)\right).$$

所以在 **固定 `zt` 的单个相邻更新** 中，均值对噪声误差的变化为：

$$\Delta\mu_\theta=-\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\Delta\varepsilon.$$

这个系数与上面的 `c_t` **不同**，不能据 `157.4` 就断言整条 DDPM 链每一步误差都放大 157 倍。实际生成误差还涉及反向采样噪声、下一步重新预测、分布偏移与模型整体训练质量。

---

<a id="q23"></a>

### **23. DDPM 中高斯分布记号与 `μ+σξ` 怎样等价？是不是和第一周的 DDPM 相同？**

是，同第一周 DDPM 的 **同一个反向采样思想**，只是变量从像素 `x_t` 变为潜变量 `z_t`。分布声明：

$$Z\sim\mathcal N(\mu,\sigma^2 I)$$

等价的构造采样：

$$Z=\mu+\sigma\xi,\qquad \xi\sim\mathcal N(0,I).$$

因为 `E[Z]=μ+σ×0=μ`，`Cov(Z)=σ² Cov(ξ)=σ²I`；其中 **方差的平方根才是标准差**。例如 `N(5,4)` 可由 `5+2ξ, ξ~N(0,1)` 产生。

在你的 Scheduler 中，若选择固定后验方差 $\sigma_t^2=\tilde\beta_t$，就能写：

$$p_\theta(z_{t-1}\mid z_t)=\mathcal N(\mu_\theta,\tilde\beta_t I) \quad\Longleftrightarrow\quad z_{t-1}=\mu_\theta+\sqrt{\tilde\beta_t}\xi.$$

注意这里不是在讲梯度，而是 **分布定义** 与 **从分布得到一个具体样本** 的两种表达。`ξ` 是当前反向步新采样的标准高斯随机变量；它与训练时构造 `z_t` 使用的 `ε` 不同。最后一步常设随机项为 0。

---

<a id="q24"></a>

### **24. LDM 中 `clip_denoised=False`，像素 DDPM 一定要裁剪到 [-1,1] 吗？**

**在当前 MiniGen 设置下，`clip_denoised=False` 合理。** 像素 DDPM 若把原图归一化到 `[-1,1]`，一些采样实现会裁剪 **预测的干净图像 `x̂0`** 到该范围，以减少不合理像素估计；这是常见策略，但 **并非数学上必须采用的规则**，不同模型与采样器可能不裁剪。

LDM 中 `z0=s·Encoder(x)` 是实值潜变量，没有逐元素固定在 `[-1,1]` 的保证；当前训练缓存有通道均值明显大于 1。贸然将 `ẑ0` 裁剪到 `[-1,1]` 会破坏 learned latent 的尺度和分布，影响解码。其他潜空间模型也可能有自己的阈值/归一化方案，不能推广成“所有 LDM 都绝对不能裁剪”。

**易错：** `clip_denoised` 通常裁剪 **估计干净状态**，不是强行让每一步带噪状态 `zt` 都落在 `[-1,1]`。

---

<a id="q25"></a>

### **25. DDIM 的 `z_s` 公式与 `DDPM.step` 一样吗？s、跳步位置和采样次数怎么确定？**

**DDIM（确定性，η=0）：** 在原训练噪声日程中选当前时间 `t` 与更早时间 `s<t`，从当前网络预测得到 `ẑ0`、`εθ(zt,t)`，然后

$$z_s=\sqrt{\bar\alpha_s}\,\hat z_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta(z_t,t).$$

这不是标准 `DDPM.step` 的“相邻后验均值加新噪声”公式。DDPM 通常逐相邻 `t→t-1`：$z_{t-1}=\mu_\theta+\sqrt{\tilde\beta_t}\xi$；确定性 DDIM 可 `t→s`（s 是原日程选中的任一合法较早索引，也可等于 `t-1`），**不再注入新的 `ξ`**。二者都使用 `εθ` 预测和相同 `ā` 表，但 **反向路径/更新规则不同**；即便 `s=t-1`，确定性 DDIM 更新也一般不等于带噪的 DDPM 更新。

“跳的是 **日程位置**”指直接索引 **训练时定义的** `ā_t` 或对应噪声水平表，而不是把原训练配置 `T=1000` 改成 `T=20`，也不是“省略数十次 `DDPM.step` 然后照搬那次相邻更新”。例如模型评估索引选为 `999→949→…→0`，各索引使用原 1000 步日程。这里的代码索引 0 仍是第一次加噪的等级；最后一次更新去往数学时间 0 的干净状态，可令该目标端点 $\bar\alpha_0=1$。

**K 如何选择？** 通常在推理时 **输入希望采样的步数 K**，采样器从原日程构造一组 **递减的时间索引**，步距可均匀，也可依据噪声变化非均匀；某些接口允许传自定义 timesteps/sigmas（等价于自己指定路径）。例如 K=20 时通常安排约 20 次模型评估/采样更新（具体接口可能在终点处理上有小差异）。用户也可以按“跨度”自定义索引，但不是所有采样器都要求输入固定跨度。

**为什么跳步有代价？** 使用 `εθ` 近似真实噪声，步长过大可能导致离散化/模型预测误差增大；DDIM 不是精确恢复原始逐步 DDPM 链。

---

<a id="q26"></a>

### **26. DDIM 为什么能复用 DDPM 训练的噪声预测模型？数学结构是不是同构？**

关键不是“两个反向公式一模一样”，而是二者可以共享训练时 **相同的单时间点带噪边缘分布**：

$$q(z_t\mid z_0)=\mathcal N\!\bigl(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)I\bigr).$$

原 DDPM 采用 Markov 前向链，DDIM 构造不同（一般为非 Markov）的过程与采样路径，但能保持每个时间点的 $q(z_t\mid z_0)$ 相同。以该单时刻边缘构造的简单噪声预测监督仍是 `εθ(zt,t)` 对 `ε` 的 MSE（在相匹配的日程/参数化条件下）。因此可复用 **同一套已训练的噪声预测权重**，替换反向采样算法；不要求 DDPM 和 DDIM 的联合轨迹分布相同，也不意味着它们每一步的更新公式相同。

直觉示例：训练模型已学会在噪声等级 t=999、749、499、249、0 下预测噪声，那么推理时可以只调用这些噪声等级，使用 DDIM 的 `t→s` 更新从较高噪声跳到较低噪声。

**注意：** “共享单时刻边缘”使简单噪声预测目标可复用；并不等同于“所有变分 ELBO 项完全相同”。实际模型的时间参数化、噪声水平映射必须与采样器匹配。

---

<a id="q27"></a>

### **27. 调用 `sample_ddpm(...)` 时执行的是训练还是采样？**

**这是推理采样。** 传入的是已经训练好的模型，采样过程中不更新参数。

```python
sampling = sample_ddpm(
    ema_model, scheduler, initial_noise,
    seed=sample_cfg["seed_reverse"],
    max_seconds=..., capture=sample_cfg["capture"],
)
```

- `ema_model`：**已训练后构造的 EMA DiT**，用于预测噪声；这里不做反向传播和优化器更新。
- `scheduler`：使用 **已训练对应的噪声日程**，按 `step` 生成更早时刻的潜变量。
- `initial_noise`：独立 $\mathcal N(0,I)$ 初始潜变量，形状本次为 `[8,4,16,16]`。
- `seed_reverse`：DDPM 中间随机项的可复现种子。
- `max_seconds`：运行时间预算；`capture`：保留指定时间步的快照以可视化。
- 输出 `completed=True, completed_steps=1000, model_calls=1000` 说明 **完整反向链执行成功**，不代表“刚训练了 1000 步”，更不保证生成图质量。

流程：`zT随机噪声 → [DiT预测εθ → Scheduler.step采样 z_{t-1}] × T → 生成 z0 → 逆缩放 z0/s → VAE Decoder → RGB`。训练流程另有 `z0真实潜变量 → 随机t/ε → zt → DiT → MSE → backward → AdamW`。

---

<a id="part-f"></a>

## **七、生成质量评估与失败定位**

> **本篇主线：** 重建对照用于检查 VAE 对真实图像的表示瓶颈，最近邻用于查看潜空间关系，生成潜变量统计用于定位离群点。

<a id="q28"></a>

### **28. 为什么要做 Reconstruction Control？它能区分哪些失败来源？**

#### **i. 两条路径使用同一解码器**

```text
重建对照：验证原图 x → Frozen VAE Encoder → 真实图像编码的 z0 → 逆缩放 → 同一 VAE Decoder → 重建图 x_rec
生成实验：纯高斯 zT → DiT + 反向 Scheduler → 生成 z0_gen → 逆缩放 → 同一 VAE Decoder → 生成图 x_gen
```

- 若 `x_rec` 对真实图标的形状、颜色、数量和位置恢复良好，说明 **VAE 对这些真实潜变量的重建链路具备能力**；若 `x_gen` 很差，就不应仅把失败归咎于“VAE 天生不能表示图标”。
- 但这 **不能独立证明** DiT 有 bug：仍可能涉及训练不足、噪声时间索引/反向实现、缩放约定、生成潜变量落在分布外等因素；它也不能证明 Decoder 对任意潜变量都能良好解码。
- 推荐保留 `原图/重建图/生成图` 的同配置对照，以及 VAE 重建量化指标（若可用，PSNR/SSIM/LPIPS）、潜变量统计、生成规则解析。要区分 **重建上限/编码瓶颈** 和 **潜空间建模与采样问题**。

**案例结果：** 当前 8 张 Validation Original vs Reconstruction 视觉上形状、颜色和位置基本一致；而无条件生成图存在大量碎块与空白，因此重点继续检查生成潜变量、DiT 训练量和采样链。

---

<a id="q29"></a>

### **29. 最近邻计算的 `g`、`h` 个数不同为什么还能算？上下两排和 `d` 分别是什么？**

#### **i. 张量形状：样本数不必相等，每个向量维度必须相等**

```python
G = sampling["images"].flatten(1)       # [8, 1024]：8 个生成潜变量
H = train_dataset.latents.flatten(1)     # [10000, 1024]：10000 个训练潜变量
D = torch.cdist(G, H)                     # [8, 10000]：每对距离
nearest_distances, nearest_indices = D.min(dim=1)  # [8] 和 [8]
```

- `g_i` 是第 `i` 张生成图的 **1024 维潜变量**；`h_j` 是第 `j` 张训练图的 **1024 维潜变量**。虽然生成集有 8 个、训练集有 10000 个，但每一对 `g_i, h_j` 的维度都是 1024，因此可以计算。
- `g_{ik}` / `h_{jk}` 分别表示这两个向量的第 `k` 个坐标，`k=1,...,1024`。

$$\boxed{d(g_i,h_j)=\sqrt{\sum_{k=1}^{1024}(g_{ik}-h_{jk})^2}},\quad j_i^*=\arg\min_{j=1}^{10000} d(g_i,h_j).$$

**手算：** 缩成二维，`g1=(1,2)`，训练样本 `h1=(2,2), h2=(4,6)`，距离分别为 `1`、`5`，故最近邻为 `h1`。对每个生成样本都做 10000 次比较并选最小值。

#### **ii. 图如何读**

- **上排**：第 `i` 个生成图像（生成潜变量解码）。
- **下排**：对应该生成潜变量的 **最近训练潜变量所对应的原始训练图像**。
- 标注 `d=27.8` 等，是对应两条 1024 维潜变量的欧氏距离，**不是像素距离、对象数、相似度百分比，也不是 FID**。
- 本次 8 个 `d` 为 `27.8、32.0、15.0、12.8、48.9、22.0、18.7、11.0`（图上四舍五入）。对 1024 维向量，对应逐元素 RMS 差异为 `d/√1024=d/32`，比如 `d=11` 对应约 0.344。

#### **iii. 用途与局限**

1. **辅助检查明显复制训练样本**：若图像非常相似且距离很小，值得进一步核查是否发生记忆/复现；但最近邻相似本身不等于证明记忆。
2. **辅助检查生成潜变量分布偏移**：如果生成最近邻距离明显超过真实验证集最近邻距离分布，值得怀疑生成结果离训练数据流形较远。但 **最近邻距离总会存在一个最小值**，找到最近邻不代表“足够近”。
3. **建议加对照**：计算 `验证潜变量 → 最近训练潜变量` 的距离分布，再比较 `生成潜变量 → 最近训练潜变量`；还可报告分位数与每通道/每样本异常。潜变量欧氏距离不等价于人的视觉感知，也不能单独判定生成质量或记忆。

---

<a id="q30"></a>

### **30. 实验 A：生成与训练潜变量的整体统计接近，极端值为何仍异常？**

**运行输出：**

| 集合 | mean | std | min | max | 样本数 |
|---|---:|---:|---:|---:|---:|
| 训练缓存潜变量 | 0.621103 | 1.279891 | -3.76449 | 4.87072 | 10000 张 |
| 生成潜变量 | 0.622369 | 1.354204 | -23.83480 | 14.57997 | 8 张 |

- **整体均值** 几乎相同（差约 0.00127），**整体标准差** 接近（生成约高 5.8%），因此不能武断称整批生成值整体尺度失控；但 **生成极端值远超训练集出现的范围**，值得检查局部离群点、某些通道或某张生成图的异常值。
- 总体均值、方差只刻画低阶统计，不能检验对象数量、颜色/形状、空间相关性、联合分布；两组数据分别只有 8 张生成 vs 10000 张训练，**极值尤其受样本数影响**，应避免单靠 min/max 推断生成分布。
- 小量极端潜变量有可能导致 Decoder 在个别位置产生强烈彩色碎块；但单靠全局统计不能证明所有失败由这些值导致。
- 下一步要按 **样本×通道** 比较均值/标准差、1%/99% 分位数、超过训练通道阈值的比例；建立 `真实验证潜变量 vs 训练最近邻` 距离分布参照，并控制训练/生成缩放一致性。

#### **i. 逐通道与逐生成样本统计**

| 通道 | 训练均值 | 生成均值 | 训练标准差 | 生成标准差 | 生成标准差相对变化 |
|---|---:|---:|---:|---:|---:|
| C0 | 2.2052 | 2.2042 | 0.7872 | 0.8302 | +5.5% |
| C1 | 0.1513 | 0.1702 | 0.5906 | 0.5490 | −7.0% |
| C2 | 0.9324 | 0.9275 | 0.6721 | 0.9680 | **+44.0%** |
| C3 | −0.8046 | −0.8125 | 0.5221 | 0.7405 | **+41.8%** |

**读表：** 四个通道均值都很接近，但 C2、C3 的生成标准差明显偏大；尚不能判断究竟是全通道普遍失真，还是某张生成样本的少数极端值拉高了标准差。

| 生成样本（零基索引） | 最小值 | 最大值 | 之前的训练最近邻距离 d |
|---|---:|---:|---:|
| 0 | −6.3667 | 4.7806 | 27.8 |
| 1 | −4.5724 | 3.6925 | 32.0 |
| 2 | −2.1612 | 3.7400 | 15.0 |
| 3 | −1.7202 | 3.9053 | 12.8 |
| **4** | **−23.8348** | **14.5800** | **48.9** |
| 5 | −3.0872 | 4.5226 | 22.0 |
| 6 | −3.1975 | 4.0652 | 18.7 |
| 7 | −1.5661 | 3.4424 | 11.0 |

**重点定位：** `index=4`（第 5 张生成图）同时具有最极端的潜变量值、最大的最近邻距离，但 **不能据此证明该图“导致”其他样本失败**，更不能把某个数值极端值当作唯一生成失败原因。下一步可先去掉第 5 张再比较通道标准差，然后统计每张图的通道级分位数与空间异常图；同时保留其余七张图的原始生成结果。

```python
# 复核：先排除 index=4 看异常是否集中在这一张
z_all = sampling["images"].float().cpu()  # [8,4,16,16]
z_wo4 = z_all[torch.arange(z_all.shape[0]) != 4]
for name, z in [("all", z_all), ("without_index4", z_wo4)]:
    print(name, z.mean((0, 2, 3)).tolist(),
          z.std((0, 2, 3), unbiased=False).tolist())
```

**谨慎结论：** 均值接近 ≠ 学会了图标的空间结构；标准差接近 ≠ 学会了潜变量的联合分布。与下一轮正式训练对比时，最好记录同种子、同噪声日程和相同评估协议，避免混入模型和采样器的变更。

**建议分析代码：**

```python
train_z = train_dataset.latents.float()
gen_z = sampling["images"].float().cpu()
# 通道统计：对 batch、H、W 求均值/标准差，保留 C
for name, z in [("train", train_z), ("gen", gen_z)]:
    print(name, "channel_mean", z.mean((0, 2, 3)).tolist())
    print(name, "channel_std", z.std((0, 2, 3), unbiased=False).tolist())
    print(name, "per_image_min", z.flatten(1).min(1).values.tolist())
    print(name, "per_image_max", z.flatten(1).max(1).values.tolist())
```

**结论边界：** 这是 **诊断线索**，不是确定的失败根因。首先需要继续训练收敛对照、检查 Scheduler/尺度与分桶指标，再综合判断。

---

<a id="part-g"></a>

## **八、第四周八道综合自测（独立练习）**

> **本篇主线：** 题目全部来自第四周末自测；必须在主知识笔记之后，避免参考答案干扰主动回忆。

> **自测说明：** 先不看答案，尝试每题用“**结论 → 为什么 → 一句公式或例子**”回答；再核对答案并标记是否独立完成。之前已独立答对的三道小题（`h=1.5`、`h=2`、重采 `ε` 会不同）不代表八道综合开放题全部掌握。

<a id="q31"></a>

### **31. 缓存固定均值之后，训练中还有哪两类主要随机性？**

**参考答案：** VAE 编码使用后验均值 `h=μϕ(x)`，只取消了 VAE 后验的随机采样 `η`。DDPM 训练仍为每个样本随机选择 **扩散时间步 `t`**、随机生成 **独立高斯噪声 `ε`**：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$

**补充：** 实际数据流还可以在 Epoch 间打乱图片顺序（Shuffle）；题目所说“两类主要随机性”特指 **扩散加噪** 的 `t`、`ε`。固定 VAE 后验均值不是把 `t` 或 `ε` 也固定。

**易错点：** `η`（VAE 后验采样）≠ `ε`（DDPM 扩散噪声）。**关联：Q10、Q11、Q2。**

---

<a id="q32"></a>

### **32. 一个 Batch 的时间步需要相同吗？为什么训练不用展开 1000 步？**

**参考答案：** **不需要相同**；每张图片可以独立抽到不同的 `tᵢ`，一个批次可以使用 `t.shape=[B]`。前向高斯马尔可夫链的累积效应有 **闭式边缘采样公式**：

$$q(z_t\mid z_0)=\mathcal N(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)I),\qquad z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

因此给定 `z₀`、时间索引对应的 `ᾱₜ` 和一份 `ε`，就能直接生成这一时刻的 `zₜ`，**无需从第 1 步逐步加噪到第 t 步**。随机抽时间步是在 Monte Carlo 估计对所有噪声等级的训练期望。

**易错点：** “训练时不用展开前向链”≠“推理时标准 DDPM 可以随意跳过逆向更新”；后一问题取决于采样算法。**关联：Q21、Q10。**

---

<a id="q33"></a>

### **33. 潜变量四通道均值不是 0，是否应当立刻标准化？**

**参考答案：** **不应立刻修改。** 当前 LDM 使用冻结 VAE 与固定约定 $z_0=s\mu_\phi(x)$（本项目 `s=0.13025`）；这属于按比例缩放，不是逐通道减均值、除标准差。通道均值非零不自动说明数据有错。随意改变训练潜变量却不修改推理/解码逆变换，会破坏模型与 Decoder 的输入尺度约定。

**正确流程：** 先检查原始图像预处理、VAE 权重与版本、后验均值策略、乘/除缩放是否一致。若要引入新的标准化 $z'=(z-\mu)/\sigma$，应明确 **只用训练集估计参数**，保存 `(μ,σ)`，重建训练缓存、重新训练/适配模型，并在解码前施行逆变换 $z=\sigma z'+\mu$；再做独立对照实验。

**易错点：** 均值非零≠需要紧急修正；某个通道的方差不等于整个张量的方差。**关联：Q3、Q4、Q8、Q9。**

---

<a id="q34"></a>

### **34. 为什么零输出的初始 MSE ≈ 1？为什么 MSE 很低仍可能只生成碎块？**

**参考答案：** 噪声标签是逐元素 $\epsilon\sim\mathcal N(0,1)$。若模型一开始输出 $\hat\epsilon=0$，则

$$\mathbb E[(\hat\epsilon-\epsilon)^2]=\mathbb E[\epsilon^2]=1.$$

即初始噪声 MSE 通常接近 1。**但噪声预测 MSE 不等于生成质量。** 高噪声时 $z_t\approx\epsilon$，统计基线也容易把 MSE 做得很低；低/中噪声阶段、潜变量多通道/空间联合结构、完整逆向链累计误差和模型训练不足，仍可能导致颜色碎块、对象缺失等生成失败。

**进一步诊断：** 同时看固定小样本拟合、独立验证分时间桶 MSE、Zero/Moment/DiT/EMA 对照、VAE 重建对照和最终随机采样质量。

**易错点：** “Loss 下降”不等于“模型会生成”；“高噪声区 MSE 低”不等于“该区间恢复干净图像简单”。**关联：Q14、Q15、Q22、Q28。**

---

<a id="q35"></a>

### **35. 训练恢复时只改变 beta_end，权重和形状不变，能否叫继续同一实验？**

**参考答案：** **不能。** `β_end` 改变通常会改变大部分时间位置的噪声系数和累计乘积，因而 **相同索引 `t` 对应不同噪声强度、不同训练输入 `zₜ` 和不同优化目标**。即使模型结构、Tensor Shape、权重文件都能加载，也已改变了训练契约；这属于 **新实验/新训练阶段**，不应标为与原轨迹严格一致的 Resume。

**恢复必须校验：** 完整 Noise Schedule（不是仅看名字）、数据和缓存身份、模型源码/架构、优化器与 EMA 状态、随机数状态、数据游标与训练配置。Checkpoint 中恢复的 `scheduler_source` 同名，不足以保证实际系数相同。

**易错点：** `load_state_dict()` 不报错，只代表形状等基本条件兼容，不能证明实验语义相同。**关联：Q8、Q9、Q19、Q20。**

---

<a id="q36"></a>

### **36. 为什么不能直接把潜变量裁剪到 [-1,1]？**

**参考答案：** 像素 DDPM 的干净归一化 RGB 值可能天然约束在 `[-1,1]`；VAE 的潜变量 `z₀` **没有这个逐元素区间约束**，本项目部分通道均值甚至大于 1。若强行 `clip(-1,1)`，会破坏真实潜变量的数值分布，导致输入 Decoder 的尺度和内容被截断，影响生成质量。因此当前 `clip_denoised=False` 合理。

**注意：** 像素 DDPM **也不一定必须裁剪**；这是某些采样实现的可选策略。LDM 也并非永远不能做任何动态阈值/范围约束，只是不能照搬像素区间。

**易错点：** `z` 不是 RGB 像素；“避免潜变量异常”≠“直接裁到 [-1,1]”。**关联：Q3、Q24。**

---

<a id="q37"></a>

### **37. 张无条件样本能否用 8 条描述算条件准确率？**

**参考答案：** **不能。** 当前 DiT 没接收文本、类别标签或任何相应条件，八张图只是由不同随机初始噪声生成；模型没有被要求根据某一条描述完成任务。事后给八张图附上描述，并据此声称“条件准确率”，不是合法的受控条件生成评估。

**可以做：** 报告无条件生成图像的解析对象数、规则符合比例、视觉多样性、生成潜变量统计与最近邻等 **诊断性指标**；但不要把解析器的成功率冒称为条件准确率，解析器本身也可能误判。要评价条件准确率，先构造真正接收条件、可重复的条件生成任务及评测协议。

**易错点：** “生成图看起来符合一句描述”≠“模型遵从了该输入条件”。**关联：Q15、Q28、Q29、Q20。**

---

<a id="q38"></a>

### **38. 把 1000 步 DDPM 仅调用 20 次，与确定性 DDIM 有什么不同？**

**参考答案：** **不能仅少调用 980 次就认为得到正确的 20 步 DDPM。** 标准 `DDPM.step` 的系数按 **相邻** 时间步 `t→t−1` 设计；直接从 `t` 跳去 `s<t−1` 却仍使用相邻更新，缺失了中间转移，生成路径不符合目标采样规则。若把调度器的训练总步数 `T=1000` 直接改成 20 且保持原 beta 起止值，还会改变终点信噪比和训练时的时间/噪声约定。

**确定性 DDIM（η=0）：** 在原噪声日程里选出一列递减时间点 `t_K>…>t_1`，每次先由模型算 `ε̂=εθ(zₜ,t)` 与 `ẑ₀`，再按对应 `ᾱ_s` 直接更新到选定的更早位置：

$$\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad z_s=\sqrt{\bar\alpha_s}\hat z_0+\sqrt{1-\bar\alpha_s}\hat\epsilon.$$

这里 `s<t` 是 **原噪声日程上选定的更早时刻**；最后从最小带噪时间点到干净边界要按 `ᾱ_{\text{clean}}=1` 处理。通常可以指定 **推理步数 K**，由采样器选择对应位置，也可以在支持的实现中明确提供时间点列表。DDIM 不等于“把 DDPM 相邻公式跳着调用”，它是不同的采样更新规则。

**细节：** 某些框架还提供经过专门推导的“祖先型跳步采样器”，但那也不是直接跳过普通 `DDPM.step` 的调用。**关联：Q21、Q25、Q26、Q27。**

核对资料：[DDPM 原论文](https://arxiv.org/abs/2006.11239)（后验与噪声参数化）、[DDIM 原论文](https://arxiv.org/abs/2010.02502)（共享加噪边缘及跳步）、[LDM 原论文](https://arxiv.org/abs/2112.10752)（空间潜变量与压缩）。
