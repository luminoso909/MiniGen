## **一、VAE 学习导航**

> 学习主题：VAE 原理与潜空间实验（基于 Week 2《VAE 原理与潜空间实验》完整 Notebook 与学习问答）  
> 核心知识点：潜空间/概率分布、近似后验/真实后验、对角高斯、KL 与 BCE、重参数化、TinyVAE、β 对照与可视化、模糊/坍塌诊断、Checkpoint/断点续训、DDPM 与 LDM 衔接  
> 约定：本笔记以 28×28 灰度 MNIST、16 维对角高斯 VAE 为主，另包含 2 维可视化实验；特别注明为「实验观察」的指标来自本周运行记录，其他简单数值为教学例子。

### **1. 学习导航**

| 模块 | 对应 Q&A | 主要问题 |
|---|---|---|
| 概念与概率基础 | Q1–Q10 | 潜空间、先验/后验、对角高斯、贝叶斯推断 |
| KL、ELBO、重参数化 | Q11–Q15 | KL 从何而来、期望权重、可微采样 |
| 损失与灰度 BCE | Q16–Q18 | BCE+KL、软目标、像素预测值的身份 |
| 训练与结果诊断 | Q19–Q24 | 不同 β/d、采样与重建、潜变量分析、失败原因 |
| 可视化与工程实现 | Q25–Q29 | 重建/生成、插值、Checkpoint、断点续训、TinyVAE 接口 |
| 跨阶段衔接 | Q30–Q31 | DDPM vs VAE Loss、潜空间扩散与论文阅读 |
| 间隔复习与纠错 | Q32–Q36 | AE vs VAE、重参数化、先验失配、模糊诊断、LDM |
| 采样补漏与手写巩固 | Q37–Q38 | 从 μ/logvar 采样到 2D 图像；从头手写 VAE 的最小任务 |

> 学习建议：**刚隔几天没学，先看「10 分钟快速复习」与 Q32–Q38；需要推导时查 Q11–Q17，分析模糊与先验生成问题时查 Q19–Q26。**

---

## **二、分钟快速复习：重新捡起 Week 2**

> **复习起点：** 你已经亲手实现过 TinyVAE、重参数化和 KL，完成了多组 β 消融、潜空间可视化、Checkpoint 与断点续训。现在的目标不是重学全部推导，而是迅速恢复「架构—公式—实验—诊断」的连接。

### **1. AE 和 VAE：先认识区别**

```text
普通 AE（确定性）： image x → Encoder → z → Decoder → x̂
VAE（概率模型）：image x → Encoder → (μ, logvar) → z=μ+exp(logvar/2)ε → Decoder → logits → x̂
                                               ε~N(0,I)
```

- **AE：** 通常直接输出确定性编码 $z=f_\phi(x)$，训练重点是重建；不必使用 KL，也没有默认保证随机先验 $z\sim N(0,I)$ 可直接生成合理图片。
- **VAE：** Encoder 输出近似后验 $q_\phi(z\mid x)=N(\mu_\phi(x),\operatorname{diag}(e^{\ell_\phi(x)}))$，其中 $\ell=\log\sigma^2$；训练中 $z\sim q_\phi(z\mid x)$，无条件生成时 $z\sim p(z)=N(0,I)$。

### **2. 必背的四个公式**

**重参数化：**

$$z=\mu+\exp(\ell/2)\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I).$$

**逐样本 KL（对角高斯对标准正态）：**

$$D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))=\frac12\sum_{j=1}^d(\mu_j^2+e^{\ell_j}-1-\ell_j).$$

**VAE 训练：**

$$L_{\beta\text{-VAE}}=\underbrace{L_{\mathrm{recon}}}_{\text{本实验使用灰度软目标 BCE}}+\beta\underbrace{L_{\mathrm{KL}}}_{\text{每图潜维求和，Batch 平均}}.$$

**DDPM 简化噪声预测目标：**

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad L_{\mathrm{DDPM,simple}}=\mathbb E\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2.$$

> 不要混淆：VAE 的 $\beta$ 是 **KL 权重**，DDPM 的 $\beta_t$ 是 **噪声调度系数**。

### **3. 核心代码与 Shape**

```python
# x: [B,1,28,28]; mu/logvar/noise/z: [B,16]
mu, logvar = model.encode(x)
z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
logits = model.decode(z)                   # [B,1,28,28]，未 sigmoid
recon = F.binary_cross_entropy_with_logits(logits, x, reduction='none').flatten(1).sum(1).mean()
kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean()
loss = recon + beta * kl
```

### **4. 三种生成路线与三个诊断层次**

| 路线 | 潜变量取自哪里 | 实验回答什么 |
|---|---|---|
| 均值重建 | $z=\mu_\phi(x)$ | 输入图片能否较准确重建？ |
| 随机重建 | $z=\mu_\phi(x)+\sigma_\phi(x)\epsilon$ | 采样扰动是否明显影响重建？ |
| 先验生成 | $z\sim N(0,I)$ | 不输入图片时能否生成合理样本？ |

- **损失层面：** Recon 与 KL 的权衡；**生成层面：** 上述三类结果；**潜空间层面：** 按维 KL、聚合后验、二维网格与插值。
- 本次 $d=16$ 对照在约 1000 步时：$\beta=0,0.5,1,4$ 的 Recon BCE/图约为 $97.10,105.67,114.56,143.71$；KL/图约为 $345.28,23.33,15.53,5.52$。**β 增大，KL 降低而重建通常变差；先验生成质量不能仅凭 16 张样本定论。**
- **重建模糊 ≠ 后验坍塌：** 可能由过强 KL、维度/网络容量不足、训练未收敛、重建目标、Decoder 对 $z$ 不敏感等引起。必须针对性验证。

### **5. 这次复习最容易忘的「采样 → 二维图片」**

**先采样一个潜向量，再让 Decoder 输出所有像素。** Encoder 的两个 Head 输出的是同形向量，而不是两张图片：

```text
x [B,1,28,28] → Encoder → μ [B,d], logvar [B,d]
                              + ε~N(0,I) [B,d]
                              → z=μ+exp(logvar/2)·ε [B,d]
                              → Decoder → logits [B,1,28,28]
                              → sigmoid → 预测灰度图 [B,1,28,28]
```

如果 `latent_dim=2`，表示潜向量有 **两个坐标**，并不是生成了一张 2 像素图片。原 MNIST 图片始终是二维的 `28×28`。详见 Q37。

### **6. 与下一阶段的桥梁**

```text
Latent Diffusion 训练：image → (pretrained) Encoder → latent z₀ → 加噪 zₜ → 去噪网络 → MSE
Latent Diffusion 生成：latent noise z_T → 多步去噪 → z₀ → (pretrained) Decoder → image
```

现代图像 LDM 往往在保留空间结构的压缩潜变量特征图上运行扩散，而不是直接把本周的 16 维 MNIST TinyVAE 当成完整生成系统。

---

## **三、潜空间、先验与近似后验**

<a id="q1"></a>

### **1. 潜空间和概率空间分别是什么？**

#### **i. 核心回答**

- **潜空间（latent space）** 是潜变量 $z$ 的取值空间。例如模型令 $z\in\mathbb R^{16}$，则潜空间是 $\mathbb R^{16}$。它是一套表示图像的隐藏坐标系；坐标不一定有可解释的单一语义。
- **概率分布** 规定在这个空间中如何随机选取坐标。例如 $p(z)=\mathcal N(0,I)$ 让不同坐标具有不同的采样可能性。
- 严格的 **概率空间** 写成 $(\Omega,\mathcal F,P)$：样本空间、事件集合与概率测度。它 **不是** 潜空间的同义词。日常说“在潜空间里放一个概率分布”通常是在讨论潜变量的取值与其概率规律，而非把两个概念等同。

#### **ii. 原理与直觉**

一张 MNIST 图像可表示为 $x\in\mathbb R^{1\times28\times28}$，共有 $784$ 个像素。编码器把图像转换为 $16$ 维的分布参数；解码器利用采样出的 $16$ 维向量重建图片。这些 $16$ 个数是潜变量坐标，而非必然对应“笔画粗细”“倾斜程度”等可解释属性。

**地图比喻**：潜空间是地图；概率分布决定随机去地图哪个位置；Decoder 决定每个位置对应什么图片。*地图上的一个位置*和*随机选位置的规律*是不同概念。

#### **iii. 公式与维度**

$$z\in\mathbb R^{16},\qquad p(z)=\mathcal N(\mathbf0,I_{16}),\qquad p_\theta(x\mid z).$$

- $p(z)$：预先指定的 **先验分布**，不需要输入图片。
- $p_\theta(x\mid z)$：解码器定义的、在给定潜变量时的图像分布。
- $q_\phi(z\mid x)$：编码器给出的 **近似后验分布**，依赖输入图片。

**易错点**：$z$ 是一次抽到的向量值；$q_\phi(z\mid x)$ 是分布，而不是一个向量值；编码器输出的 $\mu(x)$ 和 $\log\sigma^2(x)$ 是这个分布的参数。

---

<a id="q2"></a>

### **2. KL 约束如何影响潜变量保留图像信息？**

#### **i. 核心回答**

**你的担心是合理的，这正是 VAE 的信息保留与先验匹配之间的权衡。** KL 惩罚让单张图像的编码分布接近先验，但训练时还有重建损失，要求不同输入留下可辨识的信息。最终不是单独把 KL 压到零，而是在两个目标之间寻找平衡。

#### **ii. 直觉与数值例子**

先看一维潜变量：

$$p(z)=\mathcal N(0,1),\quad q(z\mid x_1)=\mathcal N(-1,0.25),\quad q(z\mid x_8)=\mathcal N(1,0.25).$$

这里高斯分布第二个参数是 **方差**，所以两种图像的标准差都是 $0.5$。两个编码分布的均值分别在 $-1$ 与 $1$，相差 $2$（即四个各自的标准差），仍然可以较好地区分。

对一维高斯 $q=\mathcal N(\mu,\sigma^2)$ 与标准正态先验：

$$D_{\mathrm{KL}}(q\Vert p)=\frac12\bigl(\mu^2+\sigma^2-1-\ln\sigma^2\bigr).$$

两个编码分布的 KL 都为：

$$\frac12(1+0.25-1-\ln0.25)\approx0.818.$$

**结论**：两个分布可以有不同的中心，同时各自与标准正态保持有限 KL；“趋近先验”不等于“必须重合”。

如果 **只** 优化 KL，确实可能得到对所有 $x$ 都有 $q_\phi(z\mid x)=p(z)$，此时 $z$ 完全不含图像信息，Decoder 无法根据 $z$ 区分数字。这就是后验坍塌（posterior collapse）的典型表现之一。实际 VAE 同时优化：

$$\mathcal L=\underbrace{\mathcal L_{\mathrm{recon}}}_{\text{保留图像信息}}+\beta\underbrace{\mathcal L_{\mathrm{KL}}}_{\text{约束偏离先验}}.$$

标准 VAE 通常 $\beta=1$；调整 $\beta$ 是超参数实验，不保证单调变好。较小的 $\beta$ 可能改善重建，但使先验采样效果变差。

#### **iii. “这么多图像能装得下吗？”**

$16$ 维不是 $16$ 个格子。每个坐标维度都允许连续取值，组合极其丰富。图像也不需要按“一张图占一个孤立格子”存储：相似图片可以共享统计结构。不过 **容量确有上限**：有限维度、噪声强度、模型能力和 KL 权重可能使可区分的信息不足；当不同输入的后验严重重叠时，重建会混淆甚至模糊。

#### **iv. 如何应对（实验建议）**

1. **比较潜变量维度**（如 $8/16/32$）：查看重建、先验采样质量及 KL 变化，而非默认维度越大越好。
2. **比较 $\beta$**（如 $0.1/1/4$）：重建与先验生成之间可能存在权衡。
3. **KL annealing**：训练初期用较小 KL 权重，再逐步提高，以缓解部分训练不稳定或坍塌问题。
4. **改善模型与分布**：加强合适的编码/解码能力、使用层次潜变量、更灵活的先验（如混合分布）等。
5. **做有条件生成**：如果希望准确控制输出对象或类别，给模型提供标签/文本等条件，而不是要求无条件随机变量自己“猜中”目标。

#### **v. 进一步原理：平均 KL 为何会限制信息？**

令 $p_{\mathrm{data}}(x)$ 为数据分布，定义聚合后验：

$$q_\phi(z)=\int p_{\mathrm{data}}(x)q_\phi(z\mid x)\,dx.$$

当相关密度与积分良好定义时，有恒等式：

$$\mathbb E_{p_{\mathrm{data}}(x)}D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)) = I_q(X;Z)+D_{\mathrm{KL}}(q_\phi(z)\Vert p(z)).$$

- $I_q(X;Z)$：$z$ 含有多少关于输入 $x$ 的信息（互信息）。
- 第二项：所有样本聚合后的编码分布离先验有多远。

因此极强的平均 KL 约束会压缩 $z$ 对输入携带的信息，而不只是让整体分布“看起来像高斯”。**每张图像的后验、聚合后验、先验是三个不同对象。**

---

<a id="q3"></a>

### **3. VAE 训练时，潜变量服从什么分布？**

#### **i. 核心回答**

**训练时给定某张输入图像 $x$，从编码器预测的近似后验分布 $q_\phi(z\mid x)$ 采样；不是直接从统一的标准正态先验 $p(z)$ 采样。** 标准正态随机数 $\epsilon$ 只在重参数化中用来制造噪声。

#### **ii. 三个对象别混淆**

| 符号 | 含义 | 是否依赖图片 $x$ | 何时使用 |
|---|---|---|---|
| $p(z)=\mathcal N(0,I)$ | 潜变量先验 | 否 | 无条件生成；KL 对齐目标 |
| $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x)))$ | 单张图的近似后验 | 是 | 训练时的采样与重建 |
| $\epsilon\sim\mathcal N(0,I)$ | 重参数化所需的辅助随机噪声 | 否 | 训练采样的计算步骤 |

#### **iii. 重参数化公式如何理解？**

$$\boxed{z=\mu+\sigma\odot\epsilon},\qquad \epsilon\sim\mathcal N(0,I).$$

- $\mu$：Encoder 给出的均值向量。
- $\sigma$：Encoder 对数方差对应的 **标准差** 向量（不是方差）。
- $\odot$：逐元素相乘，而不是矩阵乘法。
- $\epsilon$：每个维度服从标准正态的随机噪声。
- $z$：实际给 Decoder 使用的一次采样结果。

对于每个维度 $i$，因为 $\epsilon_i\sim\mathcal N(0,1)$，所以 $\mu_i+\sigma_i\epsilon_i\sim\mathcal N(\mu_i,\sigma_i^2)$。因此整个向量恰好服从 Encoder 预测的对角高斯分布。

**数值例子**：

$$\mu=[1,-0.5],\quad \log\sigma^2=[\ln 0.25,\ln 4].$$

从而

$$\sigma=\exp\left(\frac12\log\sigma^2\right)=[0.5,2].$$

若一次抽取 $\epsilon=[1,-0.5]$，则

$$z=[1,-0.5]+[0.5,2]\odot[1,-0.5]=[1.5,-1.5].$$

下次 $\epsilon$ 变化，$z$ 会变化；但固定输入 $x$ 时，$\mu$、$\sigma$ 不变（模型处于确定性的推理状态且参数不变）。

#### **iv. 为什么要重参数化？**

我们希望重建损失能够更新 Encoder 预测的 $\mu$ 和 $\log\sigma^2$。通过

$$z=\mu+\exp(0.5\,\log\sigma^2)\odot\epsilon$$

把随机性集中到独立的 $\epsilon$ 中，$z$ 对 $\mu$ 和 $\sigma$ 的运算路径可微；给定 $\epsilon$，梯度能通过这条路径回传。它也允许使用单次或少数几次采样去估计后验期望。**不要误认为 $z$ 因为用了标准正态 $\epsilon$，就一定服从标准正态；其分布由 $\mu$ 和 $\sigma$ 决定。**

#### **v. PyTorch 对应（关键实现，可运行的采样函数）**

```python
import torch

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mu, logvar: [B, latent_dim]，例如 [32, 16]
    std = torch.exp(0.5 * logvar)  # log(variance) -> standard deviation
    eps = torch.randn_like(std)     # eps ~ N(0, I)，shape 同 std
    return mu + std * eps          # z | x ~ q_phi(z|x)，shape [B, 16]

mu = torch.zeros(32, 16, requires_grad=True)
logvar = torch.zeros(32, 16, requires_grad=True)
z = reparameterize(mu, logvar)
assert z.shape == (32, 16)
z.sum().backward()
assert mu.grad is not None and logvar.grad is not None
```

`torch.distributions.Normal(mu, std).rsample()` 也提供重参数化采样；不要把普通的 `.sample()` 与 `.rsample()` 在梯度用途上混为一谈。

#### **vi. 更深一层：训练中所有样本的 $z$ 总体是什么分布？**

单张图像条件下的 $q_\phi(z\mid x)$ 是高斯。但若先随机选训练图片 $x$，再按其后验采样，得到的是 **聚合后验**：

$$q_\phi(z)=\int p_{\mathrm{data}}(x)q_\phi(z\mid x)\,dx.$$

它是不同高斯的混合，**一般不再是单一高斯分布，也不保证恰好等于 $\mathcal N(0,I)$**。VAE 的 KL 项会推动它与先验匹配。

---

<a id="q4"></a>

### **4. 无条件采样能否指定生成对象？如何提高可控性？**

#### **i. 核心回答**

**对标准的无条件 VAE，你的观察基本正确**：从 $p(z)$ 随机采样，只要求输出整体像训练数据，并不指定“必须生成数字 3”。随机性影响抽到了什么样本，但输出是否合理还取决于训练好的生成分布和 Decoder；不是“随机就一定成功”。

首先区分“对”的标准：

- **重建**：输入数字 3，希望重建仍保留数字 3 的信息，可以使用 $q_\phi(z\mid x)$ 采样或直接取均值 $z=\mu(x)$。
- **无条件生成**：没有指定目标，数字 1 或数字 8 都可能是合理结果，不能按是否“猜中某个指定数字”评估。
- **条件生成**：用户明确指定“我要一个 3”或“我要红色猫”，需要把类别/文本/结构等 **条件** 交给模型。

#### **ii. 如何提高可控性？**

1. **Conditional VAE（CVAE）**：给 Encoder/Decoder 引入标签 $y$（例如数字 3），建模 $q_\phi(z\mid x,y)$ 和 $p_\theta(x\mid z,y)$。生成时指定 $y=3$，并从模型指定的条件先验 $p(z\mid y)$（或与 $y$ 无关的固定先验）采样。
2. **潜空间编辑/插值**：已有图片可先编码获得 $z$，再插值或寻找能控制特定属性的方向；但普通 VAE 不保证某个潜变量维度自动对应单一可解释属性。
3. **更丰富的条件**：类别标签只控制类别；姿态、边缘、分割、文字等更细条件通常需要专门的模型设计和相应训练数据。
4. **固定随机种子**：能复现同一采样过程，不代表能指定采样内容或确保正确性。

**易错点**：“随机采样”与“不可控”不是同义词。生成可具有随机性，同时严格或概率性地依赖条件；只是标准 **无条件** VAE 没有外部目标输入。

---

<a id="q5"></a>

### **5. 为什么编码器要输出两个向量？**

**核心回答**：因为教学 VAE 假设 $q_\phi(z\mid x)$ 是对角高斯分布，必须给出它的均值与方差信息；Encoder 输出 $\mu$ 和 $\log\sigma^2$ 两个向量，均为 `[B, 16]`。

$$q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right).$$

不是说“任何概率分布都只需要两个向量”，而是 **选用了对角高斯假设**。使用对数方差有利于将方差参数表示为无约束的实数，并由指数函数恢复正值。

<a id="q6"></a>

### **6. 为什么训练时需要随机采样？**

**核心回答**：VAE 训练目标包含在 $q_\phi(z\mid x)$ 下对重建对数似然的期望；采样让模型能够用蒙特卡洛方式估计这一项，并让 Decoder 在潜变量存在不确定性时学习重建。

$$\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] \approx \frac1K\sum_{k=1}^K\log p_\theta(x\mid z^{(k)}).$$

训练中常用 $K=1$ 的采样估计；采用重参数化可把梯度传回 Encoder。这里的 $z^{(k)}$ 从 **近似后验** 采样，不是直接从先验采样。

<a id="q7"></a>

### **7. 为什么既有重建损失又有 KL 损失？**

**核心回答**：重建损失促进 $z$ 保留足够的输入信息，KL 损失约束 $q_\phi(z\mid x)$ 与 $p(z)$ 的偏离。两者来自 VAE 的负 ELBO（证据下界）：

$$\operatorname{ELBO}(x) =\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] -D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).$$

最大化 ELBO 等价于最小化负 ELBO：

$$\mathcal L_{\mathrm{VAE}}(x) =\underbrace{\mathbb E_q[-\log p_\theta(x\mid z)]}_{\text{重建项}} +\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))}_{\text{KL 项}}.$$

对于独立 Bernoulli 二值像素，逐像素 BCE 之和就是重建负对数似然。本实验使用灰度软目标，等价于对随机二值化目标取期望，不能将该式直接当作连续灰度的负对数密度。使用 `binary_cross_entropy_with_logits` 可直接处理 Decoder 输出的 logits，通常比先 sigmoid 再 BCE 更稳定。

对角高斯与标准正态间的 KL 可以解析计算：

$$D_{\mathrm{KL}} =\frac12\sum_{i=1}^{d}\left(\mu_i^2+\sigma_i^2-1-\log\sigma_i^2\right).$$

**求和方式要匹配**：本 Notebook 的典型写法是对像素维求和、潜变量维求和，再对 batch 取平均，避免随意把一个损失项求和而另一个全局平均。

<a id="q8"></a>

### **8. 生成时没有输入图片，潜变量从哪里来？**

**核心回答**：从预设先验 $p(z)=\mathcal N(0,I)$ 直接采样，不再需要 Encoder：

$$z\sim\mathcal N(0,I),\qquad x\sim p_\theta(x\mid z).$$

在 Bernoulli 像素解释下，Decoder 产生 `[B, 1, 28, 28]` 的 logits：

```python
# decoder 是已经训练好的模型组件
z = torch.randn(32, 16, device=next(decoder.parameters()).device)
logits = decoder(z)                   # [32, 1, 28, 28]
probabilities = torch.sigmoid(logits) # 像素为 1 的概率，也可作为灰度均值图显示
binary_samples = torch.bernoulli(probabilities)  # 真正的二值 Bernoulli 采样
```

注意：`probabilities`（概率/灰度均值图）与 `binary_samples`（每像素随机采样的 0/1 图）不是同一个对象；如果只是显示生成图，通常显示前者。

---

<a id="q9"></a>

### **9. 对角高斯的设计原因和理解是什么？**

#### **i. 核心回答**

VAE 的编码器为每张输入图像 $x$ 预测一个分布 $q_\phi(z\mid x)$。本项目选择 **16 维对角协方差高斯**：

$$q_\phi(z\mid x)=\mathcal N\bigl(\mu_\phi(x),\operatorname{diag}(\exp(\ell_\phi(x)))\bigr),\quad \ell_\phi(x)=\log\sigma_\phi^2(x).$$

- “高斯”：假设给定 $x$ 后，潜变量 $z$ 服从多元正态分布。
- “对角”：协方差矩阵的非对角元素都设为 $0$；对 **给定 $x$ 的这个高斯分布**，各潜变量维度彼此独立。
- “设计原因”：只预测每维的均值和方差，采样简单、KL 可解析求出、参数量和计算量小；**代价** 是不能直接刻画给定 $x$ 后各维度间的相关性。

#### **ii. 原理与直觉：协方差矩阵到底在描述什么？**

二维多元高斯：

$$z=\begin{bmatrix}z_1\\z_2\end{bmatrix},\qquad \Sigma=\begin{bmatrix}\sigma_1^2&\operatorname{Cov}(z_1,z_2)\\\operatorname{Cov}(z_1,z_2)&\sigma_2^2\end{bmatrix}.$$

主对角线表示每一维的自身方差；非对角线表示两维如何共同变化。举例：

$$\Sigma_{\mathrm{full}}=\begin{bmatrix}1&0.8\\0.8&1\end{bmatrix},\qquad \Sigma_{\mathrm{diag}}=\begin{bmatrix}1&0\\0&1\end{bmatrix}.$$

前者表达正相关（等密度线可呈倾斜椭圆）；后者忽略维度间相关性（等密度线轴对齐）。对多元高斯而言，不相关也意味着独立；对非高斯分布一般不成立。注意：**单张图片的条件分布** 中维度独立，不等于把不同图片的编码分布混合后的 **聚合后验** 也独立。

#### **iii. 为什么 Encoder 输出两个向量而不是完整矩阵？**

对于潜变量维数 $d$：

| 方案 | 均值参数量 | 独立协方差参数量 | 总数 | 随 $d$ 增长 |
|---|---:|---:|---:|---|
| 完整协方差高斯 | $d$ | $d(d+1)/2$ | $d+d(d+1)/2$ | $O(d^2)$ |
| 对角协方差高斯 | $d$ | $d$ | $2d$ | $O(d)$ |
| 本项目 $d=16$（完整） | 16 | 136 | **152** | — |
| 本项目 $d=16$（对角） | 16 | 16 | **32** | — |

完整协方差还需要有效地保证矩阵半正定/正定，并处理较复杂的线性代数运算；对角高斯只需确保每维方差为正。网络可输出无约束的对数方差 $\ell=\log\sigma^2$，再用 $\exp(\ell)$ 变回正方差。

#### **iv. 数值例子：二维采样与 Tensor Shape**

假定某张图片预测出：

$$\mu=\begin{bmatrix}1\\-1\end{bmatrix},\qquad \sigma^2=\begin{bmatrix}0.25\\4\end{bmatrix},\qquad \Sigma=\begin{bmatrix}0.25&0\\0&4\end{bmatrix}.$$

所以 $\sigma=[0.5,2]$。如果本次标准正态辅助噪声恰好取到 $\epsilon=[1,-0.5]$，重参数化得到：

$$z=\mu+\sigma\odot\epsilon =[1,-1]+[0.5,2]\odot[1,-0.5] =\boxed{[1.5,-2]}.$$

这里 $\epsilon\sim\mathcal N(0,I)$，但 **$z\sim\mathcal N(\mu,\operatorname{diag}(\sigma^2))$，不是标准正态**（除非参数恰好是 $\mu=0,\sigma^2=1$）。

**PyTorch 核心代码（可直接运行的独立示例）：**

```python
import torch

mu = torch.tensor([[1.0, -1.0]])                      # [B=1, d=2]
logvar = torch.log(torch.tensor([[0.25, 4.0]]))        # [1, 2]
eps = torch.tensor([[1.0, -0.5]])                     # 固定样本，便于手算复现
std = torch.exp(0.5 * logvar)                          # [1, 2] -> [0.5, 2.0]
z = mu + std * eps                                     # [1, 2] -> [1.5, -2.0]
print(z)
```

本项目 $d=16$、批大小为 $B$ 时，`mu`、`logvar`、`std`、`eps`、`z` 的形状都是 `[B,16]`。重参数化令随机性由 $\epsilon$ 承担，使重建损失能经 $z$ 对均值与对数方差求导。

#### **v. 与先验和真实后验的关系**

- 先验 $p(z)=\mathcal N(0,I)$：生成前预先指定，**不依赖图像**。
- 真实后验 $p_\theta(z\mid x)$：看见图片后，哪些潜变量更可能生成它；通常无法精确计算，且 **未必是高斯或对角高斯**。
- 近似后验 $q_\phi(z\mid x)$：Encoder 输出的 **对角高斯近似**，可训练、可采样；它逼近真实后验，但不必等于真实后验。

如果对角高斯的表达能力不足，可以考虑完整/低秩协方差、高斯混合、Normalizing Flow、层次化潜变量等更丰富的近似后验。这些方法通常增加计算和实现复杂度。

**易错点**：对角高斯不是“16 个类别”；潜空间有连续取值。对角高斯不是说输入图像的像素独立，也不是说所有图片编码后的总体潜变量都独立。

---

<a id="q10"></a>

### **10. 如何用贝叶斯公式手算离散潜变量的后验？**

#### **i. 核心回答**

贝叶斯公式：

$$\boxed{P(Z=A\mid X=x)=\frac{P(Z=A)\,P(X=x\mid Z=A)}{P(X=x)}}$$

- **先验** $P(A)$：没有看图片之前，相信原因 A 的程度。
- **似然** $P(x\mid A)$：若原因是 A，观察到这张图片的可能性。
- **联合概率** $P(A,x)$：A 发生而且观察到 x 的概率。
- **边缘似然 / 证据** $P(x)$：不管隐藏原因是什么，观察到 x 的总体概率；是归一化因子。
- **后验** $P(A\mid x)$：看到 x 后，对隐藏原因 A 的更新判断。

**似然 $P(x\mid A)$ 与后验 $P(A\mid x)$ 的条件方向相反，不能直接混为一谈。**

#### **ii. 按 Notebook 数据逐步手算**

题设隐藏变量 $Z\in\{A,B\}$，两种状态互斥且穷尽；一张已观察到的图片记为 $x$。

$$P(A)=0.5,\quad P(B)=0.5,\quad P(x\mid A)=0.8,\quad P(x\mid B)=0.2.$$

**第 1 步：先验 $\times$ 似然，得到联合概率。**

$$P(A,x)=0.5\times0.8=0.4,\qquad P(B,x)=0.5\times0.2=0.1.$$

**第 2 步：对所有潜变量状态求和，得到证据。**

$$P(x)=\sum_{z\in\{A,B\}}P(z)P(x\mid z) =0.4+0.1=\boxed{0.5}.$$

**第 3 步：分别除以证据进行归一化。**

$$P(A\mid x)=\frac{0.4}{0.5}=\boxed{0.8},\qquad P(B\mid x)=\frac{0.1}{0.5}=\boxed{0.2}.$$

| 状态 | 先验 | 似然 | 未归一化权重/联合概率 | 后验 |
|---|---:|---:|---:|---:|
| $A$ | 0.5 | 0.8 | 0.4 | **0.8** |
| $B$ | 0.5 | 0.2 | 0.1 | **0.2** |
| 总计 | 1 | — | **0.5**（证据） | **1** |

得到 $0.8$、$0.2$ 并不是因为“后验恒等于似然”，而是因为本例 **先验恰好相等**。例如若 $P(A)=0.9,P(B)=0.1$，仍用同一似然，则

$$P(x)=0.9\cdot0.8+0.1\cdot0.2=0.74, \qquad P(A\mid x)=\frac{0.72}{0.74}\approx0.973.$$

#### **iii. PyTorch 代码对应（与 Notebook 一致，可直接运行）**

```python
import torch

prior = torch.tensor([0.5, 0.5], dtype=torch.float64)     # [2]，P(A), P(B)
likelihood = torch.tensor([0.8, 0.2], dtype=torch.float64) # [2]，P(x|A), P(x|B)
weights = prior * likelihood                               # [2] -> [0.4, 0.1]
evidence = weights.sum()                                    # [] -> 0.5
posterior = weights / evidence                              # [2] -> [0.8, 0.2]
print(evidence.item(), posterior.tolist())
assert torch.allclose(posterior, torch.tensor([0.8, 0.2], dtype=torch.float64))
```

- `prior * likelihood`：按状态逐元素相乘得到联合概率。
- `.sum()`：对潜变量的 **两种可能状态** 求和，把潜变量消去。
- `/ evidence`：得到和为 1 的后验分布。

#### **iv. 为什么离散例子算得出，而 VAE 要训练 Encoder？**

上例只有两个状态，可以显式求和：

$$P(x)=\sum_z p(z)p_\theta(x\mid z).$$

实际 VAE 的 $z\in\mathbb R^{16}$ 是 **连续潜变量**，于是变成积分：

$$\boxed{p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz},\qquad p_\theta(z\mid x)=\frac{p(z)p_\theta(x\mid z)}{p_\theta(x)}.$$

由于 $p_\theta(x\mid z)$ 由神经网络参数化，这个高维积分通常无法直接精确计算，因此很难为每张图片逐个计算真实后验。VAE 用 Encoder 学习 **摊销推断（amortized inference）**：给一张图片 $x$，网络一次前向就预测 $q_\phi(z\mid x)$ 的 $\mu(x)$ 与 $\log\sigma^2(x)$，训练中通过 ELBO 同时优化 Encoder 和 Decoder。

**离散例子 $Z\in\{A,B\}$ 的后验是两点上的类别分布，不是高斯分布。** 它演示的是“先验×似然÷证据”的贝叶斯逻辑；实际 VAE 再用对角高斯来近似连续的真实后验。

**技术精度**：如果 $x$ 是连续取值，$p_\theta(x\mid z)$ 和 $p_\theta(x)$ 表示概率 **密度** 而非单点概率。本 Notebook 的 Bernoulli 像素模型在离散像素事件意义下可解释为概率。这个区分不改变贝叶斯比例关系。

---

## **四、KL 散度与重参数化**

<a id="q11"></a>

### **11. MiniGen 第五章的对角高斯 KL 如何理解？一维公式为何成立，多维和 batch 怎么归一化，手算例子怎么验证？**

> 对应 Notebook「五、对角高斯 KL：从密度展开到代码归一化」。本问题来自所展示的三张章节截图；重点是实现可用的理解，而非重做上一章的 ELBO 推导。

#### **i. 核心回答**

对一张图像，Encoder 输出对角高斯近似后验的均值向量 $\mu\in\mathbb R^d$ 和对数方差向量 $\ell=\log\sigma^2\in\mathbb R^d$；先验为 $p(z)=\mathcal N(0,I_d)$。KL 度量 **整张图片的编码分布相对先验的偏离**：

$$\boxed{D_{\mathrm{KL}}(q\Vert p)=\frac12\sum_{j=1}^{d}\bigl(\mu_j^2+\exp\ell_j-1-\ell_j\bigr).}$$

- **先对潜变量维度 $d$ 求和**：得到一张图片的联合分布 KL；对角高斯条件独立使各维 KL 可加。
- **再对 batch 大小 $B$ 求平均**：得到这个 batch 的平均每样本 KL。不要再把 $d$ 除掉，否则会改变损失尺度。
- 这个 KL **不是概率、不是重建误差**，而是非负的分布差异度量。

#### **ii. 原理与直觉：公式每一项在惩罚什么？**

一维情形：$q=\mathcal N(\mu,\sigma^2),\ p=\mathcal N(0,1)$：

$$D_{\mathrm{KL}}(q\Vert p) =\underbrace{\tfrac12\mu^2}_{\text{均值偏离 0}} +\underbrace{\tfrac12(\sigma^2-1-\log\sigma^2)}_{\text{方差偏离 1}}.$$

- 均值越偏离 $0$，$\mu^2$ 越大。
- 方差过大或过小，都使第二项变大；当 $\sigma^2=1$ 时这一项为零。
- 仅当 $\mu=0,\sigma^2=1$，一维 KL 为零。但 VAE 还必须降低重建损失，不能只追求 KL 为零，否则容易丢失输入信息。

**小数值例子**：$\mu=2,\sigma^2=1$ 时 KL 为 $2$；$\mu=0,\sigma^2=4$ 时 KL 为 $(4-1-\log4)/2\approx0.80685$；$\mu=0,\sigma^2=1$ 时 KL 为 $0$。这里 $\log$ 是自然对数。

#### **iii. 一维公式的来源：最少必要推导**

KL 定义为 $D_{\mathrm{KL}}(q\Vert p)=\mathbb E_{z\sim q}[\log q(z)-\log p(z)]$。一维高斯的对数密度：

$$\log q(z)=-\tfrac12\log(2\pi)-\log\sigma-\frac{(z-\mu)^2}{2\sigma^2}, \qquad \log p(z)=-\tfrac12\log(2\pi)-\frac{z^2}{2}.$$

相减时 $-\tfrac12\log(2\pi)$ 抵消：

$$D_{\mathrm{KL}}(q\Vert p)=-\log\sigma -\frac{\mathbb E_q[(z-\mu)^2]}{2\sigma^2} +\frac{\mathbb E_q[z^2]}2.$$

利用方差定义和恒等式 $\mathbb E_q[(z-\mu)^2]=\sigma^2$、$\mathbb E_q[z^2]=\sigma^2+\mu^2$：

$$=-\log\sigma-\tfrac12+\tfrac12(\sigma^2+\mu^2) =\boxed{\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2)}.$$

**为什么 $\mathbb E[z^2]=\sigma^2+\mu^2$？** 由 $\operatorname{Var}(z)=\mathbb E[z^2]-(\mathbb E[z])^2$ 直接移项得到；$\mathbb E[z]=\mu$。期望 $\mathbb E_q$ 指“按分布 $q$ 给不同取值加权平均”，不是从一个实际采样结果算平方。

#### **iv. 手算：两维示例（与 Notebook 一致）**

取 $\mu=(1,0)$、**方差** $\sigma^2=(4,1)$，因此 $\ell=(\log4,0)$。

| 维度 | $\mu_j$ | $\sigma_j^2$ | KL 计算 | KL 值 |
|---|---:|---:|---|---:|
| 1 | 1 | 4 | $(1^2+4-1-\log4)/2$ | $1.3068528$ |
| 2 | 0 | 1 | $(0^2+1-1-\log1)/2$ | $0$ |
| **合计** | — | — | **两维相加** | **$1.3068528$** |

若一个 batch 中有两张完全相同的上述图片，**batch 平均** 为 $(1.3068528+1.3068528)/2=1.3068528$；如果错误地还对维度平均，则结果变为 $1.3068528/2=0.6534264$。这不是同一个 $2$ 维分布的 KL。

#### **v. PyTorch 代码对应（可运行）**

```python
import torch

# B=2, d=2；两张相同样本，mu/logvar 都是 [B, d]
mu = torch.tensor([[1., 0.], [1., 0.]])
logvar = torch.tensor([[4., 1.], [4., 1.]]).log()

# [B, d]：每个样本、每个潜变量维度的 KL 贡献
kl_per_dim = 0.5 * (mu.square() + logvar.exp() - 1 - logvar)
# [B]：每张图像的 d 维联合分布 KL
kl_per_image = kl_per_dim.sum(dim=1)
# []：batch 平均标量
kl_loss = kl_per_image.mean()
print(kl_per_image.tolist(), kl_loss.item())
# [1.30685..., 1.30685...]，1.30685...

# 等价的常见写法
kl_loss2 = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
assert torch.allclose(kl_loss, kl_loss2)
```

| 变量 / 运算 | Tensor Shape | 数学含义 |
|---|---|---|
| `mu`, `logvar` | `[B, d]` | 编码器对每张图片输出的分布参数 |
| `logvar.exp()` | `[B, d]` | $\sigma^2=e^\ell$，不是标准差 |
| `kl_per_dim` | `[B, d]` | 各维 KL 贡献 |
| `.sum(dim=1)` | `[B]` | 每张图像的 $d$ 维联合 KL |
| `.mean()` | 标量 `[]` | 对 batch 平均 |

如果模型 $d=16$，就将输入张量变成 `[B,16]`，代码本身无需改动。所有运算可自动求导，训练会通过 `mu` 和 `logvar` 更新 Encoder。

#### **vi. 与重建损失的尺度配合**

若图像为 `[B,1,28,28]`，Bernoulli 重建损失常用 **每张图片对 784 个像素求和，再对 batch 求平均**，与 KL 的“每张图像对 $d$ 维求和，再对 batch 求平均”相对应：

```python
import torch.nn.functional as F

# logits, x: [B,1,28,28]; logits 是未经 sigmoid 的输出
recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
recon_loss = recon.flatten(start_dim=1).sum(dim=1).mean()
loss = recon_loss + kl_loss
```

这只是与 Notebook 相符的一种损失尺度约定。若改为对 **全部像素** 平均，或者对 **全部潜变量维度** 平均，即使名义上的 $\beta=1$ 不变，也会改变两项损失的相对权重，所以必须明确 reduction 方式。

#### **vii. 易错点与工程实践**

1. `logvar = log(sigma^2)` 是 **对数方差**，`exp(logvar)` 才是方差，`exp(0.5*logvar)` 才是标准差。
2. 把 **批量维度求平均** 与 **潜变量维度求和** 区分清楚；`mean()` 直接覆盖整个 `[B,d]` 不符合本 Notebook 的 KL 定义。
3. 对角高斯的“各维相加”依赖其分解结构；完整协方差高斯不能简单照搬逐维公式。
4. KL 理论上非负，浮点误差可能造成极小的负数；若出现明显负值、`NaN` 或溢出，应检查公式、数据和 `logvar` 数值，而不是静默裁剪掩盖训练问题。建议使用稳定的 `binary_cross_entropy_with_logits`。
5. 这里的 $\log$ 是 **自然对数**，不是以 10 为底；手算 $\log4\approx1.386294$。

---

<a id="q12"></a>

### **12. KL 散度为什么用 $\mathbb E_q[\log(q/p)]$？**

#### **i. 核心回答**

**$D_{\mathrm{KL}}(q\Vert p)=\mathbb E_q[\log(q/p)]$ 是 KL 散度的通用定义，不是从高斯分布推导出的特殊公式。** 它可以理解成：数据按 $q$ 分布产生时，使用 $p$ 而不是 $q$ 来描述这些结果，平均增加多少 **对数损失（信息量）**。

这个表达式有三个环节：

1. **$-\log p(z)$ 是事件的信息量（surprisal）**。事件越罕见，出现时带来的信息量越多；对数将独立事件的概率乘法转成信息量加法。
2. **$\log(q(z)/p(z)) = -\log p(z)-[-\log q(z)]$**，是同一事件在两套模型下的信息量之差。
3. **$\mathbb E_q$ 是按真实采样频率加权平均**：哪个事件更常由 $q$ 产生，它对总体差异的贡献就有更高的权重。

KL 并非唯一的分布差异指标；选择对数比值的平均，是因为它有信息论与编码上的意义、非负性和良好的数学性质。

#### **ii. 从信息量到公式**

在离散情况下，事件 $z$ 以概率 $p(z)$ 发生。定义它在概率模型 $p$ 下的自信息量为

$$I_p(z)=-\log p(z).$$

原因：概率为 $1$ 时信息量为 $0$，罕见事件信息量大，并且对于独立事件，$I(ab)=I(a)+I(b)$（源于 $\log(ab)=\log a+\log b$）。这里默认自然对数，单位是 nat。

对同一事件比较两个模型的信息量：

$$I_p(z)-I_q(z)=-\log p(z)+\log q(z)=\log\frac{q(z)}{p(z)}.$$

如果事件按 $q$ 发生，就对这些差异取 $q$ 下的期望：

$$\boxed{D_{\mathrm{KL}}(q\Vert p)=\sum_z q(z)\log\frac{q(z)}{p(z)}=\mathbb E_{z\sim q}\!\left[\log\frac{q(z)}{p(z)}\right].}$$

这里的 $\mathbb E_q[f(z)] = \sum_z q(z)f(z)$，就是按 $q$ 给每种情况加权平均。连续情况变成 $\int q(z)\log(q(z)/p(z))\,dz$，此时 $q(z),p(z)$ 指 **概率密度**，不是“单个连续点的概率”。

#### **iii. 手算例子：偏向正面的硬币**

假设真实采样分布 $q$ 和使用的模型 $p$ 为：

| 结果 | $q$（事件实际出现的频率） | $p$（模型判断的概率） | $\log(q/p)$ | $q\log(q/p)$ |
|---|---:|---:|---:|---:|
| 正面 H | $0.8$ | $0.5$ | $\log1.6\approx0.4700$ | $0.8(0.4700)=0.3760$ |
| 反面 T | $0.2$ | $0.5$ | $\log0.4\approx-0.9163$ | $0.2(-0.9163)=-0.1833$ |

合计：

$$D_{\mathrm{KL}}(q\Vert p) =0.8\log\frac{0.8}{0.5}+0.2\log\frac{0.2}{0.5} \approx \boxed{0.1927\ \mathrm{nat}}.$$

注意：**单项对数比值可为负，但整体 KL 不会为负**。$0.1927$ 不是某个事件的概率，也不是“19.27% 的错误率”。

#### **iv. 为什么平均以后非负？（可选的小证明）**

利用 $\log t\le t-1$（$t>0$），取 $t=p(z)/q(z)$：

$$\sum_z q(z)\log\frac{p(z)}{q(z)} \le \sum_z q(z)\left(\frac{p(z)}{q(z)}-1\right) =\sum_zp(z)-\sum_zq(z)=0.$$

两边乘以 $-1$：

$$\boxed{D_{\mathrm{KL}}(q\Vert p)\ge0.}$$

此处先假定所有相关概率为正。若某事件 $q(z)>0$、$p(z)=0$，KL 为 $+\infty$；KL 为零意味着二者几乎处处相等。

#### **v. 与熵和交叉熵的关系**

定义（离散分布）：

$$H(q)=-\mathbb E_q[\log q(z)],\qquad H(q,p)=-\mathbb E_q[\log p(z)].$$

则

$$\boxed{D_{\mathrm{KL}}(q\Vert p)=H(q,p)-H(q).}$$

在上述硬币例子中：

- $H(q)=-0.8\log0.8-0.2\log0.2\approx0.5004$ nat。
- $H(q,p)=-0.8\log0.5-0.2\log0.5\approx0.6931$ nat。
- 差值 $0.6931-0.5004\approx0.1927$ nat，与直接计算一致。

如果 $q$ 固定，最小化 $H(q,p)$ 与最小化 $D_{\mathrm{KL}}(q\Vert p)$ 对 $p$ 的最优解相同；但在 VAE 中 Encoder 的 $q_\phi(z\mid x)$ **也会变化**，不可擅自把 $H(q)$ 当成训练中不变的常数。

#### **vi. 与 VAE 和高斯公式的联系**

对于当前 VAE，把 $q$ 代成 **Encoder 给定图片 $x$ 的近似后验** $q_\phi(z\mid x)$，把 $p$ 代成 **标准正态先验** $p(z)=\mathcal N(0,I)$：

$$D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\Vert p(z)\bigr) =\mathbb E_{q_\phi(z\mid x)}[\log q_\phi(z\mid x)-\log p(z)].$$

在一维 $q=\mathcal N(\mu,\sigma^2)$、$p=\mathcal N(0,1)$ 的特殊情况下，**代入两个高斯的对数密度并求期望**，才得到上一题使用的解析式：

$$\boxed{D_{\mathrm{KL}}(q\Vert p)=\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2).}$$

**不要倒置逻辑：定义 $\to$ 代入分布 $\to$ 特定解析公式，而不是从高斯公式倒推出 KL 的定义。**

#### **vii. 易错点**

- $\mathbb E_q$ 表示从 $q$ 出发加权平均；$D_{\mathrm{KL}}(q\Vert p)$ 通常不等于 $D_{\mathrm{KL}}(p\Vert q)$，所以 KL 不是距离度量。
- 事件的 $\log(q/p)$ 可以为负，但它在 $q$ 下的平均值不能为负。
- “KL 是两个概率分布的信息差”是直觉；更精确地说，它是 **在 $q$ 下的平均对数密度比**。
- KL 可以定义于高斯、分类分布等许多分布，不局限于 VAE。

---

<a id="q13"></a>

### **13. 为什么两项对数密度都按同一个分布取期望？**

#### **i. 核心回答**

**期望的权重由下标决定，而不是由被平均的函数来自谁决定。**

- $\mathbb E_q[f(z)]$：当 $z$ 按 $q$ 发生时，函数 $f(z)$ 的平均值；所有项都按 $q$ 的概率/密度加权。
- $\mathbb E_q[\log p(z)]$ 完全合法：$p(z)$ 只是被代入的函数，不会自动把积分权重改成 $p$。
- 高斯 KL 推导使用 $\mathbb E_q[z]=\mu_q$、$\mathbb E_q[(z-\mu_q)^2]=\sigma_q^2$，因为此处的 $\mu_q,\sigma_q^2$ **明确是 $q$ 的均值和方差**；不是 $p$ 的均值和方差。

#### **ii. 直觉：比较同一批事件，才需要同样的权重**

把 $q$ 看作数据实际出现的频率，把 $p$ 看作另一套预测模型。如果按 $q$ 产生 100 次事件，比较两套模型描述 **同一批结果** 的平均信息量：

$$D_{\mathrm{KL}}(q\Vert p) =\underbrace{\mathbb E_q[-\log p(z)]}_{\text{按 }q\text{ 发生时，用 }p\text{ 描述的平均信息量}} -\underbrace{\mathbb E_q[-\log q(z)]}_{\text{按 }q\text{ 发生时，用 }q\text{ 描述的平均信息量}}.$$

若把前者换成 $\mathbb E_p[-\log p(z)]$，就变成了另一批按 $p$ 出现的事件，不能再解释为针对同一批 $q$ 样本的额外平均对数损失。它等于 $H(p)$，而原来的量是交叉熵 $H(q,p)$。反向 KL $D_{\mathrm{KL}}(p\Vert q)$ 可以使用 $p$ 作为权重，但那是另一个量。

#### **iii. 数值例子一：两个离散分布**

让随机变量 $z\in\{0,2\}$，并设：

| $z$ | $q(z)$：取期望的权重 | $p(z)$：另一个模型的概率 |
| --- | ---: | ---: |
| 0 | 0.8 | 0.6 |
| 2 | 0.2 | 0.4 |

先计算 $q$ 的均值与方差（下面所有权重都来自 $q$）：

$$\mu_q=\mathbb E_q[z]=0.8\times0+0.2\times2=0.4,$$

$$\mathbb E_q[z^2]=0.8\times0^2+0.2\times2^2=0.8,$$

$$\operatorname{Var}_q(z)=\mathbb E_q[z^2]-\mu_q^2=0.8-0.16=0.64.$$

所以 $\mathbb E_q[z^2]=\sigma_q^2+\mu_q^2=0.64+0.16=0.8$。这只是方差定义的变形，**不要求 $q$ 是高斯**。

再对来自 $p$ 的函数 $-\log p(z)$，仍然按 $q$ 加权：

$$\mathbb E_q[-\log p(z)]=-0.8\log0.6-0.2\log0.4\approx0.5919.$$

而 $\mathbb E_q[-\log q(z)]\approx0.5004$，于是：

$$D_{\mathrm{KL}}(q\Vert p)\approx0.5919-0.5004=0.0915.$$

如果错误地把第一项改成 $\mathbb E_p[-\log p(z)]$，会得到约 $0.6730$，不是同一个问题。

#### **iv. 数值例子二：关键的高斯 KL 歧义**

假设：

$$q(z)=\mathcal N(2,0.25),\qquad p(z)=\mathcal N(0,1).$$

$\log p(z)=-\tfrac12\log(2\pi)-\tfrac{z^2}{2}$。虽然函数是 $\log p(z)$，但 **KL 定义指定对 $q$ 取期望**：

$$\mathbb E_q[\log p(z)] =-\frac12\log(2\pi)-\frac12\mathbb E_q[z^2].$$

因此应使用 **$q$ 的** 均值和方差：

$$\mathbb E_q[z^2]=\sigma_q^2+\mu_q^2=0.25+2^2=4.25.$$

绝不能因为被平均的函数包含 $p$，就代入 $\mathbb E_p[z^2]=1$；后者是在 **$p$ 下取期望** 的结果。

#### **v. 为什么均值与方差恒等式可以用？**

对任何具有有限二阶矩的分布 $q$，定义：

$$\mu_q=\mathbb E_q[z],\qquad \sigma_q^2=\mathbb E_q[(z-\mu_q)^2].$$

展开平方并利用期望的线性性：

$$\begin{aligned} \sigma_q^2 &=\mathbb E_q[z^2-2\mu_qz+\mu_q^2]\\ &=\mathbb E_q[z^2]-2\mu_q\mathbb E_q[z]+\mu_q^2\\ &=\mathbb E_q[z^2]-\mu_q^2. \end{aligned}$$

所以 $\boxed{\mathbb E_q[z^2]=\sigma_q^2+\mu_q^2}$。**这个式子本身不需要 $p$ 参与，也不要求 $q$ 是高斯；只要求这些期望存在。**

#### **vi. 与 VAE 的关系和易错点**

在 VAE 的 KL $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))$ 中，固定输入 $x$ 后，采样来自近似后验 $q_\phi(z\mid x)$。对 $\log q_\phi(z\mid x)$ 和 $\log p(z)$ 的平均都必须按照这个后验进行；在解析计算时，也要使用它的 $\mu_\phi(x),\sigma_\phi^2(x)$。

- **易错 1**：把 $\mathbb E_q[\log p]$ 理解成 $\mathbb E_p[\log p]$；下标不能随被平均函数改变。
- **易错 2**：看到先验 $p=\mathcal N(0,1)$，就把 KL 里的 $\mathbb E_q[z^2]$ 当成 1；只有 $\mathbb E_p[z^2]=1$。
- **易错 3**：误认为熵差 $H(p)-H(q)$ 等于 KL；正确关系是 $D_{\mathrm{KL}}(q\Vert p)=H(q,p)-H(q)$。

---

<a id="q14"></a>

### **14. 什么时候潜变量按近似后验采样，什么时候按先验采样？**

#### **i. 核心回答**

**固定一张输入图片 $x$ 时，VAE 训练中的潜变量采样分布确实是编码器给出的近似后验 $q_\phi(z\mid x)$。** 因此训练中计算 $\mathbb E_{q_\phi(z\mid x)}[f(z)]$ 时，按 $q_\phi(z\mid x)$ 加权。这不等于说 $q$ 在数学或现实中是潜变量唯一、真正的概率分布；它只是当前计算采用的抽样规律。

在本项目的 $d$ 维对角高斯模型中：

$$q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right),\qquad z\mid x\sim q_\phi(z\mid x).$$

#### **ii. 原理与直觉**

- **训练时**：固定 $x$，编码器得到 $\mu_\phi(x)$、$\log\sigma_\phi^2(x)$；采样 $\epsilon\sim\mathcal N(0,I)$，令 $z=\mu+\sigma\odot\epsilon$，从而 $z\mid x\sim q_\phi(z\mid x)$。
- **计算期望时**：$\mathbb E_q[\log p(z)]$ 的权重来自 $q$，即使被评价的函数是 $\log p(z)$；下标决定权重，括号里的表达式决定计算对象。
- **生成时**：不输入 $x$，从先验 $p(z)=\mathcal N(0,I)$ 采样，而不是从特定图片的 $q(z\mid x)$ 采样。
- **概率推断时**：模型意义下的真实后验是 $p_\theta(z\mid x)$；编码器的 $q_\phi(z\mid x)$ 是对它的近似，两者未必相等。

#### **iii. 数值例子**

假设固定图片 $x$ 后，某一维的编码器输出 $\mu=2$、$\sigma^2=0.25$，于是 $z\mid x\sim\mathcal N(2,0.25)$。重参数化令 $\epsilon\sim\mathcal N(0,1)$、$z=2+0.5\epsilon$。若本次抽到 $\epsilon=1.2$，则 $z=2.6$。虽然辅助噪声 $\epsilon$ 来自标准正态，最终 $z$ 却来自 $q$，**不是** 标准正态。

#### **iv. 易错点**

1. **分布与样本值不同**：$q(z\mid x)$ 是分布，$z=2.6$ 是一次样本。连续分布中，$q(z\mid x)$ 的函数值是 **概率密度** 而不是点概率；对区间/区域积分才得到概率，例如 $P(a\le Z\le b\mid x)=\int_a^b q(z\mid x)\,dz$。
2. **固定图片与随机选择图片不同**：不同图片有不同的 $q(z\mid x)$；从数据集中随机选 $x$ 再采样 $z$，总体得到的是聚合后验（通常是高斯混合），未必是单个高斯。
3. **“按 $q$ 加权”有条件**：只有当期望写成 $\mathbb E_q$ 时才使用 $q$；$\mathbb E_p$ 则按 $p$ 加权。

---

<a id="q15"></a>

### **15. 第六章「重参数化：随机采样也能传回梯度」中的公式、核心函数和测试分别在做什么？**

#### **i. 核心回答**

**VAE 训练时固定输入图片 $x$，编码器预测 $\mu_\phi(x)$ 与 $\ell_\phi(x)=\log\sigma_\phi^2(x)$；将随机性转移到一个不依赖编码器参数的标准正态噪声 $\epsilon$ 中，使得潜变量采样可以表示为一条普通、可微的计算路径。**

$$\epsilon\sim\mathcal N(0,I),\qquad z=\mu_\phi(x)+\exp\!\left(\frac{\ell_\phi(x)}2\right)\odot\epsilon.$$

其中 $\odot$ 是逐元素乘法。固定 $x$ 后，所得 $z$ 的分布仍为 $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\exp\ell_\phi(x)))$；**改变的是采样的计算表达，不是目标采样分布**。

#### **ii. 直觉：为什么重建损失能回传 Encoder**

- 若采样函数返回一个不与分布参数连接的样本，Decoder 的重建损失无法沿该样本对 Encoder 的均值、方差求出 **路径梯度**。
- 改写后，先从固定分布 $\mathcal N(0,I)$ 取 $\epsilon$；在本次反向传播中把抽到的噪声视为常数，$z$ 是 $\mu,\ell$ 的显式可微函数。
- 对每一个维度 $j$：

$$\frac{\partial z_j}{\partial\mu_j}=1,\qquad \frac{\partial z_j}{\partial\ell_j}=\frac12\exp(\ell_j/2)\epsilon_j.$$

- 若损失是 $L(D_\theta(z),x)$，则 $\partial L/\partial\mu_j=(\partial L/\partial z_j)$、$\partial L/\partial\ell_j=(\partial L/\partial z_j)\tfrac12\exp(\ell_j/2)\epsilon_j$（这里只写局部链路，不含其他损失路径及分量耦合）。然后继续通过 Encoder 网络参数 $\phi$ 链式求导。
- 在常见可微和可交换微分与期望的条件下，可以将 $\mathbb E_{q_\phi}[f_\theta(z)]$ 改写为 $\mathbb E_{\epsilon\sim \mathcal N(0,I)}[f_\theta(\mu_\phi+e^{\ell_\phi/2}\odot\epsilon)]$，并通过随机样本估计路径梯度。**这不意味着所有采样天生不可导**：PyTorch `Normal.rsample()` 就实现了重参数化路径，普通 `.sample()` 通常不提供它。

#### **iii. 数值手算：对数方差不是标准差**

设一维 $\mu=1,\sigma^2=4,\ell=\log4,\epsilon=0.5$：

$$\sigma=\sqrt{4}=\exp(\ell/2)=2,\qquad z=1+2\cdot0.5=\boxed{2}.$$

如果误用 `torch.exp(log_variance)` 当标准差，则计算为 $1+4(0.5)=3$，从而产生错误方差。

截图的二维例子中：

$$\mu=[1,0],\quad \ell=[\log4,0],\quad \epsilon=[0.5,-0.25].$$

因此 $z=[2,-0.25]$。若执行 `z.sum().backward()`，由于目标函数为 $z_1+z_2$，其梯度是：

$$\mu.\mathrm{grad}=[1,1],\qquad \ell.\mathrm{grad}=[0.5,-0.125].$$

#### **iv. 最小参考代码与 Tensor Shape**

```python
import torch

def teaching_reparameterize(mean, log_variance, noise):
    # mean, log_variance, noise: all [B, d]
    if mean.shape != log_variance.shape or mean.shape != noise.shape or mean.ndim != 2:
        raise ValueError("Expected same-shaped [B,d] tensors")
    return mean + torch.exp(0.5 * log_variance) * noise  # [B, d]

def teaching_kl(mean, log_variance):
    # 每张图片先沿 d 个潜变量维度求和，再在 B 张图片上平均；结果为标量
    if mean.shape != log_variance.shape or mean.ndim != 2:
        raise ValueError("Expected same-shaped [B,d] tensors")
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(1).mean()
```

当 $B=1,d=2,\mu=[1,0],\sigma^2=[4,1]$ 时：

$$D_{\mathrm{KL}}(q\Vert p)=\frac12(1+4-1-\log4)+0=\frac{4-\log4}2\approx \boxed{1.3068528}.$$

#### **v. 截图中的测试：各项分别验证什么**

| 测试/代码 | 验证目标 | 如何解读 |
|---|---|---|
| `assert_close(z, [[2,-0.25]])` | 确定性采样公式 | 固定 `noise` 后结果完全可复现 |
| `kl_divergence(q_dist, p_dist)` 与 `teaching_kl` 一致 | KL 手写解析式 | `Independent(Normal(...), 1)` 把最后一个维度视为同一 **联合事件** 的各独立坐标，KL 会沿该维求和 |
| `z.sum().backward()` 和 `mu.grad`, `ell.grad` | 路径梯度数值 | 对当前一次固定噪声，应该得到 $[1,1]$ 和 $[0.5,-0.125]$ |
| `torch.autograd.gradcheck(...)` | 自动微分的一致性 | 用双精度有限差分近似导数，与 PyTorch autograd 比较；不是证明整个训练能收敛 |
| `teaching_kl(zeros, zeros) == 0` | 后验等于先验 | $\mu=0,\ell=0\Rightarrow\sigma^2=1$，KL 应为零 |
| 10 万次噪声采样的 `points.mean(0)`, `points.var(0)` | 检查采样分布 | 应近似 $[1,0]$ 和 $[4,1]$，不会因为随机波动而 **恰好** 相等 |
| `mc_kl = (q_dist.log_prob(points) - p_dist.log_prob(points)).mean()` | Monte Carlo 检查 KL | 实现 $\mathbb E_q[\log q(z)-\log p(z)]$ 的样本平均；应接近但不必等于解析 KL |
| `mean.repeat(3,1)` 后 KL 不变 | Batch 归一化 | 重复相同样本后，先逐维求和再按 Batch 平均，值不变 |
| 零 logits + 0.5 目标的 BCE | 重建损失缩放 | `sigmoid(0)=0.5`，每个像素 BCE 为 $\log2$；$28\times28=784$ 像素求和、对 Batch 平均后为 $784\log2\approx543.427$ |

截图的数值检查：解析 KL $\approx1.3068528$；按 `torch.distributions` 算的 KL 完全相符；Monte Carlo 估计示例约 $1.27956$，二者差约 $0.0273$，**这是有限采样误差，不表示公式有问题**。采样均值示例为 $[0.985,-0.007]$，采样方差约 $[3.967,1.002]$，都接近理论值。

#### **vi. 常见误区**

1. **方差/标准差混用**：`exp(logvar)` 给方差；`exp(0.5*logvar)` 给标准差。这里用了 $\ell=\log\sigma^2$。
2. **把 $\epsilon$ 视为网络参数**：它是外部随机噪声，固定一次样本时才对 $\mu,\ell$ 做普通求导；不需要对随机数生成器求导。
3. **采样值与分布混淆**：`z` 是一次样本，不是分布；分布由均值和方差确定。
4. **误认为蒙特卡洛 KL 必须等于解析式**：Monte Carlo 是有限样本估计；解析式在本模型设定下没有这种采样误差。
5. **以为 `gradcheck` 等于训练正确**：它只检验局部梯度数值是否一致，不检查模型架构、优化器、数据质量和生成效果。
6. **忽略像素与 Batch 的归约**：与 KL 按潜变量维度求和类似，重建 BCE 对 784 像素求和、对 Batch 取平均；若改为像素平均，两项损失相对尺度会变化。

#### **vii. 与后续项目的关系**

这一章应能独立实现 `reparameterize(mean, logvar, noise)` 与对角高斯 KL，并完成数值、梯度和统计检验。真正的项目源码仍应写入 `src/minigen/autoencoders/vae_core.py`，不能在正式实现中偷偷调用 Notebook 参考答案。训练和生成图像之前，先确认这两个核心函数数学正确。

---

## **五、VAE 损失与灰度 BCE**

<a id="q16"></a>

### **16. VAE 的损失由哪些部分组成？**

#### **i. 核心回答**

**标准 VAE 的负 ELBO（每张图像）由两部分构成：重建项 + KL 项。** 重建项要求潜变量能够解释输入图像；KL 项要求编码器的近似后验不要过度偏离可采样的先验。训练时这两项目标需要权衡，不能只压低其中一项。

$$\boxed{\mathcal L_{\mathrm{VAE}}(x)=\underbrace{-\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{重建项：期待能解释图像}}+\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))}_{\text{KL 项：靠近先验}}.}$$

- $x$：当前输入图像；$z$：潜变量的一次随机样本。
- $q_\phi(z\mid x)$：Encoder 给定图像后预测的对角高斯；$p(z)=\mathcal N(0,I)$：先验；$p_\theta(x\mid z)$：Decoder 的图像观测模型。
- 负号来源：从 **最大化** ELBO 改为优化器执行的 **最小化** 负 ELBO。
- 重建项有期望是因为同一张图像可从 $q_\phi(z\mid x)$ 抽出不同的 $z$；本项目训练通常对每张图像抽一次样本来估计该期望，并非精确积分。

#### **ii. 重建项：本项目的 BCEWithLogits**

对严格二值像素 $x_i\in\{0,1\}$，给定 $z$ 后假设 784 个像素条件独立，Decoder 给出 logit $a_i$，对应概率 $\pi_i=\operatorname{sigmoid}(a_i)$：

$$-\log p_\theta(x\mid z)=-\sum_{i=1}^{784}\bigl[x_i\log\pi_i+(1-x_i)\log(1-\pi_i)\bigr].$$

这个负对数似然恰好是按像素求和的二元交叉熵。**注意本项目训练的是灰度 MNIST，目标 $x_i\in[0,1]$，因此实际使用软标签 BCE 代理目标；不能将其宣称为连续灰度图像的精确 Bernoulli 似然。** 直接使用 `binary_cross_entropy_with_logits`，比先 sigmoid 再计算 BCE 更数值稳定。

#### **iii. KL 项：对角高斯的解析式**

若 $q_\phi(z\mid x)=\mathcal N(\mu,\operatorname{diag}(\sigma^2))$ 且 $p(z)=\mathcal N(0,I)$，则：

$$D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))=\frac12\sum_{j=1}^{d}\bigl(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\bigr).$$

- $\mu_j^2/2$ 惩罚均值远离 0。
- $(\sigma_j^2-1-\log\sigma_j^2)/2$ 惩罚方差偏离 1（无论太大还是太小）。
- 当 $\mu_j=0,\ \sigma_j^2=1$ 时，该维 KL 为 0；**这并不意味着所有图像最终都应有零 KL**，因为那可能损失输入信息并导致重建退化。

#### **iv. 手算例子**

假设一个极小的二像素二值图像，目标 $x=[1,0]$，Decoder 的概率预测 $\pi=[0.8,0.2]$，则：

$$\mathcal L_{\rm recon}=-\log 0.8-\log(1-0.2)=-2\log0.8\approx0.4463.$$

又假设二维近似后验 $\mu=[1,0]$，方差 $\sigma^2=[4,1]$：

$$\mathcal L_{\rm KL}=\tfrac12(1+4-1-\log4)+\tfrac12(0+1-1-\log1)\approx1.3069.$$

在标准权重 $\beta=1$ 的单次采样估计中，合计约 $0.4463+1.3069=1.7532$。这只是刻意选取的可手算示例，并非真实训练损失。

#### **v. PyTorch 对应与 Tensor Shape**

```python
# x, logits: [B, 1, 28, 28]; mu, logvar: [B, 16]
# 单图 784 个像素求和，随后对 B 张图求平均
recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
recon = recon.flatten(1).sum(dim=1).mean()

# 每张图 16 个潜变量维度求和，然后对 batch 平均
kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1).mean()

loss = recon + kl  # 标准 VAE 的负 ELBO 一次采样估计/软 BCE 代理实现
```

形状顺序：像素项 `[B,1,28,28] -> [B,784] -> [B] -> []`；KL 项 `[B,16] -> [B] -> []`。要使两项的相对强度符合预期，必须确认各自的 `sum`/`mean`，不能随意把像素求和换成像素平均。

实验若采用 $\mathcal L_\beta=\mathcal L_{\rm recon}+\beta\mathcal L_{\rm KL}$，$\beta=1$ 对应标准 VAE 权重；其他值是在改变重建与 KL 的权衡，**并非都等于同一个标准负 ELBO**。这里的 $\beta$ 与 DDPM 加噪调度的 $\beta_t$ 不是一回事。

#### **vi. 记忆口诀**

**重建项管“像不像原图”，KL 管“潜变量好不好按先验采样”。单图内部先求和，batch 再求平均。**

---

<a id="q17"></a>

### **17. BCE 如何处理灰度软目标？**

#### **i. 核心回答**

**本周 MNIST VAE 的目标图像 `x` 是真实灰度值，像素可取 $[0,1]$ 内的任意数（例如 $0.3$），不局限于二值的 0、1。** Encoder 接收 `x`，Decoder 输出未经 Sigmoid 的 `logits`；`sigmoid(logits)` 得到模型对每个像素的预测值 $\pi$。训练时直接使用 `binary_cross_entropy_with_logits(logits, x)`，它支持 $x_i\in[0,1]$ 的软目标。

这里的 $x_i$ 是 **真实图片的像素强度**；$\pi_i$ 在 Bernoulli 解码器模型下解释为像素为 1 的概率。对于灰度数据，使用软目标 BCE 是一种训练选择，**不能把灰度 $0.3$ 自然地等同于真实世界中“白色概率是 30%”，也不能说它就是连续灰度观测的精确 Bernoulli 似然。**

#### **ii. 一个像素的输入、输出和手算**

假设输入图片中某一像素的真实灰度为 $x_i=0.3$，Decoder 对该位置输出 logit $a_i=0$。那么

$$\pi_i=\operatorname{sigmoid}(a_i)=\frac{1}{1+e^{-a_i}}=0.5.$$

软目标 BCE：

$$\begin{aligned} L_i&=-[x_i\log\pi_i+(1-x_i)\log(1-\pi_i)]\\ &=-[0.3\log0.5+0.7\log0.5]\\ &=0.693147\ldots. \end{aligned}$$

保持 $x_i=0.3$ 不变：

| 模型预测 $\pi_i$ | BCE | 解释 |
|---:|---:|---|
| $0.3$ | $0.610864$ | 该软目标下的最小损失 |
| $0.5$ | $0.693147$ | 预测偏离目标 |
| $0.8$ | $1.193550$ | 预测偏离更大 |

**易错点：** 当 $x_i\in(0,1)$ 且预测 $\pi_i=x_i$ 时，BCE 仍不为零，因为最小值是该软目标的二元熵：

$$H(x_i)=-x_i\log x_i-(1-x_i)\log(1-x_i).$$

#### **iii. 为什么软目标 BCE 有数学意义？**

假设另有一个二元随机标签 $Y\in\{0,1\}$，且 $P(Y=1)=t$。对模型给出的成功概率 $\pi$，二分类负对数似然的期望为：

$$\begin{aligned} \mathbb E_Y[-Y\log\pi-(1-Y)\log(1-\pi)] &=-t\log\pi-(1-t)\log(1-\pi). \end{aligned}$$

它就是以 $t$ 为软目标的 BCE。**这是一种期望损失的解释，而不是在说原始 MNIST 灰度像素天然就是 Bernoulli 的概率参数。**

#### **iv. 四个像素的完整前向示例**

令真实灰度图和 Decoder logits 为

$$x=\begin{bmatrix}0&0.25\\0.75&1\end{bmatrix}, \qquad a=\begin{bmatrix}-2&0\\\log4&2\end{bmatrix}.$$

经 Sigmoid 得到预测值：

$$\pi=\operatorname{sigmoid}(a) \approx\begin{bmatrix}0.1192&0.5\\0.8&0.8808\end{bmatrix}.$$

| 像素 | $x_i$（真实灰度） | $a_i$（logit） | $\pi_i$（预测） | BCE |
|---|---:|---:|---:|---:|
| 左上 | 0 | -2 | 0.1192 | 0.1269 |
| 右上 | 0.25 | 0 | 0.5 | 0.6931 |
| 左下 | 0.75 | $\log4$ | 0.8 | 0.5697 |
| 右下 | 1 | 2 | 0.8808 | 0.1269 |

对单张图片的 4 个像素求和，得到 $L_{\mathrm{recon}}\approx1.516720$。本周真实 MNIST 有 784 个像素；每张图先对像素求和，然后对 batch 求平均。

#### **v. PyTorch 可运行代码与 Tensor Shape**

```python
import math
import torch
import torch.nn.functional as F

# 一张 2x2 灰度图： [B, C, H, W] = [1, 1, 2, 2]
target = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
logits = torch.tensor([[[[-2.0, 0.0], [math.log(4.0), 2.0]]]])

predicted_gray = torch.sigmoid(logits)          # [1,1,2,2]
per_pixel_bce = F.binary_cross_entropy_with_logits(
    logits, target, reduction="none"             # [1,1,2,2]
)
per_image_loss = per_pixel_bce.flatten(1).sum(1) # [1]
recon_loss = per_image_loss.mean()                # 标量

print("真实灰度:", target)
print("预测灰度:", predicted_gray)
print("每像素 BCE:", per_pixel_bce)
print("单图重建损失:", recon_loss.item())  # 约 1.51672
```

**`reduction="none"` 保留每个像素的损失；`flatten(1).sum(1)` 对每张图像的所有像素求和；最后 `mean()` 对 batch 求平均。** 训练用 logits 直接调用 BCEWithLogits 更数值稳定；显示图像再调用 Sigmoid。

#### **vi. 易错点**

1. `target` 是输入图像的真实像素，不是 Encoder 输出，也不是潜变量；`logits` 来自 Decoder。
2. `sigmoid(logits)` 是预测像素的概率参数，**不是另一个必须喂进 `binary_cross_entropy_with_logits` 的输入**；该函数会内部处理 Sigmoid。
3. 软目标 BCE 对齐真实灰度是一个实用训练准则，严格二值 Bernoulli 似然只对应观测取值 $0$ 或 $1$。
4. BCE 与 MSE 不同：软目标完全匹配时，BCE 通常仍大于零。
5. Loss 的整体尺度由像素维度与 batch 维度的归约方式决定；与 KL 相加时要明确求和与平均方式。

---

<a id="q18"></a>

### **18. 目标灰度 $x_i$ 和预测概率 $\pi_i$ 有什么关系？**

#### **i. 核心回答**

**在本周灰度 MNIST 的可视化与训练语境下，可以把 $x_i$ 看作真实灰度，把 $\pi_i=\operatorname{sigmoid}(a_i)$ 看作预测灰度；两者均在 $[0,1]$ 中，可以在同一个像素位置对照。** 但它们的概率语义并不完全一样：$x_i$ 是观测到的灰度目标，$\pi_i$ 严格来说是 Decoder 所定义的 Bernoulli 分布参数。

#### **ii. 数学意义和数值例子**

若 $a_i=0$，则 $\pi_i=\operatorname{sigmoid}(0)=0.5$。若输入图像这一像素的灰度为 $x_i=0.3$，训练时比较的是目标 $0.3$ 和预测参数 $0.5$：

$$\operatorname{BCE}(x_i,\pi_i) =-x_i\log\pi_i-(1-x_i)\log(1-\pi_i) =-0.3\log0.5-0.7\log0.5\approx0.6931.$$

对于二值随机像素 $X_i\mid z\sim\operatorname{Bernoulli}(\pi_i)$，有

$$\mathbb E[X_i\mid z]=0\cdot(1-\pi_i)+1\cdot\pi_i=\pi_i.$$

因此 $\pi_i$ 可以直接作为 **预期灰度** 来显示；若实际从 Bernoulli 分布采样，则得到的是随机的 0 或 1，而不是固定的 $\pi_i$。

#### **iii. 代码对应**

```python
logits = decoder(z)                  # [B, 1, 28, 28]，任意实数
pred_gray = torch.sigmoid(logits)   # [B, 1, 28, 28]，范围 [0, 1]
recon = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, x, reduction="none"
).flatten(1).sum(1).mean()
```

#### **iv. 易错点**

- $x_i=0.3$ 只是灰度强度，不意味着现实中这个像素天然服从“30% 为白”的 Bernoulli 分布；把连续灰度当作软目标是训练准则。
- $\pi_i$ 同时可用于可视化的预测灰度，但其形式化定义仍是 Bernoulli 参数；不要将“预测期望”和“一次实际采样”混为一谈。
- `binary_cross_entropy_with_logits` 接收原始 `logits`，不要先做 Sigmoid 再传给这个函数。

---

## **六、实验对照与潜空间诊断**

<a id="q19"></a>

### **19. VAE 对照实验（$d=16,\beta=0/0.5/1/4$；$d=2,\beta=1$）的现象、趋势与结论是什么？**

#### **i. 核心回答**

**对于本次相同步数的 $d=16$ 实验，$\beta$ 从 $0\to0.5\to1\to4$ 时，KL 逐渐下降、重建 BCE 逐渐上升；固定 $\beta=1$ 将维度从 16 降至 2 时，重建 BCE 明显上升。** 这体现了重建能力与先验约束之间的权衡。

#### **ii. 实验结果（各训练 1000 步）**

| 实验 | 重建 BCE/图 ↓ | KL/图 | 主要观察 |
|---|---:|---:|---|
| `d16_beta0` | **97.099** | **345.275** | 重建最好，但无 KL 惩罚，KL 随训练大幅上升 |
| `d16_beta05` | 105.667 | 23.329 | 相较于无约束模型，KL 降幅很大但重建仅部分退化；温和约束候选 |
| `d16_beta1` | 114.561 | 15.528 | 重建与先验约束相互权衡 |
| `d16_beta4` | 143.709 | **5.522** | KL 更低，但重建明显变差 |
| `d2_beta1` | 156.953 | 5.125 | 二维瓶颈下重建更困难，可用于潜空间可视化 |

**趋势：** 五组重建 BCE 随训练逐步下降；`d16_beta0` 的 KL 从 step 100 的 69.170 升至 step 1000 的 345.275；`d16_beta05`、`d16_beta1`、`d16_beta4` 的 KL 分别为 23.329、15.528、5.522。维度变化时，不能仅因总 KL 更小就认定分布“更好”，因为 KL 是对潜变量维度求和。

#### **iii. 结论与局限**

- $\beta=0$ 时，损失中没有 KL 项：`loss = recon`，但 **监测的 KL 值仍可以很大**；重建很好不代表从先验 $p(z)=\mathcal N(0,I)$ 采样能生成好图。
- $\beta=4$ 的低 KL 不等于生成质量最好；它也可能损失重建信息。
- 本次为单种子、有限步数的对照，曲线未必收敛；必须结合先验生成图像、验证结果与重复训练判断生成质量。
- **不同 $\beta$ 下，总损失 $L_{\beta}=L_{\mathrm{recon}}+\beta L_{\mathrm{KL}}$ 的定义不同，不能只比较总损失数值给模型排名。**

---

<a id="q20"></a>

### **20. 随机重建与均值重建 BCE 的对照实验，有什么现象和结论？**

#### **i. 核心回答**

随机重建使用 $z=\mu+\sigma\odot\epsilon$；均值重建使用 $z=\mu$。**本次四个 $d=16$ 模型的均值重建 BCE 都低于随机重建，且 $\beta$ 越大，两者差距越明显。**

#### **ii. 实验结果（step 1000）**

| 实验 | 随机重建 BCE/图 ↓ | 均值重建 BCE/图 ↓ | 差距（随机－均值） |
|---|---:|---:|---:|
| `d16_beta0` | 97.099 | 97.096 | **0.003** |
| `d16_beta05` | 105.667 | 101.871 | **3.796** |
| `d16_beta1` | 114.561 | 109.264 | **5.297** |
| `d16_beta4` | 143.709 | 132.648 | **11.061** |

#### **iii. 原理与结论**

- $\beta=0$ 的两种重建几乎一致：**可能** 是后验方差很小，使 $\sigma\epsilon$ 影响很弱；但也可能与 Decoder 对噪声的敏感度有关，不能只凭 BCE 差距认定方差已趋零。
- $\beta$ 增强后，潜变量采样扰动及 Decoder 的敏感性共同影响重建；本次结果是随机重建比均值重建更难，但 **并非均值重建永远优于随机重建**。
- **随机重建不是先验生成**：前者从 $q_\phi(z\mid x)$ 采样，后者从 $p(z)$ 采样。以上表格不足以判断先验生成图像的质量。
- 下一步宜对照展示均值重建、随机重建、先验生成图片；并记录 `exp(logvar).mean()` 与 `mu.square().mean()`，验证关于方差的猜测。

---

<a id="q21"></a>

### **21. 如何阅读二维先验分位数网格？**

#### **i. 核心回答**

**它是一张二维潜空间的“生成地图”，不是 144 张互相独立随机采样的数字图。** 对已经训练的二维 VAE，固定一组网格坐标 $z=(z_0,z_1)$，将每个坐标送入 Decoder，并显示 $\operatorname{sigmoid}(D_\theta(z))$；因此每个格子对应一个 **确定的潜变量坐标** 和一张像素概率图。

#### **ii. 如何读图：行、列究竟代表什么？**

本实验先用标准正态的分位数 $\Phi^{-1}$ 在 $0.05$ 到 $0.95$ 之间选择 12 个值，约覆盖 $[-1.645,1.645]$。使用 `torch.meshgrid(ticks, ticks, indexing="ij")` 时：

- **从左到右**：$z_1$ 由小变大，$z_0$ 保持不变。
- **从上到下**：$z_0$ 由小变大，$z_1$ 保持不变。
- 左上角：$(-1.645,-1.645)$；右上角：$(-1.645,+1.645)$；左下角：$(+1.645,-1.645)$；右下角：$(+1.645,+1.645)$。

因此，第一行沿横向扫描潜变量 $z_1$，第一列沿纵向扫描 $z_0$。在本次图像中，第一行从左侧类似 **0** 的形状逐渐向右侧类似 **1** 的形状变化；第一列向下逐渐出现类似 **8/3/9** 的轮廓。这是对 **当前训练模型输出的观察**，并非每个维度固定对应某种数字语义。

#### **iii. 关键代码与 Tensor Shape**

```python
normal = torch.distributions.Normal(0, 1)
ticks = normal.icdf(torch.linspace(0.05, 0.95, 12))  # [12]
coords = torch.stack(torch.meshgrid(ticks, ticks, indexing="ij"), -1).reshape(-1, 2)  # [144,2]
images = model2.decode(coords.to(DEVICE)).sigmoid()  # 通常 [144,1,28,28]
```

`coords` **不是** 从标准正态分布独立抽样的 144 个点，而是按标准正态 **分位数构造的规则网格**；调用 Decoder 时没有输入图片，也没有调用 Encoder。展示的图像是 Bernoulli 参数（像素期望），并非额外进行 Bernoulli 二值采样。

#### **iv. 为什么数字会渐变？能证明什么？**

- 相邻网格的潜变量接近，连续神经网络的输出通常也相近；插值可写为 $z(t)=(1-t)z_A+t z_B$。
- 这有助于观察 Decoder 的 **连续映射** 及潜空间的局部变化，不能证明“每一维有唯一语义”“图中每个位置都是真实数字”或“语义已经解耦”。
- 二维潜空间便于可视化，但表达能力受限；图中部分混合、模糊数字不奇怪。
- 注意图像网格的 **行对应 $z_0$、列对应 $z_1$**，而另一张散点图以 $z_0$ 为横轴、$z_1$ 为纵轴，两者坐标轴朝向不同，不能直接叠放比较。

---

<a id="q22"></a>

### **22. 如何从损失、生成和潜空间三个层面诊断 VAE？**

#### **i. 核心回答**

1. **按维 KL 很小，说明该维编码分布可能没有携带多少关于输入的信息。** 因为平均 $D_{\mathrm{KL}}(q_\phi(z_j\mid x)\Vert p(z_j))$ 很小，意味着这一维的条件后验接近先验。但它不是通用的后验坍塌证明；还应检查 Decoder 对该维度的依赖、重建是否受到影响。
2. **聚合后验更接近标准正态通常有利于先验生成，但一、二阶矩接近不等于分布相同。** 聚合后验 $q_\phi(z)=\mathbb E_x[q_\phi(z\mid x)]$ 的每维方差可用全方差公式估计：

   $$\operatorname{Var}_{q(z)}(z_j)=\operatorname{Var}_{x}(\mu_j(x))+\mathbb E_x[\exp(\ell_j(x))].$$

   即使均值接近 $0$、方差接近 $1$，也不能证明整个聚合后验是 $\mathcal N(0,I)$，更不能单凭矩指标保证生成质量。
3. **二维潜空间是观察编码结构和 Decoder 连续变化的教学工具。** 数字类别在 Encoder 均值散点图中出现一定聚集，但存在重叠；规则坐标网格对应的数字轮廓逐渐改变，不意味着类别可完美分离或各维语义解耦。

#### **ii. 对 MiniGen 项目的意义：三层评估**

| 层面 | 检查什么 | 关键提醒 |
|---|---|---|
| **损失** | Reconstruction BCE 与 KL 的权衡；$\beta$、$d$ 对指标的影响 | 重建越好不一定先验生成越好 |
| **生成** | 均值重建、随机重建、先验生成的实际图片 | 训练后验采样 $q(z\mid x)$ 不等于先验采样 $p(z)$ |
| **潜空间** | 按维 KL、聚合后验统计、二维散点与解码网格 | 低 KL、矩匹配、连续变化各只能支持有限结论 |

**最终结论：** 适度 KL 约束能改善从先验采样的可用性；约束过强可能削弱重建；潜空间维度及其实际使用情况也影响结果。要用三类证据共同判断，不能仅凭某一个 Loss 或一张网格图断言模型好坏。

---

<a id="q23"></a>

### **23. 如何诊断 VAE 重建模糊？**

#### **i. 核心回答**

**重建模糊是观察到的现象，不是对原因的诊断。** 在 VAE 中，模糊可能来自模型能力、训练程度、KL 约束、潜变量信息瓶颈、重建目标的性质，或 Decoder 对潜变量的使用不足。**不能仅凭“重建图模糊”，就宣称出现了 Posterior Collapse（后验坍塌）。**

#### **ii. 本次 MiniGen 的实际观察与解释边界**

| 配置 / 实验 | 观察到的现象 | 可以说什么 | 不能直接断言什么 |
|---|---|---|---|
| $d=16,\ \beta=0$ | 重建较清晰；KL 很高；从先验采样的数字比较难辨认 | 仅优化重建时，编码分布可能明显偏离先验 | “KL 高一定意味着每一维都有效使用” |
| $d=16,\ \beta=4$ | KL 更低；重建更模糊；部分先验样本仍不清楚 | 更强约束与更差重建在本次实验中同时出现 | “已经证明后验坍塌”或“先验生成必然最好” |
| $d=2,\ \beta=1$ | 类别散点有重叠；解码网格出现模糊或混合数字 | 二维瓶颈下，表示能力和类别分离有局限 | “数字混合就是坍塌” |
| 9→4 线性插值 | 两端可辨认，中间连续出现混合形态 | 该条潜变量路径上的 Decoder 输出连续变化 | “整个潜空间都合理”或“模型已理解数字语义” |

#### **iii. 重建模糊的六类可能原因：机制与验证方法**

| 可能原因 | 为什么可能模糊？ | 如何验证或排查？ |
|---|---|---|
| **① 模型容量不足** | MLP Encoder/Decoder 难以表达复杂的笔画、局部结构，输出可能更粗糙 | 固定数据和训练预算，增加网络宽度/深度，或比较 CNN VAE；观察验证重建 BCE 和图像细节是否改善 |
| **② 训练不充分** | 参数尚未充分优化，模型还没有学会把输入细节稳定重建出来 | 查看训练/验证 BCE 随步数变化，并比较后续 checkpoint；验证损失仍下降时，继续训练可能有收益 |
| **③ KL 约束过强** | 较大的 $\beta$ 让保留输入信息的代价升高，编码分布更接近先验，可能牺牲重建细节 | 在相同 $d$、数据和训练条件下比较 $\beta=0,0.5,1,4$ 的重建 BCE、按维 KL 与样本图像；必要时尝试 KL warm-up |
| **④ 重建目标与显示方式** | BCE 的 Bernoulli 像素概率参数经过 Sigmoid 后，被直接当作灰度值显示；若模型对一个细节存在多种可能，输出的像素均值/概率图可能显得“平均化”或模糊 | 区分 logits、Sigmoid 概率图与真正的 Bernoulli 二值采样；对比重建 BCE 和可视化；不要把灰度软目标 BCE 当作连续灰度的精确归一化似然 |
| **⑤ 潜变量维度或有效信息容量不足** | $d$ 太小，或虽然有 $d=16$、却只有少数维度携带输入信息，细节无法完整通过瓶颈 | 比较 $d=2$ 与 $d=16$；检查按维 KL、编码均值变化与删去/扰动潜维后的重建质量 |
| **⑥ Decoder 没有充分利用 $z$** | 即使输入图片不同，Decoder 输出也可能对 $z$ 不敏感，倾向给出相似的平均图像 | 固定 Decoder，替换/扰动 $z$，比较输出是否显著变化；测量重建性能是否优于忽略 $z$ 的基线；结合后验参数随 $x$ 的变化判断 |

**重要辨析：** “像素概率图看起来模糊”与“模型因 KL 过强而没有利用潜变量”是两种不同问题。只有联合观察损失、潜变量与 Decoder 行为，才能缩小原因范围。

#### **iv. 什么才是 Posterior Collapse（后验坍塌）？**

在 VAE 中，常指近似后验对于不同输入几乎都接近先验，潜变量很少承载输入信息，例如：

$$q_\phi(z\mid x)\approx p(z)=\mathcal N(0,I),\quad\text{对多数 }x\text{ 成立。}$$

同时 Decoder 对潜变量的依赖很弱。检查时应结合：

1. **统计证据**：多数潜变量维度平均 KL 很小，且 $\mu_\phi(x)$、$\sigma^2_\phi(x)$ 对输入变化不敏感。
2. **行为证据**：改变或打乱潜变量后，Decoder 输出及重建质量变化很小；或输出趋于不依赖输入的相似形态。
3. **任务证据**：检查重建损失、重建样本，比较输入依赖的生成与无条件/无潜变量基线。

单一条件不能构成普遍适用的“证明门槛”：**某几维 KL 小不等于整体后验坍塌；中间插值形状混合也不是坍塌证据。**

#### **v. 建议的最小诊断顺序**

**先分清生成路径**：是 `decode(mu)` 均值重建模糊、`decode(mu + sigma*eps)` 随机重建模糊，还是 `decode(z_prior)` 先验生成模糊？三者可能原因不同。然后：

1. **看曲线**：训练/验证重建 BCE 是否还在下降，以判断延长训练是否可能改善重建。
2. **看 KL 与分布**：记录总 KL、按维 KL、$\mu$ 统计、后验方差；对照 $\beta$ 与 $d$ 的变化。
3. **做干预**：固定模型，用均值重建与随机重建比较采样影响；扰动/交换 $z$ 检查 Decoder 是否真正使用它；必要时增加容量或延长训练。
4. **核对目标和显示**：确认 BCE 的像素求和与 batch 平均、输入范围 $[0,1]$、`sigmoid(logits)` 的显示方式一致。

#### **vi. 核心总结**

> **观察 → 提出可能机制 → 做针对性对照 → 得出有证据支持的结论。**
>
> 本次实验支持“重建质量—KL 约束—先验采样”之间存在权衡，但还不足以把某张模糊图片的原因唯一归结为后验坍塌。

---

<a id="q24"></a>

### **24. 如何解读 $\beta=0.5$ 的对照结果？**

#### **i. 核心回答**

加入 $\beta=0.5$ 后，可以更细致地观察从“只重建”到“对潜变量加约束”的变化。在相同 1000 updates 的 $d=16$ 对照中，重建 BCE 与 KL 如下：

| $\beta$ | 随机重建 BCE/图 ↓ | 均值重建 BCE/图 ↓ | KL/图 |
|---:|---:|---:|---:|
| 0 | 97.099 | 97.096 | 345.275 |
| **0.5** | **105.667** | **101.871** | **23.329** |
| 1 | 114.561 | 109.264 | 15.528 |
| 4 | 143.709 | 132.648 | 5.522 |

**观察：** $\beta=0\to0.5$，重建 BCE 上升约 $8.568$，KL 下降约 $321.946$；$\beta=0.5\to1$，重建 BCE 再上升 $8.894$，KL 降低 $7.801$。因此，**$\beta=0.5$ 在这次训练预算下是值得进一步考察的折中点**，但不能由单次结果认定它是生成质量最优超参数。

#### **ii. 为什么需要同时看先验样本？**

- $\beta=0$ 的均值/随机重建非常接近，但固定先验生成网格中有许多难辨认数字；这与编码分布严重偏离先验的风险一致。
- $\beta=0.5$ 的先验样本中，可辨认数字比 $\beta=0$ 更常见，仍有模糊笔画。
- $\beta=1$ 与 $4$ 的部分先验样本同样可辨认；单幅 16 样本网格不足以量化谁最好。

**进一步评估建议：** 使用更多固定随机种子生成图像；区分重建质量、先验采样的可辨认度、类别覆盖和多样性。所有观察仅支持当前 MNIST MLP VAE 的局部实验结论。

---

## **七、可视化、断点恢复与实现**

<a id="q25"></a>

### **25. 这段 `show_rows()`、固定验证集重建和 `fixed_prior` 生成代码在干什么？**

#### **i. 核心回答**

这不是训练，而是把 **同样的验证图片和先验随机向量** 交给不同模型做可视化对比。

| 显示内容 | 潜变量来源 | 对应数据流 | 回答什么问题 |
|---|---|---|---|
| 原图 | 无 | $x$ | 参考是什么 |
| 均值重建 | $z=\mu_\phi(x)$ | $x\to E\to\mu\to D$ | 不加随机扰动能否还原 |
| 随机重建 | $z=\mu+\sigma\epsilon$ | $x\to E\to q\to z\to D$ | 后验采样是否影响还原 |
| 先验生成 | $z\sim\mathcal N(0,I)$ | $z\to D$ | 没有输入图片，能否直接生成 |

#### **ii. 核心代码及 Shape**

```python
# x: [8,1,28,28], mean/logvar: [8,16]
mean, logvar = model.encode(x)
mean_img = model.decode(mean).sigmoid()  # [8,1,28,28]
random_img = model.decode(reparameterize(mean, logvar, eps)).sigmoid()

# 同一组 16 个标准高斯输入用于所有 d=16 模型
fixed_prior = torch.randn(16, 16, generator=torch.Generator().manual_seed(2026))
prior_imgs = model.decode(fixed_prior).sigmoid()  # [16,1,28,28]
```

`show_rows()` 只是负责把每批图片放进网格并保存；`torch.no_grad()` 关闭梯度记录；`.eval()` 切到评估模式。**固定同一输入有利于横向对比，但不同模型的相同坐标不一定具有相同语义。**

**易错点：** Decoder 最后一层输出 logits，`.sigmoid()` 产生用于展示的像素概率图；图上不是对每个 Bernoulli 像素再做一次随机二值采样。

---

<a id="q26"></a>

### **26. 最后一块潜空间插值和新进程加载 Checkpoint 在做什么？**

#### **i. 核心回答**

这部分有两个独立验收：**潜空间插值的视觉连续性** 与 **重新加载权重后的前向输出一致性**。

#### **ii. 9 → 4 的均值潜变量线性插值**

选择固定验证图片中两张图，提取其后验均值 $\mu_A,\mu_B\in\mathbb R^{16}$，在线段上取 9 个点：

$$\boxed{z(\alpha)=(1-\alpha)\mu_A+\alpha\mu_B,\quad \alpha\in\{0,0.125,\dots,1\}.}$$

```python
endpoints, _ = model.encode(val_x[:2])  # [2,16]
alpha = torch.linspace(0, 1, 9)[:, None]  # [9,1]
path = (1 - alpha) * endpoints[0] + alpha * endpoints[1]  # [9,16]
images = model.decode(path).sigmoid()  # [9,1,28,28]
```

**观察：** 图片沿这条路径从较像 9 逐渐变得较像 4，中间有混合形态。这说明 **该路径上 Decoder 输出连续变化**；不能证明全潜空间无坏点，也不能证明数字语义被清晰解耦。

#### **iii. 新进程恢复权重的前向一致性**

- 先保存一组固定潜变量和“原进程中的 Decoder logits”。
- 在新 Python 进程中重新创建 `TinyVAE(16)`，从保存的 `state_dict` 加载权重。
- 用同一组潜变量重新计算 logits，比较最大绝对误差：

$$\max\lvert \mathrm{logits}_{\mathrm{fresh}}-\mathrm{logits}_{\mathrm{reference}}\rvert  \approx5.72\times10^{-6}.$$

这一结果通过给定容差测试，说明 **固定输入下的前向推理恢复成功**。它不代表优化器、RNG 或 DataLoader 状态已经能够完整恢复训练。

---

<a id="q27"></a>

### **27. “训练恢复：保存参数为什么还不够？”以及 `StatefulBatchStream` 在测试什么？**

#### **i. 核心回答**

**能恢复推理** 与 **能从训练中断处无缝继续** 是两件事。Adam 的动量状态、随机噪声序列与数据取样位置都会影响下一步更新。

| 应保存的状态 | 缺失后可能发生什么 |
|---|---|
| 模型权重、模型配置 | 无法继续此前的网络计算 |
| 优化器状态（Adam 一/二阶动量、step） | 下一步参数更新不一致 |
| 当前 global step、调度器状态 | 学习率和进度错位 |
| RNG（Python/NumPy/Torch/CUDA 等实际使用者） | 重参数化噪声、shuffle 不一致 |
| 数据流 permutation、cursor、epoch、shuffle RNG | 恢复后样本重复或跳过 |
| EMA（如果实际使用） | 去噪或评估所用 EMA 权重不一致 |

#### **ii. 实验如何设计？**

取 19 张图片、batch size=7，一个 epoch 恰好分为 **7、7、5** 三批，用短 Batch 和跨 epoch 增加边界检查：

```text
相同模型初始化
├── A：连续训练 step 1 → ... → 8
└── B：训练到 step 3 → 保存全部训练状态 → 新建模型 → 恢复到 step 8
```

Notebook 断言 `max_parameter_error == 0`，并检查第 4～8 步的 Loss 列表完全一致。**这验证了本次 CPU 确定性设定下训练恢复的逐步一致性**，不是“所有机器、多卡、混合精度都逐位一致”的保证。

**易错点：** 仅保存 `epoch` 不能知道 epoch 内数据流的具体位置；`num_workers=0` 的简化设置不自动推广到多 worker 预取数据管线。

---

<a id="q28"></a>

### **28. `TinyVAE` 每个函数负责什么？完整 Tensor 路径是什么？**

#### **i. 核心回答**

`TinyVAE` 是一个 MNIST MLP 编解码壳，只管理网络结构，**重参数化函数由调用方注入，KL 和训练循环在类外计算**。

| 方法 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `__init__(latent_dim)` | $d=16$（默认） | 模型层 | 建 Encoder、两个 Head 和 Decoder |
| `encode(images)` | `[B,1,28,28]` | `mean, logvar`：各 `[B,d]` | 用共享隐藏特征预测对角高斯参数 |
| `decode(z)` | `[B,d]` | logits `[B,1,28,28]` | 用潜变量预测像素 logits |
| `forward(images,noise,reparameterize_fn=...)` | 图片、`[B,d]` 噪声和函数 | `logits,mean,logvar` | 串联编码、采样和解码 |

#### **ii. 完整路径（$B$ 表示 Batch Size）**

```text
images [B,1,28,28]
    │ Flatten
    ▼
  [B,784]
    │ Linear(784,256) → ReLU → Linear(256,128) → ReLU
    ▼
  [B,128]
    ├── mean_head(128,d) ──→ μ [B,d] ──┐
    └── logvar_head(128,d) → ℓ [B,d] ──┤
                                        │ reparameterize(μ,ℓ,ε)
noise ε [B,d] ──────────────────────────┘
                                        ▼
                                      z [B,d]
                                        │ Linear(d,128) → ReLU
                                        │ Linear(128,256) → ReLU
                                        │ Linear(256,784) → reshape
                                        ▼
                                logits [B,1,28,28]
                                        │
                     ┌──────────────────┴───────────────────┐
                     ▼                                      ▼
                   BCE(x)                         KL(μ, logvar)
                     └──────────────────┬───────────────────┘
                                        ▼
                                 BCE + β × KL
                                        ▼
                                    backward()
```

**注意：** 上图中 BCE、KL 与 `backward()` 属于外部训练循环，不在 `TinyVAE.forward()` 中。`encode()` 返回分布参数而非 $z$；`decode()` 可以独立接受先验 $z\sim\mathcal N(0,I)$ 进行生成。

---

<a id="q29"></a>

### **29. 如何实现重参数化与对角高斯 KL？为什么要先求和再平均？**

#### **i. 核心回答**

两个核心实现的 **数学式与维度归约正确**（假设 `mean`、`log_variance` 与 `noise` 同形，且噪声确实来自独立标准正态分布）。

```python
import torch

def reparameterize(mean, log_variance, noise):
    return mean + torch.exp(0.5 * log_variance) * noise

def kl_standard_normal(mean, log_variance):
    per_coordinate = 0.5 * (
        mean.square() + log_variance.exp() - 1 - log_variance
    )
    return per_coordinate.flatten(start_dim=1).sum(dim=1).mean()
```

对 $[B,16]$，`flatten(1)` 保持 `[B,16]`；对 `[B,C,H,W]`，会变为 `[B,C×H×W]`。随后 `sum(dim=1)` 对 **每张图的整个联合潜变量事件** 求 KL 之和，`mean()` 对 **Batch** 求平均。

$$\underbrace{[B,d]}_{\text{逐维 KL}} \xrightarrow{\mathrm{sum\ over\ latent}}[B] \xrightarrow{\mathrm{mean\ over\ batch}}\text{scalar}.$$

**为什么保留梯度？** 令 $\ell=\log\sigma^2$、$z=\mu+e^{\ell/2}\epsilon$，固定一次噪声后：

$$\frac{\partial z_j}{\partial\mu_j}=1, \qquad \frac{\partial z_j}{\partial\ell_j}=\frac12e^{\ell_j/2}\epsilon_j.$$

**测试建议：** 对解析手算值（$\mu=[1,0],\sigma^2=[4,1]$，KL 约 $1.30685$）、PyTorch `kl_divergence`、`gradcheck`、采样均值/方差、Monte Carlo KL 逐一交叉验证。

**代码维护：** 修改已经过时的“预留/当前不返回” Docstring，删除未使用的 `pending` 导入。正式使用时应检查 Shape、dtype/device 和非有限值，但不要无依据地截断 `logvar` 掩盖数值问题。

---

## **八、DDPM 与潜空间扩散**

<a id="q30"></a>

### **30. DDPM 与 VAE 的损失函数有什么区别？**

#### **i. 核心回答**

**Week 1 DDPM：** 噪声预测误差。**Week 2 VAE：** 图像重建项 + 潜变量分布的 KL 约束。这是两种 **不同的训练目标**。

#### **ii. DDPM：简化噪声预测 MSE**

令 $\alpha_t=1-\beta_t^{\mathrm{noise}}$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$，从图像 $x_0$ 和标准高斯噪声 $\epsilon$ 构造：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon, \quad\epsilon\sim\mathcal N(0,I).$$

训练模型 $\epsilon_\theta(x_t,t)$ 预测这次加入的噪声：

$$\boxed{\mathcal L_{\mathrm{DDPM,simple}}=\mathbb E_{x_0,t,\epsilon} \left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].}$$

**代码：**

```python
noise = torch.randn_like(x0)
xt = sqrt_abar_t * x0 + sqrt_1m_abar_t * noise
pred_noise = model(xt, t)
loss = torch.nn.functional.mse_loss(pred_noise, noise)
```

`F.mse_loss` 默认对全部元素取平均，而上式的平方范数为单样本各元素求和；对固定图片元素数，这主要是常数尺度的差别。该简化目标不同于 DDPM 的完整负 ELBO/KL 分解。

#### **iii. VAE：标准负 ELBO 与本周 BCE + βKL**

$$\boxed{ \mathcal L_{\mathrm{VAE}}(x)= -\mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] +D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)). }$$

第一项是期望重建负对数似然；第二项把近似后验约束到先验。教学实验使用 $\beta$ 权重：

$$\boxed{ L_{\beta}=\underbrace{-\sum_{i=1}^{784} [x_i\log\pi_i+(1-x_i)\log(1-\pi_i)]}_{\mathrm{BCE\ reconstruction}} +\underbrace{\frac\beta2\sum_{j=1}^{d} [\mu_j^2+e^{\ell_j}-1-\ell_j]}_{\mathrm{KL\ regularization}}, \quad\pi_i=\operatorname{sigmoid}(a_i),\ \ell_j=\log\sigma_j^2. }$$

上式的 BCE 是本周灰度 MNIST 的 **软目标 BCE**：$x_i\in[0,1]$ 不自动等于严格的连续灰度似然；重参数化采样让第一项的梯度传回 Encoder。实际训练先对 **每张图片的像素/潜维求和**，再对 Batch 平均。

| 项目 | DDPM | VAE |
|---|---|---|
| 训练输入 | 带噪 $x_t$ 与 $t$ | 干净图像 $x$ |
| 训练输出 | $\hat\epsilon$ | $\mu,\log\sigma^2$ 与重建 logits |
| 随机性 | 抽 $t$ 与加噪 $\epsilon$ | 从 $q_\phi(z\mid x)$ 采样 $z$ |
| 常用 Loss | 预测噪声 MSE | 重建 + KL（本实验为 BCE+KL） |
| 无条件生成 | 从噪声开始多步反向采样 | 先验采样 $z$ 后一次解码 |

**不要混淆两个“β”：** VAE 的 $\beta$ 是 KL 损失权重；DDPM 的 $\beta_t^{\mathrm{noise}}$ 是正向过程的噪声方差调度。它们不是一回事。

---

<a id="q31"></a>

### **31. 为什么在潜空间做扩散？如何读三篇原论文？**

#### **i. 核心回答**

- **VAE / Autoencoder** 学到图像到潜表示、潜表示到图像的变换。
- **DDPM** 学会通过反复去噪逼近复杂数据分布。
- **Latent Diffusion（LDM）** 借助预训练 Autoencoder，将去噪学习从高维像素空间迁移到较小的潜空间，再将生成的潜表示解码为图像。

```text
训练 LDM：image ──Encoder──> latent z₀ ──加噪──> zₜ
                                       └─ 去噪网络(预测噪声) ──> loss
生成 LDM：随机 latent z_T ──多步反向去噪──> z₀ ──Decoder──> image
```

**为什么先验证 Autoencoder？** 它在压缩过程中丢失的结构、颜色或细节，后续扩散难以无条件地补救；因此需要测量压缩率、重建能力、潜变量数值尺度及 Decoder 伪影。

**不要把本周 MLP TinyVAE 等同于现代 LDM 的图像 Autoencoder。** 后者一般保留空间特征图，并可能使用感知损失、对抗损失与更复杂的正则化；压缩越强不一定越好。

#### **ii. 本周 Notebook 的论文阅读地图**

| 论文 | 重点章节 | 读完必须回答 |
|---|---|---|
| [DDPM 原论文](https://arxiv.org/abs/2006.11239) — *Denoising Diffusion Probabilistic Models*，arXiv:2006.11239 | §2–3，算法 1/2 | 为什么能直接采样带噪图训练？噪声预测如何决定反向均值？ |
| [VAE 原论文](https://arxiv.org/abs/1312.6114) — *Auto-Encoding Variational Bayes*，arXiv:1312.6114 | §2–3，附录 B/C | 为什么需要近似后验？重参数化怎么让梯度回传？两种 KL 各指什么？ |
| [LDM 原论文](https://arxiv.org/abs/2112.10752) — *High-Resolution Image Synthesis with Latent Diffusion Models*，arXiv:2112.10752 | 图 1，§3.1–3.2，§4.1 | 为什么在潜空间去噪更省计算？压缩质量为什么决定后续上限？ |

**学习主线：** 当前 Week 2 已完成 TinyVAE 的数学核、网络、训练与可视化实验；后续先验证真实图像 Autoencoder，再讨论潜空间扩散与 DiT。DiT 是扩散模型的骨干网络选择，可与 LDM 的潜空间生成框架组合使用。

---

## **九、复习与自测**

<a id="q32"></a>

### **32. 为什么 VAE 的 Encoder 要输出 `mu` 和 `logvar`，而普通 AE 不需要？**

$\mu$ 与 $\log\sigma^2$ 确实能直接计算对标准正态的 KL，但更根本的原因是 **VAE 的 Encoder 输出一个概率分布的参数**，需要知道如何从这个分布中采样 $z$。

- **普通 Autoencoder（AE）：** $z=f_\phi(x)$，确定性的编码，再由 $\hat x=g_\theta(z)$ 重建。它没有必须使用的后验分布、标准正态先验或 KL 项。
- **Variational Autoencoder（VAE）：** Encoder 学习 $q_\phi(z\mid x)$。对角高斯假设下，使用两个 Head 输出 $\mu(x)$ 与 $\ell(x)=\log\sigma^2(x)$，以采样并计算 KL。

```text
AE:  x ──Encoder──> z ──Decoder──> x̂
VAE: x ──Encoder──> μ, ℓ ──ε~N(0,I) & 重参数化──> z ──Decoder──> x̂
                      └──────────── KL(q(z|x) || N(0,I))
```

**易错点：** AE 并非「不可能生成图片」，而是标准确定性 AE **没有天然保证** 从 $N(0,I)$ 随机采样就能得到合理图像。

---

<a id="q33"></a>

### **33. 为什么重参数化 $z=\mu+\sigma\epsilon$ 能让梯度传回 Encoder？**

使用返回值脱离计算图的采样 API（例如 PyTorch `Normal.sample()`），无法通过该返回值向分布参数回传梯度；重参数化把随机性独立放入 $\epsilon$，让给定一次噪声后的 $z$ 成为 $\mu,\ell$ 的可微函数：

$$\ell=\log\sigma^2,\quad z=\mu+e^{\ell/2}\epsilon,\qquad \frac{\partial z}{\partial\mu}=1,\quad \frac{\partial z}{\partial\ell}=\frac12e^{\ell/2}\epsilon.$$

于是梯度沿着 `reconstruction loss → Decoder → z → (mu, logvar) → Encoder` 回传。

**小例子：** $\mu=1,\sigma^2=4,\epsilon=0.5\Rightarrow z=1+2\times0.5=2$。

**易错点：** 「直接采样不可微」只是对常见朴素实现的简化说法；重参数化也不是所有随机梯度估计方法中的唯一选择。

---

<a id="q34"></a>

### **34. 如果 $\beta=0$，重建很准但 KL 很高，为什么先验生成可能变差？**

$\beta=0$ 时只有重建目标，Encoder 没有动力把每张图的后验 $q(z\mid x)$ 约束到 $p(z)=N(0,I)$ 附近。Decoder 主要见到的是 Encoder 实际输出的 $z$，但生成时却要求它处理从先验分布随机取得的 $z$；两边若失配，生成效果就可能不好。

```text
训练：x → Encoder → q(z|x) 取样 → Decoder → 重建
生成：         N(0,I) 取样 → Decoder → 新图片
           （采样来源可能与训练时明显不同）
```

- 类别之间可以共享潜空间；训练目标没有要求每个数字类别占据互不重叠的区域。
- **重要严谨性：单图平均 KL 很高，不严格推出聚合后验一定与先验相距很远。** 因为平均 KL 也可能包含潜变量携带输入信息所需的代价。可选进阶关系：

$$\mathbb E_{p_{\mathrm{data}}(x)}D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))=I_q(X;Z)+D_{\mathrm{KL}}(q_\phi(z)\Vert p(z)),$$

其中 $q_\phi(z)=\mathbb E_{p_{\mathrm{data}}(x)}q_\phi(z\mid x)$ 为聚合后验。
- **本次实验观察：** $\beta=0$ 不仅 KL/图约 $345.28$，聚合后验方差也显著偏离 1，且观察到先验生成较差。因此这个模型确实呈现明显的失配迹象；但不能凭一次小样本图片概括所有情况。

---

<a id="q35"></a>

### **35. 如果 $\beta=4$，重建模糊，可能是什么原因？能直接判断 Posterior Collapse 吗？**

重建误差增大是观察结果。过强的 KL 约束可能减少编码保留的信息，较小的潜维或网络容量也可能损害重建；要通过对照确认原因。

| 可能原因 | 为什么会模糊 | 怎样针对性验证 |
|---|---|---|
| ① KL 约束过强 | 模型减少对 $x$ 的信息编码以降低 KL，细节无法传给 Decoder | 固定其他条件，比较 $\beta=0.5/1/4$ 的 recon、KL 与生成图 |
| ② 潜变量维度不足 | $d$ 太小，编码容量有限 | 比较 $d=2$ 与 $d=16$；保持 β、步数等一致 |
| ③ 训练不充分 | 优化器还没学会足够好的重建映射 | 延长训练、看验证集 recon 是否持续下降 |
| ④ Encoder/Decoder 容量不足 | 网络表达能力有限 | 更换更强网络，并控制训练和参数量对照 |
| ⑤ 重建目标的特点 | Bernoulli/BCE 下的像素概率图可能呈现平均化或灰度过渡 | 与不同似然/重建目标对照，区分 logits、概率图和采样图 |
| ⑥ Decoder 未充分使用 $z$ | 输出对潜变量变化不敏感 | 固定模型扰动 $z$、交换不同输入的 $z$，观察输出和 recon 变化 |

**Posterior Collapse（后验坍塌）** 通常指 $q_\phi(z\mid x)$ 逼近先验且潜变量不再有效携带输入信息，Decoder 在重建时可能忽略 $z$。因此需结合 **低 KL、编码信息量、扰动 $z$ 后的 Decoder 敏感度** 等证据判断；单看低 KL 或模糊图片都不足以定论。

**结论：** 对 $\beta=4$ 模糊的首要怀疑是约束更强、信息保留减少，但其他原因仍需消融实验排查。二维插值出现混合数字也 **不是** 坍塌的直接证据。

---

<a id="q36"></a>

### **36. VAE 与 DDPM 分别优化什么？怎样组合成 Latent Diffusion？**

对应本周实现：

$$L_{\beta\text{-VAE}}=L_{\mathrm{recon}}+\beta L_{\mathrm{KL}},$$

$$L_{\mathrm{DDPM,simple}}=\mathbb E_{x_0,t,\epsilon}\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].$$

- **VAE：** 学习压缩表示及重建；**DDPM：** 学习预测噪声并进行多步去噪。
- **LDM 训练：** 通常先训练 Autoencoder，再将图片编码成潜变量 $z_0$；对 $z_0$ 加噪，在潜空间训练去噪网络。
- **LDM 生成：** 从潜空间高斯噪声逐步去噪得到 $z_0$，最后用 Decoder 解码成图片。

**精确说法：** DDPM 不是直接「塞进 VAE 的 `forward()`」就能使用；它是另一阶段在 **潜变量分布上** 训练的生成模型，并且通常使用具有空间结构的潜变量特征图（而非本周 MNIST 的 `[B,16]`）。

---

<a id="q37"></a>

### **37. 如何从均值和对数方差采样，并解码成二维图像？**

#### **i. 核心回答：采样的是 `z`，图片由 Decoder 产生**

**不是把 μ 和 logvar 两组向量直接变成二维图像**。它们为输入图片定义一个对角高斯分布；从这个分布采样得到一个潜变量向量 `z`，再用训练好的 Decoder 映射成像素。

$$q_\phi(z\mid x)=\mathcal N\left(\mu_\phi(x),\operatorname{diag}(e^{\ell_\phi(x)})\right),\qquad \ell=\log\sigma^2.$$

从这个分布中重参数化采样：

$$\boxed{\epsilon\sim\mathcal N(0,I),\qquad z=\mu+\exp(\ell/2)\odot\epsilon.}$$

之后由 Decoder 输出图像的 logits，再通过 `sigmoid` 显示灰度概率图：

$$\boxed{\hat x=\operatorname{sigmoid}(\operatorname{Decoder}_\theta(z)).}$$

#### **ii. 手算：一个只有两维的潜空间**

假设 Encoder 输出：

$$\mu=[1,2],\qquad \ell=[0,\ln 4].$$

则标准差：

$$\sigma=e^{\ell/2}=[1,2].$$

从标准正态随机抽到一次噪声：

$$\epsilon=[0.5,-1].$$

逐维计算：

$$z=[1,2]+[1,2]\odot[0.5,-1]=\boxed{[1.5,0]}.$$

**最终 `[1.5,0]` 只是二维潜向量**；要得到 28×28 MNIST，必须继续调用 `Decoder(z)`。Decoder 是训练出的非线性映射，负责从少数潜变量生成 784 个像素 logits。

#### **iii. PyTorch 的实际 Shape（`latent_dim=16`）**

```python
# images: [B,1,28,28]
mu, logvar = model.encode(images)                 # 两者都是 [B,16]
eps = torch.randn_like(mu)                       # [B,16]
std = torch.exp(0.5 * logvar)                    # [B,16]
z = mu + std * eps                              # [B,16]
logits = model.decode(z)                         # [B,1,28,28]
reconstruction = logits.sigmoid()                # [B,1,28,28]
```

本周 TinyVAE Decoder 的关键路径：

```text
z [B,16] → Linear(16,128) → ReLU → Linear(128,256) → ReLU
         → Linear(256,784) → reshape → logits [B,1,28,28]
```

有三种必须区分的路线：

| 用途 | `z` 的取得方式 | 是否需要原图 |
|---|---|---|
| 均值重建 | `z=mu` | 是 |
| 随机重建 | `z=mu+exp(logvar/2)*eps` | 是 |
| 无条件先验生成 | `z=torch.randn(B, latent_dim)` | **否** |

**容易混淆：** (1) `latent_dim=2` 是二维潜空间；MNIST 图片 `28×28` 是二维空间结构；(2) 生成新图时直接从先验采样，不需要 μ/logvar；(3) 训练 BCEWithLogits 用 Decoder 的 **logits**，仅可视化时 `sigmoid`；(4) VAE 的概率重建依赖 Encoder 预测的后验，DDPM 的生成采样则是另一套反向去噪过程。

---

<a id="q38"></a>

### **38. 如何通过手写 TinyVAE 验证对核心流程的理解？**

#### **i. 核心回答**

对于本次课程目标，**亲手重写一次 VAE 的核心代码，比单纯重看笔记更适合检验掌握程度**。之前实现/运行过，不等于过几天仍能独立写出来。这里重点是恢复「结构—Shape—采样—Loss」的主动提取能力，而不是重新写工程脚手架。

#### **ii. 最小手写任务（不提供现成完整答案）**

只用 PyTorch 基础模块，不看旧代码，手写以下五部分：

1. **Encoder**：`[B,1,28,28] → Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU`。
2. **双 Head**：用 `[B,128]` 分别预测 `mu` 与 `logvar`，都是 `[B,latent_dim]`。
3. **重参数化**：自己写 `z = mu + exp(0.5*logvar)*eps`，保持梯度。
4. **Decoder**：`[B,latent_dim] → 128 → 256 → 784 → [B,1,28,28]`；输出 **logits** 而非 sigmoid 后结果。
5. **Loss**：`recon = BCEWithLogits(logits, x)`，`KL = 0.5*(mu²+exp(logvar)-1-logvar)`；单图像素/潜维先求和，batch 再平均。

一个接口骨架（可以自己填写 `...`）：

```python
class MyVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # TODO: encoder, mean_head, logvar_head, decoder

    def encode(self, x):
        # TODO: return mu, logvar
        ...

    def reparameterize(self, mu, logvar, eps):
        # TODO: return z
        ...

    def decode(self, z):
        # TODO: return logits [B,1,28,28]
        ...

    def forward(self, x, eps):
        # TODO: return logits, mu, logvar
        ...


def vae_loss(logits, x, mu, logvar, beta=1.0):
    # TODO: return reconstruction loss + beta * KL
    ...
```

**最小验证：** 先用随机 `x∈[0,1]` 做一次前向，检查 `mu/logvar/z=[B,d]`、`logits=[B,1,28,28]`、`loss` 为标量；执行 `loss.backward()`，确认 Encoder 与 Decoder 均有梯度；再用现有 MNIST 训练器跑少量步数并查看结果。

**与下一阶段的关系：** 如果现在希望直接开始图像空间型 VAE / Latent Diffusion，完全可以先推进。手写上述 TinyVAE 是 **复习练习，不是必须先完成的前置门槛**；后续建议亲自实现卷积型 Encoder/Decoder、采样与关键 Loss，训练与日志继续复用工程工具。

---

## **十、从 TinyVAE 到图像 Autoencoder**

本周 TinyVAE 的 `z:[B,16]` 是一个 **向量**；下一阶段更常见的目标是保留空间结构的潜变量张量，例如（仅示意，并非特定模型的标准配置）：

```text
RGB image            [B,3,256,256]
  ↓ Conv Encoder（下采样）
mu/logvar            [B,4,32,32]
  ↓ 重参数化
spatial latent z     [B,4,32,32]
  ↓ Conv Decoder（上采样）
reconstructed image  [B,3,256,256]
```

**数学不必重学：** 仍然是 `z = μ + exp(logvar/2)·ε`；对于 `[B,C,H,W]`，KL 要对 `C,H,W` 求和、对 `B` 求均值。变化主要是网络结构、重建目标、潜变量的空间性与压缩率。

**下一阶段建议优先回答：** 如何确定压缩倍率？为什么用卷积保留空间结构？像素 BCE/MSE 与感知损失有何取舍？为什么必须先检查重建质量，才能训练潜空间扩散？

---

## **十一、知识关联与核心总结**

### **1. 间隔复习索引**

- **需要补的概念：** 普通 AE 的确定性编码 vs VAE 的概率编码（Q32）。
- **需要补的因果分析：** 模糊的六种原因、Posterior Collapse 的证据及验证方法（Q35）。
- **需要提高的严谨性：** 平均 KL、聚合后验失配和先验采样质量不是简单的等价关系（Q34）。
- **已经掌握的主线：** 重参数化的梯度传播（Q33）；VAE/DDPM Loss 与 LDM 的分工（Q36）。
- **下次自测建议：** 用 1 分钟解释「AE vs VAE」；用 2 分钟画出 VAE 的重建、KL 两条梯度路径；列举 3 种模糊原因并给出相应对照实验。

### **2. 五条最重要的流程（Q37 为采样生成的展开例子）**

```text
A. 训练重建：x → Encoder → (μ,ℓ) → z=μ+exp(ℓ/2)ε → Decoder → logits → BCE
B. 先验生成：z~N(0,I) → Decoder → sigmoid(logits)
C. 均值重建：x → Encoder → μ → Decoder → sigmoid(logits)
D. 潜空间插值：μ_A、μ_B → z(α)=(1-α)μ_A+αμ_B → Decoder
E. Latent Diffusion：x → Autoencoder Encoder → latent 加噪/去噪 → Decoder → 图片
```

### **3. 最终公式速查**

| 概念 | 公式 | 提醒 |
|---|---|---|
| 对角高斯近似后验 | $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(e^{\ell_\phi(x)}))$ | $\ell$ 是 **log 方差**，不是标准差 |
| 重参数化 | $z=\mu+e^{\ell/2}\odot\epsilon$，$\epsilon\sim\mathcal N(0,I)$ | 梯度可经 $z$ 回传 |
| 一维 KL | $\tfrac12(\mu^2+e^\ell-1-\ell)$ | KL 相对于标准正态先验 |
| 多维 KL | $\tfrac12\sum_j(\mu_j^2+e^{\ell_j}-1-\ell_j)$ | 先逐潜维求和，后批量平均 |
| MNIST 重建 BCE | $-\sum_i[x_i\log\pi_i+(1-x_i)\log(1-\pi_i)]$ | $\pi_i=\mathrm{sigmoid}(a_i)$；灰度软目标 |
| VAE 总损失 | $L_\beta=L_{\mathrm{recon}}+\beta L_{\mathrm{KL}}$ | 在二值 Bernoulli 观测、按图求和时，$\beta=1$ 对应负 ELBO；灰度软目标 BCE 是这里的替代损失 |
| DDPM 简化损失 | $\mathbb E\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2$ | 与完整变分目标有区别 |

### **4. 本周实验的事实与推测分开记录**

| 观察到的事实 | 合理的解释 | 仍不能直接推断 |
|---|---|---|
| $\beta$ 从 0 到 4，重建变差、KL 下降 | 更强先验正则化与信息保留权衡 | 大 $\beta$ 必然生成更好 |
| $\beta=0$ 的重建较好，但先验生成常模糊 | 训练后验与先验可能明显失配 | 所有潜变量都已被有效使用 |
| $\beta=0.5$ 重建较好、先验网格可辨认度提升 | 温和 KL 可能折中 | 单次 16 样本图能确定最优 $\beta$ |
| 二维散点类别重叠、插值数字渐变 | 表示瓶颈和 Decoder 的连续映射 | 模型学会精准控制数字语义 |
| 个别重建模糊 | 可能涉及容量、训练、目标、KL、维度或潜变量忽略 | 直接宣布 Posterior Collapse |

### **5. Week 2 结束时的实际完成项**

- 理解先验/近似后验、条件高斯、期望取权重、KL 定义与重参数化。
- 完成 TinyVAE 的 Encoder/双 Head/Decoder 与两项概率函数的实现；验证 Tensor Shape、梯度、KL 解析式与采样统计。
- 训练并对比 $d=16$ 下 $\beta=0,0.5,1,4$，以及 $d=2,\beta=1$；记录训练曲线、重建与先验生成。
- 实现按维 KL、聚合后验方差、二维潜空间网格、9→4 潜变量插值等分析；明确观察与因果诊断的区别。
- 通过固定潜变量的 Checkpoint 推理重载验证，以及 8 步训练 / 中断 3 步恢复的一致性测试（在指定 CPU 设置下）。

### **6. 自测：不看公式，尝试回答**

1. 为什么 `exp(0.5*logvar)` 才是标准差？若直接用 `exp(logvar)` 会怎样？
2. 为什么 `E_q[log p(z)]` 使用 $q$ 作为权重，而不是 $p$？
3. 为什么 KL 对潜变量维度求和、对 Batch 求平均？
4. 为什么 `BCEWithLogits` 用 logits 而不能把 sigmoid 后的概率重复传进去？
5. 先验生成不好而均值重建很好，该优先检查什么？
6. 为什么低 KL、不清晰插值或模糊图片不能单独证明后验坍塌？
7. 如果中断训练只保存模型权重，Adam 和数据读取会有什么问题？
8. 为什么 Latent Diffusion 先检查 Autoencoder 再训练去噪器？
9. 如果 `mu=[1,2]`、`logvar=[0,ln4]`、`eps=[0.5,-1]`，潜变量是什么？从这个潜变量到二维图片还差哪一步？
10. 如果把 `z:[B,16]` 改成 `z:[B,C,H,W]`，哪些 VAE 公式完全不需要变？

> **本周核心一句话：VAE 在“让潜变量保留图像信息”和“让潜变量分布易于先验采样”之间权衡；DDPM 则主要学习多步去噪。理解两者并严谨评测，是向潜空间扩散过渡的基础。**
