## **一、MiniGen · DDPM 第一阶段完整学习 Q&A**

> **范围**：DDPM 基本概率论 → 前向加噪 → 贝叶斯后验 → 变分目标的基本直觉 → Scheduler → 时间条件 U-Net → MNIST 数据 → 训练/验证 → EMA/采样 → 实验复盘 → 个人实现验收。
>
> **整理依据**：本阶段实际对话中提出的问题、此前专题 Q&A、第一周教学 Notebook 和个人 `DDPMScheduler` 代码。对重复追问合并、对不严谨说法校正，保留必要的数值例子与关键推导。
>
> **完成状态**：教学版 DDPM 已完成 MNIST 训练与完整反向采样；个人 Scheduler 的核心公式已编写，**后续 S1 验收记录已独立验证前向加噪、训练目标与反向采样核心**。
>
> **复习建议**：先读「一页总览」，再按需要查找具体 Q；不必一口气重读所有推导。这里是一次完整的阶段收官总结，不是下阶段任务清单。

---

### **1. 目录**

1. [一页总览：什么是 DDPM](#overview)
2. [概率论语言：随机变量、期望、条件、联合与边缘](#prob)
3. [前向扩散：alpha、beta、累计噪声、SNR](#forward)
4. [贝叶斯反向后验：为什么可以推导出均值和方差](#posterior)
5. [q 与 p、最大似然、KL、ELBO/VLB（当前掌握边界）](#vlb)
6. [Scheduler：接口、公式和个人实现](#scheduler)
7. [U-Net：时间编码、GroupNorm、残差与多尺度结构](#unet)
8. [MNIST 与 DataLoader：数据从哪里来](#mnist)
9. [训练、验证、随机时间与固定过拟合](#training)
10. [生成、Gaussian Sampling、EMA 和 Generator](#sampling)
11. [实验结果、Checkpoint 与可复现性](#experiment)
12. [容易混淆的概念](#review)
13. [八道综合自测](#self-check)
14. [六个核心公式与阶段验收](#core-formulas)

---

<a id="overview"></a>

## **二、一页总览：DDPM 从头到尾在干什么？**

<a id="q1"></a>

### **1. 这一节学完，整个 DDPM 究竟是什么？**

**核心回答**：DDPM 先 **预设** 如何将真实图像逐渐加噪；再训练一个网络预测带噪图中的 **累计等效噪声**；生成时从新的高斯噪声出发，借助网络预测和 Scheduler 反复采样，产生新图像。

**训练（每次仅随机选一个噪声时间步，不跑完整反向链）：**

```text
MNIST x0  ──┐
随机 t      ├── Scheduler.add_noise(x0, eps, t) → xt
标准高斯 eps┘                                        │
                                                   ▼
                                                U-Net(xt,t)
                                                   │
                                                   ▼
                                               eps_pred
                                                   │
                                                   ▼
                                            MSE(eps_pred,eps)
                                                   │
                                                   ▼
                                         backward + optimizer.step
```

**生成（运行完整或指定的反向时间网格，不计算训练 Loss）：**

```text
xT ~ N(0,I)
  ↓  U-Net(xt,t) 预测 eps_theta
  ↓  Scheduler 用 eps_theta 构造 mu_theta、variance
  ↓  x_(t-1) = mu_theta + std_t * z
  ↓  更新当前状态、时间减一，重复
x0：新生成的图像（不要求等于任何训练样本）
```

#### **i. 同一个符号在三个场景中有什么不同？**

| 变量 | 训练 | 生成 | 注意 |
|---|---|---|---|
| `x0` | 真实 MNIST 图像 | 未知；最终输出的新图像 | 生成时没有隐藏的真实答案 |
| `xt` / `sample` | 由真实 `x0` 人为加噪得到 | 反向链当前已采样出来的图像 | 都是具体的 Tensor |
| `eps` | 本次加噪的标准高斯，作为标签 | **没有** 这份标签 | 不等于本步反向随机项 |
| `eps_pred` / `epsilon_theta` | U-Net 对 `eps` 的预测 | U-Net 当前的噪声预测 | 同一套网络参数共享全部时间 |
| `z` / `step.noise` | 通常不参与简单 MSE | 为反向分布新抽的随机数 | 作用是 **采样** 而非监督 |
| `t` | 每个样本通常随机选 | 从大到小逐步迭代 | 不是 Optimizer Step |
| `theta` | 被 Optimizer 更新 | 固定不更新 | Scheduler 的 alpha/beta 通常固定 |

#### **ii. 最短可读的算法**

```python
# 一次训练迭代（示意；model/scheduler/optimizer 已正确放到设备上）
x0, _ = next(iter(train_loader))
t = torch.randint(scheduler.num_steps, (len(x0),), device=x0.device)
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)
loss = F.mse_loss(eps_pred, eps)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 一次完整 DDPM 生成（示意）
x = torch.randn(8, 1, 28, 28, device=DEVICE)
model.eval()
with torch.no_grad():
    for k in reversed(range(scheduler.num_steps)):
        t = torch.full((len(x),), k, dtype=torch.long, device=x.device)
        eps_pred = model(x, t)
        z = torch.randn_like(x)
        x = scheduler.step(eps_pred, k, x, z)
# 此时 x 为最终生成结果
```

**本阶段唯一需要记住的算法分工**：`Scheduler` 管固定数学规则；`U-Net` 学习预测；`Optimizer` 更新 U-Net；`Sampler` 循环调用已训练模型与 Scheduler。

---

<a id="prob"></a>

## **三、概率论语言：随机变量、期望、条件、联合与边缘**

<a id="q2"></a>

### **2. $x_0\sim p_{\mathrm{data}}$、$X\sim\mathcal N(0,I)$ 这种波浪号到底表示什么？**

**核心回答**：$X\sim q$ 表示随机变量 `X` **服从分布** `q`；也可以理解为“从 `q` 规定的规律中采样得到 X”。

- $x_0\sim p_{\mathrm{data}}$：从真实图片总体的分布中抽一张图；项目里用 MNIST 样本近似这个真实分布。`p_data` 一般没有已知解析式。
- $\epsilon\sim\mathcal N(0,I)$：抽一份零均值、单位协方差的高斯噪声；每个像素位置都有自己的噪声值。
- `x` 是 **某次已经抽出的数值/Tensor**；`X` 可用来强调尚未抽样的随机变量。

不要混淆“每个噪声像素服从标准高斯”和“每张图片全部像素的样本均值严格等于 0、方差严格等于 1”；后者不是必然的。

<a id="q3"></a>

### **3. $\mathbb E_{x\sim q}[f(x)]=\int q(x)f(x)\,dx$ 里 `f(x)` 又是什么？**

**核心回答**：`q` 指定 **怎么抽样**，`f` 指定 **抽到一个样本后计算什么**；期望就是对计算结果按 `q` 加权平均。

离散数值例子：`P(X=1)=0.25`、`P(X=3)=0.75`。

- 若 `f(x)=x`，则 `E[f(X)]=0.25×1+0.75×3=2.5`。
- 若 `f(x)=x²`，则 `E[f(X)]=0.25×1²+0.75×3²=7`。

连续情形用积分：若 `X ~ Uniform(0,1)`、`f(x)=x²`，

$$\mathbb E[f(X)]=\int_0^1 1\cdot x^2\,dx=\frac13.$$

在 DDPM 的变分推导中，可取：

$$f(x_{1:T})=\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)},$$

其中 **整个中间状态序列** `x_{1:T}` 充当期望的随机变量，而 `x0` 暂时固定。

<a id="q4"></a>

### **4. $\mathbb E_{x_0\sim p_{\mathrm{data}}}[\log p_\theta(x_0)]$ 右下角的条件怎么读？**

**核心回答**：意思是“在真实数据分布下抽取图片，对模型赋予它的对数密度求平均”。前面的 `max_theta` 则表示调整网络参数使平均值尽可能大：

$$\max_\theta\ \mathbb E_{x_0\sim p_{\rm data}}[\log p_\theta(x_0)].$$

若训练集只有 3 张且将其等权处理，经验目标近似为：

$$\max_\theta\frac13\sum_{i=1}^{3}\log p_\theta(x_0^{(i)}).$$

这叫 **最大似然**，但 DDPM 的 `p_theta(x0)` 一般无法直接高效精确求出；所以后面才引入变分界。

<a id="q5"></a>

### **5. $P(A\mid B)$ 与贝叶斯公式的直觉是什么？**

**核心回答**：$P(A\mid B)$ 是“已经知道 B，A 发生的概率”；条件顺序颠倒后通常不是同一个数。贝叶斯把两个方向联系起来：

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.$$

**手算例子**：盒 A 被选中概率 0.3、盒 B 概率 0.7；盒 A 红球占 80%，盒 B 红球占 20%。抽中红球后来自盒 A 的概率：

$$P(A\mid\text{红})=\frac{0.8\times0.3}{0.8\times0.3+0.2\times0.7}=\frac{0.24}{0.38}\approx0.632.$$

贝叶斯本质上是：**先验判断 + 新观察 → 更新后的判断**。连续变量中相应式子描述的是密度。

<a id="q6"></a>

### **6. 联合分布、边缘分布、生成“路径”分别是什么？**

**核心回答**：

- `p(x0,x1,x2)` 是三种状态的 **联合分布**，描述整条 `x2 → x1 → x0` 随机轨迹。
- 一条 **路径/样本轨迹** 是某次采样的具体状态序列 `(x2,x1,x0)`；不是时间轴上数值的积分。
- `p(x0)` 是只关心最终图像、不关心中间状态的 **边缘分布**。

两步反向链：

$$p_\theta(x_0,x_1,x_2)=p(x_2)p_\theta(x_1\mid x_2)p_\theta(x_0\mid x_1).$$

想求最终图片的密度，需要汇总一切中间状态：

$$p_\theta(x_0)=\iint p_\theta(x_0,x_1,x_2)\,dx_1\,dx_2.$$

推广为：

$$\boxed{p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}}.$$

$dx_{1:T}=dx_1\cdots dx_T$ 表示对 **时间步对应的图像随机向量** 积分（每张图可能有 784 维），**不是** 对时间 `t` 积分。

<a id="q7"></a>

### **7. 什么叫“采样”？已抽出来的 `sample` 为什么又是确定的？**

**核心回答**：采样是按照分布的规则随机取出一份具体数值。比如 `z = torch.randn(3)`：抽之前结果未知；抽完后 `z` 就是固定张量，除非下次重新采样。

正态采样通式：

$$z\sim\mathcal N(0,1),\quad X=\mu+\sigma z\implies X\sim\mathcal N(\mu,\sigma^2).$$

若 `mu=0.5, variance=0.09`，则 `std=0.3`：`z=-1` 抽到 0.2；`z=1` 抽到 0.8。同一分布允许不同的随机结果。代码中的 `sample` 指“当前已经拿到的图像张量”，不是抽象概率分布。

<a id="q8"></a>

### **8. 为什么“正比于高斯核”最终可以写成“等于高斯分布”？**

**核心回答**：只有完成 **归一化** 后才能写等号；不能直接把 `∝` 改成 `=`。

若已推得：

$$q(y)\propto\exp\!\left[-\frac{(y-\mu)^2}{2\sigma^2}\right],$$

合法的一维概率密度满足 $\int q(y)\,dy=1$，而高斯积分为 `sqrt(2πσ²)`，所以比例常数必须是：

$$C=\frac1{\sqrt{2\pi\sigma^2}}.$$

因而：

$$q(y)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].$$

贝叶斯公式的分母正是归一化：

$$q(x_t\mid x_0)=\int q(x_t\mid y)q(y\mid x_0)\,dy.$$

---

<a id="forward"></a>

## **四、前向扩散：单步、累计、噪声日程与信噪比**

<a id="q9"></a>

### **9. 为什么设计 $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I)$？**

**核心回答**：这是一种由研究者 **主动规定** 的、容易采样和推导的高斯马尔可夫加噪链。单步写成：

$$\boxed{x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t},\quad\alpha_t=1-\beta_t,\quad\epsilon_t\sim\mathcal N(0,I).$$

$\sqrt{\alpha_t}$ 是上一状态的幅度系数，$\sqrt{\beta_t}$ 是新增噪声的标准差。计算方差时，分别用系数的平方乘以对应随机变量的方差。如果上一状态与新噪声独立且都具有单位方差，$\alpha_t+\beta_t=1$ 可以保持单位方差；**但真实图片本身不被要求是均值 0、方差 1 的高斯**。

<a id="q10"></a>

### **10. `alpha_t` 和 `beta_t` 谁控制信号、谁控制噪声？**

- `alpha_t` 大：**本步** 对上一时刻 `x(t-1)` 的保留更强。
- `beta_t` 大：**本步** 新增高斯噪声的方差更大。
- **上一时刻本身已经有噪声**，因此它不是原始清晰图像的比例。
- 原始 `x0` 的累计保留强度必须看 `alpha_bar_t`。

例：$\alpha_1=0.9$、$\alpha_2=0.8$，则第二步中上一状态的系数平方为 0.8，但 `x0` 的累计系数平方为 `0.9×0.8=0.72`，累计噪声方差为 `1-0.72=0.28`。

<a id="q11"></a>

### **11. 多步加噪为什么能合并成一个公式？累计噪声怎么算？**

先展开两步：

$$\begin{aligned} x_1&=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1,\\ x_2&=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2\\ &=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2. \end{aligned}$$

两份 **独立** 标准高斯的线性组合仍是高斯；其噪声项协方差为：

$$\alpha_2\beta_1+\beta_2=\alpha_2(1-\alpha_1)+(1-\alpha_2)=1-\alpha_1\alpha_2.$$

所以可以把噪声项 **按分布** 写成一份标准高斯乘以标准差：

$$x_2\overset d=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{1-\alpha_1\alpha_2}\epsilon.$$

一般式：

$$\boxed{q(x_t\mid x_0)=\mathcal N\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right)},\quad \boxed{x_t\overset d=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon},$$

$$\bar\alpha_t=\prod_{s=1}^t\alpha_s,\qquad\epsilon\sim\mathcal N(0,I).$$

所谓 **累计等效噪声** 不是单步 `eps_t`，而是各步新增噪声经缩放、相加、归一化后的等效变量。对两步情形：

$$\bar\epsilon_2=\frac{\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2}{\sqrt{1-\alpha_1\alpha_2}}\sim\mathcal N(0,I).$$

**不要误读**：`x_t = ...+eps` 的直接加噪与实际逐步采样只有 **条件分布一致**，不保证同一随机轨迹上逐点相等。

#### **i. 三步日程数字例子**

| 数学步数 | `beta` | $\alpha_t=1-\beta_t$ | `alpha_bar` | 原图幅度 | 累计噪声幅度 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.1 | 0.9 | 0.900 | 0.9487 | 0.3162 |
| 2 | 0.2 | 0.8 | 0.720 | 0.8485 | 0.5292 |
| 3 | 0.3 | 0.7 | 0.504 | 0.7099 | 0.7043 |

因此训练时可以 **随机抽一个 t，直接构造 xt**；不必逐个执行前面所有加噪步。

<a id="q12"></a>

### **12. 为什么选择标准高斯噪声？最后是不是一定变成 $\mathcal N(0,I)$？**

**核心回答**：标准高斯易采样、参数简单，且让反向生成可以从一个已知分布开始。若 $\bar\alpha_T\to0$：

$$x_T=\sqrt{\bar\alpha_T}x_0+\sqrt{1-\bar\alpha_T}\epsilon\approx\epsilon\sim\mathcal N(0,I).$$

**无穷步本身不保证** 信号彻底消失；要检查 $\prod_t\alpha_t$ 的极限。固定 `0<beta<1` 或某些合适噪声日程可以使其趋零。

<a id="q13"></a>

### **13. `alpha`、`beta` 为什么随时间变化？经典取多少？**

它们共同构成 **noise schedule（噪声日程）**，控制不同时间段的噪声强度与学习难度。经典教学设置：1000 步线性 `beta` 从 `0.0001` 到 `0.02`；此时 `alpha_bar` 终点约 `4.04×10^-5`，原始信号幅度约 `0.00635`。它是 **基线而不是任何模型都最优的规定**。时间少时不能直接照搬同一组起止值而忽略终点信号。

<a id="q14"></a>

### **14. 信噪比 SNR 是 $\alpha_t/\beta_t$ 吗？高低各代表什么？**

**核心回答**：对于原图 **参考方差按 1 计** 的累计信号噪声比：

$$\boxed{\mathrm{SNR}_t=\frac{\bar\alpha_t}{1-\bar\alpha_t}}.$$

它不是单步 $\alpha_t/\beta_t$。SNR 越大，原图成分占优；越小，噪声占优。

| `alpha_bar` | SNR | 直觉 |
|---:|---:|---|
| 0.90 | 9 | 仍较清晰 |
| 0.50 | 1 | 信号与噪声平方幅度相当 |
| 0.10 | 0.111 | 噪声主导 |
| 0.01 | 0.0101 | 几乎纯噪声 |

前向让 SNR 大致下降，反向从低 SNR 逐步恢复结构。更一般地，若原图各维参考方差为 `s²`，信号方差项会额外乘 `s²`。SNR 不是“噪声像素占比”也不是模型准确率。

<a id="q15"></a>

### **15. 给定原图，`xt` 的均值/方差是什么？原图和随机噪声的数据分布有何区别？**

对于 **固定某张** `x0` 并重复抽噪声：

$$\mathbb E[x_t\mid x_0]=\sqrt{\bar\alpha_t}x_0,\qquad\mathrm{Cov}(x_t\mid x_0)=(1-\bar\alpha_t)I.$$

例：单像素 `x0=0.8`、`alpha_bar=0.36`：

$$x_t=0.6(0.8)+0.8\epsilon=0.48+0.8\epsilon,$$

所以该像素 **条件均值 0.48、条件方差 0.64**。若 `eps=1`，抽到 `xt=1.28`，完全可能超出干净 MNIST 图像的 `[-1,1]`。`x0` 来自复杂图像分布，**不是** 标准高斯。也不要把“某张图片空间像素均值”与这里的“固定原图后重复加噪的条件均值”混为一谈。

---
<a id="posterior"></a>

## **五、贝叶斯反向后验：公式为什么能推出来？**

<a id="q16"></a>

### **16. 已有前向 `q(xt|x(t-1))`，为什么还要求 $q(x_{t-1}\mid x_t,x_0)$？**

**核心回答**：为了在 **训练时已知真实原图** 的情况下，算出“当前带噪图 `xt` 的上一时刻合理状态”服从怎样的条件分布，为反向模型提供数学参照。

令 $y=x_{t-1}$，已知：

$$q(x_t\mid y)=\mathcal N(\sqrt{\alpha_t}y,\beta_t I),$$

$$q(y\mid x_0)=\mathcal N(\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I).$$

求的是 `q(y|xt,x0)`。其条件含义是：**x0、xt 都知道；y 尚未知；对 y 求概率分布**。这是“给定起点和终点推断中间状态”，不是“模型直接推断哪张图片一定正确”。

<a id="q17"></a>

### **17. 为什么贝叶斯公式能写成 $q(y\mid x_t,x_0)\propto q(x_t\mid y)q(y\mid x_0)$？**

严格式（条件密度链式法则）：

$$q(y\mid x_t,x_0)=\frac{q(x_t\mid y,x_0)q(y\mid x_0)}{q(x_t\mid x_0)}.$$

前向过程是马尔可夫链：给定上一状态 `y` 后，当前 `xt` 不再依赖更早的 `x0`，故：

$$q(x_t\mid y,x_0)=q(x_t\mid y).$$

分母 $q(x_t\mid x_0)$ 不依赖待求变量 `y`，因此若只分析 `y` 的密度形状，可写：

$$\boxed{q(y\mid x_t,x_0)\propto q(x_t\mid y)q(y\mid x_0)}.$$

**语言版**：先验 $q(y\mid x_0)$ 表示“从原图出发，这个中间态可能吗”；似然 $q(x_t\mid y)$ 表示“从这个中间态出发，观察到当前图合理吗”。相乘结合两种信息。

**关于 `∝` 和 `=`**：省掉的分母不是 1，必须最后用积分为 1 的要求恢复归一化常数（见 Q8）。

<a id="q18"></a>

### **18. 为什么两个高斯相乘又会得到高斯？展开平方到底在干什么？**

高斯密度具有 `exp(-二次型)` 的结构；两个高斯密度相乘后，指数相加，依旧是关于 `y` 的二次型，可通过 **配方** 改写成某个高斯核。

去掉与 `y` 无关的常数后，负对数的相关部分是：

$$E(y)=\frac{\lVert x_t-\sqrt{\alpha_t}y\rVert^2}{2\beta_t}+\frac{\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}{2(1-\bar\alpha_{t-1})}.$$

一维先展开：

$$\begin{aligned} (x_t-\sqrt{\alpha_t}y)^2&=x_t^2-2\sqrt{\alpha_t}x_ty+\alpha_ty^2,\\ (y-\sqrt{\bar\alpha_{t-1}}x_0)^2&=y^2-2\sqrt{\bar\alpha_{t-1}}x_0y+\bar\alpha_{t-1}x_0^2. \end{aligned}$$

将 `y²` 与 `y` 收集为：

$$\boxed{E(y)=\tfrac12A_ty^2-b_ty+\mathrm{const}},$$

$$A_t=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}},\qquad b_t=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.$$

配方：

$$\tfrac12A_ty^2-b_ty=\tfrac12A_t(y-b_t/A_t)^2+\mathrm{const}.$$

与高斯标准核：

$$\frac{(y-\mu)^2}{2\sigma^2}$$

对比，可直接读出：

$$\boxed{\tilde\mu_t=b_t/A_t,\quad\tilde\beta_t=1/A_t}.$$

**精度（Precision）** 是方差的倒数 `1/variance`，上述 `A_t` 就是精度。两个高斯约束相乘后精度相加，结果比单独使用任一个约束通常更集中。向量情形将 `y²` 换成 `||y||²`，标量 `by` 换成 `b^T y`，这里因为协方差是各向同性的 `cI`，推导仍成立。

<a id="q19"></a>

### **19. 如何从配方结果得到真实后验的均值和方差？**

由 `1/A_t` 通分：

$$\frac1{A_t}=\frac{\beta_t(1-\bar\alpha_{t-1})}{\alpha_t(1-\bar\alpha_{t-1})+\beta_t}.$$

利用 $\alpha_t+\beta_t=1$ 和 $\alpha_t\bar\alpha_{t-1}=\bar\alpha_t$，分母化为：

$$\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t.$$

最终：

$$\boxed{\tilde\beta_t=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}},$$

$$\boxed{\tilde\mu_t(x_t,x_0)=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t}.$$

对应代码：

```python
post_c0 = prev_abar.sqrt() * beta / (1 - abar)
post_ct = alpha.sqrt() * (1 - prev_abar) / (1 - abar)
post_var = beta * (1 - prev_abar) / (1 - abar)
mean = post_c0[t] * x0 + post_ct[t] * xt
```

这里的 `post_c0` 与 `post_ct` **未必相加为 1**，因为它们不是不含缩放的普通凸组合系数。

<a id="q20"></a>

### **20. 能不能用一组具体数字，把贝叶斯、配方和后验算出来？**

令：$\alpha_t=0.8,\ \beta_t=0.2,\ \bar\alpha_{t-1}=0.75,\ x_0=1,\ x_t=0.5$，因此 $\bar\alpha_t=0.6$。

第一条约束 $q(y\mid x_0)$：均值 `sqrt(0.75)=0.8660`，方差 `0.25`。

第二条似然 $q(x_t=0.5\mid y)$ 看作 `y` 的函数：中心约为 `0.5/sqrt(0.8)=0.5590`，其关于 `y` 的高斯核方差为 `beta/alpha=0.25`。

两个方差同为 0.25，合并的精度 `A=4+4=8`；后验方差 `1/8=0.125`，后验均值是两个中心的等精度平均：

$$\tilde\mu\approx(0.8660+0.5590)/2=0.7125.$$

直接代系数也得到：

$$c_0=\frac{\sqrt{0.75}\times0.2}{0.4}\approx0.4330,\quad c_t=\frac{\sqrt{0.8}\times0.25}{0.4}\approx0.5590,$$

$$\tilde\mu=c_0(1)+c_t(0.5)\approx0.7125,\quad\tilde\beta=0.125.$$

**结论**：$q(y\mid x_t=0.5,x_0=1)\approx\mathcal N(0.7125,0.125)$；这里得到的是一个 **分布**，而不是“上一时刻唯一的像素值”。

<a id="q21"></a>

### **21. 为什么公式常说 `t≥2`？最后一步发生了什么？**

当数学时间 `t=1` 时，给定真实 `x0` 的 $q(x_0\mid x_1,x_0)$ 已经没有任何不确定性，是集中在 `x0` 的 **点质量**。此时 `alpha_bar_0=1`、后验方差为 0。代码用 `previous[0]=1`，使 `post_c0[0]=1, post_ct[0]=0, post_var[0]=0`，并在最后一步不加随机项。

要分清：**代码索引 `0` 对应数学上第一步，也是反向生成最后一次更新**。

---

<a id="vlb"></a>

## **六、`q`、`p_theta`、最大似然、KL 与 VLB 的基本理解**

> **学习边界**：你曾明确表示 VLB 完整分解暂时搁置；后来已经讨论了分解的直觉。本章保留理解公式所需的最小链路和可复习推导，不要求你立即从零熟练复现全部变分证明。

<a id="q22"></a>

### **22. `q` 的前向、`q` 的反向、`p_theta` 的反向有什么区别？**

| 记号 | 条件含义 | 能否利用固定日程直接计算？ | 用途 |
|---|---|---|---|
| $q(x_t\mid x_{t-1})$ | 已知上一状态 → 当前状态 | ✅ | 单步前向加噪 |
| $q(x_t\mid x_0)$ | 已知干净图 → 任意时间带噪图 | ✅ | 训练时一次构造 `xt` |
| $q(x_{t-1}\mid x_t,x_0)$ | 原图、当前图都知道 → 上一状态 | ✅ | 贝叶斯已知原图后验，训练参考 |
| $q(x_{t-1}\mid x_t)$ | 只知道当前带噪图 → 上一状态 | 一般不易解析 | 真实数据对应的未知逆向转移 |
| $p(x_T)=\mathcal N(0,I)$ | 不给任何图像 → 初始高斯 | ✅ | 生成起点 |
| $p_\theta(x_{t-1}\mid x_t)$ | 只知道当前带噪图 → 上一状态 | 需 U-Net 预测 | 模型生成时执行的反向转移 |

真实无原图反向分布可形式上写为：

$$q(x_{t-1}\mid x_t)=\int q(x_{t-1}\mid x_t,x_0)q(x_0\mid x_t)\,dx_0.$$

难点在于未知的真实数据分布使 $q(x_0\mid x_t)$ 一般不可直接求。**知道 `x0` 的高斯后验与不知 `x0` 的真实反向分布通常不是同一个分布**。

`theta` 是 U-Net 的可训练权重参数；不是时间步，也不是 `alpha`、`beta`。反向 `p_theta` 是模型近似，不是数学上已知的精确 `q`。

<a id="q23"></a>

### **23. 为什么最大似然要对所有中间路径积分？**

模型定义整条生成路径：

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).$$

最终样本的边缘密度：

$$p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}.$$

目标是最大化真实数据平均对数密度，但高维路径积分难以高效精确计算。**能从生成链采出图片，不等于能精确计算指定图片的概率密度。**

<a id="q24"></a>

### **24. 为什么要把已知前向分布 `q` 引入积分？**

这是乘以 `q/q=1` 的代数改写（假设相关位置上密度比有定义）：

$$\begin{aligned} p_\theta(x_0)&=\int q(x_{1:T}\mid x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}dx_{1:T}\\ &=\mathbb E_{q(x_{1:T}\mid x_0)}\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]. \end{aligned}$$

`q` 是已知、能采样的前向过程。引入它让表达式变为在 `q` 下的期望，**但改写本身尚未解决全部计算难度**。

<a id="q25"></a>

### **25. Jensen 不等式、ELBO/VLB 分别是什么意思？**

因为 `log` 是凹函数，对满足相应期望存在条件的正随机变量 `X` 有：

$$\log\mathbb E[X]\ge\mathbb E[\log X].$$

**数字例子**：等概率取 `X=1` 或 `X=9`：`log E[X]=log 5≈1.609`，`E[log X]=(log1+log9)/2≈1.099`。

将 $X=p_\theta(x_{0:T})/q(x_{1:T}\mid x_0)$ 代入得到：

$$\log p_\theta(x_0)\ge\mathbb E_q\left[\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)\right].$$

右侧称 **ELBO（对数似然的下界）**。取负号后：

$$\boxed{-\log p_\theta(x_0)\le L_{\rm VLB}=\mathbb E_q[\log q-\log p_\theta]}.$$

也就是说 $L_{\mathrm{VLB}}=-\mathrm{ELBO}$，是负对数似然的 **上界**。最小化上界是有依据的代理目标，但并非等于精确计算真实图像似然。

<a id="q26"></a>

### **26. KL 为什么设计成 `E_q[log(q/p)]`，而不是单独 `log(p/q)`？**

**核心回答**：$D_{\mathrm{KL}}(q\Vert p)$ 是以 `q` 为参考、在 `q` 下加权的对数密度比：

$$\boxed{D_{\rm KL}(q\Vert p)=\mathbb E_{x\sim q}\left[\log\frac{q(x)}{p(x)}\right]}.$$

它非负；分布相同则为 0；**不对称**。`log(q/p)` 的单项可以为负数，非负性属于 **整体期望**。

离散例子：`q=(0.8,0.2)`、`p=(0.5,0.5)`，

$$D_{\rm KL}(q\Vert p)=0.8\log(1.6)+0.2\log(0.4)\approx0.193,$$

反向则：

$$D_{\rm KL}(p\Vert q)=0.5\log(0.5/0.8)+0.5\log(0.5/0.2)\approx0.223.$$

为什么有 `log`？因为 $\log(q/p)=\log q-\log p$，其期望可理解为“用 `p` 描述 `q` 的额外负对数代价”。它不是欧氏距离，也不能仅仅翻转比值而仍在 `q` 下求平均，冒充反方向 KL。

<a id="q27"></a>

### **27. VLB 最后为什么出现三项？（复习结构即可）**

利用已知原图的反向链式分解：

$$q(x_{1:T}\mid x_0)=q(x_T\mid x_0)\prod_{t=2}^{T}q(x_{t-1}\mid x_t,x_0),$$

对比模型联合分布后，可以把 `L_VLB` 的对数比值整理为：

$$\boxed{\begin{aligned} L_{\rm VLB}(x_0)={}&D_{\rm KL}(q(x_T\mid x_0)\Vert p(x_T))\\ &+\sum_{t=2}^{T}\mathbb E_{q(x_t\mid x_0)}D_{\rm KL}(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t))\\ &-\mathbb E_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1). \end{aligned}}$$

**三项的意义**：

1. **终点 KL**：固定前向终点 `q(xT|x0)` 与生成初始 $p(x_T)=\mathcal N(0,I)$ 的匹配；固定日程时不依赖网络参数。
2. **中间 KL**：训练时可计算 $q(x_{t-1}\mid x_t,x_0)$，希望模型仅用 `xt` 也能给出合理的反向分布。
3. **最后重建项**：从 `x1` 产生真实 `x0` 的负对数密度；最后一步是点质量后验，不应粗暴按正方差的高斯 KL 处理。

**不要误记**：这是一条 **变分数学目标的分解**，而你教学版训练函数使用的是更简单的噪声 MSE，不等于精确实现完整 VLB 或像素似然评估。

<a id="q28"></a>

### **28. 为什么最后能使用噪声预测 MSE，而不是显式算 KL？**

在数学时间 $t\ge2$、反向方差为固定 $\sigma_t^2>0$ 的各向同性高斯假设下，KL 中依赖 $\theta$ 的部分正比于均值平方误差。以下等价式还假设直接使用预测噪声构造均值，不对 $\hat x_0$ 施加额外裁剪：

$$D_{\rm KL}(q\Vert p_\theta)=\frac{\Vert \tilde\mu_t-\mu_\theta\Vert ^2}{2\sigma_t^2}+\text{不依赖 }\theta\text{ 的项}.$$

由前向公式估计原图：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},$$

可将反向均值整理为：

$$\boxed{\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)}.$$

用真实累计噪声 `eps` 代入可得对应后验均值，因此：

$$\tilde\mu_t-\mu_\theta=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\epsilon_\theta-\epsilon).$$

于是中间 KL 的参数相关部分为：

$$\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\lVert\epsilon_\theta-\epsilon\rVert_2^2.$$

**关键限定**：严格 VLB 对不同 `t` 有权重，并包含终点、重建项；教学版 **Simple Loss** 通常去掉该时间权重：

$$\boxed{L_{\rm simple}=\mathbb E_{x_0,t,\epsilon}\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2},$$

代码中 `F.mse_loss(..., reduction='mean')` 还对元素数取平均（相当于固定维度时再乘一个常数）。**不能宣称 Simple MSE 与完整 VLB 恒等。**

---
<a id="scheduler"></a>

## **七、Scheduler 接口与个人实现**

<a id="q29"></a>

### **29. Noise schedule、Scheduler、U-Net（Model）各干什么？**

| 名称 | 本例职责 | 参数是否训练 |
|---|---|---|
| Noise schedule | 预先规定每个时间索引 `beta/alpha/alpha_bar` | 否 |
| `DDPMScheduler` | 查表、加噪、损失、后验系数、反向采样 | 固定 Buffer，不用 Optimizer |
| `TinyUNet` | 输入带噪图和时间，预测累计等效噪声 | **是** |
| `Optimizer` | 使用梯度更新 U-Net 权重 | 自身管理优化状态 |
| `train_denoiser()` | 组织批次/循环、梯度、日志等 | 组织者，不是另一套扩散公式 |
| `sample_ddpm()` | 循环 U-Net 和 Scheduler，形成整条反向路径 | 推理时不更新权重 |

不能把 **`loss.backward()`（自动微分的反向传播）** 与 **`scheduler.step()`（扩散的反向一步）** 混为一谈。

<a id="q30"></a>

### **30. `_extract(table,timesteps,image)` 中的 table、返回值类型是什么？**

`table` 是形状 `[T]` 的一维浮点 `torch.Tensor`，通常是通过 `register_buffer` 保存的 `signals/noise_scales/post_var` 等系数表。`timesteps` 是 `[B]` 的 `torch.long` 索引。`table[timesteps]` 得到 `[B]` 的浮点 Tensor；对四维图像 reshape 为 `[B,1,1,1]`，可以正确广播到 `[B,C,H,W]`。

```python
table = torch.tensor([0.95, 0.80, 0.50])       # [T=3]
t = torch.tensor([0, 2], dtype=torch.long)    # [B=2]
a = table[t]                                  # [0.95, 0.50], shape [2]
coef = a.reshape(2, 1, 1, 1)                 # [2,1,1,1]
img = torch.ones(2, 1, 28, 28)
y = coef * img                               # [2,1,28,28]
```

原因：PyTorch 广播从 **末尾维度** 对齐，不能拿 `[B]` 直接当“每张图片的系数”乘 `[B,C,H,W]`。

<a id="q31"></a>

### **31. 为什么 `register_buffer`，而不是 `nn.Parameter` 或普通 Python 变量？**

`register_buffer` 适合 **需要跟模型一起保存与迁移设备、但不参与梯度优化** 的张量。噪声日程表由预定规则计算得出，无需训练：

```python
self.register_buffer("alpha_bars", abar.float())
```

它会随 `.to(device)` 迁移；默认持久化的 buffer 会进入 `state_dict`，但不在 `model.parameters()` 中。

<a id="q32"></a>

### **32. 为什么 `add_noise()` 和 `loss()` 要接收同一份 `noise`？**

因为这一次随机噪声既是 **出题用的扰动**，也是 **监督标签**。如果构造 `xt` 用 `eps_A`，而 Loss 却拿新抽的 `eps_B`，就是用错误答案批改模型。

```python
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)
loss = scheduler.loss(eps_pred, eps)       # 必须是同一份 eps
```

数值例：真实 `eps=-1`，预测 `eps_pred=-0.8`，MSE 为 `0.04`；若误用新噪声 `eps_B=0.2`，MSE 变成 `1.00`，监督目标已不对应输入。

<a id="q33"></a>

### **33. `F.mse_loss(..., reduction='mean')` 是什么，还有什么选项？**

`mean` 将所有 `B×C×H×W` 位置的误差平方求平均，输出标量；`sum` 求总和；`none` 保留每个元素平方误差，返回与输入同形状的 Tensor。

例如 `pred=[1,3]`、`truth=[2,5]`，平方误差 `[1,4]`：`mean=2.5`、`sum=5`、`none=[1,4]`。使用 `mean` 可以避免 Loss 单纯因为 Batch Size 或分辨率改变而等比例变大。

<a id="q34"></a>

### **34. 后验三个系数为什么可以在 `__init__()` 一次性计算？**

因为 `post_c0(t)`、`post_ct(t)`、`post_var(t)` **只依赖预设 beta 日程**，不依赖具体图片，不依赖 U-Net 参数。于是：

```python
post_c0 = prev_abar.sqrt() * beta / (1 - abar)
post_ct = alpha.sqrt() * (1 - prev_abar) / (1 - abar)
post_var = beta * (1 - prev_abar) / (1 - abar)
```

前向后验均值为 `post_c0[t] * x0 + post_ct[t] * xt`；生成时把 `x0` 替换为 `x0_hat`。**预计算系数不等于预计算每张图片的后验均值**。

<a id="q35"></a>

### **35. `step()` 到底依次做哪三件事？**

在当前代码中：

```python
# sample 是当前 xt，pred_noise 是 U-Net 给出的 eps_theta
pred_x0 = (sample - noise_scale[t] * pred_noise) / signal[t]
if clip_denoised:
    pred_x0 = pred_x0.clamp(-1, 1)
mean, var = posterior(pred_x0, sample, t)
previous = mean if t == 0 else mean + var.sqrt() * z
```

对应：

1. 由 `xt` 和预测累计噪声得到 `x0_hat`；
2. 用 `x0_hat` 与 `xt` 构造模型的反向均值，以及本步预设方差；
3. 对 `t>0` 从对应高斯采样上一状态；最后一步不加新噪声。

`pred_x0` 不保证与某张真实原图一致；这是 **生成时模型的估计**。

<a id="q36"></a>

### **36. `step()` 的 `noise` 与前向 `add_noise()` 的 `noise` 是同一份吗？**

**不是。** 三个量分工不同：

| 变量 | 来源 | 作用 |
|---|---|---|
| `eps` | 训练时 `randn_like(x0)` | 构造 xt 的累计等效噪声、MSE 真值 |
| `eps_pred` | `model(xt,t)` | U-Net 预测的累计等效噪声，构造反向均值 |
| `z` | 生成时新的 `randn_like(sample)` | 从反向高斯里抽到 **具体** 的上一状态 |

同一组训练参数、固定 `xt` 与 `t`，`eps_pred` 是确定的；但 `step` 若重新抽 `z`，输出 **通常不确定**。最后一步是例外。

<a id="q37"></a>

### **37. `len(sample)` 为什么等于 Batch Size，而不是维度数 4？**

对于 `[B,C,H,W]`：

- `len(sample) == sample.shape[0] == B`：第一维长度；
- `sample.ndim == len(sample.shape) == 4`：维度 **个数**；
- `sample.numel() == B*C*H*W`：元素总数。

因此 `torch.full((len(sample),), k)` 生成每张图一个时间索引，Shape `[B]`。

<a id="q38"></a>

### **38. `clamp(-1,1)` 是什么？与截断正态初始化一样吗？**

`clamp` 对每个元素执行：小于 -1 的改为 -1，大于 1 的改为 1，中间值不变。例 `[-1.7,-0.4,0.6,1.5]` → `[-1,-0.4,0.6,1]`。

这里限制的是 **预测干净图 `x0_hat`**，因为数据被映射到 `[-1,1]`；并非把高斯噪声整体限制到此范围。它和 `trunc_normal_` **不同**：前者是直接对数值裁切，后者描述从截断后的概率分布采样的初始化方法。

<a id="q39"></a>

### **39. 生成时为什么“知道” alpha、alpha_bar？不是应该一切都靠模型猜吗？**

因为噪声日程是 **我们自己设计的规则**，既然训练时按它加噪，生成时当然可以使用同一张表计算反向系数。这不是偷偷获得真实图片信息。**未知的是“当前带噪图像里，哪部分是符合数据分布的结构”**，这才需要 U-Net 学习。

<a id="q40"></a>

### **40. 自己写的 Scheduler 代码审查：哪些已正确，哪些测试要改？**

你提交的个人 `DDPMScheduler` 中，`linspace/cumprod/previous`、后验三系数、`_extract` 广播、`add_noise`、`loss`、`_posterior`、`step` **核心公式是正确的**；但存在两个测试逻辑错误：

1. `scheduler.loss(xt, eps)` 比较的是 **带噪图和噪声**，不是网络预测与标签。无 U-Net 时可测试 `loss(eps+0.1, eps)≈0.01`。
2. 用时间索引 `[0,1,7,19]` 给 4 张图加噪，随后对整个 Batch 使用 `step(..., timestep=19, ...)`，**时间不匹配**。应统一构造 `t=[19,19,19,19]` 后再测 `step(..., 19, ...)`。

此外检查：原占位 Docstring（“当前不返回”）需要更新；`pred_noise` 与调用方可能用的 `predicted_noise` 形参名应统一；粘贴代码中的测试缩进需检查；移除未使用的 `pending` 导入。

#### **i. 四个可独立复习的最小数值测试**

```python
scheduler = DDPMScheduler(num_steps=20)
x0 = torch.randn(4, 1, 28, 28)
eps = torch.randn_like(x0)

# 1. 不同样本可取不同 t，add_noise 输出同形
t = torch.tensor([0, 1, 7, 19], dtype=torch.long)
assert scheduler.add_noise(x0, eps, t).shape == x0.shape

# 2. MSE 目标可手算：预测比真值恒大 0.1 → 0.01
assert abs(scheduler.loss(eps + 0.1, eps).item() - 0.01) < 1e-6

# 3. 同一时间索引、真实累计 eps，z=0 时 step 返回后验均值
t_same = torch.full((4,), 19, dtype=torch.long)
xt = scheduler.add_noise(x0, eps, t_same)
mean, var = scheduler._posterior(x0, xt, t_same)
step_mean = scheduler.step(eps, 19, xt, torch.zeros_like(xt))
torch.testing.assert_close(step_mean, mean, atol=1e-6, rtol=1e-6)

# 4. 最后一步，理想预测真实 eps，应恢复 x0；不会额外加噪
t0 = torch.zeros(4, dtype=torch.long)
xt0 = scheduler.add_noise(x0, eps, t0)
recovered = scheduler.step(eps, 0, xt0, torch.randn_like(xt0))
torch.testing.assert_close(recovered, x0, atol=1e-6, rtol=1e-6)
```

**完成状态提醒**：这四项是 **建议运行的验收脚本**，不能因为它们写进笔记就宣称个人代码已通过完整对照。真正替换教学 Scheduler 仍需相同随机数、相同日程的参考实现数值比较与最小集成测试。

---
<a id="unet"></a>

## **八、时间条件 U-Net：为什么这样设计？**

<a id="q41"></a>

### **41. `timesteps`、`time_width`、`time` 是同一个东西吗？为什么默认 128？**

**不是**：

- `timesteps`：每个样本的噪声时间索引，形状 `[B]`，例如 `[10,900]`。
- `time_width=128`：时间特征向量的 **宽度**，人为选择的网络超参数；不是扩散步数。
- `time = time_embedding(timesteps)`：已经计算出来的时间特征，形状 `[B,128]`。

128 只是表达容量与算力成本的折中，64、256 等也可用。该编码把宽度均分为 Sin/Cos 两部分，所以当前实现要求使用正偶数宽度。

<a id="q42"></a>

### **42. TimeEmbedding 不就是随机初始化的乱矩阵吗？时间凭什么有作用？**

当前代码分两部分：

1. **确定的多频率正弦余弦特征**：$\omega_i=10000^{-i/(D/2)}$，`sin(t*freq_i),cos(t*freq_i)`；给同一个 `t`，这部分始终相同；第一周代码按先 sin、后 cos 的顺序拼接。
2. **可训练的 MLP**：`Linear → SiLU → Linear`，初始化权重随机，但在 MSE 的反向传播中不断被更新。

随后每个 TimeBlock 用自己的可训练 `Linear(time_width,C_out)` 将时间向量投影成 `[B,C_out]`，扩维为 `[B,C_out,1,1]`，加到 `[B,C_out,H,W]` 的特征图上：

$$h'_{b,c,i,j}=h_{b,c,i,j}+g_c(e_{t_b}).$$

例：某张图的 2 个特征通道得到 `[+0.2,-0.5]` 的时间偏置，即第 1 通道所有像素加 0.2，第 2 通道所有像素减 0.5。**它既为匹配通道数，也学习各层如何按噪声强度调节特征**。

<a id="q43"></a>

### **43. GroupNorm 如何归一化？是按组内 `C_group × H × W` 一起算吗？**

**对。** 输入 `[B,C,H,W]`，`GroupNorm(G,C)` 将每张图片的 C 个通道分为 G 组；每组含 `C/G` 个通道，在这组的 **全部通道及空间位置** 上求均值/方差，不跨 Batch，不跨组。

$$M=(C/G)HW,\quad\mu_{b,g}=\frac1M\sum_{k\in(b,g)}x_k,\quad v_{b,g}=\frac1M\sum_{k\in(b,g)}(x_k-\mu_{b,g})^2,$$

$$\hat x_k=\frac{x_k-\mu_{b,g}}{\sqrt{v_{b,g}+\epsilon}},\qquad y_k=\gamma_c\hat x_k+\beta_c.$$

**手算**：某图某组 4 个元素 `[1,2,3,4]`，均值 `2.5`，方差 `1.25`，忽略 eps 且令 `gamma=1,beta=0`，归一化约 `[-1.342,-0.447,0.447,1.342]`。

- `GroupNorm(8,64)`：每组 8 通道，总统计数是 `8×H×W`。
- Transformer 中的 LayerNorm 通常按一个 Token 的隐藏维统计；但 LayerNorm 实际归一化的轴由 `normalized_shape` 决定，并非永远只有“最后一维”。
- RMSNorm 不减均值，只按均方根缩放。

<a id="q44"></a>

### **44. TimeBlock 为何三次 SiLU？为什么不严格对称？**

当前真实顺序：

```text
Conv1 → GroupNorm1 → SiLU① → +TimeBias → SiLU②
      → Conv2 → GroupNorm2 → +Residual → SiLU③
```

① 对第一段卷积特征加入非线性；② 时间条件加入后再经非线性影响后续卷积；③ 对主分支与残差相加后的特征激活。并无数学规定必须三次或对称。你可以构造前激活、后激活等不同残差块，它们代表不同的架构设计。

数值直觉：特征 `0.3` 加时间偏置 `-0.8` 后得到 `-0.5`；经过 SiLU 后约 `-0.189`，非线性会改变简单加性调节的效果。

<a id="q45"></a>

### **45. `skip` 为什么用 `1×1 Conv`，不能用 `Linear`？**

`Conv2d(C_in,C_out,1)` 本质是 **每个空间位置共享一个通道线性变换**：

$$y_{i,j}=Wx_{i,j}+b,\quad W\in\mathbb R^{C_{out}\times C_{in}}.$$

`nn.Linear` 数学上也可以实现，但通常需要把 `[B,C,H,W]` 变成 `[B,H,W,C]` 再转回。`1×1 Conv` 更符合 CNN 的张量布局；它主要用于让残差主支与捷径分支 **通道数一致**。

<a id="q46"></a>

### **46. `base` 是什么？是否等于输入图片通道数？**

`base=32` 是基础 **特征通道宽度**，不等于输入通道、Batch Size 或时间步数。中间层通道数大致为：

```text
灰度输入 1 → 32 → 64 → 128 → 64 → 32 → 灰度输出 1
```

`base=64` 会扩大对应通道，通常增加容量与计算开销；当前 GroupNorm 的组数为 8，要求所用的输出通道可以被 8 整除。

<a id="q47"></a>

### **47. TimeBlock 输出的是什么？为什么每个层级都加入时间？**

`TimeBlock(image_features,time_embedding)` 输出的是 **中间特征图** `[B,C_out,H,W]`，不是最终噪声。只有 `TinyUNet.out` 将特征映射回图片通道数，得到 `eps_theta`。

不同时间步的噪声强度差别很大，浅层局部特征与深层整合特征都可能需要不同策略；多层注入让每层可以直接学习对时间的响应。它是合理的架构选择，但 **不是每个层都必须加时间的定理**。

<a id="q48"></a>

### **48. 什么是下采样与上采样？为什么 U-Net 先下后上？**

- 下采样：降低 H/W；本模型用 `Conv2d(k=4,stride=2,padding=1)`：`28→14→7`。它是 **带可训练卷积核** 的空间降采样。
- 上采样：提高 H/W；本模型用 `F.interpolate(mode='nearest')`：`7→14→28`，**插值操作自身无可训练参数**。

先下：扩大有效感受野，在较小空间尺度整合更大范围结构；再上：恢复逐像素预测所需的空间分辨率。分类 CNN 常只下采样并输出类别；噪声预测要返回与图像 **同形状** 的张量，因此通常需要恢复分辨率。

<a id="q49"></a>

### **49. `interpolate` 的输入输出怎么理解？**

例如最近邻插值，把：

$$\begin{bmatrix}1&2\\3&4\end{bmatrix}\quad\to\quad\begin{bmatrix}1&1&2&2\\1&1&2&2\\3&3&4&4\\3&3&4&4\end{bmatrix}.$$

它只改变空间采样密度，不会自动恢复丢失的信息。`size=skip2.shape[-2:]` 的好处是精确匹配目标 Skip 的 H/W，从而允许 `torch.cat`。

<a id="q50"></a>

### **50. 为什么有两个 Skip？它们分别跳过了什么？**

**TimeBlock 内部的残差 Skip**：

```python
return F.silu(hidden + self.skip(image))
```

将输入经恒等映射（同通道）或 `1×1 Conv`（通道数变化）后 **相加**，让信息和梯度跨过主分支卷积。

**U-Net 编码器—解码器的跨层 Skip**：

```python
hidden = torch.cat([hidden, skip2], dim=1)
```

将浅层、较高分辨率特征与上采样后的深层特征在通道维上 **拼接**，帮助恢复细节。一个是 **加法**、一个是 **拼接**；形状及目的不同。

<a id="q51"></a>

### **51. `enc1` 只接收 1 通道，RGB 图片不能用吗？**

你的版本为 MNIST 灰度图设计：`enc1=TimeBlock(1,base,...)`、输出 `Conv2d(base,1,...)`。RGB 输入 `[B,3,H,W]` 时必须 **同时把首层输入与末层输出改为 3**；中间 `base` 不必因为 RGB 而改变。

<a id="q52"></a>

### **52. TinyUNet 完整 Forward 的 Shape 是什么？**

假设 `B=2,base=32,time_width=128`：

| 操作 | Shape |
|---|---|
| `image` | `[2,1,28,28]` |
| `time_embedding(t)` | `[2,128]` |
| `enc1` | `[2,32,28,28]` |
| `down1` | `[2,32,14,14]` |
| `enc2` | `[2,64,14,14]` |
| `down2` | `[2,64,7,7]` |
| `middle` | `[2,128,7,7]` |
| `interpolate` + `cat(skip2)` | `[2,192,14,14]` |
| `dec2` | `[2,64,14,14]` |
| `interpolate` + `cat(skip1)` | `[2,96,28,28]` |
| `dec1` | `[2,32,28,28]` |
| `out` | `[2,1,28,28]` |

末层不加 Sigmoid/Softmax/Tanh，因为预测目标是 **取值不受 `[-1,1]` 限制的高斯噪声**。当前用户的教学模型先前打印约 **654,785** 个可训练参数。

<a id="q53"></a>

### **53. 为什么第一周不要求从零写完整 U-Net？**

因为这阶段最核心的个人算法目标是 DDPM 的 **加噪、后验与反向采样**。U-Net 是噪声估计主干，可以先读懂并使用 PyTorch 的 `Conv2d/GroupNorm/SiLU`；只要求能解释 `TimeEmbedding.forward`、`TimeBlock.forward`、`TinyUNet.forward` 与 Shape。之后学习 DiT 时，Transformer 的 Patch、Block 与条件调制会成为新的架构主线。

---

<a id="mnist"></a>

## **九、MNIST、Dataset 与 DataLoader：数据到底从哪里来？**

<a id="q54"></a>

### **54. MNIST 有什么数据？为什么一批既有 `images` 又有 `labels`？**

MNIST 是 28×28 单通道手写数字图像（标签 0–9），官方标准数据量：**60,000 训练、10,000 测试**。一个单样本通常是 `(image,label)`：`image` 为 `[1,28,28]`，`label` 是整数类别。

`DataLoader` 会把 B 个样本自动组织为：

```python
images, labels = next(iter(train_loader))
# images: [64,1,28,28]
# labels: [64]
```

你的 **无条件 DDPM** 不使用数字标签，因为监督目标是 **我们亲自生成的噪声 eps**，不是数字类别；故常写 `for images, _ in train_loader`。将来做类别条件生成才可能使用 `labels`。

<a id="q55"></a>

### **55. 为什么 `manifest.json` 里看不到图片？实际数据从哪里读取？**

`manifest` 是 **元数据与完整性记录**：样本数量、划分指纹、文件大小、MD5/SHA256 校验与是否使用官方测试集。**它不是存储图像像素的数据库**。

`prepare_mnist(ROOT/'data',seed=SEED)` 调用项目的数据模块及 Torchvision MNIST 去下载/读取图像与标签，并对官方训练数据做当前实验所需的划分；实际磁盘路径以 `prepare_mnist` 代码为准。Torchvision 的常见目录含 `MNIST/raw/train-images-idx3-ubyte` 等原始 IDX 文件及标签文件。

```python
import inspect
print(inspect.getsource(prepare_mnist))
print(type(train_set), len(train_set))
image, label = train_set[0]
print(image.shape, label)
```

此教学实验的划分为 **10,000 张实验训练、1,000 张实验验证**；官方测试集不参与这轮调参。

<a id="q56"></a>

### **56. Dataset 和 DataLoader 有何区别？`shuffle`、`batch_size` 等干什么？**

- `Dataset`：定义 `__len__()` 与 `__getitem__()`；知道索引 `i` 应该返回哪张图和标签。
- `DataLoader`：从 Dataset 组织 Batch、决定顺序、控制读取进程。

```python
train_loader = DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(SEED),
)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
```

`shuffle=True` 训练时随机排列样本顺序；验证用 `shuffle=False` 可稳定顺序。`num_workers=0` 在主进程加载；`generator` 控制 DataLoader 相关随机性以提高可复现性。常见流程是 `ToTensor` 将像素映射到 `[0,1]`，再 `(x-0.5)/0.5` 映射到 `[-1,1]`；显示时用 `(x+1)/2` 恢复为可视化范围。

---
<a id="training"></a>

## **十、训练与验证：为什么随机 t、为什么先过拟合？**

<a id="q57"></a>

### **57. 训练时为什么每个样本随机选择时间步，而不是顺序从 0 跑到 999？**

**核心回答**：随机时间是为了让 **同一个 U-Net** 在有限计算成本下不断接触各种噪声强度。训练时可由累计加噪公式 **直接** 构造任意时间 `xt`，不需要先走过此前所有时间。

```python
t = torch.randint(1000, (len(x0),), device=x0.device)
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
```

目标可理解为：

$$L(\theta)=\frac1T\sum_{t=1}^T\mathbb E_{x_0,\epsilon}[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2].$$

均匀抽时间步相当于使用 Monte Carlo 样本估计不同时间的平均损失。**顺序遍历也不是数学上不允许**，但随机选择便于每个 Batch 覆盖不同噪声水平。

<a id="q58"></a>

### **58. 不同 `t` 会训练不同的一套参数吗？**

**不会。** `model(x_t,t)` 对所有时间步共享 **同一套** U-Net 权重 `theta`；时间编码使它能够根据 `t` 调整内部特征。不同时间的样本贡献各自的梯度，最后一起更新同一个网络。**Scheduler 的 alpha/beta 是固定表，不随这些梯度更新。**

<a id="q59"></a>

### **59. 为什么 Warmup 只调用 `backward()` 三次，却没有训练效果？**

Warmup 用来检查前向计算、损失是否有限、梯度能否传播，以及设备短跑耗时：

```python
model.zero_grad(set_to_none=True)
loss = F.mse_loss(model(scheduler.add_noise(x0,eps,t),t), eps)
loss.backward()
```

这里 **没有 `optimizer.step()`**，所以模型参数不更新；三次随机损失没有义务连续下降。`model.eval()` 也不等于 `torch.no_grad()`：前者切换某些层的训练/评估行为，后者关闭 Autograd 记录。

<a id="q60"></a>

### **60. 为什么要故意用固定 16 张图片、固定 `t=250`、固定噪声做过拟合？**

这是经典的 **Overfit a Small Batch** 调试方法。先制造一个非常简单的“同一个输入映射到同一个噪声答案”的任务，验证 Optimizer 和模型是否有能力把它拟合下来。

固定三元组 `fixed_probe=(fixed_x,fixed_eps,fixed_t)` 使每次测试具有可比性：

```text
相同的 16 张 x0 + 同一个 t + 同一组 eps
             ↓
           相同 xt
             ↓
        更新 U-Net 100 次
             ↓
      MSE 是否显著下降？
```

**不要误会**：固定任务拟合好，只证明这组监督映射可被优化；不是证明模型能泛化到新噪声、新时间、新图片，更不是生成质量评测。

<a id="q61"></a>

### **61. 我画的图到底是 Train Loss 还是 Validation Loss？横轴是什么？**

之前固定过拟合曲线：横轴 `Optimizer steps` 是 **网络参数更新次数 1～100**；纵轴 `Fixed-pair MSE` 是 **固定训练三元组的损失**，不是扩散时间 `t`，也不是验证集损失。

你提供的实际数值：

| 检查 | MSE |
|---|---:|
| 初始固定探针 | 1.0771076679 |
| Step 50 日志 | 0.07591 |
| Step 100 日志 | 0.01059 |
| 最终固定探针 | **0.0099809347** |

`final/initial≈0.0092664`，即固定探针 MSE **下降约 99.07%**。日志中的第 100 步 Loss 与最终探针的略微差异，可能来自 **参数更新前后评估位置** 不同。这个训练测试支持“固定映射被明显拟合”，但不能推出“数字生成准确率 99%”。

<a id="q62"></a>

### **62. 换一份随机噪声，为什么 Loss 又变大？**

你在 **相同 16 张图片、相同 `t=250`** 下重新抽噪声，得到：

| 训练后测试 | MSE |
|---|---:|
| 仍用原固定噪声 | 0.00998 |
| 改成新噪声 | **0.08438** |

更换噪声后误差明显增大，说明模型对原固定配对拟合得尤其好。新噪声的 MSE 仍小于初始约 1.077，但 **图片和时间都没有改变**，不能把这一次测试当作对全部新图片和时间的泛化保证。

<a id="q63"></a>

### **63. 真正的随机 MNIST 训练与固定过拟合有什么区别？**

正式短训练使用 `train_loader` 取不同 MNIST 图片，并通常每 Batch 重新抽 `t` 和 `eps`：

```python
for x0, _ in train_loader:
    t = torch.randint(T, (len(x0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = scheduler.add_noise(x0, eps, t)
    loss = F.mse_loss(model(xt,t), eps)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`TrainConfig(max_steps=1000)` 中的 **Optimizer Step 1000** 与 `Scheduler(num_steps=1000)` 中的 **扩散时间步 1000** 是两个独立超参数，碰巧数值相同。

<a id="q64"></a>

### **64. 验证集 Loss 是怎么计算的？与训练 Loss 的区别是什么？**

**计算公式相同，数据和更新行为不同。**

训练：来自 `train_loader`；需要 `backward()`、`optimizer.step()`。

验证：来自 `val_loader`；用 `model.eval()` 和 `torch.no_grad()`，**不更新参数**；可固定 `(x0,eps,t)` 作为 Probe，比较初始模型、训练后模型、EMA 模型在 **同一个题目** 上的差异。

```python
model.eval()
with torch.no_grad():
    x0, eps, t = val_probe   # 固定 64 张验证图片 + 预先固定噪声与时间
    xt = scheduler.add_noise(x0, eps, t)
    val_loss = F.mse_loss(model(xt,t), eps)
```

真实使用需保证这些 Tensor 与模型在同一设备上。**64 张图片的一次固定 Probe 不是整个验证集上的平均指标**。

<a id="q65"></a>

### **65. 为什么随机训练的 Batch Loss 曲线会上下波动？**

因为每个 Batch 的图片、时间和噪声都随机变化，预测难度不一样。某一步 Loss 更高不必然代表模型退化。Notebook 用 `np.convolve(losses, ones(20)/20, mode='valid')` 计算 **20-step 滑动平均**，仅用于更清楚地观察趋势，不影响反向传播，也不是模型权重的 EMA。

<a id="q66"></a>

### **66. `initial_probe_loss`、`final_probe_loss`、`ema_probe_loss` 哪些是训练集，哪些是验证集？**

本次正式短训的三个 Probe 使用 **同一个固定验证 Batch**，改变的是模型权重状态：初始化模型、训练后普通模型、EMA 模型。因此三者可以公平比较网络参数的改进；但它们仍不是全验证集 MSE，更不是生成图像质量分数。

---

<a id="sampling"></a>

## **十一、反向生成：每一步到底在采什么？EMA 和 Generator 怎么回事？**

<a id="q67"></a>

### **67. `eps_pred` 预测的是从 `x0→xt` 的噪声，还是 `x(t-1)→xt` 的噪声？**

**预测的是训练直接构造 `xt` 时使用的累计等效噪声**：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon_\theta(x_t,t)\approx\epsilon.$$

不是最近一步新加入的 `eps_t`，也不是某条真实生成轨迹上可以直接“减去”的特定噪声。模型预测 `eps_theta` 的目的是用已知日程 **参数化反向分布的均值**。

<a id="q68"></a>

### **68. 预测 `eps_theta` 后，为什么又要估计 `x0_hat`？**

你学习的后验均值公式需要 `x0`，但生成时不知道真实原图。于是利用：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

先估计原图，再代入已知系数：

$$\mu_\theta(x_t,t)=c_0(t)\hat x_0+c_t(t)x_t.$$

它等价于直接写：

$$\boxed{\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)}.$$

两种写法在 **未进行 `clip_denoised` 截断时** 代数等价；开启 `x0_hat.clamp(-1,1)` 后不再保证与未截断的闭式均值逐点相同。

<a id="q69"></a>

### **69. 若知道真实 `eps`，是不是能精确得到真实 `x(t-1)`？**

**不可以这样说。** 真实累计 `eps` 结合 `xt`、`t` 可以 **代数上精确恢复构造时的 `x0`**，但无法唯一找回 **某次逐步加噪历史中实际抽到的 `x(t-1)`**，因为中间仍有随机性。

例如 `sqrt(alpha_bar)=0.8`、`sqrt(1-alpha_bar)=0.6`、`x0=1`、`eps=-0.5`，则 `xt=0.5`。用真实噪声恢复：

$$x_0=\frac{0.5-0.6(-0.5)}{0.8}=1.$$

但仅给定 `x0,xt` 时，上一步的分布 $q(x_{t-1}\mid x_t,x_0)$ 通常仍有 **非零后验方差**；只能从分布采样，并不能确定哪条历史路径发生过。

<a id="q70"></a>

### **70. U-Net 输出固定，为何 Scheduler.step 的下一张图还不固定？**

固定 `theta,xt,t` 且网络没有随机层时，`eps_pred`、`x0_hat`、`mu_theta` 和固定方差确实是确定的；但 DDPM **反向采样本身可以是随机的**：

$$\boxed{x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz},\quad z\sim\mathcal N(0,I),\quad\sigma_t=\sqrt{\sigma_t^2}.$$

若 `mu=0.5,var=0.09`，则 `std=0.3`；`z=-1` 得 `0.2`，`z=1` 得 `0.8`。**均值是分布中心，不是唯一候选结果**。最后一步不额外加随机项。

<a id="q71"></a>

### **71. 为什么反向还要重新抽 `z`，不会又把图弄坏吗？**

`z` 不是“随便加回原始噪声”——它乘以本步由 Scheduler 规定的标准差，与模型计算的均值共同生成 **符合该反向高斯分布的一份样本**。每一步 `mu_theta` 都使用更新后的 `xt` 和 `t` 重新计算，因而反向链可逐步产生结构。

**新噪声是反向概率模型固有的随机性，不等于模型预测误差。** 若模型训练不足，采样分布不准确，生成结果可能模糊或畸形；但不应理解为偏离某条必须恢复的真实路径。

<a id="q72"></a>

### **72. 一次 `sample_ddpm()` 到底执行什么？**

你提供的采样代码先：

```python
checkpoint = torch.load(training_result["checkpoint"], map_location="cpu", weights_only=True)
sampling_model = TinyUNet(base=32).to(DEVICE)
sampling_model.load_state_dict(checkpoint["ema"])
initial_noise = torch.randn((8,1,28,28), generator=torch.Generator().manual_seed(41)).to(DEVICE)
```

含义：创建 **相同架构** 的新 U-Net，加载训练后的 EMA 权重；独立生成 **8 张** 从未对应某张真实 MNIST 的高斯初始噪声图。然后 `sample_ddpm` 循环调用模型与 `scheduler.step`，保存最终 `images` 和中途 `snapshots`。

```text
initial → after k=749 → after k=499 → after k=249 → after k=99 → after k=0
```

每 **列** 对应同一张样本的不同反向状态；`k` 是 **反向采样索引**，不是优化器更新步数。`completed=False` 时不能将未完成的轨迹误称为最终生成样本。

<a id="q73"></a>

### **73. 为什么最后绘图用 `(image.clamp(-1,1)+1)/2`？**

训练的干净 MNIST 图像已归一化到 `[-1,1]`；显示时将区间映回 `[0,1]`。`image[0]` 选出单通道 `[28,28]`，`cmap='gray'` 按灰度显示。这只是 **可视化**，不改变已经完成的反向生成模型。

<a id="q74"></a>

### **74. EMA 是 `nn.Module` 自动保存或自动更新的吗？**

**不是。** EMA（指数移动平均）是一套额外维护的 **平滑模型权重**。训练器需要显式创建、更新、保存：

$$\boxed{\theta_{\rm EMA}\leftarrow d\theta_{\rm EMA}+(1-d)\theta}.$$

假设 `d=0.9`，旧 EMA 参数为 10，新模型参数为 14，则更新后 `0.9×10+0.1×14=10.4`；下一步模型参数变 12，则 EMA 新值 `0.9×10.4+0.1×12=10.56`。训练器可以复制模型，也可以管理参数字典；**都不是 `nn.Module` 自动产生**。因此你没在 `TinyUNet` 类里看到 `ema` 是正常的；应检查 `train_denoiser` 或其辅助模块中的 `ema/decay/lerp_` 逻辑。

EMA 不需要另一套 Loss 的 `backward()`；它只是根据主模型参数逐步平滑，生成时常用 `checkpoint['ema']` 加载这套权重。EMA **不保证** 验证 MSE 或生成质量永远优于当前权重。

<a id="q75"></a>

### **75. `torch.Generator()` 到底是什么？为什么两个不同 seed？**

这是 PyTorch 的 **伪随机数生成器状态对象**，与“图像生成模型”完全无关。`manual_seed(s)` 设置随机数序列的初始状态；**同一 Generator 连续调用会向前推进状态**，不会每次都返回同一个数。

```python
g1 = torch.Generator().manual_seed(41)
a = torch.randn(3, generator=g1)
b = torch.randn(3, generator=g1)            # 一般不同于 a
g2 = torch.Generator().manual_seed(41)
c = torch.randn(3, generator=g2)            # 同环境下可复现 a
```

项目采样中，`seed=41` 用于 **最初的 8 张高斯图**，`seed=42` 交给 `sample_ddpm` 控制内部反向随机数；两条随机序列可以分别固定。若同一设备/实现与所有随机调用顺序相同，生成通常可复现；跨设备或框架版本并不总保证逐位相同。

---
<a id="experiment"></a>

## **十二、实验复盘：我实际跑出了什么？Checkpoint 与数据记录**

<a id="q76"></a>

### **76. 生成的八张图片说明了什么？**

你展示的 EMA 生成图表明：从随机噪声开始，经历完整的 1000 步反向采样后，模型确实生成了 **具有手写数字笔画结构** 的 28×28 灰度图。轨迹中早期仍是噪声，中后期开始出现结构；最终有部分像 1、2、6 等数字，但也有笔画断裂、黏连、类别难辨的样本。

**支持的结论**：教学版模型、Scheduler、反向采样链已经可以工作，并且模型学到了一些 MNIST 结构。

**不支持的结论**：生成数字准确率达某个百分比、已达到论文质量、拥有强泛化、已实现大规模图像生成。

#### **i. 历史实验记录**

| 实验/观察 | 记录 | 含义 |
|---|---|---|
| 固定 16 张过拟合 | 初始 MSE `1.0771076679` → 最终 `0.0099809347` | 同一监督映射被拟合 |
| 固定图片+新噪声 | MSE `0.0843838` | 对新噪声的单次小测试 |
| 随机 MNIST 短训练 | 训练 `1000` Optimizer Steps | 是 **参数更新次数**，不是采样轮数 |
| 随机训练验证 Probe | 初始约 `1.0817`、训练后约 `0.03204`、EMA 约 `0.03137` | 固定验证探针上的对照；EMA 略低 |
| 反向生成 | `8` 张样本、`1000` 次反向时间索引迭代 | 已可见手写数字结构 |

> 表中数据是教学 Notebook 和学习记录中的历史结果，不能代表独立重复试验。材料未报告 FID、分类器准确率或全测试集似然。

<a id="q77"></a>

### **77. `summary` 字典和 `notebook_results.json` 在做什么？**

收集之前已经产生的实验配置与指标：随机种子、设备、PyTorch 版本、数据文件校验与划分、公式测试、固定拟合、训练/验证指标、采样完成状态、局限性。它 **不是重新训练或重新评估**。

```python
summary = {"seed": SEED, "fixed_pair": {...}, "training": {...}, "sampling": {...}}
(OUT / "notebook_results.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
```

`{k: result[k] for k in keys}` 是字典推导式，从大结果字典中只提取想保存的指标。`manifest.json` 是数据元信息，而 `notebook_results.json` 是 **实验运行结果**，两者职责不同。

<a id="q78"></a>

### **78. Checkpoint 是什么？为什么仅保存 U-Net 权重不能保证精确续训？**

**Checkpoint 是训练状态快照**。要恢复与不中断训练尽量一致的后续优化轨迹，通常需要：

- 主模型参数与 EMA 参数；
- Optimizer 状态（AdamW 的一阶、二阶矩、步数等）；
- 学习率调度器状态（若存在）、当前 Optimizer Step；
- 随机数发生器状态；
- 数据读取顺序与位置（若想精确恢复随机打乱的 Epoch 中途状态）。

**只有模型权重并不等于恢复了 AdamW 的历史状态**。

<a id="q79"></a>

### **79. `MinimalDenoiser` 的“三条训练路径”验证了什么？**

极简测试用 `Conv2d(1,1,1)`（单通道时只有权重和偏置两个可训练标量）、固定全零图片的小 Batch，比较：

```text
A：相同初始模型 → 训练 6 步 ─────────────────→ final_A
B：相同初始模型 → 训练 3 步 → 存 checkpoint
                                  └→ 加载并续训至总计 6 步 → final_B
```

代码使用：

```python
torch.testing.assert_close(a, b, rtol=0, atol=0)
```

要求 **普通模型参数完全相等**，随后对 `checkpoint['ema']` 的每个 Tensor 也执行严格相等检查。

通过时只说明：**在这一固定小批次、测试设备及当前实现条件下，连续训练与恢复训练得到相同权重**。不能自动扩展为 `shuffle=True` 的 DataLoader 在 Epoch 中途精确回放已验证，也不保证跨设备、跨版本逐位一致。

<a id="q80"></a>

### **80. 这些工程性内容学到什么程度就够？**

需要知道 `Checkpoint = 状态快照`、`EMA = 平滑权重`、`manifest = 数据记录`、`notebook_results.json = 实验记录`、`resume test = 对照中断前后结果`。现阶段不要求手写完整序列化框架、Loader 回放、日志工具。

---

<a id="review"></a>

## **十三、容易混淆的 15 组概念**

### **1. 最容易混淆的概念对照**

| 概念 A | 概念 B | 本质区别 |
|---|---|---|
| 扩散时间 `t` | Optimizer Step | 前者是图像噪声级别，后者是网络参数更新次数 |
| `beta_t` | `1-alpha_bar_t` | 前者为本步新噪声方差，后者为从 x0 到当前的累计等效噪声方差 |
| `alpha_t` | `alpha_bar_t` | 前者每一步保留上一状态，后者累计保留原图信号 |
| 训练真实噪声 `eps` | 推理预测 `eps_theta` | 前者是已抽样标签，后者是 U-Net 预测输出 |
| `eps_theta` | 反向采样新噪声 `z` | 一个用于估计均值，一个用于实际抽样 |
| `x0_hat` | 真实 x0 | 生成时前者是预测，真实原图并不存在于生成输入中 |
| $q(x_{t-1}\mid x_t,x_0)$ | $q(x_{t-1}\mid x_t)$ | 前者已知原图、可解析后验；后者需对未知原图边缘化 |
| `q(...)` | `p_theta(...)` | 前者规定好的正向过程及相关后验，后者是学习的反向模型 |
| 条件均值 `mu_theta` | 下一状态样本 `x_(t-1)` | 均值是分布中心，采样还需方差和随机数 |
| 方差 `var` | 标准差 `std` | `std=sqrt(var)`，采样写 `mu + std*z` |
| U-Net | Scheduler | 一个训练预测噪声，一个执行固定公式及采样 |
| TimeBlock 残差 Skip | Encoder–Decoder Skip | 内部相加 vs 跨层通道拼接 |
| `model.eval()` | `torch.no_grad()` | 切换模型某些层行为 vs 关闭梯度记录 |
| 固定训练 Probe Loss | 验证 Probe Loss | 同一训练配对拟合 vs 未参与训练图片的有限样本评估 |
| Simple Noise MSE | 完整 VLB / 精确似然 | 简化监督目标 vs 严格变分目标/密度估计 |

<a id="self-check"></a>

## **十四、八道综合自测**

<a id="q81"></a>

### **81. 给定 $\alpha_1=0.9,\ \alpha_2=0.8$，原图信号累计系数与累计噪声方差是多少？**

> $\bar\alpha_2=0.72$；原图 **幅度** 为 $\sqrt{0.72}\approx0.8485$，累计噪声 **方差** `1-0.72=0.28`。

<a id="q82"></a>

### **82. 训练时为什么不能为 `loss()` 重新生成一份噪声？**

> 因为监督目标应是 **实际构造 xt 的同一份 eps**；改成另一份随机 eps 就是标签错位。

<a id="q83"></a>

### **83. `eps_pred` 是从 `x_(t-1)` 到 `xt` 的噪声吗？**

> 否；它预测的是由 `x0` 直接构造 `xt` 的 **累计等效噪声**。

<a id="q84"></a>

### **84. $q(x_{t-1}\mid x_t,x_0)$ 能算，为什么 $q(x_{t-1}\mid x_t)$ 不能直接照搬？**

> 因为后者 **不知道真实原图**，需要综合所有可能原图的条件分布。

<a id="q85"></a>

### **85. 一个高斯后验写为 $\mathcal N(0.7,0.04)$，`z=-1` 时采样值是多少？**

> 标准差 $\sqrt{0.04}=0.2$，故 `0.7+0.2×(-1)=0.5`。

<a id="q86"></a>

### **86. `model.eval()` 后，为什么还要 `torch.no_grad()`？**

> `eval()` 不自动关闭 Autograd；生成只推理，不需要保存反向传播计算图。

<a id="q87"></a>

### **87. 固定过拟合 MSE 从 1.08 降到 0.01，是否证明图像生成准确率 99%？**

> 不证明；只能说明 **同一个固定监督任务的误差明显下降**，需看独立验证及真实生成样本。

<a id="q88"></a>

### **88. 数学上的 $p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}$ 是否对时间积分？**

> 否；它是对 **每个时间索引对应的随机图像状态** 积分，从路径联合分布中边缘化中间变量。

<a id="core-formulas"></a>

## **十五、本阶段最值得独立背出的 6 个公式**

**前向单步：**

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t.$$

**前向累计：**

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \bar\alpha_t=\prod_{s=1}^t\alpha_s.$$

**训练目标：**

$$L_{\rm simple}=\mathbb E\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2.$$

**贝叶斯后验：**

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\!\left(\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}I\right).$$

**预测原图：**

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.$$

**反向一步：**

$$x_{t-1}=\mu_\theta(x_t,t)+\sigma_tz,\quad z\sim\mathcal N(0,I),\quad t>1.$$

最后一步依照实现的边界处理不再增加新噪声。

### **1. 第一阶段完成度与下个主题衔接**

| 内容 | 证据 | 状态 |
|---|---|---|
| DDPM 前向累计加噪概念 | 已问答、写出公式和代码 | 已学习 |
| 贝叶斯后验均值与方差 | 已完成详细拆解、数值例子 | 已学习；仍可未来复盘 |
| Simple Noise MSE、随机时间 | 已讨论并完成训练实验 | 已学习 |
| 时间条件 U-Net | 读懂结构和关键 Shape | 已学习，无需当前整网手写 |
| MNIST 数据及 Batch | 已讨论 Dataset、Loader 与归一化 | 已学习 |
| 固定配对、随机训练与验证探针 | 已有具体损失结果 | 教学实验已完成 |
| EMA 生成和反向过程图 | 已生成 8 张可见数字结构的图片 | 教学实验已完成 |
| 个人 `DDPMScheduler` | S1 验收记录中的个人 DDPM 复验 | 数学核心已验收 |
| 完整 VLB/ELBO 独立手推 | 已讲直觉与关键分解，用户选择暂缓深入 | 不作为本阶段强制完成项 |

**结论**：这一节已经从“完全看不懂 DDPM 概率公式”，推进到“能识别 `q/p`、自行解释代码、完整观察训练与生成、实现核心 Scheduler 公式”。将来进入 **VAE / Latent Diffusion / DiT / Flow Matching** 时，可以把本笔记当作 DDPM 的基础参考，而不必从头重学每段训练工程代码。

---

## **十六、资料与阅读建议**

本文件按知识顺序重排，综合此前保存的 `DDPM_前向扩散八问`、`信噪比与反向后验三问`、`变分界与噪声预测损失问答`、`Scheduler 代码十一问`、`UNet 结构与时间条件十一问`、`噪声/EMA/Generator/反向采样七问`、`Probability Foundations 六问`、`训练生成与 MNIST 数据流`、`随机时间步与固定过拟合`、`实验汇总与精确断点续训`、`Scheduler 个人实现审查` 等。重复问题已合并，严谨性备注已在相关小节保留。

**复习顺序**：先看前向扩散，再看概率与贝叶斯反向后验；复习实现时查看 Scheduler、训练和反向生成；核对实验时区分教学版与个人核心验收。

公式依据：[DDPM 原论文](https://arxiv.org/abs/2006.11239)（§2–3 与算法 1、2）。
