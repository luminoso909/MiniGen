# MiniGen · DDPM 疑问速查：SNR、正反向分布、贝叶斯后验

> **扩充版说明**：保留原问题与核心答案，额外增加直觉、数值例子、代码对照与延伸。推荐先读每问的粗体结论，需要时再展开补充。


## Q1. 什么是信噪比？是 α/β 吗？我们希望它怎样变化？

**答：** 信噪比 SNR = 信号功率 / 噪声功率。前向扩散的累计表达式是

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \bar\alpha_t=\prod_{s=1}^t\alpha_s.$$

若原信号参考方差为 1，则 $\mathrm{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)$，**不是**单步 $\alpha_t/\beta_t$。若原图方差为 $\sigma_x^2$，则为 $\bar\alpha_t\sigma_x^2/(1-\bar\alpha_t)$（逐维理解）。SNR 高 → 图像信号占主导；SNR 低 → 噪声占主导。**前向**希望 SNR 从高降至接近 0，以匹配高斯先验；**反向**则从低 SNR 恢复到高 SNR。

### 数值表：SNR 是比值，不是图片中“噪声像素占比”

采用原图单位方差的参考定义：

| $\bar\alpha_t$ | 信号方差贡献 | 噪声方差贡献 | SNR | 直觉 |
|---|---:|---:|---:|---|
| 0.90 | 0.90 | 0.10 | 9.00 | 图像大体清晰 |
| 0.50 | 0.50 | 0.50 | 1.00 | 两者平均平方强度相当 |
| 0.10 | 0.10 | 0.90 | 0.111 | 噪声占优势 |
| 0.01 | 0.01 | 0.99 | 0.0101 | 几乎全是噪声 |

有时论文用 $\log\mathrm{SNR}_t$，因为 SNR 能跨越好几个数量级；也常使用分贝 $\mathrm{SNR}_{dB}=10\log_{10}\mathrm{SNR}$，例如 SNR=10 对应 10 dB。

**什么时候关注它？** 检查训练是否覆盖清晰、中等噪声、极重噪声，以及终点是否能接近简单高斯先验。它不是“网络训练得好不好”的直接准确率指标。

## Q2. 什么是正向、反向概率迭代？θ 表示什么？

**答：** 正向是人为固定的马尔可夫加噪链：

$$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I),\qquad q(x_{1:T}\mid x_0)=\prod_{t=1}^Tq(x_t\mid x_{t-1}).$$

真正的逆向分布 $q(x_{t-1}\mid x_t)$ 必须对未知原图的可能性求积分，通常无法方便地精确算出。训练时已知 $x_0$，故可以计算后验 $q(x_{t-1}\mid x_t,x_0)$。生成时没有 $x_0$，便用 $p_\theta(x_{t-1}\mid x_t)$ 近似真实逆向过程。**$\theta$ 是共享的噪声预测网络（例如 U-Net）中可学习的权重与偏置**；$t$ 作为输入告诉网络当前噪声水平。

### 用“先抽图还是先抽噪声”理解条件信息

假设训练集里同时有 3 和 8。观察一张强噪声图 $x_t$，你可能不能确定它源于哪张原图，所以 $q(x_{t-1}\mid x_t)$ 需要混合所有可能来源。

若训练中又告诉你 $x_0$ **就是某张具体的 8**，那么 $q(x_{t-1}\mid x_t,x_0)$ 可以依已知加噪规则解析计算。这并不是说生成时也会偷看真实原图！生成时只能依赖 $x_t$ 与训练得到的网络参数 $\theta$。

**对照 Python：** `model.parameters()` 中的可训练 `weight`、`bias` 属于 $\theta$；`scheduler.alpha_bars` 等是 Buffer，不是网络要学习的 $\theta$。`loss.backward()` 更新 U-Net 的梯度，跟反向采样是两件不同的事。

## Q3. 图三究竟在求什么？贝叶斯、配方和结果是什么？

**答：** 在给定 $x_t$ 和真实 $x_0$ 时，求上一状态 $y=x_{t-1}$ 的**后验均值和方差**。利用马尔可夫性和贝叶斯公式：

$$q(y\mid x_t,x_0)\propto q(x_t\mid y)q(y\mid x_0).$$

两个已知高斯分别为：

$$q(x_t\mid y)=\mathcal N(\sqrt{\alpha_t}y,\beta_t I),\quad q(y\mid x_0)=\mathcal N(\sqrt{\bar\alpha_{t-1}}x_0,(1-\bar\alpha_{t-1})I).$$

高斯密度的负对数（去掉与 $y$ 无关的常数）是：

$$\frac{\|x_t-\sqrt{\alpha_t}y\|^2}{2\beta_t}+\frac{\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2}{2(1-\bar\alpha_{t-1})}.$$

展开，收集 $y$：

$$\frac12 A_t\|y\|^2-b_t^\top y+\mathrm{const},$$

$$A_t=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}},\quad b_t=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.$$

**配方的目的：** 识别出高斯分布的均值和方差：

$$\tfrac12 A_t\|y\|^2-b_t^\top y=\tfrac12 A_t\|y-b_t/A_t\|^2+\mathrm{const}.$$

所以后验 $q(y\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I)$，其中

$$\boxed{\tilde\beta_t=\frac1{A_t}=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}}$$

$$\boxed{\tilde\mu_t=\frac{b_t}{A_t}=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t}.$$

**它的作用：** 训练时计算精确的、依赖 $x_0$ 的后验；生成时不能看到 $x_0$，但可以根据预测噪声估计 $\hat x_0$，进而得到反向均值：

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},\quad \boxed{\mu_\theta=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)}.$$

再选定反向方差 $\sigma_t^2 I$，采样 $x_{t-1}=\mu_\theta+\sigma_t z$（$z\sim\mathcal N(0,I)$）。$t=1$ 时给定 $x_0$ 的后验退化为点质量，最终一步不再额外加噪。

### 一句话总记忆

**SNR 衡量原图信息剩多少；$q$ 是已知的正向概率规则；$p_\theta$ 是学习的反向模型；贝叶斯 + 高斯配方提供“反向均值/方差”的解析模板，而噪声预测网络为生成时提供估计。**

### 先看一维数值例子，再回来看配方

取 $\alpha_t=0.8,\beta_t=0.2,\bar\alpha_{t-1}=0.75$，且已知真实 $x_0=1$、当前 $x_t=0.5$。由前向公式：

- 只看原图：上一状态 $y=x_{t-1}$ 服从 $\mathcal N(\sqrt{0.75}\times1,\ 0.25)$，中心约 **0.866**。
- 只看“如何从 y 生成当前 x_t”：以 $y$ 为变量时，似然曲线中心为 $0.5/\sqrt{0.8}\approx0.559$，等效方差为 $0.2/0.8=0.25$。

两个高斯信息的精度（方差倒数）均为 4，于是后验中心是平均数：

$$\tilde\mu_t\approx\frac{0.866+0.559}{2}=0.713,\qquad \tilde\beta_t=\frac{1}{4+4}=0.125.$$

所以后验为 $\mathcal N(0.713,0.125)$。**这不是输出一张确定的上一状态，而是求出上一状态的可能性分布。**

### “配方”做了什么？

把概率密度里关于 $y$ 的部分整理成 $\exp[-\frac12 A_ty^2+b_ty+\mathrm{const}]$，再写成 $\exp[-\frac12 A_t(y-b_t/A_t)^2+\mathrm{const}]$，就能一眼读出均值 $b_t/A_t$、方差 $1/A_t$。本质上只是**把两个高斯约束融合为一个新的高斯**。

**易错：** 第一项 $q(x_t\mid y)$ 原本是“给定 y，随机生成 $x_t$ 的分布”；在贝叶斯公式中固定观察到的 $x_t$ 后，才把它作为 y 的似然函数。不要把方向倒过来误认为 $q(y\mid x_t)$ 已知。
