# MiniGen｜DDPM：从变分界到噪声预测损失

> **扩充版说明**：保留原问题与核心答案，额外增加直觉、数值例子、代码对照与延伸。推荐先读每问的粗体结论，需要时再展开补充。


> 复习目标：理解为何从最大似然转向 VLB，以及为什么最终可训练噪声预测 MSE。记号：$x_0$ 是真实图像，$x_T$ 是噪声，$q$ 是固定前向扩散分布，$p_\theta$ 是可学习的反向生成模型。

## Q1. 为什么不能直接最大化 $\log p_\theta(x_0)$？

**答：** DDPM 通过 $T$ 步随机反向链生成图像。同一张 $x_0$ 可对应无数中间轨迹，必须把中间变量积分掉：

$$p_\theta(x_0)=\int p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)\,dx_{1:T}.$$

高维积分通常无法精确高效求解，因此转而优化可计算的变分界。

### 从一维例子理解“积分掉路径”

假设生成系统先抽一个隐变量 $z$，再由 $z$ 生成图片 $x$。你观察到 $x$，却不知道生成它时选了哪个 $z$，就要把所有可能 $z$ 的贡献汇总：离散时是 $p(x)=\sum_z p(x,z)$，连续时就是 $p(x)=\int p(x,z)dz$。

DDPM 把一个隐变量变成了整条随机轨迹 $(x_1,\ldots,x_T)$。真实图像的似然要对**所有可能轨迹**求积分；困难不在于单条轨迹采不出来，而在于把它们的概率贡献准确求和很难。

**区别：** “给定一份噪声能生成一张图”是采样问题；“计算一张指定图在模型下有多大概率密度”是似然计算问题，两者难度不同。

## Q2. 为什么引入 $q$ 和 Jensen 不等式？

**答：** $q(x_{1:T}\mid x_0)$ 是我们已知、容易采样的前向加噪链。利用它把积分改写为期望，再用 $\log$ 的凹性：

$$\log p_\theta(x_0)=\log\mathbb E_q\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]\ge\mathbb E_q\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right].$$

故 $-\log p_\theta(x_0)\le L_{\mathrm{VLB}}$。**最小化 VLB 是优化负对数似然的一个上界，不等于精确计算似然。**

### Jensen 用两个数理解，而不是先背定理

假设某个正随机变量只有两个同概率取值：$1$ 和 $4$。那么

$$\log\mathbb E[X]=\log(2.5)\approx0.916,\qquad \mathbb E[\log X]=\tfrac12(\log1+\log4)\approx0.693.$$

可见 $\log\mathbb E[X]\ge\mathbb E[\log X]$。这里的 1 和 4 **只是演示 Jensen，不是 DDPM 的真实样本权重**。

DDPM 中把 $X$ 换成联合密度比值 $p_\theta(x_{0:T})/q(x_{1:T}\mid x_0)$，就得到**可利用 q 采样估算的下界**。对下界取负号，变为最小化负对数似然的上界。

**术语对照：** `ELBO`（越大越好）$=-L_{\mathrm{VLB}}$；`VLB loss`（越小越好）是相应负号。它们不是额外的两个独立训练过程。

## Q3. VLB 为何分成三项？

**答：** 把前向链按“给定 $x_0$ 的逆序”分解，再代入生成模型分解，得到：

$$\begin{aligned}
L_{\mathrm{VLB}}={}&D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\Vert p(x_T)\bigr)\\
&+\sum_{t=2}^T\mathbb E_{q(x_t\mid x_0)}D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr)\\
&-\mathbb E_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1).
\end{aligned}$$

- **终点项**：让前向链的末端靠近生成时所用的标准高斯先验；固定 $q$ 与先验时不依赖 $\theta$。
- **中间项**：让网络的反向分布贴近可计算的、已知原图时的真实后验；这是主要学习信号。
- **最后一项**：从 $x_1$ 重建 $x_0$ 的似然项；不是零方差高斯 KL。

### 把三项看成“起点—中间—终点”检查

1. **先验匹配项**：我们的前向末端 $q(x_T\mid x_0)$ 是否已接近生成初始先验 $p(x_T)=\mathcal N(0,I)$？噪声日程固定后，这项不更新 U-Net。
2. **中间步 KL**：给定训练原图时，数学上能算出理想后验；模型的反向分布能否逼近它？这是主要训练约束。
3. **重建项**：最后从 $x_1$ 还原到 $x_0$，概率模型在真实原图处给出多大的密度？

**重要区别：** KL 讨论的是**整个概率分布**的差异；我们熟悉的 MSE 讨论数值误差。固定后验/模型方差后，两个高斯的 KL 中与网络权重相关的部分才能简化为均值 MSE，而不是任何分布的 KL 都等于 MSE。

**延伸：** 完整 VLB 还牵涉最后重建项与概率密度（或离散像素概率）；MiniGen 第一周使用的是**简化噪声 MSE**，不应对外声称实现了完整似然评测。

## Q4. 为什么“预测噪声”就决定了反向均值？

**答：** 已知前向式 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$。因此

$$x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}.$$

把它代入已求出的后验均值，可化为

$$\tilde\mu_t=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).$$

训练时噪声 $\epsilon$ 已知；生成时未知，所以让网络给出 $\epsilon_\theta(x_t,t)$ 并代替 $\epsilon$，得到 $\mu_\theta$。**网络预测噪声，采样器据此计算反向高斯的均值；网络不直接输出“上一张图片”。**

### 预测噪声为什么足够给采样器使用？

前向构造中 $x_t$、时间表系数已知，未知只有 $x_0$ 与实现该前向样本所用的 $\epsilon$。若估计出噪声，就可经代数移项得到 $\hat x_0$，再用固定后验系数计算 $\mu_\theta$。

拿 $\bar\alpha_t=0.36, x_t=0.5,\epsilon_\theta=0.2$：

$$\hat x_0=\frac{0.5-\sqrt{0.64}\times0.2}{\sqrt{0.36}}=\frac{0.34}{0.6}\approx0.567.$$

这个值只是**推测的干净像素**，不意味着真的还原了某张给定原图。网络学习的是条件统计预测：相同 $x_t,t$ 可能来自多个 $x_0$。另请记住，$\hat x_0$ 代回后验得到的是**模型反向均值**，不是训练时的精确真实后验。

## Q5. 为什么高斯 KL 最后成为噪声 MSE？

**答：** 中间步骤的两个反向分布都是高斯。当模型方差 $\sigma_t^2I$ 固定时，KL 中依赖 $\theta$ 的部分仅是均值差：

$$D_{\mathrm{KL}}=\frac{\|\tilde\mu_t-\mu_\theta\|_2^2}{2\sigma_t^2}+\mathrm{const}.$$

代入噪声形式：

$$\tilde\mu_t-\mu_\theta=\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\epsilon_\theta-\epsilon).$$

故对应训练项（$t\ge 2$）为

$$\boxed{\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\,\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2+\mathrm{const}.}$$

实务中常进一步去掉时间相关权重，采用 **简化噪声预测损失**：$L_{\mathrm{simple}}=\mathbb E_{x_0,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2$。**它是训练替代目标，不等同于完整 VLB 或精确负对数似然。**

### 从 KL 变成 MSE：看固定 t 的系数

对于两个同维高斯，若模型协方差预先固定为 $\sigma_t^2I$，那么 KL 中依赖 $\theta$ 的项为 $\|\tilde\mu_t-\mu_\theta\|^2/(2\sigma_t^2)$。又因为两个均值之差与 $\epsilon_\theta-\epsilon$ **成正比**，于是每个时间步有一份加权噪声平方误差。

假设某个时间步固定权重 $w_t=2$，一维预测真实噪声 $0.4$、模型预测 $0.1$，该步可训练误差是 $2(0.1-0.4)^2=0.18$；如果用普通简化 MSE，则是 $0.09$。这里 $w_t=2$ **仅为算例假设**，非具体 DDPM 日程的真实计算值。

**训练代码真正执行：** `eps = torch.randn_like(x0)` → `xt = add_noise(x0,eps,t)` → `eps_pred = unet(xt,t)` → `loss = mse(eps_pred,eps)`。

**两个关键限定：** (1) 去掉时间权重是实用替代目标，不能说与完整 VLB 完全等价；(2) 第一反向步的严格 VLB 项是重建似然，不能把退化后验的方差当成普通非零高斯继续套 KL。

## 一句话总复盘

**图片似然难积分 → Jensen 构造可优化的 VLB → 逐步后验 KL → 固定方差时比较高斯均值 → 均值可由噪声表示 → 训练 U-Net 预测噪声。**

**易错：** $q(x_{t-1}\mid x_t,x_0)$ 已知原图；$p_\theta(x_{t-1}\mid x_t)$ 在生成时不知道原图。均值预测是近似的，MSE 最优值对应条件期望噪声，不是每次必能恢复被抽中的真实噪声。$t=1$ 的重建项与 $t\ge2$ 的高斯 KL 应区分。
