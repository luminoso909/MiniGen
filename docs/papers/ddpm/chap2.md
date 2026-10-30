这一节非常重要！我们正式进入 DDPM 的数学核心了。🌱

**Section 2：Background** 虽然叫作「背景」，实际上已经给出了 DDPM 的基本概率模型，以及训练目标的理论基础。

结合你之前十道题的表现，我认为你已经比较熟悉前向加噪，但有三个地方需要特别深入：

1. **公式 (3)：为什么训练生成模型需要 ELBO？** 你之前对训练和推理的区别有些混淆。
2. **公式 (5)：为什么负 ELBO 可以分解为几个 KL 散度？** 这是理解 DDPM 损失函数的关键。
3. **公式 (7)：反向后验分布是怎么来的？** 你已经会使用它，但还不熟悉它的数学来源。

我们按照「**完整翻译 → 数学直觉 → 关键推导 → 理解整个过程**」的顺序进行。你已经掌握的内容简要带过，新的概念则认真展开。

# 第一部分：原文翻译


# 2. Background｜背景知识

## 2.1 扩散模型的概率定义

扩散模型 [53] 是一种具有如下形式的**潜变量模型（Latent Variable Model）**：

$$
p_\theta(x_0)
:=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

其中：

$$
x_1,\ldots,x_T
$$

是与真实数据具有相同维度的潜变量，而真实数据满足：

$$
x_0\sim q(x_0)
$$

联合分布：

$$
p_\theta(x_{0:T})
$$

被称为**反向过程（Reverse Process）**。

它被定义为一条具有可学习高斯转移分布的马尔可夫链，并从以下先验分布开始：

$$
p(x_T)=\mathcal N(x_T;0,I)
$$

具体来说：

**公式 (1)：反向生成过程**

$$
\boxed{
p_\theta(x_{0:T})
:=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}
$$

其中每一步的条件概率分布为：

$$
\boxed{
p_\theta(x_{t-1}\mid x_t)
:=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
}
\tag{1}
$$

---

## 2.2 前向扩散过程

扩散模型与其他类型潜变量模型的区别在于：

它的**近似后验分布（Approximate Posterior）**：

$$
q(x_{1:T}\mid x_0)
$$

被称为**前向过程（Forward Process）**或**扩散过程（Diffusion Process）**。

这一过程被固定为一条马尔可夫链，根据预先设定的方差调度序列：

$$
\beta_1,\ldots,\beta_T
$$

逐渐向真实数据中添加高斯噪声。

**公式 (2)：前向扩散过程**

$$
\boxed{
q(x_{1:T}\mid x_0)
:=
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
}
$$

其中：

$$
\boxed{
q(x_t\mid x_{t-1})
:=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right)
}
\tag{2}
$$

---

## 2.3 变分训练目标

模型训练通过优化**负对数似然（Negative Log-Likelihood）的变分上界**进行：

**公式 (3)：变分上界**

$$
\begin{aligned}
\mathbb E[-\log p_\theta(x_0)]
&\leq
\mathbb E_q\left[
-\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
\\[6pt]
&=
\mathbb E_q\left[
-\log p(x_T)
-
\sum_{t\geq1}
\log
\frac{p_\theta(x_{t-1}\mid x_t)}
{q(x_t\mid x_{t-1})}
\right]
\\[6pt]
&=:L
\end{aligned}
\tag{3}
$$

---

## 2.4 前向过程的方差与闭式采样

前向过程中的方差参数：

$$
\beta_t
$$

可以通过重参数化方法（Reparameterization）进行学习 [33]，也可以作为固定的超参数。

反向过程的表达能力，在一定程度上得益于我们将：

$$
p_\theta(x_{t-1}\mid x_t)
$$

设置为高斯条件分布。

这是因为，当：

$$
\beta_t
$$

足够小时，前向过程与反向过程可以具有相同的函数形式 [53]。

前向扩散过程具有一个值得注意的性质：

**我们可以通过闭式表达式（Closed-form Expression），直接在任意时间步采样得到带噪数据。**

定义：

$$
\alpha_t:=1-\beta_t
$$

以及：

$$
\bar\alpha_t:=\prod_{s=1}^{t}\alpha_s
$$

则有：

**公式 (4)：任意时间步的前向扩散分布**

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
}
\tag{4}
$$

---

## 2.5 将训练目标分解为 KL 散度

因此，我们可以通过随机梯度下降（Stochastic Gradient Descent），优化目标函数：

$$
L
$$

中的随机项，从而实现高效训练。

为了进一步降低方差，我们可以将公式 (3) 中的目标函数改写为：

**公式 (5)：变分目标的 KL 分解**

$$
\boxed{
\begin{aligned}
L=\mathbb E_q\Bigg[
&\underbrace{
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\parallel
p(x_T)
\right)
}_{L_T}
\\[6pt]
&+
\sum_{t>1}
\underbrace{
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\parallel
p_\theta(x_{t-1}\mid x_t)
\right)
}_{L_{t-1}}
\\[6pt]
&-
\underbrace{
\log p_\theta(x_0\mid x_1)
}_{-L_0}
\Bigg]
\end{aligned}
}
\tag{5}
$$

有关详细推导，请参见附录 A。

这些损失项的符号将在第 3 节中继续使用。

公式 (5) 使用 **KL 散度（Kullback–Leibler Divergence）**，直接比较模型的反向转移分布：

$$
p_\theta(x_{t-1}\mid x_t)
$$

与前向过程对应的后验分布。

当给定原始数据：

$$
x_0
$$

时，这些前向过程的后验分布是可以直接计算的：

**公式 (6)：前向过程的后验分布**

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N
\left(
x_{t-1};
\tilde\mu_t(x_t,x_0),
\tilde\beta_t I
\right)
}
\tag{6}
$$

其中：

**公式 (7)：后验均值与方差**

$$
\boxed{
\tilde\mu_t(x_t,x_0)
:=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t
}
$$

以及：

$$
\boxed{
\tilde\beta_t
:=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t
}
\tag{7}
$$

因此，公式 (5) 中所有的 KL 散度项，实际上都是**高斯分布之间的比较**。

所以，我们可以采用 Rao–Blackwell 化的方法，利用**闭式解析表达式**直接计算这些 KL 散度，而无须依赖方差较大的蒙特卡罗估计（Monte Carlo Estimates）。


以上对应论文第 2–3 页的 Section 2，保留了公式 (1)–(7) 的编号与原始数学结构。DDPM_Denoising_Diffusion_Probab…

---

# 第二部分：概念解读

## 一、先理解公式 (1)：DDPM 究竟在学习哪个概率分布？

先看这两个式子：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
$$

你之前在 Q7 中有一个疑问：

> 推理是不是需要对所有中间变量进行积分？

现在我们就把这个问题彻底解决。

### 1. 什么叫联合分布？

假设只有三个状态：

$$
x_2\rightarrow x_1\rightarrow x_0
$$

马尔可夫性质允许我们写出：

$$
p_\theta(x_0,x_1,x_2)
=
p(x_2)p_\theta(x_1\mid x_2)p_\theta(x_0\mid x_1)
$$

这描述的是三个随机变量同时取某些值的联合概率密度。

但是我们真正关心的只有最终图像：

$$
x_0
$$

因此，需要将其他变量积分掉：

$$
p_\theta(x_0)
=
\int\int
p_\theta(x_0,x_1,x_2)
\,dx_1dx_2
$$

这叫作**边缘化（Marginalization）**。

你可以这样理解：

假设一张最终图像可以由许多不同的潜变量路径生成。

那么，这张图像的边缘概率密度就需要综合考虑所有可能的中间路径。

这解释了为什么数学上需要积分。

### 2. 为什么生成图像时不需要计算积分？

因为**计算某个结果的概率**与**随机生成一个结果**是两件不同的事情。

例如：

假设：

$$
X\sim\mathcal N(0,1)
$$

你想得到一个随机数，可以直接采样：

```python
x = torch.randn(1)
```

不需要先积分计算整个高斯分布。

DDPM 也是一样。

生成时：

$$
x_T\sim\mathcal N(0,I)
$$

然后：

$$
x_{T-1}\sim p_\theta(x_{T-1}\mid x_T)
$$

接着：

$$
x_{T-2}\sim p_\theta(x_{T-2}\mid x_{T-1})
$$

如此循环，直到：

$$
x_0
$$

**核心答句：计算边缘似然需要对潜变量积分，但从已经定义好的联合生成模型中采样，不需要显式执行这些积分。**

这是你之前 Q7 中最需要澄清的概念。

---

## 二、公式 (1) 中的高斯分布究竟意味着什么？

论文定义：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

注意，这个高斯分布不是说：

> 所有生成图像都服从高斯分布。

而是说：

> 在已知当前状态的情况下，模型将下一步状态的条件分布表示为高斯分布。

这是两个完全不同的概念。

假设一张带噪图像为：

$$
x_t
$$

神经网络根据它预测：

$$
\mu_\theta(x_t,t)
$$

以及可能的协方差参数：

$$
\Sigma_\theta(x_t,t)
$$

那么可以通过：

$$
x_{t-1}
=
\mu_\theta(x_t,t)+\Sigma_\theta(x_t,t)^{1/2}z
$$

生成下一步状态，其中：

$$
z\sim\mathcal N(0,I)
$$

这里的矩阵平方根满足：

$$
\Sigma_\theta^{1/2}(\Sigma_\theta^{1/2})^\top
=
\Sigma_\theta
$$

如果协方差是各向同性的：

$$
\Sigma_\theta=\sigma_t^2I
$$

就简化为：

$$
\boxed{
x_{t-1}
=
\mu_\theta(x_t,t)+\sigma_tz
}
$$

你有没有发现？

这其实就是你在 MiniGen 的 Scheduler 中已经实现过的操作！

### 一个关键区别

论文第 2 节先给出了比较一般的形式：

$$
\mu_\theta,\Sigma_\theta
$$

但第 3.2 节会进一步固定反向方差，将主要学习任务放在反向均值上。

因此，我们现在看到的是**概率模型的一般定义**，不必误以为后面的代码一定需要两个神经网络分别预测均值和方差。

---

# 三、公式 (2)：前向扩散为什么不需要学习？

论文给出：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
\sqrt{1-\beta_t}x_{t-1},
\beta_tI
\right)
$$

你已经很熟悉了。

它等价于：

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon_t
$$

其中：

$$
\epsilon_t\sim\mathcal N(0,I)
$$

这里要理解的是：

**前向过程是我们人为选择的一套随机破坏机制。**

我们知道如何向一张图像添加噪声，不需要神经网络来学习这个过程。

而反向过程：

$$
p_\theta(x_{t-1}\mid x_t)
$$

涉及从信息不完整的状态中生成更清晰的图像。

它才需要学习。

所以：

| | Forward | Reverse |
|---|---|---|
| 概率分布 | $q$ | $p_\theta$ |
| 方向 | 清晰图像 → 噪声 | 噪声 → 清晰图像 |
| 转移方式 | 预先规定 | 由神经网络学习 |
| 是否需要训练 | 原始 DDPM 中不需要 | 需要 |
| 主要用途 | 构造训练数据与目标 | 生成新图像 |

注意，这里的「前向过程不需要学习」指的是原始 DDPM 采用固定噪声日程的实现。论文也提到，理论上可以将前向方差参数设置为可学习。

---

# 四、公式 (4)：为什么能直接跳到任意时间步？

这是你之前 Q3 回答得最好的一道题。

我们再做一次简短推导，把数学直觉固定下来。

定义：

$$
\alpha_t=1-\beta_t
$$

对于第一步：

$$
x_1=\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1
$$

第二步：

$$
x_2=\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2
$$

代入：

$$
\begin{aligned}
x_2
={}&
\sqrt{\alpha_2}
\left(
\sqrt{\alpha_1}x_0+
\sqrt{1-\alpha_1}\epsilon_1
\right)
\\
&+\sqrt{1-\alpha_2}\epsilon_2
\end{aligned}
$$

展开：

$$
x_2=
\sqrt{\alpha_1\alpha_2}x_0
+
\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1
+
\sqrt{1-\alpha_2}\epsilon_2
$$

因为两个噪声相互独立，并且均服从标准高斯分布，所以后面两项之和仍然服从高斯分布。

计算其方差：

$$
\begin{aligned}
\sigma^2
&=\alpha_2(1-\alpha_1)+(1-\alpha_2)\\
&=1-\alpha_1\alpha_2
\end{aligned}
$$

因此，可以使用一个新的标准高斯随机变量表示合并后的噪声：

$$
x_2=
\sqrt{\alpha_1\alpha_2}x_0
+
\sqrt{1-\alpha_1\alpha_2}\epsilon
$$

继续递推：

$$
\boxed{
x_t=
\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon
}
$$

其中：

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s
$$

这就得到公式 (4)。

### 为什么它如此重要？

因为训练时我们不必真的循环执行：

```python
for t in range(1000):
    x = add_noise(x, t)
```

而是可以直接写：

```python
noise = torch.randn_like(x0)

xt = (
    sqrt_alpha_bar_t * x0
    + sqrt_one_minus_alpha_bar_t * noise
)
```

这样一次运算就能获得任意时间步的带噪样本。

但仍然需要强调：

**前向过程可以直接采样任意时间步，不代表原始 DDPM 的反向过程可以直接使用相邻步公式跳过任意时间步。**

你之前的 Q5 就混淆了这两件事。

---

# 五、公式 (3)：为什么需要 ELBO？

现在进入本节的第一个理论难点。

论文写道：

$$
\mathbb E[-\log p_\theta(x_0)]
\leq L
$$

这里真正重要的是：

**为什么我们不直接最小化负对数似然，而要最小化它的一个上界？**

## 5.1 我们真正想优化什么？

假设训练集中有大量真实图像。

我们希望模型赋予这些真实图像更高的概率密度。

因此，理想目标是：

$$
\max_\theta
\mathbb E_{q(x_0)}[\log p_\theta(x_0)]
$$

等价于：

$$
\min_\theta
\mathbb E_{q(x_0)}[-\log p_\theta(x_0)]
$$

其中：

- $q(x_0)$：真实数据分布。
- $p_\theta(x_0)$：模型生成的数据分布。

我们希望：

$$
p_\theta(x_0)\approx q(x_0)
$$

问题是：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

涉及对大量潜变量进行积分。

直接计算通常不可行。

## 5.2 如何构造一个可以优化的目标？

引入已知的前向分布：

$$
q(x_{1:T}\mid x_0)
$$

利用期望的定义，可以写出：

$$
p_\theta(x_0)
=
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
$$

这里使用了一个技巧：在积分中乘除同一个概率密度。

接着取对数：

$$
\log p_\theta(x_0)
=
\log
\mathbb E_q
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
$$

由于对数函数是凹函数，根据 Jensen 不等式：

$$
\log\mathbb E[X]
\geq
\mathbb E[\log X]
$$

因此：

$$
\log p_\theta(x_0)
\geq
\mathbb E_q
\left[
\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
$$

右边就是 ELBO：

$$
\boxed{
\mathrm{ELBO}
=
\mathbb E_q
\left[
\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
}
$$

所以：

$$
\log p_\theta(x_0)\geq\mathrm{ELBO}
$$

等价地：

$$
\boxed{
-\log p_\theta(x_0)\leq-\mathrm{ELBO}
}
$$

这就是论文中的公式 (3)。

### 这里到底发生了什么？

原本我们面对一个难以直接求解的积分。

现在通过一个我们知道如何采样的分布：

$$
q(x_{1:T}\mid x_0)
$$

将目标转化为期望。

然后通过 Jensen 不等式构造一个可以优化的界。

这就是变分推断在这里发挥的作用。

### 为什么优化 ELBO 有意义？

更严格地说，对于给定的真实样本 $x_0$：

$$
\boxed{
-\mathrm{ELBO}
=
-\log p_\theta(x_0)
+
D_{\mathrm{KL}}
\left(
q(x_{1:T}\mid x_0)
\parallel
p_\theta(x_{1:T}\mid x_0)
\right)
}
$$

因为 KL 散度非负：

$$
D_{\mathrm{KL}}(P\parallel Q)\geq0
$$

所以：

$$
-\mathrm{ELBO}\geq-\log p_\theta(x_0)
$$

这也说明，**变分上界与真实负对数似然之间的差距，正是两个潜变量后验分布之间的 KL 散度。**

需要注意，最小化这个上界，并不保证在有限模型容量下恰好得到使真实负对数似然最小的参数。

但它为训练提供了一个有理论依据、可计算的目标。

到这里，你暂时只需要记住：

> **ELBO 的目的不是让生成过程变成积分，而是为难以直接计算的生成模型似然构建一个可优化的训练目标。**

---

# 六、公式 (5)：为什么要把损失拆成三个部分？

这一节是整篇 Background 的第二个重点。

原始目标：

$$
L
=
\mathbb E_q
\left[
-\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
$$

还不够方便。

于是作者将它分解为：

$$
\boxed{
L=L_T+\sum_{t=2}^{T}L_{t-1}+L_0
}
$$

这里将论文中各项的命名展开使用。

我们先理解各部分的作用。

## 6.1 第一部分：终点分布匹配

$$
L_T=
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\parallel p(x_T)
\right)
$$

它比较的是：

**真实图像经过完整扩散后的分布**与**我们预先规定的标准高斯先验**。

我们希望：

$$
q(x_T\mid x_0)
\approx
\mathcal N(0,I)
$$

这样生成时才能方便地从标准高斯噪声开始。

对于固定前向噪声日程的原始 DDPM：

$$
L_T
$$

不依赖神经网络参数 $\theta$。

因此，它在训练反向网络时可以作为常数处理。

这也是后面第 3.1 节会说明的内容。

---

## 6.2 第二部分：中间去噪分布匹配

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\parallel
p_\theta(x_{t-1}\mid x_t)
\right)
$$

这是最重要的部分。

它比较两个分布：

**分布 A：**

$$
q(x_{t-1}\mid x_t,x_0)
$$

给定真实图像 $x_0$ 和带噪图像 $x_t$，计算前向过程对应的真实后验分布。

**分布 B：**

$$
p_\theta(x_{t-1}\mid x_t)
$$

只根据当前带噪图像 $x_t$，由神经网络预测下一步的条件分布。

通过最小化 KL 散度：

$$
\min_\theta
D_{\mathrm{KL}}(A\parallel B)
$$

让模型的反向分布尽量接近训练时可以计算的目标后验分布。

### 为什么这很重要？

因为我们终于把生成模型训练转化成了一个更具体的任务：

**让神经网络预测的反向条件分布，尽可能接近前向扩散过程给出的可计算后验。**

这里请特别注意一个细节：

$$
q(x_{t-1}\mid x_t,x_0)
$$

与：

$$
q(x_{t-1}\mid x_t)
$$

并不是同一个分布。

前者额外知道原始图像，因此在 DDPM 的高斯前向过程下具有解析表达式。

后者不知道原始图像，一般不能简单地直接使用公式 (6) 的单个高斯形式。

这也是为什么论文强调：

> when conditioned on $x_0$

即：**当额外给定原始数据时，后验才具有这里使用的可计算形式。**

---

## 6.3 第三部分：最终图像重建

$$
L_0=
-\log p_\theta(x_0\mid x_1)
$$

这一项描述最后一步：

$$
x_1\rightarrow x_0
$$

模型是否能够合理地重建真实数据。

你可以把它理解为最后一步的重建负对数似然。

但要注意：

**它并不必然等于普通的像素 MSE。**

原论文后面的第 3.3 节会为离散图像数据定义具体的解码概率模型。

所以这里暂时不要直接把：

$$
L_0
$$

理解成代码里的：

```python
F.mse_loss(pred_noise, noise)
```

它们不是同一个定义。

---

## 6.4 公式 (5) 在数学上为什么成立？

这部分稍微深入一点。

我们已经知道：

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
$$

但利用贝叶斯关系，也可以重新分解为：

$$
\boxed{
q(x_{1:T}\mid x_0)
=
q(x_T\mid x_0)
\prod_{t=2}^{T}
q(x_{t-1}\mid x_t,x_0)
}
$$

也就是说，同一条前向马尔可夫链的联合分布，可以在给定 $x_0$ 的条件下，写成从 $x_T$ 开始的反向条件分布乘积。

再代入：

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t)
$$

就可以将原始目标里的对数比值分解为：

$$
\begin{aligned}
L
=
\mathbb E_q\Bigg[
&\log\frac{q(x_T\mid x_0)}{p(x_T)}
\\
&+
\sum_{t=2}^{T}
\log
\frac{
q(x_{t-1}\mid x_t,x_0)
}{
p_\theta(x_{t-1}\mid x_t)
}
\\
&-\log p_\theta(x_0\mid x_1)
\Bigg]
\end{aligned}
$$

由于 KL 散度的定义为：

$$
\boxed{
D_{\mathrm{KL}}(P\parallel Q)
=
\mathbb E_{x\sim P}
\left[
\log\frac{P(x)}{Q(x)}
\right]
}
$$

就能得到论文中的公式 (5)。

**核心答句：公式 (5) 不是重新定义了一个损失，而是利用马尔可夫分解和贝叶斯关系，将公式 (3) 的同一个变分目标精确改写为若干可计算的项。**

---

# 七、公式 (6)–(7)：反向后验分布是怎么推导出来的？

这一部分我建议你认真读。

因为它直接联系到了 MiniGen 中的 `posterior_mean`、`posterior_variance` 等变量。

你之前已经会使用公式 (7)，现在我们要理解它为什么成立。

## 7.1 我们究竟想求什么？

已知：

$$
x_0
$$

以及：

$$
x_t
$$

我们想计算：

$$
q(x_{t-1}\mid x_t,x_0)
$$

即：

**如果知道原始图像和当前带噪图像，那么前一个时间步的状态应该服从怎样的分布？**

根据贝叶斯公式：

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0)
}
$$

现在观察右边的两项。

第一项来自前向转移：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
\sqrt{\alpha_t}x_{t-1},
\beta_tI
\right)
$$

第二项来自公式 (4)：

$$
q(x_{t-1}\mid x_0)
=
\mathcal N
\left(
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right)
$$

所以：

**我们现在面对的是两个关于 $x_{t-1}$ 的高斯密度函数的乘积。**

两个高斯密度函数相乘，再经过归一化，仍然得到高斯分布。

因此：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N
\left(
\tilde\mu_t,
\tilde\beta_tI
\right)
$$

剩下的问题就是计算均值和方差。

## 7.2 先求后验方差

为了方便推导，令：

$$
y=x_{t-1}
$$

忽略与 $y$ 无关的常数项，两个高斯密度相乘后，其对数为：

$$
\begin{aligned}
\log q(y\mid x_t,x_0)
={}&
-\frac{1}{2\beta_t}
\|x_t-\sqrt{\alpha_t}y\|^2
\\
&-
\frac{1}{2(1-\bar\alpha_{t-1})}
\|y-\sqrt{\bar\alpha_{t-1}}x_0\|^2
+C
\end{aligned}
$$

展开两个平方项，只关注 $y$ 的二次项。

第一个平方项提供：

$$
-\frac{\alpha_t}{2\beta_t}\|y\|^2
$$

第二个提供：

$$
-\frac{1}{2(1-\bar\alpha_{t-1})}\|y\|^2
$$

因此，后验分布对应的精度系数为：

$$
\boxed{
\frac{1}{\tilde\beta_t}
=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}}
}
$$

这里的精度（Precision）就是方差的倒数。

通分：

$$
\frac{1}{\tilde\beta_t}
=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}
$$

由于：

$$
\alpha_t+\beta_t=1
$$

以及：

$$
\alpha_t\bar\alpha_{t-1}=\bar\alpha_t
$$

所以：

$$
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
=
1-\bar\alpha_t
$$

于是：

$$
\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t
}
$$

这就是公式 (7) 中的后验方差！

## 7.3 再求后验均值

继续观察刚刚的两个平方项。

关于 $y$ 的一次项合并后，可以得到：

$$
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0
\right)^\top y
$$

根据高斯分布的标准形式，后验均值为：

$$
\tilde\mu_t
=
\tilde\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0
\right)
$$

代入刚刚求出的后验方差：

$$
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t
$$

整理可得：

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t
}
$$

这就是公式 (7) 中的后验均值。

整个推导的关键实际上只有两件事：

1. 利用贝叶斯公式，将后验写成两个高斯密度函数的乘积。
2. 通过展开平方项、配方，求出新高斯分布的均值和方差。

它并不需要神经网络参与。

### 那为什么我们还需要神经网络？

因为这个后验均值需要：

$$
x_0
$$

但是生成阶段，我们只有：

$$
x_t
$$

并不知道真正的：

$$
x_0
$$

所以，不能直接使用这个包含真实 $x_0$ 的公式生成图像。

这就是神经网络出现的原因。

---

# 八、为什么原论文最后选择预测噪声？

这一点实际上属于论文下一节的内容，但我们可以提前建立联系，因为你已经实现过噪声预测。

根据公式 (4)：

$$
x_t=
\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon
$$

移项：

$$
\boxed{
x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}
{\sqrt{\bar\alpha_t}}
}
$$

将这个表达式代入刚刚推导的：

$$
\tilde\mu_t(x_t,x_0)
$$

经过化简：

$$
\boxed{
\tilde\mu_t
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}\epsilon
\right)
}
$$

你会发现：

后验均值现在可以仅通过以下变量计算：

$$
x_t,\quad t,\quad\epsilon
$$

我们已经有：

$$
x_t,\quad t
$$

但不知道噪声：

$$
\epsilon
$$

于是，让神经网络预测：

$$
\epsilon_\theta(x_t,t)
$$

就可以得到模型的反向均值参数：

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
}
$$

最后通过高斯分布采样：

$$
x_{t-1}
=
\mu_\theta(x_t,t)+\sigma_tz
$$

这就是 DDPM 的核心生成过程。

**最重要的直觉是：**

> DDPM 并不是先天就要求神经网络预测噪声。它首先建立反向高斯概率模型，再通过后验均值的数学表达，发现可以把反向均值参数化为噪声预测网络。

预测噪声是一种**参数化选择**。

模型也可以预测：

- 反向均值；
- 原始图像；
- 噪声；
- 其他等价参数化下的量。

不同参数化会影响训练目标、数值性质和实际效果。

而原始 DDPM 发现，噪声预测配合简化训练目标能获得很好的生成质量。

这部分会在第 3.2 节正式展开。

---

# 九、最后一句：Rao–Blackwellization 是什么？

论文最后写道：

> They can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

你现在不需要专门研究 Rao–Blackwell 定理。

只需要理解一个工程层面的直觉。

假设我们需要计算：

$$
D_{\mathrm{KL}}(P\parallel Q)
$$

一种方法是蒙特卡罗估计：

$$
D_{\mathrm{KL}}(P\parallel Q)
\approx
\frac1N\sum_{i=1}^{N}
\log\frac{P(x_i)}{Q(x_i)}
$$

其中：

$$
x_i\sim P
$$

因为使用有限个随机样本，估计结果会存在随机误差。

但如果两个分布都是高斯分布，并且均值、协方差已知，我们就可以直接利用 KL 散度的解析公式。

例如，对于两个具有相同各向同性协方差的高斯分布：

$$
P=\mathcal N(\mu_1,\sigma^2I)
$$

$$
Q=\mathcal N(\mu_2,\sigma^2I)
$$

有：

$$
\boxed{
D_{\mathrm{KL}}(P\parallel Q)
=
\frac{1}{2\sigma^2}
\|\mu_1-\mu_2\|^2
}
$$

这意味着：

**原本涉及分布比较的优化问题，可以转化为均值之间的平方误差。**

如果两者的协方差不同，还需要加入相应的协方差项。

Rao–Blackwell 化的重点是利用可解析的条件期望减少估计方差；在这里，作者利用高斯结构解析地处理了相关 KL 项，避免对中间变量进行高方差的随机估计。

而这个高斯 KL 的性质，也为后面从变分目标推导到噪声预测 MSE 提供了重要基础。

---

# 第三部分：把七个公式串成完整逻辑

我特别希望你不要把论文中的公式 (1)–(7) 当成七个彼此独立的结论。

实际上，它们共同构成了一条非常清晰的数学路线。

### ① 建立生成模型：公式 (1)

我们希望学习：

$$
p_\theta(x_0)
$$

因此设计了一条反向马尔可夫链：

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
$$

**问题：反向条件分布应该怎样学习？**

↓

### ② 建立辅助前向过程：公式 (2)

人为规定一个固定的高斯加噪过程：

$$
q(x_t\mid x_{t-1})
$$

它不需要学习，而且可以直接生成训练数据。

**问题：如何构造训练目标？**

↓

### ③ 使用变分推断：公式 (3)

因为模型的边缘似然难以直接计算，所以引入：

$$
L=-\mathrm{ELBO}
$$

通过最小化变分上界训练模型。

**问题：这个目标怎样才能高效计算？**

↓

### ④ 利用高斯性质：公式 (4)

发现可以直接计算：

$$
q(x_t\mid x_0)
$$

不需要遍历整个前向扩散链。

因此能够高效地随机抽取训练时间步。

↓

### ⑤ 分解训练目标：公式 (5)

将变分目标分解为：

$$
L_T+\sum_{t=2}^{T}L_{t-1}+L_0
$$

其中最主要的可学习部分，是让模型的反向转移分布接近对应的前向后验。

**问题：目标后验分布可以计算吗？**

↓

### ⑥ 推导后验分布：公式 (6)、(7)

根据贝叶斯公式和高斯分布性质，可以得到：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N(\tilde\mu_t,\tilde\beta_tI)
$$

其中均值、方差都有解析表达式。

**问题：生成时没有真实的 $x_0$，怎么办？**

↓

### ⑦ 使用神经网络参数化反向过程

让神经网络预测噪声：

$$
\epsilon_\theta(x_t,t)
$$

再计算：

$$
\mu_\theta(x_t,t)
$$

从而实现：

$$
x_t\rightarrow x_{t-1}
$$

最后逐步生成：

$$
\boxed{
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0
}
$$

至此，整套 DDPM 模型就建立起来了。

---

# 第四部分：你现在应该掌握到什么程度？

考虑到你已经完成 MiniGen 第一周，我不希望你把时间都花在一些目前用不到的严格证明上。

我建议这样划分：

| 内容 | 当前掌握目标 | 优先级 |
|---|---|---|
| 公式 (1) | 理解联合分布、边缘化、生成采样的区别 | 🔴 必须 |
| 公式 (2) | 能解释前向高斯过程及参数 | 🔴 必须 |
| 公式 (3) | 理解为什么需要 ELBO，以及上下界关系 | 🟠 重要 |
| 公式 (4) | 能独立推导，并实现对应代码 | 🔴 必须 |
| 公式 (5) | 理解三个损失部分的含义 | 🟠 重要 |
| 公式 (6) | 理解后验为什么是高斯分布 | 🔴 必须 |
| 公式 (7) | 理解后验均值和方差，能与代码对应 | 🔴 必须 |
| Jensen 不等式推导 | 理解基本思路即可 | 🟡 选学 |
| 完整 ELBO 分解证明 | 暂不要求独立推导 | 🟡 选学 |
| Rao–Blackwell 定理 | 知道作用即可 | ⚪ 了解 |

尤其要注意，**公式 (7) 的推导值得你看懂一次，但不需要现在就达到脱离资料完整手写的熟练度。**

相比之下，你应该能独立写出并解释：

$$
\boxed{
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon
}
$$

以及：

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
}
$$

这是你之后学习 DDIM、Latent Diffusion，以及更现代生成架构的重要基础。

---

## 最后，我希望你真正记住三句话

**第一句话：**

DDPM 的前向过程是人为规定的高斯马尔可夫链，而反向过程是需要神经网络学习的概率生成模型。

**第二句话：**

ELBO 让我们能够在无法直接计算边缘似然的情况下，通过优化可计算的变分目标来训练模型。

**第三句话，也是这一节最重要的一句话：**

**前向高斯过程使真实后验具有解析形式，而 DDPM 通过学习反向过程，使模型能够在不知道原始图像的情况下，从随机噪声逐步生成新图像。**

我认为你真正理解第三句话的时候，公式 (1)–(7) 就不再是七个需要死记硬背的数学表达式，而会成为一个相当自然的整体。

**下一节 Section 3 才会完成关键的一步：解释为什么这些看似复杂的概率推导，最终能够变成一个简洁的噪声预测 MSE 损失。**
