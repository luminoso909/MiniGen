# MiniGen · DDPM 后验：为什么从正比号变成等号？

## 原始问题
> 但是事实是 q 只是正比于这个 Gaussian 分布啊，为什么最后能划等号？

## 一针见血的答案

**正比只确定密度的形状；归一化确定倍数。** 条件概率密度 `q(y | x_t,x_0)` 必须对 `y` 积分为 1。配方后识别出一个高斯核，其归一化常数由高斯积分唯一确定，因此可以把 `∝` 变成高斯密度的 `=`。

## 公式

令 `y = x_{t-1}`。贝叶斯公式给出：

$$
q(y\mid x_t,x_0)=\frac{q(x_t\mid y)q(y\mid x_0)}{q(x_t\mid x_0)}\propto q(x_t\mid y)q(y\mid x_0).
$$

其中分母不依赖未知变量 `y`。将两个高斯密度相乘并对 `y` 配方后得到：

$$
q(y\mid x_t,x_0)\propto\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].
$$

这时**还不能直接把上式右边当作完整的概率密度**。写成：

$$
q(y\mid x_t,x_0)=C\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].
$$

由 $\int q(y\mid x_t,x_0)\,dy=1$，以及高斯积分

$$
\int_{-\infty}^{\infty}\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right]dy=\sqrt{2\pi\sigma^2},
$$

得 $C=1/\sqrt{2\pi\sigma^2}$，于是

$$
q(y\mid x_t,x_0)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right]
=\mathcal N(y;\mu,\sigma^2).
$$

**更短的证明：** 若 $q(y)=k\,\mathcal N(y;\mu,\sigma^2)$，因两边都是归一化的概率密度，积分后 $1=k\cdot1$，所以 $k=1$。

## 数值例子

假设配方得：

$$
q(y)\propto e^{-(y-3)^2}.
$$

它对应的方差是 $\sigma^2=0.5$，均值 $\mu=3$。但 $e^{-(y-3)^2}$ 自身的积分是 $\sqrt\pi$，不是 1。正确的概率密度是：

$$
q(y)=\frac{1}{\sqrt\pi}e^{-(y-3)^2}=\mathcal N(y;3,0.5).
$$

## 补充：贝叶斯分母到底去了哪里？

它并未消失！

$$
q(x_t\mid x_0)=\int q(x_t\mid y)q(y\mid x_0)\,dy
$$

正好是把分子对 `y` 积分后的归一化常数。推导分布形状时先用 `∝` 忽略它；识别出高斯后，再通过归一化恢复完整密度。

## 易错点

- `q \propto f` **不等于** `q=f`，除非已确定恰当的归一化常数。
- `\mathcal N(y;\mu,\sigma^2)` 是一个**已归一化的概率密度**；`e^{-(y-\mu)^2/(2\sigma^2)}` 只是未归一化的**高斯核**。
- 在 DDPM 中，给定 $x_0,x_t$ 的高斯后验公式通常针对 $t\ge2$；$t=1$ 时后验是集中在 $x_0$ 的点质量，不使用普通非退化高斯密度。
