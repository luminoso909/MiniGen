# MiniGen · S3 Week 5 · Flow Matching 专题学习与 Q&A

> **资料依据**：`S3_文本条件CFG与可信评测(1).ipynb` 第九节（Flow 预研），以及围绕该节提出的学习问答。\
> **学习范围**：二维高斯 + 独立端点耦合的线性条件路径；不训练图像 Flow，不把它等同于最优传输（OT），也不改动 MiniGen 已确定的 DDPM/DiT 目标。\
> **时间方向**：Flow 示例统一采用 **噪声 t=0 → 数据 t=1**。这与此前 DDPM 前向加噪的方向不同。\
> **导航**：[Q40：S4 条件图像 Flow 架构](#q40) · [总体图](#总图) · [Q37：MSE 的期望与最优解](#q37) · [Q38：直线路径为什么可以求速度](#q38) · [Q39：Euler 的 t、k、N](#q39) · [二维高斯解析基准](#gaussian) · [代码逐段](#code) · [实验结果](#results) · [知识边界](#limits)

---

<a id="总图"></a>
## 先用一张流程图理解 Flow 在做什么

```text
                         训练：学习速度
  数据端点 z_data ~ p_data         噪声端点 ε ~ N(0,I)
                |                        |
                +-----------+------------+
                            |
            随机时间 t ~ Uniform(0,1)   [B,1]
                            |
                            v
         线性插值：x_t = (1-t) ε + t z_data     [B,2]
         每对点的目标速度：u = z_data - ε       [B,2]
                            |
                       +----+----+
                       |         |
                       v         v
              网络输入 (x_t,t)   监督目标 u
                       |         |
                       v         |
                 Flow MLP       |
                       |         |
                       v         |
             预测速度 v_hat     |
                       +----+----+
                            |
                   MSE(v_hat, u)
                            |
                    Backward 更新权重

                         推理：沿速度场积分
  新噪声 x_0 ~ N(0,I)       时间 t_0 = 0
                |                 |
                +-------+---------+
                        |
                        v
              v_θ(x_k,t_k) 预测速度
                        |
                        v
             x_(k+1) = x_k + Δt · v_θ(x_k,t_k)
                        |
               k 从 0 更新到 N-1
                        |
                        v
                   x_N（t=1）
                        |
                   生成数据点
```

**关键是：训练监督的每对端点直线速度 `u`，与推理使用的、网络预测的边际速度 `v_θ(x,t)` 不是同一层概念。** 两者通过 MSE 的条件期望联系起来。

---

<a id="q37"></a>
## Q37. CFM Loss 里的期望是不是 Batch Mean？为什么最优解是条件期望？公式中的 x 又是什么？

### 原始问题

> cfm 的 loss 公式的 E，其实代表了对于每个抽样，对其结果都进行平均是吗？也就是 E 其实表示的是对 batch 的平均。那么图1处表达的含义是什么？x 是什么？这个公式为什么就能表达 mse？

### A. 两层平均要分开

讲义中的总体训练目标：

$$
\boxed{L_{\mathrm{CFM}}(\theta)=\mathbb E_{z,\epsilon,t}\left[\left\|v_\theta(x_t,t)-(z-\epsilon)\right\|_2^2\right]},
\qquad x_t=(1-t)\epsilon+tz.
$$

- $z\sim p_{\mathrm{data}}$：抽取数据端点；$\epsilon\sim\mathcal N(0,I)$：抽取噪声端点；$t\sim U(0,1)$：抽取插值时间。三个随机变量的组合会产生随机训练题目。
- **数学上的 $\mathbb E$**：对数据、噪声、时间所有可能取值的总体期望，不是特指 Batch 操作。
- **实际的 Batch Mean**：每步随机抽一批端点和时间，用有限个样本的平均近似总体期望（Monte Carlo 估计）。

Notebook 的训练代码：

```python
state, velocity = flow_path_reference(data, noise, time_value) # 两者 [256,2]
prediction = flow_net(torch.cat([state, time_value], -1)) # [256,2]
loss = (prediction - velocity).square().mean()  # scalar
```

这里 `prediction-velocity` 是 `[256,2]`：`.mean()` 对 **256 个样本 × 每样本 2 个速度分量**取平均。

$$
\widehat L_{\mathrm{batch}}=\frac1{B\,d}\sum_{i=1}^B\sum_{j=1}^d(\hat v_{ij}-u_{ij})^2,
\quad B=256,\ d=2.
$$

讲义公式按每样本**平方 L2 范数**写，意味着两个分量先相加；代码还除以分量数 $d=2$。只是整体损失乘了一个常数因子，**不会改变其理论最优预测值**。

### B. 图中的 $x$ 是一个空间位置，不是时间、更不是数据端点 $z$

$$
\boxed{v^*(x,t)=\mathbb E[z-\epsilon\mid x_t=x,\,t]}.
$$

- $x_t$：随机端点对在随机时间 $t$ 插值得到的**中间位置**，是一个随机变量；二维时形状 `[B,2]`。
- $x$：当我们把 $x_t$ 的取值固定为某个具体坐标（如 $(1.5,0)$）时，使用的记号。
- $z$：从目标数据分布中抽取的**终点**；不是 $x$，也不是 $t$。
- $v^*(x,t)$：网络只看得到当前坐标与时间时，对目标速度 $z-\epsilon$ 所能做到的最佳 MSE 预测。

可以这样读公式：**“现在已知一个点出现在 $x$，时间为 $t$。在所有可能经过它的端点配对中，真实速度取条件平均。”**

### C. 为什么这是 MSE 的最优解？

固定一个输入 $(x,t)$，令随机目标速度 $U=z-\epsilon$，你可以输出任何候选预测向量 $a$。写 $m=\mathbb E[U\mid x_t=x,t]$，则

$$
\begin{aligned}
\mathbb E[\|a-U\|^2\mid x_t=x,t]
&=\mathbb E[\|a-m+m-U\|^2\mid x_t=x,t]\\
&=\|a-m\|^2+\mathbb E[\|U-m\|^2\mid x_t=x,t],
\end{aligned}
$$

交叉项为零，因为 $\mathbb E[U-m\mid x_t=x,t]=0$。**第二项不取决于模型输出 $a$；第一项只有在 $a=m$ 时最小。** 因此上面的条件期望就是 MSE 最优预测器。

注意：**这条条件期望公式描述的是最优模型的形式，不是另写了一种 MSE 计算方式。** 即使模型达到最优，单个端点配对仍可能有不同速度，最终 Loss 也未必为零（不可约的条件方差）。

---

<a id="q38"></a>
## Q38. 线性组合为什么能表示速度？它和 ODE 怎么对上？式子中的 ε 又去哪里了？

### 原始问题

> 为什么速度明明是一个线性组合，但是却能直接被用作位置变化率？感觉对不上啊，在这个 ode 方程中是表达了什么，难道不是 v*delta_t 表达一个方向吗，但是原来公式还有加上 eps，就不是这个 ode 了啊，这两个公式对不上啊。

### A. 讲义里的线性组合定义的是位置，而不是速度

对于**一对固定端点**：起点 $\epsilon$、终点 $z$，定义

$$
\boxed{x_t=(1-t)\epsilon+tz=\epsilon+t(z-\epsilon)}.
$$

这与中学里的匀速直线运动 $\text{位置}=\text{起点}+\text{时间}\times\text{速度}$ 完全一样：

$$\frac{dx_t}{dt}=z-\epsilon=:u.$$

举例：$\epsilon=(0,0)$、$z=(2,-1)$，就有 $x_t=(2t,-t)$；这条固定直线的速度恒为 $u=(2,-1)$。

**$\epsilon$ 从未消失**：它在微分方程解中作为初始条件 $x(0)=\epsilon$。积分公式就是：

$$x(t)=x(0)+\int_0^t u\,ds=\epsilon+t(z-\epsilon).$$

这里 $u\Delta t$ 是一段**有方向、有长度的位移增量**，而不仅仅是抽象方向。

### B. 但这不等于真正推理时要沿每对端点的原直线走

上述速度 $u=z-\epsilon$ 取决于两端点。训练时知道两端点，可以监督；生成时只有随机初始点 $x_0$，**不知道未来的真实数据端点 $z$**，不能直接计算 `z-epsilon`。

Flow 模型学习的是：

$$v_\theta(x,t)\approx v^*(x,t)=\mathbb E[z-\epsilon\mid x_t=x,t].$$

它在每个空间位置、每个时间给出一个**边际平均速度**。采样解另一个 ODE：

$$\boxed{\frac{dx}{dt}=v_\theta(x,t),\quad x(0)\sim\mathcal N(0,I).}$$

推理时更新规则：$x_{k+1}\approx x_k+\Delta t\,v_\theta(x_k,t_k)$。如果速度随位置或时间变化，实际积分轨迹可以是曲线，不必是之前任意一条端点直线。

### C. 为什么两个速度不能混为一谈？一个极简的交叉例子

看两个**不同**端点配对，都在 $t=0.5$ 到达同一个 $x=1$：

| 配对 | 起点 ε | 终点 z | 中点 $x_{0.5}$ | 单条路径真实速度 $u=z-ε$ |
|---|---:|---:|---:|---:|
| A | 0 | 2 | 1 | +2 |
| B | 2 | 0 | 1 | -2 |

模型如果只看见 $x=1,t=0.5$，无法知道当时来自 A 还是 B。如果这两种配对等可能，MSE 最优速度就是 $(2+(-2))/2=0$。**因此单条直线的速度和汇总到同一个位置后的条件平均速度不一定相同。**

这就是讲义强调“监督随机点对速度，但学到的是边际场”的原因。**它也不是说 Flow 训练解决了全局最优传输（OT）的端点配对。**

---

<a id="q39"></a>
## Q39. Euler 中的 t、k 和 N 是什么关系？第一步 Flow Network 预测什么？z_t 会不会就是 t？

### 原始问题

> euler 更新公式中的 t 和 k 都代表什么？为什么又多出来一个 k？t 是确定的一个数吗？我不太懂 t 到底是怎么确定的，和步数 n 的关系？所以说这个时候神经网络预测的其实是 v 是吗？第一步 t=0 的时候，flow network 得到的是什么？z_t 是什么，z_t 不就应该直接是 t 吗？

### A. 三个符号不是同一个东西

| 符号 | 含义 | 本例中的形状/取值 |
|---|---|---|
| $t$ | 连续时间变量，限定在 $[0,1]$ | 每个样本通常 `[B,1]`，训练时随机抽 |
| $k$ | Euler 循环的整数**索引**（第几次更新） | $k=0,1,\dots,N-1$ |
| $N$ 或 `steps` | 从 t=0 积分到 t=1 的**更新总次数** | 例如 $N=4$、10、20、40 |
| $\Delta t$ | 每次前进的时间间隔 | 固定步长时 $\Delta t=1/N$ |
| $t_k$ | 第 k 次更新时使用的时间 | $t_k=k/N$ |
| $x_k$ | 第 k 次更新前点在空间中的位置 | 二维向量 `[B,2]` |
| $v_\theta(x_k,t_k)$ | 神经网络根据位置+时间**预测的速度** | `[B,2]` |

**训练与采样的时间来源不同：**训练时通常为每对端点独立抽 $t_i\sim U(0,1)$；采样时由数值积分器规定确定的网格 $t_k=k/N$（这里使用左端点 Euler 法）。

### B. 时间是标量，位置是二维向量

$$x_t\in\mathbb R^2,\qquad t\in[0,1].$$

`t=0` **只说明现在位于生成过程的起点，不是说空间坐标等于 0**。实际初始点：

$$x_0=\epsilon\sim\mathcal N(0,I),\quad\text{例如 }x_0=(1.3,-0.7).$$

第 0 步的模型输入是 `(x_0,t_0)`：

```python
# 当前 Notebook 二维场输入：[x_coordinate, y_coordinate, time]
initial = torch.randn(B, 2)     # x_0 [B,2]，不要求等于全零
time0 = torch.zeros(B, 1)       # t_0 = 0，形状 [B,1]
v0 = flow_net(torch.cat([initial, time0], dim=-1))  # [B,2] 预测速度
x1 = initial + (1 / N) * v0
```

**网络输出的是当前速度 $v$，不是时间 t、不是直接的下一位置，也不是某一个固定的训练数据点。**

### C. 完整手算：N=4 的常速度例子

假设位置从 $(0,0)$ 出发、速度恒为 $(2,-1)$，此时 $\Delta t=1/4=0.25$：

| $k$ | 读入的 $t_k=k/4$ | 更新前 $x_k$ | 速度 $v$ | 更新后 $x_{k+1}$ |
|---:|---:|---|---|---|
| 0 | 0.00 | (0, 0) | (2,-1) | (0.5,-0.25) |
| 1 | 0.25 | (0.5,-0.25) | (2,-1) | (1,-0.5) |
| 2 | 0.50 | (1,-0.5) | (2,-1) | (1.5,-0.75) |
| 3 | 0.75 | (1.5,-0.75) | (2,-1) | (2,-1) |

最后输出 $x_4$ 对应 $t_4=1$。循环里不再对 $t_4=1$ 求速度，因为已经走完四次更新。

**但这个例子是教学用“常速度场”**，它恰好与直线路径一致。真实 Flow MLP 的 $v_\theta(x,t)$ 一般随时空位置变化，Euler 并不是每步都复用同一个速度。

### D. 特别看你的二维高斯：第一次预测可能是什么？

当前 Notebook 的解析高斯基准里，数据均值 $m=(2,-1)$，噪声与数据独立。于 $t=0$ 时，$x_0=\epsilon$，于是

$$v^*(x,0)=\mathbb E[z-\epsilon\mid \epsilon=x]=m-x.$$

所以在特殊初始点 $x_0=(0,0)$，最优速度是 $(2,-1)$；在更常见的随机起点 $x_0=(1,-2)$，最优速度是 $(1,1)$。**同样是 t=0，速度也取决于位置 x**。

这恰好说明网络必须同时读入 `[B,2]` 的位置和 `[B,1]` 的时间，不能只把 t 输入模型。

---

<a id="gaussian"></a>
## 旧 Flow 课程内容：二维高斯解析速度场（统一复习）

讲义取：

$$z\sim\mathcal N(m,\sigma^2I),\quad\epsilon\sim\mathcal N(0,I),\quad z\perp\epsilon;\qquad m=(2,-1),\ \sigma=0.5.$$

令 $U=z-\epsilon$、$X=x_t=(1-t)\epsilon+tz$。由独立高斯变量的线性组合：

$$\boxed{\mathbb E[X]=tm,\quad\operatorname{Var}(X)=a_tI,\quad a_t=(1-t)^2+t^2\sigma^2.}$$

再展开交叉协方差：

$$
\begin{aligned}
\operatorname{Cov}(U,X)
&=\operatorname{Cov}(z-\epsilon,(1-t)\epsilon+tz)\\
&=[t\sigma^2-(1-t)]I.
\end{aligned}
$$

对联合高斯变量利用条件均值：

$$
\mathbb E[U\mid X=x]
=\mathbb E[U]+\operatorname{Cov}(U,X)\operatorname{Var}(X)^{-1}(x-\mathbb E[X]).
$$

因 $\mathbb E[U]=m$，得到解析最优速度：

$$\boxed{v^*(x,t)=m+\frac{t\sigma^2-(1-t)}{a_t}(x-tm).}$$

**手算校验**：$t=0.5$、$x=(1.5,0)$、$\sigma^2=0.25$，则 $a_t=0.3125$，比例系数 $[0.125-0.5]/0.3125=-1.2$，$x-tm=(0.5,0.5)$，因此

$$v^*(x,t)=(2,-1)-1.2(0.5,0.5)=\boxed{(1.4,-1.6)}.$$

### 解析 ODE 与终点验证

与此边际速度场相符的**连续精确流**可以写成：

$$x_t=tm+\sqrt{a_t}\,x_0,\quad x_0\sim\mathcal N(0,I).$$

到 $t=1$，有 $a_1=\sigma^2$，因此

$$\boxed{x_1=m+\sigma x_0.}$$

这是二维高斯例子的**解析 ODE 对初始点的映射**，不是说网络把某个训练样本一对一地送到原来随机配的 $z$。同一分布中的点，独立随机配对产生直线训练监督；学到边际速度之后，ODE 再生成自己的轨迹。

---

<a id="code"></a>
## 旧 Flow 课程内容：三段代码与实际训练链路

### 1. 训练样本构造 `flow_path_reference`

```python
def flow_path_reference(data, noise, time):
    return (1 - time) * noise + time * data, data - noise
```

输入 `data:[B,2]`、`noise:[B,2]`、`time:[B,1]`，输出 `state=x_t:[B,2]` 和 `velocity=u:[B,2]`。它是在**随机端点对定义的线性条件路径**上，抽取一个位置与该对的真实速度标签。这里的 `velocity` 不是通过网络预测得到的。

### 2. 神经网络训练 `Flow MLP`

```python
flow_net = nn.Sequential(
    nn.Linear(3, 64), nn.SiLU(),
    nn.Linear(64, 64), nn.SiLU(),
    nn.Linear(64, 2),
)

# 每个训练更新中：
data = torch.randn(256, 2) * 0.5 + torch.tensor([2., -1.])
noise = torch.randn_like(data)
time = torch.rand(256, 1)
state, target_velocity = flow_path_reference(data, noise, time)
pred_velocity = flow_net(torch.cat([state, time], dim=-1))
loss = (pred_velocity - target_velocity).square().mean()
loss.backward()   # 真正代码还有 zero_grad / optimizer.step
```

为什么输入维度是 3？二维空间坐标 $(x,y)$ 加一个时间 $t$；输出两个数是二维速度 $(v_x,v_y)$。**CFM 训练时不需要把每对样本积分到 t=1**，只在随机抽取的时间监督当前速度。

### 3. 采样器 `euler_reference`

```python
def euler_reference(field, initial, steps):
    state = initial.clone()
    trajectory = [state.clone()]
    for k in range(steps):
        time_value = torch.full((len(state), 1), k / steps, dtype=state.dtype)
        state = state + field(state, time_value) / steps
        trajectory.append(state.clone())
    return state, torch.stack(trajectory)
```

先使用人工的 `gaussian_velocity_reference` 作为 `field`，验证数值积分器；再换成训练后的 `flow_net`，考察模型能否生成目标分布。如果 `initial:[512,2]`、`steps=40`，则 `trajectory:[41,512,2]`。

---

<a id="results"></a>
## 旧 Flow 课程内容：实验结果怎样读？

### 1. 解析速度场的 Euler 步数实验（测试求解器误差）

| Euler steps | 与解析终点 $x_1=m+\sigma x_0$ 的 RMSE |
|---:|---:|
| 10 | 0.0682769 |
| 20 | 0.0352668 |
| 40 | 0.0179420 |

这说明在当前解析速度场中，Euler 数值积分更细时误差下降。它**不是**神经网络训练的 MSE，也不是完整图像生成质量。

### 2. 600 次更新的小 MLP（测试能否学习边际速度）

| 测量项 | Notebook 结果 |
|---|---:|
| 前 50 次平均 Velocity MSE | 1.76954 |
| 后 50 次平均 Velocity MSE | 0.85651 |
| 学习场生成终点均值 | `[2.03034, -1.04448]` |
| 学习场生成终点方差 | `[0.24108, 0.24352]` |
| 目标分布均值 | `[2.0, -1.0]` |
| 目标分布方差 | `[0.25, 0.25]` |

解读：Loss 下降，学习速度场做 40 步 Euler 后，生成的二维点云的均值/方差接近目标高斯。**不等于已经训练好图像 Flow，或证明任何复杂分布的质量。** 损失不会自动降到 0，因为同一个中间状态可能来自多种不同端点对，目标速度存在条件方差。

### 3. 结果图怎么读？

- **左图 `conditional velocity MSE`**：训练中 MLP 的速度预测误差，初期下降较快，后面约 0.8–1.0 波动；单次随机抽样可引起波动。
- **右图 `analytic endpoint` vs `learned Euler`**：蓝点为解析流终点，橙点为训练 MLP 经 Euler 积分后的终点；两组点云接近可以为生成分布提供初步证据。
- **右图额外曲线**：前 8 个起点沿学到的速度场经过的 Euler 轨迹；可能弯曲，不要求与训练用的端点直线相同。

---

<a id="limits"></a>
## 知识边界、最容易混淆的 9 件事

| 容易混淆 | 正确理解 |
|---|---|
| $\mathbb E$ 就是 Batch Mean | 理论期望对端点和时间的总体分布；Batch Mean 是其蒙特卡洛近似 |
| 公式中的 $x$ 等于 $t$ | $x$ 是空间位置（二维向量），$t$ 是标量时间 |
| $x_t$ 一定是时间 t | 下标 t 表示某时间的**位置**，不是 x 的值等于 t |
| $z-\epsilon$ 是位置的线性组合 | 它是两端点的**位移向量**，同时是线性插值路径关于时间的导数 |
| ODE 的每条轨迹都等于训练时的端点直线 | ODE 跟随边际速度场；训练的随机点对直线是条件监督路径 |
| Euler 的 k 是第二种时间 | k 是整数循环编号，$t_k=k/N$ 是其对应的实际时间点 |
| t=0 意味着位置是零向量 | t=0 只表示起点时刻；初始位置仍是随机噪声 $x_0=\epsilon$ |
| 最优 MSE 必定接近零 | 目标速度在固定输入下可能仍有条件方差，最优残差不一定为零 |
| 二维高斯实验等于图像 Flow 已经成功 | 只验证二维速度监督、数值积分和点云统计，不涉及高维图像训练与生成 |

### 学会这一节，至少应该能自己解释这段最小流程

```text
训练：抽 data z 和 noise ε → 抽 t → 构造 x_t 与 u=z−ε
       → 用 (x_t,t) 预测 v → MSE(v,u) → 更新网络

采样：从 x_0=ε、t_0=0 开始 → 预测 v(x_k,t_k)
       → Euler: x_(k+1)=x_k+(1/N)v(x_k,k/N)
       → N 次后到达 t=1，得到生成点
```

## 待复核 / 后续学习

- 在不改变本周教学结论的前提下，进一步学习 `conditional vs marginal probability path`、连续性方程与速度场怎样保证分布随时间变化。
- 比较普通独立耦合与 OT 耦合的差异，**本轮没有求解全局最优传输**。
- 理解 CFM、Rectified Flow 和 MeanFlow 的区别；不要因为名字都含 Flow 就认为训练目标或单步生成机制相同。
- 正式图像 Flow 留待 MiniGen 下一阶段；这周先把二维位置、时间、速度与积分的关系完全理解。

---

<a id="q40"></a>
## Q40. S3 Week 5 完结：完整 Conditional DiT + Linear CFM 架构，以及 S4 ε-DDPM 对照方案

### 原始要求

> 把目前的整个架构都放进去 Q&A 中，整理输出；下一阶段 S4 要在相同 DiT、数据、训练预算下比较 ε-DDPM 与 Linear CFM。这一节到此结束。

### 结论先行

- **Conditional DiT 是完整预测网络**，不是在它外面再串联一套 `Patchify → DiT → Cross-Attention`。图像 latent 进入其 Patchify/DiT 主干；冻结 CLIP 先在外部编码 Caption，再把文本 token 与有效位 mask 交给模型各层 Cross-Attention；时间输入经过 Time Embedding，调制原 DiT Block 的 Self-Attention 与 MLP。
- **改变的是训练路径、目标与采样器，不是必须更换 Transformer 主干。** ε-DDPM 预测噪声并用 DDIM 采样；Linear CFM 预测速度 $v$ 并用 Euler 求解 ODE。两种模型的网络结构可以相同，但实际需要**分别训练独立参数**，不能直接复用已训练噪声模型权重当速度模型。
- 以下 CFM 的文本条件架构是**S4 拟比较的方案**；本周实际完成的 Flow 实验仅是二维高斯 MLP 教学。**尚未完成图像 CFM 的正式训练与评测。**

### A. S4 拟用的 Conditional DiT：CFM 完整训练流程（纯文本）

```text
                         输入配对 (Image x, Caption y)
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
       [冻结 VAE Encoder]                        [CLIP Tokenizer]
       RGB x [B,3,H,W]                                  |
              |                                         v
              v                                 [冻结 CLIP Text Transformer]
  VAE posterior mean + scaling                          |
              |                                         v
  数据 latent z_data [B,4,h,w]                     C_y [B,L,512]
              |                                    M_y [B,L]
              |                                         |
              |                      空字符串 "" → 同一 CLIP → C_empty,M_empty
              |                                         |
              |                                         v
              |                           训练时 Condition Dropout
              |                           对每个样本选 C_y 或 C_empty，
              |                           并同步替换有效位 mask
              |                                         |
              |                                         v
              |                                C_selected [B,L,512]
              |                                M_selected [B,L]
              |                                         |
              +----------------------------+------------+
              |                            |            |
              v                            |            |
       采样 eps ~ N(0,I)                    |            |
        [B,4,h,w]                          |            |
              |                            |            |
              v                            |            |
       采样 t ~ U[0,1] [B]                 |            |
              |                            |            |
              v                            |            |
   构造 x_t=(1-t)*eps + t*z_data           |            |
       x_t [B,4,h,w]                       |            |
              |                            |            |
              v                            |            |
   +---------- CONDITIONAL DiT -----------+------------+-----------+
   |                                                              |
   |  图像 x_t ──► Patchify(p=2) ──► Linear + Position Embedding   |
   |                     |                                        |
   |                     v                                        |
   |              Image Tokens X_0 [B,N,D]                       |
   |                                                              |
   |  t ──► 连续时间编码 / Time MLP ──► T [B,D]                    |
   |                     |                                        |
   |                     +─────────────────┐                      |
   |                                       v                      |
   |              [第 0 个原始 DiT Block] ◄── T                   |
   |              - adaLN-Zero(shift/scale/gate)：Self-Attn       |
   |              - Residual                                      |
   |              - adaLN-Zero(shift/scale/gate)：MLP             |
   |              - Residual                                     |
   |                     |                                        |
   |                     v                                        |
   |                   H_0 [B,N,D]                                |
   |                     |                                        |
   |                     v                                        |
   |              LayerNorm(H_0) ──► W_Q ──► Q_0                  |
   |                                                    C_selected ──► W_KV ──► K_0,V_0
   |                                                              |
   |                       Cross-Attention(Q_0,K_0,V_0,M_selected) |
   |                     |                                        |
   |                     v                                        |
   |               W_O ──► O_0 [B,N,D]                           |
   |                     |                                        |
   |                     v                                        |
   |               X_1 = H_0 + g_0*O_0     (g_0 初始为 0)         |
   |                     |                                        |
   |                     v                                        |
   |          [第 1 个原始 DiT Block] ◄── 同一份 T                |
   |                     |                                        |
   |                     v                                        |
   |           [第 1 个 Cross-Attention] ◄── 同一份 C_selected/M |
   |                (但本层有自己 W_Q/W_KV/W_O 参数)               |
   |                     |                                        |
   |                     v                                        |
   |                 X_2 [B,N,D]                                  |
   |                     |                                        |
   |                 ... depth 层 ...                             |
   |                     |                                        |
   |                     v                                        |
   |      Final Norm + 时间调制 + Output Linear + Unpatchify     |
   |                     |                                        |
   +---------------------+----------------------------------------+
                         |
                         v
                   v_hat [B,4,h,w]
                         |
             +-----------+---------------------+
             |                                 |
             v                                 v
       目标速度 u = z_data - eps           模型预测 v_hat
             |                                 |
             +----------------+----------------+
                              |
                              v
                 MSE(v_hat, u) → Backward
                              |
                              v
                只更新 Conditional DiT；VAE/CLIP 冻结
```

**上图层级注意：** `Conditional DiT` 外部先准备 C/M，**图像 Patchify、Self-Attention、Cross-Attention、输出头都在 Conditional DiT 内部**。图像 patch 数 $N=(h/p)(w/p)$；教学例子 $h=w=16,p=2$ 时 $N=64,D=128$，因此输出仍是 `[B,4,16,16]`。如未来正式 256×256 图像对应 32×32 latent（假设 VAE 8 倍下采样），则 $p=2$ 时图像 token 数为 256，而不是继续固定为 64。

**模型中的两个时间约定不同：** S3 ε-DDPM 的离散 $t$ 通常按“数据 → 噪声”增加；此处 Flow 的连续 $t\in[0,1]$ 按“噪声 0 → 数据 1”增加。不能原封不动沿用 DDPM 的噪声时间表与采样器。

### B. CFM 生成流程（纯文本）

```text
用户 Prompt ──► Frozen CLIP ──► C_y, M_y
                               |
如使用 CFG ──► CLIP("") ──► C_empty,M_empty
                               |
随机 x_0 ~ N(0,I) [B,4,h,w]，设 t_0=0，N_step 为采样更新次数
                               |
                               v
       对 k=0,1,...,N_step-1：t_k=k/N_step, Δt=1/N_step
                               |
          +--------------------+---------------------+
          |                                          |
          v                                          v
  Conditional DiT(x_k,t_k,C_y)             同一 DiT(x_k,t_k,C_empty)
          |                                          |
          v                                          v
   预测速度 v_c                             预测速度 v_u
          |                                          |
          +--------------------+---------------------+
                               |
                               v
          若使用 CFG：v_guided=v_u+s*(v_c-v_u)
          不用 CFG：直接使用 v_c（s=1）
                               |
                               v
             Euler：x_(k+1)=x_k+Δt*v_guided
                               |
                               v
                     下一时间 t_(k+1)
                               |
                         重复直到 t=1
                               |
                               v
                    生成 latent x_1
                               |
                               v
             反缩放到 VAE 原生 latent 数值尺度
                               |
                               v
                Frozen VAE Decoder
                               |
                               v
                    RGB 图像 → PNG
```

**约束**：上图的速度版 CFG 是概念上的可选采样设计，前提是 Flow 训练时使用了对应 Condition Dropout，并验证两路速度的语义与效果；目前 MiniGen 尚未做图像 CFM，不能把这条图当作实测结果。Euler 对速度积分，绝不能把 DDPM 模型的噪声预测直接当作速度输入。

### C. S4 两套配方的受控比较（规划表）

| 比较维度 | 方案 A：ε-DDPM | 方案 B：Linear CFM | 控制要求 |
|---|---|---|---|
| 网络骨架 | Conditional DiT | 同架构 Conditional DiT | 固定层数、宽度、图文接口与参数规模；**分别训练权重** |
| 图像/文本表示 | 同一 VAE / CLIP 缓存 | 同一 VAE / CLIP 缓存 | 相同 `sample_id`、Caption、数据划分及编码版本 |
| 训练状态构造 | $z_t=\sqrt{\bar\alpha_t}z+\sqrt{1-\bar\alpha_t}\epsilon$ | $x_t=(1-t)\epsilon+tz$ | 时间方向、路径本来就不同 |
| 监督量 | 噪声 $\epsilon$ | 速度 $z-\epsilon$ | 不直接比较两种 MSE 的数值大小 |
| 训练 Loss | Noise MSE | Velocity MSE | 固定优化器、有效 Batch、学习率协议和 EMA，记录预算 |
| 采样方法 | DDIM | Euler | 不交换模型输出与采样器 |
| 每图推理预算 | NFE=10/20/40 | NFE=10/20/40 | **按分支等价 NFE** 对齐，不只对齐更新次数 |
| 随机性 | Seeds 0、1 | Seeds 0、1 | 各独立训练，固定验证请求与采样初始噪声 |
| 结果 | 生成质量、Prompt 遵循、时延与吞吐 | 同指标 | 不只看训练 Loss，也看属性成功率、失败率及效率 |

**NFE 与 CFG：**若某方法在一个更新步使用条件、无条件两路预测，则 $\mathrm{NFE}=2\times$ 更新次数。例如 CFG $s>1$ 时，NFE=10/20/40 可对应 5/10/20 次双路更新；若 $s=1$，通常一次更新只需一份预测。报告时还应给出更新次数、每次分支数、Batch Size 和延迟；把两路条件合并成 `2B` Batch 只是减少 Python 调用，并不会免费消除算力成本。

**训练预算**：讲义提出用相同数据、相同网络与优化设置比较；先做约 200 更新预估耗时，正式实验控制更新数与计算预算（例如记录 GPU 时长）两种指标。不因为算法不同就人为调整评测标准，也不将 DDPM 与 CFM 的训练 Loss 直接作为胜负依据。

### D. Week 5 的完成边界与 S4 待办

| 项目 | 本节状态 | 不能推断什么 |
|---|---|---|
| S3 教学 Notebook（CLIP、Cross-Attention、CFG、DDIM、二维 Flow Toy） | 已运行教学参考链路 | 不等于个人 `src` 的核心实现全部验收 |
| 两个教学 Seed 的条件生成与控制实验 | 有运行结果，但生成质量仍弱 | 不能宣布达到可靠文生图 |
| Flow Matching | **二维高斯 MLP 预研已做** | **图像 Flow 尚未训练** |
| `personal_DiT_CFG_Flow` 和评分器人工验收 | 仍待验收（见 Notebook 状态） | 不应写 S3 全部项目已经完成 |
| S4 DDPM vs Linear CFM 对照 | **已规划，尚未执行** | 不应编造性能差异或云端训练效果 |

### E. 后续深入（保留待办）

1. **Score → CFG 的数学深入推导**：仍按原 Q20/Q23/Q24 的待学标记，未来重新学习，不作为当前理解验收前置项。
2. **Flow 原理继续学习**：边际速度场与条件速度监督、连续性方程、Euler 离散误差、时间条件如何为连续 $t$ 编码，以及速度版本 CFG 的有效性。
3. **S4 真正执行前**：优先解决此前基础几何对象生成失真、VAE/Latent 尺度与个人 DiT/CFG 代码验收，再冻结对照配方。

---
