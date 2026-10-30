# MiniGen｜DDPM：噪声、EMA、Generator、反向采样与概率分布七问

> 复习目的：区分真实/累计噪声、U-Net 预测、反向高斯采样、EMA 和随机数生成器；明确原图与带噪图像的均值/方差。主要从 `TinyUNet + TeachingDDPMScheduler` 的代码理解，不依赖 ELBO 推导。
> 本文 Q1–Q4 为原有笔记，Q5–Q7 是后续补充；所有题目保留在同一文件中。

## Q1. ε 是随机的，εθ 是确定的吗？`step()` 输出就一定确定吗？

**结论：** `eps = torch.randn_like(x0)` 每次调用随机抽一份，但抽出之后 Tensor 数值就固定，可同时作为 `add_noise` 的输入和 MSE 的标签。对固定的 `(xt,t)`、固定 U-Net 参数，在无随机层干扰时，`eps_pred = model(xt,t)` 是确定的。**`scheduler.step()` 对 `timestep > 0` 还要抽新的随机数 `z`，所以输出通常仍然随机。**最后一步 `timestep == 0` 则直接返回均值。

| 量 | 谁产生 | 固定输入后是否确定 | 作用 |
| --- | --- | --- | --- |
| `eps`（训练） | `torch.randn_like(x0)` | 抽出后固定 | 累计加噪的监督标签 |
| `eps_pred` / $\epsilon_\theta$ | `TinyUNet(xt,t)` | 固定模型/输入时确定 | 估计累计等效噪声 |
| `z` / `noise`（生成 `step`） | 新调用的高斯采样 | 抽出后固定 | 从反向高斯分布中抽取前一状态 |
| `sample`（生成） | 初始采样及每次 `step` 更新 | 当前 Tensor 值确定 | 当前链状态 $x_t$ |

**易错：** ε 不是“每时每刻都在变化”的数；它是每次构造训练样本时新抽的一份。训练输出的 εθ 也**不保证**本身服从 $\mathcal N(0,I)$。

**扩展：** 同一初始 $x_T$，若每一步反向随机数不同，可能生成不同结果；若固定初始噪声、反向随机数序列、模型与设备等条件，结果通常可复现。`eval()` 不等于 `no_grad()`。

## Q2. EMA 是什么？`checkpoint['ema']` 保存的是什么？

EMA = **Exponential Moving Average（指数移动平均）**，平滑的是**模型的可训练权重**，不是图像状态，不是新的架构，也不需要另开一个优化器训练。

$$\theta_{\rm EMA}\leftarrow d\,\theta_{\rm EMA}+(1-d)\theta_{\rm current}.$$

**数值例子：** 设 $d=0.9$，EMA 初值 1.0；普通模型参数更新到 1.4，则 EMA 为 $0.9\times1.0+0.1\times1.4=1.04$；下一步普通参数变为 0.9，EMA 变为 $0.9\times1.04+0.1\times0.9=1.026$。EMA 波动较缓。

```python
# 示意；实际框架还要正确处理初始化与 buffer
optimizer.step()                    # 更新普通 U-Net 参数
for ema_p, p in zip(ema_model.parameters(), model.parameters()):
    ema_p.lerp_(p, 1 - decay)       # 按比例更新 EMA
```

```python
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
sampling_model = TinyUNet(base=32).to(DEVICE)
sampling_model.load_state_dict(checkpoint['ema'])
```

`checkpoint['ema']` 是 EMA 版本的模型参数/状态字典。加载后，`sampling_model` 仍是同样的 U-Net，只是权重值换成历史平滑版本。**EMA 常用于改善采样稳定性，但不保证 EMA 验证损失每次都更低。**不要和“绘制训练 Loss 的 20-step 均线”混为一谈。

## Q3. `torch.Generator()` 到底做什么？

它是 **PyTorch 随机数生成器（RNG）及其内部状态**，负责生成可复现的**伪随机数序列**，不是图像生成模型。`manual_seed(s)` 设置该序列的起始状态。

```python
g1 = torch.Generator().manual_seed(41)
a = torch.randn((3,), generator=g1)
b = torch.randn((3,), generator=g1)  # 同一个生成器继续向前走，通常和 a 不同

g2 = torch.Generator().manual_seed(41)
c = torch.randn((3,), generator=g2)
assert torch.equal(a, c)             # 相同种子、相同调用方式 -> 相同初始序列
```

**种子不是生成出的随机数本身**；它是决定随机数序列的起点。`generator=g` 使某次 `randn`/`randint` 使用指定 RNG，而不是依赖全局 RNG。

你当前的采样代码：

```python
initial_noise = torch.randn(
    (8, 1, 28, 28), generator=torch.Generator().manual_seed(41)
).to(DEVICE)
sampling_result = sample_ddpm(..., initial_noise, seed=42, ...)
```

这里的 **41** 控制*初始八张高斯噪声图*；传给 `sample_ddpm` 的 **42** 用于控制其内部随机行为（在该实现中主要用于每次 `step` 的新噪声，具体看函数源码）。两个种子没有必要相同。上面的 `torch.randn` 先在 CPU 上采样再 `.to(DEVICE)`。

## Q4. `eps_pred` 预测的是哪份噪声？为何 `step()` 又需 `mean + std*z`？

**重点纠错：** `eps_pred` 是模型从当前 $x_t$ 和时间 $t$ 估计的**从干净 $x_0$ 到当前 $x_t$ 的累计等效噪声**：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

它**不是**某一次 $x_{t-1}\to x_t$ 的新增噪声 $\epsilon_t$，也不是 `step()` 新抽的 $z$。

一次反向采样的完整步骤：

1. 当前持有一份具体的 `sample = xt`，并知道时间索引；网络 `eps_pred = model(xt,t)`。
2. 用预测噪声**估计**原图：$\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$；它不保证等于某张真实图。
3. 用固定后验系数计算模型的反向均值与方差：$\mu_\theta=c_0(t)\hat x_0+c_t(t)x_t$，$\sigma_t^2=\tilde\beta_t$（当前教学版）。
4. **从模型给出的反向分布抽取** $x_{t-1}=\mu_\theta+\sqrt{\sigma_t^2}\,z$，其中 $z\sim\mathcal N(0,I)$ 是一份**新噪声**；最后一步不额外加噪。
5. 把结果赋给 `sample`，时间减 1，继续重复；**生成不需要已知真实原图、真实 ε 或训练标签**。

为何要额外随机：均值只表示分布中心，并非唯一可能的上一步。若 $X\sim\mathcal N(\mu,\sigma^2)$，可用 $X=\mu+\sigma z$，$z\sim\mathcal N(0,1)$ 采样。**代码中乘的是标准差 `variance.sqrt()`，不是方差本身！**新增的随机性是在模拟反向条件分布，并非“不小心再次污染图片”。

### 贯穿数值例子（一个像素）

取某时间步 $\alpha_t=0.8$、$\beta_t=0.2$、$\bar\alpha_{t-1}=0.8$、$\bar\alpha_t=0.64$，且真实 $x_0=1$、本次累计 ε 为 $-0.5$。

**前向：** $x_t=0.8\times1+0.6\times(-0.5)=0.5$。

**若模型预测** $\epsilon_\theta=-0.3$（不准确），则

$$\hat x_0=\frac{0.5-0.6(-0.3)}{0.8}=0.85.$$

后验系数 $c_0=c_t=\sqrt{0.8}\times0.2/0.36\approx0.4969$；方差 $\tilde\beta_t=0.2\times0.2/0.36=1/9$。

因此模型计算的**反向均值**：

$$\mu_\theta\approx0.4969(0.85+0.5)=0.6708;\qquad \sigma_t=1/3.$$

在**相同**的预测均值下，若这次抽中 $z=-1$，则 $x_{t-1}\approx0.3375$；若抽中 $z=1$，则 $x_{t-1}\approx1.0042$。可见预测均值确定，并不意味着采样结果唯一。

如果网络恰好预测真实累计噪声 $\epsilon=-0.5$，就有 $\hat x_0=1$，对应的**真实条件后验均值**约 $0.7454$，但由于后验方差仍为 $1/9$，**仍不能得知某一次前向轨迹实际抽到的 $x_{t-1}$**。

### 最小代码：生成与训练一眼对照

```python
# 训练：已知 x0 和 eps，更新参数
x0, _ = next(iter(train_loader))
t = torch.randint(T, (len(x0),))
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)
loss = torch.nn.functional.mse_loss(eps_pred, eps)
optimizer.zero_grad(); loss.backward(); optimizer.step()

# 生成：没有 x0/eps/labels；固定已训练的模型
model.eval()
with torch.no_grad():
    sample = torch.randn(B, C, H, W)  # 初始高斯噪声
    for k in reversed(range(T)):
        times = torch.full((B,), k, dtype=torch.long)
        eps_pred = model(sample, times)
        z = torch.randn_like(sample)   # 用于反向条件分布采样的全新随机数
        sample = scheduler.step(eps_pred, k, sample, z)
    generated = sample
```

> 以上为 CPU 简化示意，若使用 MPS/CUDA，须同步处理模型、Scheduler、图片和时间索引的设备。

**最终记忆：** `eps` 是训练时自己加入的累计噪声；`eps_pred` 是 U-Net 从当前带噪图预测该累计噪声；`z` 是从反向高斯采样时额外抽的随机数；EMA 是对 U-Net 权重做平滑，Generator 是控制随机数序列的工具。


---

## Q5. EMA 是 `nn.Module` 自动注册和更新的吗？为什么 `TinyUNet` 代码里看不到？

**答案：不是。** `nn.Module` 负责管理 `Parameter`、`Buffer` 和子模块，**不自动生成 EMA 权重，也不会在 `optimizer.step()` 后自动做滑动平均**。想用 EMA，必须由训练器显式建立“影子权重”并在每次（或约定的间隔）优化器更新之后执行 EMA 更新。

$$\theta_{\mathrm{EMA}}^{(n)}=d\,\theta_{\mathrm{EMA}}^{(n-1)}+(1-d)\,\theta^{(n)},\qquad 0<d<1.$$

**数值例子：** 一维参数，初始 EMA=2.0，衰减 $d=0.99$，优化后的参数=5.0，则 EMA=$0.99\times2+0.01\times5=2.03$。EMA 只是缓慢追随普通权重。

**一种教学实现（不是断言你项目的源码这么写）：**

```python
import copy
model = TinyUNet(base=32)
ema_model = copy.deepcopy(model)   # 初始影子权重

# 训练循环里
loss.backward()
optimizer.step()                  # 先更新普通模型
with torch.no_grad():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 0.01)       # 0.99 * ema_p + 0.01 * p
```

也可以用独立参数字典或专门 EMA 辅助类，而不复制整个模型。实际实现还需处理模型 Buffer 的复制、初始化和保存。

**为什么没在 U-Net 类里看到？** EMA 通常写在 `train_denoiser()` 或其调用的帮助模块中，而不是 `TinyUNet.forward()` 中。用以下代码在你自己的 Notebook 中定位：

```python
import inspect
print(inspect.getfile(train_denoiser))
print(inspect.getsource(train_denoiser))
# 搜索 ema / decay / lerp_ / deepcopy / shadow
```

`checkpoint['ema']` 只是**训练代码自定义保存的字典键**，不是 `torch.load()` 或 `nn.Module` 的保留关键字。生成时新建同架构 U-Net，再 `load_state_dict(checkpoint['ema'])`，表示把 EMA 参数加载进去；不是又训练了新网络。

**易混淆：** (1) `model.eval()` 不会自动使用 EMA；(2) `register_buffer()` 不等于注册 EMA；(3) EMA 权重平均不同于损失曲线的滑动平均；(4) EMA 往往改善样本稳定性，但不保证所有指标都更好。

## Q6. 图片里的 $\mu_\theta$ 和对应方差是谁的？$\epsilon_\theta$ 在不同 $t$ 下为什么不同？“累计噪声”究竟怎么合成？

### A. 均值和方差描述的是“从当前 $x_t$ 向上一时刻 $x_{t-1}$ 采样”的条件分布

模型用一个高斯分布近似一步反向生成：

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(\mu_\theta(x_t,t),\sigma_t^2I\big).$$

其中

$$\boxed{\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)}$$

是**上一时刻 $x_{t-1}$ 在给定当前图像 $x_t$ 时，模型给出的条件均值**，绝不是 $x_0$、$x_t$ 或噪声的均值。它是像素级张量，形状与当前图像一致；“均值”表示重复按此反向条件分布采样时各像素的平均位置。

当前教学 Scheduler 把反向方差固定为

$$\sigma_t^2=\tilde\beta_t=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},\quad (t\ge2).$$

`posterior()` 的 `mean` 与 `variance` 分别就是这一反向高斯的中心和方差；`step()` 再用新随机数 $z$ 抽取 $x_{t-1}=\mu_\theta+\sigma_tz$。最后一步不另加随机噪声。

**重要限定：** $\tilde\beta_t$ 源自给定真实原图 $x_0$ 的解析后验 $q(x_{t-1}\mid x_t,x_0)$。把它固定为模型 $p_\theta(x_{t-1}\mid x_t)$ 的方差，是 DDPM 的模型选择；并不意味着未知的真实反向边缘分布 $q(x_{t-1}\mid x_t)$ 本身精确具有这组参数。

### B. $\epsilon_\theta(x_t,t)$ 是累计等效噪声的预测，不是单步新增噪声

前向**单步**公式：

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,\qquad\epsilon_t\overset{ind}{\sim}\mathcal N(0,I).$$

两步展开：

$$\begin{aligned}
x_1&=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1,\\
x_2&=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2\\
   &=\sqrt{\alpha_1\alpha_2}x_0+\underbrace{\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2}_{\text{截至第二步的噪声总贡献}}.
\end{aligned}$$

令 $\bar\alpha_2=\alpha_1\alpha_2$，定义**标准化的累计等效噪声**：

$$\boxed{\bar\epsilon_2=\frac{\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2}{\sqrt{1-\bar\alpha_2}}}.$$

由于独立标准高斯的线性组合仍为高斯，且合成噪声总方差为 $\alpha_2\beta_1+\beta_2=1-\bar\alpha_2$，所以 $\bar\epsilon_2\sim\mathcal N(0,I)$。

**手算例子：** $\beta_1=0.1,\beta_2=0.2$，故 $\alpha_1=0.9,\alpha_2=0.8,\bar\alpha_2=0.72$；某一次前向轨迹抽到 $\epsilon_1=1,\epsilon_2=-1$，则

$$\bar\epsilon_2=\frac{\sqrt{0.8\cdot0.1}\cdot1+\sqrt{0.2}\cdot(-1)}{\sqrt{0.28}}\approx-0.3106.$$

一般 $t$ 步可以写成：

$$\boxed{x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\bar\epsilon_t,\qquad\bar\epsilon_t\sim\mathcal N(0,I)}.$$

若你确实保存了整条前向链的逐步噪声，那么

$$\bar\epsilon_t=\frac{\sum_{s=1}^{t}\left(\sqrt{\beta_s}\prod_{j=s+1}^{t}\sqrt{\alpha_j}\right)\epsilon_s}{\sqrt{1-\bar\alpha_t}}.$$

**为什么训练代码只需一次 `randn_like`？** 因为可以直接抽一份服从同一标准高斯分布的 `eps`，用合并后的公式构造 $x_t$，其**条件分布**与逐步加噪一致；不代表是在逐元素复现某一条先前的随机轨迹。

不同 $t$、不同 $x_t$ 时，模型输出 $\epsilon_\theta(x_t,t)$ 通常不同；同一个 U-Net 的全部时间步共享参数 $\theta$，利用 TimeEmbedding 调整行为。训练用 MSE 使它预测**当前这份累计等效噪声**，而非最后一步 $\epsilon_t$。固定模型、输入和时间时预测是确定的，但预测本身不保证服从 $\mathcal N(0,I)$。

## Q7. 为什么从标准高斯 $z$ 采样后，还要变成 $X=\mu+\sigma z$？原图、前向带噪图和反向采样图的均值方差各是什么？

### A. $X$ 是随机变量；一次采样得到的 $x$ 是具体张量

写 $X\sim\mathcal N(\mu,\sigma^2)$，表示**随机变量 $X$ 的分布**具有均值 $\mu$ 和方差 $\sigma^2$；它不是“一张随机图片已经被抽出来”的意思。执行一次 `torch.randn_like` 得到的 Tensor 才是一份具体实现值。

若 $z\sim\mathcal N(0,1)$，则

$$X=\mu+\sigma z\quad\Longrightarrow\quad E[X]=\mu,\qquad\mathrm{Var}(X)=\sigma^2.$$

例如 $\mu=0.5,\sigma^2=0.09$（故标准差 $\sigma=0.3$）。抽到 $z=-1,0,1$ 时，分别得到 $X=0.2,0.5,0.8$。三个取值来自**均值 0.5、方差 0.09** 的同一个高斯分布。

**为什么不一直用 $\mathcal N(0,1)$？** 标准高斯只是一种容易采样的基础随机源。DDPM 希望每一步采样满足模型预测的**当前反向分布**，而不是每一步都抽一张全新的标准高斯图片；所以先将 $z$ 按所需标准差缩放，再平移到所需均值。

### B. 四种不同分布：绝不能混为一谈

| 对象 | 分布或约束 | 均值、方差究竟针对什么 |
| --- | --- | --- |
| 真实干净图 $x_0$ | 来自 MNIST 真实数据分布 $p_{data}$；预处理后每个像素取值 $[-1,1]$ | **没有**统一的 0 均值、1 方差保证；也一般不是高斯分布 |
| 随机加噪的 $\epsilon$ | $\mathcal N(0,I)$ | 对重复抽噪声，每个像素位置的噪声均值 0、方差 1，各维独立 |
| **固定 $x_0$ 时**的 $x_t$ | $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$ | 重复对同一原图加噪，均值是缩放后的原图，噪声带来方差 $1-\bar\alpha_t$ |
| **固定当前 $x_t$ 时**的反向 $x_{t-1}$ | $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta,\sigma_t^2I)$ | 均值是模型预测的上一时刻中心，方差是这一反向步采用的采样方差 |

**易混淆：** “对固定图像重复加噪时，某个像素位置的随机方差”不等于“一张图片所有像素值的空间方差”，也不等于“MNIST 数据集同一像素跨样本的方差”。这三种计算对象不同。

### C. 干净图变成带噪图时，均值和方差怎么变？

对固定的干净图 $x_0$，有：

$$\boxed{E[x_t\mid x_0]=\sqrt{\bar\alpha_t}x_0,\qquad \mathrm{Cov}(x_t\mid x_0)=(1-\bar\alpha_t)I.}$$

**具体像素例子：** 固定原图中的某个像素 $x_0=0.8$，假设 $\bar\alpha_t=0.36$，则

$$x_t=0.6\times0.8+0.8\epsilon=0.48+0.8\epsilon.$$

因此如果反复对**同一个** 0.8 像素加噪，$x_t$ 的均值为 **0.48**，方差为 **0.64**。某次抽到 $\epsilon=1$，实际结果是 $x_t=1.28$，**即使干净像素被约束在 $[-1,1]$，中间带噪像素也完全可能超过这个范围**。

若把 $x_0$ 本身也看作从真实数据分布随机抽出的变量，设单个像素位置 $E[x_0]=m$、$\mathrm{Var}(x_0)=v$，且噪声独立，则

$$\boxed{E[x_t]=\sqrt{\bar\alpha_t}\,m,\qquad\mathrm{Var}(x_t)=\bar\alpha_t v+(1-\bar\alpha_t).}$$

图像整体可用协方差矩阵概括：

$$\mathrm{Cov}(x_t)=\bar\alpha_t\,\mathrm{Cov}(x_0)+(1-\bar\alpha_t)I.$$

此时 **$q(x_t)$ 一般是许多高斯分布的混合，不一定本身就是高斯**。上面的闭式高斯分布特指“给定 $x_0$ 时的 $q(x_t\mid x_0)$”。

### D. 终点 $x_T$ 为什么可以从标准高斯初始化？

如果 $\bar\alpha_T\to0$，则

$$x_T=\sqrt{\bar\alpha_T}x_0+\sqrt{1-\bar\alpha_T}\epsilon\ \longrightarrow\ \epsilon\sim\mathcal N(0,I)$$

（指分布收敛，而不是有限步后每个数值等于某个固定噪声）。经典 1000 步线性 $\beta:0.0001\to0.02$ 时 $\bar\alpha_T$ 约 $4\times10^{-5}$，原图系数约 $0.00635$，噪声系数几乎为 1，因而**终点非常接近标准高斯**。有限 $T$ 的一致性是近似的，不是数学上的严格等号。

生成时正是利用这个设计，直接令 `sample = torch.randn(B,C,H,W)` 作为初始 $x_T$；随后每一步依据 $p_\theta(x_{t-1}\mid x_t)$ 采样，均值/方差一般**不再**是 0 和 1。

### E. 对应你的 `TeachingDDPMScheduler.step()`

```python
# 当前 sample 就是 x_t（已经抽出的一份具体图像张量）
eps_pred = model(sample, times)                      # 预测累计等效噪声
estimated_clean = (sample - noise_scale * eps_pred) / signal
mean, variance = scheduler.posterior(estimated_clean, sample, times)
# 反向一步，z 是重新抽的标准高斯
z = torch.randn_like(sample)
prev = mean + variance.sqrt() * z                    # t > 0 时
```

**这三个“高斯”含义不同：** `eps` 是前向训练监督标签；`z` 是为反向分布采样提供标准随机源；`p_theta` 的均值/方差由网络预测与固定 Schedule 决定。最后一步通常不加新的 `z`。

### 一眼记住的关系图

```text
真实图像 x0（MNIST，不必是高斯；像素归一化到 [-1,1]）
  │ 训练：随机 t + 随机累计 ε~N(0,I)
  ▼
xt = sqrt(abar)*x0 + sqrt(1-abar)*ε
  │
  └─ U-Net(xt,t) → eps_pred（预测累计 ε）→ MSE 更新 θ

生成：完全不需要真实 x0、真实 ε、MNIST 标签
  初始 xT ~ N(0,I)
       ↓
  U-Net(xT,T) → eps_pred
       ↓
  由 eps_pred 求 μθ；由 Schedule 查 σt²
       ↓
  抽新 z~N(0,I)，得到 x(T-1) = μθ + σt*z
       ↓
  重复至 x0（最后一步不另加随机噪声）
```

**最终总结：** $\mathcal N(0,I)$ 是训练加噪和生成采样的**标准随机源**，不是说所有图像/所有反向状态都必须满足均值 0、方差 1。每一步的 $\mu_\theta$ 描述“从当前 $x_t$ 反向生成上一步”的中心；$\sigma_t^2$ 描述这一步所需的随机性。EMA 则只是在训练过程中平滑权重，和噪声分布无关。
