# MiniGen · DDPM Scheduler 代码十一问

> **扩充版说明**：保留原问题与核心答案，额外增加直觉、数值例子、代码对照与延伸。推荐先读每问的粗体结论，需要时再展开补充。


> 复习定位：只讨论当前 `TeachingDDPMScheduler + U-Net`，不展开 ELBO。约定 `x0` 是干净图，`xt` 是带噪图；代码索引 `t=0` 对应数学上的**第一步**加噪。

### 1. `_extract` 的 `table` 是什么？取出的类型是什么？
`table` 是传入的 **1D `torch.Tensor` 查找表**（如 `self.signal`，它是注册的 buffer），形状 `[T]`、当前代码通常为 `torch.float32`。`table[timesteps]` 取出每张图片对应的系数，形状 `[B]`；再 reshape 为 `[B,1,1,1]`（图像四维时），**仍是 Tensor / float32**。`timesteps` 本身是 `[B]` 的 `torch.long` 索引。

#### 一个可打印的张量例子

```python
import torch
table = torch.tensor([0.95, 0.80, 0.50])       # [T=3], float32
steps = torch.tensor([0, 2], dtype=torch.long) # [B=2]
image = torch.zeros(2, 1, 28, 28)              # [B,C,H,W]
chosen = table[steps]                           # [2] = [0.95,0.50]
coef = chosen.reshape(2, 1, 1, 1)               # [2,1,1,1]
print(chosen.dtype, coef.shape)
```

为什么不直接用 `[B]` 乘 `[B,C,H,W]`？广播从**最后一维**开始对齐，[B] 会误当成宽度维系数（宽度不为 B 时还会报错）。改成 `[B,1,1,1]` 才代表“每张图一份系数，覆盖全部通道和像素”。`_extract` 只负责查表和 reshape，**不修改任何系数值**。

### 2. 为什么把同一个 `noise` 传入 `add_noise` 和 `loss`？
对。训练时在外部采样一次 `eps = torch.randn_like(x0)`，它既用于构造 `xt`，又是真值标签。`loss(eps_pred, eps)` 必须对照**这一次实际加入的噪声**，不能再随机抽一份。

#### 标签必须是“同一份”的例子

假设清晰像素 `x0=1`、本次噪声 `eps=-1`、本步 `alpha=0.9,beta=0.1`，则 `signal≈0.9487,noise_scale≈0.3162`，实际输入 `xt≈0.9487*1+0.3162*(-1)=0.6325`。如果网络预测 `eps_pred=-0.8`，正确监督误差是 $(-0.8-(-1))^2=0.04$。

倘若 loss 临时又采到了另一个噪声 `eps2=0.2`，误差会变成 $(-0.8-0.2)^2=1$，而且它根本不是制造 `xt` 时所用的标签。**噪声可以在函数外生成，也可以由函数返回；关键是加噪和监督使用完全相同的样本。**

### 3. Scheduler、schedule、model、U-Net 的区别？
**schedule** 是预设的噪声强度时间表（`betas/alphas/alpha_bars`）；**Scheduler** 是管理这些系数、执行加噪与反向采样公式的对象，**没有可训练参数**；当前 `model` 就是 **U-Net**，输入 `(xt,t)`，输出预测噪声 `eps_theta`，它的权重 `theta` 需要训练。

#### “谁知道什么”的速记

| 对象 | 预设/学习 | 主要输入 | 主要输出 |
|---|---|---|---|
| Noise schedule | 预设 | 时间 $t$ | 单步 $\beta_t$ 及累计系数 |
| Scheduler | 不训练 | $x_0/ x_t$、时间、噪声等 | 带噪样本或上一状态样本 |
| TinyUNet | **训练** | $(x_t,t)$ | $\epsilon_\theta(x_t,t)$ |
| Optimizer | 算法预设 + 有状态 | `model.parameters()` 的梯度 | 更新模型参数 |

`model = TinyUNet(...)` 是当前场景的具体选择；`scheduler = TeachingDDPMScheduler(...)` 是固定规则。反向采样时 U-Net 给出估计、Scheduler 决定怎么走一步。**把反向传播 `loss.backward()` 和反向扩散 `scheduler.step()` 分开记。**

### 4. `F.mse_loss(..., reduction='mean')` 是什么？
`mean`：全部元素误差平方求平均（默认）；`sum`：求和；`none`：逐元素保留误差，输出同形状张量。例：误差平方 `[1,4]` → mean=2.5，sum=5，none=`[1,4]`。

#### 为什么训练常用 mean

假设 `pred=[1,3]`，`truth=[2,5]`，误差平方为 `[1,4]`：`mean=2.5`、`sum=5`、`none=[1,4]`。

对于 `[B,C,H,W]` 噪声，`mean` 会对 **B×C×H×W 的全部元素**求平均；`sum` 的大小会随批量和分辨率明显改变。`none` 则适合自己实现按样本、按时间或按像素加权，但最终反向传播通常需要将其汇总为标量。

### 5. 反向均值系数为何可以预计算？
因为前向 `beta/alpha` 是预先指定的，已推导的条件后验均值为 `mean = c0(t)*x0 + ct(t)*xt`；`c0/ct` 及后验方差只依赖时间表，故可预先算好。**训练时**用已知 `x0` 得到真实条件后验；**生成时**用模型估计的 `x0_hat` 代替，不是精确恢复原后验。

#### 一个后验系数例子

设 $\alpha_t=0.8,\beta_t=0.2,\bar\alpha_{t-1}=0.75$，则 $\bar\alpha_t=0.6$。

$$c_0=\frac{\sqrt{0.75}\times0.2}{0.4}\approx0.4330,\qquad c_t=\frac{\sqrt{0.8}\times0.25}{0.4}\approx0.5590.$$

若 `clean=1, sample=0.5`，后验均值 `mean=c0*1+ct*0.5≈0.7125`；方差 $\tilde\beta_t=0.2\times0.25/0.4=0.125$。

**留意：** 这两个系数的和不一定等于 1，它们不是简单的“按百分比混合两张图”。生成时 `clean` 传的是模型估计的 $\hat x_0$，所以结果取决于预测质量。

### 6. 什么是“采样”？`sample` 为什么是确定值？
从分布中**随机抽一次具体数值**叫采样。如 `mu + sigma*torch.randn_like(mu)`。`sample` 只是变量名，代表**这一次已经抽到并持有的当前 `xt` Tensor**；生成开始它由 `torch.randn(...)` 初始化，随后每轮 `step` 更新。分布随机，但抽完后的数值是确定的。

#### 一次采样的现实含义

假设某个反向高斯参数是 $\mu=0.7,\sigma^2=0.04$，标准差 $\sigma=0.2$。那么 `mean + sqrt(var)*z`：若本次 `z=-1`，`sample=0.5`；若 `z=0.5`，`sample=0.8`。**相同的均值和方差，抽出来的具体值可以不同。**

`sample` 在生成开始由 `torch.randn(B,C,H,W)` 初始化为 $x_T$；每执行一次 `step(...)`，变量就被赋值为新状态。变量叫 `sample` 不表示里面仍存一个概率分布对象，它持有的是一份已经确定的 Tensor 样本。

### 7. 为什么 `len(sample)` 等于 batch size？
`torch.Tensor.__len__()` 返回**第 0 维大小**。若 `sample.shape == [8,1,28,28]`，则 `len(sample)==8`，而 `len(sample.shape)==4`（维度数）。

#### 三个很像却不同的 Python 表达式

```python
x = torch.zeros(8, 1, 28, 28)
len(x)          # 8：第 0 维的长度（batch size）
x.shape[0]      # 8：同上
len(x.shape)    # 4：维度的数量
x.ndim          # 4：同上
```

`torch.full((len(x),), i)` 因而生成 `[8]` 的时间索引；它不是创建一个只有 4 个数的时间张量。

### 8. U-Net 预测什么？`eps_theta` 要等于前向抽到的 `eps` 吗？生成时为何知道 alpha？
U-Net 输出 `eps_theta(xt,t)`；`alpha/alpha_bar` **不是模型预测的**，由预设的 schedule 给出。训练中用真实随机 `eps ~ N(0,I)` 做标签，但仅凭 `xt,t` 通常无法唯一找回那一次的 `eps`；MSE 学习的是统计上的最佳预测，**不是每次逐点还原同一随机样本**。真实 `eps` 边缘分布为标准高斯，**不保证网络输出 `eps_theta` 也服从标准高斯**。生成使用与训练匹配的、早就预设的时间表，并从 `xT ~ N(0,I)` 开始：不需要知道任何真实训练图片或原始噪声。

#### 为什么“同为高斯”并不能替代学噪声

训练标签 $\epsilon\sim\mathcal N(0,I)$，是**不考虑当前图片时的边缘分布**；网络任务却是给定 $x_t,t$，预测本次加噪中噪声的**条件信息**。同样是高斯分布，随便抽一份 `torch.randn_like(xt)` 并不知道哪些像素更像真实笔画。

一个一维反例：若 $x_0$ 只能是 0 或 1，$x_t=0.5$，同一个观测值可能来自两种不同 $x_0$ 与不同 $\epsilon$。所以网络不会保证复原每次抽到的精确噪声；MSE 最优预测是 $\mathbb E[\epsilon\mid x_t,t]$。

**生成为什么知道 alpha？** 因为生成时加载的 Scheduler 和训练时约定同一张噪声表。原图和标签未知，**规则却由我们预先制定**，两者毫不矛盾。把这里的时间表类比于训练和推理都知道的 RoPE 旋转规则就很好记。

### 9. 什么是像素截断 `clamp`？
`estimated_clean.clamp(-1,1)` 将**估计的干净图像**每个像素截到训练图像的归一化范围：`[-1.3,0.4,1.2] → [-1,0.4,1]`。不是要求高斯噪声只能在 `[-1,1]`，也不同于截断正态初始化（后者是按截断后的分布采样）。

#### 三种容易混淆的操作

```python
x = torch.tensor([-1.4, -0.2, 0.8, 1.3])
x.clamp(-1, 1)       # tensor([-1.0, -0.2, 0.8, 1.0])
```

- `clamp`：样本值已经存在，超出的值直接压到边界。
- `trunc_normal_`：初始化时**从截断后的概率分布抽样**，不是先高斯采样再简单把尾部压到边界。
- `eps ~ N(0,1)`：高斯噪声本身不受 `[-1,1]` 限制。

这个 Scheduler 的 `clip_denoised` 针对**估计的干净图** $\hat x_0$；过强截断可能损坏预测，因此是可选项。

### 10. 最短训练与生成流程？
```python
# 训练：更新 U-Net，不训练 Scheduler
x0 = next_batch()                         # [B,C,H,W]
t = torch.randint(T, (len(x0),), device=x0.device)
eps = torch.randn_like(x0)                 # 本次真实噪声
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)                    # U-Net 预测噪声
loss = scheduler.loss(eps_pred, eps)
optimizer.zero_grad(); loss.backward(); optimizer.step()

# 生成：已训练好的 U-Net + 同一噪声日程
model.eval()
with torch.no_grad():
    x = torch.randn(B, C, H, W, device=device)   # 初始 xT
    for i in reversed(range(T)):
        t = torch.full((B,), i, dtype=torch.long, device=device)
        eps_pred = model(x, t)
        x = scheduler.step(eps_pred, i, x, torch.randn_like(x))
    generated = x
```

#### 串成一张最小计算图

训练一次：

```python
x0 = images.to(device)
t = torch.randint(0, scheduler.num_steps, (len(x0),), device=device)
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)
loss = (eps_pred - eps).square().mean()
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

生成（使用**已训练**的 `model`）：

```python
model.eval()
with torch.no_grad():
    x = torch.randn(8, 1, 28, 28, device=device)
    for i in reversed(range(scheduler.num_steps)):
        t = torch.full((len(x),), i, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        x = scheduler.step(eps_pred, i, x, torch.randn_like(x))
```

训练随机抽一个 $t$，只优化一次噪声预测；生成按时间倒序调用多次 `step`。两条路线共享**同一个训练完成后的网络权重**和对应 Schedule，但并不共用具体随机噪声样本。

### 11. `step` 的 `noise` 与前向 `noise` 一样吗？函数为什么放在一起？
**不是同一个**。训练中的 `eps` 是用于制造 `xt` 的监督标签；`step` 中 `predicted_noise` 是 U-Net 输出；`step` 的 `noise` 则是**本次反向高斯采样新抽到的随机数**（末步不使用）。`add_noise/loss` 主要服务训练，`posterior/step` 主要服务反向采样；都依赖同一噪声时间表，统一封装在 Scheduler 中是为了复用与一致性。

**一句话总览：** Scheduler **知道规则**；U-Net **学习预测噪声**；训练用外部已知噪声监督，生成用 U-Net 的预测 + Scheduler 的公式一步步抽样。

#### 三份噪声一定要起不同的名字

```python
train_eps = torch.randn_like(x0)       # 训练时用于污染 x0；是监督真值
pred_eps = model(xt, t)                # U-Net 的输出，不是额外随机抽样
reverse_z = torch.randn_like(xt)       # 生成步骤用于反向高斯抽样
```

`posterior()` 使用前向日程预计算的均值和方差系数；`step()` 把模型预测接进去，完成**一小步**反向采样。因此它们同 `add_noise()/loss()` 属于不同阶段。合在一个 Scheduler 是为了共享规则与减少重复配置，**不是数学上要求必须放在同一个类里**。
