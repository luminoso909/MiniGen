# MiniGen｜DDPM 训练、生成与 MNIST 数据流

> 目标：在不依赖 ELBO/KL 推导的前提下，能够独立说清 `MNIST → DataLoader → 加噪 → U-Net → Loss → 训练权重 → 从纯噪声采样` 的完整过程。本文基于当前 `TeachingDDPMScheduler`、`TinyUNet` 和 Notebook 中的采样调用。

## 1. 训练到底学到了什么？

训练集提供干净图像 $x_0$，自己随机抽取时间 $t$ 和标准高斯噪声 $\epsilon$：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

让共享参数的 U-Net 预测 $\epsilon_\theta(x_t,t)$，用 $\mathrm{MSE}(\epsilon_\theta,\epsilon)$ 优化网络参数 $\theta$。**训练不是让每张图走一条完整的 1000 步轨迹**：任意 $t$ 都可从 $x_0$ 一次构造 $x_t$。优化器的更新步数和扩散时间步是不同概念。

```python
x0, labels = next(iter(train_loader))       # x0: [B,1,28,28]，labels: [B]
t = torch.randint(T, (len(x0),), device=x0.device)
eps = torch.randn_like(x0)
xt = scheduler.add_noise(x0, eps, t)
eps_pred = model(xt, t)                   # [B,1,28,28]
loss = F.mse_loss(eps_pred, eps)
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**更正一个易错认识**：知道合成噪声 $\epsilon$，可以从 $x_t$ **精确代数反解 $x_0$**；但一般不能唯一还原某次前向随机轨迹实际抽到的 $x_{t-1}$。已知 $x_0,x_t$ 时，上一步仍有后验随机性。反向模型学习构造合理的后验近似，不是找回历史上唯一的一条噪声轨迹。

### 数值例子：真实噪声与预测噪声

设 $\sqrt{\bar\alpha_t}=0.8$、$\sqrt{1-\bar\alpha_t}=0.6$、$x_0=1$、真实 $\epsilon=-0.5$：

$$x_t=0.8\times1+0.6\times(-0.5)=0.5.$$

已知真实 $\epsilon$，反解 $x_0=(0.5-0.6\times(-0.5))/0.8=1$。若网络预测 $\epsilon_\theta=-0.3$，则估计 $\hat x_0=(0.5-0.6\times(-0.3))/0.8=0.85$。**预测误差影响原图估计**；但这还不是某条实际前向轨迹的确切 $x_{t-1}$。

## 2. 生成时完全没有真实图像，为什么还能运行？

因为训练后 U-Net 的参数已经学到带噪图像的统计结构。生成时从全新的 $x_T\sim\mathcal N(0,I)$ 开始，反复使用模型预测噪声，Scheduler 根据已知噪声日程构造上一步的分布并采样，直到输出图像。**生成没有真实 $\epsilon$、没有 MSE、没有 optimizer，也不需要训练数据集。**

```python
model.eval()
with torch.no_grad():
    x = torch.randn(8, 1, 28, 28, device=device)  # 新的 x_T
    for k in reversed(range(scheduler.num_steps)):
        t = torch.full((len(x),), k, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        z = torch.randn_like(x)                     # 反向采样的新随机数
        x = scheduler.step(eps_pred, k, x, z)       # 得到上一步状态
    generated = x
```

**三种不同的噪声，绝不能混淆**：

| 符号/变量 | 何时产生 | 用途 |
|---|---|---|
| $\epsilon$ / `eps` | 训练时 `randn_like(x0)` | 制造 $x_t$，同时作为 MSE 真值标签 |
| $\epsilon_\theta$ / `eps_pred` | U-Net 前向计算 | 估计噪声，决定反向均值；不必服从标准高斯 |
| $z$ / `noise`（传给 `step`） | 生成时每一步新抽取 | 从反向高斯分布采样；末步通常不用 |

从 $x_T$ 开始的一张随机噪声图，不需要对应任何先前真实发生过的正向扩散。**生成是在创造新的样本，而非复原一个隐藏的原图。**

## 3. Notebook 这段采样代码逐行做什么？

```python
checkpoint = torch.load(training_result['checkpoint'],
                        map_location='cpu', weights_only=True)
sampling_model = TinyUNet(base=32).to(DEVICE)
sampling_model.load_state_dict(checkpoint['ema'])
initial_noise = torch.randn((8, 1, 28, 28),
                            generator=torch.Generator().manual_seed(41)).to(DEVICE)
sampling_result = sample_ddpm(
    sampling_model,
    TeachingDDPMScheduler(clip_denoised=True).to(DEVICE),
    initial_noise, seed=42, max_seconds=300,
)
```

- `training_result['checkpoint']`：训练时保存的权重文件路径；`torch.load` 读取文件，`map_location='cpu'` 先放到 CPU。
- `weights_only=True`：限制反序列化所允许的对象类型；不是“只读取模型权重这一项”，文件仍可有 EMA、优化器等数据。
- `TinyUNet(base=32)`：新建**相同结构**的网络；`load_state_dict(checkpoint['ema'])` 用已训练的 EMA 参数覆盖随机初始化参数。
- `initial_noise`：从标准高斯一次抽取 8 张 28×28 单通道噪声图；种子 `41` 便于复现初始图像。
- `sample_ddpm(...)`：执行逐步反向采样。`seed=42` 的精确使用方式由该函数实现决定；一般用于采样中的其他随机步骤。`max_seconds` 是运行预算，超时可能产生未完成轨迹。
- `clip_denoised=True`：把采样器内部**估计的干净图**裁到 `[-1,1]`；不是限制标准高斯噪声大小。
- `sampling_result['images']`：返回生成结果；`['snapshots']`：返回中间状态供观察；`['completed']`：判断是否走完整条反向链。

可用 `print(inspect.getsource(sample_ddpm))` 确认采样循环、随机数种子和快照保存的具体实现。

### 如何阅读采样轨迹图片？

每一列对应**同一张起始随机噪声图的不同采样时刻**。`initial → after k=749 → ... → after k=0` 表示从高噪声逐步走向数字形状；此处的 `k` 是扩散反向索引，**不是** Optimizer Step。若 `completed=False`，不能把中断结果当作最终生成。

绘图 `image[0]` 取灰度通道，`.clamp(-1,1)` 仅在绘制时限制显示数值，`(image+1)/2` 将像素映射回 `[0,1]`。图像的显示操作不负责生成。

## 4. MNIST 是什么？真实图片存在哪里？

MNIST 是**手写数字 0～9 的灰度图像数据集**；每张图像为 $28\times28$、一个通道，并带一个整数标签（数字类别）。官方标准划分是 **60,000 张训练图、10,000 张测试图**。当前项目按 Notebook 说明，从官方训练部分取 **10,000 张训练、1,000 张验证**，不使用官方测试样本参与调参；其余官方训练样本不一定都用于本实验。

数据不是保存在 `manifest.json` 里：

- `manifest` 是**数据来源、哈希校验、划分索引与数量等元信息/记录**。
- 图像和标签由 `torchvision.datasets.MNIST` 相关的数据文件提供，通常位于 `ROOT/data/MNIST/raw/`（可能含下载的压缩文件或解压后的 IDX 文件；以项目实际路径为准）。
- 当前 Notebook `from minigen.data.mnist import prepare_mnist`，项目中的具体逻辑在 `src/minigen/data/mnist.py`；查阅源码才能确认最终加载和保存路径。
- `train_set.indices` 与 `val_set.indices` 表明两者公开了各自的子集索引；不能仅从 manifest JSON 推断数据没有加载。

### DataLoader 为什么同时输出 images、labels？

底层 MNIST Dataset 单个样本返回 `(image, label)`，例如：

```python
image, label = train_set[0]
print(image.shape)   # 通常为 [1,28,28]（项目已应用 ToTensor 与标准化）
print(label)         # 0~9 的一个整数
```

`DataLoader(batch_size=64)` 将 64 个 `(image,label)` 自动组成一个 Batch：

```python
images, labels = next(iter(train_loader))
print(images.shape)  # [64,1,28,28]
print(labels.shape)  # [64]
```

MNIST 是带标签的标准数据集，所以 Dataset 默认返回标签。**当前无条件 DDPM 不用标签**；训练可以写 `for images, _ in train_loader:`。监督信号是我们自行采样的噪声 `eps`，不是数字类别 `labels`。

`shuffle=True`：训练每轮重排样本；`shuffle=False`：验证保持固定顺序；`num_workers=0`：在主进程取样和组装 Batch；`generator=...manual_seed(SEED)`：控制 DataLoader 部分随机性。

### 在你的 Notebook 中，直接查实际文件来源

```python
import inspect
from pathlib import Path

print(inspect.getsource(prepare_mnist))
print(type(train_set), type(val_set))
print('底层数据集:', type(getattr(train_set, 'dataset', train_set)))
base = getattr(train_set, 'dataset', train_set)
print('数据根目录:', getattr(base, 'root', '请查看 prepare_mnist 源码'))
print('样本:', train_set[0][0].shape, train_set[0][1])
print('数据目录里的文件:')
for p in sorted((ROOT / 'data').rglob('*')):
    if p.is_file():
        print(p.relative_to(ROOT), p.stat().st_size)
```

## 5. 训练 Loss 和验证 Loss 是怎么来的？

**公式一样，数据不同，是否更新参数也不同**。

- **训练 Loss**：从 `train_loader` 取得图片，随机抽 `t,eps`，算噪声 MSE，随后 `backward()` 和 `optimizer.step()`。
- **验证 Loss**：从 `val_loader`/`val_probe` 取得未参与训练的图片和噪声/时间，算同样的噪声 MSE，但在 `eval()` 与 `no_grad()` 下评估，**不更新参数**。
- **固定验证探针**：图片、时间、噪声都只抽取一次，训练前后不变，用于公平对比。当前 Notebook 只用 64 张验证图的一份探针，不等于完整验证集平均损失。
- **生成质量**：仍需从纯噪声实际采样观察；低 MSE 本身不是数字识别准确率或生成质量分数。

```python
model.eval()
with torch.no_grad():
    x0, eps, t = val_probe
    x0, eps, t = x0.to(DEVICE), eps.to(DEVICE), t.to(DEVICE)
    xt = scheduler.add_noise(x0, eps, t)
    val_loss = F.mse_loss(model(xt, t), eps)
```

## 6. 五个自检问题（附简答）

1. **生成时为什么没有 `clean` 也行？** 因为从标准高斯抽一张新的 $x_T$，训练后的网络提供去噪预测，不是在还原某张隐藏原图。
2. **知道真实 `eps` 就知道真实前向链中的 `x_{t-1}` 吗？** 不一定。知道它可以精确反解 `x0`；已知 `x0,xt` 时上一步通常仍有条件随机性。
3. **`labels` 为什么不送进 U-Net？** MNIST 数据集本来带数字类别标签，但当前模型是无条件噪声预测器，监督标签是 `eps`。
4. **`val_loss` 怎样计算？** 对未参与训练的图，使用同一加噪与噪声 MSE 公式，评估时不开梯度、不更新参数。
5. **训练的 `eps`、U-Net 的 `eps_pred`、生成 `step()` 中的 `z` 是否相同？** 三者来源和用途不同，绝不能混为一谈。

**最终速记：MNIST 存数据，DataLoader 组 Batch；Scheduler 规定噪声规则；U-Net 学噪声预测；训练用 MSE 改参数；生成用已训模型与 Scheduler 从纯噪声逐步抽新图。**
