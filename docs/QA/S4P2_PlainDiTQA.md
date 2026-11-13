# MiniGen · S4-02 学习 Q&A（最终分类整理版）

> **整理日期**：2026-11-12\
> **范围**：仅整理本次学习过程中你实际提出的技术问题 **Q1–Q15**；普通 Notebook 导读不自动转成提问。\
> **依据**：当前对话中的原始提问、已归档 `MiniGen_S4-02_QA.md` 与上传的 `S4_02_LatentFlow与PlainDiT训练闭环(1).ipynb`。\
> **学习状态**：S4-02 参考教学闭环已运行；个人练习 `flow_loss` 与 `sample_ode` 在上传 Notebook 输出中仍为**待完成**；S4-01 的进一步数学推导暂缓。\
> **可信度标记**：具体数值来自 Notebook 的输出；没有获取独立的 S3 `ConditionalDiTReference` 单元源码，故相关内部初始化细节会明确标注“需要查 S3 源码”。

## 一页总览：本节究竟学到了什么？

**主线**：PIL 合成图标 → 冻结 VAE 编码 latent / 冻结 CLIP 编码 token → Flow 路径 `z_t=(1−t)ε+tz` → 条件 DiT 预测速度 → 速度 MSE → Euler/Heun/CFG ODE 采样 → VAE Decoder → 颜色/形状评测。

| 维度 | 本节已核实结果 | 不能夸大的地方 |
| --- | --- | --- |
| 数据 | 64 张训练、32 张几何留出、四种颜色×形状组合 | 不是新语义组合的零样本泛化 |
| VAE/CLIP | 64px latent `[B,4,8,8]`，文本缓存 `[5,5,512]` 与 Mask `[5,5]` | CLIP 的 77 是最长序列位置，不是特征宽度 |
| 模型 | 2 层、hidden 96、4 heads、约 0.61M 参数；复用 S3 条件 DiT | Flow 并非定义了另一套必然不同的 Transformer |
| 训练 | 固定 8 样本 160 次更新：MSE `3.06988 → 0.04474`；随机 2400 次更新 | 固定监督拟合不代表泛化或采样质量 |
| 条件利用 | 验证 Right MSE `0.8822`，Wrong Color `2.5740`，Wrong Shape `0.9516` | “换词后 MSE 变大”是间接证据，不是图文匹配评分 |
| 生成 | Euler/Heun 颜色各 `16/16`，但方形各 `0/8` | 形状能力明显不足，不能把颜色准确当作整体成功 |
| 工程验证 | Padding 不变性、Cross K/V 梯度、CPU 一步精确恢复通过 | 不等于多卡或跨硬件完全可复现 |
| 个人实现 | `flow_loss`、`sample_ode` | **待完成**，不可记作已通过 |

## 分类目录

- **第一篇｜数据、主干结构与参数初始化**：[Q1](#q01)、[Q2](#q02)、[Q4](#q04)、[Q6](#q06)、[Q7](#q07)
- **第二篇｜文本条件、Padding 与 Attention Mask**：[Q5](#q05)、[Q11](#q11)、[Q12](#q12)、[Q13](#q13)、[Q15](#q15)
- **第三篇｜Flow 时间、速度与预测误差**：[Q3](#q03)、[Q10](#q10)
- **第四篇｜条件评测与无条件生成**：[Q8](#q08)、[Q9](#q09)
- **第五篇｜ODE 采样与图片网格读法**：[Q14](#q14)

## 重要公式速查

| 公式/运算 | 含义 |
| --- | --- |
| `z_t=(1−t)ε+tz` | 训练时从噪声到干净 latent 的直线路径 |
| `u=z−ε` | 这条配对路径的目标速度 |
| `MSE(v_θ(z_t,t,C,M),u)` | 训练监督目标 |
| `v* = E[u ∣ z_t,t,C]` | MSE 意义下理想的条件平均速度 |
| `scores=QKᵀ/√d_h` | Cross-Attention logits，行是图像 Query、列是文本 Key |
| `masked_fill(~valid[..., :],−∞)` | 在 logits 级别屏蔽 PAD Key 列，**不要**将 token hidden states 写成 `−∞` |
| `v_CFG=v_u+s(v_c−v_u)` | 同一状态、时间的两支速度组合 |
| `z_{k+1}=z_k+h v(z_k,t_k)` | Euler ODE 积分 |
| `z_{k+1}=z_k+h(k₁+k₂)/2` | Heun 预测-校正步 |

---


# 第一篇｜数据、主干结构与参数初始化


<a id="q01"></a>

## Q1｜为什么构造这些图像？怎么构造？是现成图片吗？只有四种图案吗？

**原问题**：我没看懂构造这些图像有什么用？怎么构造的？图像是现成的还是怎么组织形成的？只有始终图案吗（红蓝*方圆）？

### 核心回答

这些图片**不是外部数据集的成品图片**，而是 Notebook 用 PIL 在内存中程序化绘制的 64×64 图标。数据仅有 4 种「语义类别」：红圆、蓝圆、红方、蓝方，分别对应 4 条英文 caption；但每一类含多个不同中心位置与大小的几何实例。它们用来在极低计算预算下检验：数据→冻结 VAE 与 CLIP→条件 DiT→速度 MSE→ODE 采样→图像，以及训练/验证分离、文字条件是否起作用。

### 生成方法

1. 先列出 100 个候选 `(cx,cy,size)`：`cx`、`cy` 分别从 `(24,28,32,36,40)` 选取（5×5），`size` 从 `(26,30,34,38)` 选取（4），合计 `5×5×4=100`。
2. 用固定随机种子 `40201` 打乱，取前 16 组作为训练几何参数、接下来 8 组作为验证几何参数。两组几何元组不重合。
3. 每组参数分别画红圆、蓝圆、红方、蓝方：先用 `Image.new` 创建白色画布，再用 `ImageDraw.ellipse/rectangle` 绘制。先以两倍目标分辨率绘制，再用 LANCZOS 缩到 64×64，改善边缘抗锯齿。
4. RGB 数组转换到 PyTorch `(C,H,W)`，并从像素 `[0,255]` 归一化到 `[-1,1]`。
5. 得到 `train_x [64,3,64,64]`、`val_x [32,3,64,64]`；类别索引 `train_y [64]`、`val_y [32]`。每类训练 16 张、验证 8 张，共 96 张。训练与验证图像 SHA-256 无交集。

**容易误解**：并不是 96 种语义，也不是只有四张固定图片；但也不包含未知颜色/形状组合。验证主要测未见过的几何实例，而非新类别组合的零样本泛化。64×64 只是低成本教学闭环；正式 MiniGen 目标尺寸仍为 256×256。

---

---


<a id="q02"></a>

## Q2｜Plain DiT 和 Conditional DiT 是同一个东西，只是预测目标不同吗？前处理路径一样吗？

**原问题**：plain dit 其实和conditional dit一样吧，只不过conditional dit是预测eps，plain dit是预测v；在进入plain dit之前，其实与conditional dit经过的路径完全一致是吗？

### 核心回答

**对 S4-02 的这份教学代码，主干确实复用了 S3 的 `ConditionalDiTReference` 结构**：Notebook 从 S3 的带 tag 代码单元恢复类定义、检查 SHA-256，然后以 `depth=2, hidden=96, heads=4, channels=4, size=8, patch=2, text_width=512` 新建随机初始化模型。它改为用 Flow Matching 的速度标签训练。**但术语上不能把“Plain=预测 v、Conditional=预测 ε”当作普遍定义**：Plain/Conditional 描述结构或条件化形式；ε-/x₀-/v-预测描述监督目标，两者是可组合的维度。

### 哪些一样，哪些不同

- 一样：RGB → 冻结 VAE 编码并缩放；Caption → 冻结 CLIP 产生 token hidden states 和 mask；DiT 的 patch embedding、图像 Self-Attention、文本 Cross-Attention、时间调制、输出 patch 和 unpatchify 的结构接口。
- 不一样：从干净 latent `z` 和高斯噪声 `ε` 构造模型输入的**加噪/插值路径**。典型 DDPM 为 `z_t = sqrt(alpha_bar_t) z + sqrt(1-alpha_bar_t) ε`；本节 Flow 为 `z_t=(1-t)ε+t z`。
- 不一样：监督目标，DDPM 教学为预测噪声 `ε`；本节为预测速度 `u=z-ε`。**不要混淆 Flow Matching 的速度与扩散文献中另一种被称为 v-prediction 的参数化**，它们依赖各自路径的具体定义。
- 不一样：采样更新，S4-02 通过速度场做 Euler/Heun ODE，从 `t=0` 积分到 `t=1`。

### 冻结组件不等于冻结 DiT 的 Cross-Attention

VAE 和 CLIP 的预训练模型参数固定；但 DiT 内部对图像的 Q 投影、对文本的 K/V 投影等**属于 DiT 的可训练参数**。文本通过 masked Cross-Attention 进入图像 token 主序列。

---

---


<a id="q04"></a>

## Q4｜Plain DiT 内部结构是什么？请用清晰的纯文本画出来。

**原问题**：plain dit里面的架构是什么样子？用纯文本格式画出来

### 总体架构（按 S4-02 Notebook 的接口与数据流整理）

```text
[图像 x: B×3×64×64]                [Caption]
          |                            |
    冻结 VAE Encoder              冻结 CLIP
          |                            |
    z: B×4×8×8                  C: B×L×512
          |                      M: B×L
    ε ~ N(0,I), t ~ U[0,1]           |
          |                            |
    z_t=(1-t)ε+tz                    |
          |                            |
          v                            |
    Patchify(p=2)                     |
          |                            |
     B×16×16                         |
          |                            |
    Linear(16→96)                      |
          |                            |
    + Position Encoding                |
          |                            |
          v                            |
  ┌────────────────────────────────────┘
  │  DiT Block × 2
  │    image Self-Attention (+ residual / time modulation)
  │    text Cross-Attention (Q=image, K/V=text, mask=M)
  │    MLP (+ residual / time modulation)
  └───────────────────────┐
                          v
                    B×16×96
                          |
                     Velocity Head
                          |
                    B×16×16
                          |
                    Unpatchify
                          |
                    v_theta: B×4×8×8

时间另一路：t → ×1000 → 时间嵌入 / MLP → 调制 DiT 子层
训练：v_theta 与 (z−ε) 算 MSE
采样：v_theta 用于 Euler/Heun 积分，再由冻结 VAE 解码
```

单独的时间调制路径：

```text
t: [B] → ×1000 → Sinusoidal Time Embedding → Time MLP
                                           |
                         gamma / beta / residual gates
                                           |
                           注入 DiT 各个对应子层
```

> **精度说明**：S4-02 的文字明确了 Self-Attention、文本 Cross-Attention、时间条件 AdaLN/门控与输出头，且主干来自 S3 的 `ConditionalDiTReference`。上图是便于理解的**逻辑流程图**，不宣称对未随本 Notebook 一并提供的 S3 参考单元逐行核对了每一条残差在代码中的先后实现。

### 张量形状为何这样变化

- VAE：64px 压缩 8 倍空间，`[B,3,64,64] → [B,4,8,8]`。
- Patchify：latent 每块 `2×2`，共 `(8/2)^2=16` 个 patch；每块数值 `4×2×2=16`，所以 `[B,16,16]`。
- Linear：每 patch 由 16 维映射到 hidden=96，形状 `[B,16,96]`。
- 4 头 Self-Attention：每头维度 24；Cross-Attention 使用图像 Query `[B,4,16,24]` 与文本 Key/Value `[B,4,L,24]`，分数形状 `[B,4,16,L]`，文本 mask 在 softmax 前屏蔽无效位置。
- 速度头：每图像 token 输出 16 个数字，`[B,16,16]`，再还原成 `[B,4,8,8]`。这个输出表示**速度张量**，不是图像像素。

### 训练与采样如何共用一个输出

训练：

```text
v_theta(z_t,t,C,M) -> MSE(v_theta, z - ε) -> 更新 DiT 参数
```

采样：

```text
z_0 ~ N(0,I)
    -> 每个 ODE 时间点查询同一个 v_theta(z_t,t,C,M)
    -> Euler / Heun 更新 z_t
    -> z_1 / vae_scale -> 冻结 VAE Decoder -> RGB 图像
```

---


<a id="q06"></a>

## Q6｜相比 S3 的架构，Flow Matching 真正更改的地方在哪里？除了速度头还有吗？

**原问题**：相比S3的架构，flow matching真正更改的地方在哪里（速度头，还有吗）

### 核心回答

**S4-02 没有替换 DiT 的 Transformer 主干，而是复用 S3 `ConditionalDiTReference`，使用全新随机初始化的实例，改变输入路径、时间语义、监督标签和采样算法。**“速度头”在此主要是**输出语义改变**：同一形状的输出 Linear 从学习预测噪声 `ε` 改成学习预测 Flow 速度 `u=z−ε`，没有证据表明另造了不同结构的输出头。

| 部分 | S3 DDPM 噪声预测 | S4-02 Flow Matching |
|---|---|---|
| 冻结组件 | VAE 编码 latent、CLIP token 特征及 mask | 复用 |
| DiT 主干 | Patch Embedding、Self-Attention、Cross-Attention、AdaLN、MLP | 复用 `ConditionalDiTReference` |
| 输入路径 | `sqrt(alpha_bar_t) z + sqrt(1-alpha_bar_t) ε` | `z_t=(1−t)ε+t z` |
| 训练时间 | DDPM 离散步号、噪声日程 | 每样本独立 `t~Uniform[0,1)`；送进旧时间嵌入时乘 `1000` |
| 输出语义 | `epsilon_hat`，与 `ε` 同形 | `v_hat`，与 `z−ε` 同形 |
| 损失 | `MSE(epsilon_hat, ε)` | `MSE(v_hat, z−ε)` |
| 采样 | DDPM/DDIM 类型反向更新 | 从 `t=0` 到 `t=1` 做 Euler/Heun ODE 积分 |
| CFG | 组合有/无条件噪声预测 | 组合有/无条件速度预测，形式仍为 `v_u+s(v_c−v_u)` |

S4-02 中的实际训练代码：

```python
z_t, u = torch.lerp(noise, clean, t[:, None, None, None]), clean - noise
v_pred = model(z_t, 1000.0 * t, text, valid)
loss = (v_pred.float() - u.float()).square().mean()
```

**术语提醒**：Flow Matching 的 `u=dz_t/dt=z−ε` 与一些 DDPM/扩散实现的 **v-prediction 参数化**不是天然同一个量；必须结合具体路径定义。

### 验证范围

上述变化由 S4-02 Notebook 的 `velocity`、`make_pair`、`train_step`、`sample_reference` 和文字说明直接支持；S3 原引用单元没有随本次上传提供，不能声称已逐行比对原类的所有内部细节。

---

---


<a id="q07"></a>

## Q7｜AdaLN-Zero 的 modulation 中，scale 和 shift 是否也需要全零初始化？

**原问题**：modulation部分还需要全0初始化scale和shift吧

### 核心回答

**标准 adaLN-Zero 的做法是：初始化时让调制层输出的 `scale(γ)`、`shift(β)` 与残差门 `gate(g)` 全为零。** 但必须区分“标准设计”和“当前 S3 参考代码已验证的事实”：S4-02 明确说初始残差门为零（Cross-Attention 还有单独零门）和最终输出头为零，并未展示 S3 调制输出层是否也对 `γ/β` 执行零初始化。因此后者仍**需要查看 S3 的 `ConditionalDiTReference` 源码确认**。

对于某个残差子层，写作：

$$X'=X+g(t)\odot F\!\left[(1+\gamma(t))\odot\mathrm{LN}(X)+\beta(t)\right].$$

在初始化时 `γ=0, β=0, g=0`，故归一化输入不被额外缩放和平移，而残差输出被零 gate 关闭：`X'=X`。注意 `γ=0` 对应实际乘数 **`1+γ=1`**，不是把 LayerNorm 特征清零。

一种常见的 adaLN-Zero 写法是调制器最后输出 `6*d` 维（Self-Attention 和 MLP 各一组 shift、scale、gate），并对**最后一层 Linear 的 weight 和 bias 置零**：

```python
self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
nn.init.zeros_(self.adaLN_modulation[-1].weight)
nn.init.zeros_(self.adaLN_modulation[-1].bias)
```

这只是标准结构示意；S3 教学版额外还有 Cross-Attention 的独立 gate，可能采用不同的模块组织。**不能把整个 Time Embedding MLP 的所有参数清零**，否则未必符合预期的梯度流设计。初始输出 Head 若同时为零，首步梯度可能先更新 Head，再逐步传到 Transformer 主干；这不等于训练不进行。

### 查源码应关注的点

1. 生成 `shift/scale/gate` 的最后一层 `Linear` 的 `weight/bias` 是否为零；
2. Cross-Attention 的独立 gate 如何初始化；
3. 输出 velocity head（复用原预测 head）的 `weight/bias` 是否零初始化；
4. 初始化数值测试应分别检查 `γ、β、g`，不能只凭模型最终全零输出就认为所有调制参数都为零。

---

---


# 第二篇｜文本条件、Padding 与 Attention Mask


<a id="q05"></a>

## Q5｜为什么文本特征 C 是 [B,L,512]，Mask M 只有 [B,L]？二者如何配对？

**原问题**：文本特征 `C` + Mask `M` `[B,L,512] + [B,L]`：为什么 M 是两个维度？给我举个具体的数值例子说明二者怎么配对的？

### 直接回答

`C[b,l,:]` 是第 `b` 条文本中、第 `l` 个 token 的完整 **512 维 hidden state**。`M[b,l]` 是同一个位置的一个布尔值，表示该 token **能否作为 Cross-Attention 的 Key/Value 被读取**。这里屏蔽的是整个 token 位置，并非逐一屏蔽 512 个数，因此不需要 `[B,L,512]` 的 mask。

### 具体数值示例（示意数据，非 CLIP 实测数值）

取 `B=2, L=5`，两条文本为 `"a red circle"` 和 `""`（空字符串）。为节省篇幅只展示 512 维中的前三维；实际每一行还有 509 维。

```text
C.shape = [2, 5, 512]
M.shape = [2, 5]

样本0："a red circle"
位置              C[0,l,:] 前三维       M[0,l]
0 BOS             [ 0.2,-0.1, 0.8,...]   True
1 a               [ 0.4, 0.7,-0.2,...]   True
2 red             [ 0.9, 0.3, 0.1,...]   True
3 circle          [-0.5, 0.6, 0.8,...]   True
4 EOS             [ 0.1,-0.4, 0.5,...]   True

样本1：""
位置              C[1,l,:] 前三维       M[1,l]
0 BOS             [ 0.3, 0.5,-0.1,...]   True
1 EOS             [-0.2, 0.7, 0.4,...]   True
2 PAD             [ 0.4,-0.5, 0.2,...]   False
3 PAD             [ 0.1, 0.8,-0.3,...]   False
4 PAD             [-0.6, 0.2, 0.9,...]   False

M = [[1,1,1,1,1],
     [1,1,0,0,0]]
```

注意：`PAD` 的 CLIP hidden state 可能并非全零；因此必须根据显式 mask 判断，而不是根据其特征值猜测有效性。上述 token 分隔只为展示，对应本课四条 Caption 均 5 有效位置、空字符串 2 有效位置的结果。

### Cross-Attention 中如何使用

在本节示例模型里，图像 patch 共 16 个，Attention Head 共 4 个，因此每个 Query 对每个文本 token 的分数为 `[B,4,16,L]`。一个布尔位置 mask `[B,L]` 广播成 `[B,1,1,L]`，对所有 head、所有图像 Query 的同一文本位置生效：

```python
scores = scores.masked_fill(~M[:, None, None, :], float('-inf'))
weights = scores.softmax(dim=-1)
```

例如空文本样本某个 Query 的分数是 `[0.2, 1.3, 0.7, 5.0, 8.0]`，按 `[1,1,0,0,0]` 屏蔽后成为 `[0.2,1.3,-inf,-inf,-inf]`，Softmax 权重约 `[0.25,0.75,0,0,0]`，即第三到第五个位置的 Value 均不参加加权求和。

**记忆**：`C` 记录“每个 token 的内容”（512 维）；`M` 记录“每个 token 位置是否允许被读取”（1 个布尔值）。

---

---


<a id="q11"></a>

## Q11｜CLIP 最大长度 77，但实际 Caption 很短：Padding、Mask 与 CFG 空文本到底如何区分？怎样把 PAD 的注意力分数变为负无穷？

**原问题**：因为token维度为77，但是实际的序列维度到不了这么多啊。这样的话也不是空文本，该如何pad？mask其实是掩码得到空文本用的是吧，pad另有说法是吗？我们该如何让pad的部分都变成-无穷？

### 先区分三件事

1. **77 是 CLIP 文本序列的最大位置数（sequence length），不是 token 特征维度**。本课 token 特征宽度是 512。编码器初始返回 `text_full:[5,77,512]`、`mask_full:[5,77]`。
2. **Padding 是补齐序列**，例如较短 Caption 的有效 token 后补齐 PAD 位置，使所有样本具有相同长度；**valid mask 是 `[B,L]` 的布尔标记**，有效位置为 `True`，PAD 为 `False`。把 PAD 设为 False 只是在注意力计算中禁止读取它，不是把 Caption 变成空文本。
3. **CFG 空文本是主动用 `""` 作为输入，经过冻结 CLIP 编码后的另一条条件表示**。在本课中空字符串仍包含有效的 BOS 与 EOS，对应有效长度 `2`，其余位置是 PAD。`draw_batch()` 以 10% 概率把正常 Caption 的**整行文本特征与整行 mask**替换成空文本特征与 mask，而不是把正常 Caption 的 mask 全设为 False。

### 具体配对（教学示意，词元切分以实际 tokenizer 为准）

```
C.shape = [2, 5, 512]      M.shape = [2, 5]
正常 Caption:  BOS | a   | red | circle | EOS
M[0]        :   1  |  1  |  1  |   1    |  1
空字符串    :  BOS | EOS | PAD |  PAD   | PAD
M[1]        :   1  |  1  |  0  |   0    |  0
```

每个 `C[b,j,:]` 是一个完整的 512 维向量，`M[b,j]` 只控制**该 token 位置整体是否可被 attention 读取**。PAD 对应的 hidden states 不一定是零。

本课先编码为 `[5,77,512]`，然后使用 `token_count=max(lengths)+2=5`，**只截掉整个批次所有样本都无效的尾部**，成为 `[5,5,512]`；仍保留每条文本自己的 `[5,5]` mask。由于 CLIP 编码可能发生在截断之前，不能把这一步与 CLIP 内部编码时的 padding 处理混为一谈。

### 到底把谁设为 `-inf`？

**不是把 PAD 的 token ID 或其 512 维特征改成 `-inf`。** 应在 DiT 的 **Cross-Attention logits（`QKᵀ/√d_h`）上，在 softmax 前**屏蔽无效的 key 位置：

```python
# q: [B, heads, N_image, d_h]
# k: [B, heads, L_text, d_h]
# valid: [B, L_text]，True 表示有效
scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
# scores: [B, heads, N_image, L_text]
scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
weights = scores.softmax(dim=-1)
output = weights @ v
```

例如某图像 query 对五个文本位置的 logits 为 `[0.2,1.3,0.7,5.0,8.0]`，其 mask 为 `[1,1,0,0,0]`，屏蔽后 `[0.2,1.3,-inf,-inf,-inf]`，softmax 得到约 `[0.250,0.750,0,0,0]`。所有图像 query 和 attention head 共享每条文本的有效位标记，靠广播实现。

**边界情况：**如果一条文本全部 mask 为 False，整行 attention logits 全是 `-inf`，普通 softmax 将可能产生 `NaN`；本课空文本保留 BOS/EOS 两个有效位置，避免这个问题。已有的 S4-02 测试还把空文本 PAD 的特征改为 123，输出差异为 0，证明当前实现的跨注意力屏蔽有效。这份 Notebook 没有直接展示 S3 `ConditionalDiTReference` 内部的全部 mask 代码，以上 `masked_fill` 是标准机制的等价示例，而不是未经核对的源代码逐字复现。

**一句话**：`PAD` 负责凑长度，`mask` 负责阻止注意力读取 PAD，`""` 作为真实编码输入提供 CFG 无条件分支；三者不可混淆。

---

---


<a id="q12"></a>

## Q12｜用稍大一点的具体张量数值，说明 `[B,L]` Mask 如何广播到 Cross-Attention，并屏蔽 PAD

**原问题**：「给一个小型数值例子，维度大一点。」（承接 Q11：为什么不直接将 PAD 对应的 `[D]` 个特征设为 `-inf`？）

设 `B=2, L=5, D_text=4, N_image=3, heads=2, d_h=2`。这里把真实 CLIP 的 512 维缩小为 4 维，只为能够逐项手算。

```text
C.shape = [2,5,4]
C[0] = [[1,0,0,1],    # 有效 token 0
        [0,1,1,0],    # 有效 token 1
        [1,1,0,1],    # 有效 token 2
        [9,9,9,9],    # PAD
        [8,8,8,8]]    # PAD
C[1] = [[2,0,1,0],
        [0,1,0,2],
        [1,0,0,1],
        [0,2,1,2],
        [9,9,9,9]]    # PAD

M.shape = [2,5]
M = [[1,1,1,0,0],
     [1,1,1,1,0]]
```

以便于计算的投影为例，令 `K=V=reshape(C)`，按每两个维度拆成两个 head（真实 DiT 使用可学习的 K/V 投影，并不必是恒等映射）。取第一张图、head 0 的三个图像 queries：

```text
Q[0,0] = [[1,1], [1,0], [0,1]]       # [3,2]
K[0,0] = [[1,0], [0,1], [1,1], [9,9], [8,8]]  # [5,2]
```

计算 `S=QKᵀ/√2`：

```text
S[0,0] ≈ [[0.71, 0.71, 1.41, 12.73, 11.31],
          [0.71, 0.00, 0.71,  6.36,  5.66],
          [0.00, 0.71, 0.71,  6.36,  5.66]]   # [3,5]
```

**广播不是在特征 `C` 上填 `-inf`**；而是在上述 logits 上调用 `masked_fill(~M[:,None,None,:], -inf)`。由 `[2,5]` 构造的 mask 形状是 `[2,1,1,5]`，广播到 logits `[2,2,3,5]`，为同一样本的所有 head、所有图像 query 屏蔽相同 PAD 列：

```text
S_masked[0,0] ≈ [[0.71, 0.71, 1.41, -inf, -inf],
                 [0.71, 0.00, 0.71, -inf, -inf],
                 [0.00, 0.71, 0.71, -inf, -inf]]
```

逐行 Softmax，第一个图像 query 的权重约 `[0.248,0.248,0.504,0,0]`。其输出 `Σ_j weight_j * V_j` 为 `[0.752,0.752]`（近似），PAD 位置的 V 即使数值是 9 或 8，权重仍为 0，不会被聚合。第二张图的 mask 为 `[1,1,1,1,0]`，只屏蔽第 5 列。

完整张量轴的关系：

```text
C     [2,5,4]  ── learnable K/V projection, split heads ──> K,V [2,2,5,2]
image ───────────────── learnable Q projection ────────────> Q   [2,2,3,2]
Q @ Kᵀ / √2                                                  S   [2,2,3,5]
M     [2,5]    ── M[:,None,None,:] ──────────────────────────>     [2,1,1,5]
masked_fill(S, ~M, -inf) → softmax(last dim) → A [2,2,3,5]
A @ V → [2,2,3,2] → 合并 heads → [2,3,4]
```

直接做 `C.masked_fill(~M[...,None], -inf)` 虽能利用广播让整个 PAD token 的 D 个数变 `-inf`，却会让后续线性投影 `CW_K`、`CW_V` 产生不合法数值甚至 `NaN`，并非所需的注意力屏蔽方式。目标是让 PAD **被关注的概率为零**，而不是把 PAD hidden states 本身设为负无穷。

---

---


<a id="q13"></a>

## Q13｜为什么 Cross-Attention 的 Mask 只屏蔽列、不屏蔽行？为什么 Q、K 的 token 数量不同？

**原问题**：我还是不懂为什么 mask 只屏蔽列，不屏蔽行。我的问题一直在这里，然后 q 和 k 为什么 token 不一样？

### 核心直觉

Attention 分数矩阵的**行对应 Query（谁在提问）**，**列对应 Key（可以向谁读取信息）**。本课的文本 Cross-Attention 是**图像查文本**：

- `Q` 来自图像 patch 序列，教学模型 `N_image=16`，是有效的图像位置；
- `K,V` 来自 CLIP 文本序列，有效 token 与 PAD 并存，`L_text=5`；
- 两个序列的长度**无须相等**，只需每个 head 的 Q/K 特征维度相同，以便内积。

张量形状：

```text
Q: [B, heads=4, N_image=16, d_h=24]
K: [B, heads=4, L_text=5, d_h=24]
V: [B, heads=4, L_text=5, d_h=24]
S = Q @ K.transpose(-2,-1) / sqrt(24): [B,4,16,5]
valid: [B,5] → valid[:,None,None,:]: [B,1,1,5]
```

因此对 `S` 的**无效 Key 列**写 `-inf`，令所有有效图像 Query 不读取 PAD：

```text
                   text K: BOS   red   EOS   PAD   PAD
image Query 0                 0.2   0.8   0.4  -inf  -inf
image Query 1                 1.1   0.3   0.5  -inf  -inf
image Query 2                 0.6   1.4   0.2  -inf  -inf
```

这些行都对应真实图像 patch，无需屏蔽。**屏蔽某一行**意为该 Query 的输出无效或不应被使用；而文本 PAD 不属于 Q 一侧。若是文本 Self-Attention，Q/K 两侧可能都有 PAD，但常规 Key Padding Mask 仍优先屏蔽列，PAD Query 的输出另行忽略/清零；**不能对整行 logits 直接设全 `-inf` 再做普通 Softmax**，否则可能产生 NaN。

**一句话**：Cross-Attention 中，谁提供 Query 就决定行数，谁提供 Key 就决定列数；本课是有效图像 Query 向文本 Key 提问，所以仅需把无效文本 **Key 的列**屏蔽。

---

---


<a id="q15"></a>

## Q15｜“双向 Self-Attention 不用 Mask，Cross-Attention 要 Mask 且只遮列”，这种理解对吗？

**原问题**：所以说self-attn的时候不用mask（因为是双向注意力）；cross-attn的时候就要用mask了，只掩码列部分。

### 准确回答

**在本课 S4-02 的 DiT 中，这样描述“实际使用效果”基本成立；但理由应改为“是否存在 PAD，以及信息是否允许被读取”，而不是“Self/Cross 的名称”。**

- **图像 Self-Attention**：本课图像 patch 有 `N=16` 个真实位置，没有图像 PAD，又允许所有图像 patch 双向交互；所以不需要 *padding mask* 或 *causal mask*。`Q/K/V` 都来自图像，Attention scores 为 `[B,H,16,16]`。
- **图像→文本 Cross-Attention**：`Q` 来自图像 `[B,H,16,d_h]`；`K/V` 来自文本 `[B,H,L,d_h]`。文本中有 PAD，所以需要 *key padding mask*，在 `[B,H,16,L]` 的 logits 上仅屏蔽**无效文本 Key 对应的列**。每一行仍然是有效图像 Query，不需要屏蔽。
- **文本双向 Self-Attention（例如 BERT）**：即使是双向的，如果输入中含 PAD，依然需要屏蔽 PAD **Key 列**。这里的“双向”只意味着不做禁止看未来的三角形 causal mask，**不等于不用任何 Mask**。
- **文本因果 Self-Attention（例如 GPT）**：通常还要用 *causal mask*，禁止当前位置读取未来 Key；如果另有 Padding，还要同时处理 padding mask。

### 为什么不随便屏蔽行？

在 `scores = QKᵀ` 中：**行 = Query（谁在提问），列 = Key（向谁提问）**。如果 Query 一侧也有无效位置，可以在输出端忽略/置零其结果；但**不能简单给某行所有分数都填 `-inf` 后照常做 Softmax**，否则可能产生 `NaN`。

本课图像 Query 全有效，而文本 K/V 可能含 PAD；故仅屏蔽 Key 列，计算：

```python
# scores [B,heads,16,L], text_valid [B,L]
scores = scores.masked_fill(~text_valid[:, None, None, :], float('-inf'))
weights = scores.softmax(dim=-1)
# 每个图像 Query 对无效文本位置的权重 = 0
```

**最终记忆**：Attention Mask 由**可见性规则**和**Token 有效性**决定。`Self` 不等于不用 Mask，`Cross` 不等于一定要 Mask；本课之所以“图像 Self 不用、图文 Cross 遮列”，是**具体输入特性**造成的。

---


# 第三篇｜Flow 时间、速度与预测误差


<a id="q03"></a>

## Q3｜时间 t 怎么采样？是从 [0,1] 里采吗？

**原问题**：时间t是怎么采样的？是从[0, 1]里面采样吗？

### 核心回答

是。训练 `draw_batch` 中实际写法：

```python
t = torch.rand(batch_size, generator=rng)
noise = torch.randn(clean.shape, generator=rng)
z_t = torch.lerp(noise, clean, t[:, None, None, None])
u = clean - noise
```

这对应对 batch 中每张图独立采 `t ~ Uniform[0,1)`（连续理论中通常记作 `U[0,1]`），并为每张图采独立高斯噪声。时间必须广播成 `[B,1,1,1]` 才能作用到 `[B,4,8,8]` 上。

$$z_t = (1-t)\epsilon + tz = \epsilon + t(z-\epsilon), \qquad u = \frac{d z_t}{dt} = z-\epsilon.$$

- `t=0`：全是高斯噪声 `ε`；`t=1`：到达干净 latent `z`（实际训练随机抽样一般不恰好抽到 1；Notebook 另外明确测试了两个端点）。
- `t` 进训练主干前，Notebook 包装函数 `velocity(...)` 会传 `1000*t` 给旧 S3 正弦时间嵌入：**是特征刻度兼容，不是 Flow 时间范围改变**。
- 采样阶段**不随机重抽 t**，而由 ODE 步数确定时间网格 `t_i=i/N`，Euler 用单次场求值，Heun 每步两次；从 0 推进到 1。

---

---


<a id="q10"></a>

## Q10｜为什么时间 t 会影响速度 MSE？为什么当前 Right MSE 从 0.2 到 0.8 反而增加？

**原问题**：t的变化对于mse的关系，其实我没看懂到底为什么、如何分析。为什么t会影响mse的值？特别是在right下，反而呈现正向关系

### 必须区分「输入看起来干净」与「目标速度容易预测」

$$z_t=(1-t)\epsilon+tz,\qquad u=z-\epsilon.$$

- `t` 较小：`z_t` 更接近噪声 `ε`，因此噪声一端相对容易估计；但要预测目标 `u`，还需要了解图像端点 `z`，Caption 可以帮助推断其类别与统计分布。
- `t` 较大：`z_t` 更接近干净 latent `z`，但目标速度仍包含 **`−ε`**。`z_t` 很干净不等于 `ε` 已知；例如接近 `t=1` 时，`z_t≈z`，噪声端点反而更加不可辨认。
- 端点直觉：`t=0` 时 `z_t=ε`，不知道的是 `z`；`t=1` 时 `z_t=z`，不知道的是 `ε`。哪一端容易，取决于条件 Caption 对 `z` 的约束、latent 分布和模型能力，因此**不存在对所有模型/数据都单调的 t–MSE 定律**。

从 MSE 的条件期望最优预测可写出误差分解：

$$\mathbb E[(v_\theta-u)^2\mid t]=\mathbb E[(v_\theta-\mathbb E[u\mid z_t,t,C])^2\mid t]+\mathbb E[\operatorname{Var}(u\mid z_t,t,C)\mid t].$$

第一项反映模型没有学到最优条件速度的误差；第二项是即使知道 `z_t,t,C`，对随机端点仍存在的条件不确定性。这两者都可能随时间变化。

### 这份 Notebook 的实际证据与局限

| 时间 | Right MSE | Wrong Color MSE | Zero baseline | Gaussian moment baseline |
|---|---:|---:|---:|---:|
| 0.2 | 0.71936 | 2.34237 | 3.08640 | 1.29830 |
| 0.5 | 0.80130 | 2.61921 | 3.07132 | 1.64083 |
| 0.8 | 1.12589 | 2.76046 | 3.10479 | 1.45548 |

本次 Right 在**三个取样时间点**确实递增，但这不能外推为所有 t 的单调定律。更重要的是，`val_noise = randn((3,*val_z.shape))`，而验证循环 `for t, eps in zip(val_times,val_noise)` 为每个 t 使用**不同的固定噪声张量**。因此当前观测混合了时间变化与噪声样本变化，尽管每次重复验证时可复现。

如果要更干净地研究 t 的影响，应让相同验证图像和**同一组噪声 `eps=val_noise[0]`**贯穿全部时间点，再比较 Right MSE；这样每一组端点和目标 `u=z−ε` 都相同，只改变 `t` 与对应的 `z_t`。

```python
# 在已训练的 model、val_z、val_y、val_noise、text_bank、mask_bank 下执行
same_eps = val_noise[0]
model.eval()
with torch.no_grad():
    for t0 in (0.2, 0.5, 0.8):
        t_batch = torch.full((len(val_z),), t0)
        zt, u = make_pair(val_z, same_eps, t_batch)
        pred = velocity(model, zt, t_batch, text_bank[val_y], mask_bank[val_y])
        print(t0, mse_reference(pred, u).item())
model.train()
```

**结论**：本次数据表明当前模型在 t=0.8 的正确条件预测更困难（或当前样本噪声造成更高误差）；它既不意味着 `t` 越大生成质量越差，也不能只用“越接近干净图像”推导 MSE 越小。最终需结合更密的时间分桶、固定配对噪声以及 ODE 生成图像判断。

---

---


# 第四篇｜条件评测与无条件生成


<a id="q08"></a>

## Q8｜什么是 Wrong Color MSE？为什么要比较正确/错误 Caption？MSE 越来越高代表模型越好吗？

**原问题**：什么叫错误颜色mse？为什么要对比正确颜色和错误颜色mse？这个结果说明了什么？为什么会慢？错误颜色mse其实是验证集是吗？也就是说验证中如果用错误颜色caption，就会得到越来越高的mse，也就变相证明模型越来越好，能区分正确和错误的caption（因为毕竟是image和caption同时输入，所以就能得到二者匹配程度）

### 在 Notebook 中如何计算

`validate()` 使用 **32 张未见过几何参数的验证图像**（`val_z`）及固定验证随机噪声，对各时间 `t=0.2,0.5,0.8` 构造同一个 `z_t` 和真实速度标签 `u=z−ε`。随后**只更换 Caption**，分别用 `val_y`（正确）、`val_y ^ 1`（翻转红/蓝）、`val_y ^ 2`（翻转圆/方）、`4`（空文本）调用同一个 `velocity(model,z_t,t,text,mask)`。因此四种 MSE 使用同一组输入状态和目标速度，可做受控配对比较。

```text
一张蓝色方形验证图 z，固定同一个 ε 与 t
          ↓
   z_t=(1-t)ε+t z，目标 u=z−ε
          ├── 条件 C="a blue square" → v_right → MSE(v_right,u)
          └── 条件 C="a red square"  → v_wrong → MSE(v_wrong,u)
```

Notebook 2400 步结束时（对三个验证时间和 32 张图取平均）：Right≈`0.8822`；Wrong Color≈`2.5740`；Wrong Shape≈`0.9516`；Empty≈`0.9288`。错误颜色与正确颜色的间隔约 `1.692`，大于错误形状的间隔约 `0.069`。这有力支持**模型已经把颜色 Caption 用于速度预测**，但形状条件效果相对弱。

**不能理解成模型直接输出了“图片与文本匹配分数”。** DiT 输入的是带噪 latent、时间和 Caption，输出的是速度 Tensor；MSE 来自预测速度与**只有验证时才知道的真实标签**比较。该误差差距是*间接的条件利用证据*，不能单独保证生成质量或未见文本组合的泛化。Wrong Color MSE 越来越高不是训练的直接优化目标；只有在 Right MSE 低且配对对照可靠时，较大差距才有积极意义。若 Right 和 Wrong 都很高，不能说模型很好。

此外，`model.eval()` 仅切换评估模式，`@torch.no_grad()` 不计算梯度；**验证没有用错误 Caption 来更新参数**，高 Wrong MSE 不是训练失败。

---

---


<a id="q09"></a>

## Q9｜无条件模型究竟学到了什么？能生成正确图片但不一定是想要的图片吗？

**原问题**：无条件模型学习到了什么？学习到了生成正确的图片是吗？（而不一定是我们想要的）

**基本方向对，但“正确图片”要限定为训练分布里的合理样本。** 训练 `draw_batch()` 有 10% 概率将 Caption 换成空字符串的 CLIP 表示，仍提供 `z_t` 和 `t`，并使用同一 `u=z−ε` 标签训练。空条件分支学习

$$v_\theta(z_t,t,C_\varnothing)\approx\mathbb E[z-\epsilon\mid z_t,t,C_\varnothing].$$

它学习的是**不指定颜色/形状时，混合的图标 latent 分布及其速度场**；有条件分支学习指定文本条件下的速度场。无条件采样有可能生成红圆、蓝圆、红方或蓝方等训练域中的图形，但不会确保满足用户给出的具体文本意图。是否能生成清晰、合理的图片，仍需要进行 ODE 采样并查看结果，**验证速度 MSE 不能证明生成质量**。

`C_\varnothing` 不是全零向量，而是由空字符串经过冻结 CLIP 编码得到的 BOS/EOS token 表示与对应 mask。CFG 在采样时利用两路速度的差异：

$$v_{\mathrm{CFG}}=v_{\varnothing}+s(v_C-v_{\varnothing}).$$

本课的空文本验证 MSE≈`0.9288`，并非全零或毫无能力，体现无条件分支也能从 `z_t,t` 和总体分布获得预测信息。

---

---


# 第五篇｜ODE 采样与图片网格读法


<a id="q14"></a>

## Q14｜S4-02 的 4×16 采样图片网格：横排、竖排分别是什么？

**原问题**：图片表示的是什么？横排竖排都是啥？

这张网格共 `4 × 16 = 64` 个展示结果：

- **4 行（竖排）固定 Caption**，自上而下：`a red circle`、`a blue circle`、`a red square`、`a blue square`。
- **16 列（横排）分成 4 个采样方案，每组 4 个种子**，自左而右：`Empty`（无条件 Euler 20 步）、`Euler`（有条件 Euler 20 步）、`Heun`（有条件 Heun 10 步）、`Euler CFG 2`（引导强度 s=2 的 Euler 10 步）。每组内种子重复使用 `40206、40216、40226、40236`。
- 同一个 seed 在不同行/方案对应**相同初始高斯 latent**，方便把初始随机性作为控制变量；同一列不同 Caption 可以观察颜色和形状的条件响应。
- `Empty` 列的同一个种子跨 4 行应显示**完全相同的图**，因为全部替换为空 Caption；这不是错误。

实验输出（16 个结果/采样方案，规则解析失败计入分母）：

| 方案 | 颜色正确 | 形状正确 | 两者同时正确 | 解析失败 |
| --- | ---: | ---: | ---: | ---: |
| Empty | 6/16 | 6/16 | 3/16 | 4/16 |
| Euler | 16/16 | 8/16 | 8/16 | 0/16 |
| Heun | 16/16 | 8/16 | 8/16 | 0/16 |
| Euler CFG 2 | 15/16 | 9/16 | 9/16 | 1/16 |

**解读**：颜色条件明显有效；方形经常变圆润/不规则。`Heun` 与 `Euler` 的总分支等价求值预算均为 20，但积分步数不同。`CFG 2` 与普通 Euler 也同时改变了积分步数，不能把有限样本中的差异全部归因于 CFG。本节网格是固定种子的定性对照，不是独立的 64 次随机评测。

---


# 复习清单与剩余边界

1. **已理解 / 已在 Notebook 中有证据**：合成数据与冻结接口、图文 Cross-Attention 的 Q/K 轴、Key Padding Mask、速度 MSE、条件对照、ODE/CFG 采样的参考闭环、单进程 CPU 恢复检查。
2. **个人实践未完成**：`flow_loss(predicted,target)` 和 `sample_ode(field,initial,steps)`，Notebook 的实际输出为 `待完成`。应分别通过数值、梯度、shape/FP32、时间调用点、NFE、状态依赖场等测试后再勾选。
3. **数学深究暂缓**：S4-01 中的连续性方程、边际速度为何对应目标分布、ODE 误差与理论证明；已有直觉不等于完成推导。
4. **实验结论边界**：随机训练 1 个训练种子、每个提示 4 个固定采样种子；不把规则评分、训练 MSE 或单次可复现测试直接当作正式 256px 的性能证明。
5. **下一阶段**：可以先进入 S5-01（Joint Attention 和 2D RoPE），之后再补 S4-01 数学与这两个个人函数。

## 资料来源

- `S4_02_LatentFlow与PlainDiT训练闭环(1).ipynb`（当前提供的 Notebook）
- 同日 MiniGen S4-02 学习对话的 **15 条用户技术提问**
- 既有 `MiniGen_S4-02_QA.md`（Q1–Q14 内容与题号保留，本版补 Q15 并按主题重排）

> **特别说明**：文中的小型 Q/K/V、Mask 数值样例是教学计算示例，不是 CLIP 实测向量；Notebook 的输出数值则明确标注为实验结果。对于 S3 源码没有直接展示的调制层初始化细节，仅给出标准做法，不冒充已经完成的源码核验。
