## **一、L03：把空间潜变量变成可调制的 Transformer**

任务对应 S2-04。先读 [第三周 Notebook](../docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb) 第八至十章，再合上参考实现 `docs/learning/src/minigen/models/dit_block.py`。Agent 不替写主体；教学参考只存在于 Notebook。

### **1. 实现顺序与输入输出**

1. `patchify(latents, patch_size)`：输入 `[B,C,H,W]`，输出 `[B,(H/p)*(W/p),p*p*C]`。token 顺序先行后列，patch 内先行、列再通道。高度或宽度不能整除时抛出 ValueError。
2. `unpatchify(tokens, channels, height, width, patch_size)`：精确逆排列，验证 token 数和宽度。
3. 时间编码：先在自己的草稿中实现正弦余弦和 MLP，解释位置坐标与噪声时间的区别；本文件的 Block 接收已编码的 `[B,D]` 条件。
4. `DiTBlock(hidden_size, num_heads)`：前向输入 tokens `[B,N,D]`、condition `[B,D]`。使用无固定 affine 的 LayerNorm、注意力、MLP、六组条件调制和两条门控残差。
5. 初始化条件投影末层为零；不要把所有线性层设为零。初始输出精确等于输入，门控可获梯度、分支可暂时零梯度。

### **2. 先手算，再写代码**

对 `arange(16).reshape(1,2,2,4)`、patch 大小 2，第一块应为 `[0,8,1,9,4,12,5,13]`，第二块为 `[2,10,3,11,6,14,7,15]`。这项对照能发现两个错误往返函数互相抵消的问题。

$$n=r(W/p)+s,\qquad k=(ip+j)C+c$$

调制关系为：

$$Y=X+g(c)\odot f\big((1+\gamma(c))\odot\operatorname{LN}(X)+\beta(c)\big)$$

解释每个条件向量怎样广播到 token 维度，以及为什么门控为零时分支参数梯度可能为零。

## **二、验收与交接**

### **1. 运行个人核心验收**

```bash
uv run --locked python scripts/check_dit_core.py
```

只调用项目接口；当前未实现时输出“待做”并返回 2。不会从 Notebook 导入答案。实现后检查空间手排顺序、往返、异常尺寸、恒等映射、输入梯度、门控学习与条件作用。

### **2. 你需要独立讲清楚的内容**

- 从 `[B,4,16,16]` 到 `[B,64,256]` 再回到噪声张量的完整数据流。
- 为什么切 patch 可逆，但投影到隐藏维度不要求可逆。
- 注意力矩阵大小、patch 改变和实际硬件耗时之间的关系。
- 零输出头、零门控与普通参数的梯度为什么分阶段出现。
- 固定监督拟合为什么不足以证明随机时间训练和生成能力。

个人核心通过后，Agent 再连接第四周训练器。S1 的 DDPM 与 VAE 个人验收仍需完成；不能把教学 Notebook 的成功当作依赖已满足。
