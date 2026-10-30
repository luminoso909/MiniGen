# MiniGen · 第一阶段知识问答（扩充版）

> 保留你原来的问答结构，不替换已有核心公式；每问增加适量背景、数值例子、易错提醒和必要扩展。优先掌握 **直觉 → 代码 → 小数值例子 → 数学推导**。

## 阅读顺序

1. [DDPM 前向扩散八问](MiniGen_DDPM_前向扩散八问.md)：alpha/beta/alpha_bar、高斯合并、先验。
2. [SNR 与反向后验三问](MiniGen_DDPM_信噪比与反向后验三问.md)：正向/反向区别、贝叶斯后验数值示例。
3. [Scheduler 代码十一问](MiniGen_DDPM_Scheduler代码十一问.md)：查表、广播、训练/生成流程。
4. [U-Net 结构与时间条件十一问](MiniGen_UNet_结构与时间条件十一问.md)：TimeEmbedding、GroupNorm、Skip 与上下采样。
5. [变分界与噪声预测损失问答](MiniGen_DDPM_变分界与噪声预测损失问答.md)：作为进阶阅读；当前若概率基础薄弱，可先跳过。

## 复习方法

- 第一遍：只看原问答中的粗体答案及每道题的数值示例。
- 第二遍：不看答案，自己写出 `xt = add_noise(x0, eps, t)` 与 `eps_pred = model(xt,t)` 的数据流。
- 第三遍：必要时补贝叶斯后验与 VLB；暂时不会推导不影响先完成 DDPM 训练闭环。

## 统一术语

- $t$：数学上的扩散时间步；代码通常从 0 索引，`t=0` 对应第 1 次加噪。
- $\beta_t$：单步新增噪声方差；$\alpha_t=1-\beta_t$；$\bar\alpha_t=\prod_{s=1}^t\alpha_s$。
- $\epsilon$：训练时本次采样加入的真实噪声；$\epsilon_\theta$：网络预测；$z$：反向采样新抽取的随机噪声。
- **Schedule** 是时间系数方案；**Scheduler** 执行加噪/采样规则；**U-Net** 是本阶段真正学习的神经网络。
