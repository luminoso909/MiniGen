# MiniGen · 实验汇总与精确断点续训（代码复习）

> 范围：Notebook 收尾的两段代码。它们是**实验管理与可靠性测试**，不是新的 DDPM 噪声公式或训练目标。

## 1. 第一段 `summary` → `notebook_results.json`

**目的**：把之前已经算好的数据、数学检查、固定过拟合、随机训练、EMA 评测和生成结果收集成一个可复查的 JSON 文件。它不重新训练或采样。

### 保存哪些信息？

| 字段 | 意思 | 注意 |
|---|---|---|
| `scheduler_source` | Scheduler 来源 | `notebook_reference` 表示使用参考实现 |
| `src_core_implemented` | 是否完成个人核心代码 | 这里的 `False` 是**手写标记**，不会自动检测 |
| `seed`, `device`, `torch_version` | 环境和随机性 | 便于重跑、排查差异 |
| `data`, `data_resources` | 训练数据数量、哈希、官方测试集使用情况 | `manifest` 是数据说明与校验记录，不是图像本身 |
| `math_checks`, `oracle_max_error` | 先前数学与数值对照检查 | 此处只是读取先前结果，不重新执行检查 |
| `fixed_pair` | 固定图片、固定时间、固定噪声上的拟合检查 | 低损失不能证明真实生成质量 |
| `training` | 随机时间与噪声训练后的指标 | `initial/final/ema_probe_loss` 是同一验证探针的比较 |
| `sampling` | 是否完成生成、步数、模型调用次数等 | 采样完成不代表生成质量必然好 |
| `config` | 训练配置 | 用于复现 |
| `limitations` | 结果适用范围和局限 | 8 张样本、64 张验证探针、单次随机种子等 |

### Python 技巧：字典推导式

```python
picked = {k: training_result[k] for k in ["final_step", "final_probe_loss"]}
```

表示从大字典中**仅提取指定字段**。

### 写文件

```python
path = OUT / "notebook_results.json"
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
```

- `json.dumps`：Python 字典 → JSON 字符串。
- `indent=2`：两个空格缩进。
- `ensure_ascii=False`：可直接保留中文。
- `write_text`：写到磁盘。

**例**：后续将 `generated_samples.png` 与 `notebook_results.json` 一起存档，就能知道「这些图片来自哪次训练、哪种 Scheduler、哪个随机种子」，而不只剩一张无法复现的图片。

---

## 2. 第二段：测试中断后恢复训练能否与不中断完全一致

### 核心思想

比较两条路径：

```text
A. start_model ──训练 6 步──────────────→ full_model

B. start_model ──训练 3 步─保存 checkpoint
                                  └──重新创建模型并加载─再训练 3 步→ resumed_model
```

希望 A 和 B 最终**普通权重相同，EMA 权重也相同**。

### 为什么要有 `MinimalDenoiser`？

```python
class MinimalDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, image, times):
        return self.conv(image)
```

- 这是一个用于**测试训练器是否能恢复状态**的极简神经网络，不是有生成能力的完整 DDPM U-Net。
- `1×1 Conv2d(1,1,1)` 有权重和偏置，适合快速检查优化器是否正确更新和恢复。
- `times` 参数保留接口，但**不使用时间条件**。
- `mini_loader=[(torch.zeros(2,1,4,4),)]`：模拟只有一个 batch 的 DataLoader；`(tensor,)` 是单元素元组，无需标签。

### 分三次运行

```python
start_model = MinimalDenoiser()

# A：同一初值，连续跑满 6 步
full_model = copy.deepcopy(start_model)
full_result = train_denoiser(full_model, ..., max_steps=6)

# B1：同一初值，先跑 3 步并保存
partial_result = train_denoiser(copy.deepcopy(start_model), ..., max_steps=3)

# B2：新建模型，从 B1 的 checkpoint 恢复，跑到总计 6 步
resumed_model = MinimalDenoiser()
resumed_result = train_denoiser(
    resumed_model, ..., max_steps=6,
    resume_path=partial_result["checkpoint"]
)
```

**注意**：B2 的 `max_steps=6` 应理解为恢复后的**累计训练步数目标**，不是再增加六步；这一解释以训练器实现的恢复语义为前提。

`copy.deepcopy(start_model)` 确保 A、B 起点参数完全一致；仅仅重新设随机种子并不总能保证同一初始化。

### 为什么不能只保存模型权重？

希望精确恢复，通常还需要保存：

1. U-Net 的**当前参数**；
2. Optimizer 的内部状态，例如 AdamW 的动量和二阶矩估计；
3. 已完成的训练步数；
4. EMA 的平滑权重；
5. 必要的随机数生成器状态；
6. 若想覆盖任意数据加载位置，还需要恢复 DataLoader / Sampler 的顺序与位置。

否则，同样从第 4 步继续，使用的优化器状态或随机数可能已经不同，后续训练轨迹就会变化。

### 如何验证？

```python
for a, b in zip(full_model.parameters(), resumed_model.parameters()):
    torch.testing.assert_close(a, b, rtol=0, atol=0)

for key in full_saved["ema"]:
    torch.testing.assert_close(
        full_saved["ema"][key], resumed_saved["ema"][key],
        rtol=0, atol=0
    )
```

`rtol=0, atol=0`：**不允许数值误差**。如果任何比较失败，就抛出异常，后面的成功标记不会执行。

这些断言说明：**在本次极简固定数据测试中，完整训练和恢复训练得到相同的最终参数与 EMA 状态**。不能据此推断真实 `shuffle=True` 的 DataLoader 在任意 epoch 中间也能精准恢复。

### 最后的 `summary` 更新

```python
summary["cpu_fixed_loader_resume_exact"] = True
summary["experiment_elapsed_seconds"] = time.monotonic() - experiment_started
(OUT / "notebook_results.json").write_text(...)
```

在断言成功后给实验记录追加测试通过标记，并**重新写入同一份 JSON 文件**。

注意：字段名写了 `cpu`，但展示的 `TrainConfig` 调用没有显式指定 `device="cpu"`，是否确实在 CPU 运行应检查 `TrainConfig` 的默认值。

---

## 3. 这个测试不证明什么？

- 不证明 DDPM 数学公式正确（那由前面的数学检查负责）。
- 不证明 MNIST 生成效果好（要看生成样本及合理指标）。
- 不证明任何随机打乱 DataLoader 都能在 epoch 中途精确恢复（该测试只有一个固定 batch）。
- 不证明跨设备、跨 PyTorch 版本都能逐位一致。

## 4. 对当前 MiniGen 学习的要求

**需要掌握**：`summary` 是结果归档；Checkpoint 是训练状态快照；`resume_path` 继续之前的优化过程；完整训练与断点续训应能得到一致参数。

**可以暂缓**：自写 Checkpoint 序列化、AdamW 状态恢复、DataLoader shuffle 的精确回放、复杂日志系统。

**一句话总结**：第一段解决「结果能否追溯」，第二段解决「训练中断后能否可靠地接着跑」。
