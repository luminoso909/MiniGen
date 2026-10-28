## **一、L01 / S1-03 个人 DDPM 验收**

执行日期：2026 年 10 月 28 日。依赖：S1-02 已有训练工程。用户完成 DDPM 核心主体，Agent 检查数学、边界、梯度与工程衔接；没有复制 Notebook 参考到 src，也没有 fallback。

状态：**核心代码已验证**；S1-03 的个人推导与理解仍待验收，因此该阶段任务整体保留进行中。S1-04 本次只有 CPU 小模型工程冒烟，没有重新训练 MNIST，不能据此登记个人模型生成质量。其余 VAE、DiT、CFG、Flow 核心仍待用户完成。

源码：[ddpm_scheduler.py](../../src/minigen/objectives/ddpm_scheduler.py)。独立验收：[check_ddpm_core.py](../../scripts/check_ddpm_core.py)。回归测试：[test_ddpm_core.py](../../tests/test_ddpm_core.py)。结构化证据：[ddpm_core_results.json](../stage1/ddpm_core_results.json)。

### **1. 数学审查结论**

对照本地固定版本 [DDPM v2](../../docs/papers/DDPM_Denoising_Diffusion_Probabilistic_Models_2020_v2.pdf) 的式 (4)、(7)、(11)、(14) 与采样约定，当前实现采用噪声预测、简化平均 MSE 和固定后验方差，不学习方差。

| 检查点 | 结论与证据 |
| --- | --- |
| 时间编号 | 代码索引 0 对应论文第一次加噪；累计系数包含当前步，未发生偏移 |
| 前向加噪 | 三步手算累计系数 0.9、0.72、0.504，与混合 batch 时间广播结果一致 |
| 后验 | 用两个高斯的精度相加独立求解；均值、方差与实现一致 |
| 反向均值 | 用论文式 (11) 的直接噪声参数化对照，独立于实现中的干净图像估计与后验组合 |
| 随机项 | 非最终步使用后验标准差乘调用方噪声；最终索引 0 不再添加随机项 |
| 损失与梯度 | 所有像素取均值；重复 batch 不改变损失；解析梯度和双精度 gradcheck 通过 |
| 裁剪 | 只裁剪估计的干净图像，输入不原地修改；默认关闭适合后续潜变量接口 |
| 缓冲区 | 系数用 float64 构造、float32 保存；无可训练参数，可迁移 dtype 并通过 state_dict 重载 |

没有发现核心公式错误。补充了两处整数参数边界校验：num_steps 必须为正整数，step 的 timestep 必须为整数，均拒绝 bool。原先 `step(..., timestep=1.5, ...)` 会在 torch.full 中被静默截断，现在明确抛出 ValueError；核心公式、默认值与合法输入路径保持不变。

### **2. 真实数值与工程结果**

环境：Python 3.12.14、PyTorch 2.14.1、pytest 9.1.1、jaxtyping 0.3.11；macOS arm64，实际验收设备为 CPU。当前进程的 MPS/CUDA 均显示 unavailable，没有做加速设备验收。无收费资源、无新增数据或权重下载。

种子：经验分布使用独立 CPU Generator 的 20261028；工程模型初始化为 7，训练器默认种子为 20261026，采样初始噪声为 11、反向随机项为 123。工程测试输出在 pytest 隔离临时目录；无项目 Checkpoint 纳入 Git。

| 数值项 | 实测结果 |
| --- | --- |
| 手算前向最大绝对误差 | 1.1920928955078125e-7，门槛 3e-7 |
| 高斯精度后验均值 / 方差最大误差 | 0 / 0 |
| 直接噪声参数化反向均值 / 指定随机项最大误差 | 0 / 0 |
| 每个时间级别经验样本数 | 32768 |
| 经验均值 / 方差最大绝对误差 | 0.0038613081 / 0.0038020015，均低于各自理论标准误的六倍 |
| 手算 MSE / 解析梯度最大误差 | 0.43500000000000005 / 0 |

工程测试显式传入 src 中的 DDPMScheduler，并记录 scheduler_source=project_core。使用相同 8 步日程，连续训练 6 次更新与先训练 3 次、恢复到第 6 次的模型参数和 EMA 逐位一致，真实参数发生更新；完整反向链执行 8 步、8 次模型求值，结果有限、形状正确，恢复调用前的模型训练模式。

这里的模型是测试用 1×1 卷积，输入是固定小张量；结果只验证接口与恢复链路，不能作为数字生成效果。随机 loader 的精确数据位置依然由 StatefulBatchStream 的独立验收负责。

### **3. 执行命令与检查结果**

本轮保持已安装的 Notebook、latent 可选环境，实际运行：

```bash
uv run --locked --extra notebook --extra latent python scripts/check_ddpm_core.py
uv run --locked --extra notebook --extra latent python scripts/check_code_annotations.py
uv run --locked --extra notebook --extra latent pytest -q
uv run --locked --extra notebook --extra latent minigen doctor --device cpu --json
git diff --check
```

- 完整 pytest：**72 项通过**。原有两项把 DDPM 当作待实现的断言已更新；另外四项个人核心的失败边界仍被测试保护。
- 将暂存区源码、测试与已提交版本的 Notebook 导出到隔离目录，显式使用该目录的 src 复验：72 项通过，未依赖用户未暂存的 Notebook 改动。
- 接口结构检查：44 个 Python 文件、165 个函数/方法、19 个类，零问题。维度简称、多行函数签名与条件式文档沿用最新规范。
- 显式枚举 src、scripts、tests 下全部 44 个 .py 路径，分别执行 Ruff check 与 Ruff format --check，均通过。这包含默认目录排除可能漏掉的 src/minigen/data。
- CPU doctor：基础包导入、小张量前向与反向通过。
- 除 DDPM 实现、文档检查器及相关测试外，其余 37 个原有 Python 文件相对 HEAD 去除注解、文档与类型导入后，计算与控制流程 AST 一致。
- 本轮开始时登记的 14 个 Notebook/问答文件哈希全部保持一致；没有执行 Notebook。
- 全局 `ruff check .` 仍报告第一周用户 Notebook 中的 5 项原有问题；全局格式检查报告 7 个学习文件需格式化。未修改这些文件，也未放宽规则。以上 Python 检查通过不等于全局检查通过。

### **4. 仍需保留的边界与下一步**

1. 当前训练 Checkpoint 不保存或校验调度器的完整超参数。恢复必须显式重建相同 num_steps、beta 起止值和 clip_denoised；本次只验证相同日程恢复，不保证换日程后继续训练的等价性。模块 state_dict 保存系数，不包含普通 Python 属性的完整配置。
2. Notebook 仍显式使用教学参考。个人核心通过数学与短工程验收，不能自动替代此前教学证据或标记生成质量已达标；CLI train/sample/evaluate/cache 仍未实现。
3. 下一步：脱离代码解释索引 0、累计噪声、后验条件与最终步，再完成 L02 的重参数化和 KL 个人实现。S1-07 与整个 S1 未完成。

本地提交包含用户 DDPM 实现、本次边界修正、验收与状态更新，以及此前已完成但尚未提交的统一注释格式及检查器。第一周 Notebook 与 week1QA 的用户改动保留在工作区，不暂存；不创建远程、不推送。
