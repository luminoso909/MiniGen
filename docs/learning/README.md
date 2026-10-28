## **一、学习笔记导航**

所有知识学习类 Notebook 和笔记放在本目录；根目录 learning/ 仅保留任务入口。标题独立为 Markdown 单元，代码使用代码单元，交付保留真实输出。

| 阶段 | 学习笔记 | 内容与边界 |
| --- | --- | --- |
| S1 第一周 | [DDPM 原理与 MNIST 实验](S1_第1周_DDPM原理与MNIST实验.ipynb) | 全流程、概率、前向与后验推导、变分目标、教学核心、真实短训与采样；src DDPM 核心已独立验证，理解另验 |

| S1 第二周 | [VAE 原理与潜空间实验](S1_第2周_VAE原理与潜空间实验.ipynb) | 两种 ELBO 推导、KL 与梯度、四组真实短训、潜空间可视化和恢复；个人核心待做 |

论文与分层阅读任务见 [论文目录](../papers/README.md)；下一项亲手实现见 [L02 任务包](../../learning/L02_VAE核心任务.md)。

## **二、打开与重跑**

在 MiniGen 根目录使用项目环境：

```bash
uv sync --locked --extra notebook
uv run --locked --extra notebook jupyter lab docs/learning
```

选择项目环境的 Python 内核，并在笔记首个代码单元确认版本和项目根目录。也可从干净内核执行整份笔记并生成 HTML：

```bash
uv run --locked --extra notebook python scripts/execute_learning.py docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb
```

重跑会重新执行本地教学训练并更新输出，整个训练与采样预算约 30 分钟，不会调用云 GPU 或模型 API。脚本使用 .cache 下的本地 kernel spec，不注册全局内核。HTML 位于 artifacts/runs/s1_week1/learning_preview.html。

第二周使用独立 HTML 输出路径：

```bash
uv run --locked --extra notebook python scripts/execute_learning.py docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb --html-output artifacts/runs/s1_week2/learning_preview.html
```

四组短训各最多 1000 步及 5 分钟，全轮实验上限约 30 分钟。具体实际步数、设备、停止原因和固定监督失败对照见 [第二周报告](../../reports/stage1/week2.md)。

## **三、学习验收**

阅读答案后，合上参考代码完成根目录 learning/README.md 的核心任务。教学参考、项目实现和你的理解分别验收；数学代码通过不代表已经掌握，更不代替项目生成能力评测。

## **四、S2 第三周入口**

[图标数据与潜空间 DiT 基础](S2_第3周_图标数据与潜空间DiT基础.ipynb) 连接数据、空间 VAE、潜空间扩散目标、patch、位置与时间、adaLN-Zero。保留 22 个代码单元的真实输出；教学参考不替代项目核心。

使用 `--extra notebook --extra latent` 同步与运行。先调用显式资源准备脚本，再执行 Notebook，HTML 指定 `artifacts/runs/s2_week3/learning_preview.html`。单轮 VAE、缓存与拟合计算上限 30 分钟，最终执行约 14.7 秒，另计 CPU 预检和此前一次验证，合计约 117.8 秒。全量图像已生成，但缓存仅 256 张；未开展完整生成训练。

下一项个人任务为 [L03](../../learning/L03_DiT核心任务.md)；L01 代码已通过 [独立验收](../../reports/tasks/ddpm_review.md)，L02 个人核心和个人理解验收仍待做。
