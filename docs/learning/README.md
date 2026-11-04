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

## **五、S2 第四周入口**

[潜空间扩散训练与采样](S2_第4周_潜空间扩散训练与采样.ipynb) 从第三周的固定监督拟合进入完整训练与采样链路：缓存版本与分片、潜空间噪声目标、训练/验证、时间分桶、零预测和高斯矩基线、EMA、精确恢复与随机初始噪声的完整反向链。

本周显式复用第三周唯一 dit_reference 单元，避免维护第二份教学 DiT，记录源码哈希；使用已验证的 src DDPM。VAE、DiT 个人核心及你的理解仍单独验收。完整训练/验证缓存与教学短训、采样已执行，个人核心与生成质量仍另验；实际结果见 [第四周报告](../../reports/stage2/week4.md)；个人接手见 [第四周连接任务](../../learning/L03_第四周训练连接任务.md)。

使用项目环境显式执行，不覆盖旧笔记预览：

```bash
uv run --locked --extra notebook --extra latent python scripts/execute_learning.py docs/learning/S2_第4周_潜空间扩散训练与采样.ipynb --html-output artifacts/runs/s2_week4/learning_preview.html
```

这会重新执行有界本地教学实验，读写 data/ 与 artifacts/ 中相应缓存和运行目录；不会启动收费资源。报告分别解释重建质量、固定目标拟合、随机时间验证损失与随机生成质量，不把其中一种代替其余项目。

## **六、S3 全部工作入口**

[文本条件、CFG 与可信评测](S3_文本条件CFG与可信评测.ipynb) 覆盖冻结 CLIP、77-token 完整分段、Cross-Attention、条件丢弃与 CFG、两种子训练、同噪声对照、属性评分、效率、Flow 二维预研和研究设计。完整参考只在新 Notebook；个人接口与验收见 [L04](../../learning/L04_文本条件与CFG核心任务.md)。

```bash
uv sync --locked --extra notebook --extra latent --extra conditioning
uv run --locked --extra notebook --extra latent --extra conditioning python scripts/prepare_stage3_assets.py
uv run --locked --extra notebook --extra latent --extra conditioning python scripts/execute_learning.py docs/learning/S3_文本条件CFG与可信评测.ipynb --html-output artifacts/runs/s3_conditional_v3/learning_preview.html
uv run --locked python scripts/check_conditioning_core.py
uv run --locked python scripts/check_cfg_core.py
```

资源准备显式下载论文和固定 CLIP 文本组件；导入、doctor 和测试不下载。重跑 Notebook 会重新执行最多 30 分钟本地教学计算；源码或监督身份变化使用新 run_id。完整执行与质量结论见 [S3 报告](../../reports/stage3/stage3.md)，逐项状态见 [交接](../../reports/tasks/stage3.md)。

S1 最新结论以 [验收报告](../../reports/stage1/s1_acceptance_2026-10-30.md) 为准：DDPM 与 VAE 个人数学核心已验证，S1 的文档/理解收尾不自动全部完成。前几周教学历史记录中“个人待做”描述的是当时状态，不能覆盖新的验收证据。
