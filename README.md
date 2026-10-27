## **一、MiniGen 是什么**

面向可控图标与插画生成的小型生成模型实验系统。通过亲手实现核心算法、可信的训练评测，以及一次现代模型领域适配，掌握生成模型研发的完整过程。

| 路线 | 内容 | 当前状态 |
| --- | --- | --- |
| A：自研实验 | DDPM → VAE → 条件 Latent DiT → Flow Matching | 两周 DDPM/VAE 教学 Notebook 与训练工程已验证；src 核心待你完成 |
| B：现代适配 | Sana 图标领域 LoRA 与前后评测 | 尚未下载权重或执行微调 |
| 开发基础 | uv 环境、CLI 诊断、项目结构、测试与文档 | 见 [初始化验证报告](reports/bootstrap.md) |

完整路线见 [项目任务清单与学习路径](MiniGen项目任务清单与学习路径.md)。开始日期为 2026 年 10 月 26 日，第一里程碑为 11 月 30 日，最终里程碑为 12 月 15 日；个人投入每周 8–12 小时，云资源总预算 100–300 元。

### **1. 已实现与未实现**

已提供环境诊断、必要基础测试、数据与 AI 工作流规范。`train`、`sample`、`evaluate`、`cache` 只保留命令位置，执行会明确报错，退出码为 2。

第一周入口：[DDPM 原理与 MNIST Notebook](docs/learning/S1_第1周_DDPM原理与MNIST实验.ipynb)、[学习导航](docs/learning/README.md)。Notebook 显式使用获授权的教学 Scheduler，调用真实 U-Net、训练和采样工程；项目 CLI 不会自动使用这份参考。实验结果与限制见 [第一周报告](reports/stage1/week1.md)。

第二周入口：[VAE 原理与潜空间 Notebook](docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb)、[第二周报告](reports/stage1/week2.md)、[论文与阅读导航](docs/papers/README.md)。四组真实短训、数学和单进程随机顺序恢复已验证；个人核心与理解仍待验收。

五个算法学习区调用时抛出 `CoreImplementationRequired`。通过初始化测试只证明工程基础与未实现边界正确，不代表算法已验证、模型已训练或生成效果已达标。

## **二、建立与使用环境**

### **1. 基础安装与诊断**

项目使用 Python 3.12.14，uv 创建本地 `.venv`。在项目根目录执行：

```bash
uv sync --locked
uv run --locked minigen doctor
uv run --locked minigen doctor --json
uv run --locked minigen doctor --device cpu --json
uv run --locked minigen --help
```

`uv run` 自动使用项目环境，不必手动激活。uv 缓存由 `uv.toml` 放到 `.cache/uv`；不修改全局 Python、Git 或 Codex 配置。同步与运行方式依据 [uv 官方文档](https://docs.astral.sh/uv/guides/projects/)。

`doctor` 检查真实包导入、CPU/MPS/CUDA 可用性和小张量前向反向计算。默认检查 CPU 和可用的加速设备；不可用设备标为 unavailable，显式请求不可用设备时退出码为 1。小张量通过不能保证后续模型的全部算子受支持。

### **2. 手动激活与可选依赖**

```bash
source .venv/bin/activate
python -m minigen doctor --device cpu
deactivate
```

需要现代生成模型或 Notebook 时再执行对应命令：

```bash
uv sync --locked --extra generation
uv run --locked --extra generation python -c "import diffusers, transformers"
uv sync --locked --extra notebook
uv run --locked --extra notebook jupyter lab
```

两组都需要时使用 `uv sync --locked --extra generation --extra notebook`；后续 `uv run` 同样带所需 extra，避免精确同步移除未请求的可选包。锁文件包含可选依赖解析，不代表默认安装了它们。Notebook extra 不自动注册全局 kernel。

### **3. 验证与环境恢复**

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked minigen doctor --device cpu --json
```

新电脑或重建项目时，先获得 uv 和指定 Python，再使用提交的 `uv.lock` 同步。不复制 `.venv` 到其他机器。云 GPU 的 PyTorch 构建、驱动与 CUDA 兼容性须在实际云环境检查，不能用本机 MPS 检查代替；CI 仅验证 CPU 行为。

## **三、目录导航与数据管理**

```text
src/minigen/        模型、目标、采样、条件、数据、训练与评测
configs/            数据来源登记和小型清单示例
learning/           五项核心任务入口
docs/learning/      知识笔记与带真实输出的 Notebook
docs/papers/        核心论文、版本哈希与阅读任务
tests/              当前真实行为与未实现边界检查
docs/               数据规范与 Agent 工作流
templates/          数据卡、质量报告、任务及交接模板
reports/            验证、任务和研究记录
data/               本地原始、处理后数据及实际清单，默认忽略
artifacts/          缓存、权重、实验输出，默认忽略
```

数据采用图片文件加 JSONL 清单，以 YAML 登记来源。三层数据为 MNIST、准确标注的合成几何图标、OpenMoji 筛选集。SQLite 暂不引入，未来只能作为可重建索引。

来源登记见 [sources.yaml](configs/datasets/sources.yaml)，详细规则见 [数据管理](docs/data.md)。清单示例为明确标记的虚构记录，不含真实图像，不可直接用于训练。MNIST 显式获取与文件校验已实现；合成图标、OpenMoji 采集与完整质量流水线仍待做。

## **四、学习与 Agent 协作**

你负责原理、五项核心算法、整体数据流与实验结论；Codex 负责环境、数据工程、训练工程、验证、文档及交接。入口见 [学习任务](learning/README.md)、[AI 工作流](docs/ai-workflow.md) 与 [项目指令](AGENTS.md)。

你可以直接给 Codex 指令：“读取任务清单和当前交接记录，推进依赖满足的工程任务；核心实现留给我，准备清楚的输入输出、数学检查和学习任务。”

目前单 Agent 工作。四类角色模板预留数据、工程、评测和审查并行空间；未来显式启用后再分配互不冲突的修改范围。本项目不包含模型 API 编排系统，也不需要额外 API Key。

下一步是按学习导航完成 L01 / S1-03 与 [L02 / S1-06](learning/L02_VAE核心任务.md) 的个人核心。Agent 已提供独立数值对照和训练工程；不能用教学参考绕过项目核心未实现错误。VAE 验收命令为 `uv run --locked python scripts/check_vae_core.py`，当前诚实返回待做。

## **五、Git 与来源说明**

当前只初始化本地 `main` 和首次提交，不设置远程、不推送。后续发布需要确认账号、仓库和可见性。`.gitignore` 排除数据、缓存、虚拟环境、权重与凭据；文本模板、小型配置和必要报告保留。

主要算法参考 [官方 DiT](https://github.com/facebookresearch/DiT)，组件与微调参考 [Diffusers](https://github.com/huggingface/diffusers)。本次未复制两者的算法实现。数据、代码与权重的来源分别登记，不将第三方内容标为原创，也未擅自为整个项目指定开源许可证。
