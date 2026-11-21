## **一、MiniGen**

MiniGen 是一个文本条件图像生成研究项目。模型从随机初始化开始学习 **Flow Matching** 的速度目标，使用冻结的 SDXL VAE 和 CLIP ViT-B/32 文本塔，在潜空间生成 256×256 RGB 图像。

| 主干 | 图文交互 | ArtBench 配方参数量 |
| --- | --- | ---: |
| B0 · Plain Conditional DiT | 图像自注意力与文本交叉注意力 | 30,359,824 |
| B1 · Single Stream DiT | 在单流中联合处理图文 token | 30,061,456 |
| B2 · Hybrid DiT | 前段双流联合注意力，后段合为单流 | 30,061,456 |

- **ArtBench 绘画** 在 AutoDL 上完成三架构各 10,000 步训练。固定验证 Flow MSE 依次为 0.7852、0.7698、0.7712，固定描述下的生成主体仍难稳定辨认。[样图对照](reports/v2/p4/B0-B2_训练对比.html)
- **动漫头像** 完成 B0/B1/B2 的 10k 训练、B1/B2 的六组 30k 对照和 208 题盲评。主基线为 B2 50M patch1 Uniform 30k EMA，采样使用 Euler 40、CFG 3。[评测指南](docs/anime-evaluation.md)

## **二、快速开始**

使用 Python 3.12 和 uv。以下命令验证 CPU 环境及评测命令入口。

```bash
git clone https://github.com/luminoso909/MiniGen.git
cd MiniGen
uv sync --locked
uv run --locked --offline --no-sync minigen doctor --device cpu --json
uv run --locked --offline --no-sync minigen bench --help
```

代码、训练配方和评测协议随仓库维护。原始数据、处理后的图片和清单放在本地 `data/`，权重、缓存及运行结果放在 `artifacts/`。真实 P5 冒烟评测还需要本地 ArtBench 题库和对应的 P4 权重等资产。依赖、题库位置和运行方式见 [P5 Benchmark 指南](reports/v2/benchmark/README.md)。

## **三、训练与评测入口**

| 任务 | 入口 |
| --- | --- |
| 数据划分、冻结组件与缓存 | [数据规范](docs/data.md) · [训练配置导航](configs/README.md) |
| AutoDL 环境准备、训练、恢复与采样 | [云端操作手册](docs/cloud/MiniGen_云端训练与评测完整操作手册.md) |
| ArtBench 生成、评分与盲审 | [P5 Benchmark 指南](reports/v2/benchmark/README.md) |
| 头像数据与四组 30k 盲评 | [动漫头像评测指南](docs/anime-evaluation.md) |

`minigen evaluate` 计算固定配对 Flow MSE。生成质量评测使用 `minigen bench` 的采样、评分和盲审流程。

## **四、代码与文档**

- `src/minigen/` 包含模型、训练、采样、数据接口和评测实现。
- `configs/`、`scripts/` 和 `tests/` 分别维护配方、执行入口和回归测试。
- [文档导航](docs/README.md) 汇总工程、评测和学习入口。[任务索引](MiniGen项目任务清单与学习路径.md)
