## **一、MiniGen**

MiniGen 是一个文本条件图像生成研究项目。三个约 30M 参数的 Latent DiT 从随机初始化训练，共享 ArtBench 绘画、冻结的 SDXL VAE、CLIP ViT-B/32 文本塔和 Flow Matching 速度目标。模型在潜空间生成图像，最后由 VAE 解码为 256×256 RGB。

| 主干 | 图文交互 | 可训练参数 |
| --- | --- | ---: |
| B0 · Plain Conditional DiT | 图像自注意力加文本交叉注意力 | 30,359,824 |
| B1 · Single Stream DiT | 拼接图文 token，在单流中共同计算 | 30,061,456 |
| B2 · Hybrid DiT | 前段双流联合注意力，后段合为单流 | 30,061,456 |

ArtBench 的三个主干已在 AutoDL 各完成 **10,000 步**，同一组 256 条固定验证配对的 Flow MSE 分别为 0.7852、0.7698、0.7712。16 条固定描述的生成主体仍难稳定辨认，正式 MiniGen-Bench 待批准。动漫头像是独立的 B1 实验，已通过缓存回读和 AutoDL CPU 前向，CUDA 训练待执行。见 [B0 运行报告](reports/v2/p4/autodl_b0_full_10000_2026-11-12.md)、[B1/B2 运行报告](reports/v2/p4/autodl_b1_b2_full_10000_2026-11-12.md)、[三模型样图对照](reports/v2/p4/B0-B2_训练对比.html) 和 [当前交接](reports/v2/00_handoff.md)。

## **二、克隆与环境**

在 Mac 上使用具备仓库读取权限的 GitHub 账号。HTTPS 如需认证，使用本机已配置的凭据管理器。

```bash
git clone https://github.com/luminoso909/MiniGen.git
cd MiniGen
uv sync --locked --extra artbench --extra conditioning --extra latent
uv run --locked --offline --no-sync minigen --help
```

项目使用 Python 3.12 和 uv。评测环境增加 `--extra evaluation`。Mac 用于编辑代码；本项目的 Python 测试、训练和评测在 AutoDL 上执行。使用已配置的 `autodl` SSH 别名。新实例可在数据盘执行 `git clone`，或从 Mac 传入 Git bundle 后克隆固定提交。已有实例通过 rsync 同步需要更新的源码、配置和脚本。数据、模型权重、虚拟环境在远端单独准备。见 [AutoDL 环境记录](docs/cloud/MiniGen_AutoDL无卡准备.md) 和 [云端操作手册](docs/cloud/MiniGen_云端训练与评测完整操作手册.md)。

## **三、数据集与缓存**

### **1. ArtBench 主实验**

使用 ArtBench Public Domain 256×256 的固定 Parquet 版本。项目按图像及作品分组重新划分，英文 Caption 清洗后送入 CLIP。训练阶段读取 VAE latent、CLIP token 特征和 mask。Test 用于锁定后的 Benchmark。见 [数据卡](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256)、[数据规范](docs/data.md) 和 [冻结来源](configs/datasets/artbench_pd_256.yaml)。

| 划分 | 张数 | 用途 |
| --- | ---: | --- |
| Train | 37,042 | 优化主干参数 |
| Validation | 4,637 | 固定验证损失和采样对照 |
| Test | 4,619 | 独立评测 |

在 AutoDL 的项目目录运行以下步骤。首次执行需要下载固定分片和冻结模型。成功后可复用已有文件。准备脚本只使用 CPU。

```bash
ssh autodl
cd /root/autodl-tmp/minigen
(
  set -e
  for step in env dataset models worker prepare verify; do
    bash scripts/setup_autodl_cpu.sh "$step"
  done
)
```

处理结果在 `data/processed/artbench_pd_256/v1/`，包含图像、Caption、组与划分的 `manifest.jsonl`。AutoDL 可在设备就绪后独立生成全量 CUDA 缓存。P2 在 Mac 上已验收的 `v1_formal` 缓存具有独立身份，标准 `minigen train/sample` 使用该缓存。下面的 AutoDL 启动器使用远端独立全量缓存。见 [P2 数据与组件验收](reports/v2/02_p2_delivery.md)。

### **2. 动漫头像独立实验**

使用 `puruchinera/anime-faces-256` 的已清洗头像与标签。按组划分后保留 36,547 张，训练 32,892 张、验证 3,655 张。划分脚本的起点是已经准备好的 `data/processed/anime_faces_256/color_front_v1/`，其中包含清洗清单与图像。由它生成 `b1_v1/`，再编码到 `artifacts/cache/anime_faces_256/b1_v1/`。原始数据到清洗输入的工具仍保存在原实验目录。AutoDL 上已部署这套派生缓存。见 [准备与验收记录](reports/v2/00_handoff.md)、[划分入口](scripts/prepare_anime_faces_split.py) 和 [缓存入口](scripts/cache_anime_faces.py)。

## **四、终端训练、恢复与采样**

### **1. AutoDL 上运行 ArtBench**

从远端项目根目录开始。先检查 GPU 和可用显存。新构建缓存使用镜像 CUDA PyTorch，以及上一节准备好的组件依赖。已有全量缓存时，设置 `CACHE_RUN` 指向它的运行根目录并直接执行 `verify`。

```bash
cd /root/autodl-tmp/minigen
nvidia-smi
CACHE_RUN=artifacts/runs/autodl-b0-full-10000-20261112
if [ ! -s "$CACHE_RUN/full_cache/index.json" ]; then
  bash scripts/run_artbench_autodl.sh "$CACHE_RUN" build --batch-size 8
fi
bash scripts/run_artbench_autodl.sh "$CACHE_RUN" verify
```

默认以 B0 的约 30M 模型做 20 步短跑。`MODEL` 可设为 `b0`、`b1` 或 `b2`。需要按历史 ArtBench 配方重跑 10,000 步时，将 `RUN_KIND` 改成 `full`。全量模式使用 CUDA BF16、有效 batch 64 和 6 小时进程时限，开跑前先确认本轮预算。

```bash
MODEL=b0
RUN_KIND=smoke
RUN="artifacts/runs/artbench-${MODEL}-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
/root/miniconda3/bin/python - "$MODEL" "$RUN_KIND" "$RUN" "$CACHE_RUN" <<'PY'
import json
import sys
from pathlib import Path

model, kind, run, cache_run = sys.argv[1:]
config = json.loads(Path(f"configs/training/p4_{model}_cuda_pilot.json").read_text())
config["data"]["cache"] = f"{cache_run}/full_cache"
config["training"]["output_dir"] = f"{run}/train"
if kind == "smoke":
    config["data"].update(train_limit=16, validation_limit=4)
    config["training"].update(
        precision="fp32", batch_size=2, grad_accum=1, max_steps=20,
        max_seconds=300, schedule_steps=20, warmup_steps=2,
        checkpoint_every=10, validation_every=10, checkpoint_steps=[10, 20],
    )
elif kind == "full":
    every = 500 if model == "b0" else 1000
    config["data"].update(train_limit=None, validation_limit=256)
    config["training"].update(
        max_steps=10000, max_seconds=21600, schedule_steps=10000, warmup_steps=500,
        checkpoint_every=every, validation_every=every,
        checkpoint_steps=list(range(every, 10001, every)),
    )
else:
    raise SystemExit("RUN_KIND must be smoke or full")
Path(run, "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
bash scripts/run_artbench_autodl.sh "$CACHE_RUN" minigen train \
  --config "$RUN/config.json" --device cuda
bash scripts/run_artbench_autodl.sh "$CACHE_RUN" minigen sample \
  --config "$RUN/config.json" --checkpoint "$RUN/train/last.pt" \
  --output "$RUN/samples" --device cuda --split validation --count 4 \
  --seed 1234 --steps 40 --method euler --guidance 1
```

训练结果看 `train/result.json` 的 `status` 和 `step`，训练曲线取 `train/metrics.jsonl`，采样网格在 `samples/grid.png`。中断后，用同一配置、源码和数据身份恢复。

```bash
bash scripts/run_artbench_autodl.sh "$CACHE_RUN" minigen train \
  --config "$RUN/config.json" --device cuda --resume "$RUN/train/last.pt"
```

全量配方的学习率峰值为 1e-4，验证 MSE 固定取 256 条。复现历史的 16 张验证样图时，将采样的 `--count` 改为 `16`。配置来源、预算与评测步骤见 [训练配置导航](configs/README.md)、[正式实验预登记](reports/v2/final/MiniGen_正式实验预登记.md) 和 [云端任务清单](docs/cloud/云端任务清单.md)。

### **2. AutoDL 上运行动漫 B1**

先验收完整缓存与单张前向，再按独立配方运行。该配方设置 CUDA BF16、有效批量 64、最多 10,000 步。

```bash
cd /root/autodl-tmp/minigen
nvidia-smi
bash scripts/run_anime_b1_autodl.sh check --device cuda
bash scripts/run_anime_b1_autodl.sh train --config configs/training/anime_b1_10000.json
```

恢复时给 `train` 追加 `--resume artifacts/runs/anime_faces_256/b1_v1_10000/last.pt`。检查点和日志都在同一运行目录。见 [动漫 B1 配方](configs/training/anime_b1_10000.json)。

## **五、仓库层级与阅读入口**

```text
MiniGen/
├── src/minigen/           运行时 Python 包与命令入口
│   ├── models/            B0、B1、B2 主干与注意力组件
│   ├── objectives/        Flow Matching 插值、速度目标与损失
│   ├── samplers/          Euler、Heun 和 CFG
│   ├── autoencoders/      冻结 VAE 的编码、解码和缩放
│   ├── conditioning/      冻结 CLIP 文本条件
│   ├── data/              数据清单、表示缓存及读写校验
│   ├── training/          训练循环、优化、EMA、检查点与恢复
│   └── evaluation/        Benchmark、生成、评分与汇总
├── configs/               数据和模型锁、训练配方、评测协议
├── scripts/               资源准备、云端启动器、验收和诊断
├── tests/                 单元测试与工程回归
├── docs/                  数据与云端说明、学习笔记、问答和论文
├── learning/              个人练习和复验入口
├── reports/               实验报告、阶段交接和验收记录
├── data/                  本地数据、处理后的图像和清单
└── artifacts/             模型资产、缓存、训练日志和检查点
```

代码和配置随 Git 克隆。实际数据、模型权重、缓存、检查点及大型运行产物由 Git 忽略。小型结构化回执和报告放在 `reports/`。正式生成质量通过 `minigen bench` 评测，`minigen evaluate` 输出固定配对 Flow MSE。见 [Benchmark 审核包](reports/v2/benchmark/MiniGen_Benchmark_最终审核包.md)、[项目任务索引](MiniGen项目任务清单与学习路径.md) 和 [学习导航](docs/learning/README.md)。
