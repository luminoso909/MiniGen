## **一、按阶段准备数据与权重**

MiniGen v2 的正式外部数据为 ArtBench Public Domain 256x256，全量 46,298 张。B0、B1、B2 从随机初始化训练，正式生成时加载本项目训练的 Checkpoint、冻结 VAE 和冻结 CLIP 文本塔。英文 Caption 随数据集发布，经本地清洗后用于文本条件。评测模型按冻结协议单独准备。阶段与验收口径见 [v2 重构规划](MiniGen2重构规划.md)。

| 首次使用 | 数据或权重 | 给谁使用 | 是否需要外部下载 | 本机状态 |
| --- | --- | --- | --- | --- |
| P0 | 无新增数据或权重 | 仓库审计和 CPU 基础检查 | 无 | 使用现有项目环境 |
| P1 | ArtBench 固定版 Parquet | 全量接入、解码、图文审计、清洗与项目划分 | **必需**。仅取三个固定 Parquet 分片 | 状态见 [P1 交付](../reports/v2/01_p0_p1_delivery.md) |
| P1-04 → P2，P3–P7 复用 | CLIP-B/32 tokenizer 与冻结文本塔 | P1 长度校验，P2 文本特征缓存，三主干的 prompt 条件，P7 推理 | **必需，当前已有** | 固定 7 文件通过字节数和 SHA-256 核对 |
| P2，P3–P7 复用 | 冻结 SDXL VAE 编码器和解码器 | P2 的 200 张重建和潜变量缓存，三主干采样结果解码，P7 推理 | **必需，当前已有** | 固定 2 文件通过字节数和 SHA-256 核对 |
| P2，P5 复用 | LPIPS 特征网络和线性校准权重 | 200 张重建感知距离，以及预登记后的多样性统计 | **指标必需**。推荐 AlexNet 与 LPIPS v0.1 | AlexNet 与 LPIPS v0.1 Alex 校准已下载，CPU 小样通过 |
| P3 → P4 | ArtBench 冻结清单、256px 潜变量和文本特征缓存 | 三主干训练、验证与恢复 | **本地生成** | 依赖 P1/P2 验收 |
| P3 → P4，P5–P7 复用 | B0、B1、B2 的 Checkpoint 和 EMA | 训练恢复、采样、正式评测、部署 | **本项目训练生成** | v2 正式权重待训练 |
| P5 | ArtBench 冻结测试集、MiniGen-Bench 提示词、固定噪声种子与人工标签 | 分布质量、ID/组合外/改写评测及盲审 | **复用 P1 数据并自建评测集** | 候选题目和协议已锁定，人工批准待审 |
| P5 | Inception V3 特征权重、图文 CLIP 与 VQA 权重 | KID、CMMD、CLIPScore、语义遵循评分 | **正式指标所需**，按冻结协议取所选权重，见第三节 | 资产锁已建立；Inception/完整 CLIP 已实测，VQA 大权重未下载 |
| P6 | 额外种子与 10M/50M 或 Chimera 候选的 Checkpoint | 后续可选实验 | 默认复用原数据和冻结组件，自行训练 Checkpoint | 依主结果与批准决定 |
| P7 | 已验收的自训 Checkpoint 和两个冻结推理组件 | 发布复现与推理演示 | 复用 P2、P4 资产 | 干净机器按同一锁定身份恢复 |

本机状态覆盖项目的 `artifacts/models/` 与已查看的默认缓存位置。若有另存于其他位置的权重，先按相同身份与哈希核对。

本机审计与剩余事项见 [P1 审计](../reports/v2/data/artbench_audit.md)、[P2 资产回执](../reports/v2/components/p2_assets.json) 和 [资源待办](../待办清单.md)。

## **二、P1 和 P2 的具体文件**

### **1. P1 正式数据源**

源为 `alfredplpl/artbench-pd-256x256`，固定 revision 为 `4056b70025f0e3f916fb271c7a792633ba637484`。取三个原生 Parquet 分片，共 **1,517,381,098 字节**，数据卡另列可选打包文件。Parquet 足以读取全部 46,298 张 256×256 图像、英文描述、日文描述及来源元数据。[固定版本文件](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256/tree/4056b70025f0e3f916fb271c7a792633ba637484) · [分片哈希配置](../configs/datasets/artbench_pd_256.yaml)

从仓库根目录恢复本地源文件：

```bash
.venv/bin/hf download alfredplpl/artbench-pd-256x256 --repo-type dataset \
  --revision 4056b70025f0e3f916fb271c7a792633ba637484 \
  --include 'data/*.parquet' \
  --local-dir data/raw/artbench_pd_256/4056b70025f0e3f916fb271c7a792633ba637484
```

下载后核对固定 SHA-256，再离线处理。全量作品按组分到训练、验证和测试；实际清单身份及审计证据见 [数据规范](data.md) 与 [P1 审计](../reports/v2/data/artbench_audit.md)。

### **2. P1 文本来源**

ArtBench 的英文 Caption 由数据发布者提供。项目审计与清洗使用同一固定版描述及本地 CLIP tokenizer。OpenClipart 离线标注路线已经下线，旧 QwenVL 2B/8B 标注权重和独立环境已清理，不需下载。PixelArt-308K 仅作备选参考。[ArtBench 数据卡](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256)

### **3. P2 冻结组件与 LPIPS**

| 权重 | 核对结果与位置 | 恢复方法 |
| --- | --- | --- |
| `madebyollin/sdxl-vae-fp16-fix`，revision `207b116dae70ace3637169f1ddd2434b91b3a8cd` | `config.json` 和 `diffusion_pytorch_model.safetensors` 共 334,643,869 字节，均匹配。路径 `artifacts/models/sdxl_vae/207b116dae70ace3637169f1ddd2434b91b3a8cd/`。见 [VAE 身份锁](../configs/models/sdxl_vae.json) | 仅恢复锁定的这两个文件，使用 [固定版本](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/207b116dae70ace3637169f1ddd2434b91b3a8cd) 并逐文件校验 |
| `openai/clip-vit-base-patch32`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 7 文件共 256,297,846 字节，均匹配。路径 `artifacts/models/clip_text/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/`。见 [文本塔身份锁](../configs/models/clip_text.json) | 需要恢复时走 `scripts/prepare_stage3_assets.py --text-only`。缺失文本主权重时，上游组合权重需要约 605 MB 的暂存，脚本按锁提取文本塔。见 [历史恢复验收](../reports/stage3/resource_recovery.json) |
| LPIPS v0.1，候选 `net=alex` | AlexNet 主干使用 TorchVision 的 `alexnet-owt-7be5be79.pth`，另用 LPIPS 的 `lpips/weights/v0.1/alex.pth` 校准权重。建议放在 `artifacts/models/eval/lpips_alex/` 并写入本地身份 | P2 前显式准备主干，TorchVision 标注文件为约 233.1 MB。若实现的包已带匹配校准文件，则校验并复用。PSNR 与 SSIM 直接计算。见 [TorchVision AlexNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html) 与 [LPIPS 校准权重](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1) |

P2 的 200 张重建检查与单段 CLIP 验收已在 ArtBench 上实际运行；当前不缺 P2 模型权重。人工审核与正式缓存进度见 [P2 交付](../reports/v2/02_p2_delivery.md)。

## **三、P5 评测资源**

2026-11-10 按 [资产锁](../configs/models/evaluation.json) 显式准备；普通评分仅离线读取，生成条件 CLIP 文本塔不能代替评测完整 CLIP。

| 指标 | 固定权重 | 本轮实际状态 |
| --- | --- | --- |
| KID | torch-fidelity Inception V3 `weights-inception-2015-12-05-6726825d.pth`，2048 维 | 已下载、全哈希通过、CPU/MPS FP32 特征及真实小样本指标通过 |
| CMMD / CLIPScore | `openai/clip-vit-large-patch14-336`，revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`；完整 `pytorch_model.bin` 及 tokenizer/预处理配置 | 已下载，主权重 1,711,974,081 字节；本轮 PyTorch 转换后端不宣称与其他后端逐位等价 |
| LPIPS | 已有 AlexNet 与 LPIPS Alex v0.1 校准资产 | 复用 P2 文件，未重复下载；双种子辅助差异已计算 |
| VQA | `Salesforce/blip-vqa-base`，revision `787b3d35d57e49572baabd22884b3d5a05acf072` | 仅小型配置/tokenizer 已准备；1,538,800,584 字节 Safetensors **未下载**，真实 BLIP 推理待云端；接口与模拟测试不等于真实推理 |

路径按 `artifacts/models/eval/<模型>/<revision>/` 隔离；准确 URL、SHA-256 与字节数以资产锁为准。未来显式执行：

```bash
uv sync --locked --extra evaluation
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models inception clip --verify-only
# 仅在用户准备真实 VQA 时下载；本轮没有执行此命令。
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models vqa
```

KID 固定三次核与无偏 MMD，正式 100×1000 子集；CMMD 固定单位图像投影、带宽 10、biased MMD×1000；CLIPScore 固定 L/14@336、无前缀及系数 2.5。真实参考集、题目与人工批准分开记录，见 [Benchmark](../reports/v2/benchmark/README.md)。下载完成不表示绘画域评分器已通过人工校准，VQA 100+100 真实校准仍待执行。[torch-fidelity](https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_kid.py) · [CMMD](https://github.com/google-research/google-research/tree/master/cmmd) · [CLIPScore](https://github.com/jmhessel/clipscore/blob/main/clipscore.py)
