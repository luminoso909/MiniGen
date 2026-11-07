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
| P5 | ArtBench 冻结测试集、MiniGen-Bench 提示词、固定噪声种子与人工标签 | 分布质量、ID/组合外/改写评测及盲审 | **复用 P1 数据并自建评测集** | 需在正式训练前冻结提示词和协议 |
| P5 | Inception V3 特征权重、图文 CLIP 与 VQA 权重 | KID、CMMD、CLIPScore、语义遵循评分 | **正式指标所需**，按冻结协议取所选权重，见第三节 | 尚未冻结评测资产 |
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

| 指标 | 实际需要的权重 | 准备方式 |
| --- | --- | --- |
| KID | torch-fidelity 的 Inception V3 参考权重 `weights-inception-2015-12-05-6726825d.pth` | 预登记特征层、后端、权重哈希与版本，显式传入本地文件。该实现使用 TensorFlow 转换权重。见 [torch-fidelity 特征提取源码](https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py) 与 [发布文件](https://github.com/toshas/torch-fidelity/releases/tag/v0.2.0) |
| CMMD | OpenAI CLIP ViT-L/14@336 的图像塔与投影 | 优先用参考实现对应的 `ViT-L-14-336px.pt`，或经等价性核对的转换权重。锁定特征归一化、预处理和核参数。见 [CMMD 参考实现](https://github.com/google-research/google-research/blob/master/cmmd/embedding.py) 与 [参考权重映射](https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/clip/model.py) |
| CLIPScore | 带图像塔、文本塔和两侧投影的完整 CLIP，以及其预处理器和 tokenizer | 可在预登记时选完整 ViT-L/14@336 与 CMMD 共享权重，明确写出 CLIPScore 的模型变体。若选 B/32，则按固定版本另取完整 B/32。P1/P2 现有文本塔没有图像塔及投影。见 [CLIPScore 方法](https://arxiv.org/abs/2104.08718v3) 与 [完整 B/32 文件](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268) |
| VQA 属性与关系 | 一套通过 ArtBench 绘画域人工校准的 VQA 模型。独立候选 `Salesforce/blip-vqa-base` | 候选 Safetensors 约 1.54 GB，另含配置、图像预处理和 tokenizer。经标注误差与盲审校准后才锁定。评测结果须与独立人工标注对照。见 [候选模型卡](https://huggingface.co/Salesforce/blip-vqa-base) 与 [权重文件](https://huggingface.co/Salesforce/blip-vqa-base/tree/main) |

建议 P5 评测权重统一放在 `artifacts/models/eval/` 下按模型和版本隔离。正式分布参考图来自冻结的 ArtBench 测试切分，ID、组合外及改写提示词由项目生成并版本化。历史 MNIST、几何图标、OpenMoji/Sana 与 LoRA 资产按原用途归档。
