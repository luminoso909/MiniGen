## **一、正式数据版本**

MiniGen v2 使用 **ArtBench Public Domain 256x256** 全量 **46,298 张**绘画。三种生成主干共享该数据版本的训练、验证、测试划分。输入保持 **256×256 RGB**，保留上游构图。OpenClipart 路线下线，PixelArt-308K 仅保留为备选参考。[重构规划](MiniGen2重构规划.md)

| 项目 | 约定 |
| --- | --- |
| 发布与版本 | `alfredplpl/artbench-pd-256x256@4056b70025f0e3f916fb271c7a792633ba637484`，上游单一 `train` 切分 |
| 读取 | 本地固定版 Parquet 批量离线读取。校验分片字节数和 SHA-256，逐行解码并核对图像尺寸 |
| 图像 | `image` 提供图像，训练入口使用 256×256 RGB 像素 |
| 文本 | `caption` 是英文原始描述，清洗后进入文本条件；`caption_jp` 可追溯上游但不进入当前训练 |
| 溯源 | `artist` 和 `url` 用于作品溯源及审计。发布者将其描述为原 ArtBench 的公版图像子集，发布页标注 MIT |
| 本地目录 | 原始分片放 `data/raw/artbench_pd_256/4056b70025f0e3f916fb271c7a792633ba637484/data/`，项目产物放 `data/processed/artbench_pd_256/v1/` |
| 项目划分 | 固定分组种子 `20261107`。训练 37,042、验证 4,637、测试 4,619，合计覆盖全部 46,298 行 |

来源身份和许可口径见 [ArtBench 数据卡](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256) 与 [固定来源配置](../configs/datasets/artbench_pd_256.yaml)。

## **二、清洗、审计与清单**

1. **图像。** 全量检查分片身份、行数、解码和尺寸。至少抽查 200 张图像的明显模糊、拉伸、畸变和裁切，允许绘画自然笔触和柔焦表现。记录实际缺陷的样本 ID 和可复查证据。
2. **划分。** 先按作品身份分组，再生成训练、验证和测试集合。用原图哈希、解码图像哈希及来源 URL 检查跨集精确重合，记录近重复检查方法及覆盖范围。验证集负责选型，测试图及描述封存。
3. **文本。** 保留主体、场景、颜色、数量、关系和 `portrait`、`landscape`、`painting` 等基础风格词。移除可安全分离的署名、年份、作品标题归属、重复句与冗长的非视觉说明。保留原文、规则版本和处理状态。发布者注明英文 Caption 来自 Florence-2-large，清洗规则仅调整文本表达；视觉事实须在配对审计中判断。[发布者说明](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256)
4. **长度。** 使用固定 CLIP tokenizer 检查单段最多 75 个内容 token，加 BOS/EOS 后总长 77。按完整语句缩短超限 Caption，无法安全处理则单列复核。对至少 200 条配对描述检查主体、场景、颜色、数量和关系，分别报告各项的有效分母、明显错误与修订。
5. **冻结。** 清单记录样本 ID、来源 revision、原始行位置、来源 URL、作者、图像哈希、图像尺寸、组、划分、原始与清洗后 Caption、清洗和 schema 版本。正式 Loader 拒绝示例和越出数据根目录的路径。预处理、分割和标注变更使用新版本。源数据、真实图片、权重与逐项审核材料保存在本地忽略目录。

实际审计证据和 P2 进入状态见 [ArtBench 审计](../reports/v2/data/artbench_audit.md) 与 [当前交接](../reports/v2/00_handoff.md)。[来源登记](../configs/datasets/sources.yaml) 保留历史数据身份，[v1 清单示例](../configs/examples/manifest.example.jsonl) 保留旧 schema。

## **三、P2 冻结组件与正式缓存**

从 ArtBench 训练切分随机抽取 200 张，比较原图与 SDXL VAE 重建，记录 PSNR、SSIM、LPIPS、分位数和最差样本，并由用户检查边缘、细线、小物体、颜色与背景。旧几何图标的 96 张压力样本属于另一数据域。通过人工验收后构建正式潜变量缓存。

实际编码使用 VAE 后验均值，编码后乘模型配置的 `0.13025`，解码前除以同值；256px 输出为 `[b,4,32,32]`。CLIP 使用冻结清单的最终 `caption` 字段，`caption_cleaned` 是句子预算处理前的文本。输出 `[b,77,512]` 和有效 token 布尔掩码，空文本单独保存；DiT 的可训练文本投影不缓存。缓存身份绑定数据、Caption、预处理、组件锁、后验方式、缩放、mask 和精度。编码为 FP32，磁盘特征为 FP16。

缓存按 train 和 validation 组织，有序 ID 与 mmap 数组逐项对齐；不会编码 test。加载时核对清单、组件锁、数组摘要、形状和精度。使用与读取方法见 [P2 交付](../reports/v2/02_p2_delivery.md)。固定后验均值减少重复构建方差，其分布可能和预训练时的后验采样不同。

## **四、v1 历史数据与可复用资产**

| 数据 | 已有用途与证据 | v2 中的用途 |
| --- | --- | --- |
| MNIST | 官方 IDX 与固定子集，DDPM、VAE 教学与恢复测试。见 [S1 证据](../reports/stage1/s1_acceptance_2026-10-30.md) | 教学与回归 |
| 128px 合成几何图标 | 14,000 图，训练 10,000、验证 1,000，另有三类隔离测试。见 [数据卡](data_cards/synthetic_icons_week3.md)、[S2 完整缓存](../reports/stage2/week4.md) | 旧本地缓存和可重建图像已清理；保留生成器、规则解析器、配置和实验报告 |
| S3 文本与盲审 | 几何 Caption 的双段 CLIP 缓存和匿名人工评测协议。见 [S3 报告](../reports/stage3/stage3.md) | 复用缓存与盲审工程思路，在 ArtBench 绘画域重新校准 |
| OpenMoji | 旧 Sana LoRA 的计划来源。见 [历史来源登记](../configs/datasets/sources.yaml) | 归档的候选来源 |
| OpenClipart | v2 早期 SVG 与 VLM 标注试跑。[历史交付](../reports/v2/archive/openclipart/p0_p1_delivery_openclipart.md) | 已退出正式数据路线 |
