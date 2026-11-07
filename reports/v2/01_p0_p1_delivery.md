## **一、P0 与本轮结论**

P0 的仓库盘点、迁移与旧 DiT 启动测试修正已交付。P1 正式数据源已切换为 **ArtBench Public Domain 256x256**。全量 **46,298 张**已在本地验证、分组、清洗并冻结到 `v1`。可以进入 **P2 组件验收**。正式条件训练仍需处理本次发现的 Caption 语义噪声并经过后续阶段的验收。[P0 审计](00_repository_audit.md) · [当前交接](00_handoff.md) · [ArtBench 审计](data/artbench_audit.md)

## **二、切换和清理范围**

| 类别 | 本轮文件与数据 |
| --- | --- |
| 新增数据入口 | `configs/datasets/artbench_pd_256.yaml`、`scripts/prepare_artbench.py`、`src/minigen/data/artbench.py`、`src/minigen/data/artbench_caption.py`、`tests/test_artbench_caption.py` |
| 新增证据 | `reports/v2/data/artbench_audit.md`、`artbench_freeze.json`、`artbench_visual.json`、`artbench_cleanup.json`；本地 Parquet、图像、正式清单及审计页 |
| 修改入口与文档 | `pyproject.toml`、`uv.lock`、来源登记、README、规划、任务索引、学习目录、数据规范、下载清单、Agent 约定、报告导航与阶段交接 |
| 下线旧专用代码与配置 | `configs/datasets/openclipart_v2.yaml`、两个下载审计脚本、两个标注与复核脚本、`openclipart.py`、`openclipart_groups.py`、`openclipart_preview.py` 和两个 OpenClipart 专用测试 |
| 删除本地旧数据 | OpenClipart 原始下载及续传文件、旧 Xet 缓存、预览、试切分、Caption 草稿、复核页和旧下载目录。共清理 940 个文件，按逻辑大小约 1.86 GB，含上行的专用源码、配置与其编译缓存 |
| 保留通用资产 | 现有清单加载器、训练与评测、SVG 安全渲染及其 17 个通用测试（迁到 `tests/test_svg_render.py`）；SDXL VAE、CLIP 与 LPIPS 资产 |
| 历史资产 | P1 当时暂留的 QwenVL 权重和专用环境已于 P2 清理。OpenClipart 的小型聚合证据和 2B 资产锁留在历史档案；旧清单与下载入口下线 |

PixelArt-308K 只登记为参考。P1 清理逐路径记录在 [清理回执](data/artbench_cleanup.json)，旧事实见 [历史交付](archive/openclipart/p0_p1_delivery_openclipart.md)。

## **三、数据冻结与审计**

| 项目 | 固定值 |
| --- | --- |
| 来源与版本 | `alfredplpl/artbench-pd-256x256@4056b70025f0e3f916fb271c7a792633ba637484`，MiniGen `v1` |
| 样本与像素 | 46,298 张全量保留，原生 256×256 RGB JPEG，解码失败 0 |
| 使用字段 | `image` 和英文 `caption`。`caption_jp`、`artist` 和 `url` 用于溯源 |
| Caption | 保留视觉事实和基本题材词，清理可分离的艺术史文本，保存原文、动作和训练版。全部长度 ≤75 个内容 token |
| 项目划分 | 固定分组种子 `20261107`，训练 37,042、验证 4,637、测试 4,619 |
| 本地位置 | `data/raw/artbench_pd_256/<revision>/data/` 与 `data/processed/artbench_pd_256/v1/` |

从空目录独立重建，清单、摘要和抽样队列逐字节一致。全部 46,298 张文件哈希通过，组 ID、文件哈希、像素哈希和规范化来源 URL 均无跨集合冲突。[清单冻结回执](data/artbench_freeze.json)

代理逐项核对 240 对训练图文。229 张未见明显技术性缺陷，11 张待对照原作确认裁切或复制表现。Caption 有 71 条明确事实错误，47 条难以判定，122 条未见明显错误。原始 Caption 来自 Florence-2-large，清洗仅处理文本表达。人工复核为 0，当前语义数据适合做带噪声说明的组件验收输入。正式条件训练前要决定是否修订描述。若准备再分发作品或数字副本，另核对其许可。[图文审计和案例](data/artbench_audit.md) · [本地复核页](../../data/processed/artbench_pd_256/v1/audit/review.html)

## **四、验证与下一任务**

全库 pytest **131 项通过**。变更范围的 Python Ruff、格式与注解检查通过，CPU doctor 和离线锁校验通过，236 个相对文件链接已检查。全库注解检查仍报告 37 条既有问题。五份 S1–S3 Notebook 与本轮起点哈希相同。另观察到三份较新的 S5/S6 Notebook 在工作期间发生外部改动，数据切换未写入这些文件。

本轮在本机完成数据准备与审计，云和付费 API 费用为 0。P2 资产已备齐，下一任务为 **P2-01** 从冻结训练划分随机取 200 张进行 VAE 重建和人工细节验收。[P2 资产回执](components/p2_assets.json)
