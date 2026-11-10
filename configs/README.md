## **一、配置导航**

ArtBench 固定版本、三个 Parquet 分片和抽样种子见 [正式数据配置](datasets/artbench_pd_256.yaml)。全量导入、分组及 Caption 审计见 [P1 审计](../reports/v2/data/artbench_audit.md)。

| 已有配置 | 状态与用途 |
| --- | --- |
| [来源登记](datasets/sources.yaml)、[清单示例](examples/manifest.example.jsonl) | 正式及历史来源登记，清单示例保留 v1 schema |
| [几何图标配置](datasets/icons_week3.yaml) | S2 128px 几何数据的固定参数 |
| [SDXL VAE 身份](models/sdxl_vae.json)、[CLIP 文本塔身份](models/clip_text.json) | v2 冻结组件的锁定身份；ArtBench 域表示和正式缓存证据见 [P2 交付](../reports/v2/02_p2_delivery.md) |
| [S2 第四周](experiments/s2_week4.yaml)、[S3](experiments/s3.yaml) | 原有教学实验配置 |
| [旧 S4 预登记](experiments/s4_preregistered.yaml) | 已归档的扩散与 Flow 配方比较，v2 正式实验按 [重构规划](../docs/MiniGen2重构规划.md) 另建版本 |

## **二、P3 工程配置**

- [B0 30M](models/b0_30m.json)：实际 30,359,824 参数，256px 图像对应 4×32×32 潜变量。
- [本地 500 步](training/p3_local.json)：固定 16 图索引；[1000 步上限](training/p3_local_extended.json) 只延长停止条件，日程与身份不变。
- [CPU tiny](training/p3_tiny.json) 与 [tiny 模型](models/b0_tiny.json)：只检查命令闭环，不作为正式质量证据。
- [CUDA Pilot](training/p3_cuda_pilot.json)：单卡、有效 batch 64、50+50 步性能测量；尚未执行，需用户另行授权环境。

所有配置路径相对仓库根，CLI 可用 `--root` 指向其他资产根。模型 JSON 保留完整规格，训练配置与数据选择独立；未知字段明确失败。新实验使用新输出目录，恢复须保留完整配方和源代码身份。已知短训失败及实际命令见 [P3 报告](../reports/v2/03_p3_delivery.md)。
