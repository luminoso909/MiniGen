## **一、配置导航**

ArtBench 固定版本、三个 Parquet 分片和抽样种子见 [正式数据配置](datasets/artbench_pd_256.yaml)。全量导入、分组及 Caption 审计见 [P1 审计](../reports/v2/data/artbench_audit.md)。

| 已有配置 | 状态与用途 |
| --- | --- |
| [来源登记](datasets/sources.yaml)、[清单示例](examples/manifest.example.jsonl) | 正式及历史来源登记，清单示例保留 v1 schema |
| [几何图标配置](datasets/icons_week3.yaml) | S2 128px 几何数据的固定参数 |
| [SDXL VAE 身份](models/sdxl_vae.json)、[CLIP 文本塔身份](models/clip_text.json) | v2 冻结组件的锁定身份；ArtBench 域表示和正式缓存证据见 [P2 交付](../reports/v2/02_p2_delivery.md) |
| [S2 第四周](experiments/s2_week4.yaml)、[S3](experiments/s3.yaml) | 原有教学实验配置 |
| [旧 S4 预登记](experiments/s4_preregistered.yaml) | 已归档的扩散与 Flow 配方比较，v2 正式实验按 [重构规划](../docs/MiniGen2重构规划.md) 另建版本 |
