## **一、本地数据目录**

ArtBench 的固定 Parquet 分片放 `raw/artbench_pd_256/<revision>/data/`。离线导入后，图像、版本化 JSONL 清单和审计包归 `processed/artbench_pd_256/v1/`。历史教学清单沿用 `manifests/`。实际数据由 Git 忽略，来源和质量记录另有小型文档。[数据规范](../docs/data.md) · [P1 审计](../reports/v2/data/artbench_audit.md)
