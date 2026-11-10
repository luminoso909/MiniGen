## **一、配置导航**

ArtBench 固定版本、三个 Parquet 分片和抽样种子见 [正式数据配置](datasets/artbench_pd_256.yaml)。全量导入、分组及 Caption 审计见 [P1 审计](../reports/v2/data/artbench_audit.md)。

| 已有配置 | 状态与用途 |
| --- | --- |
| [来源登记](datasets/sources.yaml)、[清单示例](examples/manifest.example.jsonl) | 正式及历史来源登记，清单示例保留 v1 schema |
| [几何图标配置](datasets/icons_week3.yaml) | S2 128px 几何数据的固定参数 |
| [SDXL VAE 身份](models/sdxl_vae.json)、[CLIP 文本塔身份](models/clip_text.json) | v2 冻结组件的锁定身份；ArtBench 域表示和正式缓存证据见 [P2 交付](../reports/v2/02_p2_delivery.md) |
| [S2 第四周](experiments/s2_week4.yaml)、[S3](experiments/s3.yaml) | 原有教学实验配置 |
| [旧 S4 预登记](experiments/s4_preregistered.yaml) | 已归档的扩散与 Flow 配方比较，v2 正式实验按 [重构规划](../docs/MiniGen2重构规划.md) 另建版本 |

## **二、P3 诊断与 P4 工程配置**

- [B0 30M](models/b0_30m.json)：实际 30,359,824 参数，256px 图像对应 4×32×32 潜变量。
- [本地 500 步](training/p3_local.json)：固定 16 图索引；[1000 步上限](training/p3_local_extended.json) 只延长停止条件，日程与身份不变。
- [CPU tiny](training/p3_tiny.json) 与 [tiny 模型](models/b0_tiny.json)：只检查命令闭环，不作为正式质量证据。
- [CUDA Pilot](training/p3_cuda_pilot.json)：单卡、有效 batch 64、50+50 步性能测量；尚未执行，需用户另行授权环境。
- [单图诊断](training/p3_close_single.json) 与 [16 图有限对照](training/p3_close_sixteen.json)：独立运行目录，保留学习率、EMA 和采样对照；实际质量结论见 [P3 收尾](../reports/v2/03_p3_closeout.md)。

| 主干 | 模型规格 | 本地 8 步 | 恢复至 12 步 | 单卡 CUDA Pilot（未执行） |
| --- | --- | --- | --- | --- |
| B0，30,359,824 参数 | [B0](models/b0_30m.json) | [local](training/p4_b0_local.json) | [resume](training/p4_b0_resume.json) | [pilot](training/p4_b0_cuda_pilot.json) |
| B1，30,061,456 参数 | [B1](models/b1_30m.json) | [local](training/p4_b1_local.json) | [resume](training/p4_b1_resume.json) | [pilot](training/p4_b1_cuda_pilot.json) |
| B2，30,061,456 参数 | [B2](models/b2_30m.json) | [local](training/p4_b2_local.json) | [resume](training/p4_b2_resume.json) | [pilot](training/p4_b2_cuda_pilot.json) |

P4 三组本地配置使用相同 4 张训练图、有效 batch 4、FP32 和固定训练日程，仅验证工程闭环。恢复配置只延长停止步数，不改变日程与实验身份。三模型均完成 MPS 训练、恢复、采样与 Flow MSE 诊断；这些配置不作为正式架构质量比较。真实命令与产物见 [本地实验](../reports/v2/p4/local_experiments.md)，正式上云前置项见 [云端就绪清单](../reports/v2/p4/cloud_readiness.md)。

所有配置路径相对仓库根，CLI 可用 `--root` 指向其他资产根。模型 JSON 保留完整规格，训练配置与数据选择独立；未知字段明确失败。目标固定为 `flow_velocity`，不接受噪声预测权重混用。新实验先复制配置并设置新 `output_dir`，恢复须保留完整配方、源码哈希和数据身份；读入旧 P3 权重不能绕过跨源码恢复限制。CUDA/AMP 尚待实机验收，多卡尚未实现。

## **三、P5 Benchmark 与云端配置**

- `models/evaluation.json`：独立 Inception/完整 CLIP/BLIP 的精确下载与离线文件身份；大权重不在仓库。
- `benchmark/prompt_catalog.json`：候选题族与固定种子；生成后由 `reports/v2/benchmark/artbench_v1/lock.json` 固定内容，用户批准另存。
- `benchmark/sampling_matrix.json`：Euler/Heun 匹配 solver NFE、CFG 和 naive/SDPA 单变量扫描。
- `experiments/p5_comparison.json`：三架构/30M/seed0 与后续扩展身份；目标预算、最终采样设置及正式批准仍为空或待确认。
- `scripts/prepare_cloud_run.py`：依据对应真实 CUDA Pilot 生成全数据正式配置，不把 P4 的 12 步 smoke 配置冒充正式配方。

真实命令与参数修改、暂停恢复、正式门槛见 [云端手册](../docs/cloud/MiniGen_云端训练与评测完整操作手册.md)。


## **四、云端前候选预登记**

[precloud_preregistration.json](experiments/precloud_preregistration.json) 是数据、组件、模型和评测身份以及待测超参数的机器可读登记，**不是 train 可执行配置**，不自动批准长训。说明见 [正式实验预登记](../reports/v2/final/MiniGen_正式实验预登记.md)。待真实 CUDA Pilot 和用户批准后，才用 `scripts/prepare_cloud_run.py` 生成正式三模型配置。
