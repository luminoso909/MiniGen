## **一、当前停止点**

最新交接已补 [云端前工程最终验收](final/P3_P4_P5工程最终验收摘要.md)、[Benchmark 最终审核包](benchmark/MiniGen_Benchmark_最终审核包.md) 和 [预登记](final/MiniGen_正式实验预登记.md)。本页保留 P5 原交接；当前待处理问题与唯一下一步以这些材料及总交接为准。

P5 本地主体工程、真实小样本与云端手册已交付。**唯一下一任务是用户审核候选 Benchmark 与完整手册，然后亲自执行单卡 CUDA Pilot。** 不自动启动正式长训、P6/P7 或额外规模实验。[阶段报告](05_p5_delivery.md) · [当前总交接](00_handoff.md)

## **二、云端前置清单**

| 顺序 | 用户下一步 | 证据或入口 |
| --- | --- | --- |
| 1 | 审阅题目自然性、条件是否可判、题族近重复与组合压力边界；批准后单独留回执 | [Benchmark](benchmark/README.md)，当前只是内容冻结 |
| 2 | 审阅部署、预算、资源与恢复步骤，同步本轮真实提交和已有资产 | [操作手册](../../docs/cloud/MiniGen_云端训练与评测完整操作手册.md)，本轮未 Push |
| 3 | CPU/CUDA、AMP、恢复、实际 GPU 型号/吞吐/显存 Pilot | 现有 check_flow_device.py 与 profile_training.py；CUDA 未验 |
| 4 | 恢复 B0 单图/16 图与条件质量诊断，达标才批准长训 | P3-05 仍未通过；不靠 Loss 或烟测指标解除门槛 |
| 5 | 用匹配 Pilot 生成三架构正式配置，冻结比较预算与采样选择规则 | prepare_cloud_run.py；费用和完整步数由用户填入 |
| 6 | 准备 BLIP 权重、真实 100+100 人工校准，执行正式生成和评分 | bench review / score；本轮 Mock 不算成绩 |

所有具体命令及失败恢复见手册。DDP 未实现；首轮只做单卡。更换源码、配方或硬件不能继续宣称精确同轨迹恢复；旧 P4 权重仍可评测，但不保证跨源码续训。

## **三、已可复用的工程**

B0/B1/B2 共用训练、Flow/CFG/Euler/Heun、缓存/检查点身份与 CLI；P5 追加可恢复任务清单、独立评测特征、人工页面和报告。模型参数与 P2 缓存不变。训练回执新加硬件元数据，中间资源可由真实 checkpoint 导出；FLOPs 倍数估计和硬件计时分别报告。

真实图片目录为 `artifacts/evaluations/p5/smoke/{b0,b1,b2}/`，完整画廊为 `artifacts/evaluations/p5/smoke/report_final/gallery.html`。这些仍是训练不足的噪声纹理，不代表艺术生成已成功。小型回执在 `reports/v2/experiments/p5/`，完整图片、权重和缓存不入 Git。

本轮没有人工 Benchmark 批准、正式校准或正式架构优劣结论。数据 Caption 噪声、VAE 高频细节损失、历史 16 图质量失败继续保留，不删数据、不降低验收标准。
