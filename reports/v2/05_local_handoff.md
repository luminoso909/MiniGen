## **一、P5 本地工程入口**

本轮停在这里，等待用户下一条指令。P4 的三主干代码和小规模本地闭环已交付；正式三模型训练与质量对比尚未完成。P3 诊断见 [收尾报告](03_p3_closeout.md)，P4 状态见 [工程报告](04_p4_delivery.md)。

下一项是 **P5-01 的 Benchmark 配置与协议实现/冻结**，随后逐项实现评分器和结果汇总的本地小案例；本轮未开始这些工作，也没有生成“正式 Benchmark 已通过”的记录。

## **二、可直接复用的接口**

| 入口 / 文件 | P5 的使用方式与边界 |
| --- | --- |
| `training.config.build_model` | b0/b1/b2 工厂，共用 `model(z,t,Condition)`，同形速度输出 |
| `contracts.Condition` | `[b,n,512]` 文本和 bool mask；模型内部投影仍可训练 |
| `RepresentationCache` | 冻结 train/validation、独立空文本；test 不反复用于选型 |
| `minigen sample` | 固定 checkpoint、EMA/online、Prompt JSON 或缓存条件、seed、CFG、Euler/Heun；保存 PNG/latent/JSON |
| `samplers.flow.integrate` | 记录 solver NFE 与包含 CFG 额外分支的实际主干调用；Heun 步数不能直接等同 Euler 步数 |
| `minigen evaluate` | **仅固定配对 Flow MSE**；不能当 KID/CMMD、CLIPScore 或主体/关系准确率 |
| `scripts/inspect_flow_models.py` | 实际张量形状的 Linear+QK/PV 主导计算量，非完整硬件 FLOPs |
| `scripts/profile_training.py` | 单卡 CUDA Pilot 入口，真正 CUDA 数字仍待用户执行 |
| `scripts/plot_flow_run.py` | 同一 JSONL 格式绘制损失、梯度和学习率 |

## **三、统一产物契约**

- `run.json`（`minigen.run.v1`）：配置、源码摘要、数据/子集身份、`run_id`、本次 `attempt_id` 和时间；输出目录不是实验数学身份的一部分。
- `last.pt` / `step_*.pt`（`minigen.flow-training.v2`）：显式 `objective=flow_velocity`，模型/EMA/优化器/scaler/日程/RNG/数据游标/历史。只读兼容 P3 已知 Flow v1；错误目标拒绝。
- `metrics.jsonl`：按 optimizer step 连续记录真实损失、验证值、LR、梯度范数、样本数、数据等待和步骤耗时。
- `result.json`（`minigen.training-result.v1`）：实际停止原因、训练步数、架构/参数、模型规格、源码/数据身份、累计步骤吞吐与内存统计口径。
- `failure.json`：独立 `attempt_id`、失败时间、错误及最后已发布检查点。遇到旧成功 `result.json` 与新失败回执并存，必须比较 attempt/time，不能仅看到 result 文件就认定最近一次成功。
- `samples.json`（`minigen.samples.v1`）：同一 `run_id`、checkpoint SHA/step、架构/模型规格、source/data 身份、seed、采样器/NFE/CFG、逐项 Caption 和生成图相对路径；生成图路径相对该样本结果目录，源图路径相对 `source_image_root`。
- 诊断 JSON（`minigen.flow-evaluation.v1`）：固定时间/噪声的逐项 MSE 和明确的 `generation_quality_evaluated=false`。

模型、数据、配置或源码不一致时拒绝精确恢复，不默默改为参考实现。跨设备可以加载权重采样；跨设备继续训练的数值轨迹不保证逐位一致。`source_sha256` 绑定维护的完整 `src/minigen`；源码变更后如需继续旧实验，应回到原提交/环境，不能重写旧身份。

## **四、P5 待实现清单**

1. 固定 ID/OOD/Paraphrase 提示词与种子、训练/验证/测试用途、指标和失败判定，先版本冻结再正式训练。
2. 实现批量生成与模型/CFG/NFE/Sampler 矩阵，所有失败和全部固定样本均入清单，禁止挑图或按测试集反复调参。
3. 实现质量、文本对齐、属性/关系与效率评分，记录模型版本和适用数据域；先校准评分器，不把画家名或风格变化误当主体质量。
4. 实现统一 CSV/JSON 汇总、模型对比画廊、失败分层与不确定性记录，分开同参数、同训练步骤和同算力预算。
5. 先用合成输入和极小真实样本验证代码链路，不把 smoke 检查点用于研究性架构优劣结论。
6. P5 主体稳定后另行制作 `docs/cloud/MiniGen_云端训练与评测完整操作手册.md`，所有命令来自真实实现；CUDA、DDP 未验部分标明。

正式 P5 实验仍需要可比的 B0/B1/B2 云端训练检查点、质量和预算审核。当前三模型只有本地工程烟测，不满足正式架构比较条件。下一阶段授权由用户给出，本轮不自动继续。
