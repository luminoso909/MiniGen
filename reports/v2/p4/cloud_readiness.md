## **一、云端就绪状态**

本轮只完成可在 Mac 上验证的工程。**尚未证明 Linux/CUDA、BF16/FP16 AMP 或 GPU 性能通过**，没有租赁服务器或付费 GPU 操作。完整云端操作手册将在 P5 主体代码完成后另行生成，目标路径仍为 `docs/cloud/MiniGen_云端训练与评测完整操作手册.md`；本文件不提前给出未实现的正式评测命令。

| 能力 | 当前状态 | 上云前/上云后还需做什么 |
| --- | --- | --- |
| B0/B1/B2 随机初始化、统一 Flow | 已实现约 30M 主规格，独立 CPU 数学和梯度测试 | 固定 Git 提交、模型配置与软件环境 |
| ArtBench 与冻结 VAE/CLIP | 沿用 P2；未重建缓存、未改 Caption、256×256 不变 | 同步锁文件、Manifest/Caption 和组件资产；独立回读缓存 |
| 单卡 train/sample/evaluate | CPU/MPS 已验证；evaluate 仅固定配对 Flow MSE | CUDA 短跑验证，P5 质量指标尚待实现 |
| Checkpoint / resume | 模型、EMA、优化器、日程、scaler、数据流与 RNG 完整记录 | 相同提交/设备/配方恢复；不得把 MPS 轨迹当 CUDA 精确恢复 |
| CUDA FP32 / AMP | 代码分支与 Pilot 配置已准备，未实测 | 逐模型测试 BF16 支持、有限梯度和保存/恢复 |
| 单卡吞吐、显存和成本 | 测量入口可用，本机观测只供烟测 | 在拟租 GPU 做 50 步预热 + 50 步计时，记录真实单价再估算 |
| 多卡 DDP | **未实现** | 需新增 rank/world-size、分布式采样/RNG、有效 batch 及跨 rank 恢复测试，不提供伪装可用的 torchrun 命令 |
| 正式三模型质量 | **未完成** | B0 16 图短训仍不可辨；先处理诊断和质量门槛，不据 smoke 判优劣 |
| P5 Benchmark | **未实现/未冻结** | 在云端正式训练前冻结题目、划分、采样、指标与选型协议 |
| 正式训练预算 | **未批准** | 根据实机 Pilot 和质量改善证据由用户决定 |

## **二、已经具备的单卡 Pilot 入口**

以下命令只在用户今后提供且授权的 CUDA 环境执行。本轮没有运行它们，三份配置共用 batch=8、累积=8、有效 batch=64，模型规格各自独立。

```bash
uv run --locked --no-sync python scripts/profile_training.py --config configs/training/p4_b0_cuda_pilot.json --output artifacts/runs/cloud_b0_pilot01 --warmup 50 --measure 50 --precision bf16
uv run --locked --no-sync python scripts/profile_training.py --config configs/training/p4_b1_cuda_pilot.json --output artifacts/runs/cloud_b1_pilot01 --warmup 50 --measure 50 --precision bf16
uv run --locked --no-sync python scripts/profile_training.py --config configs/training/p4_b2_cuda_pilot.json --output artifacts/runs/cloud_b2_pilot01 --warmup 50 --measure 50 --precision bf16
```

产物是指定目录中的 `pilot.json`、`run.json`、`metrics.jsonl` 与 `last.pt`。实际完成全部预热/计时步、有限损失与梯度、可恢复检查点，才可登记该模型的 GPU Pilot 通过。显存不足时修改独立配置并重新测量，不沿用改变 batch 前的数字。使用 `--hourly-cny` 和 `--planned-steps` 记录实际报价与拟定步数，估算另计存储、保存、评测和重试。

缓存从 MPS FP32 编码后以固定 FP16 表示存储，可供 CUDA 训练读取；这并不免除云端独立回读和前向数值验收。已有三主干 `scripts/check_flow_device.py --device cuda` 接口可做固定 CPU/CUDA FP32 对照，AMP 另测。训练未实现分布式路径，第一版仍为单卡。

## **三、正式训练前仍待用户或后续阶段确认**

- 接受当前冻结 Caption 的事实噪声，还是另开带审核记录的数据修订版；本轮未改变数据。
- B0 短训质量诊断结论及后续有限实验范围；单图记忆成功不能替代多图可辨质量。
- P5 的提示词、数据划分、基线和评分器可靠性，在正式模型训练前冻结。
- 云主机、持久化数据盘、依赖/驱动、权重迁移与恢复实测；没有使用现有 SSH 别名或凭据。
- 按 GPU 实测成本决定三模型共同训练资源和中间检查点预算。
- P5 完成本地代码与小案例后，按真实入口编写完整新手云端手册，包含部署、下载、训练、恢复、评测、备份和回迁 Mac。
