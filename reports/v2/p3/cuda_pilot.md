## **一、P3-06 状态：待云端验收**

本轮没有 NVIDIA GPU，**没有运行 CUDA、没有云端消费、没有批准正式长训**。已准备有界 Pilot 和 CPU/MPS 共享训练接口；本文件是 P3 的执行入口说明，不是完整云端部署手册。

## **二、未来单卡执行入口**

在用户另外提供并授权 CUDA 环境、恢复固定 P2 数据和组件资产后，从仓库根目录执行：

```bash
uv run --locked --no-sync minigen doctor --device cuda --json
uv run --locked --no-sync python scripts/profile_training.py \
  --config configs/training/p3_cuda_pilot.json \
  --output artifacts/runs/p3_cuda_pilot_run01 \
  --warmup 50 --measure 50 --precision bf16
```

配置保持 B0 30,359,824 参数，潜空间 4×32×32；微批 8、累积 8，有效 batch 64。256 条缓存仅供性能试跑，不当作全量正式训练。显存不足时记录失败，修改独立 Pilot 配置重新测量，不能静默改变 batch 后沿用旧吞吐。CPU/MPS 无 CUDA 时入口返回 2，**不生成通过回执**。

如需估算指定训练步数费用，显式传入 `--planned-steps` 与 `--hourly-cny` 的实际值；程序只估算优化步骤，Checkpoint、采样、验证、存储和传输另计。硬件价格本轮未查询，也未填造报价。

## **三、必须验收与保存的项目**

1. GPU 名称、驱动/CUDA/PyTorch、设备精度与配置；驱动另用 `nvidia-smi` 保存。
2. 50 步预热后实际测量 50 步，有限损失/梯度；每次计时含设备同步。
3. 每步耗时、吞吐、数据等候比例、CUDA 峰值 allocated/reserved、CPU peak RSS、优化器及 EMA 张量逻辑字节数。峰值范围包含预热，逻辑字节数不等于分配器峰值。
4. `pilot.json`、`run.json`、`metrics.jsonl`、`last.pt` 位于指定输出目录；不足所需步数视为未完成。
5. FP32 设备对照可复用 `scripts/check_flow_device.py --device cuda`；CUDA AMP 数值稳定性与恢复需实机另外验收。本机通过不能替代。
6. 未来正式训练预算还取决于短训质量、Caption 噪声决定和用户审批。多卡尚未实现，不提供会误导的 `torchrun` 启动命令。

本地已验证解析帮助和 CUDA 不可用时的明确失败；性能数字保持空缺。完整云端手册等 P3–P5 主体实现稳定后另行制作。
