## **一、运行身份与范围**

2026-11-12 在用户已开启的 AutoDL vGPU 上执行 B0 长训前工程检查和一次有界短训。远端为 `autodl:/root/autodl-tmp/minigen`，本轮目录为 `artifacts/runs/autodl-b0-pretrain-20261112/`。提交为 `f0ea839925e7303832252de5c3359ba278751fe4`。RTX 4080 SUPER 可见显存 32,760 MiB，开跑前空闲，驱动 595.71.05。

复用镜像内 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128 与既有的本地组件依赖，运行时 CUDA 12.8，支持 BF16。目标机正式 `uv.lock` 环境验收仍待完成。Mac 通过 rsync 发送四个小脚本，远端 Python、缓存编码、训练及采样均在 SSH 会话中执行。生产源码与冻结配置使用上述提交。沿用之前 AutoDL 单图拟合回执 `artifacts/runs/autodl-b0-smoke-20261112-162249.Mohf/smoke.json`。

## **二、工程验收**

| 项目 | 实测 | 远端回执（相对于本轮目录） |
| --- | --- | --- |
| CUDA doctor 和定向回归 | CUDA 前后向通过，39 项测试通过，覆盖 Flow、训练恢复、三主干、CLI、缓存、数据和 Benchmark | `acceptance/doctor_cuda.json`、`acceptance/pytest.log` |
| 隔离诊断缓存 | 272 条训练、16 条验证。训练前 256 条用于吞吐，另 16 条为冻结质量子集。SDXL VAE、CLIP 在 CUDA 上 FP32 编码，独立重开逐条核对 | `pilot_cache_build.json`、`pilot_cache_readback.json` |
| B0 FP32 CPU/CUDA | 固定两条表示与同一检查点，前向最大绝对误差 `5.12e-9`，梯度最大绝对误差 `3.73e-8`，以 `atol=rtol=2e-4` 通过 | `acceptance/b0_cpu_cuda_fp32.json` |
| B0 BF16 精确恢复 | 连续 8 步与 4+4 的模型、EMA、优化器、scaler、日程、数据流、RNG 和逐步训练轨迹逐位一致 | `acceptance/b0_cuda_resume_comparison.json` |
| B0 BF16 性能 Pilot | 50 步预热、50 步测量，microbatch 8、累积 8，有效 batch 64。0.466 秒/步，137.43 图/秒，显存峰值 allocated 1.12 GiB、reserved 1.16 GiB | `profile/pilot.json` |
| 正式缓存 | `--verify-only` 失败，AutoDL 缺少预登记 `v1_formal/index.json`。已批准的 Mac MPS 缓存仍是正式版本 | `acceptance/formal-cache-readback.log`、`acceptance/formal-cache-readback.exit` |

首次定向测试启动时，镜像自带的 `pluggy` 版本与既有 pytest 不兼容。重试保留 CUDA PyTorch 和 torchvision，pytest 优先读取已有依赖，39 项全部通过。首次失败日志为 `acceptance/pytest-env-first-attempt.log`。

## **三、B0 固定 16 图短训**

训练沿用 `configs/training/p3_close_sixteen.json` 的冻结样本、FP32、batch 4、学习率 2e-4、EMA 0.99、seed 0。先完成 500 步，再从同一运行的检查点续到 1,000 步，共用 1,000 步学习率日程。独立验证使用冻结清单前 16 条。两划分各用固定 seed 1234 对照正确、空、循环打乱三种文本，采样为 EMA、Euler 40、CFG 1。

训练循环累计 **68.41 秒**，日志包含 1,000 次更新及 4,000 次样本读取。前 20 步平均训练 loss 为 **1.6911**，末 20 步为 **0.4577**。固定验证 loss 从第 50 步的 **1.5915** 升至第 1,000 步的 **1.9343**。两次训练调用的峰值 allocated 约为 0.82 GiB，峰值 reserved 约为 0.90 GiB。

| 固定配对 Flow MSE | 正确文本 | 空文本 | 打乱文本 |
| --- | ---: | ---: | ---: |
| 训练 16 张 | 0.4443 | 0.9751 | 2.3144 |
| 验证 16 张 | 1.7743 | 1.0059 | 1.8582 |

**P3-05 质量未通过。** 训练样本 `ab-30577` 的黑帽、人物轮廓和衣服可辨，其他部分训练图复现了粗略配色或布局，多数仍有明显的块状伪影。16 张验证图的生成主体普遍难以辨认，正确文本未稳定改善主体或布局。训练集上正确条件的固定配对误差最低，验证集上正确条件的误差高于空文本，支持本次条件作用集中在训练样本。上述误差度量 Flow 速度。

4 张训练图的独立数值诊断中，解析积分的最大终点误差为 `4.77e-6`，VAE 缩放因子为 `0.13025`。EMA Euler40、Heun20、Euler200、Euler40 CFG3 的目标 latent MSE 分别为 0.17966、0.18575、0.18684、0.32781。在线权重 Euler40 为 0.13387。提高积分预算或改用 Heun 在该诊断中未降低误差。所有时间切片、参数和输出保存在 `diagnosis/`。

[训练图 1–8](../../../artifacts/exports/autodl-b0-pretrain-20261112/panels/train_01-08.png) · [训练图 9–16](../../../artifacts/exports/autodl-b0-pretrain-20261112/panels/train_09-16.png) · [验证图 1–8](../../../artifacts/exports/autodl-b0-pretrain-20261112/panels/validation_01-08.png) · [验证图 9–16](../../../artifacts/exports/autodl-b0-pretrain-20261112/panels/validation_09-16.png)

## **四、产物和复现**

- 本轮 run ID 为 `17de5df3d8181989e60251aaed12a25cf13085063354799f758819d1d59759ce`，运行源码 SHA-256 为 `a27dc33a3c6e1946907be183467e2ac340afb67e4c1bfb8d2b9a303d1d5aac36`。诊断缓存 index SHA-256 为 `4e7d55edc3cfaf13959d41cd75783d0bccb2c729087a1f34a12755a919624a81`。
- `quality/last.pt` 与 `quality/step_000500.pt` 均在 AutoDL 数据盘，哈希和本轮七份配置哈希见 `acceptance/run_files_sha256.txt`。96 张原始样图在 `samples/`，顺序、条件、初噪设置与检查点哈希核对结果见 `panels/samples_verified.json`。
- 脚本在 Mac 的 `artifacts/exports/autodl_pretrain_{pilot,resume,panels}.py` 和 `artifacts/exports/autodl_pretrain_run.sh`，执行时的脚本哈希见 `acceptance/runner_sha256.txt`。面板标题的字体修正仅重排已生成图像，修正版脚本哈希见 `acceptance/panels_renderer_sha256.txt`。已有训练与采样入口通过进程内缓存读取器接入隔离子集。正式 `minigen cache --verify-only` 独立运行。
- Mac 回迁了面板、配置、日志和 JSON 回执。完整的模型检查点、诊断缓存及逐张原图尺寸生成结果保留在 AutoDL。

在同一个已就绪的 AutoDL 环境，确认 GPU 空闲后，用新的运行目录复现。

```bash
ssh autodl 'cd /root/autodl-tmp/minigen && bash artifacts/exports/autodl_pretrain_run.sh artifacts/runs/autodl-b0-pretrain-reproduction'
```

## **五、正式长训状态**

正式长训和正式 Test 均未启动。B0 质量门槛仍未通过，B1/B2 各自 CUDA Pilot 继续待做。正式缓存的目标机回读、正式锁环境、个人核心验收、Benchmark 定稿、共同训练配方和预算仍待完成。平台时价及账单未提供，实际费用待查。16:44 完成全部诊断流程，核对 GPU 空闲，实例仍开启。
