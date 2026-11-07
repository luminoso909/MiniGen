## **一、P2 工程验收**

P2 使用同一冻结版 ArtBench Public Domain 256x256：46,298 张原生 RGB 绘画，train 37,042、validation 4,637、test 4,619。清单 SHA-256 为 `0b95e7b783b2b62d6e2ff58646a95d173607a61b4184c6e250db7dfa1303cc0a`。这次没有下载新权重、训练模型或使用收费资源。

| 任务 | 实际状态 | 证据 |
| --- | --- | --- |
| P2-01 随机重建 | CPU FP32 重建 200 张固定随机训练样本及 9 张诊断样本。随机样本均值 PSNR 26.76 dB、SSIM 0.744、LPIPS 0.080 | [方法、数值与失败样本](components/vae_reconstruction_200.md) · [结构化结果](components/vae_reconstruction_200_summary.json) |
| P2-02 选定 VAE | 用户完成视觉审核，批准保留冻结的 `madebyollin/sdxl-vae-fp16-fix`，用于一般艺术绘画。细密线条、文字和小物体的高频损失记为已知局限；不因此删除样本，不更换或微调 VAE | [用户批准](components/vae_user_approval.json) · [200 张统计、9 张诊断和 LPIPS 最差 20 张](components/vae_review_evidence.json) |
| P2-03 CLIP | 46,298 条最终 Caption 离线复分词，长度 12–75；0 条空、超限、特殊 token 或计数不符。真实样本检查 mask、空条件、条件更换及下游梯度 | [文本验收](components/clip_text_acceptance.md) · [结构化结果](components/clip_text_acceptance.json) |
| P2-04 256px 表示缓存 | 本机固定 20 张对照通过；VAE 和 CLIP 使用 MPS FP32 构建正式缓存。train 37,042、validation 4,637 共 41,679 行全部通过独立回读，test 未编码 | [本机固定 20 张回执](components/artbench_local_fp32_20.json) · [正式缓存回执](components/artbench_cache_formal.json) |

原始 200 张重建汇总保留测量当时的待审核状态和哈希；以单列的用户批准作为当前视觉审核结论。完整图片材料与 ZIP 保留在本地 `reports/v2/components/vae_review/`，不纳入 Git。CLIP 的条件更换对照只说明特征能进入下游可训练投影，不能证明 B0 的提示词遵循。P1 审计在 240 对中观察到 71 条明确视觉事实错误；长度、mask 或梯度验收不会消除这类噪声。

## **二、表示缓存契约与运行**

缓存对每个样本返回 `(sample_id, latent, text, mask)`。潜变量为 FP16 `[4,32,32]`，由 FP32 后验均值乘 VAE 配置中的 `0.13025`；文本为 FP16 `[77,512]`，掩码为 bool `[77]`。空描述由同一 CLIP 编码，独立于任何配对行，只有 BOS 和 EOS 有效。模型读取时可按训练精度转换；可训练的文本投影留给 B0/B1/B2。

清单顺序、样本 ID、两套组件锁、预处理、Caption 策略、mask 和 dtype 进入身份。构建器验证源图摘要、尺寸和 Caption token 长度，把数组分批写入暂存目录并记录已完成的进度，按相同身份和参数可断点恢复，完整后原子发布。读取器独立派生预期身份，核对有序 ID、用户审核和本机数值回执、各数组 SHA-256、形状及 dtype，再通过只读 mmap 按行访问。缓存应视作只读快照；复制或搬迁后重新运行验证。

[CPU 试点](components/artbench_cache_pilot.json)实际耗时 12.108 秒。32 对特征重新编码后与磁盘 FP16 数值逐项相等，掩码和 ID 一致；FP16 量化最大绝对误差为潜变量 0.000976、有效文本位置 0.007775（仅试点，不外推成全量误差保证）。

MPS 试点在 Apple Silicon M5 上使用冻结 VAE 和 CLIP FP32 前向，关闭 MPS 算子回退和快速数学，批量 16。实际写入并回读 train 128、validation 128、空文本条件，耗时 **27.126 秒**。本地路径为 `artifacts/cache/artbench_pd_256/v1_mps_preflight`，`index.json` SHA-256 为 `f98e48422e462a5325f482b81359b1e9133c2c8733f3af7bc7920237c20f3c9b`。试点说明批量编码和写盘能够运行，不能代替固定 20 张的 CPU 数值对照或正式缓存验收。

从 MPS 试点的 train、validation 各独立回放 16 行。批量 4 与批量 16 的 FP32 潜变量、有效文本特征逐项相同；重新编码后转换为 FP16，与试点磁盘数组逐项相同，样本 ID 和掩码也一致。[批量回放回执](components/artbench_mps_batch_replay.json)。

### **1. 本机数值验收与 CUDA 边界**

正式缓存的前置检查由 `scripts/check_artbench_components.py` 执行。固定原随机重建队列的前 20 个样本，核对输入、VAE 后验 mode/均值、缩放往返、输出稳定性，以及 CLIP token、有效 mask、文本特征和空条件；以 CPU FP32 为参考分别判定两个组件的 MPS 兼容性。某组件在 MPS 不能运行或超出事前误差门槛时，才显式选择已验证的 CPU，回执保留原因。早先提出 CUDA 检查是为了直接覆盖未来训练设备；冻结编码与 NumPy 缓存构建不需要 CUDA。按最新决定，P2 本机正确性与后续 CUDA 训练运行分开，后者未执行。

**实测通过。** Apple Silicon M5、PyTorch 2.14.1、CPU 5 线程、批量 4，CPU 与 MPS 均使用 FP32，关闭 MPS 算子回退和快速数学。整个验收耗时 **86.912 秒**。下表的均值为各样本平均绝对误差中的最大值，最大值为所有比较元素的最大绝对误差；CLIP 只统计下游使用的有效 token，另检查所有输出有限。

| CPU ↔ MPS 数值对照 | 最大样本平均绝对误差 | 元素最大绝对误差 | 对应门槛（均值 / 最大值） |
| --- | ---: | ---: | ---: |
| VAE 缩放后潜变量 | 0.00001486 | 0.00376588 | 0.002 / 0.035 |
| VAE 原始解码 | 0.00004648 | 0.01033659 | 0.003 / 0.100 |
| CLIP 有效 token 特征 | 0.00000199 | 0.00004482 | 0.005 / 0.080 |

RGB 归一化、VAE 的 mode 对后验均值和正向缩放均与对照逐项相同，逆缩放最大误差为 0.000000954；本轮 CPU FP32 原始解码与旧审计对应 20 张逐项相同。VAE 潜变量和原始解码、CLIP 文本特征在各自同设备重复执行时逐项相同；两设备的 CLIP token ID 和掩码完全一致，空文本恰有 BOS/EOS 两个有效位置。选择 **VAE MPS、CLIP MPS**，无需回退 CPU。结果只覆盖固定 20 张及空条件。

### **2. 构建和回读**

从仓库根目录运行；先恢复本地资产，按需同步 `latent`、`conditioning` extras。数值验收复用原 200 张实验保留的本地 `samples.json` 和 `decoded.pt`，先校验哈希，再计算独立 CPU FP32 参考和 MPS 对照。复跑请输出到新文件。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  uv run --locked --offline --no-sync python scripts/check_artbench_components.py \
  --cpu-threads 5 --output artifacts/evaluations/p2_local_20/receipt_rerun.json
```

生产命令不会下载模型。已存在的 CPU 试点可这样独立回读：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --offline --no-sync python scripts/build_artbench_cache.py \
  --pilot-rows 16 --output artifacts/cache/artbench_pd_256/v1_pilot --verify-only
```

生成新的试点时去掉 `--verify-only` 并指定新目录。正式构建依次使用通过验收的用户批准与本机数值回执，VAE 与 CLIP 的编码设备取自后者。2026-11-07 22:41:29（Asia/Shanghai）已启动正式构建，批量 16、CPU 5 线程，日志位于 `artifacts/evaluations/p2_formal_cache/run.log`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  uv run --locked --offline --no-sync python scripts/build_artbench_cache.py \
  --approval reports/v2/components/vae_user_approval.json \
  --evidence reports/v2/components/vae_reconstruction_200_summary.json \
  --local-acceptance reports/v2/components/artbench_local_fp32_20.json \
  --batch-size 16 --cpu-threads 5 \
  --output artifacts/cache/artbench_pd_256/v1_formal
```

中断后以相同参数重跑，包括批大小和 CPU 线程数。恢复时先校验已提交的缓存段；尚未提交的批次可以重算。完整发布后在新进程对相同目录运行 `--verify-only`，不加载编码器。原始构建和独立回读回执分别保留在 `artifacts/evaluations/p2_formal_cache/build.json` 与 `verify.json`。暂存目录或单独的构建回执不能作为 P2-04 完成证据。

P3 从 `minigen.data.representation_cache` 使用 `cache_source`、`RepresentationCache.split()` 和 `blank()`；`split()` 可由 PyTorch DataLoader 批处理。正式读取先通过 `verify_cache_acceptance` 校验并获得两个回执的摘要，再交给读取器，避免使用另一版本的缓存。跨设备复制后复验；P2 的格式验收不等于已在 CUDA 运行。

### **3. 正式缓存验收**

2026-11-08 00:13:31（Asia/Shanghai）独立回读通过。MPS FP32 构建耗时 **5,515.361 秒**，发布后新进程执行 `--verify-only` 耗时 **4.345 秒**。缓存位于 `artifacts/cache/artbench_pd_256/v1_formal`，八个 NumPy 数组共 **3,631,029,392 字节**（含数组文件头），留在本机。Git 仅保存小型 [正式缓存回执](components/artbench_cache_formal.json)。

| 划分 | 独立回读通过行数 | 潜变量 | 文本特征 | 文本掩码 |
| --- | ---: | --- | --- | --- |
| train | 37,042 / 37,042 | FP16 `[37042,4,32,32]` | FP16 `[37042,77,512]` | bool `[37042,77]` |
| validation | 4,637 / 4,637 | FP16 `[4637,4,32,32]` | FP16 `[4637,77,512]` | bool `[4637,77]` |

两份回执的索引及八个数组 SHA-256 一致；索引 SHA-256 为 `d065449f4d17c8f307325b8842cd687b611b52065ada43988c28cfe26a960d00`。独立回读核对有序样本 ID、冻结清单、回执身份、数组形状和 dtype，逐行检查有限值及 Caption 掩码位置。独立空条件通过校验，只有 BOS 和 EOS 有效。test 的 4,619 张图未编码。

## **三、仓库清理与边界**

- 移除项目本地的 QwenVL 2B/8B 权重、`caption_mlx` 专用环境、可重新获取的 MNIST、可重建的 128px 几何数据与旧表示缓存。清理文件的逻辑大小合计 3,790,587,008 字节。保留 ArtBench 原始数据、冻结组件、旧实验运行记录和历史来源。[清理回执](components/p2_cleanup.json)
- 将 23 个课程模块迁到 [学习源码](../../docs/learning/src/README.md)，其中 21 个保持原字节，另两个拆出共享清单验证和 SDXL VAE 适配器。迁移前后的 10 份 Notebook 哈希一致；Notebook 及工作区中的独立问答改动不进入这次提交。旧测试和执行工具显式加载学习代码，生产包不会自动加载。
- 正式代码按数据、冻结图像编码、冻结文本编码和独立评测分层；P3 训练、采样和生成模型入口仍保留原状态。现有历史个人源码和 Notebook 的质量检查问题单独登记；CI 对维护的 `src`、`scripts` 和 `tests` 执行同一套 Ruff 和接口文档检查。

## **四、验收记录**

- 本轮本机扩展前的 P2 工程基线：全量 pytest **146 通过**。维护代码 Ruff、格式和中文接口检查通过；CPU doctor、离线锁文件与 wheel 构建通过。wheel 仅含正式 Python 包，未打包学习 overlay、数据或权重。
- 本机扩展后：全量 pytest **151 通过**，维护代码 Ruff、格式、接口检查、CPU doctor、离线锁文件和差异检查通过。新增回归覆盖回执绑定与阈值、写入中断恢复和篡改拒绝，以及掩码错位（即使文件哈希正确）。
- 全仓 Ruff 仍有 16 个历史学习/Notebook 诊断，10 个文件存在既有格式差异；留在独立范围，未修改课件或用户个人源码。Git 的尾随空格告警仅见于原字节迁移的个人 DiT 和 TinyVAE。仓库检查及哈希见 [结构化审查回执](components/p2_repository_review.json)。
- P2 验收完成；项目停在 P3-01 入口。未运行 P3、训练或 CUDA，也未消耗收费 GPU 或 API 额度。
