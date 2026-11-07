## **一、P2 工程验收**

P2 使用同一冻结版 ArtBench Public Domain 256x256：46,298 张原生 RGB 绘画，train 37,042、validation 4,637、test 4,619。清单 SHA-256 为 `0b95e7b783b2b62d6e2ff58646a95d173607a61b4184c6e250db7dfa1303cc0a`。这次没有下载新权重、训练模型或使用收费资源。

| 任务 | 实际状态 | 证据 |
| --- | --- | --- |
| P2-01 随机重建 | CPU FP32 重建 200 张固定随机训练样本及 9 张诊断样本。随机样本均值 PSNR 26.76 dB、SSIM 0.744、LPIPS 0.080 | [方法、数值与失败样本](components/vae_reconstruction_200.md) · [结构化结果](components/vae_reconstruction_200_summary.json) |
| P2-02 选定 VAE | 代理查看 31 组后建议保留候选：主体、构图和大色块通常保留；密集笔触、细线和可读文字会受损。待用户看图确认 | 本地 `artifacts/evaluations/p2_vae_artbench_200/review.html` 提供完整配对、分数和局部对照 |
| P2-03 CLIP | 46,298 条最终 Caption 离线复分词，长度 12–75；0 条空、超限、特殊 token 或计数不符。真实样本检查 mask、空条件、条件更换及下游梯度 | [文本验收](components/clip_text_acceptance.md) · [结构化结果](components/clip_text_acceptance.json) |
| P2-04 256px 表示缓存 | 试点：train 16、validation 16，独立空条件；重新加载与重编码后一致。完整 41,679 条尚未编码 | [试点回执](components/artbench_cache_pilot.json) |

CLIP 的条件更换对照只说明特征能进入下游可训练投影，不能证明 B0 的提示词遵循。P1 审计在 240 对中观察到 71 条明确视觉事实错误；长度、mask 或梯度验收不会消除这类噪声。

## **二、表示缓存契约与运行**

缓存对每个样本返回 `(sample_id, latent, text, mask)`。潜变量为 FP16 `[4,32,32]`，由 FP32 后验均值乘 VAE 配置中的 `0.13025`；文本为 FP16 `[77,512]`，掩码为 bool `[77]`。空描述由同一 CLIP 编码，独立于任何配对行，只有 BOS 和 EOS 有效。模型读取时可按训练精度转换；可训练的文本投影留给 B0/B1/B2。

清单顺序、样本 ID、两套组件锁、预处理、Caption 策略、mask 和 dtype 进入身份。构建器验证源图摘要、尺寸和 Caption token 长度，把数组分批写入暂存目录后原子发布。读取器独立派生预期身份，核对有序 ID、审核回执和各数组 SHA-256、形状及 dtype，再通过只读 mmap 按行访问。缓存应视作只读快照；复制或搬迁后重新运行验证。

CPU 试点实际耗时 12.108 秒。32 对特征重新编码后与磁盘 FP16 数值逐项相等，掩码和 ID 一致；FP16 量化最大绝对误差为潜变量 0.000976、有效文本位置 0.007775（仅试点，不外推成全量误差保证）。完整 train 和 validation 的数组净载荷预计约 3.63 GB，不含元数据和磁盘暂存。test 不编码。

从仓库根目录运行；先恢复本地资产，按需同步 `latent`、`conditioning` extras。生产命令不会下载模型。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --offline --no-sync python scripts/build_artbench_cache.py \
  --pilot-rows 16 --output artifacts/cache/artbench_pd_256/v1_pilot --verify-only
```

生成新的试点时去掉 `--verify-only` 并指定新目录。正式构建将 `--pilot-rows 16` 改为 `--approval <用户审核回执> --evidence reports/v2/components/vae_reconstruction_200_summary.json`，输出到新版本目录。审核 JSON 包含 `decision: "approved"`、`reviewed_by`、`manifest_sha256`、`vae_lock_sha256` 和 `reconstruction_sha256`。最后两个字段是对应模型锁和重建汇总文件的 SHA-256。没有用户审核时不创建回执；程序验证其身份绑定，不能替代人看图。若视觉审核不接受，保留原候选证据，排查细节缺失后再提出共享组件替换方案。

P3 从 `minigen.data.representation_cache` 使用 `cache_source`、`RepresentationCache.split()` 和 `blank()`；`split()` 可由 PyTorch DataLoader 批处理。正式读取应先调用 `verify_review`，再将返回的 SHA-256 传给读取器，避免把别次审核的缓存误用为当前版本。

## **三、仓库清理与边界**

- 移除项目本地的 QwenVL 2B/8B 权重、`caption_mlx` 专用环境、可重新获取的 MNIST、可重建的 128px 几何数据与旧表示缓存。清理文件的逻辑大小合计 3,790,587,008 字节。保留 ArtBench 原始数据、冻结组件、旧实验运行记录和历史来源。[清理回执](components/p2_cleanup.json)
- 将 23 个课程模块迁到 [学习源码](../../docs/learning/src/README.md)，其中 21 个保持原字节，另两个拆出共享清单验证和 SDXL VAE 适配器。迁移前后的 10 份 Notebook 哈希一致；Notebook 及工作区中的独立问答改动不进入这次提交。旧测试和执行工具显式加载学习代码，生产包不会自动加载。
- 正式代码按数据、冻结图像编码、冻结文本编码和独立评测分层；P3 训练、采样和生成模型入口仍保留原状态。现有历史个人源码和 Notebook 的质量检查问题单独登记；CI 对维护的 `src`、`scripts` 和 `tests` 执行同一套 Ruff 和接口文档检查。

## **四、验收记录**

- 全量 pytest：**146 通过**。维护代码 Ruff、格式和中文接口检查通过；CPU doctor、离线锁文件与 wheel 构建通过。wheel 仅含正式 Python 包，未打包学习 overlay、数据或权重。
- 全仓 Ruff 仍有 16 个历史学习/Notebook 诊断，10 个文件存在既有格式差异；留在独立范围，未修改课件或用户个人源码。Git 的尾随空格告警仅见于原字节迁移的个人 DiT 和 TinyVAE。仓库检查及哈希见 [结构化审查回执](components/p2_repository_review.json)。
- 本轮在本机完成，未消耗收费 GPU 或 API 额度。P3、完整缓存、生成能力与正式训练尚未验收。
