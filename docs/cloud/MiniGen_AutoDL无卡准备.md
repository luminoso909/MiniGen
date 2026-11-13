## **一、当前环境和准备清单**

AutoDL SSH 别名为 `autodl`。项目在数据盘 `/root/autodl-tmp/minigen`，部署源码 `f0ea839925e7303832252de5c3359ba278751fe4`。无卡容器限额为 0.5 核、2 GiB；开始前数据盘约有 61 GiB 可用。项目代码在 Mac 修改后用 rsync 同步，Python 命令在 SSH 会话中运行。当前不启动 GPU、训练或正式评测。

- [x] **A01｜连接并核对无卡资源与数据盘。** SSH 已连接；`/root/autodl-tmp` 是独立挂载；`nvidia-smi` 在无卡模式不可用。
- [x] **A02｜克隆固定提交并建立目录。** GitHub HTTPS 匿名克隆要求认证；改由 rsync 传已提交的 Git 源码包，远端验 SHA-256 后克隆，无须转交 GitHub 凭据。未上传权重、数据集或虚拟环境。
- [x] **A03｜安装并验收远端 CPU 环境。** Python 3.12.3、PyTorch 2.14.1+cpu；61 个依赖通过 `uv pip check`，`doctor_cpu.json` 的 `ok=true`。CUDA 仍不可用。
- [x] **A04｜下载和核对三份 ArtBench Parquet。** 固定 revision 的三个分片共 1,517,381,098 字节，远端逐文件 SHA-256 均通过（`dataset.exit=0`）。
- [x] **A05a｜下载并校验冻结 SDXL VAE。** 配置和权重均与锁定大小及 SHA-256 一致。
- [x] **A05b｜下载 CLIP 上游权重及 tokenizer。** 605,247,071 字节的上游权重和五个 tokenizer 文件的大小及 SHA-256 均通过（`models.exit=0`）。
- [x] **A05c｜转换并核验冻结 CLIP 文本塔。** 首次转换在 2 GiB 限额下被 OOM 终止（`prepare.initial.exit=137`）。用 [低内存转换工具](../../scripts/prepare_clip_cpu.py) 重试，配置与 252,684,896 字节文本权重均匹配原锁；无新增 OOM。重试记录在 `resume_low_memory.log`、`prepare.log`。
- [x] **A05d｜启用低内存数据处理环境。** 前两次分别解包 10,880 张和 14,080 张后因频繁换页停止，原记录已保留。现用锁定 tokenizer，进程不加载 PyTorch；隔离副本沿用原 SHA-256 函数，并将 Parquet 改为内存映射、逐行组串行读取。环境与 tokenizer 检查通过，回执见 `worker.log` 和 `worker.exit=0`。
- [x] **A06｜展开图像、重建 Caption 和划分。** 2026-11-12 15:53（北京时间）完成 46,298 张，Manifest SHA-256 匹配冻结值后发布 `v1`；见 `artbench_prepare.json` 与 `prepare.exit=0`。
- [x] **A07a｜从目标机独立回读正式图像。** 逐张核对 46,298 个文件 SHA-256；train 37,042 / validation 4,637 / test 4,619，见 `artbench_readback.json`。
- [x] **A07b｜从目标机核对 Benchmark 候选内容。** `status=verified`，内容 SHA-256 为 `ff18ef2f5538538cf54411d4d4934e135bdd712c53d62247512825aaea4c62e5`；人工审批仍是 `pending`。见 `benchmark_integrity.json`。
- [x] **A07c｜在目标机运行定向 Python 测试。** ArtBench、Caption、表征缓存和 Benchmark 共 16 项通过（7.53 秒，`verify.log`、`verify.exit=0`）。表征缓存测试使用测试夹具，不代表正式缓存已就位。
- [ ] **A08｜正式缓存迁移与回读。** 已批准缓存绑定 Mac MPS 执行身份及其验收回执。无卡 Linux 可读同一份缓存，但不能重新编码生成相同身份；当前不上传大资产，也不改冻结锁。
- [ ] **A09｜CUDA 验收。** 更换模式须由用户决定；之后检查实际 GPU 和独立 CUDA 环境，再按 [逐项验收说明](MiniGen_云端逐项验收.md) 执行。

## **二、远端路径与恢复方式**

| 路径（相对于远端项目根） | 用途 |
| --- | --- |
| `data/raw/artbench_pd_256/<revision>/data/` | 从公网下载的三份 Parquet |
| `data/processed/artbench_pd_256/v1/` | 已验收的正式图像、Manifest、Caption、划分和本地审计；同级的隐藏临时目录不是正式版本 |
| `artifacts/models/{sdxl_vae,clip_text}/` | 按 revision 锁下载的资产 |
| `.venv-cpu/`、`.cache/` | 无卡环境和下载缓存；轻量数据环境在 `.cache/artbench-worker/`；CPU wheel 有别于 Linux CUDA 锁，不能用其替代 CUDA 安装 |
| `artifacts/cloud/autodl/` | 部署总日志 `bootstrap.log`、各步 `*.log` / `*.exit`、`cpu_host_inventory.txt` |

[本地预处理脚本](../../scripts/setup_autodl_cpu.sh) 已同步。镜像仅作公开下载通道；产物须匹配 [ArtBench 来源锁](../../configs/datasets/artbench_pd_256.yaml) 和模型锁中的字节数与 SHA-256。ArtBench 以 Parquet 分发，预处理脚本直接提取内嵌图像，不需 unzip。CPU 依赖保留仓库版本约束，PyTorch 从官方 CPU wheel 索引获取，其余依赖默认使用已实测较快的 PyPI 镜像。首次慢速尝试与续跑分别记录在 `env.initial.log`、`resume_cpu.log`。

数据发布条件：46,298 张，train 37,042 / validation 4,637 / test 4,619；Manifest SHA-256 必须等于冻结值 `0b95e7b783b2b62d6e2ff58646a95d173607a61b4184c6e250db7dfa1303cc0a`。若不同，候选目录及日志保留，不能用于正式训练。

```bash
ssh autodl 'cd /root/autodl-tmp/minigen && tail -n 20 artifacts/cloud/autodl/resume_artbench_bounded.log'
```

只有原步骤停止后才重试。按失败项选择 `env`、`dataset`、`models`、`worker`、`prepare` 或 `verify`；已下载文件会复用并重新校验。`worker` 在 `env` 和 `models` 成功后执行。

```bash
ssh autodl 'cd /root/autodl-tmp/minigen && bash scripts/setup_autodl_cpu.sh dataset'
```
