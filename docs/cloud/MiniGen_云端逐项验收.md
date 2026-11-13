## **一、本地复核结论**

2026-11-12，复核源码 `f0ea839925e7303832252de5c3359ba278751fe4`。**本地未发现新的 CUDA Pilot 工程阻断。** 云端仍要核对目标提交、实际安装、持久盘和迁移资产，再由你按预算逐项执行。正式长训和 Test 继续暂停。

| 项目 | 本轮证据 | 对下一步的影响 |
| --- | --- | --- |
| 源码与依赖 | 相较工程冻结提交 `5168578`，运行源码、既有脚本、配置及依赖锁未变。预登记的 26 个文件及整包源码哈希通过；CPU doctor 通过（Python 3.12.14、PyTorch 2.14.1），63 项锁定依赖检查通过。 | Linux/CUDA 安装仍须在目标机检查。 |
| ArtBench 与冻结组件 | 46,298 张原生 256×256 图逐文件验 SHA-256，组与文件无跨划分泄漏。正式缓存独立回读 train 37,042、validation 4,637，3,631,029,392 字节；索引、payload、shape、dtype 与冻结记录一致。SDXL VAE 与 CLIP 文本塔的本地文件哈希通过。 | 同步后重跑 `data`；缺失或不符时补同步，不重建。 |
| 训练、采样、恢复和评测 | 27 项定向 pytest 通过，覆盖 Flow、训练/恢复、缓存、Benchmark、评分。三份 P4 Checkpoint 哈希与报告一致。Inception、完整 CLIP、LPIPS 权重哈希通过。 | 只复用 P4 权重做 FP32 对照；新的 CUDA 恢复从同一提交新起跑。CUDA、BF16、评测特征和真实 BLIP 仍待实测。 |
| Benchmark | 现有候选内容哈希和参考文件通过，`human_approval=pending`。120 题的数量要求未纳入评分，部分留白/场景要求不计分，颜色和绑定有重复问句；2,000 条分布参考只有 1,999 张唯一图。组合题仍是压力测试，无已验证 OOD 证据。 | 不阻断只用 train/validation 的 Pilot。建议先修订评分覆盖，再批准正式候选；若保留原候选，须明确限定结论范围。 |
| 生成质量与预算 | 历史 B0 单图可辨，固定 16 图质量未过；预登记的实际硬件、步数和预算仍为空。 | 单图通过仍须检查 16 图和条件收益；主体不可辨或正确条件无可辨收益时停止长训。 |
| Git 同步 | 本地 `HEAD` 与 `origin/main` 的本地追踪引用一致；本机直接查询 GitHub 时 DNS 失败，远端实时值未核实。新增脚本本轮未提交。 | 在目标机实际 `git fetch`，确认能检出上述提交；另行复制脚本。 |

本机直接执行了脚本的 `data`、`benchmark`、`eval-assets`。回执在 `artifacts/cloud/acceptance/local_precloud_review/`。这轮没有启用云 GPU、启动训练、做正式评分或代签人工批准。

## **二、在云端逐项运行**

当前 AutoDL 实例使用已通过回读的独立 CUDA 全量缓存。数据和冻结模型在远端准备，运行方法见 [README](../../README.md) 和 [AutoDL 环境记录](MiniGen_AutoDL无卡准备.md)。

下列命令是使用 Mac 批准的 `v1_formal` 和旧 P4 检查点的历史验收配方，适用于具备这些资产的目标机。连接、锁定依赖与资产前置见 [完整手册](MiniGen_云端训练与评测完整操作手册.md)。将 [check_cloud_readiness.sh](../../scripts/check_cloud_readiness.sh) 与目标源码一并同步。评测设备对照应使用独立目录，避免复用其他设备的特征缓存。

在云端 MiniGen 根目录执行：

```bash
export COMMIT=f0ea839925e7303832252de5c3359ba278751fe4
git fetch origin main
git switch --detach "$COMMIT"
export BATCH="pilot-$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"  # 优先沿用平台分配
check() { bash scripts/check_cloud_readiness.sh "$COMMIT" "$BATCH" "$@"; }

check cuda        # 单卡、驱动、锁定依赖、doctor、磁盘挂载
check data        # 完整 ArtBench 文件、权重与正式缓存；不重新编码
check benchmark   # 仅核对已存在的候选和参考图
```

只在新云端检出、当前没有本地改动时使用上面的 `git switch`。确认 `findmnt` 显示的是平台实际承诺的持久化盘；挂载信息本身不能证明实例释放后仍保留。脚本会给每次尝试建立独立目录。查看 `run.log`、`identity.txt` 和 `exit-code`；失败后补齐问题再重跑该步，之前的回执会保留。仅在同一主机、同一 GPU 和同一脚本版本上续用批次；新 SSH 会话继续时，把 `BATCH` 设置为原批次。

确认 GPU 的实际价格和 Pilot 时限后，在同一会话逐行运行，前一条失败就停止：

```bash
check fp32 b0
read -r -p '实际 GPU 价格（人民币/小时）：' MINIGEN_GPU_HOURLY
read -r -p '本次 AMP Pilot 进程上限（秒，1～3600）：' MINIGEN_AMP_TIMEOUT
export MINIGEN_GPU_HOURLY MINIGEN_AMP_TIMEOUT
check amp b0
check resume b0
check quality-single
```

- `fp32.json`：固定两条缓存样本的前向与梯度通过，`atol=rtol=2e-4`。旧 P4 检查点只用于权重加载。
- `pilot/pilot.json`：必须跑完 50 步预热和 50 步实测，记录 BF16、吞吐、显存峰值及实际时价。进程上限可中断超时实验，最多另有 30 秒终止宽限；它不会关闭 GPU 实例或停止平台计费。
- `resume.json`：连续 8 步和恢复后的 8 步对照模型、EMA、优化器、LR 日程、scaler、数据流、随机状态和关键训练记录。结果不一致时保留两份检查点及差值，先查 CUDA 确定性；脚本不会把差异标为通过。
- `quality-single`：看 `diagnostics/reference.png` 和诊断目录的全部对照。确认主体可辨后，才显式启动 16 图。

```bash
# 只在你看完并接受单图诊断后运行：
MINIGEN_SINGLE_REVIEWED=yes check quality-16
```

打开 `correct/grid.png`、`blank/grid.png`、`shuffled/grid.png`，结合三个 `samples.json` 对照固定 16 个样本、共用种子、真实参考图和 Caption。脚本检查三组顺序、条件和 48 张图是否齐全；图像质量始终写 `PENDING_HUMAN_REVIEW`。如果 16 图和条件收益通过人工验收，再按同样顺序对 B1/B2 分别运行 `check fp32 b1`、`check amp b1`、`check resume b1`，然后把架构换成 `b2`。若硬件不支持原生 BF16 或默认有效批量 64 的设置超显存，保留失败证据；参照手册另行确定精度或微批方案，三架构按共同口径重新测量。

## **三、Pilot 之后的独立确认**

```bash
check eval-assets     # Inception、完整 CLIP、LPIPS 只验文件
check eval-device     # 两张验证图；实际比较 Inception / CLIP 图像及文本特征
check vqa-assets      # 仅在你已按手册获取真实 BLIP 权重后执行
```

- `eval-device` 每路同时检查 CPU batch 一致性和 CUDA 对照。只允许已通过 CPU 检查的单项回退 CPU。LPIPS CUDA 仍须用手册的 `bench score --metrics lpips` 跑验证小样本；BLIP 的文件校验也不能替代真实推理或 100+100 人工校准。
- **长训之前**：完成 16 图人工审核、填实三模型 Pilot 测量及预算，批准统一训练配置，修订或明确限定 Benchmark 候选并记录审批。再按手册启动。脚本不会运行长训或正式 Test。
- **评测与释放实例之前**：先用 Validation 选择配方，批准后才用固定 Test，失败与缺失计入分母。按手册回迁结果、原始日志、配置、`last.pt` 和快照；在 Mac 对比 SHA-256、实际加载，然后再释放实例。
