## **一、登记状态与研究边界**

2026-11-10，以 `77b750685f850f1b6789995fc3dd5a1a74857b87` 为审查起点。本文是 **待用户审核、待 CUDA Pilot 填实的首轮候选预登记**，不是长训批准，也不是可直接传给 `minigen train` 的配置。对应 [机器可读登记](../../../configs/experiments/precloud_preregistration.json) 保存当前数据、组件、模型配置和评测资产的 SHA-256；云端源码应使用包含本文件的最终提交，并记录 `git rev-parse HEAD`。

主问题：同一 ArtBench 数据、冻结组件和 Flow 目标下，约 30M 的 B0/B1/B2 在生成质量、条件遵循和计算成本方面有何差异？本轮首个训练种子固定为 0；不自动补第二种子，不扩展 10M/50M 或 P6。单种子结果只作描述性比较，不声称统计显著性。

**本次不批准当前 Benchmark。** [最终审核包](../benchmark/MiniGen_Benchmark_最终审核包.md) 所列题目与评分覆盖问题必须先修正为新候选版本，或由用户明确批准限定的测量范围。内容哈希锁与人工批准是两个记录；没有批准，不运行正式 Test 评分。CUDA 工程 Pilot 只使用训练子集与独立验证条件，可在用户批准 Pilot 后进行，不需要先获得正式 Benchmark 成绩。

## **二、已固定的数据与模型**

### **1. 数据和组件身份**

| 项目 | 固定值与核验位置 |
| --- | --- |
| 数据集 | `alfredplpl/artbench-pd-256x256`，revision `4056b70025f0e3f916fb271c7a792633ba637484` |
| 图像/划分 | 全量 46,298 张；train 37,042、validation 4,637、test 4,619；原生 256×256 RGB，按组隔离 |
| Manifest SHA | `0b95e7b783b2b62d6e2ff58646a95d173607a61b4184c6e250db7dfa1303cc0a` |
| 正式缓存 | `artifacts/cache/artbench_pd_256/v1_formal`；index SHA `d065449f4d17c8f307325b8842cd687b611b52065ada43988c28cfe26a960d00` |
| 缓存规格 | latent `[n,4,32,32]`、text `[n,77,512]` 均为 float16 存储；mask `[n,77]` 为 bool。FP32 编码之后压为存储格式，训练读取转为运算精度 |
| SDXL VAE | `madebyollin/sdxl-vae-fp16-fix`，revision `207b116dae70ace3637169f1ddd2434b91b3a8cd`；posterior mode，编码乘 0.13025、解码除 0.13025 |
| 生成条件 CLIP | `openai/clip-vit-base-patch32`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`；冻结文本塔，75 内容 token + BOS/EOS，保留 attention mask |
| 已知数据局限 | Caption 有视觉事实噪声；VAE 对细线、文字、小物体有损；不据此删图、不重分组或重建本机缓存 |

完整文件、组件权重和缓存 payload 的哈希见机器登记及 [P2 正式回读](../components/artbench_cache_formal.json)。云端迁移后重新验证文件是迁移验收，不等于重新编码；Test 不进入表示训练缓存。正式训练的 `train_limit=null` 且 `train_indices=null`，不能误用 Pilot 的 16/256 图子集。验证 Flow MSE 默认读取固定前 256 个验证条目；选定后保存有序 ID，不能按结果替换。

### **2. 三模型规格与公平性**

| 模型 | 结构 | 可训练参数 | 主导前向 GFLOPs/图 |
| --- | --- | ---: | ---: |
| B0 | 8 层图像 Self-Attention + 文本 Cross-Attention；固定 2D sin/cos | 30,359,824 | 9.926 |
| B1 | 11 层图文联合 Single Stream；图像 2D RoPE | 30,061,456 | 14.894 |
| B2 | 3 层 Double Stream Joint Attention + 5 层 Single Stream；图像 2D RoPE | 30,061,456 | 10.848 |

共同 hidden=384、heads=6、patch=2、图像 token=256、MLP ratio=4；输入潜变量 `[b,4,32,32]`、文本 `[b,77,512]`、mask `[b,77]`，输出同潜变量形状的速度。参数和 FLOPs 复用 [P4 实测规格](../p4/model_specs.json)；FLOPs 只计主导矩阵乘法，不含反向、归一化、softmax、冻结组件和优化器，不是硬件计数。

这是三套具体架构的比较，位置编码、深度与调制也不同，不能把差异只归因于 Attention。三者模型配置文件不变；本轮不新增模型。相同参数规模不等于相同 FLOPs 或 GPU 时间。

## **三、共同训练配方与待测选择**

### **1. 数学、精度和优化**

$$z_t=(1-t)\epsilon+tz,\quad u_t=z-\epsilon,\quad t\sim\mathcal U[0,1),\quad \epsilon\sim\mathcal N(0,I).$$

训练用按图像元素平均的速度 MSE；采样从 t=0 积分到 t=1。文本以概率 0.1 替换成同一冻结编码器生成的真实空条件。CFG 为 $v_u+s(v_c-v_u)$，s=1 不运行空条件分支。生成器从随机初始化训练，不继承本地过拟合权重。

| 参数 | 候选/固定规则 | 冻结时机 |
| --- | --- | --- |
| Optimizer | AdamW，betas=(0.9,0.999)、eps=1e-8，weight_decay=0.01、grad_clip=1 | 共同基线；正式配方批准前复核 |
| 学习率 | 先 1e-4；仅在预定验证对照中考虑 2e-4，三模型最后共用同一选择 | Pilot 后，用户批准；不能根据 Test 改 |
| microbatch / 累积 | 候选 1/2/4/8/16，目标有效 batch=64，累积=64/microbatch；同硬件测量后选择有内存余量的配置 | Pilot 后；若 64 不可行，三者共同降档并重新登记 |
| 实际 batch | 数据流尾批可能小于上限；日志按实际 `samples` 记录，不能用上限乘步数冒充处理样本数 | 每步实测，资源比较读取真实历史 |
| 精度 | 优先 CUDA BF16；未通过稳定性验收时用 FP32。FP16 是已有但非首选的备选分支 | 各模型 CUDA 实测后；不把有限训练成功写成 AMP 数值等价 |
| EMA | 现有 Pilot 基线 0.99；仅预算允许时在验证集考虑 0.999，最终三者共用 | Pilot 后；没有实测优劣结论 |
| 日程 | 线性 warmup + cosine，min_lr_ratio=0.1；正式 steps 记为 N，warmup=`min(500,N-1,max(1,N//20))` | 与完整 N 一起冻结，恢复时不能改日程 |
| 步数/时限/费用 | 当前全部未定；先测 step_seconds、GPU 时价、显存、质量曲线和评测成本 | 用户审核 Pilot 后填写，禁止提前填一个假定长训数值 |
| 检查点 | 间隔 `max(1,N//20)`，另存 last.pt；定期验证同间隔，保存中间快照用于预算对照 | 训练前决定；生成器写的约 20 个快照并非自动 best |
| seed | 首轮三者 0；训练 RNG 与固定评测初噪分开 | 已固定首轮，额外种子需另行授权 |

正式配置只能由对应架构的真实 CUDA `pilot.json` 和相邻 `run.json`，配合用户填写的费用/步数/时限，调用 `scripts/prepare_cloud_run.py` 生成。该工具检查 Pilot 身份和费用估算，**不替代用户的质量或预算批准**。工具继承 Pilot 的 batch/LR/EMA/precision；若要选新值，先生成新 Pilot 配置、实测，再生成正式配置，不能改正式文件却沿用不匹配的旧性能估计。

### **2. 恢复、日志和质量停止条件**

检查点保存模型、EMA、AdamW、scaler、已完成步数、完整配方/日程、随机状态、数据排列与 cursor、历史日志和源码/数据身份。学习率由固定日程和 step 重算，没有独立可丢失的 scheduler 对象。精确恢复要求相同源码、设备、PyTorch 版本和配方；不同机器/内核之间不承诺逐位一致。采样可以读取旧 P4 权重，但不能据此跨源码精确续训。

同一新输出目录拒绝重复启动；中断后用 `--resume` 读取最后发布的 Checkpoint。Ctrl+C/SIGKILL 不保证保存当前更新，未发布的步数须重做；恢复时以 Checkpoint 权威历史重建日志。FP16 非有限梯度会 fail-fast，不能宣称已有自动溢出恢复机制。

**长训前必过**：B0 单图和固定 16 图的完整样本画廊、正确/空/打乱条件对照、真实 CUDA 数值/AMP/恢复、预算审批。现有证据只有单图可辨；固定 16 图仍未达到可辨主体门槛。不得用 Loss 下降、自动分数提高或部分图非纯噪声替代这一验收。失败时停止正式训练并保留配置、曲线、全量固定样本与失败原因。

## **四、评测候选、分母和选择规则**

### **1. 条件与采样**

当前候选为 `artbench-bench-v1`，内容 SHA `ff18ef2f5538538cf54411d4d4934e135bdd712c53d62247512825aaea4c62e5`，状态 pending。正式 Prompt：80 ID、40 composition_stress、40 配对改写；种子 20261110/20261111，共 320 图/模型。独立 Validation 80 题，种子 20261112/20261113。ID 只具备训练 Caption 词证据；组合题 `ood_verified=false`，不得报告已证实的 OOD 泛化率。

分布参考为 Test 的 2,000 条记录（1,999 个唯一图片/组，存在同组重复权重），用各条冻结 Caption 一图对应一参考记录。是否改成组去重抽样须在最终批准前决定，新版本重新锁定；不改变正式数据划分。Test 只索引和完整性检查，不参与本地调参。

采样候选复用 [固定矩阵](../../../configs/benchmark/sampling_matrix.json)：Euler CFG1 NFE10/20/40；Heun CFG1 5/10/20 步对应 NFE10/20/40；Euler NFE40 补 CFG2/3；另做 naive/SDPA 单变量性能对照。CFG2/3 每个 solver evaluation 有两次主干调用，NFE 相同不等于总计算相同。默认质量诊断用 EMA、Euler40、CFG1、FP32；正式选择仍为空。

选择规则提案：只在 Validation 上，以可辨内容和经校准的条件遵循优先，满足质量要求的候选中选实测成本低者；共同选择一套主配置，完整矩阵作为补充。若需要架构独立最优配置，另表列出，不能混进共同配置主比较。选择及人审标准在看到 Test 输出前记录；当前未校准 VQA 不承担选型真值。

### **2. 指标和失败处理**

| 指标 | 固定口径 | 失败/分母 |
| --- | --- | --- |
| KID | torch-fidelity Inception 2048；三次多项式核、无偏 MMD；100×1,000 子集；允许合法负数，报告抽样 std 与 seed | 正式分布 2,000 对记录；有效子集结果带覆盖率和 incomplete，不将缺图算成满样本成绩 |
| CMMD | 单位归一化 CLIP L/14@336 图像投影，RBF sigma=10，biased MMD×1,000，分块求和 | 同分布参考/生成集合，失败及缺失明确保留 |
| CLIPScore | L/14@336，无前缀，2.5×max(cos,0)，记录截断和配对顺序 | 均值是可计算配对均值，同时报告 attempted/valid，不能把缺失默认记成完整 |
| LPIPS | AlexNet 校准资产；同 prompt 双 seed 配对差异 | 报告有效配对数、无效图和遵循；噪声差异不是有益多样性 |
| VQA | BLIP base，确定性答案；正确/错误/不可判断/执行失败分开 | 每类仅适用题作分母；未知和失败不算对，保守成功率保留生成失败；重复问句不得当独立证据 |
| 人工盲评 | 匿名架构、固定顺序种子、质量/遵循 1–5 与错误标签；未作答不自动得分 | 交代分配数、已答数、缺图/未知，不补造人工数据 |

评测模型与库版本见 [资产锁](../../../configs/models/evaluation.json)、`uv.lock` 和评分回执；不与生成条件 CLIP 文本塔混用。真实 BLIP 和 100 真图+100 生成/退化图的逐类人审校准待云端/用户执行。当前 8+8 页面仅工具验收，既有 24 张烟测指标不是正式结果。

### **3. 资源与比较**

先报告同更新步数（确实存在的 checkpoint），再报告同 GPU 时间/近似 FLOPs。相同预算用不超过目标的最近已有 checkpoint，默认最大相对缺口 10%，记录目标、实际与偏差；没有对应点就留空，不插值造点。相同 GPU/精度/评测协议才可比；硬件未知时不比较 GPU Hours。

FLOPs 可用 `3 × 主导前向矩阵 FLOPs × 实际处理样本数` 近似，必须标注忽略项；不得称为硬件实测。费用按账单另存；本机秒数不能换成 CUDA 费用。正式首轮至少包括三模型各 2,000 分布图及 320 Prompt 图；Validation 网格、校准、特征提取、备份和存储另估，不只预算训练。

## **五、目录、审批和变更管理**

| 产物 | 路径约定 |
| --- | --- |
| 真实数据/Caption | `data/processed/artbench_pd_256/v1/`；Parquet 在 `data/raw/artbench_pd_256/` 下按下载脚本组织 |
| 模型/缓存 | 资产锁的 `local_path`；`artifacts/cache/artbench_pd_256/v1_formal/` |
| Pilot / 正式配置 | `artifacts/cloud/configs/`，唯一文件名，含 architecture/seed/用途 |
| Checkpoint / 日志 | `artifacts/runs/cloud/<architecture>_<purpose>_<version>/`，run.json、metrics.jsonl、last.pt、step 快照 |
| 图片/评分/匿名页面 | `artifacts/evaluations/cloud/` 下按 run_id、checkpoint、Benchmark、采样设置分开 |
| 小型证据/最终表格 | `reports/v2/experiments/`、`reports/v2/final/`；不提交大权重/缓存/整批图片 |

上述目录是统一命名提案，实际 CLI 的 `--output`/配置输出路径须与手册一致，并记录原路径。`run_id` 是源码/配方/数据契约摘要，不是随意改名的目录名。

批准顺序：用户审核本文/手册并授权 Pilot 预算 → CUDA 实测与 B0 质量过关 → 填实共同配方、各模型预算和采样选择规则 → 解决 Benchmark 审核问题并签发匹配新内容哈希的独立回执 → 用户批准正式训练。Benchmark 必须在正式训练前定稿，采样配置只在 Validation 上选择后才开 Test。

任何条件变化均保留旧版本和旧结果，新增登记版本，记录原因和受影响比较。不得改旧 Checkpoint 身份以绕过恢复检查，不得把候选 JSON 的 `approved` 自动改成 true。硬件、时价、训练步数、实测吞吐、CUDA/AMP、真实 VQA 与人审字段当前均待测/待批。
