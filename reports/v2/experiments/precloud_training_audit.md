## **一、审查结论与证据边界**

本次以 `77b750685f850f1b6789995fc3dd5a1a74857b87` 为基线，只读复核 P3/P4 的模型、Flow、条件、训练、缓存和恢复链路。21 个所查源码文件与该提交一致，未发现需要改动模型或训练算法的确定性缺陷；参数统计重新构建三种正式规格后核对。复用 P4 的正式规格 MPS 训练证据和 P5 的真实采样/评分证据，只补充 **12 项 CPU 定向回归，4.54 秒，全部通过**。未重复训练正式规格、重建缓存、下载权重或运行 CUDA。

**本地工程通过不等于生成质量通过。** [P3 收尾](../03_p3_closeout.md) 的单图记忆产生可辨人物，原固定 16 图仍有结构和高频纹理问题；P3-05 质量验收未通过。该项不阻止用户执行有上限的诊断 Pilot，但阻止直接批准正式长训。没有证据将其归因于已确认的 Flow/VAE/CFG 算法错误，也不能据单图成功排除条件学习不足等假设。

机器可读核对见 [源码摘要与参数量](precloud_training_audit.json)。历史 [P4 验收](../04_p4_delivery.md) 与 [P5 检查回执](p5/checks.json) 保持原样，217 项是上一轮全量测试证据，不写成本轮重新运行结果。

## **二、模型与数学契约**

| 项目 | 当前实现与核对 | 状态 |
| --- | --- | --- |
| B0 | 8 层图像 Self-Attention + 文本 Cross-Attention；30,359,824 参数 | 重新构建计数一致；正式规格 MPS 8+4 步历史实测 |
| B1 | 11 层双向单流；30,061,456 参数 | 重新构建计数一致；正式规格 MPS 8+4 步历史实测 |
| B2 | 3 层双流 + 5 层单流；30,061,456 参数 | 重新构建计数一致；正式规格 MPS 8+4 步历史实测 |
| 统一输入输出 | 输入和速度输出 `[b,4,32,32]`；时间 `[b]`；条件 `[b,77,512]` 及布尔 mask `[b,77]`；图像规格 256×256 | 工厂共享 `model(z,t,Condition)`；不含冻结组件训练参数 |
| Flow 训练 | $z_t=(1-t)\epsilon+tz$，目标 $z-\epsilon$，时间均匀抽样；FP32 平均速度 MSE | 训练从噪声 0 向数据 1，和采样一致 |
| Euler/Heun | 0→1 积分；Euler NFE=steps，Heun NFE=2×steps，最终正确器包含 t=1 | NFE 另计 CFG 分支，非 DDPM 离散日程 |
| CFG | $v_u+s(v_c-v_u)$；s=0/1 只执行所需分支，其他 s 每次两次主干调用 | 不把相同 solver NFE 误称相同主干成本 |
| 条件丢弃 | 以同一 CLIP 编码器的真实空文本同时替换 features 和 mask | 空文本有 BOS/EOS；不以全零 features 偷代真实空条件 |
| Padding | B0 key mask；B1/B2 key mask 加每层无效 query 清零；投影前清除 padding 数值 | 已有非零输出头/门控的前向、梯度、扰动与追加 padding 测试 |
| 注意力后端 | 主干局部 SDPA/naive，默认 SDPA，state_dict 键不变 | 既有 CPU 前向/梯度对照通过；CUDA kernel 行为待测 |

源文件入口：[Flow 目标](../../../src/minigen/objectives/flow.py)、[积分器](../../../src/minigen/samplers/flow.py)、[模型工厂](../../../src/minigen/training/config.py)、[B0](../../../src/minigen/models/plain_dit.py)、[共享双流/单流](../../../src/minigen/models/joint.py)。B0 的二维 sin/cos 与 B1/B2 的二维 RoPE、层数和调制均不同；比较是三套完整架构，不能归因于一个注意力因素。

## **三、冻结表示与训练状态**

### **1. 只复用已批准缓存**

VAE 为固定 SDXL FP32 冻结模型，输入 RGB `[-1,1]`，编码取 posterior mode 并乘 `0.13025`，解码先除相同系数。训练器读取的 latent 已经缩放，不重复乘系数。CLIP-B/32 文本塔冻结，最多 75 内容 token 加 BOS/EOS，mask 保留真实有效位置。

P2 缓存使用 float16 储存 `[4,32,32]` latent、`[77,512]` 文本及 bool `[77]` mask；训练加载后转 FP32，CUDA AMP 只作用于主干计算。缓存身份绑定 Manifest、Caption、两组件锁、VAE 用户审核、本机数值验收、编码/缩放规范和 payload hashes。`open_cache` 核对来源/审批/索引/全部 payload 摘要、shape/dtype，再固定子集有序 ID；验证集不与训练合并，正式 Test 不进入训练缓存。

已验收 train 37,042、validation 4,637，详见 [P2 独立回读](../components/artbench_cache_formal.json)。本轮不重跑 41,679 张完整回读或编码；云端迁移后仍须对传输后的文件做独立回读，不能用本机旧回执替代新机器文件完整性检查。

### **2. 优化与恢复机制**

| 检查点 | 审查结果与限制 |
| --- | --- |
| AdamW | 优化器状态和参数组保存恢复；LR 按固定 warmup/cosine 配方及零起始更新编号计算，无隐藏 Scheduler 对象 |
| 梯度累积 | 一次先抽取整个有效 batch 的时间、噪声与条件丢弃；微批损失按实际样本数加权。末尾不足 batch 保留，实际 `samples` 入日志 |
| EMA | 优化更新后参数线性更新、buffers 复制；EMA 为 eval 且无梯度。采样明确选择 EMA 或在线权重 |
| 梯度裁剪 | unscale 后裁剪，非有限梯度立即报错；不会把失败更新计为完成 |
| 完整状态 | model、EMA、optimizer、GradScaler、step、固定日程、history、Python/NumPy/CPU/设备 RNG、数据排列/游标/排序 RNG、配置/源码/数据身份均保存 |
| 原子发布 | 临时文件完整写入并 fsync 后替换 `last.pt`；完整更新边界保存；失败保留最后已发布检查点 |
| 恢复核验 | 必须相同源码哈希、设备类型、Torch 版本、模型、配方和数据身份；只允许延长 max_steps/max_seconds，不允许通过延长停止上限改变 schedule_steps |
| 日志修复 | Checkpoint 的 history 为权威，删除崩溃后超前日志尾部；恢复后的 step/loss/LR/grad_norm/samples 在 CPU 小案例逐项一致 |
| 输出保护 | 有 last.pt 或 metrics.jsonl 的训练目录要求显式 resume；运行目录加非阻塞文件锁；sample/evaluate 拒绝已有目标 |
| 失败回执 | 普通训练异常记录 attempt_id、异常和最后有效 checkpoint，旧成功结果保留。消费者须比较最新 attempt，不以旧 result.json 认定新尝试成功 |

`KeyboardInterrupt`/进程被杀、机器断电不承诺保存当前未完成更新或写入 failure.json。恢复只能回到已发布 Checkpoint；受控 `max_seconds` 在完整更新边界退出并保存。`max_seconds` 是 **每次调用** 的时间限制，恢复次数会累计成本，必须另管总预算。

训练恢复契约记录设备类型而不锁具体 GPU 型号；运行回执另记 hardware。换 GPU/驱动/算子或精度时，即使兼容性检查未拒绝，也不能宣称逐位同轨迹。旧 P3/P4 权重可被当前代码显式读取评测，不表示可跨源码严格续训。

## **四、本轮最小回归与已复用的真实闭环**

执行命令：

```bash
uv run --locked --offline --no-sync pytest -q -p no:cacheprovider \
  tests/test_p3_cli.py tests/test_flow_training.py
```

结果为 **12 passed in 4.54s**，完整日志 `artifacts/runs/precloud_audit/training_targeted_tests.txt`。其中：

- B0/B1/B2 分别通过真实 CLI 的 2 步训练→保存→延长到 3 步恢复→Heun/CFG3 采样→固定配对 Flow MSE，验证实际模型切换、run_id/架构/输出语义及文件生成；冻结缓存与 VAE 用明确夹具替代，因此不是本轮重新验收真实 VAE。
- 三种真实小规格主干各执行连续 6 步与 3+3 步 CPU 对照，model、EMA、optimizer、CPU RNG、数据排列及逐步 Loss 一致。
- 微批累积与整批结果对照、验证 RNG 隔离、日程独立于停止上限、超时保存、已有目录拒绝、目标/旧 schema 边界和失败回执保留通过。
- 注入 NaN 的失败发生在真实训练器中，保留有效检查点及旧成功回执；错数据身份不能覆写旧失败记录。

真实数据、冻结组件、三种 **正式 30M 规格** 的训练/恢复/采样已有 [P4 本地记录](../p4/local_experiments.md)；现有检查点到真实评测和报告的后半链已有 [P5 本地记录](p5_local.md)。这些记录与本轮夹具回归共同覆盖工程边界，但不是宣称本轮将整个正式规格链重新跑过。P5 的 24 张图仅为工程冒烟，图像仍是不可辨纹理，没有模型质量排名结论。

## **五、CUDA Pilot 必须补齐的验收**

| CUDA 项目 | 已有实现 | 云端应检查 | 未通过的后果 |
| --- | --- | --- | --- |
| Linux 单卡 | 路径从 root 解析，CPU/MPS/CUDA 统一接口；训练不依赖 MPS 专属 API | 离线资产加载、CUDA doctor、实际 GPU/软件版本、数据回读 | 停止 Pilot，先修环境/迁移 |
| FP32 数值 | `check_flow_device.py` 的固定两样本前向/梯度对照 | B0/B1/B2 各自与 CPU 参考，预设 atol=rtol=2e-4，不临时放宽 | 保留误差，定位 kernel/版本或实现 |
| AMP | CUDA BF16/FP16 autocast，FP16 使用 GradScaler；MPS/CPU 只开放 FP32 | BF16 支持、有限 Loss/梯度/更新、对照 FP32、采样/恢复一致性 | 不批准该精度正式训练 |
| FP16 特别限制 | finite-gradient fail-fast 位于 scaler.step 前 | 溢出时会报错停止，不是自动跳过并降低 scale 后继续；优先诊断 FP32/BF16 | 保存失败、显式改新配方并重验 |
| CUDA 恢复 | 同一单卡契约与 RNG 分支已写，CPU 精确回归通过 | 连续 8 步 vs 4+4、全部状态/LR/日志和实际数值差异；非确定性须记录 | 不批准长期运行 |
| 资源预算 | `profile_training.py` 区分预热/测量，记录 actual GPU、吞吐、峰值显存 | B0/B1/B2 单独测量；费用估算加验证/保存/生成开销 | 不采用本机速度伪造云预算 |
| 生成质量 | 单图、固定 16 图配置及条件对照入口已存在 | 原固定小样本主体可辨、结构/颜色合理，正确条件相对空/打乱有作用；查看完整结果 | P3-05 保持未过，停止正式长训 |
| 采样精度 | `sample`/`evaluate` 与当前 Benchmark 采样使用 FP32 | 不把训练 BF16 写成 BF16 采样性能；按真实精度记录 | 修正记录，不混算效率 |
| DDP | **未实现** | 本期单进程单设备；CUDA_VISIBLE_DEVICES 仅选择一张卡 | 不提供可直接多卡运行的假命令 |

当前没有本地工程阻断用户在审核后执行有限 CUDA Pilot；**CUDA 本身尚未通过、16 图质量未通过、正式预算/配置尚待批准**，均是正式长训前必须解决的门槛。本轮没有修改历史结果、冻结数据、B0/B1/B2 架构或原质量标准。
