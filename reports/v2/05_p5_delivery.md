## **一、交付结论与阶段状态**

2026-11-10，从 `12ac4f21ca5f61b7882ae45ca5c9cc8c0df025e2` 开始。**P5 本地主体工程和真实小样本验收已完成；P5 正式实验尚未完成。** 本轮交付候选 Benchmark、独立评测器、可恢复生成、人评/校准工具、性能与预算汇总，以及完整云端操作手册。停在用户审核 Benchmark 与手册、准备亲自执行 CUDA Pilot 的入口。

| 任务 | 本轮已执行 | 仍待完成 |
| --- | --- | --- |
| P5-01 | 80 ID、40 组合压力、40 配对改写、80 验证题；2,000 Test 参考索引、16 smoke 与 100 calibration 验证索引；组/文件隔离与内容锁 | 用户语义审核与独立批准；正式生成。组合题不宣称真实视觉 OOD |
| P5-02 | KID/CMMD/CLIPScore/LPIPS 真实本机计算；BLIP 真实适配器、Mock/失败契约、人评导入导出及校准工具 | BLIP 大权重与真实推理；100+100 人工校准和正式盲评 |
| P5-03 | 局部 naive/SDPA、前向/梯度/padding 对照；9 组本机 Euler/Heun/NFE/CFG 性能测量 | CUDA/AMP 与正式采样质量矩阵 |
| P5-04 | 同 step/同预算选择、逐检查点真实资源回执、近似 FLOPs 方法及曲线；缺测轴为空 | 可比较正式检查点、同 GPU 时间/预算的研究结论 |
| P5-05 | 24 张真实生成图、逐项指标/哈希/失败清单、画廊、必要曲线、命令和完整云端手册 | 全量正式评分、人工结果及最终架构比较 |
| P5-06 | 架构/规模/种子/消融字段预留 | 未授权、未运行 P6/P7 或额外研究实验 |

B0 30,359,824 参数，B1/B2 各 30,061,456。ArtBench 46,298 张、256×256、冻结 VAE/CLIP、Flow Matching 不变，P2 缓存未重建。本轮未执行正式或短训，只读取 P4 step=12 检查点。P3-05 的 16 图可辨主体门槛继续未通过，不能因评测工具可用而解除。

## **二、代码与资产**

- `evaluation/benchmark.py` 固定题目/参考/协议，独立批准回执才允许正式执行；普通本地生成仅允许 validation。
- `evaluation/assets.py` 与 `prepare_eval_assets.py` 提供显式下载和精确文件校验；普通评分不隐式联网。
- `evaluation/features.py` 管 Inception/完整 CLIP 的顺序特征、版本绑定缓存及文本截断记录；`metrics.py` 实现 KID/CMMD/CLIPScore，LPIPS 复用 P2 资产。
- `evaluation/vqa.py` 使用直接 BLIP 模型类，保留答案、未知和执行失败；概率不作为已校准置信度。评分器将 VQA 部分/全部执行错误标为不完整，CLI 返回 1。
- `evaluation/generation.py` 按 Prompt/seed 构造独立 CPU 初噪，保存逐图哈希、完成状态、错误和调用次数，支持身份校验后的恢复。
- `models/attention.py` 只切换项目主干，默认 SDPA；不改权重键或冻结组件。训练器仅新增实际硬件元数据，计算流程未改。
- `evaluation/review.py` 提供匿名离线页面、进度、JSON 导入导出和逐类校准；无真实人评时没有一致率。
- `evaluation/profiling.py`、`reporting.py` 区分 solver NFE/实际主干求值、主干/全流程耗时、MPS 当前内存/CUDA 峰值；拒绝混合模拟、不完整或不兼容的预算比较。
- `export_checkpoint_receipt.py` 从真实中间状态提取步数、耗时和实际样本量；可选 FLOPs 为有明确定义的前向倍数近似，旧缺失硬件保持空。
- `prepare_cloud_run.py` 只接受匹配架构、完整模型规格与实际 CUDA Pilot，要求用户填写步数/时限/费用上限；生成全数据配置和约 20 个预定快照，不启动训练。

命令为 `minigen bench prepare/generate/score/review/profile/report`。原 `evaluate` 仍表示固定配对 Flow MSE。新 `evaluation` 依赖组进入 uv 锁，CI 安装该组只做 CPU 测试，不下载权重。

本轮实际下载 Inception 与完整 CLIP L/14@336；BLIP 仅小型配置/tokenizer，1.54GB 主权重未下载。所有固定文件的 URL、revision、SHA 和大小见 [评测资产锁](../../configs/models/evaluation.json)。CLIPScore 是 L/14@336 无前缀变体；CMMD 的 PyTorch 端口不宣称与 Scenic/其他后端逐位相同。[资源导航](../../docs/MiniGen2下载清单.md)

## **三、真实验收与限制**

完整 pytest **217 项通过**，相对 P4 基线 180 项增加 37 项。Python Ruff、格式、注释检查通过；注释检查覆盖 112 文件、563 函数、71 类，没有结构问题。CPU doctor、离线锁检查、安装依赖兼容检查、wheel 构建与 HTML 包含检查通过。回合前 698 个受保护文件 SHA-256 全部保持，用户已有修改和删除状态保留。[检查回执](experiments/p5/checks.json)

实际 Inception/CLIP CPU/MPS FP32 与 batch 一致性通过，固定阈值 5e-4；两张验证图最大误差 2.09e-5。三模型均完成 8/8 图生成及 16 张验证参考评分；首次完整烟测 92.756 秒。主 Agent 查看全部 24 张缩略图，图像均为噪声纹理、主体不可辨；**工程通过，不代表生成质量通过**。[真实实验与指标](experiments/p5_local.md)

浏览器检查了 24 图匿名页及 8+8 校准页首屏；24/24 图像加载成功，页面仍为 0/24 和 0/16 已记录。没有提交真实人工答案。合成测试夹具用于检查往返、问题配对和失败分母，不能作为用户批准。手册通过 43 个 Shell 片段语法、7 个 Python 片段 AST 和 49 条真实 CLI 参数解析检查；这些是本地解析验收，**不是 CUDA 执行**。[手册检查](benchmark/cloud_manual_checks.json)

候选协议内容 SHA：`ff18ef2f5538538cf54411d4d4934e135bdd712c53d62247512825aaea4c62e5`。正式参考只做索引及文件完整性检查。Prompt 概念词来自训练 Caption 检索；目前没有可信绘画类别真值，也没有完整视觉 OOD 证据。验证和测试的主体目录及题族标识隔离，相关概念/表达仍需人工审核；颜色与属性绑定题部分共享判断证据，不应视为独立测量。[候选说明](benchmark/README.md)

## **四、必要修复与未执行事项**

审查期间修复了不完整/模拟成绩进入预算比较、缺失实际 GPU 型号混比、错误 Pilot 与模型规格不匹配、云端路径回迁后审核读图失效、失败图片在匿名页被丢弃，以及 VQA 执行错误仍被标为完成的问题。上述拒绝路径或往返均有相应测试。实际回读生成只改变尝试回执元数据，旧评分保留；最终指标重新绑定最终评分实现与清单哈希。

未执行：真实 BLIP、100+100 人工校准、全量 2,000 图指标/320 Prompt 图正式比较、CUDA/AMP、付费 GPU、正式训练、DDP、P6/P7。CUDA 字段和实际账单缺失处保持 null；本轮新增云/API 收费 **0 元**，不把本机秒数换算成 GPU 小时。

## **五、交付导航与停止点**

- [完整云端操作手册](../../docs/cloud/MiniGen_云端训练与评测完整操作手册.md)
- [按依赖排序的云端任务清单](../../docs/cloud/云端任务清单.md)
- [Benchmark 候选及人工审核重点](benchmark/README.md)
- [真实本机实验与图片路径](experiments/p5_local.md)
- [云端入口交接](05_p5_handoff.md)

本轮仅提交源码、配置、测试、协议、小型回执和必要图表；模型、特征/表示缓存与大批量图片由 Git 忽略。提交名称为 `feat(p5): 完成本地评测工程与云端训练操作手册`，不推送。下一步由用户先审核协议和手册，再亲自运行 CUDA Pilot；任何正式长训仍须先解决既有质量、预算与审批门槛。
