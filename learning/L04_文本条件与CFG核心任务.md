## **一、目标、依赖与整体感**

L04 对应 S3-01、S3-02。先完成 L03 的 patch、adaLN-Zero 和个人验收，再连接文本条件。完整讲解与参考仅在 [S3 Notebook](../docs/learning/S3_文本条件CFG与可信评测.ipynb)，src 接口没有自动 fallback。

| 你负责的核心 | 输入 → 输出 | 位置 | 独立验收 |
| --- | --- | --- | --- |
| Cross-Attention | image `[b,n_x,d]`、text `[b,n_y,d_c]`、mask `[b,n_y]` → `[b,n_x,d]` | src/minigen/conditioning/cross_attention.py | scripts/check_conditioning_core.py |
| CFG | 同形无条件/条件预测、scale → 同形引导预测 | src/minigen/samplers/cfg.py | scripts/check_cfg_core.py |
| 条件模型整合 | noisy、t、text、mask → 噪声 | 在个人 DiT 完成后接入，不预写答案 | mask、梯度、缓存对齐、对照实验 |

Agent 已负责冻结编码器、缓存、数据顺序、训练与采样、指标、成本和报告。你需要理解各步输入、核心公式和证据能支持的结论，不必重写文件下载或日志系统。

## **二、先手算，再实现**

### **1. 文本检索**

画出 `q:[b,H,n_x,d_h]`、`k/v:[b,H,n_y,d_h]` 与分数 `[b,H,n_x,n_y]`。解释除以根号头宽度的原因，mask 的 True 约定，以及 padding 为什么不能改变输出。

个人接口约定 `q=Linear(d,d)`、`kv=Linear(d_c,2d)`、`out=Linear(d,d)`，便于验收器在同一投影上用 Torch SDPA 做独立数值对照。完整多头重排和 softmax 由你写。每条 mask 至少一个有效位；非法输入明确报错。外层负责残差和门控，接口本身不偷偷增加第二次残差。

### **2. CFG**

写出 $u+s(c-u)$，手算 $u=(1,2)$、$c=(3,1)$、$s=0,1,3$。说明其与原论文 $(1+w)c-wu$ 的参数关系 $s=1+w$。实现只组合预测，不在核心函数中加载模型或抽随机数。

丢弃整条条件时必须同时替换 token 和 mask；空条件来自同一编码器的空文本，不是全 False mask。先解释训练为何见过空条件，再解释采样如何调用它。

## **三、可直接执行的验收**

```bash
uv run --locked python scripts/check_dit_core.py
uv run --locked python scripts/check_conditioning_core.py
uv run --locked python scripts/check_cfg_core.py
uv run --locked --extra latent --extra conditioning --extra notebook pytest tests/test_stage3.py
```

个人未实现入口返回待做和退出码 2；不能为了通过而导入 Notebook。数学验收通过后，再由 Agent 用你的对象替换显式调用参数、创建新 run_id 做小型整合验证。旧教学权重仍标为教学来源，不因接口通过变成你亲自训练的模型。

## **四、理解验收与交接**

合上参考回答：token 数不同为什么可以注意力？77 长度包含什么？跨段编码有什么局限？CFG=1 为什么只需一条分支？mask 改反时会怎样？固定 $z_t$ 的输出变化与生成属性变化差在哪里？

完成条件代码后，你只需说明修改文件和一句原理总结，Agent 负责独立数值/梯度、完整 pytest、缓存与恢复、同噪声对照和报告。若条件收益仍不成立，优先定位链路与训练，不打开最终测试找样例。

当前状态：L03、L04 个人实现与理解仍分别待验收。S1 的 VAE 两项个人函数已通过代码验收；尚未清理的旧文档/旧测试不表示 VAE 数学失败。

审核不要求你手工维护 JSON 或目录：你给出判断，Agent 负责匿名视图、标签录入、合并和一致性统计。
