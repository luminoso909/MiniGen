## **一、P2-03 CLIP 文本组件验收**

**已验证。** 正式文本塔可在离线 CPU FP32 下严格加载。ArtBench v1 的 46,298 条正式英文 Caption 都能完整进入单段窗口。两条真实 Caption 能改变固定图像 query 的注意力输出，冻结文本塔没有梯度。这个实验验证输入与梯度链路，生成模型的条件遵循仍需在 P3 之后评估。

| 项目 | 本次固定值 |
| --- | --- |
| 来源 | `openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| 锁定资产 | [本地锁](../../../configs/models/clip_text.json)中 7 个文件全部通过大小和 SHA-256 校验；Safetensors 权重摘要为 `73c3428fc48ba1fe924733f5bc81282980f2c7a128851745e75ab343e231e504`；模型加载无缺键、多余键或尺寸冲突 |
| 组件接口 | [FrozenCLIPText](../../../src/minigen/conditioning/clip.py) 返回 `last_hidden_state` `[b, 77, 512]`、布尔有效位 `[b, 77]` 和 token ID `[b, 77]`；下游可训练投影不属于文本塔 |
| 运行环境 | macOS arm64、CPU、Python 3.12.14、Torch 2.14.1、Transformers 5.20.2；实测约 2.6 秒，本地资产，无付费资源 |
| 清单身份 | [ArtBench v1 冻结回执](../data/artbench_freeze.json)中固定的清单与 tokenizer 锁摘要均匹配 |

### **1、全量长度与 mask**

- 重新分词 37,042 条训练、4,637 条验证、4,619 条测试描述，只比较机械长度与清单，未对测试集做视觉分析或选型。全量最小／中位数／最大长度为 **12／46／75**，其中 525 条正好 75。空描述、超限、内嵌特殊 token、清单长度不符均为 0。
- 两条真实描述、空字符串、纯空白、单 token、恰好 75 个 token 的同批特征为 `[6, 77, 512]` FP32；有效位数为 **14、35、2、2、3、77**。76 个内容 token 被显式拒绝。
- 真实词表中 `BOS=49406`、`EOS=PAD=49407`。按位置产生 mask：BOS、内容、EOS 可见，右侧填充不可见。空条件有且只有 BOS/EOS 有效，空批量拒绝。填充 K/V 扰动 1000 后，注意力最大输出差为 **0**。

### **2、固定图像的条件链路与存储误差**

按冻结清单 SHA-256 核对训练图 `ab-00000`，以同一图像生成 query；条件分别来自 `ab-00000` 和 `ab-07820`。第二条作为反事实替换。仅用随机初始化的图像 query 投影、文本 K/V 投影及 PyTorch 注意力执行一次前向和反向，种子为 `20261107`，没有优化步骤。

| 检查 | 实测结果 |
| --- | ---: |
| 两条 Caption 共有有效位置的特征平均绝对差 | 0.778775 |
| 固定图像的注意力输出平均绝对差 | 0.367069 |
| 文本 K/V 投影梯度范数 | 2.058603 |
| 图像 query 投影梯度范数 | 0.019641 |
| CLIP 参数总量／可训练参数／有梯度的参数张量 | 63,165,952／0／0 |
| FP32 → FP16 → FP32：平均绝对误差／最大绝对误差／RMSE | 0.000129572／0.003863335／0.000219795 |

FP16 误差仅在 6 条验收输入的 68,096 个可见特征值上统计。该结果记录小样本存储误差，不代表整套缓存已写入、全部图文语义已正确或生成质量保持不变。

### **3、复现**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --offline --no-sync \
  python scripts/check_clip_p2.py --device cpu
uv run --locked --offline --no-sync pytest -q -p no:cacheprovider tests/test_clip_p2.py
```

脚本默认写入被 Git 忽略的 `artifacts/v2/p2/clip_acceptance.json`。本次统计与资产摘要保存在 [实测 JSON](clip_text_acceptance.json)。5 项无权重单测通过，覆盖相同 ID 的 EOS/PAD、下游梯度、非法输入和文件锁拒绝路径。
