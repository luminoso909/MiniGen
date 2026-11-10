## **一、审核结论与批准边界**

本次对 `77b750685f850f1b6789995fc3dd5a1a74857b87` 的 MiniGen-Bench 候选及实际源码进行最终审查。**候选的内容锁、数量、配对及数据隔离检查通过；目前不建议直接将它批准为“全文条件遵循／真实 OOD”正式基准。** 下文列出的评分覆盖、重复问题和语义边界需要用户决定。它们不阻断 B0 的验证集 CUDA Pilot，但在解决前阻断正式 Benchmark 成绩与相应研究结论。

本轮没有更改题目、候选锁、数据划分或批准回执，没有读取测试图作调参，没有运行模型或下载资产。此前 217 项回归与真实本地评测证据继续复用；本次新增的是全量题目、清单和协议的只读核验及逐题审查，不冒充人工视觉审核。

- 候选：[benchmark.json](artbench_v1/benchmark.json)；锁：[lock.json](artbench_v1/lock.json)。
- 内容 SHA-256：`ff18ef2f5538538cf54411d4d4934e135bdd712c53d62247512825aaea4c62e5`。
- 文件 SHA-256：`8574e2a05ae797780ccdf427e7aef1fe16932f7771ff5544fdb35041a96b8cc4`。内容规范哈希与文件字节哈希用途不同，不可互换。
- 机器可读审查：[benchmark_audit.json](final_review/benchmark_audit.json)，含全部 240 题、40 对改写逐项核对和问题 ID。
- **审批状态仍是 `pending`；本文件是 Agent 的审核意见，不是用户批准。**

## **二、题目定义、分布与隔离**

### **1. 数量和使用规则**

| 集合 | 实际数量 | 固定噪声种子 | 用途 |
| --- | ---: | --- | --- |
| ID | 80 | 20261110、20261111 | 20 主体 × 数量一/二、颜色、画面左侧四类请求 |
| Composition stress | 40 | 20261110、20261111 | 20 对主体 × 左侧/上方附颜色；`ood_verified=false` |
| Paraphrase | 40 | 20261110、20261111 | ID 单数和双数各 20 题的明确成对改写 |
| 主测试合计 | 160 Prompt / 320 图每模型 | 同上 | 最终条件评测，不参与采样选型 |
| Validation | 80 Prompt / 160 图每模型 | 20261112、20261113 | 20 个不同主体目录，承担调参和局部验证 |
| 分布参考 | 2,000 Test 记录 | 选择 seed=20261110 | 按冻结 Caption 一条生成一图；正式 KID/CMMD |

80/40/40 是 Prompt 数量，320 是双种子图片数量，不能混用。所有 Prompt 的精确字符串与规范化字符串均无重复；与 train/validation/test 的 Caption 做小写及标点规范化后，没有完整字符串重合。这只能排除精确文本复制，不能证明视觉或近义语义隔离。

### **2. ID 与组合压力的证据边界**

人物、动物、庭院、河流风景、乡村建筑、静物和室内题材与 ArtBench 绘画场景相容。主体词在训练 Caption 中有检索命中，但只有词汇出现证据。颜色、数量、场景及关系的完整组合是否常见没有经过视觉确认；因此当前 `id` 应理解为“词汇受支持的常规绘画请求”，不能称为严格证明处于训练联合分布内。

原 Plan 的 `Composition-OOD` 位置现在明确使用 `composition_stress`。训练图片没有封存指定未见组合，没有视觉缺席证明，不能报告“真正 OOD 成功率”。例如 `composition-03-1` 的 horse above bridge、`composition-12-1` 的 table above tower 属于不寻常绘画构图，可作为控制压力题，但不能据此假定它们是 ArtBench 常见情境或严格 OOD。

### **3. Paraphrase 完整检查**

40 对均将 `A painting showing {内容}.` 改写成 `Depict {同一内容} as a painted artwork.`，对象、数量、场景词逐字保留；结构化要求、语义族和噪声种子完全一致。审查没有发现添加或删除视觉事实。配对为 `paraphrase-id-XX-0/1` → `id-XX-0/1`，XX=00–19。

这只是一个固定句式的轻量改写测试，尚不覆盖同义词替换、复杂句式、词序变化、颜色或空间题的表达鲁棒性。不应将它描述为广泛语言泛化。数值对比应按配对 Prompt 和相同 seed 报告，不能把 40 个改写及原题视为完全独立样本。

### **4. 数据隔离与剩余近义风险**

正式 Manifest 仍为 train 37,042、validation 4,637、test 4,619。三组两两的 `group_id` 与文件 SHA-256 交集均为 0。本轮读取清单核对，沿用 P5 参考图文件校验，不重复读 2,000 图或计算测试特征。

验证/主测试目录的主体字符串和题族 ID 不重合，但没有做到概念完全独立：例如 rose/flower、ship/boat 为上下位或近义关系；girl/woman、boy/man 共享人物语义；cat/dog 和 pear/apple 共享更高层类别。相同四种模板也被两组共同使用。可接受它作为“独立题目、近域调参”，但不能宣称“视觉概念与语言模板彻底隔离”。若研究需要概念级留出，须建立新版本再批准，不能事后删改正式结果对应的题目。

**2,000 条参考记录只有 1,999 个独立文件和组**：`ab-28721` 与 `ab-46291` 的 SHA-256 同为 `e5c59655e44b85fe7fa5ca54b5e15c1948b3d64fd13d7dc3506b19a89e26b6ad`。这是同 Test 内的重复记录，不是跨划分泄漏。当前索引按记录抽样，两条都进入分布指标；须选择接受这项轻微加权并披露，或另发按唯一组抽样的新版本。不得声称已有 2,000 张相互独立的图片。参考 Caption 有 1,987 个唯一字符串，818 个非空 artist 字段值；artist 不是可信题材/画派真值，不能推导类别覆盖率。

16 张 smoke 验证参考全部包含在 100 张 calibration 验证参考内，并非两套互不重叠的参考。二者都与 Test 参考隔离；复用验证图可以测试工具，但正式评测器可靠性报告应披露这个关系，不把 smoke 与 calibration 当作独立证据相加。

## **三、正式批准前必须解决的事项**

### **1. 题义与结构化评分范围**

| 问题 | 精确范围 | 影响与建议 |
| --- | --- | --- |
| 明示数量未进入评分 | `id-XX-2/3`、`validation-XX-2/3`（XX=00–19）；全部 `composition-XX-0/1`；共 120 题，其中主测试 80 题 | 颜色/位置题写 one，组合题写 one A/one B，但无 count。当前 `all_requirements_correct` 不是全文全部条件正确 |
| 右侧留空未评分 | 全部 `id-XX-3`、`validation-XX-3`；共 40 题 | 左侧存在对象不等于右侧留空；需要补判定或从全文遵循声明中排除 |
| 场景/背景没有独立判定 | ID 的 0/1/2 共 60 题、全部 40 改写、Validation 的 0/1/2 共 60 题 | 例如 woman in garden 中 garden 不在要求表。不能将通过对象和数量称为场景正确 |
| 同一颜色问题重复计权 | 全部 `id-XX-2` 和 `validation-XX-2`；共 40 题 | color 与 binding 生成完全相同问句；主测试 20 题 × 2 seeds 重复贡献 40 个问题，不能作为两份独立属性证据 |
| 同色组合不检验颜色互换 | `composition-00-1/03-1/06-1/09-1/12-1/18-1`；6 题 | 两对象同色，可检验颜色遵循，不能检验颜色交换/绑定区分能力 |
| 左/上语义未操作化 | 40 左侧构图题及 20 上方组合题 | 需要人评约定画面二维位置、主体中心/范围、重叠/遮挡、不可判；上方不是物理悬浮的必然要求 |
| 可数实例边界较模糊 | Validation 家族 07 mountain、10 cloud、16 fence，共 12 题 | 连续山脉、云团、栅栏段如何计数须在人评规范明确；不是自动把这些题删掉 |

这里的 ID 简写完整展开在 [逐题审查 JSON](final_review/benchmark_audit.json)。这些是正式评分解释上的问题，不是模型训练代码缺陷。

审查起点还发现数量 VQA 问句直接套用单数主体，例如 `How many woman are visible...`。本轮已局部修正为 `How many instances of {subject} are visible in this painting?`，对象问句同时避免 `a apple` 冠词问题，并以 woman/child/sheep/horses 四种类别及 apple 对象检查问句、问题 ID、要求和答案保持一致；VQA/校准/评分定向测试 14 项通过。候选 Prompt、分母和批准状态未变，真实 BLIP 仍未执行。新评测源码哈希与历史回执不同，不能把旧评分改标为新版本；训练契约锁定全包源码，因此旧检查点不得跨本轮源码精确续训。

### **2. 两种可审阅的处理方案**

**方案 A，推荐：发布新的候选版本后再批准。** 维持 80/40/40 和数据来源边界，补齐确实需要考察的数量与场景/留空要求；颜色与绑定去除重复加权或明确分开但相关的分项；补充构图与计数判定规范；如需要独立参考，按唯一组重新选满 2,000。若改变题文、要求、分母、参考或评分规则，都须新目录、新版本、新内容锁和新的用户回执，旧候选及历史 smoke 证据保持原身份。不能把新题的协议套到旧成绩。

**方案 B：用户明确批准一个有限范围基准。** 保留题文，仅把自动结果称为“登记子要求满足率”；全文遵循由人工独立评分，未覆盖部分不得被 VQA 总成功率替代。接受当前近域验证、轻量改写、同组重复参考的加权及 color/binding 的相关性，按类别分开报告，不将微平均值宣称为独立视觉事实准确率，也不宣称颜色交换能力或真实 OOD。这个限定范围仍属于评分解释协议的正式变更，应形成独立可追溯修订及用户回执，不能仅在口头说明后把旧候选直接盖章。

本轮没有擅自选择方案，也没有修改候选。正式训练前需要用户决定并冻结，避免根据最终模型效果改题；验证集 CUDA Pilot 可以先验证工程和既有 P3 质量门槛，不运行这批主测试题选型。

## **四、指标协议、真实状态及分母**

### **1. 评测器身份与定义**

| 指标 | 当前固定实现 | 已有证据与待验 |
| --- | --- | --- |
| KID | torch-fidelity Inception 2048；未归一化特征；三次核、无偏 MMD²；100 次 × 1,000 子集；seed 20261110；std 的 ddof=0；保留合法负值 | NumPy/参考实现对照及真实本地加载通过；正式 2,000 图与 CUDA 未执行 |
| CMMD | 完整 CLIP L/14@336 图像投影归一化；RBF sigma=10；包含对角线的 biased MMD² × 1,000；CPU float64 分块聚合 | 分块数值/真实本机通过；不宣称其他后端逐位等价 |
| CLIPScore | 同一完整 L/14@336 图文投影；不添加前缀；`2.5 × max(cosine,0)`；记录截断与长度 | 真实本地提取通过；是显式变体，不能与不同模型/前缀得分直接比较 |
| LPIPS | P2 已有 AlexNet 特征及校准权重；同 Prompt 两 seed 距离平均 | 真实本地 4 对/模型通过；多样性辅助，不等于质量或条件遵循 |
| VQA | `Salesforce/blip-vqa-base`；直接 `BlipForQuestionAnswering` 与 processor；FP32、greedy、`max_new_tokens=16`；原始答案及有限词规范化 | 真实适配器已实现；本地仅 Mock/契约，权重大文件未下载、真实推理/绘画域可靠性未验 |
| 人工盲评 | 匿名图像/模型文件名，固定乱序；质量与遵循各 1–5，错误类型及逐要求判定；私有映射分离 | 导入导出/空答案/失败图片流程已测；没有真实人工评分与一致率 |

资产锁：[evaluation.json](../../../configs/models/evaluation.json)。CLIP revision 为 `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`；BLIP revision 为 `787b3d35d57e49572baabd22884b3d5a05acf072`；Inception 固定 `v0.2.0-tf2015` 权重 SHA 起始 `6726825d`。LPIPS 的 AlexNet 与校准 SHA 见 [P2 资产回执](../components/p2_assets.json)。生成条件 CLIP-B/32 文本塔不充作评测 CLIP。权重本地路径、精确 SHA、大小和下载 URL 都在锁/回执中，普通评分离线执行。

P5 实测依赖记录为 torch-fidelity 0.3.0、Transformers 5.20.2、PyTorch 2.14.1、Pillow 12.3.0、NumPy 2.5.4；来源为 [检查回执](../experiments/p5/checks.json)，本轮没有重装依赖。迁移后的 CUDA 栈须另行记录实际版本和一致性，不能凭 Mac 回执宣称 CUDA 已通过。

### **2. 分母与失败规则**

| 项目 | 分母/聚合 | 缺图、不可判和缺失值 |
| --- | --- | --- |
| 分布 KID/CMMD | 正式 2,000 参考记录与 2,000 按相同 Caption 分布生成任务；KID 100×1000 子集 | 缺图不伪造特征；只在有效子集计算并标记 `complete=false`、覆盖率和错误。不能当完整正式比较 |
| CLIPScore | 有效图像与同序 Caption 的逐图分数均值 | 无效图不填 0；完整任务仍保留失败，结果不完整。要同时报告 expected/valid，而非只报幸存样本均值 |
| LPIPS | 每个同 Prompt 完整双种子对 | 缺一张就不能形成该对；结合完整生成清单列缺失对。分布任务每 Caption 单 seed，通常不适用 |
| VQA 分项 | 每类别全部适用问题；主测试两 seed 当前 object=400、count=160、color=40、binding=120、relation=120，共 840 | correct/错误/不可判/执行失败均保留。成功率=correct/全部适用；覆盖率=(correct+wrong)/全部适用 |
| VQA 总体 | 当前源码 `value` 是所有问题的微平均；`all_requirements_correct` 是逐图登记要求全对的数量 | 主测试适用图分母为 320，失败图总体失败。不能把微平均与逐图综合成功率混同，也不能把重复问题当独立证据 |
| 人工评分 | 实际有对应评分的样本，另列 expected/answered/complete | 未填不自动计零，也不自动推断。失败图保留审核占位，不能删掉；完整性检查不能替代审核人判断 |
| 校准一致率 | 人工与自动可判配对，额外列全部配对/不可判/未配对/逐类混淆 | 没有人工输入时不存在一致率；只对可判子集的比例必须带覆盖率，不代表总体可靠性 |

真实 BLIP 在绘画域至少需要约 100 真图 + 100 生成或退化图的独立人评校准。当前 8+8 材料只验证工具链、主体/场景问题，不足以校准五种属性，也不是模型生成图。即使以后真实 BLIP 命令执行完成，回执仍会显示 `calibrated=false`，不能自行升级为人工真值。预期答案来自 Prompt 要求，人工校准才判断图片是否满足。

### **3. Validation、Test 与效率对照**

采样选型只用 Validation，冻结选择后才执行主 Prompt Test 和独立分布 Test。Euler 10/20/40 步对应 solver NFE 10/20/40；Heun 5/10/20 步对应相同 solver NFE。CFG1 单条件分支，CFG2/3 额外无条件分支须单列实际主干调用数。不能把相同 solver NFE 称为相同 CFG 总计算。

[采样矩阵](../../../configs/benchmark/sampling_matrix.json) 与评测资产锁已绑定在候选内容中。naive/SDPA 是局部主干后端对照，冻结 VAE/CLIP 不切换。延迟记录硬件、精度、batch、预热/同步、主干与端到端耗时；MPS 当前内存不是 CUDA 峰值，缺失 CUDA 数据保持空值。不同结果类型 `real_local`、`simulated`、`formal` 不能混入一张正式比较表。

## **五、用户审核清单与下一步**

- [ ] 确认使用“组合压力”名称，不宣称真正 Composition-OOD；接受 ID 只有训练词汇证据的限制。
- [ ] 在方案 A/B 中确定评分范围，解决数量、场景/右侧留空遗漏与重复颜色问题；如改内容/协议，发布新版本再审。
- [ ] 确认左/上、数量实例、遮挡、多色及不可判的人工规则；决定同色组合如何报告。
- [ ] 确认近域 Validation 和轻量 Paraphrase 的用途，不扩大泛化声明。
- [ ] 确认 2,000 参考记录中的同组重复处理方式，以及 smoke/calibration 参考重叠的披露。
- [ ] 核对 KID/CMMD/CLIPScore 变体、分母、失败处理与 LPIPS 辅助用途。
- [ ] 计划真实 BLIP、五类问题 100+100 校准和匿名人工盲评；未实测部分继续待验。
- [ ] 完成上述决定后，批准最终明确的内容哈希与协议，独立保留 reviewer/reviewed_at/benchmark_sha256 回执。

**停止点：等待用户审核。** 本文件不触发正式训练、正式评分或批准；CUDA Pilot 仅使用训练子集和独立验证条件，并保留 P3 固定 16 图质量尚未通过的停止条件。

## **六、全部题目逐项审查索引**

以下表逐一列出本轮核对的题文、配对和需决定事项。标记是审查提示，不是自动拒绝题目，也不表示用户已经批准。`场景`=场景短语未独立评分；`数量`=显式数量未评分；`留空`=右侧留空未评分；`位置`=二维定位规则待定；`重复`=颜色/绑定同问句；`同色`=不能测颜色交换；`上方`=二维/物理关系须澄清；`实例`=可数单位需规定。

### **1. ID：80 题**

| ID | Prompt | 配对原题 | 审查标记 |
| --- | --- | --- | --- |
| `id-00-0` | A painting showing one woman wearing a dress in a garden. | — | 场景 |
| `id-00-1` | A painting showing two women wearing a dress in a garden. | — | 场景 |
| `id-00-2` | A painting showing one woman wearing a dress in a garden. The woman's dress is blue. | — | 数量、场景、重复 |
| `id-00-3` | A painting showing one woman on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-01-0` | A painting showing one man wearing a coat in a room. | — | 场景 |
| `id-01-1` | A painting showing two men wearing a coat in a room. | — | 场景 |
| `id-01-2` | A painting showing one man wearing a coat in a room. The man's coat is brown. | — | 数量、场景、重复 |
| `id-01-3` | A painting showing one man on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-02-0` | A painting showing one dog resting beside a country path. | — | 场景 |
| `id-02-1` | A painting showing two dogs resting beside a country path. | — | 场景 |
| `id-02-2` | A painting showing one dog resting beside a country path. The dog is black. | — | 数量、场景、重复 |
| `id-02-3` | A painting showing one dog on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-03-0` | A painting showing one horse standing in a meadow. | — | 场景 |
| `id-03-1` | A painting showing two horses standing in a meadow. | — | 场景 |
| `id-03-2` | A painting showing one horse standing in a meadow. The horse is brown. | — | 数量、场景、重复 |
| `id-03-3` | A painting showing one horse on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-04-0` | A painting showing one bird sitting on a branch. | — | 场景 |
| `id-04-1` | A painting showing two birds sitting on a branch. | — | 场景 |
| `id-04-2` | A painting showing one bird sitting on a branch. The bird is white. | — | 数量、场景、重复 |
| `id-04-3` | A painting showing one bird on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-05-0` | A painting showing one tree beside a river in a landscape. | — | 场景 |
| `id-05-1` | A painting showing two trees beside a river in a landscape. | — | 场景 |
| `id-05-2` | A painting showing one tree beside a river in a landscape. The tree is green. | — | 数量、场景、重复 |
| `id-05-3` | A painting showing one tree on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-06-0` | A painting showing one flower in a still life. | — | 场景 |
| `id-06-1` | A painting showing two flowers in a still life. | — | 场景 |
| `id-06-2` | A painting showing one flower in a still life. The flower is red. | — | 数量、场景、重复 |
| `id-06-3` | A painting showing one flower on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-07-0` | A painting showing one boat floating on calm water. | — | 场景 |
| `id-07-1` | A painting showing two boats floating on calm water. | — | 场景 |
| `id-07-2` | A painting showing one boat floating on calm water. The boat is blue. | — | 数量、场景、重复 |
| `id-07-3` | A painting showing one boat on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-08-0` | A painting showing one house in a rural landscape. | — | 场景 |
| `id-08-1` | A painting showing two houses in a rural landscape. | — | 场景 |
| `id-08-2` | A painting showing one house in a rural landscape. The house is white. | — | 数量、场景、重复 |
| `id-08-3` | A painting showing one house on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-09-0` | A painting showing one church in a village landscape. | — | 场景 |
| `id-09-1` | A painting showing two churches in a village landscape. | — | 场景 |
| `id-09-2` | A painting showing one church in a village landscape. The church is white. | — | 数量、场景、重复 |
| `id-09-3` | A painting showing one church on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-10-0` | A painting showing one bridge crossing a river. | — | 场景 |
| `id-10-1` | A painting showing two bridges crossing a river. | — | 场景 |
| `id-10-2` | A painting showing one bridge crossing a river. The bridge is brown. | — | 数量、场景、重复 |
| `id-10-3` | A painting showing one bridge on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-11-0` | A painting showing one vase in a painted still life. | — | 场景 |
| `id-11-1` | A painting showing two vases in a painted still life. | — | 场景 |
| `id-11-2` | A painting showing one vase in a painted still life. The vase is blue. | — | 数量、场景、重复 |
| `id-11-3` | A painting showing one vase on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-12-0` | A painting showing one table inside a quiet room. | — | 场景 |
| `id-12-1` | A painting showing two tables inside a quiet room. | — | 场景 |
| `id-12-2` | A painting showing one table inside a quiet room. The table is brown. | — | 数量、场景、重复 |
| `id-12-3` | A painting showing one table on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-13-0` | A painting showing one chair inside a painted interior. | — | 场景 |
| `id-13-1` | A painting showing two chairs inside a painted interior. | — | 场景 |
| `id-13-2` | A painting showing one chair inside a painted interior. The chair is red. | — | 数量、场景、重复 |
| `id-13-3` | A painting showing one chair on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-14-0` | A painting showing one window on the wall of an interior. | — | 场景 |
| `id-14-1` | A painting showing two windows on the wall of an interior. | — | 场景 |
| `id-14-2` | A painting showing one window on the wall of an interior. The window frame is white. | — | 数量、场景、重复 |
| `id-14-3` | A painting showing one window on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-15-0` | A painting showing one book in a still life. | — | 场景 |
| `id-15-1` | A painting showing two books in a still life. | — | 场景 |
| `id-15-2` | A painting showing one book in a still life. The book cover is red. | — | 数量、场景、重复 |
| `id-15-3` | A painting showing one book on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-16-0` | A painting showing one bowl in a still life. | — | 场景 |
| `id-16-1` | A painting showing two bowls in a still life. | — | 场景 |
| `id-16-2` | A painting showing one bowl in a still life. The bowl is white. | — | 数量、场景、重复 |
| `id-16-3` | A painting showing one bowl on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-17-0` | A painting showing one apple in a painted still life. | — | 场景 |
| `id-17-1` | A painting showing two apples in a painted still life. | — | 场景 |
| `id-17-2` | A painting showing one apple in a painted still life. The apple is red. | — | 数量、场景、重复 |
| `id-17-3` | A painting showing one apple on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-18-0` | A painting showing one bottle in a still life. | — | 场景 |
| `id-18-1` | A painting showing two bottles in a still life. | — | 场景 |
| `id-18-2` | A painting showing one bottle in a still life. The bottle is green. | — | 数量、场景、重复 |
| `id-18-3` | A painting showing one bottle on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `id-19-0` | A painting showing one tower rising above a city landscape. | — | 场景 |
| `id-19-1` | A painting showing two towers rising above a city landscape. | — | 场景 |
| `id-19-2` | A painting showing one tower rising above a city landscape. The tower is brown. | — | 数量、场景、重复 |
| `id-19-3` | A painting showing one tower on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |

### **2. 组合压力：40 题**

| ID | Prompt | 配对原题 | 审查标记 |
| --- | --- | --- | --- |
| `composition-00-0` | An artistic painting showing one woman to the left of one boat. | — | 数量 |
| `composition-00-1` | An artistic painting showing one woman above one boat. The woman's dress is blue, and the boat is blue. | — | 数量、同色、上方 |
| `composition-01-0` | An artistic painting showing one man to the left of one house. | — | 数量 |
| `composition-01-1` | An artistic painting showing one man above one house. The man's coat is brown, and the house is white. | — | 数量、上方 |
| `composition-02-0` | An artistic painting showing one dog to the left of one church. | — | 数量 |
| `composition-02-1` | An artistic painting showing one dog above one church. The dog is black, and the church is white. | — | 数量、上方 |
| `composition-03-0` | An artistic painting showing one horse to the left of one bridge. | — | 数量 |
| `composition-03-1` | An artistic painting showing one horse above one bridge. The horse is brown, and the bridge is brown. | — | 数量、同色、上方 |
| `composition-04-0` | An artistic painting showing one bird to the left of one vase. | — | 数量 |
| `composition-04-1` | An artistic painting showing one bird above one vase. The bird is white, and the vase is blue. | — | 数量、上方 |
| `composition-05-0` | An artistic painting showing one tree to the left of one table. | — | 数量 |
| `composition-05-1` | An artistic painting showing one tree above one table. The tree is green, and the table is brown. | — | 数量、上方 |
| `composition-06-0` | An artistic painting showing one flower to the left of one chair. | — | 数量 |
| `composition-06-1` | An artistic painting showing one flower above one chair. The flower is red, and the chair is red. | — | 数量、同色、上方 |
| `composition-07-0` | An artistic painting showing one boat to the left of one window. | — | 数量 |
| `composition-07-1` | An artistic painting showing one boat above one window. The boat is blue, and the window frame is white. | — | 数量、上方 |
| `composition-08-0` | An artistic painting showing one house to the left of one book. | — | 数量 |
| `composition-08-1` | An artistic painting showing one house above one book. The house is white, and the book cover is red. | — | 数量、上方 |
| `composition-09-0` | An artistic painting showing one church to the left of one bowl. | — | 数量 |
| `composition-09-1` | An artistic painting showing one church above one bowl. The church is white, and the bowl is white. | — | 数量、同色、上方 |
| `composition-10-0` | An artistic painting showing one bridge to the left of one apple. | — | 数量 |
| `composition-10-1` | An artistic painting showing one bridge above one apple. The bridge is brown, and the apple is red. | — | 数量、上方 |
| `composition-11-0` | An artistic painting showing one vase to the left of one bottle. | — | 数量 |
| `composition-11-1` | An artistic painting showing one vase above one bottle. The vase is blue, and the bottle is green. | — | 数量、上方 |
| `composition-12-0` | An artistic painting showing one table to the left of one tower. | — | 数量 |
| `composition-12-1` | An artistic painting showing one table above one tower. The table is brown, and the tower is brown. | — | 数量、同色、上方 |
| `composition-13-0` | An artistic painting showing one chair to the left of one woman. | — | 数量 |
| `composition-13-1` | An artistic painting showing one chair above one woman. The chair is red, and the woman's dress is blue. | — | 数量、上方 |
| `composition-14-0` | An artistic painting showing one window to the left of one man. | — | 数量 |
| `composition-14-1` | An artistic painting showing one window above one man. The window frame is white, and the man's coat is brown. | — | 数量、上方 |
| `composition-15-0` | An artistic painting showing one book to the left of one dog. | — | 数量 |
| `composition-15-1` | An artistic painting showing one book above one dog. The book cover is red, and the dog is black. | — | 数量、上方 |
| `composition-16-0` | An artistic painting showing one bowl to the left of one horse. | — | 数量 |
| `composition-16-1` | An artistic painting showing one bowl above one horse. The bowl is white, and the horse is brown. | — | 数量、上方 |
| `composition-17-0` | An artistic painting showing one apple to the left of one bird. | — | 数量 |
| `composition-17-1` | An artistic painting showing one apple above one bird. The apple is red, and the bird is white. | — | 数量、上方 |
| `composition-18-0` | An artistic painting showing one bottle to the left of one tree. | — | 数量 |
| `composition-18-1` | An artistic painting showing one bottle above one tree. The bottle is green, and the tree is green. | — | 数量、同色、上方 |
| `composition-19-0` | An artistic painting showing one tower to the left of one flower. | — | 数量 |
| `composition-19-1` | An artistic painting showing one tower above one flower. The tower is brown, and the flower is red. | — | 数量、上方 |

### **3. 成对改写：40 题**

| ID | Prompt | 配对原题 | 审查标记 |
| --- | --- | --- | --- |
| `paraphrase-id-00-0` | Depict one woman wearing a dress in a garden as a painted artwork. | id-00-0 | 场景 |
| `paraphrase-id-00-1` | Depict two women wearing a dress in a garden as a painted artwork. | id-00-1 | 场景 |
| `paraphrase-id-01-0` | Depict one man wearing a coat in a room as a painted artwork. | id-01-0 | 场景 |
| `paraphrase-id-01-1` | Depict two men wearing a coat in a room as a painted artwork. | id-01-1 | 场景 |
| `paraphrase-id-02-0` | Depict one dog resting beside a country path as a painted artwork. | id-02-0 | 场景 |
| `paraphrase-id-02-1` | Depict two dogs resting beside a country path as a painted artwork. | id-02-1 | 场景 |
| `paraphrase-id-03-0` | Depict one horse standing in a meadow as a painted artwork. | id-03-0 | 场景 |
| `paraphrase-id-03-1` | Depict two horses standing in a meadow as a painted artwork. | id-03-1 | 场景 |
| `paraphrase-id-04-0` | Depict one bird sitting on a branch as a painted artwork. | id-04-0 | 场景 |
| `paraphrase-id-04-1` | Depict two birds sitting on a branch as a painted artwork. | id-04-1 | 场景 |
| `paraphrase-id-05-0` | Depict one tree beside a river in a landscape as a painted artwork. | id-05-0 | 场景 |
| `paraphrase-id-05-1` | Depict two trees beside a river in a landscape as a painted artwork. | id-05-1 | 场景 |
| `paraphrase-id-06-0` | Depict one flower in a still life as a painted artwork. | id-06-0 | 场景 |
| `paraphrase-id-06-1` | Depict two flowers in a still life as a painted artwork. | id-06-1 | 场景 |
| `paraphrase-id-07-0` | Depict one boat floating on calm water as a painted artwork. | id-07-0 | 场景 |
| `paraphrase-id-07-1` | Depict two boats floating on calm water as a painted artwork. | id-07-1 | 场景 |
| `paraphrase-id-08-0` | Depict one house in a rural landscape as a painted artwork. | id-08-0 | 场景 |
| `paraphrase-id-08-1` | Depict two houses in a rural landscape as a painted artwork. | id-08-1 | 场景 |
| `paraphrase-id-09-0` | Depict one church in a village landscape as a painted artwork. | id-09-0 | 场景 |
| `paraphrase-id-09-1` | Depict two churches in a village landscape as a painted artwork. | id-09-1 | 场景 |
| `paraphrase-id-10-0` | Depict one bridge crossing a river as a painted artwork. | id-10-0 | 场景 |
| `paraphrase-id-10-1` | Depict two bridges crossing a river as a painted artwork. | id-10-1 | 场景 |
| `paraphrase-id-11-0` | Depict one vase in a painted still life as a painted artwork. | id-11-0 | 场景 |
| `paraphrase-id-11-1` | Depict two vases in a painted still life as a painted artwork. | id-11-1 | 场景 |
| `paraphrase-id-12-0` | Depict one table inside a quiet room as a painted artwork. | id-12-0 | 场景 |
| `paraphrase-id-12-1` | Depict two tables inside a quiet room as a painted artwork. | id-12-1 | 场景 |
| `paraphrase-id-13-0` | Depict one chair inside a painted interior as a painted artwork. | id-13-0 | 场景 |
| `paraphrase-id-13-1` | Depict two chairs inside a painted interior as a painted artwork. | id-13-1 | 场景 |
| `paraphrase-id-14-0` | Depict one window on the wall of an interior as a painted artwork. | id-14-0 | 场景 |
| `paraphrase-id-14-1` | Depict two windows on the wall of an interior as a painted artwork. | id-14-1 | 场景 |
| `paraphrase-id-15-0` | Depict one book in a still life as a painted artwork. | id-15-0 | 场景 |
| `paraphrase-id-15-1` | Depict two books in a still life as a painted artwork. | id-15-1 | 场景 |
| `paraphrase-id-16-0` | Depict one bowl in a still life as a painted artwork. | id-16-0 | 场景 |
| `paraphrase-id-16-1` | Depict two bowls in a still life as a painted artwork. | id-16-1 | 场景 |
| `paraphrase-id-17-0` | Depict one apple in a painted still life as a painted artwork. | id-17-0 | 场景 |
| `paraphrase-id-17-1` | Depict two apples in a painted still life as a painted artwork. | id-17-1 | 场景 |
| `paraphrase-id-18-0` | Depict one bottle in a still life as a painted artwork. | id-18-0 | 场景 |
| `paraphrase-id-18-1` | Depict two bottles in a still life as a painted artwork. | id-18-1 | 场景 |
| `paraphrase-id-19-0` | Depict one tower rising above a city landscape as a painted artwork. | id-19-0 | 场景 |
| `paraphrase-id-19-1` | Depict two towers rising above a city landscape as a painted artwork. | id-19-1 | 场景 |

### **4. Validation：80 题**

| ID | Prompt | 配对原题 | 审查标记 |
| --- | --- | --- | --- |
| `validation-00-0` | A painting showing one girl wearing a skirt in a courtyard. | — | 场景 |
| `validation-00-1` | A painting showing two girls wearing a skirt in a courtyard. | — | 场景 |
| `validation-00-2` | A painting showing one girl wearing a skirt in a courtyard. The girl's skirt is red. | — | 数量、场景、重复 |
| `validation-00-3` | A painting showing one girl on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-01-0` | A painting showing one boy wearing a shirt in a courtyard. | — | 场景 |
| `validation-01-1` | A painting showing two boys wearing a shirt in a courtyard. | — | 场景 |
| `validation-01-2` | A painting showing one boy wearing a shirt in a courtyard. The boy's shirt is blue. | — | 数量、场景、重复 |
| `validation-01-3` | A painting showing one boy on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-02-0` | A painting showing one cat sitting on a doorstep. | — | 场景 |
| `validation-02-1` | A painting showing two cats sitting on a doorstep. | — | 场景 |
| `validation-02-2` | A painting showing one cat sitting on a doorstep. The cat is black. | — | 数量、场景、重复 |
| `validation-02-3` | A painting showing one cat on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-03-0` | A painting showing one cow standing beside a fence. | — | 场景 |
| `validation-03-1` | A painting showing two cows standing beside a fence. | — | 场景 |
| `validation-03-2` | A painting showing one cow standing beside a fence. The cow is brown. | — | 数量、场景、重复 |
| `validation-03-3` | A painting showing one cow on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-04-0` | A painting showing one sheep standing on a hillside. | — | 场景 |
| `validation-04-1` | A painting showing two sheep standing on a hillside. | — | 场景 |
| `validation-04-2` | A painting showing one sheep standing on a hillside. The sheep is white. | — | 数量、场景、重复 |
| `validation-04-3` | A painting showing one sheep on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-05-0` | A painting showing one goat standing among rocks. | — | 场景 |
| `validation-05-1` | A painting showing two goats standing among rocks. | — | 场景 |
| `validation-05-2` | A painting showing one goat standing among rocks. The goat is white. | — | 数量、场景、重复 |
| `validation-05-3` | A painting showing one goat on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-06-0` | A painting showing one rose in a floral still life. | — | 场景 |
| `validation-06-1` | A painting showing two roses in a floral still life. | — | 场景 |
| `validation-06-2` | A painting showing one rose in a floral still life. The rose is pink. | — | 数量、场景、重复 |
| `validation-06-3` | A painting showing one rose on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-07-0` | A painting showing one mountain behind a broad valley. | — | 场景、实例 |
| `validation-07-1` | A painting showing two mountains behind a broad valley. | — | 场景、实例 |
| `validation-07-2` | A painting showing one mountain behind a broad valley. The mountain is blue. | — | 数量、场景、重复、实例 |
| `validation-07-3` | A painting showing one mountain on the left side of the composition, with empty space on the right. | — | 数量、留空、位置、实例 |
| `validation-08-0` | A painting showing one ship sailing across the sea. | — | 场景 |
| `validation-08-1` | A painting showing two ships sailing across the sea. | — | 场景 |
| `validation-08-2` | A painting showing one ship sailing across the sea. The ship is black. | — | 数量、场景、重复 |
| `validation-08-3` | A painting showing one ship on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-09-0` | A painting showing one castle above a wooded valley. | — | 场景 |
| `validation-09-1` | A painting showing two castles above a wooded valley. | — | 场景 |
| `validation-09-2` | A painting showing one castle above a wooded valley. The castle is gray. | — | 数量、场景、重复 |
| `validation-09-3` | A painting showing one castle on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-10-0` | A painting showing one cloud above a painted seascape. | — | 场景、实例 |
| `validation-10-1` | A painting showing two clouds above a painted seascape. | — | 场景、实例 |
| `validation-10-2` | A painting showing one cloud above a painted seascape. The cloud is white. | — | 数量、场景、重复、实例 |
| `validation-10-3` | A painting showing one cloud on the left side of the composition, with empty space on the right. | — | 数量、留空、位置、实例 |
| `validation-11-0` | A painting showing one pear in an arrangement of fruit. | — | 场景 |
| `validation-11-1` | A painting showing two pears in an arrangement of fruit. | — | 场景 |
| `validation-11-2` | A painting showing one pear in an arrangement of fruit. The pear is yellow. | — | 数量、场景、重复 |
| `validation-11-3` | A painting showing one pear on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-12-0` | A painting showing one hat in a simple still life. | — | 场景 |
| `validation-12-1` | A painting showing two hats in a simple still life. | — | 场景 |
| `validation-12-2` | A painting showing one hat in a simple still life. The hat is black. | — | 数量、场景、重复 |
| `validation-12-3` | A painting showing one hat on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-13-0` | A painting showing one cup on a kitchen shelf. | — | 场景 |
| `validation-13-1` | A painting showing two cups on a kitchen shelf. | — | 场景 |
| `validation-13-2` | A painting showing one cup on a kitchen shelf. The cup is blue. | — | 数量、场景、重复 |
| `validation-13-3` | A painting showing one cup on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-14-0` | A painting showing one plate on a kitchen shelf. | — | 场景 |
| `validation-14-1` | A painting showing two plates on a kitchen shelf. | — | 场景 |
| `validation-14-2` | A painting showing one plate on a kitchen shelf. The plate is white. | — | 数量、场景、重复 |
| `validation-14-3` | A painting showing one plate on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-15-0` | A painting showing one basket on a stone floor. | — | 场景 |
| `validation-15-1` | A painting showing two baskets on a stone floor. | — | 场景 |
| `validation-15-2` | A painting showing one basket on a stone floor. The basket is brown. | — | 数量、场景、重复 |
| `validation-15-3` | A painting showing one basket on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-16-0` | A painting showing one fence beside a pasture. | — | 场景、实例 |
| `validation-16-1` | A painting showing two fences beside a pasture. | — | 场景、实例 |
| `validation-16-2` | A painting showing one fence beside a pasture. The fence is white. | — | 数量、场景、重复、实例 |
| `validation-16-3` | A painting showing one fence on the left side of the composition, with empty space on the right. | — | 数量、留空、位置、实例 |
| `validation-17-0` | A painting showing one door on a building facade. | — | 场景 |
| `validation-17-1` | A painting showing two doors on a building facade. | — | 场景 |
| `validation-17-2` | A painting showing one door on a building facade. The door is green. | — | 数量、场景、重复 |
| `validation-17-3` | A painting showing one door on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-18-0` | A painting showing one rock beside the coast. | — | 场景 |
| `validation-18-1` | A painting showing two rocks beside the coast. | — | 场景 |
| `validation-18-2` | A painting showing one rock beside the coast. The rock is gray. | — | 数量、场景、重复 |
| `validation-18-3` | A painting showing one rock on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
| `validation-19-0` | A painting showing one lamp inside a domestic interior. | — | 场景 |
| `validation-19-1` | A painting showing two lamps inside a domestic interior. | — | 场景 |
| `validation-19-2` | A painting showing one lamp inside a domestic interior. The lamp is yellow. | — | 数量、场景、重复 |
| `validation-19-3` | A painting showing one lamp on the left side of the composition, with empty space on the right. | — | 数量、留空、位置 |
