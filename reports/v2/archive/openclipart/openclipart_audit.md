## **一、OpenClipart 历史审计**

此报告是已下线的 P1 试样记录。当前数据源及阶段状态见 [P0/P1 交付](../../01_p0_p1_delivery.md)。

2026-11-06（Asia/Shanghai）。完成固定版本的来源与试样审计。400 条试样中 231 条通过安全栅格化，预览、试切分和 120 条待标注队列已生成。目视抽查给出了首版风格筛选规则。正式数据需要更广的风格筛选、标注模型试验和人工复核，当前样本为审计用途。

| 阶段 | 当前证据 | 状态 |
| --- | --- | --- |
| P1-01 来源、许可与抽样 | 固定 18 个原始分片的前缀和独立回执，固定种子抽取 400 条 | 有界审计完成，推断范围限于前缀抽样框 |
| P1-02 画风与可用数 | 72 张目视抽查，初版留删规则和样本索引 | 试样规则已试用，完整语料的可用量待后续筛选 |
| P1-03 渲染与隔离 | 231 张 RGB，101 个组，试切分 185/23/23，已检查跨集冲突 | 试样链路已验证，正式数据仍待筛选、复核并重新切分 |
| P1-04 Caption 模型试验 | 120 条有哈希的试验队列，本地候选固定版本 | 等待一套可运行的 Caption VLM，尚无模型指标 |
| P1-05 标注、人工复核与冻结 | 清单字段与人工复核口径，保留草稿与实际状态 | 待 P1-04 及正式数据筛选，当前没有冻结训练清单 |
| P1-06 多主体覆盖 | 9 张风格候选中仅 1 张有明确多主体互动 | 试样初判。组合场景需要更广证据及用户决定 |

下载资源、已核对的本机权重和后续阶段对应关系见 [P0–P7 下载清单](../../../../docs/MiniGen2下载清单.md)。

## **二、固定来源与取样范围**

镜像为 `kawwaaa/openclipart`，固定 revision `df38e5ef633be067d397df7da19a9f009e69b3bd`。固定树 18 个 `.jsonl.zst` 共 22,340,611,707 字节。卡片所述 178,604 条是上游声明。Viewer API 实测只提供 9,309 条部分行，不能作为全库抽样框。原始 JSONL 包含 SVG 正文、原站页、作者、创建时间、标签与许可。镜像声明 CC0-1.0，原站也说明用 CC0 接收素材。逐条检查原站 URL、行许可和 `pd_issue` 等待审标记，保留原站出处。来源见 [固定镜像](https://huggingface.co/datasets/kawwaaa/openclipart/tree/df38e5ef633be067d397df7da19a9f009e69b3bd)、[数据卡](https://huggingface.co/datasets/kawwaaa/openclipart) 和 [OpenClipart FAQ](https://openclipart.org/faq)。

13 个分片使用 6 MiB 前缀。其余 5 个的首轮响应中断，改用 3 MiB 后通过。最终保留 **93 MiB（97,517,568 字节）** 的精确 HTTP 206 响应。每个回执保存来源、范围、完整分片身份与实际前缀 SHA-256。首轮范围请求上限 108 MiB，第二轮只对五个缺失分片请求至多 15 MiB。两轮正式范围请求的响应体上限合计 123 MiB，另有少量访问探针和仓库元数据。93 MiB 指验证通过并落盘的前缀，失败传输的实际字节未逐项计量。本轮云与付费 API 支出 0。

每个前缀最多读取 120 条完整行，另设单行 4 MiB 和解码 128 MiB 上限。超长完整行仍保留抽样位置，末尾残行剔除。实际抽样框有 1,146 条完整行。以种子 `20261106` 对分片分层抽取 400 条，涵盖 2006–2025 年的可识别日期。该设计横跨原始分片，但只覆盖各分片的前缀。原始行顺序可能与标题、主题、作者或系列有关，不能用本次留存率推算全库。逐分片选中行号、前缀哈希、日期与标签分布见 [审计统计](openclipart_audit_summary.json)。

## **三、栅格化、去重和试切分**

使用锁定的 CairoSVG 2.9.1。SVG 进入渲染器前经过静态检查，拒绝外部资源、脚本、危险引用和超出尺寸或复杂度限制的内容。渲染进程禁止资源加载，限制 CPU 时间、墙钟时间和输出画布。macOS 上采用静态复杂度和画布限制，Linux worker 另有地址空间上限。安全子集会排除部分合法但复杂的 SVG，因此结果反映当前管线的安全可渲染率。

| 项目 | 结果 |
| --- | ---: |
| 总试样 | 400 |
| 进入 SVG 校验与渲染 | 392 |
| 成功生成 256×256 RGB | 231 |
| 拒绝的资源或图形功能 | 93 |
| 超出静态复杂度或尺寸限制 | 62 |
| 完整 JSONL 行超限 | 8 |
| 其他失败 | 6 |
| 成功图原始内容带透明区域 | 226/231 |
| 成功图含 SVG 文本元素 | 8/231 |
| 横幅、竖幅、方形 | 107、95、29 |

等比缩放并居中填充白底，保留源尺寸、viewBox、内容框与原透明性。已逐张重新验证 231 张 PNG 的模式、尺寸、可解码性和 SHA-256。

以作品 ID、已知作者、SVG 与 PNG 精确哈希、dHash 距离不超过 3 的近重候选形成连通组。231 张共 101 组，最大组 25 张。试切分按组得到训练 185、验证 23、测试 23。发现 5 对 dHash 候选（1 对 dHash 完全相同，4 对距离 1–3），逐对复查无跨集候选。该诊断没有穷尽近重，同作者归组也可能偏保守。所有记录为 `audit_only` 且 `training_eligible=false`。正式切分还需核对同源变体和语义近重并封存测试组。

## **四、目标画风与多主体试样**

在 231 张成功预览中按分片抽查 72 张，固定种子 `20261108`。代理目视筛选的结果是保留候选 9、边界 27、排除 36。逐行决定见 [筛选记录](openclipart_style_review.json)。这 72 张的判定用于拟定规则，人工审核字段为空。

| 决定 | 首版规则与实例 |
| --- | --- |
| 保留候选 | 主体语义明确，扁平色块、适量细节、轮廓清楚，缺少明显水印、徽标与大段文字。例 `01-0027` 企鹅、`05-0083` 狗 |
| 边界 | 识别得到主体，但以线稿、阴影、单色剪影或低对比度为主。例 `13-0026` 单色狗、`08-0062` 阴影较重的肖像 |
| 排除 | 标志、文字图、照片或写实纹理、抽象纹样、裁切/空白、主题不清或同一图中并置重复变体。例 `14-0059` 书封文字、`15-0067` 限速牌 |

严格候选中明确的多主体互动只有 `03-0036`。器物部件、背景、重复图标和装饰元素单列，不用于空间关系真值。接下来优先从同一 OpenClipart 来源补查带有两个可辨识主体的自然场景。若仍不足，再单独提报同源组合场景方案，避免为增加数量而引入不受控拼贴。当前无法给出 8,000–12,000 张目标集的可信有效量。

## **五、Caption 试验与正式冻结的前置**

`artifacts/audits/openclipart_p1_01/preview/caption_pilot.jsonl` 保存 120 个不同样本 ID、图片路径、PNG 哈希与出处。队列覆盖多个分片和难度，其中包含待进一步风格复核的图像。先用固定 VLM 运行该队列并保留原始响应，记录 JSON 有效率、事实错误、幻觉、延迟和峰值内存。Caption 提示只描述画面可确认的对象、颜色、数量、相对关系和风格，不能判断的字段写 `unknown` 或 `uncertain`。

输出需要 `caption_en`、`objects`、`relations`、`style_label`、`uncertain` 和可追溯的模型、revision、提示模板。用本地 CLIP tokenizer 校验 75 内容 token 上限，超限重写或拒绝。试验结果与人工评分分字段保存。正式数据冻结前分层人工复核至少 200 张，保留初始标注和修订。没有模型试验和人工证据时，队列空字段保持空。

## **六、复现与检查**

```bash
uv sync --locked --inexact --extra openclipart
uv run --locked --offline --no-sync python scripts/audit_openclipart.py \
  --config configs/datasets/openclipart_v2.yaml --dry-run
uv run --locked --offline --no-sync python scripts/audit_openclipart.py \
  --config configs/datasets/openclipart_v2.yaml --download-only \
  --output-dir artifacts/audits/openclipart_p1_01
uv run --locked --offline --no-sync python scripts/audit_openclipart.py \
  --config configs/datasets/openclipart_v2.yaml --offline \
  --output-dir artifacts/audits/openclipart_p1_01
```

`uv --offline` 限制 uv 自身获取依赖。脚本的 `--download-only` 是显式范围下载，脚本的 `--offline` 只复核本地回执和生成预览。已有 `preview/` 时拒绝覆盖。真实结果见本地 `preview/records.jsonl`、`summary.json`、`contact_sheets/` 与 `style_review/`，完整源和 PNG 保持忽略。报告中的统计与筛选记录是可审查的小型证据。

新增 31 项离线测试覆盖范围读取、截断 Zstandard 帧、SVG 安全与等比栅格化、组级切分。新 Python 文件通过 Ruff 和格式检查。全量 pytest 为 137 通过、1 失败，失败为旧 bootstrap 用例仍预期已实现的 DiT `patchify` 抛出待实现异常。

P1 文档链接与改动范围已复核。五份 S1–S3 Notebook 的 SHA-256 匹配 P1 开始前基线，已有依赖版本保持一致。冻结 VAE 与 CLIP 文本塔的 9 个文件重新核对大小和 SHA-256 全部通过。
