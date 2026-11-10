## **一、先明确执行顺序和验收边界**

本手册依据 P5 新增的真实终端接口编写。**CUDA 命令已实现，但本轮未在 CUDA 上执行；读到命令不表示该项已经验收通过。** 本轮停在用户审核 Benchmark 与本手册的入口。后续由用户决定租机、费用和运行时长，然后按本手册逐项操作。

正式规格固定为 ArtBench Public Domain 46,298 张、256×256 RGB、冻结 SDXL VAE 与 CLIP-B/32 文本塔、Flow Matching、B0/B1/B2 约 30M。训练 37,042 张，验证 4,637 张，测试 4,619 张；正式缓存只编码训练和验证。P3 的 16 图可辨质量门槛尚未通过，不能因为 Pilot 的损失有限就直接开始长训。

| 顺序 | 工作 | 必须通过的门槛 | 失败时停在哪里 |
| --- | --- | --- | --- |
| 0 | 审核协议、固定代码提交和预算 | Benchmark 人工批准单独留档；确认单卡范围 | 不运行正式 Benchmark 或长训 |
| 1 | Linux 环境、资产迁移和缓存独立回读 | CUDA 可用、版本可追踪、37,042/4,637 行及哈希一致 | 修复环境或迁移，不重写数据锁 |
| 2 | B0 CPU/CUDA FP32、AMP、恢复和 GPU Pilot | 有限数值、可恢复、显存和耗时实测 | 定位数值问题，不能用 MPS 结果代替 |
| 3 | B0 受控单图/16 图生成和条件对照 | 按既有标准判断主体可辨和条件作用 | 不进入正式三模型长训 |
| 4 | B1/B2 独立 Pilot、共同训练配方 | 各自实测，三模型配方公平；费用可接受 | 修改后重新 Pilot |
| 5 | B0/B1/B2 正式训练与中间检查点 | 有效 batch、数据、优化配方和预定比较点冻结 | 保留失败记录，恢复有效检查点 |
| 6 | 验证集选择采样器、CFG、NFE | 验证数据专用；选择结果在测试前冻结 | 不借主测试反复选参数 |
| 7 | 正式生成、指标、VQA 校准与盲评 | 协议审批、全量任务、缺失与失败明确记录 | 不把不完整成绩包装成正式完整结果 |
| 8 | 同步结果、核对备份、释放实例 | 模型与回执离机，独立哈希校验通过 | 不先删除唯一副本 |

所有路径默认相对于仓库根目录。以下 `bash` 命令在云服务器执行；标明 **Mac** 的段落在 Mac 执行。每次重新连接后先进入同一仓库并设置环境。所有 `cloud_*` 目录都是新实验目录，已有目录不得覆盖。

## **二、租机、SSH、持久化磁盘与代码**

### **1. 选择和连接实例**

选择提供 Linux、单张 NVIDIA GPU、SSH 和持久化磁盘的实例。GPU 型号和价格由用户依据真实报价选择，本手册不预设某型号一定放得下，也不把示例步数当作批准预算。先租短时段跑 Pilot，再决定正式预算。记录开机时间、计费方式、GPU 时价以及停机后磁盘是否继续计费。

在供应商控制台获取主机地址、用户名和 SSH 端口；私钥只留本机，不写进仓库、配置或日志。在 Mac 的 `~/.ssh/config` 添加自己真实的连接信息，例如：

```sshconfig
Host minigen-gpu
    HostName YOUR_SERVER_ADDRESS
    User YOUR_SERVER_USER
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

**Mac：**先用 `ssh minigen-gpu` 连接，按供应商信息核对首次连接的主机指纹。云端执行：

```bash
nvidia-smi
df -h
pwd
```

确认 GPU 型号、驱动和挂载点。下文以 `/workspace` 为已经挂载的持久化磁盘举例；若供应商使用别处，统一替换 `MINIGEN_ROOT`，不要把数据存进可能随实例销毁的临时系统盘。

### **2. 同步本轮代码，而不是假设 GitHub 已经更新**

本轮仅本地提交，**未自动 Push**。Mac 上检查最终交付的提交哈希，将其填入后续 `MINIGEN_COMMIT`。用户可自行推送相应分支后从云端克隆；也可以传 Git bundle，不需要开放私有数据。

**Mac，选择 GitHub 路线时：**

```bash
cd /Users/lucius/projects/personal/MiniGen
git status --short
git rev-parse HEAD
git push origin HEAD
```

这一步是用户未来主动执行的发布操作。本手册没有代为推送。云端：

```bash
export MINIGEN_ROOT=/workspace/MiniGen
export MINIGEN_COMMIT=填入本轮完整提交哈希
git clone https://github.com/luminoso909/MiniGen.git "$MINIGEN_ROOT"
cd "$MINIGEN_ROOT"
git checkout --detach "$MINIGEN_COMMIT"
git rev-parse HEAD
git status --short
```

若不推送：Mac 执行 `git bundle create /tmp/MiniGen.bundle HEAD`，用 `scp /tmp/MiniGen.bundle minigen-gpu:/workspace/` 传输，云端 `git clone /workspace/MiniGen.bundle /workspace/MiniGen`，再 checkout 同一哈希。Git bundle 只携带已提交内容，不携带本地缓存、权重或用户未提交笔记。

验收：云端 HEAD 与最终交付哈希一致。代码变更会影响精确恢复；已开训后不要在同一实验中 `git pull` 或升级依赖。

## **三、Python、uv、PyTorch 与环境验证**

### **1. 创建云端独立环境**

目的：使用仓库的 `.python-version` 和 `uv.lock`，不复制 Mac 的 `.venv`。前置：云端已安装 Git、Python 安装所需系统组件、`curl`，以及供应商提供的 NVIDIA 驱动。普通训练使用 PyTorch wheel，不要求自行编译 CUDA 内核。

如果尚无 uv，按 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/) 安装；下面是官方独立安装器入口。网络代理按服务器实际条件配置，不在命令中公开访问令牌。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd "$MINIGEN_ROOT"
export UV_CACHE_DIR="$MINIGEN_ROOT/.cache/uv"
export HF_HOME="$MINIGEN_ROOT/.cache/huggingface"
uv python install 3.12.14
uv sync --locked --extra latent --extra conditioning --extra artbench --extra evaluation
uv run --locked --no-sync python -c 'import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
```

uv 按当前 Linux 平台安装锁定包，Mac wheel 不能迁移使用。若 Linux wheel 不可获得、`torch.version.cuda` 为 `None` 或 CUDA 不可用，停止并核查安装输出、GPU 驱动和 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。不要随手 `pip install -U torch` 后仍宣称使用原锁；需要变更 CUDA wheel 来源时，先记录新环境、重新解析兼容依赖并重跑本章检查。

### **2. 环境和基础代码检查**

```bash
mkdir -p artifacts/cloud/environment
uv run --locked --no-sync minigen doctor --device cuda --json > artifacts/cloud/environment/doctor_cuda.json
uv run --locked --no-sync pytest -p no:cacheprovider
uv run --locked --no-sync ruff check src scripts tests
uv run --locked --no-sync ruff format --check src scripts tests
uv run --locked --no-sync python scripts/check_code_annotations.py
uv run --locked --no-sync minigen bench --help
uv run --locked --no-sync python -c 'import torch; print({"gpu":torch.cuda.get_device_name(0),"bf16":torch.cuda.is_bf16_supported()})'
nvidia-smi > artifacts/cloud/environment/nvidia-smi.txt
uv pip freeze > artifacts/cloud/environment/packages.txt
git rev-parse HEAD > artifacts/cloud/environment/commit.txt
```

验收：doctor 的 CUDA 小张量前反向通过、测试通过；记录实际 Python/PyTorch/CUDA/GPU。测试不会证明真实 CUDA AMP、吞吐或生成质量，这些后面独立验收。BF16 不支持时可选 FP16 或 FP32，但须修改三模型共同配置并独立测试，不能静默换精度。

失败恢复：检查 GPU 是否分配给容器、驱动是否正常；`nvidia-smi` 中的 CUDA 版本是驱动能力信息，不等于 `torch.version.cuda`。先修环境，不能通过把 `--device cuda` 改成 CPU 来登记 CUDA 通过。

## **四、数据、冻结组件与正式缓存迁移**

### **1. 推荐路线：同步 Mac 上已验收的资产**

目的：原样复用 Caption、组划分、VAE mode 编码和 P2 正式缓存，避免重新编码引入新身份。前置：本机 P2 验收产物仍在、云端磁盘空间充足。正式缓存约 3.63 GB；处理后图像、模型、检查点和评测图片需另计。检查点包含模型、EMA、优化器等，明显大于单份推理权重；按计划快照数量预留空间。

**Mac：**

```bash
cd /Users/lucius/projects/personal/MiniGen
export MINIGEN_HOST=minigen-gpu
export MINIGEN_REMOTE_ROOT=/workspace/MiniGen
ssh "$MINIGEN_HOST" "mkdir -p '$MINIGEN_REMOTE_ROOT/data/processed/artbench_pd_256' '$MINIGEN_REMOTE_ROOT/artifacts/cache/artbench_pd_256' '$MINIGEN_REMOTE_ROOT/artifacts/models'"
rsync -av --partial --progress data/processed/artbench_pd_256/v1/ "$MINIGEN_HOST:$MINIGEN_REMOTE_ROOT/data/processed/artbench_pd_256/v1/"
rsync -av --partial --progress artifacts/cache/artbench_pd_256/v1_formal/ "$MINIGEN_HOST:$MINIGEN_REMOTE_ROOT/artifacts/cache/artbench_pd_256/v1_formal/"
rsync -av --partial --progress artifacts/models/sdxl_vae/ "$MINIGEN_HOST:$MINIGEN_REMOTE_ROOT/artifacts/models/sdxl_vae/"
rsync -av --partial --progress artifacts/models/clip_text/ "$MINIGEN_HOST:$MINIGEN_REMOTE_ROOT/artifacts/models/clip_text/"
rsync -av --partial --progress artifacts/models/eval/ "$MINIGEN_HOST:$MINIGEN_REMOTE_ROOT/artifacts/models/eval/"
```

末尾 `/` 表示同步目录内容。没有 `--delete`，不会因本机缺文件而删除远端结果。网络中断后重跑同一 rsync 命令；先校验再开始训练。只需要正式资产，旧几何数据、OpenClipart、QwenVL、历史教学权重均不作为云端训练前置。

### **2. 下载替代路线：固定 revision，不写回身份锁**

已成功同步的文件不必重下。ArtBench 原始分片可从固定版本获取：

```bash
uv run --locked --no-sync hf download alfredplpl/artbench-pd-256x256 \
  --repo-type dataset --revision 4056b70025f0e3f916fb271c7a792633ba637484 \
  --include 'data/*.parquet' \
  --local-dir data/raw/artbench_pd_256/4056b70025f0e3f916fb271c7a792633ba637484
uv run --locked --no-sync hf download madebyollin/sdxl-vae-fp16-fix \
  config.json diffusion_pytorch_model.safetensors \
  --revision 207b116dae70ace3637169f1ddd2434b91b3a8cd \
  --local-dir artifacts/models/sdxl_vae/207b116dae70ace3637169f1ddd2434b91b3a8cd
uv run --locked --no-sync python scripts/prepare_stage3_assets.py --text-only
uv run --locked --no-sync python - <<'PYVERIFY'
import hashlib
import json
from pathlib import Path
for lock_name in ("sdxl_vae", "clip_text"):
    lock = json.loads(Path(f"configs/models/{lock_name}.json").read_text())
    for name, record in lock["files"].items():
        path = Path(lock["local_path"]) / name
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"], path
        assert path.stat().st_size == record["bytes"], path
    print(lock_name, lock["revision"], "all files verified")
PYVERIFY
```

CLIP 命令从固定上游权重提取文本塔，并与 **已有** `configs/models/clip_text.json` 的所有哈希严格对照，失败则停止，不更新期望值。它不是完整评测 CLIP。不要用旧 `prepare_week3_assets.py` 代替上述 VAE 下载命令，该脚本还维护历史论文和锁文件。

若需要重新展开 ArtBench，可先写 **新的候选目录**：

```bash
uv run --locked --no-sync python scripts/prepare_artbench.py \
  --output-dir artifacts/cloud/artbench_rebuilt_candidate
sha256sum artifacts/cloud/artbench_rebuilt_candidate/manifest.jsonl
```

只有候选 Manifest SHA-256 等于冻结值 `0b95e7b783b2b62d6e2ff58646a95d173607a61b4184c6e250db7dfa1303cc0a`，且组、Caption、行顺序及所有图像哈希相同，才可作为冻结数据的恢复副本。优先同步已验收 `v1/`；不以“样本数相同”替代身份一致。不重建 P2 正式缓存，也不重新划分数据。

独立评测资产显式下载，普通 score 命令只离线加载：

```bash
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models inception clip
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models inception clip --verify-only
```

真实 VQA 待后续云端执行时再显式获取：

```bash
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models vqa
uv run --locked --no-sync python scripts/prepare_eval_assets.py --models vqa --verify-only
```

评测 CLIP 是完整 ViT-L/14@336，不能用生成条件 CLIP 文本塔代替。BLIP VQA 使用锁定 `Salesforce/blip-vqa-base`。下载文件清单、revision、SHA-256 和本地位置都在 `configs/models/evaluation.json`；不把网络下载成功当成模型推理通过。

LPIPS 优先同步 `artifacts/models/eval/lpips_alex/`；缺失时使用以下按既有 P2 记录下载并校验的命令，不修改锁：

```bash
uv run --locked --no-sync python - <<'PY'
import hashlib
import json
import urllib.request
from pathlib import Path

assets = json.loads(Path("reports/v2/components/p2_assets.json").read_text())["lpips_alex"]
for name in ("backbone", "calibration"):
    item = assets[name]
    path = Path(item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(item["source"], temporary)
        with temporary.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]
        temporary.replace(path)
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]
    print(name, path.stat().st_size, "verified")
PY
```

下载后以 CPU FP32 为基准做实际 CUDA 特征对照：

```bash
uv run --locked --no-sync python scripts/check_eval_backends.py \
  --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
  --output artifacts/cloud/acceptance/eval_cpu_cuda_fp32.json --device cuda
```

此项真实提取两张固定 **验证图** 的 Inception、CLIP 图像和文本特征，检查 CPU 批量一致性及 CUDA 对照，最大绝对误差阈值固定为 5e-4。逐项看 `device_passed`/`selected_device`，不是只看总体 complete；若某组件失败，显式采用已通过的 CPU，不能放宽阈值。评分入口一次接受一个设备，可将不同指标拆成独立 score 文件，记录实际设备。不计算正式测试集指标。

### **3. 云端独立回读正式缓存**

目的：验证迁移没有损坏，仍与被冻结数据及批准回执绑定。该命令只读缓存，不加载编码器、不重新编码，Linux 不需要 MPS。

```bash
mkdir -p artifacts/cloud/acceptance
uv run --locked --no-sync minigen cache \
  --output artifacts/cache/artbench_pd_256/v1_formal \
  --approval reports/v2/components/vae_user_approval.json \
  --evidence reports/v2/components/vae_reconstruction_200_summary.json \
  --local-acceptance reports/v2/components/artbench_local_fp32_20.json \
  --verify-only > artifacts/cloud/acceptance/cache_readback.json
uv run --locked --no-sync minigen bench prepare \
  --output reports/v2/benchmark/artbench_v1
```

验收：`verified_rows` 为 train 37,042、validation 4,637；latent `[4,32,32]`、FP16，text `[77,512]`、FP16，mask `[77]`、bool；顺序、样本 ID、源哈希和 payload SHA 一致。text 的 77 位置含 BOS/EOS，75 是内容 token 上限。缓存中的 latent 已经乘 VAE scaling factor 0.13025，训练不能再乘一次；解码适配层负责除回该系数。

`bench prepare` 在已冻结目录上核对内容，不重选题、不进行测试集特征评测。它同时核对 2,000 个正式参考文件的哈希。数据冻结锁和内容锁不相同时停止；不要手改锁使错误文件“通过”。

恢复：rsync 补齐损坏文件后重跑；保留失败日志。这里复用 MPS 编码的已批准缓存，不宣称云端 CUDA 编码与 MPS 逐位等价。

## **五、先完成 B0 CUDA 正确性、AMP、恢复与质量门槛**

### **1. FP32 CPU/CUDA 对照**

目的：验证同一模型、同一固定数据在 CUDA 上的前向和梯度。前置：缓存回读通过、已同步一个 P4 B0 工程检查点。历史 P4 检查点可用于评测，不能用来跨源码精确续训。Mac 可将 `artifacts/runs/p4_b0_local/last.pt` 同步到云端同一路径；若实际历史路径不同，以 `reports/v2/p4/b0_result.json` 的 checkpoint 为准。

**Mac：**只同步本次需要评测的三个工程检查点，不同步全部历史训练目录。

```bash
cd /Users/lucius/projects/personal/MiniGen
for model in b0 b1 b2; do
  ssh minigen-gpu "mkdir -p /workspace/MiniGen/artifacts/runs/p4_${model}_local"
  rsync -av --partial --progress "artifacts/runs/p4_${model}_local/last.pt" \
    "minigen-gpu:/workspace/MiniGen/artifacts/runs/p4_${model}_local/last.pt"
 done
```

**云端：**

```bash
uv run --locked --no-sync python scripts/check_flow_device.py \
  --config configs/training/p4_b0_resume.json \
  --checkpoint artifacts/runs/p4_b0_local/last.pt \
  --device cuda --output artifacts/cloud/acceptance/b0_cpu_cuda_fp32.json
```

验收：输出 status=passed，固定 `atol=rtol=2e-4` 不随结果放宽；检查 missing gradient、输出与梯度最大误差。此项不等于 AMP 通过。失败时先查软件版本、SDPA 和 GPU 数值设置，保留差值，不增加容差掩盖问题。

### **2. 100 更新的 AMP 和资源 Pilot**

目的：实测单卡微批、梯度累积、AMP、优化器和 EMA；建立费用估算。前置：FP32 对照通过，确认 BF16 支持。三份现有 Pilot 配置为 microbatch=8、grad_accum=8、有效 batch=64，固定 256 条训练表示。可以因显存调整 microbatch/累积，但应保持有效 batch，并重新测量所有架构。

用户填写真实时价，以下不提供假定报价：

```bash
export MINIGEN_GPU_HOURLY=填入每小时人民币价格
uv run --locked --no-sync python scripts/profile_training.py \
  --config configs/training/p4_b0_cuda_pilot.json \
  --output artifacts/runs/cloud_b0_pilot01 \
  --warmup 50 --measure 50 --precision bf16 \
  --hourly-cny "$MINIGEN_GPU_HOURLY"
```

验收：`pilot.json` status=measured，确实完成 50 步预热和 50 步计时；loss/梯度有限、训练结果 device=cuda、检查点可加载；记录 GPU 型号、精度、吞吐、step seconds、CUDA 峰值 allocated/reserved。成本只是优化步估算，保存、评测、下载、磁盘及重试另计。

产物：`artifacts/runs/cloud_b0_pilot01/{pilot.json,run.json,metrics.jsonl,result.json,last.pt}`。OOM 时复制 Pilot 配置到 `artifacts/cloud/configs/`，减少 batch、增加累积并使用新输出目录重跑；不能拿旧时延为新配置估价。BF16/FP16 数值异常时保留 `failure.json`，先用 FP32 同数据对照，不能直接推进正式训练。

### **3. 独立验证恢复，固定学习率日程**

先用独立短实验做 **连续 8 步** 与 **4+4 步恢复**，不要在已完成的 Pilot 里改日程。以下只生成三份小配置：

```bash
uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path

base = json.loads(Path("configs/training/p4_b0_cuda_pilot.json").read_text())
base["training"].update(batch_size=2, grad_accum=1, precision="bf16",
                        max_seconds=300, schedule_steps=20, warmup_steps=2,
                        checkpoint_every=4, checkpoint_steps=[4, 8], validation_every=100)
base["data"].update(train_limit=8, validation_limit=4)
folder = Path("artifacts/cloud/configs")
folder.mkdir(parents=True, exist_ok=True)
for name, steps, run in (("continuous", 8, "continuous"), ("part1", 4, "split"),
                          ("part2", 8, "split")):
    item = json.loads(json.dumps(base))
    item["training"].update(max_steps=steps, output_dir=f"artifacts/runs/cloud_resume_{run}")
    path = folder / f"resume_{name}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(item, indent=2) + "\n")
PY
uv run --locked --no-sync minigen train --config artifacts/cloud/configs/resume_continuous.json --device cuda
uv run --locked --no-sync minigen train --config artifacts/cloud/configs/resume_part1.json --device cuda
uv run --locked --no-sync minigen train --config artifacts/cloud/configs/resume_part2.json \
  --resume artifacts/runs/cloud_resume_split/last.pt --device cuda
```

实际对照并保存回执：

```bash
uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path
import torch
from minigen.training.checkpoint import load_checkpoint

left = load_checkpoint(Path("artifacts/runs/cloud_resume_continuous/last.pt"))
right = load_checkpoint(Path("artifacts/runs/cloud_resume_split/last.pt"))
assert left["step"] == right["step"] == 8
report = {"step": 8, "device": "cuda", "model": {}, "ema": {}}
for section in ("model", "ema"):
    errors = {key: float((value - right[section][key]).abs().max())
              for key, value in left[section].items() if value.is_floating_point()}
    report[section] = {"max_abs": max(errors.values()),
                       "bitwise_equal": all(torch.equal(v, right[section][k])
                                            for k, v in left[section].items())}
report["schedule_equal"] = left["scheduler"] == right["scheduler"]
Path("artifacts/cloud/acceptance/cuda_resume_comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
```

验收：两个最终 step=8，数据 cursor/排列、优化器、EMA、scaler、RNG、日程均保存且恢复无身份错误；模型与 EMA 的差值须实际检查。上述回执只比较参数和日程，恢复链路的优化器/随机状态覆盖另由完整工程测试支持；不要把该小脚本称为所有状态逐项比较。相同源码、配方、设备条件下应保持受控轨迹。CUDA 算子若存在非确定性，应记录实际差值和算子限制，不能写成跨设备逐位恢复。基础 CPU 精确恢复测试不代替此次 CUDA 实测。

### **4. 受控单图和 16 图质量诊断**

目的：解决“loss 下降但采样不可辨”的未完成门槛。前置：数值和恢复已通过；这是小样本诊断，不是正式训练。复用 P3 已冻结索引和配方，显式生成新 CUDA 目录：

```bash
uv run --locked --no-sync python - <<'PY'
import json
from pathlib import Path

folder = Path("artifacts/cloud/configs")
folder.mkdir(parents=True, exist_ok=True)
for name in ("single", "sixteen"):
    item = json.loads(Path(f"configs/training/p3_close_{name}.json").read_text())
    item["training"].update(device="cuda", precision="fp32", max_seconds=600,
                            output_dir=f"artifacts/runs/cloud_quality_{name}")
    target = folder / f"quality_{name}.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(item, indent=2) + "\n")
PY
uv run --locked --no-sync minigen train --config artifacts/cloud/configs/quality_single.json --device cuda
uv run --locked --no-sync python scripts/diagnose_flow_quality.py \
  --config artifacts/cloud/configs/quality_single.json \
  --checkpoint artifacts/runs/cloud_quality_single/last.pt \
  --output artifacts/evaluations/cloud_quality_single --device cuda
```

先看单图的 `reference.png`、online/EMA、Euler40/Heun20、Euler200 与 CFG3 对照；oracle 端点误差是积分检查，不是学习成功。单图通过后才运行：

```bash
uv run --locked --no-sync minigen train --config artifacts/cloud/configs/quality_sixteen.json --device cuda
uv run --locked --no-sync python scripts/diagnose_flow_quality.py \
  --config artifacts/cloud/configs/quality_sixteen.json \
  --checkpoint artifacts/runs/cloud_quality_sixteen/last.pt \
  --output artifacts/evaluations/cloud_quality_sixteen --device cuda
for mode in correct blank shuffled; do
  uv run --locked --no-sync minigen sample \
    --config artifacts/cloud/configs/quality_sixteen.json \
    --checkpoint artifacts/runs/cloud_quality_sixteen/last.pt \
    --output "artifacts/evaluations/cloud_condition_$mode" --device cuda \
    --condition-mode "$mode" --count 8 --seed 1234 --steps 40 --guidance 1
 done
```

验收：按原 Plan 和 P3 报告判断固定 16 图中主体可辨及条件作用，保留全部固定图，不只选好看的。正确、空白和打乱条件共用初噪；单次差异不等于统计结论。若仍不可辨，停止长训，记录 time-slice 速度误差、EMA/online、solver 差异和训练预算，查明后再确定下一次有限实验；不能提高步数直到挑出一张成功图便宣称通过。

## **六、三模型正式配置、训练与恢复**

### **1. 各自完成 Pilot，统一冻结训练协议**

B0 质量门槛通过后，对 B1/B2 重复 FP32 数值检查、AMP/恢复检查，并运行各自 Pilot：

```bash
uv run --locked --no-sync python scripts/profile_training.py \
  --config configs/training/p4_b1_cuda_pilot.json --output artifacts/runs/cloud_b1_pilot01 \
  --warmup 50 --measure 50 --precision bf16 --hourly-cny "$MINIGEN_GPU_HOURLY"
uv run --locked --no-sync python scripts/profile_training.py \
  --config configs/training/p4_b2_cuda_pilot.json --output artifacts/runs/cloud_b2_pilot01 \
  --warmup 50 --measure 50 --precision bf16 --hourly-cny "$MINIGEN_GPU_HOURLY"
```

B1/B2 复验使用对应 `p4_b1_resume.json`/`p4_b2_resume.json` 和 checkpoint；不能把 B0 的数值通过复制给另两个模型。其训练、采样和恢复接口一致，差异在模型构造配置。

正式运行前记录：Git 提交、三份模型 JSON、真实参数量、数据/组件哈希、精度、有效 batch、学习率与日程、AdamW、条件丢弃、EMA、训练种子、统一更新步数、检查点间隔、验证指标、费用上限。相同参数量不表示相同 FLOPs；等步数、等 GPU 时间和估算 FLOPs 对照分别报告。

### **2. 用真实 Pilot 生成正式配置**

目的：将已实测可运行的配方转换为全训练集配置。前置：对应真实 CUDA `pilot.json` 与相邻 `run.json` 身份匹配；用户根据 Pilot 填写步数、时限和预算，不能用本地 12 步配置当正式方案。

```bash
export MINIGEN_STEPS=填入三模型共同更新步数
export MINIGEN_SECONDS=填入每次允许的秒数
export MINIGEN_BUDGET=填入每模型训练预算人民币
mkdir -p artifacts/cloud/configs
for model in b0 b1 b2; do
  uv run --locked --no-sync python scripts/prepare_cloud_run.py \
    --architecture "$model" --pilot "artifacts/runs/cloud_${model}_pilot01/pilot.json" \
    --output "artifacts/cloud/configs/${model}_formal.json" \
    --run-dir "artifacts/runs/cloud_${model}_formal_s0" \
    --steps "$MINIGEN_STEPS" --max-seconds "$MINIGEN_SECONDS" \
    --hourly-cny "$MINIGEN_GPU_HOURLY" --budget-cny "$MINIGEN_BUDGET"
 done
```

产物：三份正式 JSON 和 `.budget.json`。生成器使用全训练集、256 条固定配对验证样本，保留约 20 个预定 step 快照，按实测时延增加 25% 运行预留；这仍不是收费账单，也不包含全部存储/评测费用。输入估算超出时限或预算会拒绝生成，不能通过填假时价绕过。

更改超参数的方法：在新配置中修改 `training.batch_size`、`grad_accum`、`learning_rate`、`precision`、`max_steps`、`max_seconds`、`warmup_steps`、`schedule_steps`、`checkpoint_steps` 等明确键；模型路径由 `model` 指定。改变会影响时延或显存的参数后重做对应 Pilot。**microbatch × grad_accum 是单卡有效 batch；max_steps 是 optimizer 更新数。** 三模型必须遵守同一预先确定的比较方案。

配置解析检查，不启动训练：

```bash
uv run --locked --no-sync python - <<'PY'
from pathlib import Path
from minigen.commands import read_experiment
for architecture in ("b0", "b1", "b2"):
    config, data = read_experiment(Path.cwd(), Path(f"artifacts/cloud/configs/{architecture}_formal.json"))
    print(architecture, config.model, config.batch_size * config.grad_accum,
          config.max_steps, config.precision, data["train_limit"], config.checkpoint_steps)
PY
```

### **3. 启动、查看日志、保存和继续训练**

前置：人工确认质量和预算门槛，不并行挤在同一张 GPU。使用服务器已经安装的 tmux 保持 SSH 断线后进程继续运行；缺少时先由供应商环境方式安装。打开会话：

```bash
tmux new -s minigen
cd "$MINIGEN_ROOT"
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync minigen train \
  --config artifacts/cloud/configs/b0_formal.json --device cuda
```

`Ctrl+B` 后按 `D` 离开 tmux，不是终止训练；`tmux attach -t minigen` 回到会话。B0 完成并检查结果后，分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync minigen train --config artifacts/cloud/configs/b1_formal.json --device cuda
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync minigen train --config artifacts/cloud/configs/b2_formal.json --device cuda
```

另一个 SSH 窗口查看：

```bash
tail -n 10 artifacts/runs/cloud_b0_formal_s0/metrics.jsonl
nvidia-smi
uv run --locked --no-sync python scripts/plot_flow_run.py \
  --run artifacts/runs/cloud_b0_formal_s0 \
  --output artifacts/evaluations/cloud_b0_loss.png
```

产物包括 `run.json`（完整身份/配置）、`metrics.jsonl`、`result.json`、`last.pt`、`step_000100.pt` 形式的预定快照（编号按实际步数）。指标包含 loss、梯度、学习率、步耗时、数据等待和固定配对验证。`plot_flow_run.py` 需要已经写出 result.json；首次训练尚未结束时先看 metrics.jsonl，不将缺少结束回执当作训练失败。`result.json` 是一次调用的结束回执；存在旧成功结果不意味着最近一次失败已经修复，需同时看 `failure.json`、attempt_id 与时间。

训练器没有远程暂停按钮。预先用 `max_seconds` 安排受控分段，退出前保存最后状态；强制断电或 `kill -9` 只能回到已发布的 checkpoint。`Ctrl+C` 不承诺保存当前未完成更新，所以先确认 `last.pt` 的更新时间。恢复命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-sync minigen train \
  --config artifacts/cloud/configs/b0_formal.json \
  --resume artifacts/runs/cloud_b0_formal_s0/last.pt --device cuda
```

保持同一源码、模型/数据/配置身份、学习率日程和随机流。`max_steps` 是恢复后总目标，不是再加若干步；只延长停止上限不应更改固定 `schedule_steps`。迁移机器导致设备/算子变化时，不宣称逐位精确恢复。身份冲突时保留原文件，读取报错定位；不能删除校验代码或修改 checkpoint 内的哈希。

独立采样和固定配对 MSE：

```bash
uv run --locked --no-sync minigen sample --config artifacts/cloud/configs/b0_formal.json \
  --checkpoint artifacts/runs/cloud_b0_formal_s0/last.pt \
  --output artifacts/evaluations/cloud_b0_preview --device cuda \
  --split validation --count 8 --seed 1234 --steps 40 --method euler --guidance 1
uv run --locked --no-sync minigen evaluate --config artifacts/cloud/configs/b0_formal.json \
  --checkpoint artifacts/runs/cloud_b0_formal_s0/last.pt \
  --output artifacts/evaluations/cloud_b0_flow_mse.json --device cuda --count 256
```

`evaluate` 仍然只计算固定配对 Flow MSE，不是图像质量指标。图像质量使用下一章 `bench`。

## **七、先在验证集选择采样配置，再运行正式 Benchmark**

### **1. 人工审核候选 Benchmark 并保存批准回执**

前置：阅读 [Benchmark 说明](../../reports/v2/benchmark/README.md) 与实际 `benchmark.json`。确认 80 ID、40 composition_stress、40 paired paraphrase、80 独立 validation，主测试种子 20261110/20261111，验证种子 20261112/20261113。组合压力题不具有已证明的 OOD 身份；训练 Caption 检索只支持概念词出现，不证明视觉组合绝未见过。不要把自动内容哈希当成人工批准。

只有用户实际审核完毕，才手工创建批准回执，例如 `artifacts/cloud/benchmark_approval.json`：

```json
{
  "benchmark_sha256": "填入 reports/v2/benchmark/artbench_v1/lock.json 中的内容哈希",
  "approved": true,
  "reviewer": "填写实际审核人",
  "reviewed_at": "填写实际审核时间和时区"
}
```

字段 `benchmark_sha256` 是 lock 的 **规范化内容哈希**，不是直接 `sha256sum benchmark.json` 的文件字节哈希。此回执是审批记录，不是数字签名。没有审核时不创建 approved=true。题目、规则或种子变更后需新版本、新审核，不编辑旧锁绕过。

### **2. 验证集单变量扫描与性能测量**

目的：用独立 validation prompts 选择设置，主测试不参与选择。前置：固定同一 checkpoint/设备/精度/batch/初噪。`bench profile` 测时而不保存完整质量图，质量对照须另执行 generate+score。

```bash
uv run --locked --no-sync minigen bench profile \
  --config artifacts/cloud/configs/b0_formal.json \
  --checkpoint artifacts/runs/cloud_b0_formal_s0/last.pt \
  --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
  --output artifacts/evaluations/cloud_b0_profile.json \
  --device cuda --count 4 --warmup 2 --repeats 5 \
  --matrix configs/benchmark/sampling_matrix.json
```

矩阵：Euler NFE10/20/40，Heun 分别 5/10/20 steps 对齐 NFE10/20/40；再固定 Euler NFE40 扫 CFG2/3；同一设置补 naive/SDPA。`bench` 中 `--solver-nfe` 是速度场求值预算；普通 `sample --steps` 是积分步数。CFG 非 1 时额外无条件分支单独计入 model evaluations，不能说相同 solver NFE 即相同总计算。

以下完整验证题质量扫描每个设置 80×2=160 张。先选少量 `--count 4` 排错，正式选型运行时移除 `--count`：

```bash
for setting in euler-10-1 euler-20-1 euler-40-1 heun-10-1 heun-20-1 heun-40-1 euler-40-2 euler-40-3; do
  IFS=- read -r method nfe cfg <<< "$setting"
  uv run --locked --no-sync minigen bench generate \
    --config artifacts/cloud/configs/b0_formal.json \
    --checkpoint artifacts/runs/cloud_b0_formal_s0/last.pt \
    --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
    --output "artifacts/evaluations/cloud_validation/$setting" --split validation \
    --device cuda --batch-size 4 --method "$method" --solver-nfe "$nfe" --guidance "$cfg" \
    --scope real_local --max-seconds 1800
  uv run --locked --no-sync minigen bench score \
    --generation "artifacts/evaluations/cloud_validation/$setting/samples.json" \
    --output "artifacts/evaluations/cloud_validation/$setting/score.json" \
    --metrics clipscore lpips --device cuda --scope real_local
 done
```

`real_local` 是工程/探索范围标签，**不意味着只能在本机执行**；真实设备另外记录。通过盲评和后续已校准 VQA 判断条件遵循，不能单凭 CLIPScore 定型。保存选择结果和理由，然后冻结所有模型共同使用的设置。若按架构分别选设置，必须事先声明“每模型同样验证预算”，不能事后给偏好模型更多搜索。

### **3. 正式分布评测与 Prompt 评测分开生成**

目的：真实分布指标使用 2,000 条测试 Caption 各一张；结构化条件评测使用 160 题各两个种子。前置：Benchmark 人工批准、采样设置在 validation 上确定、模型/checkpoint 比较点已冻结。下面 Euler40/CFG1 **只演示命令形式**，真正运行时一致替换为已冻结选择。

```bash
for model in b0 b1 b2; do
  uv run --locked --no-sync minigen bench generate \
    --config "artifacts/cloud/configs/${model}_formal.json" \
    --checkpoint "artifacts/runs/cloud_${model}_formal_s0/last.pt" \
    --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
    --output "artifacts/evaluations/formal/${model}_distribution" \
    --split distribution --device cuda --batch-size 4 --method euler --solver-nfe 40 \
    --guidance 1 --scope formal --approval artifacts/cloud/benchmark_approval.json \
    --max-seconds 1800
  uv run --locked --no-sync minigen bench generate \
    --config "artifacts/cloud/configs/${model}_formal.json" \
    --checkpoint "artifacts/runs/cloud_${model}_formal_s0/last.pt" \
    --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
    --output "artifacts/evaluations/formal/${model}_prompts" \
    --split test --device cuda --batch-size 4 --method euler --solver-nfe 40 \
    --guidance 1 --scope formal --approval artifacts/cloud/benchmark_approval.json \
    --max-seconds 1800
 done
```

时限是每次调用上限，达到后保留 pending 项，按原命令附 `--resume` 继续；已失败项需附 `--retry-failed`。输出中的 `samples.json` 每项含 caption、seed、要求、图片哈希、完成/失败状态。每个 `(prompt_id,seed)` 独立构造 CPU 初噪，批次划分和重试不改初噪；跨设备结果不保证逐位一致。损坏图片不会静默当完成，身份不一致会拒绝恢复。

验收：distribution 预期 2,000 张、prompts 预期 320 张；全任务都保留，包括失败和缺图。**formal 模式禁止用 `--count` 缩小后冒充正式完整结果**。分批推理实际显存以当前GPU测量；显存不足先降低 batch 并保留变更说明，身份门槛拒绝原目录时使用新目录，不修改旧回执。

### **4. 分别执行图像指标、真实 VQA 和双种子多样性**

```bash
for model in b0 b1 b2; do
  uv run --locked --no-sync minigen bench score \
    --generation "artifacts/evaluations/formal/${model}_distribution/samples.json" \
    --reference reports/v2/benchmark/artbench_v1/benchmark.json \
    --output "artifacts/evaluations/formal/${model}_distribution/score.json" \
    --metrics kid cmmd clipscore --device cuda --batch-size 4 --scope formal
  uv run --locked --no-sync minigen bench score \
    --generation "artifacts/evaluations/formal/${model}_prompts/samples.json" \
    --reference reports/v2/benchmark/artbench_v1/benchmark.json \
    --output "artifacts/evaluations/formal/${model}_prompts/score.json" \
    --metrics clipscore vqa lpips --vqa-backend blip \
    --device cuda --batch-size 4 --scope formal
 done
```

验收与解释：KID 正式 100 次、每次 1,000 子样本，无偏估计可为负；CMMD 为归一化完整 CLIP 图像投影、RBF 带宽 10、biased MMD ×1000；CLIPScore 使用 L/14@336、无文字前缀、2.5×max(cosine,0)，不是原论文另一 CLIP 变体的同一数值口径。LPIPS 只是双种子差异，不能独立证明质量好或文本遵循。

VQA 必须 `--vqa-backend blip` 并实际加载锁定资产；Mock 只允许显式 simulated 工程测试，不能进入正式成绩。保存原始回答、规范化答案、correct/wrong/indeterminate/error、适用分母、覆盖率和保守成功率；概率不是已校准置信度。真实 VQA 在人工校准前只是自动判断，不当作可靠真值。

评分逐项失败保留错误和覆盖率；`--resume` 在身份相同且图片校验通过时直接复用已完整的评分回执；部分失败会重试整个请求列表，仍可复用身份匹配的图像特征缓存。检查 JSON 的 `complete`、metric status 和 coverage，不能只看进程退出码。缺图时可计算子集的结果标为 incomplete，不作为全量正式结果。

## **八、人工校准和匿名盲评**

### **1. 先验证工具，再完成真实校准**

已交付的 8 张验证图+8 张退化图只是页面与数据流验收，位于 `artifacts/evaluations/p5/calibration/review/review.html`。未收到人工标注，没有一致率；仅 caption claim 的页面不能覆盖对象/数量/颜色/绑定/关系五类 VQA 校准。

正式校准需要 100 张真实验证图以及 100 张生成或退化图，另行提供可信结构化 requirements。上游 Caption 不是视觉真值，必须人工核对后才能作为已知属性。不要用 Test reference 做校准调参。准备通用 100+100 待审材料的入口：

```bash
uv run --locked --no-sync minigen bench review calibrate \
  --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
  --output artifacts/evaluations/cloud_calibration_100 --count 100
```

这条默认命令只产生 caption claim 任务及退化谱系，**不自动完成逐类 VQA 校准**。先核对 `benchmark.json` 的 `calibration_references` 前 100 项，人工看原图，给每个 sample_id 填写实际可见事实，保存 `artifacts/cloud/calibration_annotations.json`。格式如下；内容仅是说明格式的例子，不能原样充当这 100 张的真值：

```json
{
  "source": "human_annotations",
  "samples": [
    {
      "sample_id": "替换为真实验证图ID",
      "requirements": [
        {"kind": "object", "subject": "horse", "expected": "yes"},
        {"kind": "count", "subject": "horse", "expected": "2"},
        {"kind": "color", "subject": "horse", "expected": "brown"},
        {"kind": "binding", "subject": "horse", "expected": "brown"},
        {"kind": "relation", "subject": "horse", "object": "tree", "expected": "left"}
      ]
    }
  ]
}
```

每图仅写实际适用且人工可判断的属性，不强行给没有主体的抽象画添加计数/空间关系。为覆盖五类问题，需要在冻结校准样本中记录真实适用数量；若某类别样本不足，如实报告不足，不根据模型表现改抽样。

带人工标注创建 **新的目录**，显式保留标注文件哈希，并生成可直接评分的 `samples.json`：

```bash
uv run --locked --no-sync minigen bench review calibrate \
  --benchmark reports/v2/benchmark/artbench_v1/benchmark.json \
  --annotations artifacts/cloud/calibration_annotations.json \
  --output artifacts/evaluations/cloud_calibration_annotated --count 100
uv run --locked --no-sync minigen bench score \
  --generation artifacts/evaluations/cloud_calibration_annotated/samples.json \
  --output artifacts/evaluations/cloud_calibration_annotated/vqa_score.json \
  --metrics vqa --vqa-backend blip --device cuda --batch-size 4 --scope real_local
```

将 `cloud_calibration_annotated/review/review.html` 复制给真实审核者。原图与退化图分别判断 claim，不预设退化一定答错；模糊后无法判断用 undecidable。完成评分并导出 JSON 后：

```bash
uv run --locked --no-sync minigen bench review import \
  --package artifacts/evaluations/cloud_calibration_annotated/review \
  --answers artifacts/evaluations/cloud_calibration_annotated/user_answers.json \
  --output artifacts/evaluations/cloud_calibration_annotated/human_result.json
uv run --locked --no-sync minigen bench review compare \
  --human artifacts/evaluations/cloud_calibration_annotated/human_result.json \
  --automatic artifacts/evaluations/cloud_calibration_annotated/vqa_score.json \
  --output artifacts/evaluations/cloud_calibration_annotated/agreement.json
```

验收：200 张都参与、每条已批准要求有匹配 question_id，真实模型原始回答和人工 verdict 独立保存，按 kind 输出混淆/未知/执行失败/覆盖率/样本量。没有满足实际样本量或人审不完整时，不输出“正式校准已通过”。这部分不是从 Caption 自动制造真值。

### **2. 跨模型匿名盲评**

目的：让审核者只看图、提示词和评分项，不知道模型架构和文件名。前置：三模型使用同一固定 Prompt/seed，缺失图片也进入失败任务。导出只将公开任务和重新编码的图片嵌入 HTML，私有映射单独存放：

```bash
uv run --locked --no-sync minigen bench review export \
  --generations artifacts/evaluations/formal/b0_prompts/samples.json \
                artifacts/evaluations/formal/b1_prompts/samples.json \
                artifacts/evaluations/formal/b2_prompts/samples.json \
  --output artifacts/evaluations/cloud_blind_review --seed 20261110
```

将 `review.html` 复制回 Mac 后直接浏览器打开即可，不需要服务、网络或在线模型。只把 HTML 发给审核者，不把同目录 `private_mapping.json` 一起发出。页面支持放大图像、质量/遵循 1–5 分、错误类型、多项 claim 判断、localStorage 保存以及 JSON 导入/导出。真实评分后填写 reviewer 和 reviewed_at 再导出答案。

```bash
uv run --locked --no-sync minigen bench review import \
  --package artifacts/evaluations/cloud_blind_review \
  --answers artifacts/evaluations/cloud_blind_review/user_answers.json \
  --output artifacts/evaluations/cloud_blind_review/human_result.json
```

验收：package ID 匹配、题号完整、分数/标签合法、没有漏答；partial 不是 complete。浏览器存储可能被清理，应及时导出 JSON。没有人工答案时只交待审包，不写一致率、不将自动指标替代人工。

跨模型盲评提供质量与遵循判断；逐类自动—人工一致率使用上一小节的独立校准流程，不能用没有结构化真值的默认 caption claim 页面代替。

## **九、汇总、成本对照与中间检查点**

### **1. 基于真实回执生成结果表与画廊**

```bash
uv run --locked --no-sync minigen bench report \
  --scores artifacts/evaluations/formal/b0_distribution/score.json \
           artifacts/evaluations/formal/b1_distribution/score.json \
           artifacts/evaluations/formal/b2_distribution/score.json \
           artifacts/evaluations/formal/b0_prompts/score.json \
           artifacts/evaluations/formal/b1_prompts/score.json \
           artifacts/evaluations/formal/b2_prompts/score.json \
  --training artifacts/runs/cloud_b0_formal_s0/result.json \
             artifacts/runs/cloud_b1_formal_s0/result.json \
             artifacts/runs/cloud_b2_formal_s0/result.json \
  --profiles artifacts/evaluations/cloud_b0_profile.json \
  --output artifacts/evaluations/cloud_final
```

产物包括 JSON/CSV、曲线、画廊和 Markdown；按 actual scope 分开工程、模拟和正式结果。没有的数据留空，不从 MPS smoke 外推 CUDA hours，不从无账单推算实际费用。正式失败也保留，不能挑选所有成功条目组成“完整”结果。

### **2. 等更新步数和等预算是两种问题**

在预定相同 step 的真实快照上分别运行 generate+score，才有质量—训练步数曲线。使用 `last.pt` 只能代表当前实际 step，不能给它换个文件名充当早期 checkpoint。快照路径以训练输出为准；每个 checkpoint 绑定自己的 step、哈希和累积训练证据。

先从实际快照导出该步的资源回执，避免把最终总耗时误配给早期模型。以下 `000100` 仅表示文件名格式，运行时替换为配置中 **真实存在的预定快照编号**：

```bash
uv run --locked --no-sync python scripts/export_checkpoint_receipt.py \
  --checkpoint artifacts/runs/cloud_b0_formal_s0/step_000100.pt \
  --output artifacts/evaluations/formal/b0_step000100_training.json \
  --flops-inspection reports/v2/p4/model_specs.json
```

随后以同一 `step_000100.pt` 执行上一章生成/评分，将评分 JSON 与这份 step 专属 JSON 一起交给 `bench report --scores ... --training ...`。每个预定检查点重复此过程，report 可读同一 run_id 的多个真实 step。`export_checkpoint_receipt.py` 在 CPU 读取权重和历史、不启动训练；可选 FLOPs 估算严格核对模型与输入形状，使用 3×主要前向矩阵 FLOPs×历史实际处理样本数，并明确排除非矩阵算子、优化器、EMA、采样和冻结编码器。旧检查点缺少 GPU 型号时该值保留空，不补猜。硬件型号记录的是保存时的设备；同预算实验应从头到尾使用同一 GPU 型号，跨硬件迁移需新建运行并分开登记，不把混合设备历史算作同 GPU 预算。

预算比较使用实际已存在的检查点，下面演示以一个已冻结的目标 GPU hours 进行比较：

```bash
export MINIGEN_TARGET_GPU_HOURS=填入预先确定的目标小时数
uv run --locked --no-sync minigen bench report \
  --scores artifacts/evaluations/formal/b0_distribution/score.json \
           artifacts/evaluations/formal/b1_distribution/score.json \
           artifacts/evaluations/formal/b2_distribution/score.json \
  --training artifacts/runs/cloud_b0_formal_s0/result.json \
             artifacts/runs/cloud_b1_formal_s0/result.json \
             artifacts/runs/cloud_b2_formal_s0/result.json \
  --comparison-mode gpu_hours --target "$MINIGEN_TARGET_GPU_HOURS" --tolerance 0.1 \
  --output artifacts/evaluations/cloud_budget_comparison
```

`steps`、`gpu_hours`、`training_flops` 是可选比较维度；上例只允许已有预算落在目标的 90%～100%，不超过目标，必须在观察最终成绩前决定。缺失匹配资源记录、scope 不同、非完整结果或 GPU 型号未知时不能形成合法比较。没有命中预算的快照，就报告不具备该对照，不插值生成不存在的模型得分。

训练 FLOPs 若使用“主干前向 FLOPs × 近似反向倍数 × 已处理样本”需记录估算方法、未计项和实际步数，不能声称硬件实测；先具备显式估计字段再执行该维度比较。训练时长和真实收费分别记录，不把运行秒数乘报价称为账单。

## **十、保存、回迁与释放云实例**

### **1. 统一目录与备份优先级**

| 内容 | 路径 | 备份和 Git 规则 |
| --- | --- | --- |
| 源码、协议、模型规格 | `src/`、`configs/`、`reports/v2/benchmark/` | Git 固定提交；不混入真实权重 |
| 原始/处理后数据 | `data/raw/`、`data/processed/artbench_pd_256/v1/` | 可按锁恢复，但 Caption/Manifest 精确版本必须保存 |
| 正式缓存 | `artifacts/cache/artbench_pd_256/v1_formal/` | 保留哈希与验收回执，备份已有批准版本 |
| 冻结模型 | `artifacts/models/` | 锁定 revision/hash，Git 忽略 |
| 训练实验 | `artifacts/runs/cloud_<model>_formal_s0/` | 优先备份 last.pt、预定快照、run/result/metrics/failure |
| 生成与评测 | `artifacts/evaluations/formal/` | 图片、samples.json、评分、特征身份与失败齐全 |
| 云端环境和配置 | `artifacts/cloud/` | 环境、预算回执、批准、解析后的完整配置需回迁 |

恢复训练前确认还有足够磁盘写临时 checkpoint 和原子替换后的正式文件。磁盘满时先暂停，核对已完成离机备份后清理可重建特征缓存；不能删除唯一 optimizer/EMA checkpoint。不要在训练写入时移动正在使用的 last.pt。

### **2. 回迁 Mac 并独立校验**

云端退出训练或等待当前保存完成后，记录哈希：

```bash
find artifacts/runs/cloud_b0_formal_s0 artifacts/runs/cloud_b1_formal_s0 artifacts/runs/cloud_b2_formal_s0 \
  -type f -print0 | sort -z | xargs -0 sha256sum > artifacts/cloud/training_backup.sha256
```

**Mac：**

```bash
cd /Users/lucius/projects/personal/MiniGen
mkdir -p artifacts/cloud_return
rsync -av --partial --progress minigen-gpu:/workspace/MiniGen/artifacts/cloud/ artifacts/cloud_return/cloud/
rsync -av --partial --progress minigen-gpu:/workspace/MiniGen/artifacts/runs/ artifacts/cloud_return/runs/
rsync -av --partial --progress minigen-gpu:/workspace/MiniGen/artifacts/evaluations/ artifacts/cloud_return/evaluations/
```

保存在 `cloud_return/` 避免覆盖历史本地实验。哈希列表中的相对路径以云端仓库为根，校验时把 `artifacts/runs/` 映射到 `artifacts/cloud_return/runs/`：

```bash
uv run --locked --no-sync python - <<'PY'
import hashlib
from pathlib import Path

receipt = Path("artifacts/cloud_return/cloud/training_backup.sha256")
checked = 0
for line in receipt.read_text().splitlines():
    digest, relative = line.split("  ", 1)
    original = Path(relative)
    path = Path("artifacts/cloud_return/runs") / original.relative_to("artifacts/runs")
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    assert actual == digest, path
    checked += 1
print({"verified_files": checked})
PY
```

不要在错误目录直接校验后忽略报错。对关键 last.pt、快照、run.json 和评分文件逐项比对 SHA-256，并从 Mac 离线加载至少一份 checkpoint、打开一份 HTML 和图片抽查。

释放实例前确认：所有进程已停止；三模型需要的 checkpoint/optimizer/EMA、环境和配置、所有日志、评分、审核 JSON、失败回执均回迁；费用与磁盘账单已记录；云磁盘是否保留已明确。关闭终端不等于关停计费实例。用户在供应商控制台自行停止/释放，不能先删持久化磁盘再寻找唯一模型。

## **十一、多卡状态与常见错误**

### **1. 当前仅支持单卡，不提供虚构的 torchrun 命令**

DDP 尚未实现。`CUDA_VISIBLE_DEVICES=0` 只选择单卡，增大可见 GPU 数不会自动并行。未来扩展需要 rank/world-size 初始化、分布式数据排列和重复控制、梯度同步与 no_sync、全局有效 batch、跨 rank RNG/Checkpoint、rank0 日志、验证汇总与实际多卡恢复测试。扩展后有效 batch 为 microbatch×accum×world_size；学习率与日程是否变化需要新配方和公平对照，不沿用单卡结果冒充多卡吞吐。

### **2. 故障定位表**

| 现象 | 先查什么 | 恢复方法与禁止事项 |
| --- | --- | --- |
| CUDA unavailable | nvidia-smi、torch.version.cuda、容器 GPU 暴露 | 修复环境；不把 CPU 执行登记 CUDA |
| OOM | 模型/精度、microbatch、并发进程、显存峰值 | 降 microbatch/增累积、新 Pilot；采样减 batch；不改 256px/主干 |
| Cache hash mismatch | rsync 是否中断、路径、正式/试点缓存混用 | 补齐准确文件重新回读；不改锁 |
| Frozen CLIP 转换哈希不同 | 锁定 Transformers/PyTorch 版本、上游文件 | 同步原冻结组件；不接受新哈希自动覆盖 |
| Resume source/config mismatch | Git 提交、训练 JSON、数据/组件锁、日程 | 回到原身份；跨源码只能独立评测或显式新实验 |
| Loss 有限但不可辨 | 单图、16 图、time slices、EMA、条件、solver | 保留证据并停止长训；不按 loss 判质量 |
| `bench generate` 输出存在 | 是否同一恢复任务、内容哈希与初噪 | 相同身份用 --resume；新设置用新目录 |
| 指标 NaN/执行失败 | 图片损坏、范围、权重哈希、样本数、设备 | 留失败及分母；修复后 --resume，不丢弃失败 |
| KID 小样本为负 | 无偏 MMD 估计本来可负 | 如实报告；不能截成 0 |
| VQA 不可判断/错误答案 | 模型能力、题型、人工校准、原始输出 | 分开记录覆盖率与保守成功率，不强判正确 |
| 人工导入 partial | 漏题、未填分数、package ID、reviewer/time | 回原页面补真实审核；不能脚本填满 |
| SSH 掉线 | tmux 中进程是否仍运行 | 重连后 attach；不要重复启动同一实验 |
| 磁盘满 | 快照数量、特征缓存、临时文件 | 先备份校验再清理；不删除唯一 last.pt |

## **十二、执行记录与最后的停止点**

每个云端任务登记：开始/结束时间、精确命令、提交、配置/数据/组件/Benchmark 身份、GPU 型号、精度、成功/失败、实际产物、实测资源和实际账单。保持 P3-05 质量、P3-06 CUDA、P4 正式实验、P5 真实 VQA/人工校准这些验收分别记录。

本轮交付手册不代表已经运行上述云端命令，也不自动批准正式长训。**下一步由用户审核 Benchmark 与手册，然后亲自执行单卡 CUDA Pilot；P6/P7、额外规模、额外种子、多卡和研究结论不在本轮范围。** 分任务勾选使用 [云端任务清单](云端任务清单.md)，每个对号必须对应真实回执。
