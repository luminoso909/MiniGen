#!/usr/bin/env bash
# 在 MiniGen 根目录逐项验收。每项写独立回执；绝不下载、长训或批准 Benchmark。
set -Eeuo pipefail

usage() {
    cat <<'HELP'
用法：bash scripts/check_cloud_readiness.sh <完整Git SHA> <验收批次> <步骤> [b0|b1|b2]

步骤（建议按顺序；每次只执行一项）：
  data              核对预登记、冻结组件、全部 ArtBench 文件和正式缓存
  benchmark         核对候选内容与参考图；不批准、不打分
  cuda              Linux 单卡、驱动、设备、可选依赖及磁盘记录
  fp32 MODEL        固定样本 CPU/CUDA 前反向对照，需同步 P4 Checkpoint
  amp MODEL         50+50 BF16 Pilot；需 MINIGEN_GPU_HOURLY、MINIGEN_AMP_TIMEOUT（秒）
  resume MODEL      BF16 连续 8 步与 4+4 对照
  quality-single    B0 单图有界诊断，产物待人工查看
  quality-16        B0 固定 16 图三种条件，需先人工查看并接受单图
                    （显式设置 MINIGEN_SINGLE_REVIEWED=yes）
  eval-assets       Inception、完整 CLIP、LPIPS 离线哈希检查
  eval-device       在两张验证图上检查 Inception/CLIP CPU/CUDA 对照
  vqa-assets        BLIP 离线资产检查，不执行推理

回执：artifacts/cloud/acceptance/<验收批次>/<步骤>[-模型].XXXXXX/
同一批次可重试；保留每次日志。脚本不安装依赖、不联网、不创建批准回执。
HELP
}

fail() {
    printf '停止：%s\n' "$*" >&2
    exit 2
}

file_sha256() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
[[ $# -ge 3 && $# -le 4 ]] || { usage >&2; exit 2; }
expected=$1 batch=$2 stage=$3 model=${4:-}
[[ $expected =~ ^[0-9a-f]{40}$ ]] || fail '需要完整的 40 位 Git SHA'
[[ $batch =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || fail '验收批次名称无效'
case "$stage" in
    fp32|amp|resume) [[ $model =~ ^b[012]$ ]] || fail '必须选择 b0、b1 或 b2' ;;
    data|benchmark|cuda|quality-single|quality-16|eval-assets|eval-device|vqa-assets)
        [[ $# == 3 ]] || fail '本步骤不接受模型参数' ;;
    *) fail "未知步骤：$stage" ;;
esac

root=$(git rev-parse --show-toplevel)
script_sha=$(file_sha256 "${BASH_SOURCE[0]}")
cd "$root"
[[ $(git rev-parse HEAD) == "$expected" ]] || fail 'HEAD 不等于目标提交'
locked=(src configs reports/v2/components reports/v2/data reports/v2/p4 reports/v2/benchmark/artbench_v1)
[[ -z $(git status --porcelain --untracked-files=no -- "${locked[@]}" scripts pyproject.toml uv.lock .python-version) ]] ||
    fail '源码、脚本、配置或冻结回执存在已跟踪的本地改动'
[[ -z $(git ls-files --others --exclude-standard -- "${locked[@]}") ]] ||
    fail '运行源码、配置或冻结回执下存在未跟踪文件'

base="$root/artifacts/cloud/acceptance/$batch"
mkdir -p "$base"
if ! (set -o noclobber; printf '%s\n' "$expected" > "$base/commit") 2>/dev/null; then
    [[ $(cat "$base/commit") == "$expected" ]] || fail '该批次已绑定到其他提交'
fi
receipt=$(mktemp -d "$base/$stage${model:+-$model}.XXXXXX")
# 禁止 uv 同步或尝试隐式拉取，HF 组件也只能离线读取。
unset PYTHONPATH PYTHONOPTIMIZE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
run=(uv run --locked --offline --no-sync)
run_logged() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}
save_json() {
    local target=$1
    shift
    printf '+'
    printf ' %q' "$@"
    printf ' > %q\n' "$target"
    "$@" | tee "$target"
}
need() {
    local candidate
    for candidate in "$base/$1".*/exit-code; do
        if [[ -f $candidate && $(cat "$candidate") == 0 ]] &&
            grep -Fxq "script_sha256=$script_sha" "${candidate%/*}/identity.txt"; then
            return 0
        fi
    done
    fail "同一批次缺少已通过的 $1 回执"
}
pilot_config="configs/training/p4_${model}_cuda_pilot.json"
bench=reports/v2/benchmark/artbench_v1

execute_step() {
printf 'UTC=%s\ncommit=%s\nscript_sha256=%s\nstep=%s\nmodel=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$(date -u +%FT%TZ)" "$expected" "$script_sha" "$stage" "$model" "${CUDA_VISIBLE_DEVICES:-}" |
    tee "$receipt/identity.txt"
case "$stage" in
data)
    "${run[@]}" python - "$receipt" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import minigen
from minigen.data.manifest import load_manifest

root = Path.cwd()
assert Path(minigen.__file__).resolve().is_relative_to(root / "src"), "错误的 MiniGen 安装"
folder = Path(sys.argv[1])
spec = json.loads(Path("configs/experiments/precloud_preregistration.json").read_text())
frozen = spec["frozen_inputs"]
components = {}
for name in ("sdxl_vae", "clip_text"):
    lock = json.loads(Path(f"configs/models/{name}.json").read_text())
    components[name] = lock["revision"]
    for filename, record in lock["files"].items():
        path = Path(lock["local_path"]) / filename
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"], path
        assert path.stat().st_size == record["bytes"], path

files = frozen["files_sha256"] | spec["evaluation_implementation_sha256"] | {
    item["config_file"]: item["config_sha256"] for item in spec["models"]
}
for name, expected in files.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
package = root / "src/minigen"
digest = hashlib.sha256()
for file in sorted(package.rglob("*.py")):
    digest.update(str(file.relative_to(package)).encode() + b"\0" + file.read_bytes())
assert digest.hexdigest() == spec["delivery_runtime_source_sha256"], "训练源码身份变化"

data = Path("data/processed/artbench_pd_256/v1")
rows = load_manifest(data / "manifest.jsonl", data, verify_hashes=True)
assert Counter(row["split"] for row in rows) == frozen["splits"]
by_group, by_hash = {}, {}
for row in rows:
    for key, lookup in (("group_id", by_group), ("sha256", by_hash)):
        assert lookup.setdefault(row[key], row["split"]) == row["split"], row["sample_id"]
print("Verified image hashes:", len(rows))
(folder / "inputs.json").write_text(json.dumps({
    "verified_images": len(rows), "splits": frozen["splits"],
    "component_revisions": components, "source_sha256": digest.hexdigest()
}, indent=2) + "\n")
PY
    save_json "$receipt/cache.json" "${run[@]}" minigen cache \
        --output artifacts/cache/artbench_pd_256/v1_formal --verify-only \
        --approval reports/v2/components/vae_user_approval.json \
        --evidence reports/v2/components/vae_reconstruction_200_summary.json \
        --local-acceptance reports/v2/components/artbench_local_fp32_20.json
    "${run[@]}" python - "$receipt/cache.json" <<'PY'
import json
import sys
from pathlib import Path

cache = json.loads(Path(sys.argv[1]).read_text())
spec = json.loads(Path("configs/experiments/precloud_preregistration.json").read_text())
frozen = spec["frozen_inputs"]
assert cache["mode"] == "formal" and cache["verified_only"] is True
assert cache["verified_rows"] == {key: frozen["splits"][key] for key in ("train", "validation")}
for field in ("manifest_sha256", "cache_index_sha256", "storage", "payload_sha256"):
    assert cache[field] == frozen[field], field
PY
    ;;
benchmark)
    [[ -s $bench/benchmark.json && -s $bench/lock.json ]] || fail '缺少既有候选或内容锁'
    save_json "$receipt/benchmark.json" "${run[@]}" minigen bench prepare --output "$bench"
    ;;
cuda)
    [[ $(uname -s) == Linux ]] || fail '此步需要 Linux NVIDIA 主机'
    run_logged nvidia-smi
    run_logged df -h "$root" "$base"
    run_logged findmnt -T "$base"
    run_logged uv --version
    run_logged uv sync --locked --offline --check --inexact --no-install-project \
        --extra latent --extra conditioning --extra artbench --extra evaluation
    run_logged "${run[@]}" python -c '
import platform
import diffusers, safetensors, torch, transformers
assert torch.cuda.device_count() == 1, "请只暴露一张 CUDA GPU"
print("Python", platform.python_version(), "PyTorch", torch.__version__, "CUDA runtime", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0), "BF16", torch.cuda.is_bf16_supported(including_emulation=False))
print("diffusers", diffusers.__version__, "transformers", transformers.__version__, "safetensors", safetensors.__version__)
'
    save_json "$receipt/doctor_cuda.json" "${run[@]}" minigen doctor --device cuda --json
    ;;
fp32)
    need cuda
    need data
    checkpoint="artifacts/runs/p4_${model}_local/last.pt"
    [[ -s $checkpoint ]] || fail "请先同步 $checkpoint"
    "${run[@]}" python - "$checkpoint" "$model" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint, model = sys.argv[1:]
expected = json.loads(Path(f"reports/v2/p4/{model}_samples.json").read_text())["checkpoint_sha256"]
with Path(checkpoint).open("rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
print("P4 checkpoint SHA-256:", actual)
assert actual == expected, "同步的 P4 Checkpoint 与报告不一致"
PY
    run_logged "${run[@]}" python scripts/check_flow_device.py \
        --config "configs/training/p4_${model}_resume.json" --checkpoint "$checkpoint" \
        --device cuda --output "$receipt/fp32.json"
    ;;
amp)
    need "fp32-$model"
    [[ ${MINIGEN_AMP_TIMEOUT:-} =~ ^[1-9][0-9]{0,3}$ ]] ||
        fail '请用 MINIGEN_AMP_TIMEOUT 设置进程运行上限（秒，1～3600）'
    (( MINIGEN_AMP_TIMEOUT <= 3600 )) || fail 'Pilot 运行上限不得超过 3600 秒'
    [[ -n ${MINIGEN_GPU_HOURLY:-} ]] || fail '请设置 MINIGEN_GPU_HOURLY 为实际人民币/小时'
    run_logged "${run[@]}" python -c 'import torch; assert torch.cuda.is_bf16_supported(including_emulation=False)'
    run_logged timeout --kill-after=30s "${MINIGEN_AMP_TIMEOUT}s" "${run[@]}" python scripts/profile_training.py \
        --config "$pilot_config" --output "$receipt/pilot" --warmup 50 --measure 50 \
        --precision bf16 --hourly-cny "$MINIGEN_GPU_HOURLY"
    ;;
resume|quality-single|quality-16)
    case "$stage" in
        resume) need "amp-$model"; source=$pilot_config ;;
        quality-single) need resume-b0; source=configs/training/p3_close_single.json ;;
        quality-16)
            need quality-single
            [[ ${MINIGEN_SINGLE_REVIEWED:-} == yes ]] || fail '先看单图结果，再决定是否执行 16 图'
            source=configs/training/p3_close_sixteen.json ;;
    esac
    printf 'source_config=%s\n' "$source"
    "${run[@]}" python - "$source" "$receipt" "$stage" <<'PY'
import json
import sys
from pathlib import Path

source, directory, stage = sys.argv[1:]
folder = Path(directory)
base = json.loads(Path(source).read_text())
training = base["training"]
training["device"] = "cuda"
if stage == "resume":
    training.update(batch_size=2, grad_accum=1, precision="bf16", max_seconds=300,
                    schedule_steps=20, warmup_steps=2, checkpoint_every=4,
                    checkpoint_steps=[4, 8], validation_every=100)
    base["data"].update(train_limit=8, validation_limit=4)
    for name, steps, run in (("continuous", 8, "continuous"), ("part1", 4, "split"),
                              ("part2", 8, "split")):
        training.update(max_steps=steps, output_dir=str(folder / run))
        (folder / f"{name}.json").write_text(json.dumps(base, indent=2) + "\n")
else:
    training.update(precision="fp32", max_seconds=600, output_dir=str(folder / "train"))
    (folder / "quality.json").write_text(json.dumps(base, indent=2) + "\n")
PY
    if [[ $stage == resume ]]; then
        for name in continuous part1; do
            run_logged "${run[@]}" minigen train --config "$receipt/$name.json" --device cuda
        done
        run_logged "${run[@]}" minigen train --config "$receipt/part2.json" \
            --resume "$receipt/split/last.pt" --device cuda
        "${run[@]}" python - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

import torch
from minigen.training.checkpoint import load_checkpoint

def equal(
    left: object,
    right: object,
) -> bool:
    """递归比较 Checkpoint 状态，包括嵌套张量。

    Args:
        left: 连续运行保存的状态或其中一项。
        right: 恢复运行的对应项。

    Results:
        bool: 数据类型、结构和值均一致时为真。
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right

folder = Path(sys.argv[1])
left = load_checkpoint(folder / "continuous/last.pt")
right = load_checkpoint(folder / "split/last.pt")
fields = ("contract", "model", "ema", "optimizer", "scheduler", "scaler", "stream", "rng")
checks = {key: equal(left[key], right[key]) for key in fields}
checks["eight_updates"] = left["step"] == right["step"] == len(left["history"]) == len(right["history"]) == 8
keys = ("step", "loss", "lr", "grad_norm", "samples")
checks["history"] = [[row[key] for key in keys] for row in left["history"]] == [
    [row[key] for key in keys] for row in right["history"]
]
report = {"status": "passed" if all(checks.values()) else "needs_investigation",
          "checks": checks, "step": right["step"]}
report["max_abs"] = {
    section: max(float((value - right[section][key]).abs().max())
                 for key, value in left[section].items() if value.is_floating_point())
    for section in ("model", "ema")
}
(folder / "resume.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not all(checks.values()):
    sys.exit("恢复未证实逐位一致；保留两个 Checkpoint，检查确定性和轨迹后再推进")
PY
    else
        cfg="$receipt/quality.json"
        checkpoint="$receipt/train/last.pt"
        run_logged "${run[@]}" minigen train --config "$cfg" --device cuda
        run_logged "${run[@]}" python scripts/diagnose_flow_quality.py --config "$cfg" \
            --checkpoint "$checkpoint" --output "$receipt/diagnostics" --device cuda
        if [[ $stage == quality-16 ]]; then
            for condition in correct blank shuffled; do
                run_logged "${run[@]}" minigen sample --config "$cfg" --checkpoint "$checkpoint" \
                    --output "$receipt/$condition" --device cuda --split train --condition-mode "$condition" \
                    --count 16 --seed 1234 --steps 40 --method euler --weights ema --guidance 1
            done
            "${run[@]}" python - "$receipt" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
modes = ("correct", "blank", "shuffled")
reports = {mode: json.loads((root / mode / "samples.json").read_text()) for mode in modes}
ids = [row["sample_id"] for row in reports["correct"]["records"]]
assert len(ids) == len(set(ids)) == 16
for mode, report in reports.items():
    assert (report["seed"], report["split"], report["condition_mode"]) == (1234, "train", mode)
    assert [row["sample_id"] for row in report["records"]] == ids
    assert all((root / mode / row["image_path"]).is_file() for row in report["records"])
assert all(not row["caption"] for row in reports["blank"]["records"])
assert [row["condition_sample_id"] for row in reports["shuffled"]["records"]] == ids[1:] + ids[:1]
print("16 IDs, shared noise seed, three conditions and 48 images verified")
PY
        fi
        printf 'PENDING_HUMAN_REVIEW\n' > "$receipt/quality-status"
        printf '图片已生成，等待人工确认；执行成功不代表质量达标。\n'
    fi
    ;;
eval-assets)
    run_logged "${run[@]}" python scripts/prepare_eval_assets.py --models inception clip --verify-only
    "${run[@]}" python - <<'PY'
import hashlib
import json
from pathlib import Path

assets = json.loads(Path("reports/v2/components/p2_assets.json").read_text())["lpips_alex"]
for key in ("backbone", "calibration"):
    record = assets[key]
    with Path(record["path"]).open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"], key
    print("LPIPS", key, "verified")
PY
    ;;
eval-device)
    need cuda
    need eval-assets
    need benchmark
    run_logged "${run[@]}" python scripts/check_eval_backends.py \
        --benchmark "$bench/benchmark.json" --output "$receipt/eval_fp32.json" --device cuda
    "${run[@]}" python - "$receipt/eval_fp32.json" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())["results"]
assert {row["component"] for row in rows} == {"inception", "clip_images", "clip_text"}
for row in rows:
    print(row["component"], "CPU batch:", row["batch_passed"], "CUDA:", row["device_passed"],
          "selected:", row["selected_device"])
if not all(row["batch_passed"] and row["device_passed"] for row in rows):
    sys.exit("评测 CUDA 未完全通过；仅对 CPU batch 通过的组件考虑分项使用 CPU")
PY
    ;;
vqa-assets)
    run_logged "${run[@]}" python scripts/prepare_eval_assets.py --models vqa --verify-only
    printf '仅核对权重；真实推理和人工校准仍待验。\n'
    ;;
esac
}

# 单独保留命令退出码；tee 写入失败也不能算通过。
set +e
(set -e; execute_step) 2>&1 | tee "$receipt/run.log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}
if [[ $rc == 0 ]]; then
    rc=${codes[1]}
fi
printf '%s\n' "$rc" > "$receipt/exit-code" || exit 1
printf '退出码 %s；回执 %s\n' "$rc" "$receipt"
exit "$rc"
