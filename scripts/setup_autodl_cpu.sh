#!/usr/bin/env bash
# AutoDL 无卡模式：只准备环境、公开输入与数据清单，不开 GPU、不启动训练。
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"
[[ $root == /root/autodl-tmp/minigen ]] || { echo '请在 /root/autodl-tmp/minigen 内执行' >&2; exit 2; }
mountpoint -q /root/autodl-tmp
mkdir -p .cache/{bootstrap,uv,huggingface} artifacts/cloud/autodl
export UV_CACHE_DIR="$root/.cache/uv" HF_HOME="$root/.cache/huggingface"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONOPTIMIZE
bootstrap_python=${MINIGEN_BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python3.12}
uv="$root/.cache/bootstrap/bin/uv"
cpu="$root/.venv-cpu/bin/python"
hub=${MINIGEN_HF_ENDPOINT:-https://hf-mirror.com}
case "$hub" in https://huggingface.co|https://hf-mirror.com) ;; *) echo '请选择官方站点或已知镜像' >&2; exit 2 ;; esac

fetch_locked() {
    local url=$1 file=$2 bytes=$3 sha=$4 part
    part="$file.partial"
    mkdir -p "$(dirname "$file")"
    if [[ -f $file ]]; then
        [[ $(stat -c %s "$file") == "$bytes" ]]
        printf '%s  %s\n' "$sha" "$file" | sha256sum -c -
        return
    fi
    if [[ ! -f $part || $(stat -c %s "$part") != "$bytes" ]]; then
        curl -fLsS --retry 3 --connect-timeout 12 --max-time 1200 -C - -o "$part" "$url"
    fi
    [[ $(stat -c %s "$part") == "$bytes" ]]
    printf '%s  %s\n' "$sha" "$part" | sha256sum -c -
    mv -- "$part" "$file"
}

asset_list() {
    "$bootstrap_python" - "$hub" "$1" <<'PY'
import json
import sys
from pathlib import Path

hub, step = sys.argv[1:]
if step == "dataset":
    data = json.loads(Path("reports/v2/data/artbench_freeze.json").read_text())
    revision = data["source_revision"]
    entries = [
        (f"{hub}/datasets/{data['source_id']}/resolve/{revision}/{row['path']}",
         f"data/raw/artbench_pd_256/{revision}/{row['path']}", row["bytes"], row["sha256"])
        for row in data["raw_shards"]
    ]
else:
    vae = json.loads(Path("configs/models/sdxl_vae.json").read_text())
    clip = json.loads(Path("configs/models/clip_text.json").read_text())
    entries = [
        (f"{hub}/{lock['repo_id']}/resolve/{lock['revision']}/{name}",
         f"{lock['local_path']}/{name}", info["bytes"], info["sha256"])
        for lock in (vae, clip)
        for name, info in lock["files"].items()
        if lock is vae or name not in {"model.safetensors", "config.json"}
    ]
    if not all((Path(clip["local_path"]) / name).is_file()
               for name in ("config.json", "model.safetensors")):
        entries.append((
            f"{hub}/{clip['repo_id']}/resolve/{clip['revision']}/pytorch_model.bin",
            f"{clip['local_path']}/upstream.bin", clip["upstream_bytes"], clip["upstream_sha256"],
        ))
for entry in entries:
    print(*entry, sep="\t")
PY
}

# 数据审计只需 tokenizer 和 VAE 模块里的纯 SHA-256 函数。独立进程不加载 PyTorch。
prepare_worker() {
    local worker="$root/.cache/artbench-worker" source_path
    source_path="$worker/src"
    mkdir -p "$source_path"
    [[ -x $worker/venv/bin/python ]] || "$uv" venv --python "$bootstrap_python" "$worker/venv"
    "$uv" pip install --python "$worker/venv/bin/python" \
        --default-index "${MINIGEN_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" \
        --constraint artifacts/cloud/autodl/cpu-constraints.txt numpy pillow pyyaml pyarrow transformers
    cp -R src/minigen "$source_path/"
    "$bootstrap_python" - "$source_path/minigen" <<'PY'
import ast
import sys
from pathlib import Path

package = Path(sys.argv[1])
path = package / "autoencoders/sdxl.py"
source = Path("src/minigen/autoencoders/sdxl.py").read_text()
function = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "hash_file")
path.write_text("import hashlib\nfrom pathlib import Path\n\n" +
                ast.get_source_segment(source, function) + "\n")

# 同一来源顺序和列，只改变 Parquet 缓冲方式；最终仍须匹配冻结 Manifest。
reader = package / "data/artbench.py"
original = reader.read_text()
changes = (
    ("    from pyarrow.parquet import ParquetFile\n", "    import pyarrow as pa\n"
     "    from pyarrow.parquet import ParquetFile\n"),
    ("ParquetFile(path)", "ParquetFile(path, memory_map=True, pre_buffer=False)"),
    ("            for batch in file.iter_batches(batch_size=128, columns=SOURCE_FIELDS):\n"
     "                for row in batch.to_pylist():\n"
     "                    yield name, index, row\n"
     "                    index += 1\n",
     "            for group in range(file.num_row_groups):\n"
     "                for batch in file.iter_batches(\n"
     "                    batch_size=128, columns=SOURCE_FIELDS, row_groups=[group],\n"
     "                    use_threads=False,\n"
     "                ):\n"
     "                    for row in batch.to_pylist():\n"
     "                        yield name, index, row\n"
     "                        index += 1\n"
     "                pa.default_memory_pool().release_unused()\n"),
)
for before, after in changes:
    if original.count(before) != 1:
        raise SystemExit("ArtBench reader changed; review the low-memory adapter")
    original = original.replace(before, after)
compile(original, str(reader), "exec")
reader.write_text(original)
PY
    PYTHONPATH="$source_path" "$worker/venv/bin/python" - <<'PY'
import sys
from pathlib import Path

from minigen.data.artbench_caption import load_clip_token_counter

counter = load_clip_token_counter(Path.cwd(), Path("configs/models/clip_text.json"))
print(f"ArtBench worker tokenizer smoke: {counter('a landscape painting')} tokens")
if "torch" in sys.modules:
    raise RuntimeError("ArtBench worker unexpectedly loaded PyTorch")
PY
}

run_artbench() {
    local worker="$root/.cache/artbench-worker"
    if [[ -x $worker/venv/bin/python && -d $worker/src/minigen ]]; then
        ARROW_DEFAULT_MEMORY_POOL=system MALLOC_ARENA_MAX=2 \
            PYTHONPATH="$worker/src" "$worker/venv/bin/python" scripts/prepare_artbench.py "$@"
    else
        "$cpu" scripts/prepare_artbench.py "$@"
    fi
}

case "${1:-}" in
env)
    if [[ ! -x $uv ]]; then
        PIP_CACHE_DIR="$root/.cache/bootstrap/pip" "$bootstrap_python" -m pip install \
            --target "$root/.cache/bootstrap" 'uv>=0.9,<1'
    fi
    [[ -x $cpu ]] || "$uv" venv --python "$bootstrap_python" .venv-cpu
    "$bootstrap_python" - <<'PY'
import tomllib
from pathlib import Path

packages = tomllib.loads(Path("uv.lock").read_text())["package"]
constraints = {}
for package in packages:
    name = package["name"]
    if name != "minigen":
        constraints.setdefault(name, set()).add(package["version"])
Path("artifacts/cloud/autodl/cpu-constraints.txt").write_text("".join(
    f"{name}=={next(iter(versions))}\n"
    for name, versions in sorted(constraints.items()) if len(versions) == 1
))
PY
    "$uv" pip install --python "$cpu" --torch-backend cpu \
        --default-index "${MINIGEN_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}" \
        --constraint artifacts/cloud/autodl/cpu-constraints.txt \
        -e '.[artbench,conditioning,latent]' pytest
    "$uv" pip check --python "$cpu"
    "$uv" pip freeze --python "$cpu" > artifacts/cloud/autodl/cpu-packages.txt
    .venv-cpu/bin/minigen doctor --device cpu --json | tee artifacts/cloud/autodl/doctor_cpu.json
    ;;
dataset|models)
    list=$(asset_list "$1")
    while IFS=$'\t' read -r url file size sha; do
        fetch_locked "$url" "$file" "$size" "$sha"
    done <<< "$list"
    if [[ $1 == models ]]; then
        read -r clipdir cliprepo cliprev < <("$bootstrap_python" -c '
import json
from pathlib import Path
lock = json.loads(Path("configs/models/clip_text.json").read_text())
print(lock["local_path"], lock["repo_id"], lock["revision"])
')
        if [[ ! -f $clipdir/config.json || ! -f $clipdir/model.safetensors ]]; then
            curl -fLsS --retry 3 --connect-timeout 12 --max-time 60 \
                "$hub/$cliprepo/raw/$cliprev/config.json" -o "$clipdir/upstream_config.json.partial"
            mv -- "$clipdir/upstream_config.json.partial" "$clipdir/upstream_config.json"
        fi
    fi
    ;;
worker)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    prepare_worker
    ;;
prepare)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    clipdir=$("$bootstrap_python" -c 'import json; print(json.load(open("configs/models/clip_text.json"))["local_path"])')
    if [[ ! -s $clipdir/config.json || ! -s $clipdir/model.safetensors ]]; then
        [[ -s $clipdir/upstream.bin && -s $clipdir/upstream_config.json ]] ||
            { echo '先完成 models 下载' >&2; exit 2; }
    fi
    "$cpu" scripts/prepare_clip_cpu.py
    # 模型加载器只接受锁定文件。转换成功后，保留上游归档并移出组件目录。
    for file in upstream.bin upstream_config.json; do
        if [[ -f $clipdir/$file ]]; then
            archive="$root/.cache/bootstrap/clip-upstream/$(basename "$clipdir")"
            mkdir -p "$archive"
            if [[ -e $archive/$file ]]; then
                cmp -s "$clipdir/$file" "$archive/$file" || {
                    echo "上游归档已有不同内容：$archive/$file" >&2
                    exit 1
                }
            fi
            mv -- "$clipdir/$file" "$archive/$file"
        fi
    done
    freeze=reports/v2/data/artbench_freeze.json
    expected=$("$cpu" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$freeze")
    target=data/processed/artbench_pd_256/v1
    candidate=data/processed/artbench_pd_256/v1.autodl-candidate
    output=$candidate
    [[ ! -d $target ]] || output=$target
    run_artbench --output-dir "$output" | tee artifacts/cloud/autodl/artbench_prepare.json
    printf '%s  %s\n' "$expected" "$output/manifest.jsonl" | sha256sum -c -
    if [[ $output == "$candidate" ]]; then
        [[ ! -e $target ]] || { echo '目标目录已存在，停止发布' >&2; exit 1; }
        mv -T -- "$candidate" "$target"
    fi
    ;;
verify)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    bench=reports/v2/benchmark/artbench_v1
    [[ -s $bench/benchmark.json && -s $bench/lock.json ]] ||
        { echo '缺少已冻结的 Benchmark 内容或锁' >&2; exit 2; }
    [[ -s data/processed/artbench_pd_256/v1/manifest.jsonl ]] ||
        { echo '先完成 prepare' >&2; exit 2; }
    run_artbench --output-dir data/processed/artbench_pd_256/v1 |
        tee artifacts/cloud/autodl/artbench_readback.json
    .venv-cpu/bin/minigen bench prepare --output "$bench" | tee artifacts/cloud/autodl/benchmark_integrity.json
    "$cpu" -m pytest -q -p no:cacheprovider tests/test_artbench.py tests/test_artbench_caption.py \
        tests/test_representation_cache.py tests/test_benchmark.py
    ;;
*)
    echo '用法：bash scripts/setup_autodl_cpu.sh {env|dataset|models|worker|prepare|verify}' >&2
    exit 2
    ;;
esac
