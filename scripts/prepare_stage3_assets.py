"""显式准备 S3 论文及冻结 CLIP 文本组件；不启动训练。"""

import argparse
import json
import urllib.error
from pathlib import Path

import torch
from prepare_week3_assets import fetch, sha256
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPO = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
UPSTREAM_HASH = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
PAPERS = [
    (
        "2207.12598v1",
        "CFG_Classifier_Free_Diffusion_Guidance",
        "Classifier-Free Diffusion Guidance",
        ["Jonathan Ho", "Tim Salimans"],
        2022,
        "stage3_core",
    ),
    (
        "2103.00020v1",
        "CLIP_Learning_Transferable_Visual_Models",
        "Learning Transferable Visual Models From Natural Language Supervision",
        ["Alec Radford et al."],
        2021,
        "stage3_text_encoder",
    ),
    (
        "2210.02747v2",
        "Flow_Matching_for_Generative_Modeling",
        "Flow Matching for Generative Modeling",
        ["Yaron Lipman et al."],
        2023,
        "stage3_flow_preparation",
    ),
    (
        "2410.06940v4",
        "REPA_Representation_Alignment_for_Generation",
        "Representation Alignment for Generation: "
        "Training Diffusion Transformers Is Easier Than You Think",
        ["Sihyun Yu et al."],
        2025,
        "stage3_research_card",
    ),
    (
        "2510.11690v1",
        "RAE_Diffusion_Transformers_with_Representation_Autoencoders",
        "Diffusion Transformers with Representation Autoencoders",
        ["Boyang Zheng et al."],
        2025,
        "stage3_research_card",
    ),
    (
        "2505.13447v1",
        "MeanFlow_Mean_Flows_for_One_step_Generative_Modeling",
        "Mean Flows for One-step Generative Modeling",
        ["Zhengyang Geng et al."],
        2025,
        "stage3_research_card",
    ),
]


def fetch_resource(
    url: str,
    path: Path,
) -> None:
    """复用已固定 URL 的本地文件，网络瞬时失败最多重试三次。

    Args:
        url: 固定 revision 的公开下载地址。
        path: 资源路径；已存在时由上层身份校验，不重新下载。

    Raises:
        URLError/OSError: 三次仍失败时透传最后错误。

    副作用:
        必要时访问公开网络并保存文件。
    """
    if path.exists():
        return
    for attempt in range(3):
        try:
            # Git 文本文件使用官方 raw 地址，避开 resolve-cache 重定向的连接故障。
            fetch(url.replace("/resolve/", "/raw/"), path)
            return
        except (urllib.error.URLError, OSError):
            if attempt == 2:
                raise


def papers() -> None:
    """核验所有既有论文（含第一周），按固定版本补齐 S3 阅读材料。

    Raises:
        ValueError: 已冻结论文哈希或 PDF 标题/版本身份不符。
        OSError/网络/PDF 解析错误: 获取或读取资源失败时透传。

    副作用:
        按需访问公开 arXiv、保存 PDF、更新来源清单并打印核验摘要。
    """
    folder = ROOT / "docs/papers"
    manifest = folder / "manifest.json"
    records = json.loads(manifest.read_text())
    for record in records:
        path = folder / record["file"]
        if not path.exists():
            fetch(record["download_url"], path)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen paper hash mismatch: {path.name}")
    for version, stem, title, authors, year, role in PAPERS:
        path = folder / f"{stem}_{version}.pdf"
        if not path.exists():
            fetch(f"https://arxiv.org/pdf/{version}", path)
        reader = PdfReader(path)
        texts = [page.extract_text() for page in reader.pages]
        first = "".join(texts[0].lower().split()).replace("-", "")
        expected = "".join(title.lower().split()).replace("-", "")
        if expected not in first or version not in "".join(texts).replace(" ", ""):
            raise ValueError(f"Paper identity mismatch: {version}")
        record = {
            "file": path.name,
            "title": title,
            "authors": authors,
            "year": year,
            "version": version,
            "role": role,
            "source": f"https://arxiv.org/abs/{version}",
            "download_url": f"https://arxiv.org/pdf/{version}",
            "retrieved": "2026-11-04",
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "pages": len(reader.pages),
            "pdf_parse_verified": True,
        }
        records = [r for r in records if r["version"] != version] + [record]
        print(json.dumps(record, ensure_ascii=False), flush=True)
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def text_model() -> None:
    """从固定 OpenAI CLIP 权重严格提取文本网络，保存本地 Safetensors。

    Raises:
        ValueError: 上游或本地冻结文件哈希不匹配。
        ImportError/RuntimeError: 缺少 conditioning 依赖或文本权重严格加载失败。
        OSError/网络错误: 文件访问或公开资源下载失败时透传。

    副作用:
        下载约 605 MB 上游组合权重并暂存；校验后只保留约 254 MB 文本组件。
        创建本地模型与固定身份文件；不加载视觉网络，不下载图像生成主干。
    """
    from transformers import CLIPConfig, CLIPTextModel

    lock_path = ROOT / "configs/models/clip_text.json"
    if lock_path.exists():
        identity = json.loads(lock_path.read_text())
        for name, info in identity["files"].items():
            if sha256(ROOT / identity["local_path"] / name) != info["sha256"]:
                raise ValueError(f"Frozen CLIP file mismatch: {name}")
        print("Reused verified frozen CLIP text component", flush=True)
        return
    directory = ROOT / "artifacts/models/clip_text" / REVISION
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "upstream.bin"
    if not raw.exists():
        fetch(f"https://huggingface.co/{REPO}/resolve/{REVISION}/pytorch_model.bin", raw)
    if sha256(raw) != UPSTREAM_HASH:
        raise ValueError("CLIP upstream weight hash mismatch")
    original_config = directory / "upstream_config.json"
    fetch_resource(f"https://huggingface.co/{REPO}/resolve/{REVISION}/config.json", original_config)
    config = CLIPConfig.from_dict(json.loads(original_config.read_text())).text_config
    model = CLIPTextModel(config)
    state = torch.load(raw, map_location="cpu", weights_only=True)
    text_state = {key: value for key, value in state.items() if key.startswith("text_model.")}
    # Transformers 5 的独立文本模型去掉了 text_model 前缀；仅做明确的一一重命名，仍 strict。
    if "embeddings.token_embedding.weight" in model.state_dict():
        text_state = {key.removeprefix("text_model."): value for key, value in text_state.items()}
    # 旧版本把确定性的 position_ids 写入权重；新版本不持久化该 buffer。
    for key in ("embeddings.position_ids", "text_model.embeddings.position_ids"):
        if key in text_state and key not in model.state_dict():
            if not torch.equal(text_state[key], torch.arange(config.max_position_embeddings)[None]):
                raise ValueError("Unexpected legacy CLIP position_ids")
            del text_state[key]
    model.load_state_dict(text_state, strict=True)
    model.save_pretrained(directory, safe_serialization=True)
    names = ["config.json", "model.safetensors"]
    for name in [
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        fetch_resource(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", directory / name)
        names.append(name)
    identity = {
        "repo_id": REPO,
        "revision": REVISION,
        "retrieved": "2026-11-04",
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "upstream_sha256": UPSTREAM_HASH,
        "upstream_bytes": raw.stat().st_size,
        "conversion": (
            "CLIPTextModel strict text_model.* extraction; explicit prefix mapping for v5; "
            "verified deterministic position_ids omitted; safetensors; no projection"
        ),
        "local_path": directory.relative_to(ROOT).as_posix(),
        "files": {
            name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in names
        },
    }
    lock_path.write_text(json.dumps(identity, indent=2) + "\n")
    raw.unlink()
    original_config.unlink()
    print(json.dumps(identity), flush=True)


def main() -> None:
    """按命令行选择准备论文及显式文本组件。

    Raises:
        SystemExit: 参数错误；资源准备函数的网络、身份及文件异常透传。

    副作用:
        执行选中的显式下载并打印资源记录。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    if not args.text_only:
        papers()
    if not args.papers_only:
        text_model()


if __name__ == "__main__":
    main()
