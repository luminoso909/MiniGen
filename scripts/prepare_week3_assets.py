"""显式下载并核验公开论文与冻结 VAE 单组件，不启动训练。"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPO = "madebyollin/sdxl-vae-fp16-fix"


def sha256(
    path: Path,
) -> str:
    """对本地论文或权重文件计算分块 SHA-256。

    Args:
        path: 待核验的文件 Path。

    Raises:
        OSError: 文件缺失或无法读取时透传。

    Results:
        str: 十六进制摘要，供已有资源与固定身份比较。
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(
    url: str,
    path: Path,
) -> None:
    """显式下载公开资源到临时文件，成功后替换目标。

    Args:
        url: 已选定的公开资源 URL。
        path: 目标本地路径，父目录按需创建。

    Raises:
        URLError/HTTPError/TimeoutError: 网络或服务失败时透传；OSError: 本地写入失败。

    副作用:
        访问远程 URL；写临时文件，成功后替换本地目标。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "MiniGen-study/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(path)


def papers() -> None:
    """复核已有核心论文，并获取固定 DiT v2 后更新来源清单。

    Raises:
        ValueError: 已有论文哈希不符或 DiT 首页标题不匹配；PDF 解析、网络和文件错误透传。

    副作用:
        按需下载 PDF，更新论文来源清单并打印核验结果。
    """
    folder = ROOT / "docs/papers"
    manifest_path = folder / "manifest.json"
    records = json.loads(manifest_path.read_text())
    for record in records:
        path = folder / record["file"]
        if not path.exists() or sha256(path) != record["sha256"]:
            fetch(record["download_url"], path)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Existing paper hash mismatch: {path.name}")
        PdfReader(path).pages[0].extract_text()
    version = "2212.09748v2"
    filename = "DiT_Scalable_Diffusion_Models_with_Transformers_2212.09748v2.pdf"
    path = folder / filename
    if not path.exists():
        fetch(f"https://arxiv.org/pdf/{version}", path)
    reader = PdfReader(path)
    title = "Scalable Diffusion Models with Transformers"
    first = " ".join(reader.pages[0].extract_text().split())
    if title.lower() not in first.lower():
        raise ValueError("DiT PDF title verification failed")
    for page in reader.pages:
        page.extract_text()
    record = {
        "file": filename,
        "title": title,
        "authors": ["William Peebles", "Saining Xie"],
        "year": 2023,
        "first_submitted_year": 2022,
        "version": version,
        "role": "week3_core",
        "source": f"https://arxiv.org/abs/{version}",
        "download_url": f"https://arxiv.org/pdf/{version}",
        "retrieved": "2026-10-28",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "pages": len(reader.pages),
        "pdf_parse_verified": True,
    }
    records = [r for r in records if r["version"] != version] + [record]
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"papers": len(records), "new": record}, ensure_ascii=False), flush=True)


def model() -> None:
    """显式获取或复用固定 SDXL VAE 单组件，校验其配置与 Safetensors。

    Raises:
        ValueError: 冻结模型文件哈希不符；JSON、网络、文件或必要身份字段错误透传。

    副作用:
        按需下载固定组件，写本地权重与模型身份，打印资源记录。
    """
    lock_path = ROOT / "configs/models/sdxl_vae.json"
    if lock_path.exists():
        record = json.loads(lock_path.read_text())
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}") as response:
            revision = json.load(response)["sha"]
        record = {"repo_id": REPO, "revision": revision, "files": {}}
    directory = ROOT / "artifacts/models/sdxl_vae" / record["revision"]
    for name in ["config.json", "diffusion_pytorch_model.safetensors"]:
        path = directory / name
        known = record["files"].get(name)
        if not path.exists() or (known and sha256(path) != known["sha256"]):
            print(f"Downloading {name}", flush=True)
            fetch(f"https://huggingface.co/{REPO}/resolve/{record['revision']}/{name}", path)
        digest = sha256(path)
        if known and digest != known["sha256"]:
            raise ValueError(f"Frozen model hash mismatch: {name}")
        record["files"][name] = {"sha256": digest, "bytes": path.stat().st_size}
    record.update(
        retrieved="2026-10-28",
        source=f"https://huggingface.co/{REPO}",
        local_path=directory.relative_to(ROOT).as_posix(),
        use="frozen_autoencoder_only",
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-only", action="store_true")
    args = parser.parse_args()
    papers()
    if not args.papers_only:
        model()
