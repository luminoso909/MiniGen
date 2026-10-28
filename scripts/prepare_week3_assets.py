"""Explicit public downloads, fixed identities, hashes and PDF verification; no training."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPO = "madebyollin/sdxl-vae-fp16-fix"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "MiniGen-study/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(path)


def papers():
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


def model():
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
