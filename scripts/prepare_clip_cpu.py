"""在内存受限的 CPU 主机复现已锁定的 MiniGen CLIP 文本塔。"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def matches(
    path: Path,
    size: int,
    digest: str,
) -> bool:
    """检查一个资源是否与固定身份一致。

    Args:
        path: 本地文件路径。
        size: 期望文件大小（字节）。
        digest: 期望的 SHA-256 十六进制摘要。

    Raises:
        OSError: 文件存在但无法读取。

    Results:
        bool: 文件存在且大小与哈希都一致。
    """
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == digest


def main() -> None:
    """从已下载的锁定归档提取 CLIP 文本参数，验收后才发布。

    Raises:
        OSError: 本地锁、模型文件或目标目录不可读写。
        ValueError: 输入资源、配置或输出不符合身份锁。
        RuntimeError: mmap 加载、模型严格匹配或序列化失败。

    副作用:
        在冻结组件目录写入经校验的配置和权重，打印资源状态。
    """
    lock = json.loads(Path("configs/models/clip_text.json").read_text())
    directory = Path(lock["local_path"])
    model_files = ("config.json", "model.safetensors")
    for name, spec in lock["files"].items():
        path = directory / name
        if (name not in model_files or path.exists()) and not matches(
            path, spec["bytes"], spec["sha256"]
        ):
            raise ValueError(f"CLIP file identity mismatch: {path}")
    if all((directory / name).is_file() for name in model_files):
        print("Reused verified frozen CLIP text component")
        return

    import torch
    from transformers import CLIPConfig, CLIPTextModel

    archive = directory / "upstream.bin"
    if not matches(archive, lock["upstream_bytes"], lock["upstream_sha256"]):
        raise ValueError("CLIP upstream weight hash mismatch")
    upstream = json.loads((directory / "upstream_config.json").read_text())
    config = CLIPConfig.from_dict(upstream).text_config
    # 避免在 CPU 上随机初始化一整份参数。mmap 不会预读未使用的视觉权重。
    with torch.device("meta"):
        model = CLIPTextModel(config)
    state = torch.load(archive, map_location="cpu", weights_only=True, mmap=True)
    text = {key: value for key, value in state.items() if key.startswith("text_model.")}
    del state
    if "embeddings.token_embedding.weight" in model.state_dict():
        text = {key.removeprefix("text_model."): value for key, value in text.items()}
    for key in ("embeddings.position_ids", "text_model.embeddings.position_ids"):
        if key in text and key not in model.state_dict():
            if not torch.equal(text[key], torch.arange(config.max_position_embeddings)[None]):
                raise ValueError("Unexpected legacy CLIP position_ids")
            del text[key]
    model.load_state_dict(text, strict=True, assign=True)
    print("Validated CLIP text state; serializing locked component", flush=True)

    with TemporaryDirectory(prefix=".autodl-clip-", dir=directory) as temporary:
        output = Path(temporary)
        model.save_pretrained(output, safe_serialization=True)
        for name in model_files:
            spec = lock["files"][name]
            if not matches(output / name, spec["bytes"], spec["sha256"]):
                raise ValueError(f"CLIP conversion does not match the frozen lock: {name}")
        for name in model_files:
            (output / name).replace(directory / name)
    print("Converted and verified frozen CLIP text component")


if __name__ == "__main__":
    main()
