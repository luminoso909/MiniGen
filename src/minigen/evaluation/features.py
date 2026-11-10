"""只从锁定本地资产提取评测特征；CLIP 与生成条件文本塔严格分离。"""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from minigen.autoencoders.sdxl import hash_file


class FeatureSuite:
    """延迟加载评测模型，图像特征按内容、模型、预处理和设备身份缓存。

    Args:
        root: 仓库根目录。
        asset_lock: 含 models.inception/clip 的身份文件。
        device: 实际 FP32 推理设备。
        batch_size: 每次模型调用的图片或文本数量。
        cache_dir: 可选特征缓存；None 使用项目忽略目录。
    """

    def __init__(
        self,
        root: Path,
        asset_lock: Path,
        device: str = "cpu",
        batch_size: int = 4,
        cache_dir: Path | None = None,
    ) -> None:
        """读取身份但不下载、不立即加载权重。

        Args:
            root: 项目根目录。
            asset_lock: 资产锁路径。
            device: CPU、MPS 或 CUDA。
            batch_size: 正整数批量。
            cache_dir: 忽略目录中的缓存位置。

        Raises:
            ValueError: 批量非法。
            OSError/KeyError: 身份文件不可读或字段缺失。

        副作用:
            保存实例配置和空模型引用。
        """
        if batch_size < 1:
            raise ValueError("Feature batch_size must be positive")
        self.root = root.resolve()
        self.lock = self.root / asset_lock
        self.identity = json.loads(self.lock.read_text())["models"]
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.cache = cache_dir or self.root / "artifacts/cache/evaluation"
        self.models: dict[str, Any] = {}
        self.text_lengths: list[dict[str, int | bool]] = []

    def close(self) -> None:
        """卸载当前实例持有的模型，便于在统一内存设备顺序评分。

        副作用:
            清空模型引用；设备分配器是否回收由 PyTorch 决定。
        """
        self.models.clear()

    def _model(
        self,
        name: str,
    ) -> Any:
        """校验完整资产再按需构造冻结模型；不隐式请求网络。

        Args:
            name: inception 或 clip。

        Raises:
            ValueError/OSError: 资产不匹配或模型无法加载。
            ImportError: 所需可选依赖未安装。

        Results:
            Any: 底层评测模型，clip 项为模型与 tokenizer 元组。

        副作用:
            在实际设备加载模型并存入实例缓存。
        """
        if name not in self.models:
            from minigen.evaluation.assets import verify_asset

            directory = verify_asset(self.root, self.lock, name)
            if name == "inception":
                from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

                weights = directory / next(
                    key for key in self.identity[name]["files"] if key.endswith(".pth")
                )
                model = FeatureExtractorInceptionV3(
                    "inception-v3-compat", ["2048"], feature_extractor_weights_path=str(weights)
                )
                self.models[name] = model.eval().requires_grad_(False).to(self.device)
            elif name == "clip":
                from transformers import CLIPModel, CLIPTokenizerFast

                model = (
                    CLIPModel.from_pretrained(
                        directory, local_files_only=True, use_safetensors=False
                    )
                    .eval()
                    .requires_grad_(False)
                    .float()
                    .to(self.device)
                )
                tokenizer = CLIPTokenizerFast.from_pretrained(directory, local_files_only=True)
                self.models[name] = (model, tokenizer)
            else:
                raise ValueError(f"Unknown feature model {name}")
        return self.models[name]

    @torch.inference_mode()
    def _images(
        self,
        paths: list[Path],
        name: str,
    ) -> Float[Tensor, "n d"]:
        """逐批提取图像特征，原子缓存每图结果并保留输入顺序。

        Args:
            paths: 非空图像路径列表，尺寸必须为正式 256×256 RGB。
            name: inception 或 clip。

        Raises:
            ValueError: 图像尺寸、模型特征或缓存非法。
            OSError/RuntimeError: 图片、权重、缓存读取或推理失败。

        Results:
            Float[Tensor, "n d"]: CPU FP32 特征，Inception 为 2048 维，CLIP 为 768 维。

        副作用:
            按需加载模型并原子写入忽略目录的特征缓存。
        """
        if not paths:
            raise ValueError("Image list is empty")
        # 先核验资产，缓存命中不能绕过权重版本和文件完整性约束。
        model = self._model(name)
        signature = {
            "model": self.identity[name],
            "preprocess": ("rgb256;clip:PIL-bicubic336-normalize-v1;inception:uint8-native"),
            "device": str(self.device),
            "precision": "float32",
            "torch": str(torch.__version__),
            "pillow": version("Pillow"),
            "backend_version": version("transformers" if name == "clip" else "torch-fidelity"),
            "implementation_sha256": hash_file(Path(__file__)),
        }
        namespace = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        destination = self.cache / namespace
        destination.mkdir(parents=True, exist_ok=True)
        result: list[Tensor | None] = [None] * len(paths)
        missing = []
        for i, path in enumerate(paths):
            cache = destination / f"{hash_file(path)}.pt"
            if cache.exists():
                value = torch.load(cache, map_location="cpu", weights_only=True)
                if (
                    value.shape != ((2048,) if name == "inception" else (768,))
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("Invalid cached feature")
                result[i] = value
            else:
                missing.append((i, path, cache))
        for start in range(0, len(missing), self.batch_size):
            items = missing[start : start + self.batch_size]
            images = []
            for _, path, _ in items:
                with Image.open(path) as im:
                    if im.size != (256, 256) or im.mode != "RGB":
                        raise ValueError(f"Expected native RGB256 image: {path}")
                    if name == "clip":
                        image = TF.to_tensor(
                            TF.resize(im, [336, 336], interpolation=InterpolationMode.BICUBIC)
                        )
                        image = TF.normalize(
                            image,
                            [0.48145466, 0.4578275, 0.40821073],
                            [0.26862954, 0.26130258, 0.27577711],
                        )
                    else:
                        image = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
                    images.append(image)
            batch = torch.stack(images).to(self.device)
            if name == "clip":
                backbone, _ = model
                output = backbone.visual_projection(
                    backbone.vision_model(pixel_values=batch, return_dict=True).pooler_output
                )
            else:
                output = model(batch)[0]
            output = output.detach().float().cpu()
            width = 768 if name == "clip" else 2048
            if output.shape != (len(items), width) or not torch.isfinite(output).all():
                raise ValueError("Invalid extracted feature shape or values")
            for (index, _, path), row in zip(items, output, strict=True):
                temporary = path.with_suffix(".tmp")
                torch.save(row.clone(), temporary)
                temporary.replace(path)
                result[index] = row
        return torch.stack([row for row in result if row is not None])

    def inception(
        self,
        paths: list[Path],
    ) -> Float[Tensor, "n 2048"]:
        """提取未归一化的 TensorFlow 兼容 Inception 池化特征。

        Args:
            paths: 有序 RGB256 图像路径。

        Raises:
            ValueError/OSError/RuntimeError: 图像、资产、缓存或推理无效。

        Results:
            Float[Tensor, "n 2048"]: CPU FP32 图像特征。
        """
        return self._images(paths, "inception")

    def clip_images(
        self,
        paths: list[Path],
    ) -> Float[Tensor, "n 768"]:
        """提取 CLIP L/14@336 图像投影，由指标层实施单位归一化。

        Args:
            paths: 有序 RGB256 图像路径。

        Raises:
            ValueError/OSError/RuntimeError: 图像、资产、缓存或推理无效。

        Results:
            Float[Tensor, "n 768"]: CPU FP32 投影，未归一化。
        """
        return self._images(paths, "clip")

    @torch.inference_mode()
    def clip_text(
        self,
        texts: list[str],
    ) -> Float[Tensor, "n 768"]:
        """不加前缀编码 Caption，显式记录 77 token 上限造成的截断。

        Args:
            texts: 与待评分图严格同序的非空文本列表。

        Raises:
            ValueError: 文本为空或输出特征非法。
            OSError/RuntimeError: 权重、tokenizer 或推理失败。

        Results:
            Float[Tensor, "n 768"]: CPU FP32 文本投影。

        副作用:
            加载 CLIP，更新 text_lengths 为本次各文本长度和截断标记。
        """
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("CLIPScore needs nonempty text")
        model, tokenizer = self._model("clip")
        lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        self.text_lengths = [{"tokens_before_truncation": n, "truncated": n > 77} for n in lengths]
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            inputs = tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            pooled = model.text_model(**inputs, return_dict=True).pooler_output
            outputs.append(model.text_projection(pooled).float().cpu())
        features = torch.cat(outputs)
        if features.shape != (len(texts), 768) or not torch.isfinite(features).all():
            raise ValueError("Invalid CLIP text features")
        return features
