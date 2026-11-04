"""S3 条件教学、缓存、采样计数、请求评分及恢复的独立小案例。"""

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

pytest.importorskip("cv2")
from minigen.conditioning.text import FrozenTextEncoder, text_cache
from minigen.data.icons import render
from minigen.evaluation.conditional import (
    paired_difference,
    score_request,
    shuffled_indices,
    summarize,
)
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.samplers.conditional import sample_conditional
from minigen.training.conditional import train_conditional
from minigen.training.denoising import TrainConfig


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """显式提取 W3 及 S3 标记的唯一教学单元，不执行 Notebook 实验。

    Raises:
        AssertionError: 核心单元不唯一。

    Results:
        dict[str,Any]: 仅供测试的异构命名空间；不是个人核心 fallback。
    """
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    for path, tags in [
        (Path("docs/learning/S2_第3周_图标数据与潜空间DiT基础.ipynb"), ["dit_reference"]),
        (
            Path("docs/learning/S3_文本条件CFG与可信评测.ipynb"),
            ["conditional_reference", "cfg_reference", "flow_reference"],
        ),
    ]:
        notebook = json.loads(path.read_text())
        for tag in tags:
            found = [c for c in notebook["cells"] if tag in c.get("metadata", {}).get("tags", [])]
            assert len(found) == 1
            exec(compile("".join(found[0]["source"]), str(path), "exec"), namespace)
    return namespace


class FakeTokenizer:
    """只用于验证分段保存信息，不冒充预训练文本语义。"""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        """将字符映射到小型整数表。

        Args:
            texts: 字符串列表。
            add_special_tokens: 测试接口要求为 False。
            truncation: 测试接口要求为 False。

        Raises:
            AssertionError: 被调用者开启隐式特殊符号或截断。

        Results:
            dict: input_ids 为每个字符的确定映射。
        """
        assert not add_special_tokens and not truncation
        return {"input_ids": [[3 + ord(c) % 10 for c in text] for text in texts]}


class FakeTextModel(nn.Module):
    """有限上下文的纯接口替身；不用于生成能力实验。"""

    def __init__(self) -> None:
        """建立小嵌入和上下文上限。

        副作用:
            创建随机可训练参数；适配器随后冻结它。
        """
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=5, hidden_size=4)
        self.embedding = nn.Embedding(16, 4)

    def forward(
        self,
        input_ids: Int[Tensor, "b n"],
        attention_mask: Int[Tensor, "b n"],
    ) -> SimpleNamespace:
        """返回逐 token 嵌入，测试不使用注意力算法。

        Args:
            input_ids: [b,n] 词表索引。
            attention_mask: [b,n] 掩码；替身只验证形状。

        Raises:
            AssertionError: 两个输入形状不一致。

        Results:
            SimpleNamespace: last_hidden_state 为 [b,n,4]。
        """
        assert input_ids.shape == attention_mask.shape
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_text_chunking_mask_and_cache(
    tmp_path: Path,
) -> None:
    """验证空文本、完整分段、过长拒绝、缓存重载和身份/载荷拒绝。

    Args:
        tmp_path: pytest 独立临时目录。

    Raises:
        AssertionError: 形状、mask、冻结或缓存保证不成立。

    副作用:
        写入并主动损坏临时缓存，构造随机测试参数。
    """
    encoder = FrozenTextEncoder(
        FakeTextModel(), FakeTokenizer(), {"revision": "fake"}, max_chunks=2
    )
    values, masks, lengths = encoder.encode(["", "abcde"])
    assert values.shape == (2, 10, 4) and lengths == [0, 5]
    assert masks.sum(-1).tolist() == [2, 9]
    assert not any(p.requires_grad for p in encoder.model.parameters())
    with pytest.raises(ValueError, match="capacity"):
        encoder.encode(["abcdefg"])
    first = text_cache(encoder, ["abcde", ""], tmp_path)
    again = text_cache(encoder, ["abcde", ""], tmp_path)
    torch.testing.assert_close(first[0], again[0])
    assert first[2] == again[2]
    with pytest.raises(ValueError, match="Incompatible"):
        text_cache(encoder, ["other"], tmp_path)
    payload = tmp_path / "tokens.npz"
    payload.write_bytes(payload.read_bytes() + b"damage")
    with pytest.raises(ValueError, match="corrupted"):
        text_cache(encoder, ["abcde", ""], tmp_path)


def test_cross_attention_and_gradient(
    reference: dict[str, Any],
) -> None:
    """检查 masked key 不影响输出，零输出初始化与多步文本梯度。

    Args:
        reference: 显式教学类。

    Raises:
        AssertionError: mask、输出或梯度与设计不符。

    副作用:
        创建随机输入并执行五步小型优化。
    """
    torch.manual_seed(13)
    cross = reference["TextCrossAttentionReference"](8, 6, 2)
    x, text = torch.randn(2, 3, 8), torch.randn(2, 4, 6)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    changed = text.clone()
    changed[~mask] = 10000
    q = cross.q(x).reshape(2, 3, 2, 4).transpose(1, 2)
    k, v = cross.kv(text).reshape(2, 4, 2, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    independent = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :]
    )
    expected = cross.out(independent.transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(cross(x, text, mask), expected)
    torch.testing.assert_close(cross(x, text, mask), cross(x, changed, mask))
    with pytest.raises(ValueError):
        cross(x, text, torch.zeros_like(mask))
    model = reference["ConditionalDiTReference"](depth=1, hidden=16, heads=2, size=4, text_width=6)
    z, t = torch.randn(2, 4, 4, 4), torch.tensor([1, 2])
    assert model(z, t, text, mask).count_nonzero() == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        (model(z, t, text, mask) - z).square().mean().backward()
        optimizer.step()
    assert model.cross[0].kv.weight.grad.abs().sum() > 0


def test_cfg_drop_and_flow_contract(
    reference: dict[str, Any],
) -> None:
    """独立检查 CFG 手算值、整条丢弃与高斯速度积分误差。

    Args:
        reference: Notebook 唯一数学核心。

    Raises:
        AssertionError: 数值边界或误差收敛不符。

    副作用:
        消耗测试随机状态。
    """
    u, c = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 1.0])
    cfg = reference["cfg_reference"]
    torch.testing.assert_close(cfg(u, c, 0), u)
    torch.testing.assert_close(cfg(u, c, 1), c)
    torch.testing.assert_close(cfg(u, c, 3), torch.tensor([7.0, -1.0]))
    values, mask = torch.ones(4, 3, 2), torch.ones(4, 3, dtype=torch.bool)
    empty, empty_mask = torch.zeros(1, 3, 2), torch.tensor([[1, 1, 0]], dtype=torch.bool)
    dropped, dropped_mask = reference["drop_condition_reference"](
        values, mask, empty, empty_mask, 1
    )
    assert dropped.count_nonzero() == 0
    torch.testing.assert_close(dropped_mask, empty_mask.expand_as(mask))
    velocity = reference["gaussian_velocity_reference"]
    torch.testing.assert_close(
        velocity(torch.tensor([[1.5, 0.0]]), torch.tensor([[0.5]])), torch.tensor([[1.4, -1.6]])
    )
    initial = torch.tensor([[1.0, -0.5]])
    exact = torch.tensor([[2.0, -1.0]]) + 0.5 * initial
    errors = [
        float((reference["euler_reference"](velocity, initial, s)[0] - exact).square().sum())
        for s in (10, 20, 40)
    ]
    assert errors[2] < errors[1] < errors[0]


class TinyConditional(nn.Module):
    """工程测试网络，线性 token 均值影响每个像素。"""

    def __init__(self) -> None:
        """建立参数，不冒充 DiT。

        副作用:
            创建随机测试参数。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.projection = nn.Linear(3, 1)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Int[Tensor, " b"],
        text: Float[Tensor, "b n 3"],
        mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, "b c h w"]:
        """用带 mask 文本均值生成同形预测。

        Args:
            z: 空间状态。
            t: 时间索引，替身不依赖它。
            text: 三维 token 特征。
            mask: 有效位。

        Results:
            Float[Tensor,"b c h w"]: 同形噪声预测。
        """
        pooled = (text * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.weight * z + self.projection(pooled)[:, :, None, None]


@pytest.mark.parametrize("scale,branches", [(0.0, 1), (1.0, 1), (3.0, 2)])
def test_sampler_counts_and_mode_restore(
    reference: dict[str, Any],
    scale: float,
    branches: int,
) -> None:
    """检查完整 DDIM、双分支合批计数以及训练模式恢复。

    Args:
        reference: 显式 CFG 教学核心。
        scale: 测试引导权重。
        branches: 每步等价分支求值数。

    Raises:
        AssertionError: 完成、计数或模式不符。

    副作用:
        构造测试网络并执行短 CPU 采样。
    """
    model = TinyConditional().train()
    result = sample_conditional(
        model,
        DDPMScheduler(10),
        torch.zeros(2, 1, 2, 2),
        torch.ones(2, 3, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(1, 3, 3),
        torch.ones(1, 3, dtype=torch.bool),
        reference["cfg_reference"],
        method="ddim",
        steps=4,
        scale=scale,
    )
    assert model.training and result["completed"]
    assert result["model_calls"] == 4 and result["effective_nfe"] == 4 * branches
    assert result["branch_examples"] == 8 * branches


def test_conditional_resume_and_contract_rejection(
    reference: dict[str, Any],
    tmp_path: Path,
) -> None:
    """连续四步对照二步后恢复，比较模型/优化器/RNG/数据状态，并拒绝条件变化。

    Args:
        reference: 显式条件丢弃函数。
        tmp_path: 临时训练目录。

    Raises:
        AssertionError: 同环境恢复不精确。

    副作用:
        执行三次极小 CPU 训练并写临时权重。
    """
    torch.manual_seed(4)
    data, text_ids = torch.randn(7, 1, 2, 2), torch.tensor([0, 1, 0, 1, 0, 1, 0])
    features, masks = torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool)
    identity = {"drop_source_sha256": "0" * 64}
    model_id = {"source": "test", "source_sha256": "1" * 64, "architecture": {"test": True}}
    initial = TinyConditional().state_dict()
    saved = []
    for name, steps, resume in [
        ("full", 4, None),
        ("part", 2, None),
        ("resumed", 4, str(tmp_path / "part/latest.pt")),
    ]:
        model = TinyConditional()
        model.load_state_dict(initial)
        result = train_conditional(
            model,
            DDPMScheduler(10),
            data,
            text_ids,
            features,
            masks,
            0,
            reference["drop_condition_reference"],
            TrainConfig(
                str(tmp_path / name),
                max_steps=steps,
                seed=3,
                device="cpu",
                scheduler_source="project_core",
            ),
            identity=identity,
            model_identity=model_id,
            batch_size=3,
            resume_path=resume,
        )
        saved.append(torch.load(result["checkpoint"], weights_only=True))
    for key in saved[0]["model"]:
        torch.testing.assert_close(saved[0]["model"][key], saved[2]["model"][key], rtol=0, atol=0)
    for key in saved[0]["ema"]:
        torch.testing.assert_close(saved[0]["ema"][key], saved[2]["ema"][key], rtol=0, atol=0)
    for key, state in saved[0]["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, saved[2]["optimizer"]["state"][key][field], rtol=0, atol=0
            )
    assert saved[0]["rng"]["python"] == saved[2]["rng"]["python"]
    assert saved[0]["rng"]["numpy"] == saved[2]["rng"]["numpy"]
    torch.testing.assert_close(saved[0]["rng"]["torch"], saved[2]["rng"]["torch"], rtol=0, atol=0)
    assert saved[0]["data_state"]["cursor"] == saved[2]["data_state"]["cursor"]
    torch.testing.assert_close(
        saved[0]["data_state"]["order"], saved[2]["data_state"]["order"], rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="contract changed"):
        train_conditional(
            TinyConditional(),
            DDPMScheduler(10),
            data,
            text_ids,
            features,
            masks,
            0,
            reference["drop_condition_reference"],
            TrainConfig(
                str(tmp_path / "bad"), max_steps=4, seed=3, scheduler_source="project_core"
            ),
            identity=identity,
            model_identity=model_id,
            batch_size=3,
            drop_probability=0.2,
            resume_path=str(tmp_path / "part/latest.pt"),
        )


def test_scoring_hidden_positions_binding_and_failure() -> None:
    """验证不使用隐藏中心、绑定错误不能得综合成功、失败保持分母。

    Raises:
        AssertionError: 评分或置换违反请求口径。
    """
    objects = [
        {"color": "red", "shape": "circle", "center": [25, 40], "bbox": [10, 25, 40, 55]},
        {"color": "yellow", "shape": "square", "center": [90, 40], "bbox": [75, 25, 105, 55]},
    ]
    record = {
        "attributes": {
            "objects": objects,
            "relations": [{"subject": 0, "predicate": "left of", "object": 1}],
        }
    }
    image = render(objects, 128)
    assert score_request(image, record)["all"] == 1
    # 隐藏中心写错不应影响请求评分；bbox 只用于绘制原图，不用于生成评分。
    record["attributes"]["objects"][0]["center"] = [999, 999]
    assert score_request(image, record)["all"] == 1
    wrong = json.loads(json.dumps(objects))
    wrong[0]["shape"], wrong[1]["shape"] = "square", "circle"
    score = score_request(render(wrong, 128), record)
    assert score["color"] == score["shape"] == 1 and score["binding"] == score["all"] == 0
    from PIL import Image

    blank = score_request(Image.new("RGB", (128, 128), "white"), record)
    summary = summarize([score_request(image, record), blank])
    assert summary["images"] == 2 and summary["all"] == summary["parse_failure"] == 0.5
    different = json.loads(json.dumps(record))
    different["attributes"]["objects"][0]["color"] = "green"
    assert shuffled_indices([record, different]) == [1, 0]
    with pytest.raises(ValueError):
        shuffled_indices([record, record])
    difference = paired_difference([1, 0], [0, 0])
    assert difference["difference"] == 0.5 and difference["pairs"] == 2


class OracleNoise(nn.Module):
    """已知常量干净值的解析噪声 oracle，只检验采样更新。"""

    def __init__(self) -> None:
        """注册独立测试日程。

        副作用:
            创建固定日程缓冲区。
        """
        super().__init__()
        self.schedule = DDPMScheduler(10)

    def forward(
        self,
        z: Float[Tensor, "b c h w"],
        t: Int[Tensor, " b"],
        text: Float[Tensor, "b n 3"],
        mask: Bool[Tensor, "b n"],
    ) -> Float[Tensor, "b c h w"]:
        """解析出当前状态相对固定 clean=0.25 的噪声。

        Args:
            z: 当前状态。
            t: 日程索引。
            text: 占位条件，oracle 不使用。
            mask: 占位 mask。

        Results:
            Float[Tensor,"b c h w"]: 使干净估计为 0.25 的解析预测；不是学到的生成能力。
        """
        # 从同一累计系数计算根号，避免两份预舍入表的微小差异被逆除法放大。
        a = self.schedule.alpha_bars[t, None, None, None]
        signal, scale = a.sqrt(), (1 - a).sqrt()
        return (z - signal * 0.25) / scale


@pytest.mark.parametrize("method,steps", [("ddim", 3), ("ddpm", 10)])
def test_sampling_endpoint_against_analytic_clean(
    reference: dict[str, Any],
    method: str,
    steps: int,
) -> None:
    """用解析噪声检查完整端点，防止将索引 0 错当成干净系数。

    Args:
        reference: 显式 CFG 函数。
        method: 采样方法。
        steps: 完整更新次数。

    Raises:
        AssertionError: 端点未恢复独立已知的常量 clean。

    副作用:
        构造测试噪声和模型，执行短采样。
    """
    result = sample_conditional(
        OracleNoise(),
        DDPMScheduler(10),
        torch.randn(2, 1, 2, 2),
        torch.ones(2, 3, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(1, 3, 3),
        torch.ones(1, 3, dtype=torch.bool),
        reference["cfg_reference"],
        method=method,
        steps=steps,
    )
    torch.testing.assert_close(result["images"], torch.full((2, 1, 2, 2), 0.25), rtol=0, atol=2e-6)
    assert result["completed"]


def test_sampling_timeout_and_exception_restore(
    reference: dict[str, Any],
) -> None:
    """超时不能冒充最终样本，模型异常也须恢复原训练模式。

    Args:
        reference: 显式 CFG 函数。

    Raises:
        AssertionError: 完成状态、拒绝规则或模式恢复错误。

    副作用:
        构造小模型并执行有界/故障采样。
    """
    arguments = [
        DDPMScheduler(10),
        torch.zeros(2, 1, 2, 2),
        torch.ones(2, 3, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(1, 3, 3),
        torch.ones(1, 3, dtype=torch.bool),
        reference["cfg_reference"],
    ]
    timed = sample_conditional(
        TinyConditional(), *arguments, method="ddim", steps=3, max_seconds=1e-12
    )
    assert not timed["completed"] and timed["completed_steps"] == 0
    bad_model = nn.Identity().train()
    with pytest.raises(TypeError):
        sample_conditional(bad_model, *arguments, method="ddim", steps=3)
    assert bad_model.training
    with pytest.raises(ValueError, match="every step"):
        sample_conditional(TinyConditional(), *arguments, method="ddpm", steps=3)


def test_review_gate_rejects_missing_duplicate_and_disagreement() -> None:
    """检查人工门槛的空标签、重复标识与低一致性拒绝；标签仅为测试 fixture。

    Raises:
        AssertionError: 缺证据被接受或完整测试记录被错误拒绝。

    副作用:
        动态导入只含定义的门槛脚本，不执行其 main 或测试采样。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("s3_gate", "scripts/evaluate_stage3_holdout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.review_gate([])[0]
    rows = []
    score = {
        "count": 1,
        "color": 1,
        "shape": 1,
        "binding": 1,
        "all": 1,
        "parse_failure": 0,
        "has_relation": 1,
        "relation_all": 1,
    }
    for kind in ("original", "degraded"):
        for i in range(100):
            rows.append(
                {
                    "review_id": f"{kind}_{i}",
                    "parser": score.copy(),
                    "human": score.copy(),
                    "reviewer": "unit_test_fixture_only",
                }
            )
    assert module.review_gate(rows)[0]
    assert not module.review_gate(rows + [rows[0]])[0]
    for r in rows[:10]:
        r["human"]["all"] = 0
    assert not module.review_gate(rows)[0]


def test_blind_review_hides_results_and_preserves_labels(
    tmp_path: Path,
) -> None:
    """匿名审核不暴露来源/机器结果；重新准备不能覆盖已填写的人类标签。

    Args:
        tmp_path: 独立临时目录。

    Raises:
        AssertionError: 匿名化、合并或已填保护不符合约定。

    副作用:
        创建小型假图像和临时人工审核记录，仅用于工程测试。
    """
    from PIL import Image

    from minigen.evaluation.conditional import merge_human_review, prepare_blind_review

    Image.new("RGB", (8, 8), "white").save(tmp_path / "source.png")
    source = [
        {
            "review_id": "generated_0_correct_000",
            "image_path": "source.png",
            "request": "test fixture",
            "parser": {"all": 1},
            "human": None,
            "reviewer": None,
        }
    ]
    path = prepare_blind_review(source, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert "parser" not in rows[0] and "correct" not in rows[0]["review_id"]
    rows[0]["human"], rows[0]["reviewer"] = {"all": 0}, "test_fixture_not_real_review"
    path.write_text(json.dumps(rows[0]) + "\n")
    before = path.read_bytes()
    assert prepare_blind_review(source, tmp_path) == path and path.read_bytes() == before
    merged = merge_human_review(path, tmp_path / "review_key.json")
    assert merged[0]["human"]["all"] == 0 and merged[0]["parser"]["all"] == 1


def test_blind_review_rejects_request_and_image_changes(
    tmp_path: Path,
) -> None:
    """匿名审核若请求或图像变化必须拒绝合并，不能评错另一批数据。

    Args:
        tmp_path: 临时资源路径。

    Raises:
        AssertionError: 身份变化未被识别。

    副作用:
        写入并改变临时审核图像及 JSONL。
    """
    from PIL import Image

    from minigen.evaluation.conditional import merge_human_review, prepare_blind_review

    Image.new("RGB", (8, 8), "white").save(tmp_path / "source.png")
    source = [
        {
            "review_id": "original_000",
            "image_path": "source.png",
            "request": "fixture",
            "parser": {"all": 1},
        }
    ]
    path = prepare_blind_review(source, tmp_path)
    row = json.loads(path.read_text())
    row["request"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="identity changed"):
        merge_human_review(path, tmp_path / "review_key.json")


@pytest.fixture
def assets_module(
    monkeypatch: pytest.MonkeyPatch,
) -> "ModuleType":
    """加载显式准备脚本的定义，测试不访问远程资源。

    Args:
        monkeypatch: pytest 路径/函数替换工具。

    Raises:
        ImportError: 脚本缺必需依赖时透传；pypdf 未安装则该组跳过。

    Results:
        ModuleType: 独立脚本模块，ROOT 和网络函数由具体测试替换。

    副作用:
        临时增加 scripts 的导入路径，加载脚本定义。
    """
    import importlib.util

    pytest.importorskip("pypdf")
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "s3_assets_check", "scripts/prepare_stage3_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_asset_lock(
    module: "ModuleType",
    root: Path,
    payloads: dict[str, bytes],
) -> Path:
    """写固定身份的临时测试版本，模拟 Git 带来身份但没有忽略目录。

    Args:
        module: 提供 repo/revision/upstream 常量的脚本模块。
        root: 独立临时项目根。
        payloads: 期望文件的内容，函数只登记哈希、不创建这些文件。

    Results:
        Path: 已写入的模型身份文件。

    副作用:
        在临时目录写 configs/models/clip_text.json。
    """
    import hashlib

    lock = root / "configs/models/clip_text.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "repo_id": module.REPO,
                "revision": module.REVISION,
                "upstream_sha256": module.UPSTREAM_HASH,
                "local_path": f"artifacts/models/clip_text/{module.REVISION}",
                "files": {
                    name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in payloads.items()
                },
            }
        )
    )
    return lock


def test_asset_lock_restores_missing_tokenizer_without_weights(
    assets_module: "ModuleType",
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已有 lock 的缺失小文件应按固定版本恢复，而非 FileNotFound 或整份权重重下。

    Args:
        assets_module: 独立资源脚本。
        tmp_path: 临时项目路径。
        monkeypatch: 替换 ROOT 与下载函数。

    Raises:
        AssertionError: 资源或冻结身份不符；测试网络由本地替身阻断。

    副作用:
        写临时身份和小型假 tokenizer 文件，不下载真实资源。
    """
    payloads = {"tokenizer.json": b'{"fixture": true}'}
    lock = write_asset_lock(assets_module, tmp_path, payloads)
    before = lock.read_bytes()
    monkeypatch.setattr(assets_module, "ROOT", tmp_path)
    calls = []

    def fetch_fixture(
        url: str,
        path: Path,
    ) -> None:
        """用本地固定内容代替网络下载。

        Args:
            url: 固定公开资源 URL，用于验证 revision。
            path: 临时目标路径。

        副作用:
            记录调用并写入测试字节。
        """
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[path.name])

    monkeypatch.setattr(assets_module, "fetch_resource", fetch_fixture)
    assets_module.text_model()
    assert lock.read_bytes() == before and len(calls) == 1
    assert assets_module.REVISION in calls[0] and calls[0].endswith("tokenizer.json")


def test_asset_lock_with_absent_converted_weights_enters_download(
    assets_module: "ModuleType",
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新克隆有 lock 无权重时走显式上游下载；不直接对缺失文件计算哈希。

    Args:
        assets_module: 资源脚本。
        tmp_path: 临时项目根。
        monkeypatch: 替换路径和下载。

    Raises:
        AssertionError: 未进入固定资源下载，或冻结身份被改写。

    副作用:
        建立临时配置；显式下载在本地替身入口被终止，不访问网络。
    """
    pytest.importorskip("transformers")
    lock = write_asset_lock(assets_module, tmp_path, {"config.json": b"fixture"})
    before = lock.read_bytes()
    monkeypatch.setattr(assets_module, "ROOT", tmp_path)

    def stop_download(
        url: str,
        path: Path,
    ) -> None:
        """确认公开固定权重入口后阻断测试下载。

        Args:
            url: 需要包含固定 revision 的 URL。
            path: 临时 upstream.bin 路径。

        Raises:
            AssertionError: 不是预期权重入口。
            RuntimeError: 测试主动阻断下载，证明已到达正确分支。
        """
        assert assets_module.REVISION in url and url.endswith("pytorch_model.bin")
        assert path.name == "upstream.bin"
        raise RuntimeError("expected explicit upstream retrieval")

    monkeypatch.setattr(assets_module, "fetch", stop_download)
    with pytest.raises(RuntimeError, match="expected explicit upstream retrieval"):
        assets_module.text_model()
    assert lock.read_bytes() == before


def test_asset_lock_rejects_present_corruption(
    assets_module: "ModuleType",
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已存在但哈希错误的文件应拒绝，不能把错误内容重新登记为期望值。

    Args:
        assets_module: 独立准备脚本。
        tmp_path: 临时根目录。
        monkeypatch: 修改 ROOT。

    Raises:
        AssertionError: 损坏未被拒绝或 lock 被覆盖。

    副作用:
        写临时损坏文件，不下载任何内容。
    """
    lock = write_asset_lock(assets_module, tmp_path, {"tokenizer.json": b"expected"})
    before = lock.read_bytes()
    folder = tmp_path / f"artifacts/models/clip_text/{assets_module.REVISION}"
    folder.mkdir(parents=True)
    (folder / "tokenizer.json").write_bytes(b"broken")
    monkeypatch.setattr(assets_module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="file mismatch"):
        assets_module.text_model()
    assert lock.read_bytes() == before
