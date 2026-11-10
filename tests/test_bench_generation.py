"""任务初噪、求值预算与可恢复逐图生成记录的独立测试。"""

import json
from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from minigen._typing import JsonDict
from minigen.evaluation import benchmark, generation, profiling
from minigen.evaluation.generation import (
    execute_tasks,
    initial_noise,
    integration_steps,
    make_tasks,
)
from minigen.evaluation.profiling import sampling_matrix


def fixture_tasks() -> list[JsonDict]:
    """生成两个语义配对提示、两个种子的四项任务。

    Results:
        list: 明确标记的合成测试任务。
    """
    benchmark = {
        "prompts": [
            {
                "prompt_id": f"v{i}",
                "split": "validation",
                "caption": caption,
                "seeds": [7, 19],
                "requirements": [],
            }
            for i, caption in enumerate(["a tree", "one tree"])
        ]
    }
    return make_tasks(benchmark, "validation", None)


def fixed_renderer(
    tasks: list[JsonDict],
) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
    """将固定初噪压到图像范围，仅验证持久化协议。

    Args:
        tasks: 合成测试任务。

    Results:
        tuple: [b,3,4,4] 合成图与显式模拟标记，不是模型输出。
    """
    return initial_noise(tasks, (3, 4, 4)).sigmoid(), {
        "scope": "simulated",
        "solver_nfe": 1,
        "model_evaluations": 1,
    }


def test_noise_and_nfe_are_independent_of_batch_order() -> None:
    """重排、分批、恢复不能改变给定 seed 的 CPU 初噪。

    Raises:
        AssertionError: 初噪或 Euler/Heun/CFG 对照定义错误。
    """
    tasks = fixture_tasks()
    noise = initial_noise(tasks, (4, 3, 3))
    torch.testing.assert_close(
        noise,
        torch.cat([initial_noise(tasks[:1], (4, 3, 3)), initial_noise(tasks[1:], (4, 3, 3))]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(initial_noise(tasks[::-1], (4, 3, 3)), noise.flip(0), atol=0, rtol=0)
    torch.testing.assert_close(noise[0], noise[2], atol=0, rtol=0)
    assert integration_steps("heun", 10) == 5 and integration_steps("euler", 10) == 10
    with pytest.raises(ValueError, match="even"):
        integration_steps("heun", 3)
    settings = sampling_matrix()
    assert len(settings) == 9
    assert {(r["method"], r["solver_nfe"]) for r in settings[:6]} == {
        (m, n) for m in ["euler", "heun"] for n in [10, 20, 40]
    }


def test_resume_keeps_failures_and_checks_identity_and_image_hash(
    tmp_path: Path,
) -> None:
    """部分失败不消失；明确重试后恢复，并拒绝篡改图片和错误身份。

    Args:
        tmp_path: 隔离输出目录。

    Raises:
        AssertionError: 分母、状态或校验边界错误。

    副作用:
        写合成图、失败和恢复回执；不加载外部权重。
    """
    tasks = fixture_tasks()
    identity = {"scope": "simulated", "checkpoint": "fixture"}
    calls = 0

    def interrupted(
        batch: list[JsonDict],
    ) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
        """使第二批产生可记录的设备异常。

        Args:
            batch: 合成任务批。

        Raises:
            RuntimeError: 第二次调用模拟设备失败。

        Results:
            tuple: 模拟图像及运行记录。

        副作用:
            更新外层调用计数。
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated device error")
        return fixed_renderer(batch)

    report = execute_tasks(tasks, identity, tmp_path, 2, interrupted)
    assert (
        report["completed_samples"] == 2
        and report["failed_samples"] == 2
        and not report["complete"]
    )
    unchanged = execute_tasks(tasks, identity, tmp_path, 1, fixed_renderer, resume=True)
    assert unchanged["failed_samples"] == 2
    completed = execute_tasks(
        tasks, identity, tmp_path, 1, fixed_renderer, resume=True, retry_failed=True
    )
    assert completed["complete"] and completed["expected_samples"] == 4
    hashes = [r["sha256"] for r in completed["records"]]
    repeated = execute_tasks(tasks, identity, tmp_path, 4, fixed_renderer, resume=True)
    assert [r["sha256"] for r in repeated["records"]] == hashes
    with pytest.raises(FileExistsError):
        execute_tasks(tasks, identity, tmp_path, 2, fixed_renderer)
    with pytest.raises(ValueError, match="identity"):
        execute_tasks(
            tasks, {**identity, "checkpoint": "other"}, tmp_path, 2, fixed_renderer, resume=True
        )
    (tmp_path / completed["records"][0]["image_path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="image"):
        execute_tasks(tasks, identity, tmp_path, 2, fixed_renderer, resume=True)


def test_interrupt_publishes_pending_receipt_and_can_resume(
    tmp_path: Path,
) -> None:
    """模拟进程级中断，已完成批保持，剩余任务作为 pending 再启动。

    Args:
        tmp_path: 隔离目录。

    Raises:
        AssertionError: 中断后清单丢失或完成项重复执行。

    副作用:
        产生部分回执并恢复全部合成任务。
    """
    tasks = fixture_tasks()
    calls = 0

    def interrupt(
        batch: list[JsonDict],
    ) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
        """第二批直接中断，模拟 SIGINT。

        Args:
            batch: 本次任务。

        Raises:
            KeyboardInterrupt: 第二次调用。

        Results:
            tuple: 第一批的合成结果。

        副作用:
            更新调用数。
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return fixed_renderer(batch)

    identity = {"scope": "simulated"}
    with pytest.raises(KeyboardInterrupt):
        execute_tasks(tasks, identity, tmp_path, 2, interrupt)
    receipt = json.loads((tmp_path / "samples.json").read_text())
    assert [r["status"] for r in receipt["records"]] == [
        "completed",
        "completed",
        "pending",
        "pending",
    ]
    final = execute_tasks(tasks, identity, tmp_path, 1, fixed_renderer, resume=True)
    assert final["complete"] and len(final["batches"]) == 3


def test_formal_generation_checks_approval_before_loading_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式入口先要求完整任务与真实人工批准，拒绝后不加载资产。

    Args:
        tmp_path: 临时根目录。
        monkeypatch: 替换协议检查以隔离外部资源。

    Raises:
        AssertionError: 缺少人工批准仍可执行或批准参数未传递。

    副作用:
        仅在测试内替换协议检查函数，不进行模型加载。
    """

    def denied(
        path: Path,
        require_approved: bool = False,
        approval: Path | None = None,
    ) -> JsonDict:
        """拒绝尚未发生的人工审批，并检查调用参数。

        Args:
            path: 题目路径。
            require_approved: 正式生成必须为真。
            approval: 本案例没有批准文件。

        Raises:
            AssertionError: 参数与正式入口不符。
            ValueError: 人工批准尚未发生。

        """
        assert path == tmp_path / "benchmark.json"
        assert require_approved and approval is None
        raise ValueError("Human approval pending")

    monkeypatch.setattr(benchmark, "load_benchmark", denied)
    arguments = {
        "root": tmp_path,
        "config": Path("config.json"),
        "checkpoint": Path("missing.pt"),
        "benchmark": Path("benchmark.json"),
        "output": Path("results"),
        "scope": "formal",
    }
    with pytest.raises(ValueError, match="full tasks"):
        generation.run_generation(**arguments, count=1)
    with pytest.raises(ValueError, match="approval pending"):
        generation.run_generation(**arguments)
    with pytest.raises(ValueError, match="validation only"):
        generation.run_generation(**{**arguments, "scope": "real_local", "split": "test"})
    assert not (tmp_path / "results").exists()


def test_profile_records_real_calls_and_keeps_cuda_fields_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用显式模拟适配器验证预热、重复、求值记录与设备内存边界。

    Args:
        tmp_path: 隔离结果目录。
        monkeypatch: 替换加载和推理，避免测试下载模型。

    Raises:
        AssertionError: 预热进入计时、重复缺失或 CPU 被当作 CUDA。

    副作用:
        写带模拟元数据的性能结果，仅验证控制流程。
    """
    definition = {
        "manifest_sha256": "fixture_manifest",
        "prompts": [
            {
                "prompt_id": "profile_fixture",
                "caption": "a tree",
                "split": "validation",
                "seeds": [7, 19],
                "requirements": [],
            }
        ],
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(definition))
    (tmp_path / "matrix.json").write_text(json.dumps(profiling.sampling_matrix()[:1]))
    model = torch.nn.Linear(1, 1)
    model.attention_backend = "sdpa"
    session = generation.GenerationSession(
        model=model,
        text=None,
        vae=None,
        blank=None,
        metadata={
            "scope": "simulated",
            "checkpoint": "fixture",
            "data_identity": {"source": {"manifest_sha256": "fixture_manifest"}},
        },
    )
    calls = 0

    def read_definition(
        path: Path,
    ) -> JsonDict:
        """读取测试协议。

        Args:
            path: 本地 JSON。

        Results:
            JsonDict: 合成题目。
        """
        return json.loads(path.read_text())

    def load_fixture(
        root: Path,
        config: Path,
        checkpoint: Path,
        device: str,
        weights: str,
        attention_backend: str,
    ) -> generation.GenerationSession:
        """返回只用于测试调度的组件占位。

        Args:
            root: 隔离根。
            config: 不读取的占位配置路径。
            checkpoint: 不读取的占位权重路径。
            device: 要求的设备。
            weights: 权重选择。
            attention_backend: 初始执行后端。

        Results:
            GenerationSession: 显式带 simulated 身份的测试会话。
        """
        return session

    def render_fixture(
        current: generation.GenerationSession,
        tasks: list[JsonDict],
        method: str,
        solver_nfe: int,
        guidance: float,
    ) -> tuple[Float[Tensor, "b 3 h w"], JsonDict]:
        """生成确定性计时样例，只检查保存的计数而不测真实性能。

        Args:
            current: 测试会话。
            tasks: 固定题目与种子。
            method: 积分方法。
            solver_nfe: 求值预算。
            guidance: CFG 系数。

        Results:
            tuple: 合成图与带模拟标记的时长。

        副作用:
            增加测试调用计数。
        """
        nonlocal calls
        calls += 1
        return fixed_renderer(tasks)[0], {
            "scope": "simulated",
            "denoise_seconds": float(calls),
            "pipeline_seconds": float(calls) + 1,
            "solver_nfe": solver_nfe,
            "model_evaluations": solver_nfe,
        }

    monkeypatch.setattr(benchmark, "load_benchmark", read_definition)
    monkeypatch.setattr(profiling, "load_session", load_fixture)
    monkeypatch.setattr(profiling, "render_batch", render_fixture)
    result = profiling.run_profile(
        tmp_path,
        Path("config.json"),
        Path("checkpoint.pt"),
        Path("benchmark.json"),
        Path("profile.json"),
        warmup=1,
        repeats=2,
        matrix=Path("matrix.json"),
    )
    assert calls == 3 and result["complete"] and result["scope"] == "simulated"
    row = result["results"][0]
    assert len(row["measurements"]) == 2 and row["denoise_seconds_mean"] == 2.5
    assert row["images_per_repeat"] == 2 and row["cuda_peak_allocated_bytes"] is None
    assert row["mps_peak_allocated_bytes"] is None
    assert all(value["scope"] == "simulated" for value in row["measurements"])
