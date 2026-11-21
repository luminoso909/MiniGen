"""CPU 回读六组 30k 训练、编号快照、恢复状态和数据配对。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

import minigen

RUN = Path("artifacts/runs/anime_faces_256/six_30k_v1")
STEPS = (3000, 5000, 7000, 10000, 13000, 15000, 17000, 20000, 23000, 25000, 27000, 30000)
PARAMETERS = {
    "b1_30m_p2": 30061456,
    "b2_30m_p2": 30061456,
    "b1_50m_p2": 48681232,
    "b2_50m_p2": 48681232,
    "b1_50m_p1": 48672004,
    "b2_50m_p1": 48672004,
}


def read_json(
    path: Path,
) -> dict[str, Any]:
    """读取一个已发布 JSON 回执。

    Args:
        path: 要读取的运行或结果 JSON 路径。

    Raises:
        OSError/ValueError: 文件读取或 JSON 解码失败。

    Results:
        dict: 完整回执对象。
    """
    return json.loads(path.read_text())


def file_sha256(
    path: Path,
) -> str:
    """计算完整文件的内容哈希。

    Args:
        path: 检查点文件路径。

    Raises:
        OSError: 读取文件失败。

    Results:
        str: 文件内容的 SHA-256。
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(
    contract: dict[str, Any],
) -> str:
    """按训练引擎的序列化规则复算恢复契约摘要。

    Args:
        contract: 运行的完整恢复契约。

    Raises:
        ValueError/TypeError: 契约不能序列化为有限 JSON。

    Results:
        str: 与 run_id 对应的 SHA-256。
    """
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def same_stream_and_rng(
    first: dict[str, Any],
    second: dict[str, Any],
    label: str,
) -> None:
    """在不同架构的同一步比较样本顺序与随机状态。

    Args:
        first: 参考检查点，包含 stream 和 rng 字段。
        second: 待比较的检查点。
        label: 断言失败时显示的组名和步数。

    Raises:
        AssertionError: 任一数据顺序、游标或随机状态不同。
        KeyError: 检查点缺少所需状态。
    """
    for key in ("version", "shuffle", "size", "batch_size", "epoch", "cursor"):
        assert first["stream"][key] == second["stream"][key], (label, "stream", key)
    for key in ("order", "generator"):
        assert torch.equal(first["stream"][key], second["stream"][key]), (label, "stream", key)
    for key in ("python", "numpy"):
        assert first["rng"][key] == second["rng"][key], (label, "rng", key)
    for key in ("cpu", "accelerator"):
        assert torch.equal(first["rng"][key], second["rng"][key]), (label, "rng", key)


def check_checkpoint(
    path: Path,
    step: int,
    run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """安全映射一个检查点并核对状态、张量规格和累计样本。

    Args:
        path: 使用 weights_only=True 和 CPU mmap 读取的检查点路径。
        step: 预期的累计优化更新数。
        run: 当前运行的 run.json，用于校验身份、配置和状态规格。

    Raises:
        AssertionError: 检查点身份、日志、优化器或张量规格不符合协议。
        OSError/RuntimeError/ValueError/KeyError: 文件加载、状态解析或字段读取失败。

    Results:
        tuple: 检查点状态和验收摘要。
            状态: 保留 CPU mmap 张量供组间配对及最终 last 比对。
            摘要: 文件哈希、字节数、样本数、等价轮数、验证误差和优化器状态数量。
    """
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    label = str(path)
    assert state["schema"] == "minigen.flow-training.v2", label
    assert state["objective"] == "flow_velocity", label
    assert state["step"] == state["scheduler"]["step"] == len(state["history"]) == step, label
    assert state["contract"] == run["contract"], label
    assert state["data_identity"] == run["contract"]["data"], label
    assert json.loads(json.dumps(state["config"])) == run["config"], label
    assert state["stream"]["data_id"] == run["run_id"], label
    history = state["history"]
    assert [row["step"] for row in history] == list(range(1, step + 1)), label
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in history), (
        label
    )
    assert all(math.isfinite(row["val_loss"]) for row in history if "val_loss" in row), label
    assert len([row for row in history if "val_loss" in row]) == step // 1000, label
    rows = run["contract"]["data"]["subsets"]["train"]["count"]
    batch = run["config"]["batch_size"] * run["config"]["grad_accum"]
    epoch_steps = math.ceil(rows / batch)
    tail = rows % batch or batch
    expected_samples = step * batch - (step // epoch_steps) * (batch - tail)
    assert sum(row["samples"] for row in history) == expected_samples, label
    assert all(
        row["samples"] == (tail if row["step"] % epoch_steps == 0 else batch) for row in history
    ), label
    assert {float(entry["step"]) for entry in state["optimizer"]["state"].values()} == {step}, label
    expected_tensors = run["contract"]["tensors"]
    for family in ("model", "ema"):
        actual = {
            key: [list(value.shape), str(value.dtype)] for key, value in state[family].items()
        }
        assert actual == expected_tensors, (label, family)
    return state, {
        "file": path.name,
        "step": step,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "samples_seen": expected_samples,
        "equivalent_epochs": expected_samples / rows,
        "validation_flow_mse": history[-1]["val_loss"],
        "adamw_state_entries": len(state["optimizer"]["state"]),
    }


def main() -> None:
    """核对全部六组，再向标准输出写完整验收 JSON。

    Raises:
        AssertionError: 配方、源码、回执、检查点或组间随机配对不符合冻结协议。
        OSError/RuntimeError/ValueError/KeyError: 回执或检查点读取失败。

    副作用:
        将当前 CPU 线程数设为 2，读取所有检查点，向 stdout 写验收 JSON；不分配 GPU。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    runs = {arm: read_json(root / RUN / arm / "run.json") for arm in PARAMETERS}
    results = {arm: read_json(root / RUN / arm / "result.json") for arm in PARAMETERS}
    reference_run = next(iter(runs.values()))
    source = Path(minigen.__file__).resolve().parent
    source_digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_digest.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
    assert source_digest.hexdigest() == reference_run["contract"]["source_sha256"]
    common = dict(reference_run["config"])
    common.pop("model")
    common.pop("output_dir")
    assert common["seed"] == 42 and common["max_steps"] == 30000
    assert common["schedule_steps"] == 30000 and common["warmup_steps"] == 500
    assert common["learning_rate"] == 1e-4 and common["min_lr_ratio"] == 0.1
    assert common["batch_size"] * common["grad_accum"] == 64
    assert common["precision"] == "bf16" and common["device"] == "cuda"
    assert common["condition_dropout"] == 0.1 and common["ema_decay"] == 0.99
    assert common["checkpoint_every"] == common["validation_every"] == 1000
    assert tuple(common["checkpoint_steps"]) == STEPS
    arms = {}
    for arm, expected_parameters in PARAMETERS.items():
        run, result = runs[arm], results[arm]
        config = dict(run["config"])
        config.pop("model")
        config.pop("output_dir")
        assert config == common, arm
        assert run["contract"]["source_sha256"] == reference_run["contract"]["source_sha256"], arm
        assert run["contract"]["data"] == reference_run["contract"]["data"], arm
        assert result["status"] == "completed" and result["step"] == 30000, arm
        assert result["performance"]["optimizer_steps"] == 30000, arm
        assert result["attempt_id"] == run["attempt_id"], arm
        assert result["run_id"] == run["run_id"] == identity(run["contract"]), arm
        assert result["source_sha256"] == run["contract"]["source_sha256"], arm
        assert (
            result["trainable_parameters"] == run["trainable_parameters"] == expected_parameters
        ), arm
        assert result["train_samples"] == 32892 and result["validation_samples"] == 256, arm
        assert set(path.name for path in (root / RUN / arm).glob("step_*.pt")) == {
            f"step_{step:06d}.pt" for step in STEPS
        }, arm
        arms[arm] = {"result": result, "snapshots": []}
    parity_steps = []
    for step in STEPS:
        states = {}
        for arm in PARAMETERS:
            state, receipt = check_checkpoint(
                root / RUN / arm / f"step_{step:06d}.pt", step, runs[arm]
            )
            states[arm] = state
            arms[arm]["snapshots"].append(receipt)
        reference = next(iter(states.values()))
        for arm, state in states.items():
            same_stream_and_rng(reference, state, f"{arm}/{step}")
        parity_steps.append(step)
        if step == STEPS[-1]:
            for arm, state in states.items():
                folder = root / RUN / arm
                last, receipt = check_checkpoint(folder / "last.pt", step, runs[arm])
                assert last["history"] == state["history"], arm
                assert last["scheduler"] == state["scheduler"], arm
                same_stream_and_rng(state, last, f"{arm}/last")
                for family in ("model", "ema"):
                    assert all(
                        torch.equal(state[family][key], value)
                        for key, value in last[family].items()
                    ), (arm, family)
                metrics = [
                    json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
                ]
                assert metrics == state["history"], arm
                arms[arm]["snapshots"].append(receipt)
                assert results[arm]["performance"]["samples_seen"] == receipt["samples_seen"], arm
                windows = {"first_100": state["history"][:100], "last_100": state["history"][-100:]}
                arms[arm]["loss_windows"] = {
                    key: fmean(row["loss"] for row in window) for key, window in windows.items()
                }
    print(
        json.dumps(
            {
                "status": "verified",
                "six_model_training_recipe": common,
                "same_source_and_data": True,
                "same_data_stream_and_rng_at_steps": parity_steps,
                "last_matches_final_numbered_model_and_ema": True,
                "arms": arms,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
