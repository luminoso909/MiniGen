"""验证环境、CLI 与个人核心的真实行为。"""

import json
import subprocess
import sys

import pytest
import torch

from minigen._learning import CoreImplementationRequired
from minigen.conditioning.cross_attention import TextCrossAttention
from minigen.doctor import diagnose
from minigen.models.dit_block import DiTBlock, patchify, unpatchify
from minigen.objectives.flow_objective import euler_step, linear_path, velocity_loss
from minigen.samplers.cfg import classifier_free_guidance


def run_cli(
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """用当前解释器启动 CLI 子进程并捕获文本输出。

    Args:
        args: 展开的命令参数，每项为字符串；不包含解释器或模块名。

    Raises:
        OSError: 无法启动解释器时透传。

    Results:
        subprocess.CompletedProcess[str]: CompletedProcess，含 returncode、stdout、stderr；
            非零退出不主动抛异常。

    副作用:
        启动 CLI 子进程，捕获其 stdout/stderr；子进程诊断可创建本地缓存。
    """
    return subprocess.run(
        [sys.executable, "-m", "minigen", *args], capture_output=True, text=True, check=False
    )


def test_doctor_json_real_cpu_autograd() -> None:
    """验证 doctor 的 JSON 报告来自真实 CPU 前向反向与包导入。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        运行诊断或 CLI 子进程，可能创建项目本地缓存。
    """
    result = run_cli("doctor", "--device", "cpu", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["ok"]
    assert report["devices"]["cpu"]["status"] == "passed"
    assert report["devices"]["cpu"]["loss"] > 0
    assert all(item["status"] == "passed" for item in report["packages"].values())


@pytest.mark.parametrize("command", ["train", "sample", "evaluate", "cache"])
def test_unimplemented_commands_fail_honestly(
    command: str,
) -> None:
    """验证未实现命令明确返回 2，不伪造标准输出。

    Args:
        command: 参数化的未实现命令名称。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        运行诊断或 CLI 子进程，可能创建项目本地缓存。
    """
    result = run_cli(command)
    assert result.returncode == 2
    assert "尚未实现" in result.stderr
    assert not result.stdout


def test_cli_help_and_version() -> None:
    """验证帮助、版本和缺少子命令的退出约定。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        运行诊断或 CLI 子进程，可能创建项目本地缓存。
    """
    assert run_cli("--help").returncode == 0
    assert "minigen 0.1.0" in run_cli("--version").stdout
    assert run_cli().returncode == 2


def test_requested_unavailable_device_is_failure() -> None:
    """验证显式请求不可用设备时总体诊断判为失败。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。

    副作用:
        运行诊断或 CLI 子进程，可能创建项目本地缓存。
    """
    report = diagnose("cpu")
    for device, availability in report["devices"].items():
        if availability["status"] == "unavailable":
            requested = diagnose(device)
            assert not requested["ok"]
            assert requested["devices"][device]["status"] == "unavailable"


def test_personal_dit_patch_order_and_round_trip() -> None:
    """验证个人 DiT 的切块顺序、逆变换及零门控初始化。

    Raises:
        AssertionError: patch 数值、恢复结果或初始化输出不符合契约。

    副作用:
        初始化临时 DiT Block，消耗当前进程的 Torch 随机状态。
    """
    latents = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    # 先按 patch 网格从左到右、从上到下，再按 patch 内像素顺序交织通道。
    expected = latents.new_tensor(
        [
            [0, 16, 1, 17, 4, 20, 5, 21],
            [2, 18, 3, 19, 6, 22, 7, 23],
            [8, 24, 9, 25, 12, 28, 13, 29],
            [10, 26, 11, 27, 14, 30, 15, 31],
        ]
    ).unsqueeze(0)
    tokens = patchify(latents, 2)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    torch.testing.assert_close(unpatchify(tokens, 2, 4, 4, 2), latents, rtol=0, atol=0)
    block = DiTBlock(hidden_size=8, num_heads=2)
    torch.testing.assert_close(block(tokens, latents.new_ones(1, 8)), tokens, rtol=0, atol=0)


def test_remaining_core_interfaces_never_silently_fall_back() -> None:
    """验证尚待个人实现的文本注意力、CFG 和 Flow 接口明确报错。

    Raises:
        AssertionError: 结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    latents = torch.zeros(2, 4, 4, 4)
    time = torch.zeros(2)
    image_tokens = torch.zeros(2, 3, 8)
    text_tokens = torch.zeros(2, 4, 12)
    text_mask = torch.ones(2, 4, dtype=torch.bool)
    # 每个 lambda 的契约为 Callable[[], object]；预期调用都以待实现异常终止。
    calls = [
        lambda: TextCrossAttention(hidden=8, text_width=12, heads=2)(
            image_tokens, text_tokens, text_mask
        ),
        lambda: classifier_free_guidance(latents, latents, 1.0),
        lambda: linear_path(latents, latents, time),
        lambda: velocity_loss(latents, latents),
        lambda: euler_step(latents, latents, 0.1),
    ]
    for call in calls:
        with pytest.raises(CoreImplementationRequired, match="learning/README.md"):
            call()
