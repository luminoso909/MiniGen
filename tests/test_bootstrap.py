"""验证环境、CLI 与个人核心未实现边界的真实行为。"""

import json
import subprocess
import sys

import pytest
import torch

from minigen._learning import CoreImplementationRequired
from minigen.autoencoders.vae_core import kl_standard_normal, reparameterize
from minigen.doctor import diagnose
from minigen.models.dit_block import DiTBlock, patchify, unpatchify
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.objectives.flow_objective import euler_step, linear_path, velocity_loss
from minigen.samplers.cfg import classifier_free_guidance


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """用当前解释器启动 CLI 子进程并捕获文本输出。

    1 输入：
        args：展开的命令参数，每项为字符串；不包含解释器或模块名。

    2 输出：
        CompletedProcess，含 returncode、stdout、stderr；非零退出不主动抛异常。

    3 报错（Raises）：
        OSError：无法启动解释器时透传。
    """
    return subprocess.run(
        [sys.executable, "-m", "minigen", *args], capture_output=True, text=True, check=False
    )


def test_doctor_json_real_cpu_autograd() -> None:
    """验证 doctor 的 JSON 报告来自真实 CPU 前向反向与包导入。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    result = run_cli("doctor", "--device", "cpu", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["ok"]
    assert report["devices"]["cpu"]["status"] == "passed"
    assert report["devices"]["cpu"]["loss"] > 0
    assert all(item["status"] == "passed" for item in report["packages"].values())


@pytest.mark.parametrize("command", ["train", "sample", "evaluate", "cache"])
def test_unimplemented_commands_fail_honestly(command: str) -> None:
    """验证未实现命令明确返回 2，不伪造标准输出。

    1 输入：
        command：参数化的未实现命令名称。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    result = run_cli(command)
    assert result.returncode == 2
    assert "尚未实现" in result.stderr
    assert not result.stdout


def test_cli_help_and_version() -> None:
    """验证帮助、版本和缺少子命令的退出约定。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    assert run_cli("--help").returncode == 0
    assert "minigen 0.1.0" in run_cli("--version").stdout
    assert run_cli().returncode == 2


def test_requested_unavailable_device_is_failure() -> None:
    """验证显式请求不可用设备时总体诊断判为失败。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    report = diagnose("cpu")
    for device, availability in report["devices"].items():
        if availability["status"] == "unavailable":
            requested = diagnose(device)
            assert not requested["ok"]
            assert requested["devices"][device]["status"] == "unavailable"


def test_core_interfaces_never_silently_fall_back() -> None:
    """逐项验证五类学习核心仍抛出待实现异常。

    1 输入：
        无显式参数；使用对象状态、闭包或命令行上下文。

    2 输出：
        无返回值；以断言验证该项契约，临时资源只写 pytest 目录。

    3 报错（Raises）：
        AssertionError：结果偏离预期时失败；pytest.raises 中约定的异常被捕获，其他异常令测试失败。
    """
    x = torch.zeros(2, 4, 4, 4)
    t = torch.zeros(2, dtype=torch.long)
    schedule = DDPMScheduler()
    # 每个 lambda 的契约为 Callable[[], object]；预期调用都以待实现异常终止。
    calls = [
        lambda: schedule.add_noise(x, x, t),
        lambda: schedule.loss(x, x),
        lambda: schedule.step(x, 0, x, x),
        lambda: reparameterize(x, x, x),
        lambda: kl_standard_normal(x, x),
        lambda: patchify(x, 2),
        lambda: unpatchify(x, 4, 4, 4, 2),
        lambda: DiTBlock()(x, x),
        lambda: classifier_free_guidance(x, x, 1.0),
        lambda: linear_path(x, x, t.float()),
        lambda: velocity_loss(x, x),
        lambda: euler_step(x, x, 0.1),
    ]
    for call in calls:
        with pytest.raises(CoreImplementationRequired, match="learning/README.md"):
            call()
