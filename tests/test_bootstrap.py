"""Validate the bootstrap's real behavior, not the unfinished generation algorithms."""

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


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "minigen", *args], capture_output=True, text=True, check=False
    )


def test_doctor_json_real_cpu_autograd():
    result = run_cli("doctor", "--device", "cpu", "--json")
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["ok"]
    assert report["devices"]["cpu"]["status"] == "passed"
    assert report["devices"]["cpu"]["loss"] > 0
    assert all(item["status"] == "passed" for item in report["packages"].values())


@pytest.mark.parametrize("command", ["train", "sample", "evaluate", "cache"])
def test_unimplemented_commands_fail_honestly(command):
    result = run_cli(command)
    assert result.returncode == 2
    assert "尚未实现" in result.stderr
    assert not result.stdout


def test_cli_help_and_version():
    assert run_cli("--help").returncode == 0
    assert "minigen 0.1.0" in run_cli("--version").stdout
    assert run_cli().returncode == 2


def test_requested_unavailable_device_is_failure():
    report = diagnose("cpu")
    for device, availability in report["devices"].items():
        if availability["status"] == "unavailable":
            requested = diagnose(device)
            assert not requested["ok"]
            assert requested["devices"][device]["status"] == "unavailable"


def test_core_interfaces_never_silently_fall_back():
    x = torch.zeros(2, 4, 4, 4)
    t = torch.zeros(2, dtype=torch.long)
    schedule = DDPMScheduler()
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
