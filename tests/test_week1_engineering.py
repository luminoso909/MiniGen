"""Engineering contracts; analytic stubs are explicit test doubles, not DDPM references."""

import copy

import pytest
import torch
from torch import nn

from minigen._learning import CoreImplementationRequired
from minigen.models.tiny_unet import TinyUNet
from minigen.objectives.ddpm_scheduler import DDPMScheduler
from minigen.samplers.ddpm_loop import sample_ddpm
from minigen.training.denoising import TrainConfig, seed_all, train_denoiser


class AnalyticTrainingStub:
    num_steps = 4

    def add_noise(self, clean, noise, timesteps):
        return clean + noise

    def loss(self, prediction, noise):
        return (prediction - noise).square().mean()


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, image, times):
        return self.layer(image)


def test_unet_shapes_and_time_condition_gradient():
    torch.set_num_threads(2)
    model = TinyUNet(base=8, time_width=16)
    image = torch.randn(2, 1, 28, 28)
    result = model(image, torch.tensor([0, 999]))
    assert result.shape == image.shape
    result.square().mean().backward()
    assert model.time_embedding.mlp[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cpu_checkpoint_resume_matches_continuous_with_fixed_loader(tmp_path):
    torch.set_num_threads(2)
    seed_all(4)
    original = SmallModel()
    batches = [(torch.zeros(2, 1, 4, 4),)]
    full = copy.deepcopy(original)
    train_denoiser(
        full,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "full"), max_steps=6, checkpoint_every=6),
    )
    partial = copy.deepcopy(original)
    first = train_denoiser(
        partial,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "partial"), max_steps=3, checkpoint_every=3),
    )
    resumed = SmallModel()
    train_denoiser(
        resumed,
        AnalyticTrainingStub(),
        batches,
        TrainConfig(str(tmp_path / "resume"), max_steps=6, checkpoint_every=6),
        resume_path=first["checkpoint"],
    )
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=True)
    for key in a["ema"]:
        torch.testing.assert_close(a["ema"][key], b["ema"][key], rtol=0, atol=0)
    assert a["step"] == b["step"] == 6


def test_pending_project_scheduler_is_not_replaced(tmp_path):
    with pytest.raises(CoreImplementationRequired):
        train_denoiser(
            SmallModel(),
            DDPMScheduler(),
            [(torch.zeros(2, 1, 4, 4),)],
            TrainConfig(str(tmp_path), max_steps=1),
        )


def test_full_reverse_loop_and_timeout_are_distinguished():
    class SamplingStub:
        num_steps = 4

        def step(self, predicted, timestep, sample, noise):
            return sample + 1

    model = SmallModel()
    initial = torch.zeros(2, 1, 4, 4)
    result = sample_ddpm(model, SamplingStub(), initial, capture=[0])
    assert result["completed"] and result["completed_steps"] == 4
    torch.testing.assert_close(result["images"], torch.full_like(initial, 4))
    short = sample_ddpm(model, SamplingStub(), initial, max_seconds=1e-12)
    assert not short["completed"] and short["completed_steps"] == 0
    assert model.training
