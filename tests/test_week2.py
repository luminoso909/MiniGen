"""Week-two math references and real engineering contracts, without learner-core fallback."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from minigen._learning import CoreImplementationRequired
from minigen.autoencoders import vae_core
from minigen.data.stateful import StatefulBatchStream
from minigen.models.tiny_vae import TinyVAE
from minigen.training.denoising import TrainConfig, seed_all, train_denoiser
from minigen.training.vae import VAEConfig, evaluate_vae, reconstruction_loss, train_vae


@pytest.fixture
def reference():
    notebook = json.loads(Path("docs/learning/S1_第2周_VAE原理与潜空间实验.ipynb").read_text())
    cells = [
        c for c in notebook["cells"] if "vae_reference" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(cells) == 1
    scope = {"torch": torch}
    exec("".join(cells[0]["source"]), scope)
    return scope["teaching_reparameterize"], scope["teaching_kl"]


def test_reference_math_gradients_and_reduction(reference):
    reparam, kl = reference
    mean = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    logvar = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], dtype=torch.float64, requires_grad=True)
    noise = torch.randn_like(mean)
    assert torch.autograd.gradcheck(reparam, (mean, logvar, noise))
    assert torch.autograd.gradcheck(kl, (mean, logvar))
    q = torch.distributions.Normal(mean, (logvar / 2).exp())
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    independent = torch.distributions.kl_divergence(q, p).sum(1).mean()
    torch.testing.assert_close(kl(mean, logvar), independent)
    torch.testing.assert_close(kl(mean.repeat(3, 1), logvar.repeat(3, 1)), independent)
    assert kl(torch.zeros_like(mean), torch.zeros_like(logvar)) == 0
    with pytest.raises(ValueError):
        reparam(mean, logvar, noise[:, :1])


def test_project_core_still_explicitly_pending():
    values = torch.zeros(2, 3)
    with pytest.raises(CoreImplementationRequired):
        vae_core.reparameterize(values, values, values)
    with pytest.raises(CoreImplementationRequired):
        vae_core.kl_standard_normal(values, values)


def test_data_stream_short_batch_epoch_and_recovery():
    dataset = TensorDataset(torch.arange(19))
    stream = StatefulBatchStream(dataset, 7, data_id="v1", seed=8)
    first = [next(stream)[0] for _ in range(3)]
    assert [len(x) for x in first] == [7, 7, 5]
    assert torch.equal(torch.cat(first).sort().values, torch.arange(19))
    state = stream.state_dict()
    following = [next(stream)[0] for _ in range(5)]
    other = StatefulBatchStream(dataset, 7, data_id="v1", seed=999)
    other.load_state_dict(state)
    assert all(torch.equal(a, next(other)[0]) for a in following)
    with pytest.raises(ValueError, match="identity"):
        StatefulBatchStream(dataset, 7, data_id="v2").load_state_dict(state)
    with pytest.raises(ValueError, match="configuration"):
        StatefulBatchStream(dataset, 8, data_id="v1").load_state_dict(state)


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare_nested(x, y)
    else:
        assert a == b


class EngineeringObjectiveStub:
    """Test double only: not an implementation or evaluation of DDPM mathematics."""

    num_steps = 7

    def __init__(self):
        self.noises = []
        self.times = []

    def add_noise(self, clean, noise, timesteps):
        self.noises.append(noise.clone())
        self.times.append(timesteps.clone())
        return clean + noise

    def loss(self, prediction, noise):
        return (prediction - noise).square().mean()


class EngineeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, image, times):
        return self.layer(image)


@pytest.mark.parametrize("split", [2, 3, 5])
def test_denoiser_exact_resume_order_noise_optimizer_and_ema(tmp_path, split):
    torch.set_num_threads(2)
    seed_all(11)
    dataset = TensorDataset(torch.arange(19.0).reshape(19, 1, 1, 1) / 19)
    original = EngineeringModel()

    def stream():
        return StatefulBatchStream(dataset, 7, data_id="19-v1", seed=6)

    full_scheduler = EngineeringObjectiveStub()
    full = train_denoiser(
        copy.deepcopy(original),
        full_scheduler,
        stream(),
        TrainConfig(str(tmp_path / "full"), max_steps=8),
    )
    first = train_denoiser(
        copy.deepcopy(original),
        EngineeringObjectiveStub(),
        stream(),
        TrainConfig(str(tmp_path / "partial"), max_steps=split),
    )
    next_scheduler = EngineeringObjectiveStub()
    resumed = train_denoiser(
        EngineeringModel(),
        next_scheduler,
        stream(),
        TrainConfig(str(tmp_path / "resumed"), max_steps=8, require_exact_resume=True),
        resume_path=first["checkpoint"],
    )
    a = torch.load(full["checkpoint"], weights_only=True)
    b = torch.load(resumed["checkpoint"], weights_only=True)
    for key in ("model", "optimizer", "ema", "rng", "data_state", "step"):
        compare_nested(a[key], b[key])
    compare_nested(full_scheduler.noises[split:], next_scheduler.noises)
    compare_nested(full_scheduler.times[split:], next_scheduler.times)
    assert [r["loss"] for r in full["metrics"][split:]] == [r["loss"] for r in resumed["metrics"]]


def test_old_checkpoint_cannot_claim_exact_recovery(tmp_path):
    first = train_denoiser(
        EngineeringModel(),
        EngineeringObjectiveStub(),
        [(torch.zeros(2, 1, 2, 2),)],
        TrainConfig(str(tmp_path / "first"), max_steps=1),
    )
    with pytest.raises(ValueError, match="no exact"):
        train_denoiser(
            EngineeringModel(),
            EngineeringObjectiveStub(),
            [(torch.zeros(2, 1, 2, 2),)],
            TrainConfig(str(tmp_path / "next"), max_steps=2, require_exact_resume=True),
            resume_path=first["checkpoint"],
        )


def test_vae_training_resume_and_validation_rng(tmp_path, reference):
    reparam, kl = reference
    seed_all(31)
    images = torch.rand(19, 1, 28, 28)
    dataset = TensorDataset(images)
    original = TinyVAE(2)

    def stream():
        return StatefulBatchStream(dataset, 7, data_id="images01", seed=4)

    def run(model, folder, steps, resume=None):
        return train_vae(
            model,
            stream(),
            VAEConfig(str(tmp_path / folder), max_steps=steps),
            reparameterize_fn=reparam,
            kl_fn=kl,
            validation=images[:3],
            resume_path=resume,
        )

    a = run(copy.deepcopy(original), "full", 6)
    first = run(copy.deepcopy(original), "partial", 2)
    b = run(TinyVAE(2), "resumed", 6, first["checkpoint"])
    saved_a = torch.load(a["checkpoint"], weights_only=True)
    saved_b = torch.load(b["checkpoint"], weights_only=True)
    for key in ("model", "optimizer", "rng", "data_state", "step"):
        compare_nested(saved_a[key], saved_b[key])
    state = torch.get_rng_state().clone()
    evaluate_vae(original, images, reparameterize_fn=reparam, kl_fn=kl)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match="configuration"):
        train_vae(
            TinyVAE(2),
            stream(),
            VAEConfig(str(tmp_path / "bad"), beta=4),
            reparameterize_fn=reparam,
            kl_fn=kl,
            validation=images,
            resume_path=first["checkpoint"],
        )


def test_bce_scale_and_bad_target():
    x = torch.zeros(2, 1, 28, 28)
    assert abs(reconstruction_loss(x, x).item() - 784 * torch.log(torch.tensor(2.0)).item()) < 0.001
    with pytest.raises(ValueError):
        reconstruction_loss(x, x - 1)


def test_fixed_batch_stream_keeps_images_and_noise_aligned():
    stream = StatefulBatchStream(TensorDataset(torch.arange(32)), 32, data_id="fit", shuffle=False)
    first = next(stream)[0]
    for _ in range(3):
        assert torch.equal(next(stream)[0], first)
    state = stream.state_dict()
    with pytest.raises(ValueError):
        StatefulBatchStream(stream.dataset, 32, data_id="fit", shuffle=True).load_state_dict(state)
