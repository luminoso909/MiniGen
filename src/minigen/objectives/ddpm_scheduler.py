"""L01 / S1-03: discrete diffusion math; t is a batch of integer indices."""

from torch import Tensor

from minigen._learning import pending


class DDPMScheduler:
    """Own the noise schedule, forward perturbation and reverse sampling math."""

    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Inputs [B,C,H,W], [B,C,H,W], [B]; output has the image shape."""
        pending("L01 add_noise")

    def loss(self, predicted_noise: Tensor, noise: Tensor) -> Tensor:
        """Return a scalar; explicitly choose reduction over batch and features."""
        pending("L01 noise_prediction_loss")

    def step(self, predicted_noise: Tensor, timestep: int, sample: Tensor, noise: Tensor) -> Tensor:
        """One reverse step; handle the last step without extra random perturbation."""
        pending("L01 reverse_step")
