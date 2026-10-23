"""L05 / S4-01: continuous time increases from noise (0) to data (1)."""

from torch import Tensor

from minigen._learning import pending


def linear_path(clean: Tensor, noise: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """Return perturbed input and target velocity; time [B], images [B,C,H,W]."""
    pending("L05 linear_path")


def velocity_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Return a scalar mean-squared velocity regression objective."""
    pending("L05 velocity_loss")


def euler_step(state: Tensor, velocity: Tensor, delta_time: float) -> Tensor:
    """Update toward increasing time; output shape matches state."""
    pending("L05 euler_step")
