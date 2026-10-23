"""L02 / S1-06: diagonal-Gaussian reparameterization and KL."""

from torch import Tensor

from minigen._learning import pending


def reparameterize(mean: Tensor, log_variance: Tensor, noise: Tensor) -> Tensor:
    """Same-shaped inputs; preserve a gradient path to both distribution parameters."""
    pending("L02 reparameterize")


def kl_standard_normal(mean: Tensor, log_variance: Tensor) -> Tensor:
    """Scalar KL(q || N(0,I)); document sum versus mean over latent dimensions."""
    pending("L02 kl_standard_normal")
