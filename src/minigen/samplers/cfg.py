"""L04 / S3-02: use unconditioned + scale * conditional difference."""

from torch import Tensor

from minigen._learning import pending


def classifier_free_guidance(unconditioned: Tensor, conditioned: Tensor, scale: float) -> Tensor:
    """Same-shaped predictions; scale 0 is unconditional and 1 is conditional."""
    pending("L04 classifier_free_guidance")
