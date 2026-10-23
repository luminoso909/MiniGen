"""L03 / S2-04: spatial patch order and adaLN-Zero are learner-owned."""

from torch import Tensor, nn

from minigen._learning import pending


def patchify(latents: Tensor, patch_size: int) -> Tensor:
    """[B,C,H,W] -> [B,(H/p)*(W/p),C*p*p]; preserve an explicit spatial order."""
    pending("L03 patchify")


def unpatchify(tokens: Tensor, channels: int, height: int, width: int, patch_size: int) -> Tensor:
    """Invert patchify exactly, using the same ordering convention."""
    pending("L03 unpatchify")


class DiTBlock(nn.Module):
    """Interface only; define projections, modulation and initialization yourself."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        """tokens [B,N,D], conditioning [B,D]; output [B,N,D]."""
        pending("L03 DiTBlock / adaLN-Zero")
