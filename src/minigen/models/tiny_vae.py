"""MLP encoder/decoder wiring; core distribution operations are supplied by the caller."""

from torch import nn


class TinyVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )
        self.mean_head = nn.Linear(128, latent_dim)
        self.log_variance_head = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def encode(self, images):
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("Expected [B,1,28,28]")
        hidden = self.encoder(images)
        return self.mean_head(hidden), self.log_variance_head(hidden)

    def decode(self, latent):
        return self.decoder(latent).reshape(-1, 1, 28, 28)

    def forward(self, images, noise, *, reparameterize_fn):
        mean, log_variance = self.encode(images)
        latent = reparameterize_fn(mean, log_variance, noise)
        return self.decode(latent), mean, log_variance
