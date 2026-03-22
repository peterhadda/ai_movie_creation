from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encoder(image)
        return latent_vector

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        reconstructed_image = self.decoder(latent_vector)
        return reconstructed_image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encode(image)
        reconstructed_image = self.decode(latent_vector)
        return reconstructed_image


def initialize_autoencoder(input_dim: int = 784, latent_dim: int = 32) -> Autoencoder:
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    return model


def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.mse_loss(reconstructed, original)
    return loss


def save_autoencoder(model: Autoencoder, model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
    }
    torch.save(checkpoint, path)


def load_autoencoder(model_path: str | Path) -> Autoencoder:
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    model = initialize_autoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
