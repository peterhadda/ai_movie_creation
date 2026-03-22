from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.phase3_image.autoencoder import initialize_autoencoder, load_autoencoder


def encode_image(image_input: torch.Tensor) -> torch.Tensor:
    flattened_image = image_input.flatten()
    image_embedding = flattened_image / flattened_image.norm().clamp_min(1e-8)
    return image_embedding


def _project_embedding(shared_embedding: torch.Tensor, output_dim: int) -> torch.Tensor:
    repeats = (output_dim + shared_embedding.numel() - 1) // shared_embedding.numel()
    projected = shared_embedding.repeat(repeats)[:output_dim]
    return projected


def generate_image(
    shared_embedding: torch.Tensor,
    model_path: str | Path,
    output_path: str | Path,
) -> Path:
    model_file = Path(model_path)
    if model_file.exists():
        model = load_autoencoder(model_file)
    else:
        model = initialize_autoencoder()

    latent_vector = _project_embedding(shared_embedding, model.latent_dim).unsqueeze(0)
    with torch.no_grad():
        generated_image = model.decode(latent_vector).view(28, 28).clamp(0.0, 1.0)

    image_array = (generated_image.cpu().numpy() * 255).astype("uint8")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array, mode="L").save(output_file)
    return output_file
