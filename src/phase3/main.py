from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from src.autoencoder import initialize_autoencoder, save_autoencoder
from src.image_dataset import (
    create_image_data_loader,
    flatten_images,
    load_mnist_dataset,
    split_image_dataset,
)
from src.image_train import initialize_autoencoder_optimizer, run_autoencoder_training
from src.utils import ensure_directory_exists, load_config, log_message, save_training_history


def save_reconstruction_samples(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    output_path: str | Path,
) -> Path:
    model.eval()
    ensure_directory_exists(Path(output_path).parent)

    with torch.no_grad():
        flattened_images = flatten_images(sample_batch)
        reconstructed_batch = model(flattened_images).view(-1, 1, 28, 28)
        comparison = torch.cat([sample_batch[:8], reconstructed_batch[:8]], dim=0)
        save_image(comparison, output_path, nrow=8)

    return Path(output_path)


def run_autoencoder_pipeline(config_path: str | Path = "config_phase3.json") -> dict[str, Any]:
    config = load_config(config_path)
    phase3_config = config["phase3"]

    data_dir = phase3_config.get("data_dir", "data/raw/mnist")
    batch_size = phase3_config.get("batch_size", 64)
    learning_rate = phase3_config.get("learning_rate", 0.001)
    num_epochs = phase3_config.get("num_epochs", 5)
    latent_dim = phase3_config.get("latent_dim", 32)
    train_ratio = phase3_config.get("train_ratio", 0.8)
    val_ratio = phase3_config.get("val_ratio", 0.1)
    test_ratio = phase3_config.get("test_ratio", 0.1)
    download = phase3_config.get("download", True)

    log_message("Loading MNIST dataset")
    dataset = load_mnist_dataset(data_dir=data_dir, train=True, download=download)
    train_dataset, val_dataset, test_dataset = split_image_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_loader = create_image_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_image_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_image_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

    log_message("Initializing autoencoder")
    model = initialize_autoencoder(input_dim=28 * 28, latent_dim=latent_dim)
    optimizer = initialize_autoencoder_optimizer(model, learning_rate=learning_rate)

    log_message("Training autoencoder")
    trained_model, training_history = run_autoencoder_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=num_epochs,
    )

    sample_batch, _ = next(iter(test_loader))
    model_output_path = phase3_config["output"]["model"]
    history_output_path = phase3_config["output"]["training_history"]
    reconstruction_output_path = phase3_config["output"]["reconstructions"]

    save_autoencoder(trained_model, model_output_path)
    save_training_history(training_history, history_output_path)
    save_reconstruction_samples(trained_model, sample_batch, reconstruction_output_path)

    return {
        "trained_model": trained_model,
        "training_history": training_history,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "model_path": model_output_path,
        "history_path": history_output_path,
        "reconstructions_path": reconstruction_output_path,
    }
