from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.autoencoder import reconstruction_loss
from src.image_dataset import flatten_images


def initialize_autoencoder_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def train_autoencoder(
    model: nn.Module,
    batch_images: torch.Tensor,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    flattened_images = flatten_images(batch_images)
    reconstructed_images = model(flattened_images)
    loss = reconstruction_loss(flattened_images, reconstructed_images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_autoencoder_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    optimizer: optim.Optimizer,
) -> float:
    total_loss = 0.0
    for batch_images, _ in train_loader:
        total_loss += train_autoencoder(model, batch_images, optimizer)
    train_loss = total_loss / max(len(train_loader), 1)
    return float(train_loss)


def validate_autoencoder(
    model: nn.Module,
    val_loader: DataLoader[Any],
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_images, _ in val_loader:
            flattened_images = flatten_images(batch_images)
            reconstructed_images = model(flattened_images)
            loss = reconstruction_loss(flattened_images, reconstructed_images)
            total_loss += float(loss.item())

    val_loss = total_loss / max(len(val_loader), 1)
    return float(val_loss)


def run_autoencoder_training(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    training_history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_autoencoder_epoch(model, train_loader, optimizer)
        val_loss = validate_autoencoder(model, val_loader)
        training_history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    trained_model = model
    return trained_model, training_history
