from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def initialize_audio_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def compute_audio_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.mse_loss(predicted, target)
    return loss


def train_audio_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0

    for input_sequence, target_sequence in train_loader:
        predicted_sequence = model(input_sequence)
        loss = compute_audio_loss(predicted_sequence, target_sequence)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(train_loader), 1)


def validate_audio_model(
    model: nn.Module,
    val_loader: DataLoader[Any],
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_sequence, target_sequence in val_loader:
            predicted_sequence = model(input_sequence)
            loss = compute_audio_loss(predicted_sequence, target_sequence)
            total_loss += float(loss.item())

    return total_loss / max(len(val_loader), 1)


def run_audio_training_loop(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    training_history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_audio_model(model, train_loader, optimizer)
        val_loss = validate_audio_model(model, val_loader)
        training_history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    trained_model = model
    return trained_model, training_history
