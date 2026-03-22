from __future__ import annotations

from typing import Any

from torch import nn, optim
from torch.utils.data import DataLoader


def initialize_loss_function() -> nn.Module:
    return nn.MSELoss()


def initialize_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate)


def train_one_video_batch(
    model: nn.Module,
    batch_sequences,
    batch_targets,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    predicted_frames = model(batch_sequences)
    loss = loss_fn(predicted_frames, batch_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    total_loss = 0.0
    for batch_sequences, batch_targets in train_loader:
        total_loss += train_one_video_batch(model, batch_sequences, batch_targets, loss_fn, optimizer)
    return total_loss / max(len(train_loader), 1)


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader[Any],
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    with __import__("torch").no_grad():
        for batch_sequences, batch_targets in val_loader:
            predicted_frames = model(batch_sequences)
            loss = loss_fn(predicted_frames, batch_targets)
            total_loss += float(loss.item())
    return total_loss / max(len(val_loader), 1)


def run_video_training_loop(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    training_history: list[dict[str, float]] = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validate_one_epoch(model, val_loader, loss_fn)
        training_history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
    return model, training_history
