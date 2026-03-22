from __future__ import annotations

from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

loss_fn: nn.Module | None = None
optimizer: optim.Optimizer | None = None
learning_rate = 0.001
num_epochs = 20


def initialize_loss_function(task_type: str) -> nn.Module:
    if task_type != "classification":
        raise ValueError(f"Unsupported task_type: {task_type}")
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def initialize_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


def train_one_batch(
    model: nn.Module,
    batch_X: torch.Tensor,
    batch_y: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    y_pred = model(batch_X)
    loss = loss_fn(y_pred, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    batch_loss = float(loss.item())
    return batch_loss


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for batch_X, batch_y in train_loader:
        batch_loss = train_one_batch(model, batch_X, batch_y, loss_fn, optimizer)
        total_loss += batch_loss

        with torch.no_grad():
            logits = model(batch_X)
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += int((predictions == batch_y).sum().item())
            total_examples += len(batch_y)

    train_loss = total_loss / max(len(train_loader), 1)
    train_accuracy = correct_predictions / max(total_examples, 1)
    return float(train_loss), float(train_accuracy)


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader[Any],
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            total_loss += float(loss.item())

            predictions = torch.argmax(y_pred, dim=1)
            correct_predictions += int((predictions == batch_y).sum().item())
            total_examples += len(batch_y)

    val_loss = total_loss / max(len(val_loader), 1)
    val_accuracy = correct_predictions / max(total_examples, 1)
    return float(val_loss), float(val_accuracy)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    training_history: list[dict[str, float]] = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_fn)
        training_history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

    trained_model = model
    return trained_model, training_history
