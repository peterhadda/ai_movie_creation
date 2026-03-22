from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

y_true: list[int] = []
y_pred: list[int] = []


def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    matches = sum(int(expected == predicted) for expected, predicted in zip(y_true, y_pred))
    return matches / len(y_true)


def build_confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).tolist()


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[Any],
    loss_fn: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            total_loss += float(loss.item())

            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predictions.tolist())

    test_loss = total_loss / max(len(test_loader), 1)
    test_accuracy = compute_accuracy(y_true, y_pred)
    return float(test_loss), float(test_accuracy), y_true, y_pred


def generate_evaluation_report(
    test_loss: float,
    test_accuracy: float,
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, Any]:
    evaluation_report = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": build_confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return evaluation_report
