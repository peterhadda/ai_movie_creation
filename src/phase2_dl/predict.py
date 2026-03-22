from __future__ import annotations

import torch
from torch import nn

new_record: torch.Tensor | None = None
predictions: list[int] = []


def predict_single_sample(model: nn.Module, sample_tensor: torch.Tensor) -> int:
    model.eval()
    with torch.no_grad():
        outputs = model(sample_tensor.unsqueeze(0))
        prediction = int(torch.argmax(outputs, dim=1).item())
    return prediction


def predict_batch(model: nn.Module, batch_tensor: torch.Tensor) -> list[int]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch_tensor)
        predictions = torch.argmax(outputs, dim=1).tolist()
    return predictions
