from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

input_size = 0
hidden_size = 16
output_size = 0
model: nn.Module | None = None


class MediaTypeNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.network(inputs)
        return outputs


def initialize_network(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> MediaTypeNetwork:
    model = MediaTypeNetwork(input_size, hidden_size, output_size)
    return model


def initialize_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> MediaTypeNetwork:
    return initialize_network(input_size, hidden_size, output_size)


def save_model(model: MediaTypeNetwork, model_path: str | Path) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "output_size": model.output_size,
    }
    torch.save(checkpoint, Path(model_path))


def load_model(model_path: str | Path) -> MediaTypeNetwork:
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    model = initialize_network(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        checkpoint["output_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_labels(model: nn.Module, X: torch.Tensor) -> list[int]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred = torch.argmax(logits, dim=1).tolist()
    return y_pred


def predict_probabilities(model: nn.Module, X: torch.Tensor) -> list[list[float]]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_proba = torch.softmax(logits, dim=1).tolist()
    return y_proba
