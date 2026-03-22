from __future__ import annotations

from pathlib import Path

import torch

from src.common.io_utils import ensure_directory_exists, save_evaluation_report, save_training_history


def save_video_model(model, model_path: str | Path, input_shape: tuple[int, int, int], hidden_size: int) -> Path:
    path = Path(model_path)
    ensure_directory_exists(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "hidden_size": hidden_size,
        },
        path,
    )
    return path


__all__ = ["save_video_model", "save_training_history", "save_evaluation_report"]
