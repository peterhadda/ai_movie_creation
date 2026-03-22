from __future__ import annotations

import torch


def compute_frame_reconstruction_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean((y_true - y_pred) ** 2).item())


def measure_temporal_consistency(generated_frames: torch.Tensor) -> float:
    if generated_frames.shape[0] < 2:
        return 0.0
    frame_deltas = generated_frames[1:] - generated_frames[:-1]
    return float(torch.mean(torch.abs(frame_deltas)).item())


def generate_video_evaluation_report(metrics: dict[str, float]) -> dict[str, float]:
    return metrics
