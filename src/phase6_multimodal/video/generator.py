from __future__ import annotations

from pathlib import Path

import torch

from src.phase5_video.generate import assemble_video_from_frames, generate_frame_sequence
from src.phase5_video.model import initialize_video_model
from src.phase5_video.video_ingestion import generate_moving_square_video
from src.phase5_video.frame_processing import convert_frames_to_tensor


def encode_video(video_input: torch.Tensor) -> torch.Tensor:
    flattened_video = video_input.flatten()
    video_embedding = flattened_video / flattened_video.norm().clamp_min(1e-8)
    return video_embedding


def _project_embedding(shared_embedding: torch.Tensor, output_dim: int) -> torch.Tensor:
    repeats = (output_dim + shared_embedding.numel() - 1) // shared_embedding.numel()
    return shared_embedding.repeat(repeats)[:output_dim]


def _fallback_seed_sequence(shared_embedding: torch.Tensor, sequence_length: int, frame_size: tuple[int, int]) -> torch.Tensor:
    brightness = float(shared_embedding.abs().mean().item())
    step_size = max(1, int(1 + brightness * 4))
    frames = generate_moving_square_video(
        frame_count=sequence_length,
        frame_size=frame_size,
        square_size=6,
        step_size=step_size,
    )
    return convert_frames_to_tensor(frames)


def generate_video(
    shared_embedding: torch.Tensor,
    model_path: str | Path,
    output_path: str | Path,
    fps: int = 8,
) -> Path:
    model_file = Path(model_path)

    if model_file.exists():
        checkpoint = torch.load(model_file, map_location="cpu")
        model = initialize_video_model(
            input_shape=tuple(checkpoint["input_shape"]),
            hidden_size=checkpoint["hidden_size"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        input_shape = tuple(checkpoint["input_shape"])
        sequence_length = 4
        seed_sequence = _project_embedding(
            shared_embedding,
            sequence_length * input_shape[0] * input_shape[1] * input_shape[2],
        ).view(sequence_length, *input_shape)
        generated_frames = generate_frame_sequence(model, seed_sequence, num_future_frames=6)
    else:
        seed_sequence = _fallback_seed_sequence(shared_embedding, 4, (32, 32))
        generated_frames = seed_sequence

    return assemble_video_from_frames(generated_frames, fps=fps, output_path=output_path)
