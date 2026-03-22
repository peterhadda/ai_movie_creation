from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def resize_frames(video_frames: list[Image.Image], frame_size: tuple[int, int]) -> list[Image.Image]:
    return [frame.resize(frame_size) for frame in video_frames]


def normalize_frames(video_frames: list[Image.Image]) -> list[torch.Tensor]:
    normalized_frames: list[torch.Tensor] = []
    for frame in video_frames:
        frame_tensor = torch.tensor(list(frame.getdata()), dtype=torch.float32)
        frame_tensor = frame_tensor.view(frame.height, frame.width, 3).permute(2, 0, 1) / 255.0
        normalized_frames.append(frame_tensor)
    return normalized_frames


def convert_frames_to_tensor(video_frames: list[Image.Image] | list[torch.Tensor]) -> torch.Tensor:
    if not video_frames:
        raise ValueError("video_frames cannot be empty")

    if isinstance(video_frames[0], torch.Tensor):
        return torch.stack(video_frames)

    normalized_frames = normalize_frames(video_frames)  # type: ignore[arg-type]
    return torch.stack(normalized_frames)


def save_extracted_frames(video_frames: list[Image.Image], output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, frame in enumerate(video_frames):
        frame_path = output_path / f"frame_{index:03d}.png"
        frame.save(frame_path)
        saved_paths.append(frame_path)
    return saved_paths
