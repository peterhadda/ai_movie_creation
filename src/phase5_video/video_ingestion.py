from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageSequence


def load_video(video_path: str | Path) -> Image.Image:
    return Image.open(video_path)


def extract_frames(video_path: str | Path) -> list[Image.Image]:
    with Image.open(video_path) as video:
        video_frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(video)]
    if not video_frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    return video_frames


def extract_video_metadata(video_path: str | Path) -> dict[str, Any]:
    with Image.open(video_path) as video:
        frame_count = sum(1 for _ in ImageSequence.Iterator(video))
        duration_ms = video.info.get("duration", 100)
        fps = 1000 / duration_ms if duration_ms else 10.0
        width, height = video.size
    return {
        "frame_count": frame_count,
        "duration_ms_per_frame": duration_ms,
        "fps": fps,
        "frame_size": (width, height),
    }


def generate_moving_square_video(
    frame_count: int,
    frame_size: tuple[int, int],
    square_size: int = 8,
    step_size: int = 2,
) -> list[Image.Image]:
    width, height = frame_size
    video_frames: list[Image.Image] = []
    max_x = max(width - square_size, 1)
    max_y = max(height - square_size, 1)

    for index in range(frame_count):
        frame_array = np.zeros((height, width, 3), dtype=np.uint8)
        x_position = min((index * step_size) % max_x, width - square_size)
        y_position = min((index * step_size) % max_y, height - square_size)
        frame_array[y_position : y_position + square_size, x_position : x_position + square_size] = 255
        video_frames.append(Image.fromarray(frame_array, mode="RGB"))

    return video_frames
