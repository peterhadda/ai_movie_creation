from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def generate_next_frame(model, seed_sequence: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(seed_sequence.unsqueeze(0))[0]


def generate_frame_sequence(
    model,
    seed_sequence: torch.Tensor,
    num_future_frames: int,
) -> torch.Tensor:
    generated_frames: list[torch.Tensor] = []
    current_sequence = seed_sequence.clone()

    for _ in range(num_future_frames):
        next_frame = generate_next_frame(model, current_sequence)
        generated_frames.append(next_frame)
        current_sequence = torch.cat([current_sequence[1:], next_frame.unsqueeze(0)], dim=0)

    return torch.stack(generated_frames)


def assemble_video_from_frames(
    generated_frames: torch.Tensor,
    fps: int,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames: list[Image.Image] = []
    for frame in generated_frames:
        frame_array = (frame.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_frames.append(Image.fromarray(frame_array))

    duration_ms = int(1000 / max(fps, 1))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return path
