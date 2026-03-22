from __future__ import annotations

import torch
from torch import nn

from src.audio_data import frame_vectors_to_spectrogram, postprocess_audio, spectrogram_to_waveform


def generate_audio(model: nn.Module, seed_sequence: torch.Tensor, length: int) -> torch.Tensor:
    model.eval()
    generated_frames = seed_sequence.clone()
    current_sequence = seed_sequence.unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            predicted_sequence = model(current_sequence)
            next_frame = predicted_sequence[:, -1, :]
            generated_frames = torch.cat([generated_frames, next_frame], dim=0)
            current_sequence = torch.cat([current_sequence[:, 1:, :], next_frame.unsqueeze(1)], dim=1)

    return generated_frames


def generated_frames_to_waveform(
    generated_audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    target_length: int | None = None,
) -> torch.Tensor:
    generated_spectrogram = frame_vectors_to_spectrogram(generated_audio)
    audio_waveform = spectrogram_to_waveform(
        generated_spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        length=target_length,
    )
    return postprocess_audio(audio_waveform)
