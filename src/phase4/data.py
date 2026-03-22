from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import wavfile


def load_audio(file_path: str | Path) -> tuple[torch.Tensor, int]:
    sample_rate, audio_waveform = wavfile.read(file_path)

    if audio_waveform.ndim > 1:
        audio_waveform = audio_waveform.mean(axis=1)

    waveform = torch.tensor(audio_waveform, dtype=torch.float32)
    peak = waveform.abs().max().item()
    if peak > 0:
        waveform = waveform / peak

    return waveform, int(sample_rate)


def trim_audio(audio_waveform: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    active_indices = torch.nonzero(audio_waveform.abs() > threshold, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        return audio_waveform

    start_index = int(active_indices[0].item())
    end_index = int(active_indices[-1].item()) + 1
    return audio_waveform[start_index:end_index]


def pad_audio(audio_waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = audio_waveform.numel()
    if current_length >= target_length:
        return audio_waveform[:target_length]

    padding = torch.zeros(target_length - current_length, dtype=audio_waveform.dtype)
    return torch.cat([audio_waveform, padding], dim=0)


def waveform_to_spectrogram(
    audio_waveform: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
) -> torch.Tensor:
    window = torch.hann_window(n_fft)
    spectrogram = torch.stft(
        audio_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    return spectrogram


def spectrogram_to_waveform(
    spectrogram: torch.Tensor,
    n_fft: int = 256,
    hop_length: int = 128,
    length: int | None = None,
) -> torch.Tensor:
    window = torch.hann_window(n_fft)
    audio_waveform = torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=length,
    )
    return audio_waveform


def spectrogram_to_frame_vectors(spectrogram: torch.Tensor) -> torch.Tensor:
    real_frames = spectrogram.real.transpose(0, 1)
    imag_frames = spectrogram.imag.transpose(0, 1)
    return torch.cat([real_frames, imag_frames], dim=1)


def frame_vectors_to_spectrogram(frame_vectors: torch.Tensor) -> torch.Tensor:
    feature_count = frame_vectors.shape[1] // 2
    real_frames = frame_vectors[:, :feature_count]
    imag_frames = frame_vectors[:, feature_count:]
    return torch.complex(real_frames, imag_frames).transpose(0, 1)


def prepare_audio_sequences(
    spectrograms: Iterable[torch.Tensor],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_sequences: list[torch.Tensor] = []
    target_sequences: list[torch.Tensor] = []

    for spectrogram in spectrograms:
        frame_vectors = spectrogram_to_frame_vectors(spectrogram)
        if frame_vectors.shape[0] <= sequence_length:
            continue

        for start_index in range(frame_vectors.shape[0] - sequence_length):
            input_sequence = frame_vectors[start_index : start_index + sequence_length]
            target_sequence = frame_vectors[start_index + 1 : start_index + sequence_length + 1]
            input_sequences.append(input_sequence)
            target_sequences.append(target_sequence)

    if not input_sequences:
        raise ValueError("No audio sequences were created. Increase the data size or reduce sequence_length.")

    return torch.stack(input_sequences), torch.stack(target_sequences)


def postprocess_audio(audio_waveform: torch.Tensor) -> torch.Tensor:
    peak = audio_waveform.abs().max().item()
    if peak > 0:
        return audio_waveform / peak
    return audio_waveform


def save_audio(audio_waveform: torch.Tensor, sample_rate: int, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized_waveform = postprocess_audio(audio_waveform).clamp(-1.0, 1.0)
    int_waveform = (normalized_waveform.cpu().numpy() * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, int_waveform)
    return path


def generate_sine_wave_dataset(
    sample_rate: int,
    duration: float,
    frequencies: list[float],
    examples_per_frequency: int,
) -> list[torch.Tensor]:
    total_samples = int(sample_rate * duration)
    timeline = torch.linspace(0, duration, total_samples, dtype=torch.float32)
    waveforms: list[torch.Tensor] = []

    for frequency in frequencies:
        for example_index in range(examples_per_frequency):
            phase = (example_index / max(examples_per_frequency, 1)) * torch.pi
            audio_waveform = torch.sin(2 * torch.pi * frequency * timeline + phase)
            waveforms.append(postprocess_audio(audio_waveform))

    return waveforms
