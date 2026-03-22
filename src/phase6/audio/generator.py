from __future__ import annotations

from pathlib import Path

import torch

from src.phase4.data import postprocess_audio, save_audio
from src.phase4.generate import generate_audio as generate_audio_frames
from src.phase4.generate import generated_frames_to_waveform
from src.phase4.model import initialize_audio_model


def encode_audio(audio_input: torch.Tensor) -> torch.Tensor:
    flattened_audio = audio_input.flatten()
    audio_embedding = flattened_audio / flattened_audio.norm().clamp_min(1e-8)
    return audio_embedding


def _project_embedding(shared_embedding: torch.Tensor, output_dim: int) -> torch.Tensor:
    repeats = (output_dim + shared_embedding.numel() - 1) // shared_embedding.numel()
    return shared_embedding.repeat(repeats)[:output_dim]


def _fallback_waveform(shared_embedding: torch.Tensor, sample_rate: int, duration: float) -> torch.Tensor:
    total_samples = int(sample_rate * duration)
    timeline = torch.linspace(0, duration, total_samples, dtype=torch.float32)
    base_frequency = 220.0 + float(shared_embedding[0].abs().item() * 440.0)
    overtone = 440.0 + float(shared_embedding[1].abs().item() * 660.0) if shared_embedding.numel() > 1 else 440.0
    waveform = 0.6 * torch.sin(2 * torch.pi * base_frequency * timeline)
    waveform += 0.4 * torch.sin(2 * torch.pi * overtone * timeline)
    return postprocess_audio(waveform)


def generate_audio(
    shared_embedding: torch.Tensor,
    model_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    duration: float = 1.0,
) -> Path:
    model_file = Path(model_path)

    if model_file.exists():
        checkpoint = torch.load(model_file, map_location="cpu")
        model = initialize_audio_model(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            output_size=checkpoint["output_size"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        sequence_length = checkpoint.get("sequence_length", 8)
        n_fft = checkpoint.get("n_fft", 256)
        hop_length = checkpoint.get("hop_length", 128)
        sample_rate = checkpoint.get("sample_rate", sample_rate)
        seed_sequence = _project_embedding(shared_embedding, sequence_length * checkpoint["input_size"]).view(
            sequence_length,
            checkpoint["input_size"],
        )
        generated_audio = generate_audio_frames(model, seed_sequence, length=12)
        waveform = generated_frames_to_waveform(
            generated_audio,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    else:
        waveform = _fallback_waveform(shared_embedding, sample_rate, duration)

    return save_audio(postprocess_audio(waveform), sample_rate, output_path)
