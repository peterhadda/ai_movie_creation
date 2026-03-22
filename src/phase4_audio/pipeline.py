from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.common.config import load_config
from src.common.io_utils import ensure_directory_exists, save_evaluation_report, save_training_history
from src.common.logging_utils import log_message
from src.phase4_audio.data import (
    generate_sine_wave_dataset,
    load_audio,
    pad_audio,
    postprocess_audio,
    prepare_audio_sequences,
    save_audio,
    trim_audio,
    waveform_to_spectrogram,
)
from src.phase4_audio.generate import generate_audio, generated_frames_to_waveform
from src.phase4_audio.model import initialize_audio_model
from src.phase4_audio.train import initialize_audio_optimizer, run_audio_training_loop


def _load_waveforms_from_directory(audio_dir: str | Path, target_length: int) -> list[torch.Tensor]:
    waveforms: list[torch.Tensor] = []
    for audio_path in sorted(Path(audio_dir).glob("*.wav")):
        audio_waveform, _ = load_audio(audio_path)
        trimmed_audio = trim_audio(audio_waveform)
        padded_audio = pad_audio(trimmed_audio, target_length)
        waveforms.append(postprocess_audio(padded_audio))
    return waveforms


def _build_audio_loaders(
    input_sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], TensorDataset, TensorDataset, TensorDataset]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    dataset = TensorDataset(input_sequences, target_sequences)
    dataset_size = len(dataset)

    train_size = max(1, int(dataset_size * train_ratio))
    val_size = max(1, int(dataset_size * val_ratio))
    test_size = dataset_size - train_size - val_size

    if test_size < 1:
        test_size = 1
        if train_size >= val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def run_audio_generation_pipeline(config_path: str | Path = "configs/phase4.json") -> dict[str, Any]:
    config = load_config(config_path)["phase4"]

    sample_rate = config.get("sample_rate", 16000)
    duration = config.get("duration", 1.0)
    target_length = int(sample_rate * duration)
    n_fft = config.get("n_fft", 256)
    hop_length = config.get("hop_length", 128)
    sequence_length = config.get("sequence_length", 8)
    hidden_size = config.get("hidden_size", 64)
    learning_rate = config.get("learning_rate", 0.001)
    num_epochs = config.get("num_epochs", 5)
    batch_size = config.get("batch_size", 8)
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)

    audio_source_dir = Path(config.get("audio_dir", "data/raw/audio"))
    if audio_source_dir.exists() and list(audio_source_dir.glob("*.wav")):
        log_message("Loading audio waveforms from data directory")
        waveforms = _load_waveforms_from_directory(audio_source_dir, target_length)
    else:
        log_message("No audio files found. Generating synthetic sine-wave dataset")
        waveforms = generate_sine_wave_dataset(
            sample_rate=sample_rate,
            duration=duration,
            frequencies=config.get("synthetic_frequencies", [220.0, 440.0, 880.0]),
            examples_per_frequency=config.get("examples_per_frequency", 4),
        )

    spectrograms = [
        waveform_to_spectrogram(audio_waveform, n_fft=n_fft, hop_length=hop_length)
        for audio_waveform in waveforms
    ]

    input_sequences, target_sequences = prepare_audio_sequences(
        spectrograms,
        sequence_length=sequence_length,
    )

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = _build_audio_loaders(
        input_sequences,
        target_sequences,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    input_size = input_sequences.shape[2]
    output_size = target_sequences.shape[2]

    log_message("Initializing audio sequence model")
    model = initialize_audio_model(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer = initialize_audio_optimizer(model, learning_rate=learning_rate)

    log_message("Training audio model")
    trained_model, training_history = run_audio_training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=num_epochs,
    )

    seed_sequence = input_sequences[0]
    generated_frames = generate_audio(
        trained_model,
        seed_sequence=seed_sequence,
        length=config.get("generation_length", 12),
    )
    generated_waveform = generated_frames_to_waveform(
        generated_frames,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    model_path = Path(config["output"]["model"])
    history_path = Path(config["output"]["training_history"])
    generated_audio_path = Path(config["output"]["generated_audio"])
    evaluation_path = Path(config["output"]["report"])

    ensure_directory_exists(model_path.parent)
    torch.save(
        {
            "state_dict": trained_model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "sequence_length": sequence_length,
            "sample_rate": sample_rate,
        },
        model_path,
    )
    save_training_history(training_history, history_path)
    save_audio(generated_waveform, sample_rate, generated_audio_path)

    evaluation_report = {
        "sample_rate": sample_rate,
        "waveform_count": len(waveforms),
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "test_sequences": len(test_dataset),
        "generated_audio_path": str(generated_audio_path),
    }
    save_evaluation_report(evaluation_report, evaluation_path)

    return {
        "trained_model": trained_model,
        "training_history": training_history,
        "generated_audio": generated_waveform,
        "generated_audio_path": generated_audio_path,
        "report_path": evaluation_path,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
