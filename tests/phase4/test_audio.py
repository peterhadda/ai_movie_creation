import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.phase4.data import (
    frame_vectors_to_spectrogram,
    generate_sine_wave_dataset,
    load_audio,
    pad_audio,
    postprocess_audio,
    prepare_audio_sequences,
    save_audio,
    spectrogram_to_frame_vectors,
    spectrogram_to_waveform,
    trim_audio,
    waveform_to_spectrogram,
)
from src.phase4.generate import generate_audio, generated_frames_to_waveform
from src.phase4.main import run_audio_generation_pipeline
from src.phase4.model import initialize_audio_model
from src.phase4.train import (
    compute_audio_loss,
    initialize_audio_optimizer,
    run_audio_training_loop,
    train_audio_model,
    validate_audio_model,
)


class AudioGenerationTests(unittest.TestCase):
    def test_audio_load_trim_pad_and_save_round_trip(self) -> None:
        waveform = torch.cat([torch.zeros(50), torch.ones(100) * 0.5, torch.zeros(50)])

        trimmed_audio = trim_audio(waveform, threshold=0.1)
        padded_audio = pad_audio(trimmed_audio, target_length=160)
        processed_audio = postprocess_audio(padded_audio)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            save_audio(processed_audio, sample_rate=8000, output_path=audio_path)
            loaded_waveform, sample_rate = load_audio(audio_path)

            self.assertEqual(sample_rate, 8000)
            self.assertEqual(trimmed_audio.numel(), 100)
            self.assertEqual(padded_audio.numel(), 160)
            self.assertEqual(loaded_waveform.ndim, 1)

    def test_spectrogram_conversion_and_sequence_prep(self) -> None:
        waveform = torch.sin(torch.linspace(0, 8 * torch.pi, 512))
        spectrogram = waveform_to_spectrogram(waveform, n_fft=64, hop_length=32)
        reconstructed_waveform = spectrogram_to_waveform(
            spectrogram,
            n_fft=64,
            hop_length=32,
            length=waveform.numel(),
        )
        frame_vectors = spectrogram_to_frame_vectors(spectrogram)
        restored_spectrogram = frame_vectors_to_spectrogram(frame_vectors)
        input_sequences, target_sequences = prepare_audio_sequences([spectrogram], sequence_length=4)

        self.assertEqual(spectrogram.shape, restored_spectrogram.shape)
        self.assertEqual(reconstructed_waveform.shape[0], waveform.shape[0])
        self.assertEqual(input_sequences.ndim, 3)
        self.assertEqual(target_sequences.ndim, 3)

    def test_audio_training_and_generation(self) -> None:
        waveforms = generate_sine_wave_dataset(
            sample_rate=4000,
            duration=0.2,
            frequencies=[220.0, 440.0],
            examples_per_frequency=2,
        )
        spectrograms = [waveform_to_spectrogram(waveform, n_fft=64, hop_length=32) for waveform in waveforms]
        input_sequences, target_sequences = prepare_audio_sequences(spectrograms, sequence_length=4)

        dataset = torch.utils.data.TensorDataset(input_sequences, target_sequences)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        model = initialize_audio_model(
            input_size=input_sequences.shape[2],
            hidden_size=16,
            output_size=target_sequences.shape[2],
        )
        optimizer = initialize_audio_optimizer(model, learning_rate=0.001)

        train_loss = train_audio_model(model, loader, optimizer)
        val_loss = validate_audio_model(model, loader)
        trained_model, history = run_audio_training_loop(model, loader, loader, optimizer, num_epochs=2)
        seed_sequence = input_sequences[0]
        generated_frames = generate_audio(trained_model, seed_sequence, length=3)
        generated_waveform = generated_frames_to_waveform(generated_frames, n_fft=64, hop_length=32)
        loss = compute_audio_loss(model(input_sequences[:1]), target_sequences[:1])

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertEqual(len(history), 2)
        self.assertEqual(generated_frames.ndim, 2)
        self.assertEqual(generated_waveform.ndim, 1)
        self.assertEqual(loss.ndim, 0)

    def test_run_audio_generation_pipeline_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            config = {
                "phase4": {
                    "audio_dir": str(temp_dir_path / "audio"),
                    "sample_rate": 4000,
                    "duration": 0.2,
                    "n_fft": 64,
                    "hop_length": 32,
                    "sequence_length": 4,
                    "hidden_size": 16,
                    "learning_rate": 0.001,
                    "num_epochs": 2,
                    "batch_size": 2,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "synthetic_frequencies": [220.0, 440.0],
                    "examples_per_frequency": 2,
                    "generation_length": 4,
                    "output": {
                        "model": str(temp_dir_path / "models" / "audio_sequence_model.pt"),
                        "training_history": str(temp_dir_path / "reports" / "audio_training_history.json"),
                        "generated_audio": str(temp_dir_path / "reports" / "generated_audio.wav"),
                        "report": str(temp_dir_path / "reports" / "audio_generation_report.json")
                    }
                }
            }

            config_path = temp_dir_path / "config_phase4.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_audio_generation_pipeline(config_path)

            self.assertIn("generated_audio", result)
            self.assertTrue((temp_dir_path / "models" / "audio_sequence_model.pt").exists())
            self.assertTrue((temp_dir_path / "reports" / "audio_training_history.json").exists())
            self.assertTrue((temp_dir_path / "reports" / "generated_audio.wav").exists())
            self.assertTrue((temp_dir_path / "reports" / "audio_generation_report.json").exists())
