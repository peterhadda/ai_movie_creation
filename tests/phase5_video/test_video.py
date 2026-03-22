import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.phase5_video.dataset import (
    build_video_dataset,
    create_frame_sequences,
    create_video_dataloader,
    split_video_dataset,
)
from src.phase5_video.evaluate import (
    compute_frame_reconstruction_loss,
    generate_video_evaluation_report,
    measure_temporal_consistency,
)
from src.phase5_video.frame_processing import (
    convert_frames_to_tensor,
    resize_frames,
    save_extracted_frames,
)
from src.phase5_video.generate import assemble_video_from_frames, generate_frame_sequence, generate_next_frame
from src.phase5_video.pipeline import run_video_generation_pipeline
from src.phase5_video.model import initialize_video_model
from src.phase5_video.train import (
    initialize_loss_function,
    initialize_optimizer,
    run_video_training_loop,
    train_one_epoch,
    train_one_video_batch,
    validate_one_epoch,
)
from src.phase5_video.video_ingestion import extract_frames, extract_video_metadata, generate_moving_square_video


class VideoGenerationTests(unittest.TestCase):
    def test_video_ingestion_and_frame_processing(self) -> None:
        frames = generate_moving_square_video(frame_count=8, frame_size=(24, 24))
        resized_frames = resize_frames(frames, (16, 16))
        frame_tensor = convert_frames_to_tensor(resized_frames)

        self.assertEqual(len(frames), 8)
        self.assertEqual(resized_frames[0].size, (16, 16))
        self.assertEqual(frame_tensor.shape, (8, 3, 16, 16))

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_extracted_frames(resized_frames, Path(temp_dir) / "frames")
            self.assertEqual(len(saved_paths), 8)

    def test_gif_extraction_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "sample.gif"
            frames = generate_moving_square_video(frame_count=5, frame_size=(16, 16))
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            extracted = extract_frames(gif_path)
            metadata = extract_video_metadata(gif_path)

            self.assertEqual(len(extracted), 5)
            self.assertEqual(metadata["frame_count"], 5)

    def test_dataset_model_training_generation_and_evaluation(self) -> None:
        frames = generate_moving_square_video(frame_count=12, frame_size=(16, 16))
        frame_tensor = convert_frames_to_tensor(frames)
        frame_sequences, target_frames = create_frame_sequences(frame_tensor, sequence_length=4)
        dataset = build_video_dataset(frame_sequences, target_frames)
        train_dataset, val_dataset, test_dataset = split_video_dataset(dataset, 0.7, 0.15, 0.15)
        train_loader = create_video_dataloader(train_dataset, batch_size=2, shuffle=True)
        val_loader = create_video_dataloader(val_dataset, batch_size=2, shuffle=False)
        test_loader = create_video_dataloader(test_dataset, batch_size=2, shuffle=False)

        model = initialize_video_model(input_shape=(3, 16, 16), hidden_size=32)
        loss_fn = initialize_loss_function()
        optimizer = initialize_optimizer(model, learning_rate=0.001)

        batch_sequences, batch_targets = next(iter(train_loader))
        batch_loss = train_one_video_batch(model, batch_sequences, batch_targets, loss_fn, optimizer)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validate_one_epoch(model, val_loader, loss_fn)
        trained_model, history = run_video_training_loop(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=2,
        )

        seed_sequence = frame_sequences[0]
        predicted_next_frame = generate_next_frame(trained_model, seed_sequence)
        generated_frames = generate_frame_sequence(trained_model, seed_sequence, num_future_frames=3)

        self.assertGreaterEqual(batch_loss, 0.0)
        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertEqual(len(history), 2)
        self.assertEqual(predicted_next_frame.shape, (3, 16, 16))
        self.assertEqual(generated_frames.shape, (3, 3, 16, 16))

        test_batch_sequences, test_batch_targets = next(iter(test_loader))
        with torch.no_grad():
            test_predictions = trained_model(test_batch_sequences)
        metrics = {
            "frame_reconstruction_loss": compute_frame_reconstruction_loss(test_batch_targets, test_predictions),
            "temporal_consistency": measure_temporal_consistency(generated_frames),
        }
        report = generate_video_evaluation_report(metrics)
        self.assertIn("frame_reconstruction_loss", report)

        with tempfile.TemporaryDirectory() as temp_dir:
            gif_path = Path(temp_dir) / "generated.gif"
            assembled_path = assemble_video_from_frames(generated_frames, fps=8, output_path=gif_path)
            self.assertTrue(assembled_path.exists())

    def test_run_video_generation_pipeline_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            config = {
                "phase5": {
                    "video_path": str(temp_dir_path / "missing.gif"),
                    "frame_size": [16, 16],
                    "fps": 8,
                    "sequence_length": 4,
                    "hidden_size": 32,
                    "learning_rate": 0.001,
                    "num_epochs": 2,
                    "batch_size": 2,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "generated_future_frames": 3,
                    "synthetic_frame_count": 12,
                    "square_size": 4,
                    "step_size": 2,
                    "output": {
                        "frames_dir": str(temp_dir_path / "outputs" / "frames"),
                        "model": str(temp_dir_path / "models" / "video_prediction_model.pt"),
                        "training_history": str(temp_dir_path / "reports" / "video_training_history.json"),
                        "report": str(temp_dir_path / "reports" / "video_evaluation_report.json"),
                        "generated_video": str(temp_dir_path / "outputs" / "generated_video.gif")
                    }
                }
            }

            config_path = temp_dir_path / "config_phase5.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_video_generation_pipeline(config_path)

            self.assertIn("generated_frames", result)
            self.assertTrue((temp_dir_path / "models" / "video_prediction_model.pt").exists())
            self.assertTrue((temp_dir_path / "reports" / "video_training_history.json").exists())
            self.assertTrue((temp_dir_path / "reports" / "video_evaluation_report.json").exists())
            self.assertTrue((temp_dir_path / "outputs" / "generated_video.gif").exists())
