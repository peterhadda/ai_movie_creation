from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.common.utils import load_config, log_message
from src.phase5.dataset import (
    build_video_dataset,
    create_frame_sequences,
    create_video_dataloader,
    split_video_dataset,
)
from src.phase5.evaluate import (
    compute_frame_reconstruction_loss,
    generate_video_evaluation_report,
    measure_temporal_consistency,
)
from src.phase5.frame_processing import (
    convert_frames_to_tensor,
    resize_frames,
    save_extracted_frames,
)
from src.phase5.generate import assemble_video_from_frames, generate_frame_sequence, generate_next_frame
from src.phase5.model import initialize_video_model
from src.phase5.storage import save_evaluation_report, save_training_history, save_video_model
from src.phase5.train import (
    initialize_loss_function,
    initialize_optimizer,
    run_video_training_loop,
)
from src.phase5.video_ingestion import extract_frames, extract_video_metadata, generate_moving_square_video


def run_video_generation_pipeline(config_path: str | Path = "config_phase5.json") -> dict[str, Any]:
    config = load_config(config_path)["phase5"]
    raw_video_path = Path(config.get("video_path", "data/raw/video/sample.gif"))
    frame_size = tuple(config.get("frame_size", [32, 32]))
    sequence_length = config.get("sequence_length", 4)
    hidden_size = config.get("hidden_size", 128)
    learning_rate = config.get("learning_rate", 0.001)
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 4)
    fps = config.get("fps", 8)
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    generated_future_frames = config.get("generated_future_frames", 6)

    if raw_video_path.exists():
        log_message("Loading video frames from source file")
        video_frames = extract_frames(raw_video_path)
        metadata = extract_video_metadata(raw_video_path)
    else:
        log_message("No source video found. Generating synthetic moving-square clip")
        video_frames = generate_moving_square_video(
            frame_count=config.get("synthetic_frame_count", 20),
            frame_size=frame_size,
            square_size=config.get("square_size", 8),
            step_size=config.get("step_size", 2),
        )
        metadata = {
            "fps": fps,
            "frame_count": len(video_frames),
            "frame_size": frame_size,
        }

    resized_frames = resize_frames(video_frames, frame_size)
    frame_tensor = convert_frames_to_tensor(resized_frames)
    saved_frame_paths = save_extracted_frames(
        resized_frames,
        config["output"]["frames_dir"],
    )

    frame_sequences, target_frames = create_frame_sequences(frame_tensor, sequence_length)
    dataset = build_video_dataset(frame_sequences, target_frames)
    train_dataset, val_dataset, test_dataset = split_video_dataset(dataset, train_ratio, val_ratio, test_ratio)
    train_loader = create_video_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_video_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_video_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    input_shape = tuple(frame_tensor.shape[1:])
    model = initialize_video_model(input_shape=input_shape, hidden_size=hidden_size)
    loss_fn = initialize_loss_function()
    optimizer = initialize_optimizer(model, learning_rate)

    log_message("Training video prediction model")
    trained_model, training_history = run_video_training_loop(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        num_epochs,
    )

    seed_sequence = frame_sequences[0]
    predicted_next_frame = generate_next_frame(trained_model, seed_sequence)
    generated_frames = generate_frame_sequence(trained_model, seed_sequence, generated_future_frames)

    test_batch_sequences, test_batch_targets = next(iter(test_loader))
    with torch.no_grad():
        test_predictions = trained_model(test_batch_sequences)

    metrics = {
        "frame_reconstruction_loss": compute_frame_reconstruction_loss(test_batch_targets, test_predictions),
        "temporal_consistency": measure_temporal_consistency(generated_frames),
    }
    evaluation_report = generate_video_evaluation_report(metrics)

    model_path = save_video_model(
        trained_model,
        config["output"]["model"],
        input_shape=input_shape,
        hidden_size=hidden_size,
    )
    history_path = save_training_history(training_history, config["output"]["training_history"])
    report_path = save_evaluation_report(evaluation_report, config["output"]["report"])
    video_output_path = assemble_video_from_frames(generated_frames, fps=fps, output_path=config["output"]["generated_video"])

    return {
        "video_frames": video_frames,
        "frame_sequences": frame_sequences,
        "target_frames": target_frames,
        "predicted_next_frame": predicted_next_frame,
        "generated_frames": generated_frames,
        "evaluation_report": evaluation_report,
        "training_history": training_history,
        "metadata": metadata,
        "saved_frame_paths": saved_frame_paths,
        "model_path": model_path,
        "history_path": history_path,
        "report_path": report_path,
        "generated_video_path": video_output_path,
    }
