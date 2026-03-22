from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def create_frame_sequences(
    video_frames: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_sequences: list[torch.Tensor] = []
    target_frames: list[torch.Tensor] = []

    for start_index in range(video_frames.shape[0] - sequence_length):
        frame_sequence = video_frames[start_index : start_index + sequence_length]
        target_frame = video_frames[start_index + sequence_length]
        frame_sequences.append(frame_sequence)
        target_frames.append(target_frame)

    if not frame_sequences:
        raise ValueError("Not enough frames to create sequences")

    return torch.stack(frame_sequences), torch.stack(target_frames)


def build_video_dataset(
    frame_sequences: torch.Tensor,
    target_frames: torch.Tensor,
) -> TensorDataset:
    return TensorDataset(frame_sequences, target_frames)


def split_video_dataset(
    dataset: TensorDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

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
    return train_dataset, val_dataset, test_dataset


def create_video_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[Any]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
