from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

X_tensor: torch.Tensor | None = None
y_tensor: torch.Tensor | None = None
train_dataset: TensorDataset | None = None
val_dataset: TensorDataset | None = None
test_dataset: TensorDataset | None = None
train_loader: DataLoader[Any] | None = None
val_loader: DataLoader[Any] | None = None
test_loader: DataLoader[Any] | None = None


def convert_features_to_tensor(X: list[list[float]]) -> torch.Tensor:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor


def convert_target_to_tensor(y: list[int]) -> torch.Tensor:
    y_tensor = torch.tensor(y, dtype=torch.long)
    return y_tensor


def create_tensor_dataset(X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> TensorDataset:
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset


def split_dataset(
    dataset: TensorDataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    dataset_size = len(dataset)
    if dataset_size < 3:
        raise ValueError("Dataset must contain at least 3 rows for train/val/test splitting")

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


def create_data_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[Any]:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
