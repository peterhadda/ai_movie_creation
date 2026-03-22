from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def load_mnist_dataset(
    data_dir: str = "data/raw/mnist",
    train: bool = True,
    download: bool = True,
) -> Dataset[Any]:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )
    return dataset


def split_image_dataset(
    dataset: Dataset[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    dataset_size = len(dataset)
    if dataset_size < 3:
        raise ValueError("Dataset must contain at least 3 samples")

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


def create_image_data_loader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[Any]:
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def flatten_images(batch_images: torch.Tensor) -> torch.Tensor:
    return batch_images.view(batch_images.size(0), -1)
