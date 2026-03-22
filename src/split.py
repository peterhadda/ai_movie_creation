from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Any

from sklearn.model_selection import train_test_split

test_size = 0.2
X_train: list[list[Any]] = []
X_test: list[list[Any]] = []
y_train: list[Any] = []
y_test: list[Any] = []


def split_train_test(
    X: list[list[Any]],
    y: list[Any],
    test_size: float = test_size,
) -> tuple[list[list[Any]], list[list[Any]], list[Any], list[Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


def check_class_distribution(y: list[Any]) -> dict[Any, int]:
    class_counts = dict(Counter(y))
    return class_counts


def stratified_split(
    X: list[list[Any]],
    y: list[Any],
    test_size: float = test_size,
) -> tuple[list[list[Any]], list[list[Any]], list[Any], list[Any]]:
    class_counts = check_class_distribution(y)
    number_of_classes = len(class_counts)
    test_count = ceil(len(y) * test_size)
    train_count = len(y) - test_count
    can_stratify = (
        number_of_classes > 1
        and min(class_counts.values()) >= 2
        and test_count >= number_of_classes
        and train_count >= number_of_classes
    )

    if not can_stratify:
        return split_train_test(X, y, test_size=test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
