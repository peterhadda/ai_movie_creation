from __future__ import annotations

from typing import Any

from src.model import fit_model, initialize_model
from src.split import stratified_split

model_config = {"model_type": "logistic_regression"}
trained_model: Any = None


def train_baseline_model(X_train: list[list[Any]], y_train: list[Any]) -> Any:
    model = initialize_model(model_config["model_type"])
    trained_model = fit_model(model, X_train, y_train)
    return trained_model


def run_training_pipeline(X: list[list[Any]], y: list[Any]) -> tuple[Any, list[list[Any]], list[Any]]:
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    if len(set(y_train)) < 2:
        X_train = X
        y_train = y
        X_test = X
        y_test = y

    trained_model = train_baseline_model(X_train, y_train)
    return trained_model, X_test, y_test
