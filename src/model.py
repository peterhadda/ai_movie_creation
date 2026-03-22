from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

from src.storage import load_model_artifact, save_model_artifact

model_type = "logistic_regression"
model: LogisticRegression | None = None


def initialize_model(model_type: str = model_type) -> LogisticRegression:
    if model_type != "logistic_regression":
        raise ValueError(f"Unsupported model_type: {model_type}")
    return LogisticRegression(max_iter=1000, random_state=42)


def fit_model(
    model: LogisticRegression,
    X_train: list[list[Any]],
    y_train: list[Any],
) -> LogisticRegression:
    trained_model = model.fit(X_train, y_train)
    return trained_model


def predict_labels(model: LogisticRegression, X: list[list[Any]]) -> list[Any]:
    y_pred = model.predict(X).tolist()
    return y_pred


def predict_probabilities(model: LogisticRegression, X: list[list[Any]]) -> list[list[float]]:
    y_proba = model.predict_proba(X).tolist()
    return y_proba


def save_model(model: LogisticRegression, model_path: str | Path) -> None:
    save_model_artifact(model, model_path)


def load_model(model_path: str | Path) -> LogisticRegression:
    model = load_model_artifact(model_path)
    return model
