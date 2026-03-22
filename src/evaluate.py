from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.model import predict_labels

y_true: list[Any] = []
y_pred: list[Any] = []


def compute_accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_recall(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_f1_score(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def build_confusion_matrix(y_true: list[Any], y_pred: list[Any]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).tolist()


def evaluate_model(model: Any, X_test: list[list[Any]], y_test: list[Any]) -> dict[str, Any]:
    y_pred = predict_labels(model, X_test)
    evaluation_report = {
        "accuracy_score": compute_accuracy(y_test, y_pred),
        "precision_score": compute_precision(y_test, y_pred),
        "recall_score": compute_recall(y_test, y_pred),
        "f1_score": compute_f1_score(y_test, y_pred),
        "confusion_matrix": build_confusion_matrix(y_test, y_pred),
        "y_true": y_test,
        "y_pred": y_pred,
    }
    return evaluation_report
