from __future__ import annotations

from typing import Any

from src.model import predict_labels

new_record: list[Any] | dict[str, Any] | None = None
predictions: list[Any] = []


def _normalize_records(records: list[list[Any]] | list[dict[str, Any]]) -> list[list[Any]]:
    if not records:
        return []
    if isinstance(records[0], dict):
        return [list(record.values()) for record in records]
    return records


def predict_single_record(model: Any, record: list[Any] | dict[str, Any]) -> Any:
    normalized_record = [list(record.values())] if isinstance(record, dict) else [record]
    prediction = predict_labels(model, normalized_record)[0]
    return prediction


def predict_batch(model: Any, records: list[list[Any]] | list[dict[str, Any]]) -> list[Any]:
    normalized_records = _normalize_records(records)
    predictions = predict_labels(model, normalized_records)
    return predictions


def format_prediction_output(predictions: list[Any]) -> list[dict[str, Any]]:
    formatted_predictions = [
        {"record_index": index, "prediction": prediction}
        for index, prediction in enumerate(predictions)
    ]
    return formatted_predictions
