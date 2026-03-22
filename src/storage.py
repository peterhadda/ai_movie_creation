from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from src.utils import ensure_directory_exists


def save_as_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory_exists(output.parent)

    fieldnames = list(records[0].keys()) if records else []
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output


def save_as_json(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    return output


def save_processing_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return output


def save_model_artifact(model: Any, output_path: str | Path) -> Path:
    output = Path(output_path)
    ensure_directory_exists(output.parent)

    torch.save(model, output)
    return output


def load_model_artifact(model_path: str | Path) -> Any:
    return torch.load(Path(model_path), map_location="cpu")


def save_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return output


def save_predictions(predictions: list[Any], path: str | Path) -> Path:
    output = Path(path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2)
    return output


def save_training_history(training_history: list[dict[str, float]], history_path: str | Path) -> Path:
    output = Path(history_path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(training_history, file, indent=2)
    return output
