from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

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
