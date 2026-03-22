from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_directory_exists(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_message(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] [{level.upper()}] {message}")


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    if value is None:
        return default

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return default
        value = stripped

    try:
        return target_type(value)
    except (TypeError, ValueError):
        return default


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        clean_key = str(key).strip()
        normalized[clean_key] = value.strip() if isinstance(value, str) else value
    return normalized


def deep_copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deepcopy(records)
