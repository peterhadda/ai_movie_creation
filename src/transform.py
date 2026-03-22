from __future__ import annotations

from typing import Any

from src.utils import deep_copy_records, safe_cast


SIZE_UNITS = {
    "bytes": 1,
    "kb": 1024,
    "mb": 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
}


def rename_columns(records: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for record in records:
        updated = {}
        for key, value in record.items():
            updated[mapping.get(key, key)] = value
        renamed.append(updated)
    return renamed


def normalize_categories(
    records: list[dict[str, Any]], field: str, mapping: dict[str, str]
) -> list[dict[str, Any]]:
    normalized = deep_copy_records(records)
    for record in normalized:
        value = record.get(field)
        if isinstance(value, str):
            record[field] = mapping.get(value.strip().lower(), value.strip().lower())
    return normalized


def _convert_size_to_bytes(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return value

    normalized = value.replace(",", "").strip().lower()
    parts = normalized.split()
    if len(parts) == 1:
        numeric = safe_cast(parts[0], float)
        return int(numeric) if numeric is not None else value

    numeric = safe_cast(parts[0], float)
    multiplier = SIZE_UNITS.get(parts[1])
    if numeric is None or multiplier is None:
        return value
    return int(numeric * multiplier)


def convert_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = deep_copy_records(records)
    for record in converted:
        if "size" in record:
            record["size_bytes"] = _convert_size_to_bytes(record["size"])
        if "duration" in record:
            duration = record["duration"]
            if isinstance(duration, str):
                numeric = safe_cast(duration, float)
                if numeric is not None:
                    record["duration_seconds"] = numeric
            elif isinstance(duration, (int, float)):
                record["duration_seconds"] = float(duration)
    return converted


def select_relevant_fields(records: list[dict[str, Any]], selected_fields: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        selected.append({field: record.get(field) for field in selected_fields})
    return selected


def transform_dataset(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    renamed = rename_columns(records, config.get("rename_columns", {}))
    normalized = normalize_categories(
        renamed,
        config.get("category_field", "category"),
        config.get("category_mapping", {}),
    )
    converted = convert_units(normalized)
    return select_relevant_fields(converted, config.get("selected_fields", []))
