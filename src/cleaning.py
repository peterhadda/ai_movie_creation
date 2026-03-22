from __future__ import annotations

from datetime import datetime
from typing import Any

from src.utils import deep_copy_records, safe_cast


DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%B %d %Y", "%Y.%m.%d")


def remove_duplicates(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    for record in records:
        signature = tuple(sorted(record.items()))
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(record)

    return deduplicated, len(records) - len(deduplicated)


def fill_missing_values(
    records: list[dict[str, Any]], strategy: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    filled_records = deep_copy_records(records)
    fill_counts: dict[str, int] = {}

    for record in filled_records:
        for field, fill_value in strategy.items():
            value = record.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                record[field] = fill_value
                fill_counts[field] = fill_counts.get(field, 0) + 1

    return filled_records, fill_counts


def strip_text_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = deep_copy_records(records)
    for record in cleaned:
        for key, value in record.items():
            if isinstance(value, str):
                record[key] = value.strip()
    return cleaned


def standardize_date_formats(records: list[dict[str, Any]], date_fields: list[str]) -> list[dict[str, Any]]:
    cleaned = deep_copy_records(records)
    for record in cleaned:
        for field in date_fields:
            value = record.get(field)
            if not value:
                continue
            for date_format in DATE_FORMATS:
                try:
                    record[field] = datetime.strptime(str(value).strip(), date_format).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
    return cleaned


def clean_numeric_fields(records: list[dict[str, Any]], numeric_fields: dict[str, str]) -> list[dict[str, Any]]:
    cleaned = deep_copy_records(records)
    for record in cleaned:
        for field, numeric_type in numeric_fields.items():
            value = record.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = value.replace(",", "").replace("_", "").strip()
            else:
                normalized = value

            if numeric_type == "int":
                cast_value = safe_cast(normalized, int)
            else:
                cast_value = safe_cast(normalized, float)

            if cast_value is not None:
                record[field] = cast_value
    return cleaned


def clean_dataset(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stripped = strip_text_fields(records)
    deduplicated, duplicates_removed = remove_duplicates(stripped)
    filled, fill_counts = fill_missing_values(deduplicated, config.get("fill_missing", {}))
    numeric_cleaned = clean_numeric_fields(filled, config.get("numeric_fields", {}))
    date_cleaned = standardize_date_formats(numeric_cleaned, config.get("date_fields", []))

    report = {
        "duplicates_removed": duplicates_removed,
        "missing_values_filled": fill_counts,
    }
    return date_cleaned, report
