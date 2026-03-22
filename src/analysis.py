from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any


def count_total_records(records: list[dict[str, Any]]) -> int:
    return len(records)


def summarize_missing_values(records: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        for field, value in record.items():
            if value is None or (isinstance(value, str) and value.strip() == ""):
                summary[field] = summary.get(field, 0) + 1
    return summary


def compute_basic_statistics(records: list[dict[str, Any]], numeric_fields: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for field in numeric_fields:
        values = [record[field] for record in records if isinstance(record.get(field), (int, float))]
        if not values:
            continue
        stats[field] = {
            "min": min(values),
            "max": max(values),
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
        }
    return stats


def count_by_category(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter()
    for record in records:
        value = record.get(field)
        if value is not None:
            counter[str(value)] += 1
    return dict(counter)


def generate_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_records": count_total_records(records),
        "missing_values": summarize_missing_values(records),
        "statistics": compute_basic_statistics(records, config.get("numeric_summary_fields", [])),
        "category_counts": count_by_category(records, config.get("category_field", "category")),
    }
