from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


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


def _ensure_list_of_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [normalize_record(record) for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return [normalize_record(record) for record in data["records"] if isinstance(record, dict)]
        return [normalize_record(data)]
    raise ValueError("Expected a JSON object or a list of objects.")


def read_csv_file(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty: {csv_path}")
        records = [normalize_record(row) for row in reader]

    if not records:
        raise ValueError(f"CSV file has no data rows: {csv_path}")
    return records


def read_json_file(path: str | Path) -> list[dict[str, Any]]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _ensure_list_of_records(data)


def fetch_data_from_api(url: str, timeout: int = 10) -> list[dict[str, Any]]:
    try:
        with urlopen(url, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            if status_code != 200:
                raise ValueError(f"API request failed with status code {status_code}")
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        raise ValueError(f"API request failed with status code {error.code}") from error
    except URLError as error:
        raise ValueError(f"API request failed: {error.reason}") from error

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("API returned invalid JSON") from error

    return _ensure_list_of_records(data)


def load_raw_data(source_type: str, source_path_or_url: str) -> list[dict[str, Any]]:
    source = source_type.strip().lower()
    if source == "csv":
        return read_csv_file(source_path_or_url)
    if source == "json":
        return read_json_file(source_path_or_url)
    if source == "api":
        return fetch_data_from_api(source_path_or_url)
    raise ValueError(f"Unsupported source type: {source_type}")


def check_required_fields(record: dict[str, Any], required_fields: list[str]) -> list[str]:
    missing_fields: list[str] = []
    for field in required_fields:
        value = record.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing_fields.append(field)
    return missing_fields


def check_data_types(record: dict[str, Any], schema: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for field, expected_type in schema.items():
        value = record.get(field)
        if value is None or value == "":
            continue

        if expected_type == "str":
            continue
        if expected_type == "int" and safe_cast(value, int) is None:
            issues.append(f"{field}: expected int")
        elif expected_type == "float" and safe_cast(value, float) is None:
            issues.append(f"{field}: expected float")
    return issues


def validate_record(
    record: dict[str, Any],
    schema: dict[str, str],
    required_fields: list[str],
) -> dict[str, Any]:
    missing_fields = check_required_fields(record, required_fields)
    type_issues = check_data_types(record, schema)
    issues = [f"missing required field: {field}" for field in missing_fields] + type_issues
    return {"is_valid": not issues, "issues": issues, "record": record}


def validate_dataset(
    records: list[dict[str, Any]],
    schema: dict[str, str],
    required_fields: list[str],
) -> dict[str, Any]:
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}

    for record in records:
        result = validate_record(record, schema, required_fields)
        if result["is_valid"]:
            valid_records.append(record)
            continue

        invalid_records.append({"record": record, "issues": result["issues"]})
        for issue in result["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    return {
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "issue_counts": issue_counts,
    }


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
    records: list[dict[str, Any]],
    strategy: dict[str, Any],
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


DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%B %d %Y", "%Y.%m.%d")


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


def save_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return output


def save_training_history(training_history: list[dict[str, float]], history_path: str | Path) -> Path:
    output = Path(history_path)
    ensure_directory_exists(output.parent)

    with output.open("w", encoding="utf-8") as file:
        json.dump(training_history, file, indent=2)
    return output
