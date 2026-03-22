from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.utils import normalize_record


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
