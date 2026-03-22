from __future__ import annotations

from typing import Any

from src.common.utils import deep_copy_records, safe_cast


SIZE_UNITS = {
    "bytes": 1,
    "kb": 1024,
    "mb": 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
}


def _print_feature_debug(title: str, value: Any) -> None:
    print(f"\n--- {title} ---")
    if isinstance(value, list):
        for item in value:
            print(item)
        return
    print(value)


def rename_columns(records: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for record in records:
        updated = {}
        for key, value in record.items():
            updated[mapping.get(key, key)] = value
        renamed.append(updated)
    return renamed


def normalize_categories(
    records: list[dict[str, Any]],
    field: str,
    mapping: dict[str, str],
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


def select_feature_columns(
    records: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    return [{column: record.get(column) for column in feature_columns} for record in records]


def select_target_column(records: list[dict[str, Any]], target_column: str) -> list[Any]:
    missing_target_rows = [
        index for index, record in enumerate(records) if record.get(target_column) is None
    ]
    if missing_target_rows:
        raise ValueError(
            f"Target column '{target_column}' is missing for rows: {missing_target_rows}"
        )
    return [record[target_column] for record in records]


def encode_categorical_features(
    records: list[dict[str, Any]],
    categorical_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    encoded_records = [record.copy() for record in records]
    encoders: dict[str, dict[str, int]] = {}

    for column in categorical_columns:
        categories = sorted(
            {
                str(record.get(column))
                for record in encoded_records
                if record.get(column) is not None
            }
        )
        column_encoder = {category: index for index, category in enumerate(categories)}
        encoders[column] = column_encoder

        for record in encoded_records:
            value = record.get(column)
            if value is not None:
                record[column] = column_encoder[str(value)]

    return encoded_records, encoders


def scale_numeric_features(
    records: list[dict[str, Any]],
    numeric_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    scaled_records = [record.copy() for record in records]
    scalers: dict[str, dict[str, float]] = {}

    for column in numeric_columns:
        numeric_values = [
            float(record[column])
            for record in scaled_records
            if isinstance(record.get(column), (int, float))
        ]
        if not numeric_values:
            continue

        min_value = min(numeric_values)
        max_value = max(numeric_values)
        range_value = max_value - min_value
        scalers[column] = {"min": min_value, "max": max_value}

        for record in scaled_records:
            value = record.get(column)
            if not isinstance(value, (int, float)):
                continue
            if range_value == 0:
                record[column] = 0.0
            else:
                record[column] = round((float(value) - min_value) / range_value, 6)

    return scaled_records, scalers


def build_feature_matrix(
    records: list[dict[str, Any]],
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> list[list[Any]]:
    selected_records = select_feature_columns(records, feature_columns)
    encoded_records, _ = encode_categorical_features(selected_records, categorical_columns)
    scaled_records, _ = scale_numeric_features(encoded_records, numeric_columns)
    return [[record.get(column) for column in feature_columns] for record in scaled_records]


def build_target_vector(records: list[dict[str, Any]], target_column: str) -> list[Any]:
    return select_target_column(records, target_column)


def encode_target_labels(target_values: list[Any]) -> tuple[list[int], dict[str, int], dict[int, str]]:
    categories = sorted({str(target_value) for target_value in target_values})
    target_encoder = {category: index for index, category in enumerate(categories)}
    target_decoder = {index: category for category, index in target_encoder.items()}
    y_encoded = [target_encoder[str(target_value)] for target_value in target_values]
    return y_encoded, target_encoder, target_decoder


def prepare_ml_dataset(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    feature_columns = config.get("feature_columns", [])
    categorical_columns = config.get("categorical_feature_columns", [])
    numeric_columns = config.get("numeric_feature_columns", [])
    target_column = config.get("target_column")
    debug_features = config.get("debug_features", False)

    selected_records = select_feature_columns(records, feature_columns)
    encoded_records, encoders = encode_categorical_features(selected_records, categorical_columns)
    scaled_records, scalers = scale_numeric_features(encoded_records, numeric_columns)

    feature_matrix = [
        [record.get(column) for column in feature_columns] for record in scaled_records
    ]
    target_vector = build_target_vector(records, target_column) if target_column else []
    y_encoded, target_encoder, target_decoder = encode_target_labels(target_vector)

    if debug_features:
        _print_feature_debug("PROCESSED RECORDS", records)
        _print_feature_debug("SELECTED FEATURE RECORDS", selected_records)
        _print_feature_debug("ENCODED RECORDS", encoded_records)
        _print_feature_debug("SCALED RECORDS", scaled_records)
        _print_feature_debug("FEATURE COLUMNS", feature_columns)
        _print_feature_debug("TARGET COLUMN", target_column)
        _print_feature_debug("ENCODERS", encoders)
        _print_feature_debug("SCALERS", scalers)
        _print_feature_debug("FEATURE MATRIX", feature_matrix)
        _print_feature_debug("TARGET VECTOR", target_vector)
        _print_feature_debug("ENCODED TARGET VECTOR", y_encoded)

    return {
        "X": feature_matrix,
        "y": y_encoded,
        "y_raw": target_vector,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "feature_matrix": feature_matrix,
        "target_vector": target_vector,
        "target_encoder": target_encoder,
        "target_decoder": target_decoder,
        "encoders": encoders,
        "scalers": scalers,
    }
