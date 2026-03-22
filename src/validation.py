from __future__ import annotations

from typing import Any

from src.utils import safe_cast


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
    record: dict[str, Any], schema: dict[str, str], required_fields: list[str]
) -> dict[str, Any]:
    missing_fields = check_required_fields(record, required_fields)
    type_issues = check_data_types(record, schema)
    issues = [f"missing required field: {field}" for field in missing_fields] + type_issues
    return {"is_valid": not issues, "issues": issues, "record": record}


def validate_dataset(
    records: list[dict[str, Any]], schema: dict[str, str], required_fields: list[str]
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
