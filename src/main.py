from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis import generate_summary
from src.cleaning import clean_dataset
from src.ingestion import load_raw_data
from src.storage import save_as_csv, save_as_json, save_processing_report
from src.transform import transform_dataset
from src.utils import load_config, log_message
from src.validation import validate_dataset


def run_pipeline(config_path: str | Path = "config.json") -> dict[str, Any]:
    config = load_config(config_path)
    log_message("Loading raw data")
    raw_records = load_raw_data(config["source_type"], config["source_path"])

    log_message("Validating dataset")
    validation_result = validate_dataset(
        raw_records,
        config["schema"],
        config["required_fields"],
    )

    log_message("Cleaning valid records")
    cleaned_records, cleaning_report = clean_dataset(validation_result["valid_records"], config)

    log_message("Transforming cleaned records")
    processed_records = transform_dataset(cleaned_records, config)

    log_message("Generating summary")
    summary = generate_summary(processed_records, config)

    report = {
        "raw_rows": len(raw_records),
        "valid_rows": len(validation_result["valid_records"]),
        "invalid_rows": len(validation_result["invalid_records"]),
        "validation_issues": validation_result["issue_counts"],
        "duplicates_removed": cleaning_report["duplicates_removed"],
        "missing_values_handled": cleaning_report["missing_values_filled"],
        "summary": summary,
    }

    csv_path = save_as_csv(processed_records, config["output"]["csv"])
    json_path = save_as_json(processed_records, config["output"]["json"])
    report_path = save_processing_report(report, config["output"]["report"])

    log_message(f"Saved processed CSV to {csv_path}")
    log_message(f"Saved processed JSON to {json_path}")
    log_message(f"Saved processing report to {report_path}")
    log_message(f"Pipeline complete. Usable records: {summary['total_records']}")
    return report


if __name__ == "__main__":
    run_pipeline()
