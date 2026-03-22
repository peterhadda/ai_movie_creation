from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis import generate_summary
from src.cleaning import clean_dataset
from src.evaluate import evaluate_model
from src.features import prepare_ml_dataset
from src.ingestion import load_raw_data
from src.model import predict_labels, save_model
from src.storage import (
    save_as_csv,
    save_as_json,
    save_evaluation_report,
    save_processing_report,
)
from src.train import run_training_pipeline
from src.transform import transform_dataset
from src.utils import load_config, log_message
from src.validation import validate_dataset


def _build_processed_records(config_path: str | Path) -> dict[str, Any]:
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

    log_message("Preparing ML dataset")
    ml_dataset = prepare_ml_dataset(processed_records, config)

    log_message("Generating summary")
    summary = generate_summary(processed_records, config)

    report = {
        "raw_rows": len(raw_records),
        "valid_rows": len(validation_result["valid_records"]),
        "invalid_rows": len(validation_result["invalid_records"]),
        "validation_issues": validation_result["issue_counts"],
        "duplicates_removed": cleaning_report["duplicates_removed"],
        "missing_values_handled": cleaning_report["missing_values_filled"],
        "ml_dataset": {
            "feature_columns": ml_dataset["feature_columns"],
            "target_column": ml_dataset["target_column"],
            "feature_row_count": len(ml_dataset["feature_matrix"]),
            "target_count": len(ml_dataset["target_vector"]),
            "encoders": ml_dataset["encoders"],
            "scalers": ml_dataset["scalers"],
        },
        "summary": summary,
    }

    csv_path = save_as_csv(processed_records, config["output"]["csv"])
    json_path = save_as_json(processed_records, config["output"]["json"])
    report_path = save_processing_report(report, config["output"]["report"])

    log_message(f"Saved processed CSV to {csv_path}")
    log_message(f"Saved processed JSON to {json_path}")
    log_message(f"Saved processing report to {report_path}")
    log_message(f"Pipeline complete. Usable records: {summary['total_records']}")
    return {
        "config": config,
        "processed_records": processed_records,
        "ml_dataset": ml_dataset,
        "processing_report": report,
    }


def run_ml_pipeline(config_path: str | Path = "config.json") -> dict[str, Any]:
    pipeline_state = _build_processed_records(config_path)
    config = pipeline_state["config"]

    X = pipeline_state["ml_dataset"]["X"]
    y = pipeline_state["ml_dataset"]["y"]

    log_message("Training baseline model")
    trained_model, X_test, y_test = run_training_pipeline(X, y)

    log_message("Predicting test labels")
    y_pred = predict_labels(trained_model, X_test)

    log_message("Evaluating model")
    evaluation_report = evaluate_model(trained_model, X_test, y_test)

    model_output_path = config["output"].get("model", "data/processed/trained_model.pkl")
    evaluation_output_path = config["output"].get(
        "evaluation_report",
        "data/processed/evaluation_report.json",
    )

    save_model(trained_model, model_output_path)
    save_evaluation_report(evaluation_report, evaluation_output_path)

    return {
        "X": X,
        "y": y,
        "trained_model": trained_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "evaluation_report": evaluation_report,
    }


def run_pipeline(config_path: str | Path = "config.json") -> dict[str, Any]:
    ml_pipeline_result = run_ml_pipeline(config_path)
    return ml_pipeline_result


if __name__ == "__main__":
    run_ml_pipeline()
