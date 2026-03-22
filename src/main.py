from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dataset import (
    convert_features_to_tensor,
    convert_target_to_tensor,
    create_data_loader,
    create_tensor_dataset,
    split_dataset,
)
from src.evaluate import evaluate_model, generate_evaluation_report
from src.features import prepare_ml_dataset, transform_dataset
from src.model import initialize_network, save_model
from src.train import initialize_loss_function, initialize_optimizer, run_training_loop
from src.utils import (
    clean_dataset,
    generate_summary,
    load_config,
    load_raw_data,
    log_message,
    save_as_csv,
    save_as_json,
    save_evaluation_report,
    save_processing_report,
    save_training_history,
    validate_dataset,
)


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
    target_decoder = pipeline_state["ml_dataset"]["target_decoder"]

    training_config = config.get("training", {})
    hidden_size = training_config.get("hidden_size", 16)
    learning_rate = training_config.get("learning_rate", 0.001)
    num_epochs = training_config.get("num_epochs", 20)
    batch_size = training_config.get("batch_size", 4)
    train_ratio = training_config.get("train_ratio", 0.6)
    val_ratio = training_config.get("val_ratio", 0.2)
    test_ratio = training_config.get("test_ratio", 0.2)

    log_message("Converting features and targets to tensors")
    X_tensor = convert_features_to_tensor(X)
    y_tensor = convert_target_to_tensor(y)

    log_message("Creating datasets and loaders")
    dataset = create_tensor_dataset(X_tensor, y_tensor)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

    log_message("Initializing neural network")
    input_size = len(X[0]) if X else 0
    output_size = len(target_decoder)
    model = initialize_network(input_size, hidden_size, output_size)
    loss_fn = initialize_loss_function("classification")
    optimizer = initialize_optimizer(model, learning_rate)

    log_message("Training neural network")
    trained_model, training_history = run_training_loop(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        num_epochs,
    )

    log_message("Evaluating model")
    test_loss, test_accuracy, y_true, y_pred = evaluate_model(
        trained_model,
        test_loader,
        loss_fn,
    )
    evaluation_report = generate_evaluation_report(test_loss, test_accuracy, y_true, y_pred)
    evaluation_report["label_mapping"] = {
        "target_decoder": {str(key): value for key, value in target_decoder.items()}
    }

    model_output_path = config["output"].get("model", "data/processed/trained_model.pkl")
    evaluation_output_path = config["output"].get(
        "evaluation_report",
        "data/processed/evaluation_report.json",
    )
    history_output_path = config["output"].get(
        "training_history",
        "data/processed/training_history.json",
    )

    save_model(trained_model, model_output_path)
    save_evaluation_report(evaluation_report, evaluation_output_path)
    save_training_history(training_history, history_output_path)

    return {
        "X": X,
        "y": y,
        "X_tensor": X_tensor,
        "y_tensor": y_tensor,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "trained_model": trained_model,
        "training_history": training_history,
        "y_test": y_true,
        "y_pred": y_pred,
        "evaluation_report": evaluation_report,
    }


def run_pipeline(config_path: str | Path = "config.json") -> dict[str, Any]:
    ml_pipeline_result = run_ml_pipeline(config_path)
    return ml_pipeline_result


if __name__ == "__main__":
    run_ml_pipeline()
