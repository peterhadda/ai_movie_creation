import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.cleaning import clean_dataset
from src.dataset import (
    convert_features_to_tensor,
    convert_target_to_tensor,
    create_data_loader,
    create_tensor_dataset,
    split_dataset,
)
from src.evaluate import (
    build_confusion_matrix,
    compute_accuracy,
    evaluate_model,
    generate_evaluation_report,
)
from src.features import build_feature_matrix, prepare_ml_dataset
from src.main import run_ml_pipeline
from src.model import (
    initialize_network,
    load_model,
    predict_labels,
    predict_probabilities,
    save_model,
)
from src.predict import predict_batch, predict_single_sample
from src.train import (
    initialize_loss_function,
    initialize_optimizer,
    run_training_loop,
    train_one_batch,
    train_one_epoch,
    validate_one_epoch,
)
from src.transform import transform_dataset
from src.validation import validate_dataset


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "schema": {
                "file_name": "str",
                "format": "str",
                "duration": "float",
                "width": "int",
                "height": "int",
                "size": "str",
                "category": "str",
                "created_date": "str",
            },
            "required_fields": ["file_name", "format", "category", "created_date"],
            "fill_missing": {"format": "unknown"},
            "numeric_fields": {"duration": "float", "width": "int", "height": "int"},
            "date_fields": ["created_date"],
            "category_field": "category",
            "category_mapping": {
                "video file": "video",
                "image": "image",
                "vid": "video",
                "audio": "audio",
            },
            "selected_fields": [
                "file_name",
                "format",
                "duration_seconds",
                "width",
                "height",
                "size_bytes",
                "category",
                "created_date",
            ],
            "feature_columns": ["format", "duration_seconds", "width", "height", "size_bytes"],
            "categorical_feature_columns": ["format"],
            "numeric_feature_columns": ["duration_seconds", "width", "height", "size_bytes"],
            "target_column": "category",
            "training": {
                "hidden_size": 8,
                "learning_rate": 0.01,
                "num_epochs": 5,
                "batch_size": 2,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            },
            "output": {
                "csv": "data/processed/test_media_metadata_clean.csv",
                "json": "data/processed/test_media_metadata_clean.json",
                "report": "data/processed/test_processing_report.json",
                "model": "data/processed/test_trained_model.pt",
                "evaluation_report": "data/processed/test_evaluation_report.json",
                "training_history": "data/processed/test_training_history.json",
            },
        }
        self.training_X = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.05, 0.0, 0.05, 0.0, 0.05],
            [0.95, 1.0, 0.95, 1.0, 0.95],
        ]
        self.training_y = [0, 0, 1, 1, 0, 1]

    def test_validation_separates_invalid_records(self) -> None:
        records = [
            {"file_name": "clip.mp4", "format": "mp4", "duration": "10", "width": "1920", "height": "1080", "size": "1 MB", "category": "video", "created_date": "2026-03-01"},
            {"file_name": "bad.mp4", "format": "mp4", "duration": "bad", "width": "1920", "height": "1080", "size": "1 MB", "category": "video", "created_date": "2026-03-01"},
        ]
        result = validate_dataset(records, self.config["schema"], self.config["required_fields"])
        self.assertEqual(len(result["valid_records"]), 1)
        self.assertEqual(len(result["invalid_records"]), 1)

    def test_clean_transform_and_feature_prep_build_encoded_targets(self) -> None:
        records = [
            {"file_name": " trailer.mp4 ", "format": "mp4", "duration": "120.5", "width": "1920", "height": "1080", "size": "1.5 MB", "category": "video file", "created_date": "2026/03/01"},
            {"file_name": "song.wav", "format": "wav", "duration": "200", "width": "0", "height": "0", "size": "2 MB", "category": "audio", "created_date": "2026/03/02"},
        ]
        cleaned, _ = clean_dataset(records, self.config)
        transformed = transform_dataset(cleaned, self.config)
        ml_dataset = prepare_ml_dataset(transformed, self.config)

        self.assertEqual(transformed[0]["category"], "video")
        self.assertEqual(ml_dataset["feature_columns"], self.config["feature_columns"])
        self.assertEqual(ml_dataset["y_raw"], ["video", "audio"])
        self.assertEqual(set(ml_dataset["target_encoder"].keys()), {"audio", "video"})
        self.assertEqual(len(ml_dataset["X"]), 2)

    def test_build_feature_matrix_preserves_row_alignment(self) -> None:
        records = [
            {"format": "mp4", "duration_seconds": 50.0, "width": 640, "height": 360, "size_bytes": 10},
            {"format": "wav", "duration_seconds": 150.0, "width": 1280, "height": 720, "size_bytes": 110},
        ]

        matrix = build_feature_matrix(
            records,
            self.config["feature_columns"],
            self.config["categorical_feature_columns"],
            self.config["numeric_feature_columns"],
        )

        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), len(self.config["feature_columns"]))
        self.assertEqual(matrix[0][0], 0)
        self.assertEqual(matrix[1][0], 1)

    def test_dataset_module_creates_tensors_and_loaders(self) -> None:
        X_tensor = convert_features_to_tensor(self.training_X)
        y_tensor = convert_target_to_tensor(self.training_y)
        dataset = create_tensor_dataset(X_tensor, y_tensor)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.6, 0.2, 0.2)
        train_loader = create_data_loader(train_dataset, batch_size=2, shuffle=True)

        batch_X, batch_y = next(iter(train_loader))

        self.assertEqual(X_tensor.dtype, torch.float32)
        self.assertEqual(y_tensor.dtype, torch.long)
        self.assertEqual(len(train_dataset) + len(val_dataset) + len(test_dataset), len(dataset))
        self.assertEqual(batch_X.shape[1], len(self.training_X[0]))
        self.assertEqual(batch_y.ndim, 1)

    def test_model_module_initializes_predicts_and_persists(self) -> None:
        model = initialize_network(input_size=5, hidden_size=8, output_size=2)
        X_tensor = convert_features_to_tensor(self.training_X)

        probabilities = predict_probabilities(model, X_tensor)
        predictions = predict_labels(model, X_tensor)

        self.assertEqual(len(probabilities), len(self.training_X))
        self.assertEqual(len(predictions), len(self.training_X))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pt"
            save_model(model, model_path)
            loaded_model = load_model(model_path)
            self.assertEqual(len(predict_labels(loaded_model, X_tensor)), len(self.training_X))

    def test_training_functions_run_one_batch_epoch_and_loop(self) -> None:
        X_tensor = convert_features_to_tensor(self.training_X)
        y_tensor = convert_target_to_tensor(self.training_y)
        dataset = create_tensor_dataset(X_tensor, y_tensor)
        train_dataset, val_dataset, _ = split_dataset(dataset, 0.6, 0.2, 0.2)
        train_loader = create_data_loader(train_dataset, batch_size=2, shuffle=True)
        val_loader = create_data_loader(val_dataset, batch_size=1, shuffle=False)

        model = initialize_network(input_size=5, hidden_size=8, output_size=2)
        loss_fn = initialize_loss_function("classification")
        optimizer = initialize_optimizer(model, learning_rate=0.01)

        batch_X, batch_y = next(iter(train_loader))
        batch_loss = train_one_batch(model, batch_X, batch_y, loss_fn, optimizer)
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, loss_fn)
        trained_model, training_history = run_training_loop(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=3,
        )

        self.assertGreaterEqual(batch_loss, 0.0)
        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertGreaterEqual(train_accuracy, 0.0)
        self.assertGreaterEqual(val_accuracy, 0.0)
        self.assertEqual(len(training_history), 3)
        self.assertTrue(hasattr(trained_model, "forward"))

    def test_evaluate_and_predict_modules_return_expected_outputs(self) -> None:
        X_tensor = convert_features_to_tensor(self.training_X)
        y_tensor = convert_target_to_tensor(self.training_y)
        dataset = create_tensor_dataset(X_tensor, y_tensor)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.6, 0.2, 0.2)
        train_loader = create_data_loader(train_dataset, batch_size=2, shuffle=True)
        val_loader = create_data_loader(val_dataset, batch_size=1, shuffle=False)
        test_loader = create_data_loader(test_dataset, batch_size=1, shuffle=False)

        model = initialize_network(input_size=5, hidden_size=8, output_size=2)
        loss_fn = initialize_loss_function("classification")
        optimizer = initialize_optimizer(model, learning_rate=0.01)
        trained_model, _ = run_training_loop(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            num_epochs=2,
        )

        test_loss, test_accuracy, y_true, y_pred = evaluate_model(
            trained_model,
            test_loader,
            loss_fn,
        )
        evaluation_report = generate_evaluation_report(test_loss, test_accuracy, y_true, y_pred)
        single_prediction = predict_single_sample(trained_model, X_tensor[0])
        batch_predictions = predict_batch(trained_model, X_tensor)

        self.assertIn("test_loss", evaluation_report)
        self.assertIn("confusion_matrix", evaluation_report)
        self.assertGreaterEqual(compute_accuracy(y_true, y_pred), 0.0)
        self.assertEqual(len(build_confusion_matrix(y_true, y_pred)), 2)
        self.assertIn(single_prediction, [0, 1])
        self.assertEqual(len(batch_predictions), len(self.training_X))

    def test_run_ml_pipeline_persists_phase_two_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            config = dict(self.config)
            config["source_type"] = "csv"
            config["source_path"] = "data/raw/media_metadata.csv"
            config["debug_features"] = False
            config["training"] = {
                "hidden_size": 8,
                "learning_rate": 0.01,
                "num_epochs": 3,
                "batch_size": 2,
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
            }
            config["output"] = {
                "csv": str(temp_dir_path / "processed.csv"),
                "json": str(temp_dir_path / "processed.json"),
                "report": str(temp_dir_path / "processing_report.json"),
                "model": str(temp_dir_path / "trained_model.pt"),
                "evaluation_report": str(temp_dir_path / "evaluation_report.json"),
                "training_history": str(temp_dir_path / "training_history.json"),
            }

            config_path = temp_dir_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_ml_pipeline(config_path)

            self.assertIn("X_tensor", result)
            self.assertIn("training_history", result)
            self.assertIn("evaluation_report", result)
            self.assertTrue((temp_dir_path / "trained_model.pt").exists())
            self.assertTrue((temp_dir_path / "evaluation_report.json").exists())
            self.assertTrue((temp_dir_path / "training_history.json").exists())


if __name__ == "__main__":
    unittest.main()
