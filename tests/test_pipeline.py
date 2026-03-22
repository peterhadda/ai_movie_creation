import json
import tempfile
import unittest
from pathlib import Path

from src.cleaning import clean_dataset
from src.evaluate import evaluate_model
from src.features import build_feature_matrix, prepare_ml_dataset
from src.main import run_ml_pipeline
from src.model import (
    fit_model,
    initialize_model,
    load_model,
    predict_labels,
    predict_probabilities,
    save_model,
)
from src.predict import format_prediction_output, predict_batch, predict_single_record
from src.split import check_class_distribution, split_train_test, stratified_split
from src.train import run_training_pipeline
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
            "category_mapping": {"video file": "video", "image": "image", "vid": "video"},
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
            "output": {
                "csv": "data/processed/test_media_metadata_clean.csv",
                "json": "data/processed/test_media_metadata_clean.json",
                "report": "data/processed/test_processing_report.json",
                "model": "data/processed/test_trained_model.pkl",
                "evaluation_report": "data/processed/test_evaluation_report.json",
            },
        }
        self.training_X = [
            [0, 0.0, 0.0, 0.0, 0.0],
            [0, 0.1, 0.1, 0.1, 0.1],
            [1, 0.9, 0.9, 0.9, 0.9],
            [1, 1.0, 1.0, 1.0, 1.0],
        ]
        self.training_y = ["video", "video", "audio", "audio"]

    def test_validation_separates_invalid_records(self) -> None:
        records = [
            {"file_name": "clip.mp4", "format": "mp4", "duration": "10", "width": "1920", "height": "1080", "size": "1 MB", "category": "video", "created_date": "2026-03-01"},
            {"file_name": "bad.mp4", "format": "mp4", "duration": "bad", "width": "1920", "height": "1080", "size": "1 MB", "category": "video", "created_date": "2026-03-01"},
        ]
        result = validate_dataset(records, self.config["schema"], self.config["required_fields"])
        self.assertEqual(len(result["valid_records"]), 1)
        self.assertEqual(len(result["invalid_records"]), 1)

    def test_clean_and_transform_standardizes_output(self) -> None:
        records = [
            {"file_name": " trailer.mp4 ", "format": "mp4", "duration": "120.5", "width": "1920", "height": "1080", "size": "1.5 MB", "category": "video file", "created_date": "2026/03/01"},
            {"file_name": " trailer.mp4 ", "format": "mp4", "duration": "120.5", "width": "1920", "height": "1080", "size": "1.5 MB", "category": "video file", "created_date": "2026/03/01"},
        ]
        cleaned, report = clean_dataset(records, self.config)
        transformed = transform_dataset(cleaned, self.config)

        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(transformed[0]["category"], "video")
        self.assertEqual(transformed[0]["created_date"], "2026-03-01")
        self.assertEqual(transformed[0]["size_bytes"], 1572864)

    def test_prepare_ml_dataset_encodes_and_scales_features(self) -> None:
        records = [
            {
                "file_name": "clip_a.mp4",
                "format": "mp4",
                "duration_seconds": 100.0,
                "width": 1000,
                "height": 500,
                "size_bytes": 100,
                "category": "video",
                "created_date": "2026-03-01",
            },
            {
                "file_name": "clip_b.wav",
                "format": "wav",
                "duration_seconds": 200.0,
                "width": 2000,
                "height": 1000,
                "size_bytes": 300,
                "category": "audio",
                "created_date": "2026-03-02",
            },
        ]

        ml_dataset = prepare_ml_dataset(records, self.config)

        self.assertEqual(ml_dataset["feature_columns"], self.config["feature_columns"])
        self.assertEqual(ml_dataset["y"], ["video", "audio"])
        self.assertEqual(ml_dataset["encoders"]["format"], {"mp4": 0, "wav": 1})
        self.assertEqual(ml_dataset["X"][0], [0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ml_dataset["X"][1], [1, 1.0, 1.0, 1.0, 1.0])

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

    def test_split_module_returns_expected_outputs(self) -> None:
        X_train, X_test, y_train, y_test = split_train_test(self.training_X, self.training_y)
        self.assertEqual(len(X_train) + len(X_test), len(self.training_X))
        self.assertEqual(len(y_train) + len(y_test), len(self.training_y))
        self.assertEqual(check_class_distribution(self.training_y), {"video": 2, "audio": 2})

    def test_stratified_split_keeps_class_balance(self) -> None:
        X = [[index] for index in range(10)]
        y = ["video"] * 5 + ["audio"] * 5

        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.4)

        self.assertEqual(check_class_distribution(y_train), {"video": 3, "audio": 3})
        self.assertEqual(check_class_distribution(y_test), {"video": 2, "audio": 2})

    def test_model_module_trains_predicts_and_persists(self) -> None:
        model = initialize_model()
        trained_model = fit_model(model, self.training_X, self.training_y)

        y_pred = predict_labels(trained_model, self.training_X)
        y_proba = predict_probabilities(trained_model, self.training_X)

        self.assertEqual(len(y_pred), len(self.training_y))
        self.assertEqual(len(y_proba), len(self.training_X))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            save_model(trained_model, model_path)
            loaded_model = load_model(model_path)
            self.assertEqual(predict_labels(loaded_model, self.training_X), y_pred)

    def test_evaluate_and_predict_modules_return_expected_shapes(self) -> None:
        trained_model = fit_model(initialize_model(), self.training_X, self.training_y)

        evaluation_report = evaluate_model(trained_model, self.training_X, self.training_y)
        batch_predictions = predict_batch(trained_model, self.training_X)
        single_prediction = predict_single_record(trained_model, self.training_X[0])
        formatted_predictions = format_prediction_output(batch_predictions)

        self.assertIn("accuracy_score", evaluation_report)
        self.assertIn("confusion_matrix", evaluation_report)
        self.assertEqual(len(batch_predictions), len(self.training_X))
        self.assertIn(single_prediction, self.training_y)
        self.assertEqual(formatted_predictions[0]["record_index"], 0)

    def test_run_training_pipeline_returns_model_and_test_split(self) -> None:
        trained_model, X_test, y_test = run_training_pipeline(self.training_X, self.training_y)
        self.assertTrue(hasattr(trained_model, "predict"))
        self.assertEqual(len(X_test), len(y_test))

    def test_run_ml_pipeline_persists_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            config = dict(self.config)
            config["source_type"] = "csv"
            config["source_path"] = "data/raw/media_metadata.csv"
            config["debug_features"] = False
            config["output"] = {
                "csv": str(temp_dir_path / "processed.csv"),
                "json": str(temp_dir_path / "processed.json"),
                "report": str(temp_dir_path / "processing_report.json"),
                "model": str(temp_dir_path / "trained_model.pkl"),
                "evaluation_report": str(temp_dir_path / "evaluation_report.json"),
            }

            config_path = temp_dir_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_ml_pipeline(config_path)

            self.assertIn("trained_model", result)
            self.assertIn("evaluation_report", result)
            self.assertTrue((temp_dir_path / "trained_model.pkl").exists())
            self.assertTrue((temp_dir_path / "evaluation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
