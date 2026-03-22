import unittest

from src.cleaning import clean_dataset
from src.features import build_feature_matrix, prepare_ml_dataset
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
        }

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
        self.assertEqual(ml_dataset["target_vector"], ["video", "audio"])
        self.assertEqual(ml_dataset["encoders"]["format"], {"mp4": 0, "wav": 1})
        self.assertEqual(ml_dataset["feature_matrix"][0], [0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ml_dataset["feature_matrix"][1], [1, 1.0, 1.0, 1.0, 1.0])

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


if __name__ == "__main__":
    unittest.main()
