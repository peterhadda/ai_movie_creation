import unittest

from src.cleaning import clean_dataset
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


if __name__ == "__main__":
    unittest.main()
