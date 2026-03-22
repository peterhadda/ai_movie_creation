# Media Data Pipeline

This project is a Phase 0 data-engineering exercise. It ingests raw media metadata, validates it, cleans it, transforms it into a consistent schema, stores the processed output, and generates a simple inspection report.

It also includes a first feature-engineering step that converts processed records into an ML-ready feature matrix and target vector.

## Structure

```text
media_data_pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── tests/
├── config.json
├── main.py
└── requirements.txt
```

## What the pipeline does

1. Loads raw data from CSV, JSON, or an API.
2. Validates required fields and expected types.
3. Cleans duplicates, trims text, fills configured missing values, and standardizes dates.
4. Converts media categories and units into a consistent format.
5. Saves processed records as CSV and JSON.
6. Builds encoded and scaled model inputs from configured feature columns.
7. Writes a processing report with counts, feature-prep metadata, and summary statistics.

## Run

```bash
python main.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Sample dataset

The sample raw file at `data/raw/media_metadata.csv` contains intentionally messy records so you can see validation and cleaning behavior in action.
