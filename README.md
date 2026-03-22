# Media DL Pipeline

This project is a tabular deep learning pipeline for media-type classification. It loads raw media metadata, validates and cleans it, engineers structured features, converts them into tensors, trains a feedforward neural network with PyTorch, evaluates the model, and saves the resulting artifacts.

The current target is `category` / media type classification such as:

- `image`
- `audio`
- `video`

## Structure

```text
media_dl_pipeline/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- features/
|-- models/
|-- reports/
|-- src/
|   |-- dataset.py
|   |-- evaluate.py
|   |-- features.py
|   |-- main.py
|   |-- model.py
|   |-- predict.py
|   |-- train.py
|   `-- utils.py
|-- tests/
|-- config.json
|-- main.py
`-- requirements.txt
```

## Pipeline Flow

```text
raw records
  -> validation
  -> cleaning
  -> feature preparation
  -> X, y
  -> X_tensor, y_tensor
  -> dataset split
  -> train/validation/test loaders
  -> neural network
  -> training loop
  -> evaluation
  -> saved model and reports
```

## Modules

- [src/utils.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/utils.py): data loading, validation, cleaning, summary generation, and file saving helpers
- [src/features.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/features.py): transformation, feature encoding, scaling, and target encoding
- [src/dataset.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/dataset.py): tensor conversion, dataset creation, splitting, and dataloaders
- [src/model.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/model.py): feedforward PyTorch network, prediction helpers, save/load
- [src/train.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/train.py): loss, optimizer, batch training, epoch training, validation, full training loop
- [src/evaluate.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/evaluate.py): test evaluation, accuracy, confusion matrix, evaluation report
- [src/predict.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/predict.py): single-sample and batch prediction
- [src/main.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/main.py): end-to-end pipeline entrypoint

## Outputs

By default the pipeline writes:

- processed CSV to `data/processed/`
- processed JSON to `data/processed/`
- trained model to `models/`
- processing and evaluation reports to `reports/`
- training history to `reports/`

All output paths and training settings are configurable in [config.json](/C:/Users/Aymen/Desktop/ai_movie_creation/config.json).

## Run

```bash
python main.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Notes

- The included sample dataset is useful for pipeline development, but it is too small for meaningful deep learning performance.
- For serious Phase 2 training, use a larger and more balanced dataset with at least a few hundred rows.
