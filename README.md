# Media AI Pipeline

This repository is organized by phase so each learning track stays isolated:

- Phase 2: tabular deep learning for media-type classification
- Phase 3: image generation with an MNIST autoencoder
- Phase 4: audio generation with waveform, spectrogram, and sequence modeling foundations

Shared helpers live in `src/common`, while each phase has its own package, entrypoint, config, and tests.

## Structure

```text
ai_movie_creation/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- features/
|-- models/
|-- reports/
|-- src/
|   |-- common/
|   |   `-- utils.py
|   |-- phase2/
|   |   |-- dataset.py
|   |   |-- evaluate.py
|   |   |-- features.py
|   |   |-- main.py
|   |   |-- model.py
|   |   |-- predict.py
|   |   `-- train.py
|   |-- phase3/
|   |   |-- autoencoder.py
|   |   |-- dataset.py
|   |   |-- main.py
|   |   `-- train.py
|   |-- phase4/
|   |   |-- data.py
|   |   |-- generate.py
|   |   |-- main.py
|   |   |-- model.py
|   |   `-- train.py
|   `-- ...
|-- tests/
|   |-- phase2/
|   |-- phase3/
|   `-- phase4/
|-- config.json
|-- config_phase3.json
|-- config_phase4.json
|-- main.py
|-- main_phase3.py
|-- main_phase4.py
`-- requirements.txt
```

## Entry Points

- [main.py](/C:/Users/Aymen/Desktop/ai_movie_creation/main.py): runs the Phase 2 tabular pipeline
- [main_phase3.py](/C:/Users/Aymen/Desktop/ai_movie_creation/main_phase3.py): runs the Phase 3 image autoencoder pipeline
- [main_phase4.py](/C:/Users/Aymen/Desktop/ai_movie_creation/main_phase4.py): runs the Phase 4 audio generation pipeline

## Phase 2 Flow

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

## Phase Packages

- [src/common/utils.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/common/utils.py): shared loading, cleaning, reporting, and save helpers
- [src/phase2](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase2): structured-data deep learning pipeline
- [src/phase3](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase3): image-generation starter built around an autoencoder
- [src/phase4](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase4): audio-generation starter with spectrogram and sequence modeling

## Outputs

By default the project writes:

- processed tabular data to `data/processed/`
- trained models to `models/`
- reports, histories, reconstructions, and generated audio to `reports/`

## Run

```bash
python main.py
python main_phase3.py
python main_phase4.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Notes

- Phase 2 uses the sample media metadata file and is mainly for pipeline structure, not model quality.
- Phase 3 uses MNIST and saves reconstruction samples.
- Phase 4 supports real `.wav` files from `data/raw/audio/`, but falls back to synthetic sine waves if that folder is empty.
