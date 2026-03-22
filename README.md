# AI Movie Creation

This repository is now organized by phase and by responsibility:

- `data/` for datasets
- `models/` for saved weights
- `outputs/` for generated media
- `reports/` for metrics and histories
- `configs/` for phase configs
- `scripts/` for runnable entrypoints
- `src/` for implementation code

The refactor keeps behavior intact while removing the older root-level `main_phaseX.py` and wrapper-module duplication.

## Structure

```text
ai_movie_creation/
|-- configs/
|   |-- phase2.json
|   |-- phase3.json
|   |-- phase4.json
|   |-- phase5.json
|   `-- phase6.json
|-- data/
|   |-- raw/
|   |   |-- media_metadata.csv
|   |   |-- mnist/
|   |   `-- videos/
|   |-- processed/
|   |   |-- phase0/
|   |   |-- phase3/
|   |   |-- phase4/
|   |   `-- phase5/
|   `-- features/
|-- models/
|   |-- phase2/
|   |-- phase3/
|   |-- phase4/
|   `-- phase5/
|-- outputs/
|   |-- phase3/
|   |-- phase4/
|   |-- phase5/
|   `-- phase6/
|-- reports/
|   |-- phase0/
|   |-- phase1/
|   |-- phase2/
|   |-- phase3/
|   |-- phase4/
|   |-- phase5/
|   `-- phase6/
|-- scripts/
|   |-- run_phase2.py
|   |-- run_phase3.py
|   |-- run_phase4.py
|   |-- run_phase5.py
|   `-- run_phase6.py
|-- src/
|   |-- common/
|   |   |-- config.py
|   |   |-- io_utils.py
|   |   |-- logging_utils.py
|   |   |-- paths.py
|   |   `-- utils.py
|   |-- phase0_data_pipeline/
|   |-- phase1_ml/
|   |-- phase2_dl/
|   |-- phase3_image/
|   |-- phase4_audio/
|   |-- phase5_video/
|   `-- phase6_multimodal/
|-- tests/
|   |-- phase2_dl/
|   |-- phase3_image/
|   |-- phase4_audio/
|   |-- phase5_video/
|   `-- phase6_multimodal/
|-- README.md
`-- requirements.txt
```

## Phase Packages

- [src/phase2_dl](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase2_dl): tabular deep learning pipeline for media classification
- [src/phase3_image](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase3_image): MNIST autoencoder image-generation starter
- [src/phase4_audio](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase4_audio): waveform, spectrogram, and sequence-based audio generation starter
- [src/phase5_video](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase5_video): next-frame prediction and short video continuation starter
- [src/phase6_multimodal](/C:/Users/Aymen/Desktop/ai_movie_creation/src/phase6_multimodal): multimodal orchestration layer for text-to-image, text-to-audio, and text-to-video

## Shared Utilities

- [src/common/config.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/common/config.py): config loading
- [src/common/logging_utils.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/common/logging_utils.py): logging
- [src/common/io_utils.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/common/io_utils.py): file and report IO helpers
- [src/common/paths.py](/C:/Users/Aymen/Desktop/ai_movie_creation/src/common/paths.py): project path constants

## Run

```bash
python scripts/run_phase2.py
python scripts/run_phase3.py
python scripts/run_phase4.py
python scripts/run_phase5.py
python scripts/run_phase6.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Notes

- Phase 2 still performs the end-to-end tabular preprocessing flow and writes cleaned tabular artifacts into `data/processed/phase0/`.
- Phase 4 falls back to synthetic sine-wave audio when `data/raw/audio/` is empty.
- Phase 5 falls back to a synthetic moving-square clip when `data/raw/videos/sample.gif` is missing.
- Phase 6 is a lightweight orchestration layer that reuses the image, audio, and video generators instead of retraining new multimodal foundation models.
