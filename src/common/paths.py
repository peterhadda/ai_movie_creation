from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

__all__ = ["PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "OUTPUTS_DIR", "REPORTS_DIR"]
