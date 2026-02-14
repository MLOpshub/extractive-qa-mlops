from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "artifacts" / "models"))
MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", PROJECT_ROOT / "mlruns"))
