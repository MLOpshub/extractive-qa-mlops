from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: Dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_model_artifacts(model_dir: Path) -> None:
    # Upload the whole directory as artifacts
    mlflow.log_artifacts(str(model_dir), artifact_path="model")
