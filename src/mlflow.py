from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def set_run_tags(team: str, env: Optional[str] = None) -> None:
    """
    Run tags:
    - team: pass from config
    - env: optional; auto-detect if not provided
    - platform: system info
    """
    detected_env = "colab" if os.getenv("COLAB_RELEASE_TAG") else "local"
    mlflow.set_tag("team", team)
    mlflow.set_tag("env", env or detected_env)
    mlflow.set_tag("platform", platform.platform())


def log_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters"""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics"""
    mlflow.log_metrics(metrics, step=step)


def log_config_artifact(config_path: str | Path) -> None:
    """Log the training config file (train.yaml) as an MLflow artifact"""
    mlflow.log_artifact(str(config_path), artifact_path="config")


def log_model_artifacts(model_dir: Path) -> None:
    """Upload the whole model directory as MLflow artifacts"""
    mlflow.log_artifacts(str(model_dir), artifact_path="model")
