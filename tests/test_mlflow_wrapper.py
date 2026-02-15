from pathlib import Path
from unittest.mock import MagicMock

import extractive_qa_mlops.mlflow as mlf


def test_setup_mlflow_calls_mlflow_api(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(mlf, "mlflow", mock)

    mlf.setup_mlflow("file:/tmp/mlruns", "exp1")

    mock.set_tracking_uri.assert_called_once_with("file:/tmp/mlruns")
    mock.set_experiment.assert_called_once_with("exp1")


def test_set_run_tags_detects_env_and_sets_tags(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(mlf, "mlflow", mock)

    # Force local env
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    mlf.set_run_tags(team="team-x")

    # called with expected keys
    mock.set_tag.assert_any_call("team", "team-x")
    mock.set_tag.assert_any_call("env", "local")
    # platform tag exists (value varies)
    assert any(call.args[0] == "platform" for call in mock.set_tag.call_args_list)


def test_set_run_tags_honors_env_argument(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(mlf, "mlflow", mock)

    mlf.set_run_tags(team="team-x", env="ci")

    mock.set_tag.assert_any_call("env", "ci")


def test_log_params_metrics_and_artifacts(monkeypatch, tmp_path: Path):
    mock = MagicMock()
    monkeypatch.setattr(mlf, "mlflow", mock)

    mlf.log_params({"lr": 1e-3, "epochs": 3})
    mock.log_params.assert_called_once()

    mlf.log_metrics({"f1": 12.3}, step=7)
    mock.log_metrics.assert_called_once_with({"f1": 12.3}, step=7)

    cfg = tmp_path / "train.yaml"
    cfg.write_text("x: 1", encoding="utf-8")
    mlf.log_config_artifact(cfg)
    mock.log_artifact.assert_called_once()
    assert mock.log_artifact.call_args[0][0].endswith("train.yaml")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mlf.log_model_artifacts(model_dir)
    mock.log_artifacts.assert_called_once()
    assert mock.log_artifacts.call_args[0][0].endswith("model")
