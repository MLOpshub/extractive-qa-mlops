from __future__ import annotations

import argparse
import shutil
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
import mlflow
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.settings import MODELS_DIR, MLRUNS_DIR
from src.data import load_squad, preprocess_train_features, preprocess_eval_features
from src.evaluate import compute_em_f1
from src.mlflow import setup_mlflow, log_params, log_metrics, log_model_artifacts


@dataclass
class TrainConfig:
    dataset_name: str = "squad"
    model_name: str = "bert-base-uncased"
    max_length: int = 384
    doc_stride: int = 128
    num_train_samples: int | None = 2000
    num_eval_samples: int | None = 500
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-5
    num_train_epochs: float = 1.0
    seed: int = 42

    run_name: str = "run"
    output_subdir: str = "experiments"
    save_best_to: str = "best"

    mlflow_experiment: str = "bert_squad"
    mlflow_tracking_uri: str = ""  # if empty -> file:<MLRUNS_DIR>


def read_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = TrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Force types
    cfg.learning_rate = float(cfg.learning_rate)
    cfg.num_train_epochs = float(cfg.num_train_epochs)
    cfg.per_device_train_batch_size = int(cfg.per_device_train_batch_size)
    cfg.per_device_eval_batch_size = int(cfg.per_device_eval_batch_size)
    cfg.max_length = int(cfg.max_length)
    cfg.doc_stride = int(cfg.doc_stride)
    if cfg.num_train_samples is not None:
        cfg.num_train_samples = int(cfg.num_train_samples)
    if cfg.num_eval_samples is not None:
        cfg.num_eval_samples = int(cfg.num_eval_samples)

    return cfg


def ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def copy_dir(src: Path, dst: Path) -> None:
    ensure_empty_dir(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/train.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    set_seed(cfg.seed)

    # MLflow setup
    tracking_uri = cfg.mlflow_tracking_uri.strip()
    if not tracking_uri:
        tracking_uri = f"file:{MLRUNS_DIR.resolve()}"
    setup_mlflow(tracking_uri=tracking_uri, experiment_name=cfg.mlflow_experiment)

    # Paths
    run_dir = MODELS_DIR / cfg.output_subdir / cfg.run_name
    best_dir = MODELS_DIR / cfg.save_best_to
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_squad(cfg.dataset_name)
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    if cfg.num_train_samples is not None:
        train_ds = train_ds.select(range(min(cfg.num_train_samples, len(train_ds))))
    if cfg.num_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(cfg.num_eval_samples, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Preprocess train
    train_features = train_ds.map(
        lambda x: preprocess_train_features(
            x, tokenizer=tokenizer, max_length=cfg.max_length, doc_stride=cfg.doc_stride
        ),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    # Preprocess eval (keep offsets)
    eval_examples = eval_ds
    eval_features = eval_examples.map(
        lambda x: preprocess_eval_features(
            x, tokenizer=tokenizer, max_length=cfg.max_length, doc_stride=cfg.doc_stride
        ),
        batched=True,
        remove_columns=eval_examples.column_names,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    # TrainingArguments (version-safe)
    ta_kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        fp16=True,
        seed=cfg.seed,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        ta_kwargs["evaluation_strategy"] = "epoch"
    else:
        ta_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in sig:
        ta_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    # Prepare lists for metrics
    eval_examples_list = [eval_examples[i] for i in range(len(eval_examples))]
    eval_features_list = [eval_features[i] for i in range(len(eval_features))]

    def compute_metrics(eval_pred):
        # Logits -> EM/F1
        start_logits, end_logits = eval_pred.predictions
        return compute_em_f1(
            eval_examples=eval_examples_list,
            eval_features=eval_features_list,
            raw_predictions=(np.array(start_logits), np.array(end_logits)),
        )

    # Start MLflow run
    with mlflow.start_run(run_name=cfg.run_name):
        log_params(
            {
                "dataset_name": cfg.dataset_name,
                "model_name": cfg.model_name,
                "max_length": cfg.max_length,
                "doc_stride": cfg.doc_stride,
                "train_samples": cfg.num_train_samples,
                "eval_samples": cfg.num_eval_samples,
                "train_batch_size": cfg.per_device_train_batch_size,
                "eval_batch_size": cfg.per_device_eval_batch_size,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.num_train_epochs,
                "seed": cfg.seed,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=eval_features,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        # Log metrics
        metrics: Dict[str, float] = {}
        if "train_loss" in train_result.metrics:
            metrics["train_loss"] = float(train_result.metrics["train_loss"])
        for k, v in eval_result.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        log_metrics(metrics)

        # Save final model
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

        # Log artifacts
        log_model_artifacts(run_dir)

    # Update best model folder
    copy_dir(run_dir, best_dir)

    print(f"Saved run model to: {run_dir}")
    print(f"Updated best model at: {best_dir}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
