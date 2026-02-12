from __future__ import annotations

import argparse
import shutil
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import yaml
import mlflow
import numpy as np
import random

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
from src.mlflow import (
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifacts,
    set_run_tags,
    log_config_artifact,
)
from src.best_model import load_best_meta, is_better, save_best_meta


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

    run_name: str = "auto"
    output_subdir: str = "experiments"
    save_best_to: str = "best"

    mlflow_experiment: str = "bert_squad"
    mlflow_tracking_uri: str = ""  # if empty -> file:<MLRUNS_DIR>
    
    fp16: bool = True
    logging_steps: int = 2000
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01 

    team_name: str = "Tengzhe_Deepthi_Reda"

    log_ctx_stats: bool = True
    ctx_stats_n: int = 200

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
    cfg.log_ctx_stats = bool(cfg.log_ctx_stats) if isinstance(cfg.log_ctx_stats, bool) else str(cfg.log_ctx_stats).lower() == "true"
    cfg.ctx_stats_n = int(cfg.ctx_stats_n)

    return cfg


def remove_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)

def copy_dir(src: Path, dst: Path) -> None:
    remove_dir(dst)
    shutil.copytree(src, dst)  

# automatic naming experiments
def fmt_lr(lr: float) -> str:
    s = f"{lr:.0e}"          # 3e-05
    return s.replace("e-0", "e-").replace("e+0", "e+")

def fmt_epoch(e: float) -> str:
    return str(int(e)) if float(e).is_integer() else str(e).replace(".", "p")

def resolve_run_name(cfg: TrainConfig) -> None:
    if (not cfg.run_name) or (cfg.run_name == "auto"):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_short = cfg.model_name.split("/")[-1]
        cfg.run_name = (
            f"{ts}_{model_short}_lr{fmt_lr(cfg.learning_rate)}"
            f"_bs{cfg.per_device_train_batch_size}_e{fmt_epoch(cfg.num_train_epochs)}"
            f"_l{cfg.max_length}_s{cfg.doc_stride}"
        )

def sanity_check_answers(ds, n: int = 50, seed: int = 42) -> Dict[str, float]:
    """SQuAD v1 sanity check: answer text should appear in context (sampled)."""
    n = min(n, len(ds))
    rnd = random.Random(seed)
    idxs = rnd.sample(range(len(ds)), n)

    ok = 0
    for i in idxs:
        ex = ds[i]
        ans = ex["answers"]["text"][0]   # v1: always has an answer
        if ans in ex["context"]:
            ok += 1

    return {
        "answer_in_context_rate": float(ok / n) if n > 0 else 0.0,
        "answer_checked": float(n),
    }

def sample_context_stats(ds, n: int = 200, seed: int = 42) -> Dict[str, float]:
    """Sample N examples and compute context length stats (chars)."""
    n = min(n, len(ds))
    rnd = random.Random(seed)
    idxs = rnd.sample(range(len(ds)), n)
    lens = sorted(len(ds[i]["context"]) for i in idxs)
    return {
        "ctx_len_mean": float(sum(lens) / len(lens)),
        "ctx_len_p50": float(lens[len(lens) // 2]),
        "ctx_len_p90": float(lens[int(len(lens) * 0.9) - 1]),
        "ctx_len_max": float(lens[-1]),
    }

def log_data_profile(train_ds, eval_ds, train_features, eval_features, ctx_stats_n: int, seed: int) -> Dict[str, float]:
    """Compute light-weight dataset and feature stats to log to MLflow."""
    stats: Dict[str, float] = {}

    stats["n_train_examples"] = float(len(train_ds))
    stats["n_eval_examples"] = float(len(eval_ds))

    stats["n_train_features"] = float(len(train_features))
    stats["n_eval_features"] = float(len(eval_features))

    stats["train_feat_per_ex"] = float(len(train_features) / max(1, len(train_ds)))
    stats["eval_feat_per_ex"] = float(len(eval_features) / max(1, len(eval_ds)))

    stats.update(sample_context_stats(train_ds, n=ctx_stats_n, seed=seed))

    return stats

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/train.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    resolve_run_name(cfg)
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
        report_to=[],
        seed=cfg.seed,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
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
        #team/env/platform
        set_run_tags(team=cfg.team_name)

        # Save the exact config used for this run (train.yaml)
        log_config_artifact(args.config)

        # Light data profiling (before training)
        if cfg.log_ctx_stats:
            stats = log_data_profile(
                train_ds=train_ds,
                eval_ds=eval_ds,
                train_features=train_features,
                eval_features=eval_features,
                ctx_stats_n=cfg.ctx_stats_n,
                seed=cfg.seed,
            )
            for k, v in stats.items():
                mlflow.log_metric(k, v)

        # Tiny sanity check (before training)
        check = sanity_check_answers(train_ds, n=50, seed=cfg.seed)
        mlflow.log_metrics(check)

        log_params(
            {
                "dataset_name": cfg.dataset_name,
                "model_name": cfg.model_name,
                "max_length": cfg.max_length,
                "doc_stride": cfg.doc_stride,
                "train_samples": cfg.num_train_samples,
                "train_size": len(train_ds),
                "eval_size": len(eval_ds),
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

        # Get current EM/F1
        cur_f1 = float(eval_result.get("eval_f1", eval_result.get("f1", -1.0)))
        cur_em = float(eval_result.get("eval_em", eval_result.get("em", -1.0)))


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
    # Update best (F1 first, EM second)
    best_meta = load_best_meta(best_dir)
    best_f1 = float(best_meta.get("best_f1", -1.0))
    best_em = float(best_meta.get("best_em", -1.0))

    if is_better(cur_f1, cur_em, best_f1, best_em):
        copy_dir(run_dir, best_dir)
        save_best_meta(
            best_dir,
            {
                "best_f1": cur_f1,
                "best_em": cur_em,
                "best_run": cfg.run_name,
                "best_path": str(best_dir),
            },
        )
        print(f"Updated best model at: {best_dir} (f1={cur_f1:.2f}, em={cur_em:.2f})")
    else:
        print(f"Keep best model (best_f1={best_f1:.2f}, best_em={best_em:.2f})")

    print(f"Saved run model to: {run_dir}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
