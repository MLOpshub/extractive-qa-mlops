from __future__ import annotations

import argparse
import shutil
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from datetime import datetime

import os
import platform
import yaml
import mlflow
import numpy as np
import random
import json
import math
import re
import string

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    set_seed,
)

from extractive_qa_mlops.settings import MODELS_DIR, MLRUNS_DIR
from extractive_qa_mlops.data import (
    load_squad,
    preprocess_train_features,
    preprocess_eval_features,
)
from extractive_qa_mlops.evaluate import compute_em_f1
from extractive_qa_mlops.mlflow import (
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifacts,
    set_run_tags,
    log_config_artifact,
)
from extractive_qa_mlops.best_model import load_best_meta, is_better, save_best_meta


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
    cfg.log_ctx_stats = (
        bool(cfg.log_ctx_stats)
        if isinstance(cfg.log_ctx_stats, bool)
        else str(cfg.log_ctx_stats).lower() == "true"
    )
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
    s = f"{lr:.0e}"  # 3e-05
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
        ans = ex["answers"]["text"][0]  # v1: always has an answer
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


def log_data_profile(
    train_ds, eval_ds, train_features, eval_features, ctx_stats_n: int, seed: int
) -> Dict[str, float]:
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


# standard SQuAD normalize
def normalize_answer(s: str) -> str:
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def write_error_report(
    run_dir: Path,
    trainer: Trainer,
    eval_features,  # Dataset (NOT list)
    eval_examples_list,
    eval_features_list,  # list only for offset_mapping lookup
    seed: int = 42,
    n_total: int = 30,
    n_each: int = 10,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    ctx_max_chars: int = 600,
) -> Path:
    """
    Generate a compact error analysis report:
      - 10 worst: wrong & high confidence
      - 10 best:  correct & high confidence
      - 10 random
    Save to run_dir/reports/error_cases.json and return the path.
    """
    rnd = random.Random(seed)

    # only sample n_total examples to keep it fast
    n_total = min(n_total, len(eval_examples_list))
    sampled_idxs = rnd.sample(range(len(eval_examples_list)), n_total)
    sampled_exids = {eval_examples_list[i]["id"] for i in sampled_idxs}

    # predict logits on eval_features (Dataset)
    pred_out = trainer.predict(eval_features)
    start_logits, end_logits = pred_out.predictions

    # index features by example_id
    feats_by_exid = {}
    for fi, feat in enumerate(eval_features_list):
        exid = feat["example_id"]
        if exid in sampled_exids:
            feats_by_exid.setdefault(exid, []).append(fi)

    rows = []
    for i in sampled_idxs:
        ex = eval_examples_list[i]
        exid = ex["id"]
        context = ex["context"]
        question = ex["question"]
        gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

        cand = []
        for fi in feats_by_exid.get(exid, []):
            s = start_logits[fi]
            e = end_logits[fi]
            offsets = eval_features_list[fi]["offset_mapping"]

            # top indices
            s_top = sorted(range(len(s)), key=lambda j: s[j], reverse=True)[
                :n_best_size
            ]
            e_top = sorted(range(len(e)), key=lambda j: e[j], reverse=True)[
                :n_best_size
            ]

            for si in s_top:
                for ei in e_top:
                    if ei < si or (ei - si + 1) > max_answer_length:
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    a, _ = offsets[si]
                    _, b = offsets[ei]
                    if a is None or b is None or b <= a:
                        continue
                    text = context[a:b]
                    score = float(s[si] + e[ei])
                    cand.append((score, float(s[si]), float(e[ei]), text))

        if not cand:
            pred, score, sl, el, conf = "", 0.0, 0.0, 0.0, 0.0
        else:
            cand.sort(key=lambda x: x[0], reverse=True)
            score, sl, el, pred = cand[0]

            # confidence = softmax(best_score among top-K)
            topk = cand[: min(len(cand), n_best_size)]
            m = topk[0][0]
            exps = [math.exp(c[0] - m) for c in topk]
            conf = float(exps[0] / sum(exps))

        # EM with standard normalize
        correct = normalize_answer(pred) == normalize_answer(gold)

        rows.append(
            {
                "id": exid,
                "question": question,
                "context": (
                    context[:ctx_max_chars]
                    + ("..." if len(context) > ctx_max_chars else "")
                ),
                "gold_answer": gold,
                "pred_answer": pred,
                "correct_em": bool(correct),
                "confidence": conf,
                "start_logit": sl,
                "end_logit": el,
                "score": score,
            }
        )

    # select cases
    wrong = sorted(
        [r for r in rows if not r["correct_em"]],
        key=lambda r: r["confidence"],
        reverse=True,
    )
    right = sorted(
        [r for r in rows if r["correct_em"]],
        key=lambda r: r["confidence"],
        reverse=True,
    )

    worst = wrong[:n_each]
    best = right[:n_each]

    rnd_rows = rows[:]
    rnd.shuffle(rnd_rows)
    random_cases = rnd_rows[:n_each]

    report = {
        "summary": {
            "n_eval_examples_sampled": len(rows),
            "n_correct_em": sum(1 for r in rows if r["correct_em"]),
            "n_incorrect": sum(1 for r in rows if not r["correct_em"]),
        },
        "cases": {
            "worst": worst,
            "best": best,
            "random": random_cases,
        },
    }

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "error_cases.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def write_model_card_and_metadata(
    run_dir: Path,
    cfg: TrainConfig,
    metrics: Dict[str, float],
    data_stats: Dict[str, float],
    env_name: str,
    platform_str: str,
    tracking_uri: str,
) -> None:
    """Write MODEL_CARD.md and metadata.json into run_dir (copied to best if updated)."""
    em = float(metrics.get("eval_em", metrics.get("em", -1.0)))
    f1 = float(metrics.get("eval_f1", metrics.get("f1", -1.0)))

    # Model card (human readable)
    card = f"""# Model Card

## Overview
- **Base model**: {cfg.model_name}
- **Dataset**: {cfg.dataset_name}
- **Task**: Extractive Question Answering (SQuAD-style)
- **Validation metrics**: EM={em:.4f}, F1={f1:.4f}

## Training data sizes
- train examples: {int(data_stats.get("n_train_examples", -1))}
- eval examples: {int(data_stats.get("n_eval_examples", -1))}
- train features (after sliding window): {int(data_stats.get("n_train_features", -1))}
- eval features (after sliding window): {int(data_stats.get("n_eval_features", -1))}
- train feat/example: {data_stats.get("train_feat_per_ex", -1.0):.3f}
- eval feat/example: {data_stats.get("eval_feat_per_ex", -1.0):.3f}

## Runtime
- env: {env_name}
- platform: {platform_str}

## Training config (key)
- max_length: {cfg.max_length}
- doc_stride: {cfg.doc_stride}
- learning_rate: {cfg.learning_rate}
- epochs: {cfg.num_train_epochs}
- batch_size(train): {cfg.per_device_train_batch_size}
- fp16: {cfg.fp16}
- seed: {cfg.seed}

## Error analysis
- report path: reports/error_cases.json

## Intended use
- Course project / research experiments on SQuAD v1 style QA.
- Offline evaluation and demonstrations (not production-grade).

## Limitations
- Long context: may miss answers far from the selected window.
- Entity/number confusion when multiple similar mentions exist.
- Sensitive to paraphrases / wording differences.

## Notes
- MLflow tracking URI: {tracking_uri}
"""
    (run_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    # Metadata (machine readable)
    meta = {
        "base_model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "task": "extractive_qa",
        "metrics": {"em": em, "f1": f1},
        "data_sizes": {k: float(v) for k, v in data_stats.items()},
        "runtime": {"env": env_name, "platform": platform_str},
        "train_config": {
            "max_length": cfg.max_length,
            "doc_stride": cfg.doc_stride,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
            "fp16": cfg.fp16,
            "seed": cfg.seed,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
        },
        "error_analysis": {"path": "reports/error_cases.json"},
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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

    # Avoid accidental overwrite
    if run_dir.exists():
        i = 1
        while (MODELS_DIR / cfg.output_subdir / f"{cfg.run_name}_{i}").exists():
            i += 1
        cfg.run_name = f"{cfg.run_name}_{i}"
        run_dir = MODELS_DIR / cfg.output_subdir / cfg.run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    # Detect runtime env (for model card / metadata)
    detected_env = "colab" if os.getenv("COLAB_RELEASE_TAG") else "local"
    env_name = os.getenv("ENV_NAME", detected_env)
    platform_str = platform.platform()

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

    # Data sizes (for model card / metadata)
    data_stats: Dict[str, float] = {
        "n_train_examples": float(len(train_ds)),
        "n_eval_examples": float(len(eval_ds)),
        "n_train_features": float(len(train_features)),
        "n_eval_features": float(len(eval_features)),
        "train_feat_per_ex": float(len(train_features) / max(1, len(train_ds))),
        "eval_feat_per_ex": float(len(eval_features) / max(1, len(eval_ds))),
    }

    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    # TrainingArguments
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
        disable_tqdm=False,
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

    # Prepare lists for compute_em_f1()
    eval_examples_list = [eval_examples[i] for i in range(len(eval_examples))]
    eval_features_list = [eval_features[i] for i in range(len(eval_features))]

    # Start MLflow run
    with mlflow.start_run(run_name=cfg.run_name):
        # team/env/platform
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
            data_stats.update(stats)

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
        )

        train_result = trainer.train()

        # Always compute EM/F1 via predict (stable across versions)
        pred_out = trainer.predict(eval_features)
        start_logits, end_logits = pred_out.predictions

        qa_metrics = compute_em_f1(
            eval_examples=eval_examples_list,
            eval_features=eval_features_list,
            raw_predictions=(np.array(start_logits), np.array(end_logits)),
        )

        # current EM/F1
        cur_f1 = float(qa_metrics.get("f1", -1.0))
        cur_em = float(qa_metrics.get("em", qa_metrics.get("exact_match", -1.0)))

        # Error analysis report (sample 30)
        report_path = write_error_report(
            run_dir=run_dir,
            trainer=trainer,
            eval_features=eval_features,
            eval_examples_list=eval_examples_list,
            eval_features_list=eval_features_list,
            seed=cfg.seed,
            n_total=30,
            n_each=10,
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # Log metrics (train + qa + runtime)
        metrics: Dict[str, float] = {}
        if "train_loss" in train_result.metrics:
            metrics["train_loss"] = float(train_result.metrics["train_loss"])

        # log qa metrics with eval_ prefix
        for k, v in qa_metrics.items():
            if isinstance(v, (int, float, np.floating)):
                metrics[f"eval_{k}"] = float(v)

        log_metrics(metrics)

        # Write MODEL_CARD + metadata (copied to best if updated)
        write_model_card_and_metadata(
            run_dir=run_dir,
            cfg=cfg,
            metrics=metrics,
            data_stats=data_stats,
            env_name=env_name,
            platform_str=platform_str,
            tracking_uri=tracking_uri,
        )
        mlflow.log_artifact(str(run_dir / "MODEL_CARD.md"), artifact_path="reports")
        mlflow.log_artifact(str(run_dir / "metadata.json"), artifact_path="reports")

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
