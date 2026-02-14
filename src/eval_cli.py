from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.data import load_squad, preprocess_eval_features
from src.evaluate import compute_em_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to saved model folder")
    parser.add_argument("--dataset", default="squad", help="squad or squad_v2")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--samples", type=int, default=500, help="Eval subset size")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_dir))
    model.eval()

    # Load eval data
    ds = load_squad(args.dataset)
    eval_ds = ds["validation"]
    if args.samples is not None:
        eval_ds = eval_ds.select(range(min(args.samples, len(eval_ds))))

    eval_features = eval_ds.map(
        lambda x: preprocess_eval_features(
            x,
            tokenizer=tokenizer,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
        ),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # Prepare lists
    eval_examples_list = [eval_ds[i] for i in range(len(eval_ds))]
    eval_features_list = [eval_features[i] for i in range(len(eval_features))]

    # Run forward
    start = time.time()
    start_logits_all = []
    end_logits_all = []

    for feat in eval_features_list:
        inputs = {
            k: np.array([feat[k]]) for k in ["input_ids", "attention_mask"] if k in feat
        }
        if "token_type_ids" in feat:
            inputs["token_type_ids"] = np.array([feat["token_type_ids"]])

        with np.errstate(all="ignore"):
            out = model(**{k: torch_tensor(v) for k, v in inputs.items()})

        start_logits_all.append(out.start_logits.detach().cpu().numpy()[0])
        end_logits_all.append(out.end_logits.detach().cpu().numpy()[0])

    elapsed = time.time() - start

    # Metrics
    res = compute_em_f1(
        eval_examples=eval_examples_list,
        eval_features=eval_features_list,
        raw_predictions=(np.stack(start_logits_all), np.stack(end_logits_all)),
    )

    avg_ms = (elapsed / len(eval_features_list)) * 1000.0
    print(f"EM: {res['em']:.2f} | F1: {res['f1']:.2f}")
    print(f"Avg forward time: {avg_ms:.2f} ms/feature")


def torch_tensor(x: np.ndarray):
    # Numpy -> torch tensor (lazy import)
    import torch

    return torch.from_numpy(x)


if __name__ == "__main__":
    main()
