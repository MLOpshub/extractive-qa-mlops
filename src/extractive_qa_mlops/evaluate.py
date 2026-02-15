from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
import evaluate


def postprocess_qa_predictions(
    examples: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Dict[str, str]:
    # Convert logits -> text answers
    start_logits, end_logits = raw_predictions

    # Map example id -> index
    ex_id_to_idx = {ex["id"]: i for i, ex in enumerate(examples)}

    # Collect feature indices per example
    feats_per_ex: Dict[int, List[int]] = {}
    for i, f in enumerate(features):
        ex_idx = ex_id_to_idx[f["example_id"]]
        feats_per_ex.setdefault(ex_idx, []).append(i)

    preds: Dict[str, str] = {}

    for ex_idx, ex in enumerate(examples):
        context = ex["context"]
        best_score = -1e9
        best_text = ""

        for fi in feats_per_ex.get(ex_idx, []):
            s = start_logits[fi]
            e = end_logits[fi]
            offsets = features[fi]["offset_mapping"]

            start_idxs = np.argsort(s)[-n_best_size:][::-1]
            end_idxs = np.argsort(e)[-n_best_size:][::-1]

            for si in start_idxs:
                for ei in end_idxs:
                    if si >= len(offsets) or ei >= len(offsets):
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    if ei < si:
                        continue
                    if (ei - si + 1) > max_answer_length:
                        continue

                    start_char, _ = offsets[si]
                    _, end_char = offsets[ei]
                    score = float(s[si] + e[ei])

                    if score > best_score:
                        best_score = score
                        best_text = context[start_char:end_char]

        preds[ex["id"]] = best_text

    return preds


def compute_em_f1(
    eval_examples: List[Dict[str, Any]],
    eval_features: List[Dict[str, Any]],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, float]:
    # EM/F1 for SQuAD
    metric = evaluate.load("squad")
    pred_dict = postprocess_qa_predictions(
        eval_examples, eval_features, raw_predictions
    )

    predictions = [{"id": k, "prediction_text": v} for k, v in pred_dict.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    res = metric.compute(predictions=predictions, references=references)
    return {"em": float(res["exact_match"]), "f1": float(res["f1"])}
