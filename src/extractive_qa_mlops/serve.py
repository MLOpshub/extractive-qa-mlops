from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from extractive_qa_mlops.settings import QASettings, MLRUNS_DIR


_TOKENIZER = None
_MODEL = None
_MODEL_PATH = None


def get_model_path() -> Path:
    model_dir = MLRUNS_DIR / "best"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {model_dir}")

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]

    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required model files in {model_dir}: {', '.join(missing)}"
        )

    return model_dir


def get_model_and_tokenizer():
    global _TOKENIZER, _MODEL, _MODEL_PATH

    if _TOKENIZER is None or _MODEL is None:
        model_path = get_model_path()
        _MODEL_PATH = str(model_path)

        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_PATH)
        _MODEL = AutoModelForQuestionAnswering.from_pretrained(_MODEL_PATH)
        _MODEL.eval()

    return _TOKENIZER, _MODEL


def answer_question(
    context: str,
    question: str,
    settings: Optional[QASettings] = None,
) -> Dict[str, Any]:
    settings = settings or QASettings()

    context = (context or "").strip()
    question = (question or "").strip()

    empty = {
        "answer": "",
        "score": 0.0,
        "start": -1,
        "end": -1,
        "model_path": _MODEL_PATH or str(get_model_path()),
        "settings": {
            "max_context_chars": settings.max_context_chars,
            "max_answer_chars": settings.max_answer_chars,
        },
    }

    if not context or not question:
        return empty

    tokenizer, model = get_model_and_tokenizer()

    truncated_context = context[: settings.max_context_chars]

    inputs = tokenizer(
        question,
        truncated_context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    answer = answer[: settings.max_answer_chars]

    start_score = torch.softmax(start_logits, dim=1)[0][start_idx].item()
    end_score = torch.softmax(end_logits, dim=1)[0][end_idx].item()
    score = float((start_score + end_score) / 2)

    return {
        "answer": answer,
        "score": score,
        "start": start_idx,
        "end": end_idx,
        "model_path": _MODEL_PATH or str(get_model_path()),
        "settings": {
            "max_context_chars": settings.max_context_chars,
            "max_answer_chars": settings.max_answer_chars,
        },
    }


def build_chunks(
    context: str, chunk_size: int = 500, overlap: int = 50
) -> Dict[str, Any]:
    from extractive_qa_mlops.text_utils import chunk_text

    chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
    return {"num_chunks": len(chunks), "chunks": chunks}


# from dataclasses import asdict
# from typing import Any, Dict, List, Optional

# from extractive_qa_mlops.text_utils import best_window_match, chunk_text, normalize_text
# from extractive_qa_mlops.settings import QASettings


# def answer_question(
#     context: str,
#     question: str,
#     settings: Optional[QASettings] = None,
# ) -> Dict[str, Any]:
#     """
#     Lightweight QA baseline used for CI/tests.

#     This is intentionally model-free: it provides a deterministic answer
#     without requiring trained artifacts.
#     """
#     settings = settings or QASettings()

#     ctx = normalize_text(context)[: settings.max_context_chars]
#     q = normalize_text(question)

#     # Default empty response (stable schema for tests)
#     empty = {
#         "answer": "",
#         "score": 0.0,
#         "start": -1,
#         "end": -1,
#         "settings": asdict(settings),
#     }

#     if not ctx or not q:
#         return empty

#     # Find a token from the question that appears in the context
#     tokens = sorted({t for t in q.split() if len(t) >= 3}, key=len, reverse=True)

#     match_start = -1
#     matched_token = ""
#     for tok in tokens:
#         idx = best_window_match(ctx, tok)
#         if idx != -1:
#             match_start = idx
#             matched_token = tok
#             break

#     if match_start == -1:
#         # Fallback: return first "word-like" chunk from context
#         # (avoid relying on capitalization which may not exist in normalized text)
#         m = re.search(r"\b\w+\b", ctx)
#         if not m:
#             return empty

#         start, end = m.start(), m.end()
#         return {
#             "answer": ctx[start:end],
#             "score": 0.1,
#             "start": start,
#             "end": end,
#             "settings": asdict(settings),
#         }

#     window = 80
#     start = max(0, match_start - window)
#     end = min(len(ctx), match_start + len(matched_token) + window)

#     snippet = ctx[start:end].strip()
#     score = min(1.0, (len(matched_token) / 10.0)) if matched_token else 0.2

#     return {
#         "answer": snippet[: settings.max_answer_chars],
#         "score": float(score),
#         "start": int(start),
#         "end": int(end),
#         "settings": asdict(settings),
#     }


# def build_chunks(
#     context: str, chunk_size: int = 500, overlap: int = 50
# ) -> Dict[str, Any]:
#     """
#     Utility used by tests: returns a list of chunks and a count.
#     """
#     chunks: List[str] = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
#     return {"num_chunks": len(chunks), "chunks": chunks}
