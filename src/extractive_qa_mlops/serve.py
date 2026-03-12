from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from extractive_qa_mlops.paths import MODELS_DIR, PROJECT_ROOT


_TOKENIZER = None
_MODEL = None
_MODEL_PATH = None
_SERVE_CONFIG = None


def load_serve_config() -> dict:
    global _SERVE_CONFIG

    if _SERVE_CONFIG is None:
        config_path = PROJECT_ROOT / "configs" / "serve.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            _SERVE_CONFIG = yaml.safe_load(f) or {}

    return _SERVE_CONFIG


def get_model_path() -> Path:
    model_dir = MODELS_DIR / "best"

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
    settings: Optional[dict] = None,
) -> Dict[str, Any]:
    qa_cfg = settings or load_serve_config().get("qa", {})

    max_context_chars = qa_cfg.get("max_context_chars", 4000)
    max_answer_chars = qa_cfg.get("max_answer_chars", 200)
    inference_max_length = qa_cfg.get("inference_max_length", 512)

    context = (context or "").strip()
    question = (question or "").strip()

    empty = {
        "answer": "",
        "score": 0.0,
        "start": -1,
        "end": -1,
        "model_path": _MODEL_PATH or str(get_model_path()),
        "settings": {
            "max_context_chars": max_context_chars,
            "max_answer_chars": max_answer_chars,
            "inference_max_length": inference_max_length,
        },
    }

    if not context or not question:
        return empty

    tokenizer, model = get_model_and_tokenizer()

    truncated_context = context[:max_context_chars]

    inputs = tokenizer(
        question,
        truncated_context,
        return_tensors="pt",
        truncation=True,
        max_length=inference_max_length,
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
    answer = answer[:max_answer_chars]

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
            "max_context_chars": max_context_chars,
            "max_answer_chars": max_answer_chars,
            "inference_max_length": inference_max_length,
        },
    }


def build_chunks(
    context: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, Any]:
    from extractive_qa_mlops.text_utils import chunk_text

    chunk_cfg = load_serve_config().get("chunking", {})
    chunk_size = (
        chunk_size if chunk_size is not None else chunk_cfg.get("chunk_size", 500)
    )
    overlap = overlap if overlap is not None else chunk_cfg.get("overlap", 50)

    chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
    return {"num_chunks": len(chunks), "chunks": chunks}
