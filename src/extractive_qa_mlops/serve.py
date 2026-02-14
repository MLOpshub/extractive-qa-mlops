from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from extractive_qa_mlops.text_utils import best_window_match, chunk_text, normalize_text
from extractive_qa_mlops.settings import QASettings


def answer_question(
    context: str,
    question: str,
    settings: Optional[QASettings] = None,
) -> Dict[str, Any]:
    """
    Lightweight QA baseline used for CI/tests.

    This is intentionally model-free: it provides a deterministic answer
    without requiring trained artifacts.
    """
    settings = settings or QASettings()

    ctx = normalize_text(context)[: settings.max_context_chars]
    q = normalize_text(question)

    # Default empty response (stable schema for tests)
    empty = {
        "answer": "",
        "score": 0.0,
        "start": -1,
        "end": -1,
        "settings": asdict(settings),
    }

    if not ctx or not q:
        return empty

    # Find a token from the question that appears in the context
    tokens = sorted({t for t in q.split() if len(t) >= 3}, key=len, reverse=True)

    match_start = -1
    matched_token = ""
    for tok in tokens:
        idx = best_window_match(ctx, tok)
        if idx != -1:
            match_start = idx
            matched_token = tok
            break

    if match_start == -1:
        # Fallback: return first "word-like" chunk from context
        # (avoid relying on capitalization which may not exist in normalized text)
        m = re.search(r"\b\w+\b", ctx)
        if not m:
            return empty

        start, end = m.start(), m.end()
        return {
            "answer": ctx[start:end],
            "score": 0.1,
            "start": start,
            "end": end,
            "settings": asdict(settings),
        }

    window = 80
    start = max(0, match_start - window)
    end = min(len(ctx), match_start + len(matched_token) + window)

    snippet = ctx[start:end].strip()
    score = min(1.0, (len(matched_token) / 10.0)) if matched_token else 0.2

    return {
        "answer": snippet[: settings.max_answer_chars],
        "score": float(score),
        "start": int(start),
        "end": int(end),
        "settings": asdict(settings),
    }


def build_chunks(
    context: str, chunk_size: int = 500, overlap: int = 50
) -> Dict[str, Any]:
    """
    Utility used by tests: returns a list of chunks and a count.
    """
    chunks: List[str] = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
    return {"num_chunks": len(chunks), "chunks": chunks}
