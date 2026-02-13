import re
from dataclasses import asdict
from typing import Any, Dict, Optional

from .data import best_window_match, chunk_text, normalize_text
from .settings import QASettings


def answer_question(
    context: str,
    question: str,
    settings: Optional[QASettings] = None,
) -> Dict[str, Any]:
    settings = settings or QASettings()

    ctx = normalize_text(context)[: settings.max_context_chars]
    q = normalize_text(question)

    if not ctx or not q:
        return {
            "answer": "",
            "score": 0.0,
            "start": -1,
            "end": -1,
            "settings": asdict(settings),
        }

    tokens = sorted({t for t in q.split(" ") if len(t) >= 3}, key=len, reverse=True)

    match_start = -1
    matched_token = None
    for tok in tokens:
        idx = best_window_match(ctx, tok)
        if idx != -1:
            match_start = idx
            matched_token = tok
            break

    if match_start == -1:
        # Fallback: return a simple "entity-like" answer from context (first capitalized word).
        m = re.search(r"\b[A-Z][a-z]{2,}\b", ctx)
        if not m:
            return {
                "answer": "",
                "score": 0.0,
                "start": -1,
                "end": -1,
                "settings": asdict(settings),
            }

        start = m.start()
        end = m.end()
        return {
            "answer": ctx[start:end],
            "score": 0.1,
            "start": start,
            "end": end,
            "settings": asdict(settings),
        }

    window = 80
    start = max(0, match_start - window)
    end = min(len(ctx), match_start + len(matched_token or "") + window)
    snippet = ctx[start:end].strip()

    score = min(1.0, (len(matched_token or "") / 10.0))

    return {
        "answer": snippet[: settings.max_answer_chars],
        "score": score,
        "start": start,
        "end": end,
        "settings": asdict(settings),
    }


def build_chunks(
    context: str, chunk_size: int = 500, overlap: int = 50
) -> Dict[str, Any]:
    chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
    return {"num_chunks": len(chunks), "chunks": chunks}
