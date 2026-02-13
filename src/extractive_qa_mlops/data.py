import re
from typing import Iterable, List


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


def best_window_match(context: str, answer: str) -> int:
    context_n = context.lower()
    answer_n = answer.lower().strip()
    if not answer_n:
        return -1
    return context_n.find(answer_n)


def iter_non_empty(items: Iterable[str]) -> List[str]:
    return [normalize_text(x) for x in items if normalize_text(x)]
