from __future__ import annotations

from typing import Iterable, List


def normalize_text(text: str) -> str:
    """Normalize whitespace (matches tests)."""
    return " ".join((text or "").strip().split())


def iter_non_empty(items: Iterable[str]) -> List[str]:
    """Strip items and keep only non-empty strings (matches tests)."""
    out: List[str] = []
    for x in items:
        s = (x or "").strip()
        if s:
            out.append(s)
    return out


def best_window_match(text: str, needle: str) -> int:
    """Case-insensitive substring match; return start index or -1 (matches tests)."""
    if not text or not needle:
        return -1
    return text.lower().find(needle.lower())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping character chunks.

    Tests expect:
    - chunk_size=0 raises ValueError
    - for 120 chars with chunk_size=50 overlap=10 => len(chunks) >= 3
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    text = text or ""
    if not text:
        return []

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size - 1 if chunk_size > 1 else 0

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step

    return chunks
