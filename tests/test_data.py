import pytest

from extractive_qa_mlops.data import (
    best_window_match,
    chunk_text,
    iter_non_empty,
    normalize_text,
)


def test_normalize_text():
    assert normalize_text("  hello   world ") == "hello world"


def test_chunk_text_basic():
    text = "a" * 120
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 3


def test_chunk_text_invalid():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)


def test_best_window_match():
    assert best_window_match("Hello World", "world") == 6


def test_iter_non_empty():
    assert iter_non_empty([" a ", "", "b"]) == ["a", "b"]
