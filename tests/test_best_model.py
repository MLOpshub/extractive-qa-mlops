import json
from pathlib import Path

from extractive_qa_mlops.best_model import is_better, load_best_meta, save_best_meta


def test_load_best_meta_when_missing(tmp_path: Path):
    best_dir = tmp_path / "best"
    meta = load_best_meta(best_dir)
    assert meta["best_f1"] == -1.0
    assert meta["best_em"] == -1.0


def test_save_and_load_best_meta(tmp_path: Path):
    best_dir = tmp_path / "best"
    payload = {"best_f1": 77.7, "best_em": 55.5, "note": "ok"}

    save_best_meta(best_dir, payload)

    # file exists and is valid json
    meta_path = best_dir / "metadata.json"
    assert meta_path.exists()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["best_f1"] == 77.7

    # load returns the same content
    loaded = load_best_meta(best_dir)
    assert loaded["best_em"] == 55.5
    assert loaded["note"] == "ok"


def test_is_better_prefers_higher_f1():
    assert is_better(new_f1=0.9, new_em=0.0, best_f1=0.8, best_em=1.0) is True
    assert is_better(new_f1=0.7, new_em=999, best_f1=0.8, best_em=0.0) is False


def test_is_better_ties_break_on_em():
    assert is_better(new_f1=0.8, new_em=0.6, best_f1=0.8, best_em=0.5) is True
    assert is_better(new_f1=0.8, new_em=0.4, best_f1=0.8, best_em=0.5) is False
