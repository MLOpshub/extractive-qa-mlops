from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock


import extractive_qa_mlops.data as data_mod


class FakeBatchEncoding(dict):
    """Mimic HuggingFace BatchEncoding enough for our functions."""

    def __init__(
        self,
        base: Dict[str, Any],
        seq_ids_per_feature: List[List[int | None]],
    ):
        super().__init__(base)
        self._seq_ids_per_feature = seq_ids_per_feature

    def sequence_ids(self, i: int):
        return self._seq_ids_per_feature[i]


class FakeTokenizer:
    """Callable tokenizer that returns a controlled BatchEncoding."""

    def __init__(self, batch_encoding: FakeBatchEncoding):
        self._be = batch_encoding
        self.calls = []

    def __call__(self, questions, contexts, **kwargs):
        # Record call for assertions if needed
        self.calls.append((questions, contexts, kwargs))
        return self._be


def test_load_squad_calls_datasets_load_dataset(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(data_mod, "load_dataset", mock)

    data_mod.load_squad("squad")

    mock.assert_called_once_with("squad")


def test_preprocess_train_features_no_answer_sets_zeros():
    examples = {
        "question": ["  what?  "],
        "context": ["Hello Paris world"],
        "answers": [{"answer_start": [], "text": []}],
    }

    be = FakeBatchEncoding(
        base={
            "overflow_to_sample_mapping": [0],
            "offset_mapping": [[(0, 5), (6, 11), (12, 17)]],
        },
        # seq ids: first token question(0), then context(1), then context(1)
        seq_ids_per_feature=[[0, 1, 1]],
    )
    tok = FakeTokenizer(be)

    out = data_mod.preprocess_train_features(examples, tok, max_length=8, doc_stride=2)

    assert out["start_positions"] == [0]
    assert out["end_positions"] == [0]


def test_preprocess_train_features_answer_outside_context_sets_zeros():
    # answer starts before context span in offsets -> should become (0,0)
    examples = {
        "question": ["where"],
        "context": ["Hello Paris world"],
        "answers": [{"answer_start": [0], "text": ["Hello"]}],
    }

    be = FakeBatchEncoding(
        base={
            "overflow_to_sample_mapping": [0],
            # Context offsets start later (simulate truncation / mismatch)
            "offset_mapping": [[(10, 15), (16, 21), (22, 27)]],
        },
        seq_ids_per_feature=[[0, 1, 1]],
    )
    tok = FakeTokenizer(be)

    out = data_mod.preprocess_train_features(examples, tok)

    assert out["start_positions"] == [0]
    assert out["end_positions"] == [0]


def test_preprocess_train_features_answer_inside_context_finds_token_span():
    # answer "Paris" is at chars 6..11, matches offsets index 1 exactly
    examples = {
        "question": ["where"],
        "context": ["Hello Paris world"],
        "answers": [{"answer_start": [6], "text": ["Paris"]}],
    }

    be = FakeBatchEncoding(
        base={
            "overflow_to_sample_mapping": [0],
            "offset_mapping": [[(0, 5), (6, 11), (12, 17)]],
        },
        # token 0 is question, tokens 1..2 are context
        seq_ids_per_feature=[[0, 1, 1]],
    )
    tok = FakeTokenizer(be)

    out = data_mod.preprocess_train_features(examples, tok)

    # Expect "Paris" to map to token index 1..1
    assert out["start_positions"] == [1]
    assert out["end_positions"] == [1]


def test_preprocess_eval_features_masks_non_context_offsets_and_sets_example_id():
    examples = {
        "id": ["ex1"],
        "question": ["  q  "],
        "context": ["Hello Paris world"],
    }

    be = FakeBatchEncoding(
        base={
            "overflow_to_sample_mapping": [0],
            "offset_mapping": [[(0, 1), (2, 3), (4, 5)]],
        },
        # token 0 is question(0), token 1 is context(1), token 2 is padding(None)
        seq_ids_per_feature=[[0, 1, None]],
    )
    tok = FakeTokenizer(be)

    out = data_mod.preprocess_eval_features(examples, tok)

    # offset_mapping: keep only seq_id == 1, others -> None
    assert out["offset_mapping"][0][0] is None
    assert out["offset_mapping"][0][1] == (2, 3)
    assert out["offset_mapping"][0][2] is None

    assert out["example_id"] == ["ex1"]
