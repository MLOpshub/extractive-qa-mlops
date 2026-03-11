import torch

import extractive_qa_mlops.serve as serve_mod
from extractive_qa_mlops.serve import answer_question, build_chunks


def test_answer_question_empty(monkeypatch):
    monkeypatch.setattr(serve_mod, "_MODEL_PATH", "dummy-model-path")

    out = answer_question("", "")

    assert out["answer"] == ""
    assert out["score"] == 0.0
    assert out["model_path"] == "dummy-model-path"


def test_answer_question_match(monkeypatch):
    monkeypatch.setattr(serve_mod, "_MODEL_PATH", "dummy-model-path")

    class DummyTokenizer:
        def __call__(self, question, context, return_tensors, truncation, max_length):
            return {
                "input_ids": torch.tensor([[101, 2000, 3000, 102]]),
            }

        def decode(self, answer_ids, skip_special_tokens=True):
            return "Paris"

    class DummyOutputs:
        def __init__(self):
            self.start_logits = torch.tensor([[0.1, 0.2, 0.9, 0.1]])
            self.end_logits = torch.tensor([[0.1, 0.2, 0.9, 0.1]])

    class DummyModel:
        def __call__(self, **inputs):
            return DummyOutputs()

    def fake_get_model_and_tokenizer():
        return DummyTokenizer(), DummyModel()

    monkeypatch.setattr(serve_mod, "get_model_and_tokenizer", fake_get_model_and_tokenizer)

    context = "Paris is beautiful. I studied data engineering in Paris."
    question = "What city is mentioned?"

    out = answer_question(context, question)

    assert out["score"] > 0.0
    assert out["answer"] == "Paris"
    assert out["model_path"] == "dummy-model-path"


def test_build_chunks():
    out = build_chunks("a" * 200, chunk_size=50, overlap=10)
    assert out["num_chunks"] > 0