import numpy as np

import extractive_qa_mlops.evaluate as ev


def test_postprocess_qa_predictions_picks_best_span():
    examples = [{"id": "ex1", "context": "Hello Paris world"}]

    # One feature for ex1 with offsets for tokens: [Hello][Paris][world]
    features = [
        {
            "example_id": "ex1",
            "offset_mapping": [(0, 5), (6, 11), (12, 17)],
        }
    ]

    # logits pick "Paris" (index 1)
    start_logits = np.array([[0.0, 10.0, 0.0]])
    end_logits = np.array([[0.0, 10.0, 0.0]])

    preds = ev.postprocess_qa_predictions(
        examples=examples,
        features=features,
        raw_predictions=(start_logits, end_logits),
        n_best_size=2,
        max_answer_length=10,
    )
    assert preds["ex1"] == "Paris"


def test_compute_em_f1_uses_metric_and_returns_floats(monkeypatch):
    # Fake metric object
    class DummyMetric:
        def compute(self, predictions, references):
            # basic assertions that shapes are right
            assert isinstance(predictions, list) and len(predictions) == 1
            assert isinstance(references, list) and len(references) == 1
            return {"exact_match": 50.0, "f1": 60.0}

    # Monkeypatch evaluate.load inside module
    monkeypatch.setattr(ev.evaluate, "load", lambda name: DummyMetric())

    eval_examples = [
        {
            "id": "ex1",
            "context": "Hello Paris world",
            "answers": {"text": ["Paris"], "answer_start": [6]},
        }
    ]
    eval_features = [
        {
            "example_id": "ex1",
            "offset_mapping": [(0, 5), (6, 11), (12, 17)],
        }
    ]
    start_logits = np.array([[0.0, 10.0, 0.0]])
    end_logits = np.array([[0.0, 10.0, 0.0]])

    out = ev.compute_em_f1(eval_examples, eval_features, (start_logits, end_logits))
    assert out["em"] == 50.0
    assert out["f1"] == 60.0
    assert isinstance(out["em"], float)
    assert isinstance(out["f1"], float)
