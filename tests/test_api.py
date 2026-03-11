from fastapi.testclient import TestClient

import extractive_qa_mlops.api as api_mod
from extractive_qa_mlops.api import app

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_qa_endpoint(monkeypatch) -> None:
    def fake_answer_question(context: str, question: str):
        return {
            "answer": "Paris",
            "score": 0.99,
            "start": 0,
            "end": 5,
            "model_path": "dummy-model-path",
        }

    monkeypatch.setattr(api_mod, "answer_question", fake_answer_question)

    payload = {
        "context": "France is a country in Europe. Paris is the capital of France.",
        "question": "What is the capital of France?",
    }

    response = client.post("/qa", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris"
    assert body["score"] == 0.99