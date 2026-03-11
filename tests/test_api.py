from fastapi.testclient import TestClient

from extractive_qa_mlops.api import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert "message" in body
    assert body["message"] == "Extractive QA API is running"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "max_context_chars" in body
    assert "max_answer_chars" in body


def test_qa_endpoint() -> None:
    payload = {
        "context": "France is a country in Europe. Paris is the capital of France.",
        "question": "What is the capital of France?",
    }

    response = client.post("/qa", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "answer" in body
    assert "score" in body
    assert "start" in body
    assert "end" in body
    assert "settings" in body


def test_chunks_endpoint() -> None:
    payload = {
        "context": "This is a test context. " * 50,
        "chunk_size": 100,
        "overlap": 20,
    }

    response = client.post("/chunks", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "num_chunks" in body
    assert "chunks" in body
    assert isinstance(body["chunks"], list)