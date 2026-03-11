from extractive_qa_mlops.serve import answer_question, build_chunks


def test_answer_question_empty():
    out = answer_question("", "")
    assert out["answer"] == ""
    assert out["score"] == 0.0


def test_answer_question_match():
    context = "Paris is beautiful. I studied data engineering in Paris."
    question = "What city is mentioned?"
    out = answer_question(context, question)
    assert out["score"] > 0.0
    assert "Paris" in out["answer"]


def test_build_chunks():
    out = build_chunks("a" * 200, chunk_size=50, overlap=10)
    assert out["num_chunks"] > 0
