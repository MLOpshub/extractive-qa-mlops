import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")

app = FastAPI(title="QA Model API")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

class QARequest(BaseModel):
    question: str
    context: str
    max_answer_len: int = 80

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/qa")
def qa(req: QARequest):
    inputs = tokenizer(req.question, req.context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start = int(torch.argmax(outputs.start_logits, dim=-1))
        end = int(torch.argmax(outputs.end_logits, dim=-1)) + 1

    # safety: if end < start, fix
    if end <= start:
        end = start + 1

    answer_ids = inputs["input_ids"][0][start:end]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # limit length a bit
    if len(answer) > req.max_answer_len:
        answer = answer[: req.max_answer_len].rstrip()

    return {"answer": answer, "start": start, "end": end}
