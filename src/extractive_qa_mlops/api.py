from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from extractive_qa_mlops.serve import answer_question, build_chunks
from extractive_qa_mlops.settings import QASettings

app = FastAPI(
    title="Extractive QA API",
    version="1.0.0",
    description="FastAPI service for lightweight extractive QA",
)


class QARequest(BaseModel):
    context: str = Field(..., min_length=1, description="Input context paragraph")
    question: str = Field(..., min_length=1, description="Question to answer")


class ChunkRequest(BaseModel):
    context: str = Field(..., min_length=1, description="Input context paragraph")
    chunk_size: int = Field(500, ge=50, le=5000)
    overlap: int = Field(50, ge=0, le=1000)


@app.get("/")
def root() -> dict:
    return {
        "message": "Extractive QA API is running",
        "docs": "/docs",
        "health": "/health",
        "qa_endpoint": "/qa",
        "chunks_endpoint": "/chunks",
    }


@app.get("/health")
def health() -> dict:
    settings = QASettings()
    return {
        "status": "ok",
        "max_context_chars": settings.max_context_chars,
        "max_answer_chars": settings.max_answer_chars,
    }


@app.post("/qa")
def qa(request: QARequest) -> dict:
    return answer_question(
        context=request.context,
        question=request.question,
    )


@app.post("/chunks")
def chunks(request: ChunkRequest) -> dict:
    return build_chunks(
        context=request.context,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )