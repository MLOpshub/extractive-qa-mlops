FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic transformers torch

ENV PYTHONPATH=/app/src
ENV MLRUNS_DIR=/app/artifacts/mlruns

EXPOSE 8000

CMD ["uvicorn", "extractive_qa_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
