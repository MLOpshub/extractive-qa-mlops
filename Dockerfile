FROM python:3.11-slim

WORKDIR /app

# System deps (often needed by tokenizers/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# If you use pyproject.toml + uv.lock, easiest is pip install.
# Option A: requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Option B: pyproject.toml (no lock install here, simplest)
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic transformers torch

# Expose FastAPI
EXPOSE 8000

# Default model dir inside container (we copied models/ into /app/models)
ENV MODEL_DIR=/app/models/best

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]