# Extractive Question Answering (BERT) — End-to-End MLOps Pipeline

This project builds an **extractive Question Answering** system using **BERT fine-tuned on SQuAD**, then serves it via **FastAPI** and ships it with **Docker**.

Pipeline (high-level):
SQuAD dataset → preprocessing/tokenization → BERT QA training on Colab GPU → save model artifacts (all artifacts in one Google Drive) → download `models/best/` to local → FastAPI loads model → Docker

---

## 1 Project Overview

**Goal:** Given a *question* and a *context paragraph*, return the **answer span** extracted from the context.

**Why this project:** It demonstrates an end-to-end workflow that includes:
- reproducible environment & dependencies
- training + evaluation
- model artifact management
- API serving
- containerization
- CI/testing hooks

### 2.2 Setup & Usage

**A) Training on Colab (GPU)**
1. Open the training notebook in Colab.
2. Run:
   - Set environment
   - Run `sweep.py` and save all models
   - Evaluate best model on SQuAD
   - Run MLflow UI via Colab

> **all artifacts** to a single Google Drive folder (**https://drive.google.com/drive/folders/10aV6fYGGOgDyS4CmPr3TiVpQJ7fiwEjA?usp=drive_link**).

**B) Download best model to local**
Download the best checkpoint folder from Drive into:

**C) Run FastAPI locally**
Install dependencies:
```bash
uv sync
# or: pip install -r requirements.txt
```
---

## 2 Problem Definition & Data

### Task
Extractive QA:
- Input: `question`, `context`
- Output: `answer_text`, `start_char`, `end_char`, and a confidence score

### Dataset
- **SQuAD v1 (Stanford Question Answering Dataset)**
We use SQuAD to fine-tune a pretrained BERT-style model for span prediction.

### Data split
- Train / Validation (standard split provided by dataset)

---

## 3 System Architecture

### 3.1 Repository Structure (*need to be modified after later*)
.
├── app/
│ ├── main.py # FastAPI entrypoint
│ ├── schemas.py # Pydantic request/response models
│ └── qa_inference.py # Model loading + prediction utilities
├── src/
│ ├── data/
│ │ ├── load_dataset.py # SQuAD loading
│ │ └── preprocess.py # tokenization + span alignment
│ ├── train.py # training script (Colab/local)
│ └── evaluate.py # evaluation script (optional)
├── models/
│ └── best/ # downloaded best checkpoint from Google Drive
├── tests/
│ └── test_api.py # basic API test(s)
├── Dockerfile
├── pyproject.toml
├── uv.lock # if using uv
└── README.md

### 3.2 Data & preprocessing
   - Load SQuAD
   - Clean and normalize text (lightweight)
   - Tokenize with a Hugging Face tokenizer
   - Map character-level answer spans → token-level start/end indices

### 3.3 Training (Colab GPU)
   - Fine-tune a BERT QA model on SQuAD
   - After training, we run Trainer.predict() on validation features and compute EM/F1 via our compute_em_f1() post-processing
   - Save checkpoints and the best model

### 3.4 Artifact management (Google Drive)

All training outputs are stored under a single Google Drive folder so that they persist across Colab sessions.
We keep **two top-level folders**:

- `artifacts/` — human-friendly training outputs (models, reports, experiment notes)
- `mlruns/` — MLflow **file store** (runs/metrics/params + logged artifacts)

**Drive layout:**
```txt
<extractive-qa-mlops>/
├─ artifacts/
│  └─ models/
│     ├─ best/
│     │  ├─ checkpoints/
│     │  ├─ reports/
│     │  │  └─ error_cases.json
│     │  ├─ MODEL_CARD.md
│     │  ├─ metadata.json
│     │  └─ model.safetensors + tokenizer files (config.json, tokenizer.json, vocab.txt, ...)
│     └─ experiments/
│        ├─ <run_name_1>/
│        │  ├─ checkpoints/
│        │  ├─ reports/
│        │  │  └─ error_cases.json
│        │  ├─ MODEL_CARD.md
│        │  ├─ metadata.json
│        │  └─ model.safetensors + tokenizer files
│        └─ <run_name_2>/ ...
└─ mlruns/
   └─ <experiment_id>/
      ├─ meta.yaml
      └─ <run_id>/
         ├─ metrics/
         ├─ params/
         ├─ tags/
         ├─ meta.yaml
         └─ artifacts/   (e.g., model/...)
```

**Best model selection:** after each run, we update `artifacts/models/best/` using **F1 (primary) and EM (secondary)** so downstream components can always load the latest best checkpoint.

### 3.5 Local inference
   - Download `models/best/` from Drive to local project folder
   - FastAPI loads the model at startup and serves predictions

### 3.6 Containerization
   - Docker image for running the inference service consistently

---

## 4 MLOps Practices

This repository aims to follow the course expectations (structure, reproducibility, serving, containerization, testing, etc.).

### 4.1 Reproducible environment
- Python dependencies are pinned (see `pyproject.toml` / lockfile)
- Recommended: `uv` for env & dependency management

### 4.2 Version control (Git/GitHub)
- Feature branches + PR reviews
- Clear commit messages with meaningful changes

### 4.3 Testing & code quality
- Unit tests (recommended target coverage per course)
- Pre-commit hooks (formatting/linting)

### 4.4 Model artifacts
Saved artifacts are organized so the inference service can load them directly:
- `models/best/` is the canonical folder used by FastAPI
- Each run produces a `MODEL_CARD.md` + `metadata.json` + `reports/error_cases.json` for traceability
- `models/best/` is automatically updated using F1 (primary) / EM (secondary)
---

## 5 Monitoring & Reliability

Basic reliability considerations:
- data profiling metrics (e.g., feature counts, context length stats)
- `/health` endpoint to confirm the API is running
- structured logs for requests and inference latency
- input validation via Pydantic models
- graceful error messages for missing/invalid input

---

## 6 Team Collaboration

- Work is tracked via Git commit history and PR reviews.
- Each member should understand the full pipeline: data → training → artifacts → serving.

> Team members
- Member 1: _Tengzhe ZHANG_ —> Data Preprocessing, Training and Evaluation
- Member 2 _RAJAGOPAL GAJENDRA Deepthi_ —> Serving
- Member 3 _MOURAD Reda_ —> Docker, CI, CD and Monitoring

---

## 7 Limitations & Future Work

### 7.1 Limitations
- Extractive QA depends on the answer being present in the provided context.
- Performance depends on domain similarity between SQuAD and real user contexts.

### 7.2 Future work
- Add retrieval (RAG-style) to fetch relevant contexts automatically
- Add better confidence calibration and abstention (“no answer”)
- Improve monitoring (metrics, dashboards)
- Automate model registry + deployment using MLflow
- Improve span decoding/post-processing (n-best aggregation, length constraints, normalization) to reduce boundary errors and close the EM–F1 gap.

---
