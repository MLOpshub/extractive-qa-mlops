from pathlib import Path

import pytest
from transformers import AutoTokenizer

MODEL_DIR = Path("artifacts/models")


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local model directory not available")
def test_local_model_loads():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    assert tokenizer is not None