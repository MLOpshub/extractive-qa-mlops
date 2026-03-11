from pathlib import Path

import pytest
from transformers import AutoTokenizer

MODEL_DIR = Path("artifacts/mlruns/best")


def _has_tokenizer_files(model_dir: Path) -> bool:
    required = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return model_dir.exists() and all((model_dir / f).exists() for f in required)


@pytest.mark.skipif(
    not _has_tokenizer_files(MODEL_DIR),
    reason="Complete local tokenizer artifacts are not available",
)
def test_local_model_loads():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    assert tokenizer is not None