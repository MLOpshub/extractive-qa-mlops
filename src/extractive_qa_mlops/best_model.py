from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_best_meta(best_dir: Path) -> Dict[str, Any]:
    # Read best metadata
    meta_path = best_dir / "metadata.json"
    if not meta_path.exists():
        return {"best_f1": -1.0, "best_em": -1.0}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def is_better(new_f1: float, new_em: float, best_f1: float, best_em: float) -> bool:
    # F1 first, EM second
    if new_f1 > best_f1:
        return True
    if new_f1 == best_f1 and new_em > best_em:
        return True
    return False


def save_best_meta(best_dir: Path, meta: Dict[str, Any]) -> None:
    # Write best metadata
    best_dir.mkdir(parents=True, exist_ok=True)
    meta_path = best_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
