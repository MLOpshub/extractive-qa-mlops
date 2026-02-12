from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into base and return base."""
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def build_train_cmd(train_entry: str, merged_config: Path) -> list[str]:
    """
    train_entry supports:
      - module path: "src.train"  -> python -m src.train
      - file path:   "src/train.py" -> python src/train.py
    """
    if train_entry.endswith(".py") or "/" in train_entry or "\\" in train_entry:
        return [sys.executable, train_entry, "--config", str(merged_config)]
    return [sys.executable, "-m", train_entry, "--config", str(merged_config)]


def main() -> None:
    root = repo_root()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="configs/train.yaml", help="Baseline train config")
    parser.add_argument("--sweep", default="configs/sweep.yaml", help="Sweep config")
    parser.add_argument(
        "--train_entry",
        default="src.train",
        help='Train entry: module ("src.train") or script path ("src/train.py")',
    )
    parser.add_argument("--out_dir", default="configs/_sweep_tmp", help="Temp dir for merged configs")
    args = parser.parse_args()

    train_path = resolve_from_root(args.train, root)
    sweep_path = resolve_from_root(args.sweep, root)
    out_dir = resolve_from_root(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    experiments = sweep_cfg.get("experiments") or []

    if not experiments:
        raise ValueError("No experiments found in sweep.yaml (expected key: experiments)")

    for exp in experiments:
        name = exp.get("name") or "exp"
        overrides = exp.get("overrides") or {}

        merged = deepcopy(base_cfg)
        deep_update(merged, overrides)

        merged.setdefault("run_name", "auto")

        merged_path = out_dir / f"train_{name}.yaml"
        merged_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

        print(f"\n=== Running experiment: {name} ===")
        cmd = build_train_cmd(args.train_entry, merged_path)
        print("CMD:", " ".join(cmd))

        subprocess.run(cmd, check=True, cwd=str(root))

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
