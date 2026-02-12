from __future__ import annotations

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into base"""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="configs/train.yaml", help="Baseline train config")
    parser.add_argument("--sweep", default="configs/sweep.yaml", help="Sweep config")
    parser.add_argument("--train_py", default="train.py", help="Path to train.py")
    parser.add_argument("--out_dir", default="configs/_sweep_tmp", help="Temp dir for merged configs")
    args = parser.parse_args()

    train_path = Path(args.train)
    sweep_path = Path(args.sweep)
    train_py = Path(args.train_py)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    experiments = sweep_cfg.get("experiments", [])

    if not experiments:
        raise ValueError("No experiments found in sweep.yaml (expected key: experiments)")

    for exp in experiments:
        name = exp.get("name", "exp")
        overrides = exp.get("overrides", {}) or {}

        merged = deepcopy(base_cfg)
        deep_update(merged, overrides)

        if not merged.get("run_name"):
            merged["run_name"] = "auto"

        merged_path = out_dir / f"train_{name}.yaml"
        merged_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

        print(f"\n=== Running experiment: {name} ===")
        cmd = ["python", str(train_py), "--config", str(merged_path)]
        print("CMD:", " ".join(cmd))

        # Run and fail fast if any experiment fails
        subprocess.run(cmd, check=True)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
