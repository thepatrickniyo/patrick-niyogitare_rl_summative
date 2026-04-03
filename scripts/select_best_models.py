#!/usr/bin/env python3
"""Pick best eval_mean_return per algorithm from results/training_log.csv → best_models.json."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    log_path = ROOT / "results" / "training_log.csv"
    if not log_path.exists():
        raise SystemExit(f"Missing {log_path}; run training first.")

    by_algo: dict[str, list[dict]] = defaultdict(list)
    with log_path.open() as f:
        for row in csv.DictReader(f):
            by_algo[row["algo"]].append(row)

    best: dict = {}
    for algo, rows in by_algo.items():
        def score(r):
            return float(r.get("eval_mean_return", 0.0))

        top = max(rows, key=score)
        entry = {
            "algo": algo,
            "path": top["model_path"],
            "eval_mean_return": float(top["eval_mean_return"]),
            "run_index": int(top["run_index"]),
        }
        hk = "hp_hidden"
        if algo == "reinforce" and hk in top and top[hk] not in ("", None):
            entry["hidden"] = int(float(top[hk]))
        best[algo] = entry

    out = {"best": best, "note": "Regenerate after full hyperparameter sweep."}
    out_path = ROOT / "results" / "best_models.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(out_path)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
