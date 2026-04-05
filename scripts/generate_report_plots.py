#!/usr/bin/env python3
"""
Build matplotlib figures from results/training_log.csv for report.md.

Run from project root:
  PYTHONPATH=. python scripts/generate_report_plots.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "report" / "figures"
LOG_PATH = ROOT / "results" / "training_log.csv"


def load_aggregated() -> dict[tuple[str, int], dict]:
    """Best row per (algo, run_index) after filtering short smoke runs."""
    rows: list[dict[str, str]] = []
    with LOG_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    kept: list[dict[str, str]] = []
    for r in rows:
        algo = r.get("algo", "").strip()
        try:
            ts = int(float(r.get("timesteps") or 0))
        except ValueError:
            ts = 0
        if algo in ("dqn", "ppo", "a2c") and ts < 50_000:
            continue
        if algo == "reinforce" and ts < 100:
            continue
        kept.append(r)

    by: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for r in kept:
        key = (r["algo"], int(r["run_index"]))
        by[key].append(r)

    best: dict[tuple[str, int], dict] = {}
    for key, lst in by.items():
        top = max(lst, key=lambda x: float(x.get("eval_mean_return") or -1e9))
        best[key] = top
    return best


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = load_aggregated()

    algos_sweep = ["dqn", "reinforce", "ppo"]
    colors = {"dqn": "#2563eb", "reinforce": "#16a34a", "ppo": "#ea580c"}
    run_indices = list(range(10))

    # --- Figure 1: Mean eval return vs hyperparameter run index (3 lines + error bars) ---
    fig1, ax1 = plt.subplots(figsize=(10, 5.5))
    for algo in algos_sweep:
        means: list[float] = []
        stds: list[float] = []
        for ri in run_indices:
            row = agg.get((algo, ri))
            if row is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row["eval_mean_return"]))
                stds.append(float(row.get("eval_std_return") or 0))
        arr_m = np.array(means, dtype=float)
        arr_s = np.array(stds, dtype=float)
        ax1.plot(
            run_indices,
            arr_m,
            "o-",
            label=algo.upper(),
            color=colors[algo],
            linewidth=2,
            markersize=6,
        )
        ax1.fill_between(
            run_indices,
            arr_m - arr_s,
            arr_m + arr_s,
            alpha=0.15,
            color=colors[algo],
        )

    ax1.set_xlabel("Hyperparameter run index (0–9)", fontsize=11)
    ax1.set_ylabel("Mean evaluation return (20 eval episodes)", fontsize=11)
    ax1.set_title("Hyperparameter sweep: post-training evaluation performance", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(run_indices)
    fig1.tight_layout()
    p1 = OUT_DIR / "fig1_eval_return_vs_run_index.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # --- Figure 2: Best vs mean return per algorithm ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    labels: list[str] = []
    best_vals: list[float] = []
    mean_vals: list[float] = []
    for algo in algos_sweep + ["a2c"]:
        vals = [
            float(agg[(algo, ri)]["eval_mean_return"])
            for ri in run_indices
            if (algo, ri) in agg
        ]
        if not vals:
            continue
        labels.append(algo.upper())
        best_vals.append(max(vals))
        mean_vals.append(float(np.mean(vals)))

    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w / 2, best_vals, w, label="Best run (max)", color="#1d4ed8")
    ax2.bar(x + w / 2, mean_vals, w, label="Mean over 10 runs", color="#93c5fd")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Evaluation mean return")
    ax2.set_title("Best vs average performance by algorithm")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = OUT_DIR / "fig2_best_vs_mean_by_algo.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # --- Figure 3: Three stacked subplots — bar per run index for each algo ---
    fig3, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, algo in zip(axes, algos_sweep):
        vals = [
            float(agg[(algo, ri)]["eval_mean_return"]) if (algo, ri) in agg else 0.0
            for ri in run_indices
        ]
        ax.bar(run_indices, vals, color=colors[algo], edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Mean return")
        ax.set_title(f"{algo.upper()} — all 10 hyperparameter configurations")
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("Run index")
    fig3.suptitle("Evaluation return for each hyperparameter row", fontsize=14, y=1.02)
    fig3.tight_layout()
    p3 = OUT_DIR / "fig3_per_algo_bars.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    print("Wrote:", p1.relative_to(ROOT))
    print("Wrote:", p2.relative_to(ROOT))
    print("Wrote:", p3.relative_to(ROOT))


if __name__ == "__main__":
    main()
