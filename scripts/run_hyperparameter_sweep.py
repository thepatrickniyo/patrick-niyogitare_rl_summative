#!/usr/bin/env python3
"""
Launch all 10 hyperparameter runs for each algorithm (40 jobs total).

Uses subprocess so each process has a clean SB3 state. Reduce timesteps for smoke tests.

  PYTHONPATH=. python scripts/run_hyperparameter_sweep.py --timesteps 20000 --episodes 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=2500, help="REINFORCE episodes per run")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    env = {"PYTHONPATH": str(ROOT), **dict(__import__("os").environ)}
    cmds: list[list[str]] = []
    for i in range(10):
        cmds.append(
            [
                sys.executable,
                str(ROOT / "training" / "dqn_training.py"),
                "--run_index",
                str(i),
                "--timesteps",
                str(args.timesteps),
            ]
        )
    for i in range(10):
        for algo in ("ppo", "a2c"):
            cmds.append(
                [
                    sys.executable,
                    str(ROOT / "training" / "pg_training.py"),
                    "--algo",
                    algo,
                    "--run_index",
                    str(i),
                    "--timesteps",
                    str(args.timesteps),
                ]
            )
    for i in range(10):
        cmds.append(
            [
                sys.executable,
                str(ROOT / "training" / "pg_training.py"),
                "--algo",
                "reinforce",
                "--run_index",
                str(i),
                "--episodes",
                str(args.episodes),
            ]
        )

    print(f"Planned {len(cmds)} training runs.")
    if args.dry_run:
        for c in cmds:
            print(" ", " ".join(c))
        return

    for idx, c in enumerate(cmds):
        print(f"\n>>> [{idx + 1}/{len(cmds)}] ", " ".join(c))
        subprocess.run(c, cwd=str(ROOT), env=env, check=True)

    print("\nDone. Inspect results/training_log.csv and update results/best_models.json.")


if __name__ == "__main__":
    main()
