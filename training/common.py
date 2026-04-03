"""Shared helpers: project root on path, env factory, logging."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.custom_env import CodetyAILearningEnv  # noqa: E402


def make_env(seed: Optional[int] = None) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return CodetyAILearningEnv(render_mode=None)

    return _init


def append_result_row(path: Path, row: dict[str, Any]) -> None:
    """Append one CSV row; unions columns across algorithms so headers stay consistent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                rows.append(dict(old))
    rows.append(dict(row))
    all_keys: list[str] = []
    seen: set[str] = set()
    for rr in rows:
        for k in rr:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for rr in rows:
            w.writerow({k: rr.get(k, "") for k in all_keys})


def evaluate_policy(
    env: CodetyAILearningEnv,
    predict_fn: Callable[[np.ndarray], int],
    n_episodes: int = 15,
    seed_base: int = 0,
) -> dict[str, float]:
    rewards: list[float] = []
    lengths: list[int] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        trunc = False
        total = 0.0
        steps = 0
        while not (done or trunc):
            a = predict_fn(obs)
            obs, r, done, trunc, _ = env.step(a)
            total += float(r)
            steps += 1
        rewards.append(total)
        lengths.append(steps)
    return {
        "eval_mean_return": float(np.mean(rewards)),
        "eval_std_return": float(np.std(rewards)),
        "eval_mean_len": float(np.mean(lengths)),
    }
