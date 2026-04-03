#!/usr/bin/env python3
"""
Record a short rollout with a uniformly random policy (no trained model).
Writes a static GIF under static/ for assignment submission / reports.

  PYTHONPATH=. python demos/record_random_rollout.py --steps 250 --out static/random_agent_rollout.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from environment.custom_env import CodetyAILearningEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--out", type=str, default="static/random_agent_rollout.gif")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Include on-screen objective + reward legend (same as main.py --demo)",
    )
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = CodetyAILearningEnv(render_mode="rgb_array", demo_overlay=args.demo)
    env.reset(seed=args.seed)
    frames: list[np.ndarray] = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    rng = np.random.default_rng(args.seed)
    for _ in range(args.steps):
        a = int(rng.integers(0, env.action_space.n))
        obs, _, done, trunc, _ = env.step(a)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if done or trunc:
            obs, _ = env.reset()
        if len(frames) >= args.steps + 1:
            break

    env.close()
    if not frames:
        raise SystemExit("No frames captured; check pygame / rgb_array rendering.")
    imageio.mimsave(str(out_path), frames, fps=8, loop=0)
    print(f"Saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
