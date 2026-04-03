#!/usr/bin/env python3
"""
Run the best-performing trained agent with optional Pygame GUI and verbose logging.

Usage:
  PYTHONPATH=. python main.py --algo dqn --model-path models/dqn/run_2.zip
  PYTHONPATH=. python main.py --config results/best_models.json --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.custom_env import CodetyAILearningEnv  # noqa: E402


def load_sb3_model(path: Path, algo: str):
    from stable_baselines3 import A2C, DQN, PPO

    algo = algo.lower()
    if algo == "dqn":
        return DQN.load(path)
    if algo == "ppo":
        return PPO.load(path)
    if algo == "a2c":
        return A2C.load(path)
    raise ValueError(f"Unknown SB3 algo: {algo}")


def run_episode(
    env: CodetyAILearningEnv,
    predict,
    max_steps: int,
    verbose: bool,
) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = trunc = False
    while not (done or trunc) and steps < max_steps:
        action = predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if verbose:
            print(
                f"  step={steps:4d}  action={int(action)}  "
                f"reward={reward:8.2f}  skill={info['skill']:.1f}  "
                f"conf={info['confidence']:.1f}  eng={info['engagement']}  "
                f"projects={info['projects']}  mentors={info['mentorship_sessions']}"
            )
    return {
        "return": total_reward,
        "steps": steps,
        "projects": info.get("projects", 0),
        "skill": info.get("skill", 0.0),
        "job_ready": info.get("job_ready", False),
        "dropout": info.get("dropout", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c", "reinforce"], default="dqn")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--config", type=str, default="results/best_models.json")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--render", action="store_true", help="Pygame human window")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="REINFORCE MLP hidden size (must match training run if not using best_models.json)",
    )
    args = parser.parse_args()

    model_path = args.model_path
    algo = args.algo
    if model_path is None:
        cfg_path = ROOT / args.config
        if not cfg_path.exists():
            print(
                "No --model-path and no results/best_models.json. "
                "Train first, then create best_models.json (see README). "
                "Falling back to random policy for smoke test.",
                file=sys.stderr,
            )
            algo = "random"
        else:
            cfg = json.loads(cfg_path.read_text())
            entry = cfg.get("best", {}).get(algo)
            if entry is None:
                print(f"No entry for algo={algo} in config; using random.", file=sys.stderr)
                algo = "random"
            else:
                model_path = entry["path"]
                algo = entry.get("algo", algo)

    render_mode = "human" if args.render else None
    env = CodetyAILearningEnv(render_mode=render_mode)

    if algo == "random":

        def predict(obs):
            return int(env.action_space.sample())

    elif algo == "reinforce":
        import torch

        from training.reinforce_trainer import PolicyNet

        if model_path is None:
            raise SystemExit("reinforce requires --model-path")
        path = ROOT / model_path
        obs_dim = int(np.prod(env.observation_space.shape))
        n_act = int(env.action_space.n)
        hidden = int(args.hidden) if args.hidden is not None else 128
        cfg_path = ROOT / args.config
        if cfg_path.exists() and args.hidden is None:
            cfg = json.loads(cfg_path.read_text())
            ent = cfg.get("best", {}).get("reinforce", {})
            hidden = int(ent.get("hidden", hidden))
        net = PolicyNet(obs_dim, n_act, hidden=hidden)
        state = torch.load(path, map_location="cpu")
        net.load_state_dict(state)
        net.eval()

        def predict(obs):
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return int(torch.argmax(net(x), dim=-1).item())

    else:
        model = load_sb3_model(ROOT / model_path, algo)

        def predict(obs):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)

    print("=== CodetyAI — Adaptive learning & mentorship (evaluation) ===")
    print(f"Algorithm: {algo}  model: {model_path}")
    print(
        "Objective: maximize student success — skill, confidence, projects, "
        "mentorship — and reach job-ready state (STEM employability)."
    )
    print(
        "Reward events: +10 project done, +15 strong skill gain (≥3/step), +20 job-ready, "
        "−10 engagement drop, −20 dropout (see environment/custom_env.py)."
    )

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1} / {args.episodes} (seed={args.seed + ep}) ---")
        env.reset(seed=args.seed + ep)
        t0 = time.time()
        stats = run_episode(
            env,
            predict,
            max_steps=env.MAX_EPISODE_STEPS,
            verbose=args.verbose,
        )
        dt = time.time() - t0
        print(
            f"Summary: return={stats['return']:.3f}  steps={stats['steps']}  "
            f"skill_end={stats['skill']:.1f}  projects={stats['projects']}  "
            f"job_ready={stats['job_ready']}  dropout={stats['dropout']}  "
            f"wall_time={dt:.2f}s"
        )

    env.close()


if __name__ == "__main__":
    main()
