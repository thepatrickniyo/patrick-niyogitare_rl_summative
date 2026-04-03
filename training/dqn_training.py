#!/usr/bin/env python3
"""
Train DQN (Stable-Baselines3) on CodetyAILearningEnv (CodetyAI platform sim).

Example:
  PYTHONPATH=. python training/dqn_training.py --run_index 0 --timesteps 80000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from training.common import ROOT, CodetyAILearningEnv, append_result_row, evaluate_policy
from training.hyperparam_runs import DQN_RUNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_index", type=int, default=0, help="Index 0..9 into DQN_RUNS")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models_dir", type=str, default="models/dqn")
    parser.add_argument("--results_csv", type=str, default="results/training_log.csv")
    args = parser.parse_args()

    hp = DQN_RUNS[args.run_index % len(DQN_RUNS)]
    out_dir = ROOT / args.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"run_{args.run_index}.zip"

    def make_m() -> Monitor:
        e = CodetyAILearningEnv(render_mode=None)
        return Monitor(e)

    train_env = DummyVecEnv([make_m])
    eval_env = CodetyAILearningEnv(render_mode=None)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=hp["learning_rate"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        gamma=hp["gamma"],
        train_freq=hp["train_freq"],
        target_update_interval=hp["target_update_interval"],
        exploration_fraction=hp["exploration_fraction"],
        exploration_final_eps=hp["exploration_final_eps"],
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(ROOT / "results" / "tb" / "dqn"),
    )

    eval_cb = EvalCallback(
        DummyVecEnv([lambda: Monitor(CodetyAILearningEnv(render_mode=None))]),
        best_model_save_path=str(out_dir / f"best_run_{args.run_index}"),
        log_path=str(ROOT / "results" / "eval" / "dqn"),
        eval_freq=max(args.timesteps // 10, 1000),
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(model_path))

    metrics = evaluate_policy(
        eval_env,
        lambda obs: int(model.predict(obs, deterministic=True)[0]),
        n_episodes=20,
        seed_base=args.seed,
    )
    row = {
        "algo": "dqn",
        "run_index": args.run_index,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "model_path": str(model_path.relative_to(ROOT)),
        **{f"hp_{k}": v for k, v in hp.items()},
        **metrics,
    }
    append_result_row(ROOT / args.results_csv, row)

    summary = {
        "algo": "dqn",
        "run_index": args.run_index,
        "eval_mean_return": metrics["eval_mean_return"],
        "model_path": str(model_path.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
