#!/usr/bin/env python3
"""
Policy-gradient family on the same custom environment:
  - ppo   : Proximal Policy Optimization (SB3)
  - a2c   : Advantage Actor-Critic (SB3)
  - reinforce : Monte Carlo REINFORCE (custom PyTorch)

Example:
  PYTHONPATH=. python training/pg_training.py --algo ppo --run_index 3 --timesteps 100000
  PYTHONPATH=. python training/pg_training.py --algo reinforce --run_index 0 --episodes 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from training.common import ROOT, CodetyAILearningEnv, append_result_row, evaluate_policy
from training.hyperparam_runs import A2C_RUNS, PPO_RUNS, REINFORCE_RUNS
from training.reinforce_trainer import ReinforceConfig, ReinforceTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce"], required=True)
    parser.add_argument("--run_index", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=120_000, help="SB3 algos only")
    parser.add_argument("--episodes", type=int, default=2500, help="REINFORCE only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models_dir", type=str, default="models/pg")
    parser.add_argument("--results_csv", type=str, default="results/training_log.csv")
    args = parser.parse_args()

    out_base = ROOT / args.models_dir / args.algo
    out_base.mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        hp = PPO_RUNS[args.run_index % len(PPO_RUNS)]

        def make_m() -> Monitor:
            return Monitor(CodetyAILearningEnv(render_mode=None))

        train_env = DummyVecEnv([make_m])
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=hp["n_epochs"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            clip_range=hp["clip_range"],
            ent_coef=hp["ent_coef"],
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(ROOT / "results" / "tb" / "ppo"),
        )
        eval_cb = EvalCallback(
            DummyVecEnv([lambda: Monitor(CodetyAILearningEnv(render_mode=None))]),
            best_model_save_path=str(out_base / f"best_run_{args.run_index}"),
            log_path=str(ROOT / "results" / "eval" / "ppo"),
            eval_freq=max(args.timesteps // 10, 1000),
            deterministic=True,
            render=False,
        )
        model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
        path = out_base / f"run_{args.run_index}.zip"
        model.save(str(path))
        eval_env = CodetyAILearningEnv(render_mode=None)
        metrics = evaluate_policy(
            eval_env,
            lambda obs: int(model.predict(obs, deterministic=True)[0]),
            n_episodes=20,
            seed_base=args.seed,
        )
        row = {
            "algo": "ppo",
            "run_index": args.run_index,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "model_path": str(path.relative_to(ROOT)),
            **{f"hp_{k}": v for k, v in hp.items()},
            **metrics,
        }

    elif args.algo == "a2c":
        hp = A2C_RUNS[args.run_index % len(A2C_RUNS)]

        def make_m2() -> Monitor:
            return Monitor(CodetyAILearningEnv(render_mode=None))

        train_env = DummyVecEnv([make_m2])
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(ROOT / "results" / "tb" / "a2c"),
        )
        eval_cb = EvalCallback(
            DummyVecEnv([lambda: Monitor(CodetyAILearningEnv(render_mode=None))]),
            best_model_save_path=str(out_base / f"best_run_{args.run_index}"),
            log_path=str(ROOT / "results" / "eval" / "a2c"),
            eval_freq=max(args.timesteps // 10, 1000),
            deterministic=True,
            render=False,
        )
        model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
        path = out_base / f"run_{args.run_index}.zip"
        model.save(str(path))
        eval_env = CodetyAILearningEnv(render_mode=None)
        metrics = evaluate_policy(
            eval_env,
            lambda obs: int(model.predict(obs, deterministic=True)[0]),
            n_episodes=20,
            seed_base=args.seed,
        )
        row = {
            "algo": "a2c",
            "run_index": args.run_index,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "model_path": str(path.relative_to(ROOT)),
            **{f"hp_{k}": v for k, v in hp.items()},
            **metrics,
        }

    else:
        hp = REINFORCE_RUNS[args.run_index % len(REINFORCE_RUNS)]
        env = CodetyAILearningEnv(render_mode=None)
        cfg = ReinforceConfig(
            lr=hp["lr"],
            gamma=hp["gamma"],
            hidden=hp["hidden"],
            entropy_coef=hp["entropy_coef"],
            max_grad_norm=hp["max_grad_norm"],
        )
        trainer = ReinforceTrainer(env, cfg, seed=args.seed)
        trainer.train(total_episodes=args.episodes, log_every=max(args.episodes // 10, 1))
        path = out_base / f"run_{args.run_index}.pt"
        torch.save(trainer.policy.state_dict(), path)
        metrics = evaluate_policy(
            env,
            trainer.predict,
            n_episodes=20,
            seed_base=args.seed,
        )
        row = {
            "algo": "reinforce",
            "run_index": args.run_index,
            "timesteps": args.episodes,
            "seed": args.seed,
            "model_path": str(path.relative_to(ROOT)),
            **{f"hp_{k}": v for k, v in hp.items()},
            **metrics,
        }

    append_result_row(ROOT / args.results_csv, row)
    print(json.dumps({"algo": row["algo"], "run_index": args.run_index, **metrics}, indent=2))


if __name__ == "__main__":
    main()
