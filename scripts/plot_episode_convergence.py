#!/usr/bin/env python3
"""
Record per-episode training returns and plot convergence (episodes vs return).

Uses best hyperparameter rows from the sweep: DQN run 4, PPO run 2, A2C run 2,
REINFORCE run 9. Writes report/figures/fig4_episode_convergence.png and
report/convergence_metrics.json.

  PYTHONPATH=. python scripts/plot_episode_convergence.py
  PYTHONPATH=. python scripts/plot_episode_convergence.py --quick   # shorter run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import CodetyAILearningEnv
from training.hyperparam_runs import A2C_RUNS, DQN_RUNS, PPO_RUNS, REINFORCE_RUNS
from training.reinforce_trainer import ReinforceConfig, ReinforceTrainer


class EpisodeRewardCallback(BaseCallback):
    """Collect Monitor 'episode' info from VecEnv after each env step."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos") or []:
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
        return True


def trailing_ma(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(w - 1, len(x)):
        out[i] = float(x[i - w + 1 : i + 1].mean())
    return out


def metrics(
    rewards: list[float],
    smooth_w: int = 15,
    plateau_tail: float = 0.2,
) -> dict:
    r = np.asarray(rewards, dtype=float)
    n = len(r)
    if n == 0:
        return {"n_episodes": 0}
    ma = trailing_ma(r, smooth_w)
    finite = ma[~np.isnan(ma)]
    max_ma = float(np.max(finite)) if len(finite) else float(np.nan)
    ep_90 = None
    for i in range(len(ma)):
        if not np.isnan(ma[i]) and max_ma > 0 and ma[i] >= 0.9 * max_ma:
            ep_90 = int(i + 1)
            break
    tail_n = max(3, int(np.ceil(plateau_tail * n)))
    plateau_mean = float(np.mean(r[-tail_n:]))
    plateau_std = float(np.std(r[-tail_n:]))
    cv = plateau_std / (abs(plateau_mean) + 1e-8)
    return {
        "n_episodes": int(n),
        "smooth_window": smooth_w,
        "max_trailing_mean_return": max_ma,
        "episodes_to_90pct_of_max_ma": ep_90,
        "tail_mean_return": plateau_mean,
        "tail_std_return": plateau_std,
        "tail_coefficient_of_variation": round(cv, 4),
        "mean_return_all_episodes": float(np.mean(r)),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Shorter training for smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timesteps = 22_000 if args.quick else 55_000
    rf_eps = 120 if args.quick else 320

    seed = args.seed
    out_fig = ROOT / "report" / "figures" / "fig4_episode_convergence.png"
    out_json = ROOT / "report" / "convergence_metrics.json"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    smooth_w = 12
    series: dict[str, list[float]] = {}

    # --- SB3 algorithms ---
    def make_mon() -> Monitor:
        return Monitor(CodetyAILearningEnv(render_mode=None))

    # DQN best row 4
    hp_d = DQN_RUNS[4]
    cb_d = EpisodeRewardCallback()
    env_d = DummyVecEnv([make_mon])
    model_d = DQN(
        "MlpPolicy",
        env_d,
        learning_rate=hp_d["learning_rate"],
        buffer_size=min(hp_d["buffer_size"], 50_000),
        batch_size=hp_d["batch_size"],
        gamma=hp_d["gamma"],
        train_freq=hp_d["train_freq"],
        target_update_interval=hp_d["target_update_interval"],
        exploration_fraction=hp_d["exploration_fraction"],
        exploration_final_eps=hp_d["exploration_final_eps"],
        verbose=0,
        seed=seed,
    )
    model_d.learn(total_timesteps=timesteps, callback=cb_d, progress_bar=False)
    series["DQN"] = list(cb_d.episode_rewards)
    env_d.close()

    # PPO best row 2
    hp_p = PPO_RUNS[2]
    cb_p = EpisodeRewardCallback()
    env_p = DummyVecEnv([make_mon])
    model_p = PPO(
        "MlpPolicy",
        env_p,
        learning_rate=hp_p["learning_rate"],
        n_steps=hp_p["n_steps"],
        batch_size=hp_p["batch_size"],
        n_epochs=hp_p["n_epochs"],
        gamma=hp_p["gamma"],
        gae_lambda=hp_p["gae_lambda"],
        clip_range=hp_p["clip_range"],
        ent_coef=hp_p["ent_coef"],
        verbose=0,
        seed=seed,
    )
    model_p.learn(total_timesteps=timesteps, callback=cb_p, progress_bar=False)
    series["PPO"] = list(cb_p.episode_rewards)
    env_p.close()

    # A2C best row 2
    hp_a = A2C_RUNS[2]
    cb_a = EpisodeRewardCallback()
    env_a = DummyVecEnv([make_mon])
    model_a = A2C(
        "MlpPolicy",
        env_a,
        learning_rate=hp_a["learning_rate"],
        n_steps=hp_a["n_steps"],
        gamma=hp_a["gamma"],
        gae_lambda=hp_a["gae_lambda"],
        ent_coef=hp_a["ent_coef"],
        vf_coef=hp_a["vf_coef"],
        max_grad_norm=hp_a["max_grad_norm"],
        verbose=0,
        seed=seed,
    )
    model_a.learn(total_timesteps=timesteps, callback=cb_a, progress_bar=False)
    series["A2C"] = list(cb_a.episode_rewards)
    env_a.close()

    # REINFORCE run 9 hyperparams
    hp_r = REINFORCE_RUNS[9]
    cfg = ReinforceConfig(
        lr=hp_r["lr"],
        gamma=hp_r["gamma"],
        hidden=hp_r["hidden"],
        entropy_coef=hp_r["entropy_coef"],
        max_grad_norm=hp_r["max_grad_norm"],
    )
    env_r = CodetyAILearningEnv(render_mode=None)
    trainer = ReinforceTrainer(env_r, cfg, seed=seed)
    hist = trainer.train(total_episodes=rf_eps, log_every=max(rf_eps, 1))
    series["REINFORCE"] = [float(x) for x in hist]
    env_r.close()

    # --- Plot 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    colors = {"DQN": "#2563eb", "PPO": "#ea580c", "A2C": "#9333ea", "REINFORCE": "#16a34a"}
    order = ["DQN", "PPO", "A2C", "REINFORCE"]
    all_m: dict[str, dict] = {}

    for ax, name in zip(axes.flat, order):
        rew = series.get(name, [])
        all_m[name] = metrics(rew, smooth_w=smooth_w)
        all_m[name]["training_budget"] = (
            f"{timesteps} env steps (SB3)" if name != "REINFORCE" else f"{rf_eps} training episodes"
        )
        if not rew:
            ax.set_title(f"{name} (no episodes logged)")
            continue
        x = np.arange(1, len(rew) + 1)
        ax.plot(x, rew, alpha=0.35, color=colors[name], label="Episode return")
        ma = trailing_ma(np.array(rew), smooth_w)
        ax.plot(x, ma, color=colors[name], linewidth=2, label=f"{smooth_w}-ep trailing mean")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Return")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Training convergence (seed={seed}); SB3 budget={timesteps} steps; "
        f"REINFORCE={rf_eps} episodes",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    out_json.write_text(json.dumps(all_m, indent=2))
    print("Wrote", out_fig.relative_to(ROOT))
    print("Wrote", out_json.relative_to(ROOT))
    for k, v in all_m.items():
        print(k, v)


if __name__ == "__main__":
    main()
