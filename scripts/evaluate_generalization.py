#!/usr/bin/env python3
"""
Evaluate best policies on held-out reset seeds (initial student profiles).

Compares episode returns on:
  - "Near-train" seeds (0–49): small IDs often used in debugging / nearby regime
  - "Far held-out" seeds (10_000–10_049): disjoint from typical training seed 42

Writes report/figures/fig5_generalization.png and report/generalization_metrics.json

  PYTHONPATH=. python scripts/evaluate_generalization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import CodetyAILearningEnv


def episode_return(env: CodetyAILearningEnv, predict, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    total = 0.0
    while not (done or trunc):
        a = predict(obs)
        obs, r, done, trunc, _ = env.step(a)
        total += float(r)
    return total


def collect_returns(
    env: CodetyAILearningEnv,
    predict: Callable[[np.ndarray], int],
    seeds: list[int],
) -> np.ndarray:
    out = []
    for s in seeds:
        out.append(episode_return(env, predict, s))
    return np.array(out, dtype=float)


def load_predictors():
    """Best checkpoints from hyperparameter sweep (aligns with report tables)."""
    import torch

    from training.reinforce_trainer import PolicyNet

    preds = {}

    dqn = DQN.load(ROOT / "models/dqn/run_4.zip")

    def pred_dqn(obs):
        a, _ = dqn.predict(obs, deterministic=True)
        return int(a)

    preds["DQN"] = pred_dqn

    ppo = PPO.load(ROOT / "models/pg/ppo/run_2.zip")

    def pred_ppo(obs):
        a, _ = ppo.predict(obs, deterministic=True)
        return int(a)

    preds["PPO"] = pred_ppo

    a2c = A2C.load(ROOT / "models/pg/a2c/run_2.zip")

    def pred_a2c(obs):
        a, _ = a2c.predict(obs, deterministic=True)
        return int(a)

    preds["A2C"] = pred_a2c

    # REINFORCE run 9 (best in sweep); best_models.json may still point to run_0
    path_r = ROOT / "models/pg/reinforce/run_9.pt"
    if not path_r.exists():
        path_r = ROOT / "models/pg/reinforce/run_0.pt"
    obs_dim = 6
    n_act = 5
    hidden = 112  # run_9 hyperparam
    if path_r.name.startswith("run_0"):
        hidden = 128
    net = PolicyNet(obs_dim, n_act, hidden=hidden)
    state = torch.load(path_r, map_location="cpu")
    net.load_state_dict(state)
    net.eval()

    def pred_rf(obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = torch.clamp(net(x), -40.0, 40.0)
            return int(torch.argmax(logits, dim=-1).item())

    preds["REINFORCE"] = pred_rf

    return preds


def main() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    near_seeds = list(range(50))
    far_seeds = list(range(10_000, 10_050))

    preds = load_predictors()
    env = CodetyAILearningEnv(render_mode=None)

    results: dict[str, dict] = {}
    series_near: dict[str, np.ndarray] = {}
    series_far: dict[str, np.ndarray] = {}

    for name, pred in preds.items():
        r_near = collect_returns(env, pred, near_seeds)
        r_far = collect_returns(env, pred, far_seeds)
        series_near[name] = r_near
        series_far[name] = r_far
        results[name] = {
            "near_seeds_range": f"{near_seeds[0]}–{near_seeds[-1]}",
            "far_seeds_range": f"{far_seeds[0]}–{far_seeds[-1]}",
            "near_mean_return": float(np.mean(r_near)),
            "near_std_return": float(np.std(r_near)),
            "far_mean_return": float(np.mean(r_far)),
            "far_std_return": float(np.std(r_far)),
            "generalization_gap": float(np.mean(r_far) - np.mean(r_near)),
            "pct_change_far_vs_near": float(
                100.0 * (np.mean(r_far) - np.mean(r_near)) / (abs(np.mean(r_near)) + 1e-8)
            ),
        }

    env.close()

    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = ROOT / "report" / "generalization_metrics.json"

    # --- Figure: grouped bars mean ± std + overlaid strip of per-seed returns ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    names = list(preds.keys())
    x = np.arange(len(names))
    w = 0.35

    for ax, label, data_dict in (
        (
            axes[0],
            f"Near-train band\n(seeds {near_seeds[0]}–{near_seeds[-1]})",
            series_near,
        ),
        (
            axes[1],
            f"Held-out band\n(seeds {far_seeds[0]}–{far_seeds[-1]})",
            series_far,
        ),
    ):
        means = [float(np.mean(data_dict[n])) for n in names]
        stds = [float(np.std(data_dict[n])) for n in names]
        ax.bar(x, means, yerr=stds, capsize=4, color=["#2563eb", "#ea580c", "#9333ea", "#16a34a"], alpha=0.85)
        for i, n in enumerate(names):
            ys = data_dict[n]
            jitter = (rng.random(len(ys)) - 0.5) * 0.15
            ax.scatter(np.full(len(ys), i) + jitter, ys, s=8, alpha=0.35, color="black", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Episode return (1 ep. per seed)")
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Generalization: same policies, different reset seeds (initial student profiles)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_generalization.png", dpi=150)
    plt.close(fig)

    out_json.write_text(json.dumps(results, indent=2))
    print("Wrote", (out_dir / "fig5_generalization.png").relative_to(ROOT))
    print("Wrote", out_json.relative_to(ROOT))
    for k, v in results.items():
        print(k, "gap", round(v["generalization_gap"], 2), "pct", round(v["pct_change_far_vs_near"], 2))


if __name__ == "__main__":
    main()
