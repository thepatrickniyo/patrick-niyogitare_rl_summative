"""
Ten distinct hyperparameter combinations per algorithm for summative tuning.

Template-aligned knobs (learning rate, batch / buffer, gamma, exploration,
entropy / clip, n_steps, etc.) — each row is one training run index 0..9.
"""

from __future__ import annotations

# --- DQN (value-based) ---
DQN_RUNS: list[dict] = [
    {"learning_rate": 1e-4, "buffer_size": 50_000, "batch_size": 32, "gamma": 0.99, "train_freq": 4, "target_update_interval": 1_000, "exploration_fraction": 0.2, "exploration_final_eps": 0.05},
    {"learning_rate": 3e-4, "buffer_size": 100_000, "batch_size": 64, "gamma": 0.99, "train_freq": 4, "target_update_interval": 2_000, "exploration_fraction": 0.15, "exploration_final_eps": 0.02},
    {"learning_rate": 5e-4, "buffer_size": 80_000, "batch_size": 128, "gamma": 0.995, "train_freq": 2, "target_update_interval": 500, "exploration_fraction": 0.25, "exploration_final_eps": 0.08},
    {"learning_rate": 2.5e-4, "buffer_size": 200_000, "batch_size": 256, "gamma": 0.99, "train_freq": 8, "target_update_interval": 4_000, "exploration_fraction": 0.1, "exploration_final_eps": 0.01},
    {"learning_rate": 1e-3, "buffer_size": 40_000, "batch_size": 32, "gamma": 0.97, "train_freq": 1, "target_update_interval": 500, "exploration_fraction": 0.35, "exploration_final_eps": 0.1},
    {"learning_rate": 6e-5, "buffer_size": 150_000, "batch_size": 64, "gamma": 0.999, "train_freq": 4, "target_update_interval": 3_000, "exploration_fraction": 0.18, "exploration_final_eps": 0.04},
    {"learning_rate": 4e-4, "buffer_size": 60_000, "batch_size": 48, "gamma": 0.98, "train_freq": 4, "target_update_interval": 1_500, "exploration_fraction": 0.22, "exploration_final_eps": 0.06},
    {"learning_rate": 2e-4, "buffer_size": 120_000, "batch_size": 96, "gamma": 0.99, "train_freq": 6, "target_update_interval": 2_500, "exploration_fraction": 0.12, "exploration_final_eps": 0.03},
    {"learning_rate": 7e-4, "buffer_size": 90_000, "batch_size": 128, "gamma": 0.985, "train_freq": 3, "target_update_interval": 800, "exploration_fraction": 0.28, "exploration_final_eps": 0.07},
    {"learning_rate": 1.5e-4, "buffer_size": 250_000, "batch_size": 64, "gamma": 0.993, "train_freq": 4, "target_update_interval": 5_000, "exploration_fraction": 0.08, "exploration_final_eps": 0.015},
]

# --- PPO (proximal policy optimization) ---
PPO_RUNS: list[dict] = [
    {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 128, "n_epochs": 8, "gamma": 0.995, "gae_lambda": 0.92, "clip_range": 0.15, "ent_coef": 0.005},
    {"learning_rate": 5e-4, "n_steps": 256, "batch_size": 32, "n_epochs": 15, "gamma": 0.99, "gae_lambda": 0.98, "clip_range": 0.25, "ent_coef": 0.01},
    {"learning_rate": 2.5e-4, "n_steps": 2048, "batch_size": 256, "n_epochs": 6, "gamma": 0.997, "gae_lambda": 0.9, "clip_range": 0.1, "ent_coef": 0.0},
    {"learning_rate": 8e-4, "n_steps": 128, "batch_size": 16, "n_epochs": 20, "gamma": 0.98, "gae_lambda": 0.97, "clip_range": 0.3, "ent_coef": 0.02},
    {"learning_rate": 4e-4, "n_steps": 768, "batch_size": 96, "n_epochs": 12, "gamma": 0.99, "gae_lambda": 0.94, "clip_range": 0.18, "ent_coef": 0.002},
    {"learning_rate": 6e-5, "n_steps": 1536, "batch_size": 192, "n_epochs": 5, "gamma": 0.999, "gae_lambda": 0.96, "clip_range": 0.12, "ent_coef": 0.0},
    {"learning_rate": 1e-3, "n_steps": 384, "batch_size": 48, "n_epochs": 8, "gamma": 0.97, "gae_lambda": 0.99, "clip_range": 0.22, "ent_coef": 0.015},
    {"learning_rate": 2e-4, "n_steps": 640, "batch_size": 80, "n_epochs": 10, "gamma": 0.992, "gae_lambda": 0.93, "clip_range": 0.16, "ent_coef": 0.008},
    {"learning_rate": 3.5e-4, "n_steps": 896, "batch_size": 112, "n_epochs": 9, "gamma": 0.988, "gae_lambda": 0.955, "clip_range": 0.19, "ent_coef": 0.003},
]

# --- A2C (actor-critic, synchronous advantage) ---
A2C_RUNS: list[dict] = [
    {"learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "n_steps": 8, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.0, "vf_coef": 0.25, "max_grad_norm": 1.0},
    {"learning_rate": 1e-3, "n_steps": 4, "gamma": 0.98, "gae_lambda": 1.0, "ent_coef": 0.02, "vf_coef": 0.5, "max_grad_norm": 0.3},
    {"learning_rate": 5e-5, "n_steps": 16, "gamma": 0.999, "gae_lambda": 0.92, "ent_coef": 0.005, "vf_coef": 0.4, "max_grad_norm": 2.0},
    {"learning_rate": 2e-4, "n_steps": 10, "gamma": 0.99, "gae_lambda": 0.98, "ent_coef": 0.015, "vf_coef": 0.35, "max_grad_norm": 0.7},
    {"learning_rate": 4.5e-4, "n_steps": 6, "gamma": 0.993, "gae_lambda": 0.97, "ent_coef": 0.008, "vf_coef": 0.45, "max_grad_norm": 0.6},
    {"learning_rate": 9e-4, "n_steps": 3, "gamma": 0.985, "gae_lambda": 1.0, "ent_coef": 0.025, "vf_coef": 0.55, "max_grad_norm": 0.4},
    {"learning_rate": 1.2e-4, "n_steps": 20, "gamma": 0.997, "gae_lambda": 0.94, "ent_coef": 0.0, "vf_coef": 0.3, "max_grad_norm": 1.5},
    {"learning_rate": 6e-4, "n_steps": 7, "gamma": 0.991, "gae_lambda": 0.96, "ent_coef": 0.012, "vf_coef": 0.38, "max_grad_norm": 0.55},
    {"learning_rate": 2.5e-4, "n_steps": 12, "gamma": 0.989, "gae_lambda": 0.93, "ent_coef": 0.004, "vf_coef": 0.42, "max_grad_norm": 0.85},
]

# --- REINFORCE (Monte Carlo policy gradient; custom trainer) ---
REINFORCE_RUNS: list[dict] = [
    {"lr": 1e-3, "gamma": 0.99, "hidden": 128, "entropy_coef": 0.0, "max_grad_norm": 1.0},
    {"lr": 3e-4, "gamma": 0.995, "hidden": 256, "entropy_coef": 0.01, "max_grad_norm": 0.5},
    {"lr": 5e-4, "gamma": 0.98, "hidden": 64, "entropy_coef": 0.005, "max_grad_norm": 2.0},
    {"lr": 1e-4, "gamma": 0.999, "hidden": 192, "entropy_coef": 0.0, "max_grad_norm": 1.5},
    {"lr": 7e-4, "gamma": 0.99, "hidden": 160, "entropy_coef": 0.02, "max_grad_norm": 0.3},
    {"lr": 2e-3, "gamma": 0.97, "hidden": 96, "entropy_coef": 0.015, "max_grad_norm": 0.8},
    {"lr": 4.5e-4, "gamma": 0.992, "hidden": 200, "entropy_coef": 0.003, "max_grad_norm": 1.2},
    {"lr": 8e-5, "gamma": 0.996, "hidden": 128, "entropy_coef": 0.0, "max_grad_norm": 2.5},
    {"lr": 6e-4, "gamma": 0.988, "hidden": 144, "entropy_coef": 0.008, "max_grad_norm": 0.65},
    {"lr": 1.5e-3, "gamma": 0.985, "hidden": 112, "entropy_coef": 0.025, "max_grad_norm": 0.45},
]
