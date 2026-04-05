"""Vanilla REINFORCE with optional entropy bonus (not provided by SB3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment.custom_env import CodetyAILearningEnv


class PolicyNet(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_act),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReinforceConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    hidden: int = 128
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0


class ReinforceTrainer:
    def __init__(self, env: CodetyAILearningEnv, cfg: ReinforceConfig, seed: int = 0):
        self.env = env
        self.cfg = cfg
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_obs = int(np.prod(env.observation_space.shape))
        n_act = int(env.action_space.n)
        self.policy = PolicyNet(n_obs, n_act, hidden=cfg.hidden)
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def _episode(self) -> Tuple[float, int]:
        obs, _ = self.env.reset()
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []
        done = trunc = False
        steps = 0
        while not (done or trunc):
            x = torch.as_tensor(np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0), dtype=torch.float32).unsqueeze(0)
            logits = torch.clamp(self.policy(x), -40.0, 40.0)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, r, done, trunc, _ = self.env.step(int(action.item()))
            rewards.append(float(r))
            steps += 1
        if not log_probs:
            return 0.0, 0
        returns: list[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) < 2:
            returns_t = returns_t * 0.0
        else:
            rs_std = returns_t.std()
            if rs_std < 1e-5:
                returns_t = returns_t - returns_t.mean()
            else:
                returns_t = (returns_t - returns_t.mean()) / (rs_std + 1e-8)
        pol_loss = torch.stack([-lp * Gt for lp, Gt in zip(log_probs, returns_t)]).sum()
        ent_loss = torch.stack(entropies).sum()
        loss = pol_loss - self.cfg.entropy_coef * ent_loss
        if not torch.isfinite(loss):
            return float(sum(rewards)), steps
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return float(sum(rewards)), steps

    def train(self, total_episodes: int, log_every: int = 50) -> List[float]:
        history: List[float] = []
        for ep in range(total_episodes):
            ret, _ = self._episode()
            history.append(ret)
            if (ep + 1) % log_every == 0:
                print(f"REINFORCE ep {ep + 1}  mean_last_{log_every}={np.mean(history[-log_every:]):.3f}")
        return history

    def predict(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = torch.clamp(self.policy(x), -40.0, 40.0)
            return int(torch.argmax(logits, dim=-1).item())
