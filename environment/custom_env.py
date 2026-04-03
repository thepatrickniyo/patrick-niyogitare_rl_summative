"""
Adaptive AI Learning & Mentorship Optimization (CodetyAI) — custom Gymnasium environment.

Simulates one student on the CodetyAI platform. The RL agent schedules lessons,
projects, mentor sessions, and revision to improve employability (STEM youth in Rwanda).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CodetyAction(IntEnum):
    """Platform decisions the policy controls each step."""

    BEGINNER_LESSON = 0
    ADVANCED_LESSON = 1
    HANDS_ON_PROJECT = 2
    CONNECT_MENTOR = 3
    RECOMMEND_REVISION = 4


class CodetyAILearningEnv(gym.Env):
    """
    Observation (normalized, float32):
      skill / 100, engagement / 2, projects / cap, mentorship / cap,
      confidence / 100, program progress (step / max_steps)

    Engagement: 0 = low, 1 = medium, 2 = high (stored as int, observed normalized).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    MAX_PROJECTS_OBS = 6
    MAX_MENTOR_OBS = 10
    MAX_EPISODE_STEPS = 200

    JOB_READY_SKILL = 75.0
    JOB_READY_CONFIDENCE = 70.0
    MIN_PROJECTS_JOB_READY = 2

    R_PROJECT_COMPLETE = 10.0
    R_SKILL_IMPROVE = 15.0
    R_JOB_READY = 20.0
    R_ENGAGEMENT_DROP = -10.0
    R_DROPOUT = -20.0

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        demo_overlay: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.demo_overlay = demo_overlay
        self._renderer: Optional[Any] = None

        self.action_space = spaces.Discrete(len(CodetyAction))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._skill = 25.0
        self._confidence = 35.0
        self._engagement = 1  # 0 low, 1 med, 2 high
        self._projects = 0
        self._mentorship_sessions = 0
        self._step_count = 0
        self._low_engagement_streak = 0
        self._last_action = 0
        self._skill_history: list[float] = [25.0]
        self._episode_return = 0.0
        self._terminal_status = "active"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._skill = float(self._rng.uniform(12.0, 32.0))
        self._confidence = float(self._rng.uniform(18.0, 42.0))
        self._engagement = int(self._rng.integers(0, 2))  # start low or medium
        self._projects = int(self._rng.integers(0, 2))
        self._mentorship_sessions = 0
        self._step_count = 0
        self._low_engagement_streak = 0
        self._last_action = -1
        self._skill_history = [self._skill]
        self._episode_return = 0.0
        self._terminal_status = "active"

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._skill / 100.0,
                self._engagement / 2.0,
                min(self._projects / self.MAX_PROJECTS_OBS, 1.0),
                min(self._mentorship_sessions / self.MAX_MENTOR_OBS, 1.0),
                self._confidence / 100.0,
                min(self._step_count / float(self.MAX_EPISODE_STEPS), 1.0),
            ],
            dtype=np.float32,
        )

    def _is_job_ready(self) -> bool:
        return (
            self._skill >= self.JOB_READY_SKILL
            and self._confidence >= self.JOB_READY_CONFIDENCE
            and self._projects >= self.MIN_PROJECTS_JOB_READY
        )

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_count += 1
        a = int(action)
        self._last_action = a
        reward = 0.0
        skill_before = self._skill
        eng_before = self._engagement

        # --- Stochastic transitions driven by pedagogical action ---
        if a == CodetyAction.BEGINNER_LESSON:
            gain = float(self._rng.uniform(2.0, 7.0))
            self._skill = min(100.0, self._skill + gain)
            self._confidence = min(100.0, self._confidence + self._rng.uniform(1.0, 4.0))
            if self._rng.random() < 0.55:
                self._engagement = min(2, self._engagement + 1)

        elif a == CodetyAction.ADVANCED_LESSON:
            if self._skill >= 42.0:
                self._skill = min(100.0, self._skill + self._rng.uniform(5.0, 12.0))
                self._confidence = min(100.0, self._confidence + self._rng.uniform(2.0, 6.0))
                if self._rng.random() < 0.45:
                    self._engagement = min(2, self._engagement + 1)
            else:
                self._skill = min(100.0, self._skill + self._rng.uniform(0.5, 3.0))
                self._confidence = max(0.0, self._confidence - self._rng.uniform(2.0, 8.0))
                if self._rng.random() < 0.5:
                    self._engagement = max(0, self._engagement - 1)

        elif a == CodetyAction.HANDS_ON_PROJECT:
            if self._skill >= 28.0 and self._rng.random() < 0.72:
                self._projects += 1
                self._skill = min(100.0, self._skill + self._rng.uniform(4.0, 14.0))
                self._confidence = min(100.0, self._confidence + self._rng.uniform(3.0, 10.0))
                reward += self.R_PROJECT_COMPLETE
                if self._rng.random() < 0.6:
                    self._engagement = min(2, self._engagement + 1)
            else:
                self._confidence = max(0.0, self._confidence - self._rng.uniform(3.0, 9.0))
                self._engagement = max(0, self._engagement - 1)

        elif a == CodetyAction.CONNECT_MENTOR:
            self._mentorship_sessions += 1
            self._skill = min(100.0, self._skill + self._rng.uniform(2.0, 6.0))
            self._confidence = min(100.0, self._confidence + self._rng.uniform(5.0, 12.0))
            if self._rng.random() < 0.65:
                self._engagement = min(2, self._engagement + 1)

        else:  # RECOMMEND_REVISION
            self._skill = min(100.0, self._skill + self._rng.uniform(1.0, 4.0))
            self._confidence = min(100.0, self._confidence + self._rng.uniform(2.0, 5.0))
            self._engagement = min(2, self._engagement + 1)

        # Skill improvement reward (+15 if meaningful gain in one step)
        delta_skill = self._skill - skill_before
        if delta_skill >= 3.0:
            reward += self.R_SKILL_IMPROVE

        # Engagement drop penalty
        if self._engagement < eng_before:
            reward += self.R_ENGAGEMENT_DROP

        # Low engagement streak → dropout risk
        if self._engagement == 0:
            self._low_engagement_streak += 1
        else:
            self._low_engagement_streak = 0

        terminated = False
        truncated = self._step_count >= self.MAX_EPISODE_STEPS

        if self._is_job_ready():
            reward += self.R_JOB_READY
            terminated = True

        dropout = False
        if not terminated:
            if self._low_engagement_streak >= 4 or (
                self._engagement == 0 and self._confidence < 12.0
            ):
                reward += self.R_DROPOUT
                terminated = True
                dropout = True

        reward_f = float(reward)
        self._episode_return += reward_f
        if terminated:
            if dropout:
                self._terminal_status = "dropout"
            elif self._is_job_ready():
                self._terminal_status = "success"
            else:
                self._terminal_status = "timeout"
        elif truncated:
            self._terminal_status = "timeout"
        else:
            self._terminal_status = "active"

        self._skill_history.append(self._skill)
        if len(self._skill_history) > 80:
            self._skill_history = self._skill_history[-80:]

        info = {
            "skill": self._skill,
            "confidence": self._confidence,
            "engagement": self._engagement,
            "projects": self._projects,
            "mentorship_sessions": self._mentorship_sessions,
            "last_action": a,
            "job_ready": self._is_job_ready(),
            "dropout": dropout,
            "step": self._step_count,
            "episode_return": self._episode_return,
            "terminal_status": self._terminal_status,
        }
        return self._get_obs(), reward_f, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import CodetyRenderer

            self._renderer = CodetyRenderer(
                width=920,
                height=640,
                fps=self.metadata["render_fps"],
                demo_overlay=self.demo_overlay,
            )
        return self._renderer.render(
            skill=self._skill,
            confidence=self._confidence,
            engagement=self._engagement,
            projects=self._projects,
            mentorship=self._mentorship_sessions,
            last_action=self._last_action,
            step=self._step_count,
            skill_history=np.array(self._skill_history, dtype=np.float32),
            episode_return=self._episode_return,
            terminal_status=self._terminal_status,
            mode=self.render_mode,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def register_env() -> None:
    gym.register(
        id="CodetyAI-Learning-v0",
        entry_point="environment.custom_env:CodetyAILearningEnv",
        max_episode_steps=CodetyAILearningEnv.MAX_EPISODE_STEPS,
    )
