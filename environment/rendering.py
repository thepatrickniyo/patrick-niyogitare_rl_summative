"""Pygame visualization for the CodetyAI learning-platform RL environment."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore

ACTION_LABELS = {
    -1: "(start)",
    0: "Beginner lesson",
    1: "Advanced lesson",
    2: "Hands-on project",
    3: "Mentor session",
    4: "Revision",
}

ENGAGEMENT_LABEL = {0: "Low", 1: "Medium", 2: "High"}


class CodetyRenderer:
    """Progress bars, last action, and skill trajectory for CodetyAI simulation."""

    def __init__(self, width: int = 720, height: int = 520, fps: int = 12) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for rendering. pip install pygame")
        pygame.init()
        pygame.display.set_caption("CodetyAI — Adaptive Learning & Mentorship (RL)")
        self.width = width
        self.height = height
        self.fps = fps
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None

    def _ensure_display(self, mode: str) -> pygame.Surface:
        assert pygame is not None
        if mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode((self.width, self.height))
                self._clock = pygame.time.Clock()
                self._font = pygame.font.SysFont("arial", 20)
                self._font_small = pygame.font.SysFont("arial", 16)
            return self._screen
        surf = pygame.Surface((self.width, self.height))
        if self._font is None:
            self._font = pygame.font.SysFont("arial", 20)
            self._font_small = pygame.font.SysFont("arial", 16)
        return surf

    def _bar(
        self,
        surf: "pygame.Surface",
        x: int,
        y: int,
        w: int,
        h: int,
        frac: float,
        bg: tuple,
        fill: tuple,
    ) -> None:
        assert pygame is not None
        pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=6)
        fw = max(0, int(w * max(0.0, min(1.0, frac))))
        if fw > 0:
            pygame.draw.rect(surf, fill, (x, y, fw, h), border_radius=6)
        pygame.draw.rect(surf, (90, 94, 102), (x, y, w, h), 2, border_radius=6)

    def render(
        self,
        skill: float,
        confidence: float,
        engagement: int,
        projects: int,
        mentorship: int,
        last_action: int,
        step: int,
        skill_history: np.ndarray,
        mode: str = "human",
    ) -> Optional[np.ndarray]:
        assert pygame is not None
        surf = self._ensure_display(mode)
        surf.fill((28, 32, 40))
        assert self._font is not None and self._font_small is not None

        title = self._font.render(
            "CodetyAI — student journey (simulated)", True, (240, 244, 250)
        )
        surf.blit(title, (20, 14))

        y = 52
        line = 36
        surf.blit(
            self._font_small.render(
                f"Step {step}  |  Projects: {projects}  |  Mentor sessions: {mentorship}",
                True,
                (200, 204, 212),
            ),
            (20, y),
        )
        y += line

        surf.blit(
            self._font_small.render(
                f"Engagement: {ENGAGEMENT_LABEL.get(engagement, '?')}",
                True,
                (180, 220, 200),
            ),
            (20, y),
        )
        y += line + 8

        surf.blit(self._font_small.render("Skill", True, (220, 222, 228)), (20, y))
        self._bar(surf, 120, y, 520, 22, skill / 100.0, (45, 48, 55), (80, 140, 220))
        surf.blit(
            self._font_small.render(f"{skill:.0f}/100", True, (230, 232, 238)),
            (650, y + 2),
        )
        y += 34

        surf.blit(self._font_small.render("Confidence", True, (220, 222, 228)), (20, y))
        self._bar(surf, 120, y, 520, 22, confidence / 100.0, (45, 48, 55), (120, 200, 140))
        surf.blit(
            self._font_small.render(f"{confidence:.0f}/100", True, (230, 232, 238)),
            (650, y + 2),
        )
        y += 44

        act_label = ACTION_LABELS.get(last_action, "?")
        surf.blit(
            self._font.render(f"Last action: {act_label}", True, (250, 210, 120)),
            (20, y),
        )
        y += 36

        # Skill over time (sparkline)
        chart_y = y
        chart_h = 120
        pygame.draw.rect(
            surf, (38, 42, 50), (20, chart_y, self.width - 40, chart_h), border_radius=8
        )
        surf.blit(
            self._font_small.render("Skill growth over time", True, (180, 184, 192)),
            (30, chart_y + 8),
        )
        if len(skill_history) >= 2:
            pts = skill_history[-min(60, len(skill_history)) :]
            xs = np.linspace(30, self.width - 50, num=len(pts))
            scale_y = chart_h - 36
            base = chart_y + chart_h - 14
            prev = None
            for i, val in enumerate(pts):
                py = base - (float(val) / 100.0) * scale_y
                px = int(xs[i])
                if prev:
                    pygame.draw.line(surf, (100, 170, 250), prev, (px, int(py)), 2)
                prev = (px, int(py))
        else:
            surf.blit(
                self._font_small.render("(collecting data…)", True, (120, 124, 132)),
                (40, chart_y + 50),
            )

        if mode == "human":
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            return None

        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    def close(self) -> None:
        if pygame is None:
            return
        if self._screen is not None:
            pygame.display.quit()
        self._screen = None
        self._clock = None


# Legacy name kept for any external reference
TrafficRenderer = CodetyRenderer
