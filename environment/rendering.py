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

STATUS_BADGE = {
    "active": ("Learning in progress", (100, 180, 255)),
    "success": ("Job-ready — success!", (90, 220, 130)),
    "dropout": ("Dropout risk — episode ends", (255, 120, 120)),
    "timeout": ("Time limit (max steps)", (220, 190, 100)),
}


class CodetyRenderer:
    """
    Game-style dashboard: progress bars, last action, skill curve, optional
    on-screen checklist for assignment video recordings.
    """

    def __init__(
        self,
        width: int = 920,
        height: int = 640,
        fps: int = 12,
        demo_overlay: bool = False,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for rendering. pip install pygame")
        pygame.init()
        pygame.display.set_caption("CodetyAI — RL demo (screen recording)")
        self.width = width
        self.height = height
        self.fps = fps
        self.demo_overlay = demo_overlay
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._font_title: Optional[pygame.font.Font] = None

    def _ensure_display(self, mode: str) -> pygame.Surface:
        assert pygame is not None
        if mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode((self.width, self.height))
                self._clock = pygame.time.Clock()
                self._font_title = pygame.font.SysFont("arial", 22, bold=True)
                self._font = pygame.font.SysFont("arial", 19)
                self._font_small = pygame.font.SysFont("arial", 15)
            return self._screen
        surf = pygame.Surface((self.width, self.height))
        if self._font_title is None:
            self._font_title = pygame.font.SysFont("arial", 22, bold=True)
            self._font = pygame.font.SysFont("arial", 19)
            self._font_small = pygame.font.SysFont("arial", 15)
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

    def _draw_demo_panel(self, surf: "pygame.Surface", x: int, y: int, w: int, h: int) -> None:
        """Static text for narrated screen recordings (objective + rewards)."""
        assert pygame is not None and self._font_small is not None
        pygame.draw.rect(surf, (32, 36, 44), (x, y, w, h), border_radius=10)
        pygame.draw.rect(surf, (70, 120, 200), (x, y, w, h), 2, border_radius=10)
        lines = [
            "OBJECTIVE: Maximize employability — guide learner to job-ready",
            "(skill ≥75, confidence ≥70, ≥2 projects).",
            "",
            "REWARDS:  +10 project done   +15 skill jump (≥3/step)",
            "           +20 job-ready      −10 engagement drop   −20 dropout",
        ]
        yy = y + 10
        for line in lines:
            col = (210, 215, 225) if line.strip() else (0, 0, 0)
            if line.strip():
                surf.blit(self._font_small.render(line, True, col), (x + 12, yy))
            yy += 18

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
        episode_return: float = 0.0,
        terminal_status: str = "active",
        mode: str = "human",
    ) -> Optional[np.ndarray]:
        assert pygame is not None
        surf = self._ensure_display(mode)
        surf.fill((24, 28, 36))
        assert self._font is not None and self._font_small is not None and self._font_title is not None

        # Header — game-style title strip
        pygame.draw.rect(surf, (38, 44, 58), (0, 0, self.width, 56))
        surf.blit(
            self._font_title.render("CodetyAI — simulated student journey", True, (245, 248, 255)),
            (20, 10),
        )
        surf.blit(
            self._font_small.render(
                "Agent picks: lesson · project · mentor · revision  (RL policy)",
                True,
                (160, 170, 190),
            ),
            (20, 34),
        )

        badge_text, badge_col = STATUS_BADGE.get(
            terminal_status, ("Learning in progress", (100, 180, 255))
        )
        pygame.draw.rect(surf, badge_col, (self.width - 320, 12, 300, 32), border_radius=6)
        surf.blit(
            self._font_small.render(badge_text, True, (20, 22, 28)),
            (self.width - 308, 18),
        )

        y = 68
        line = 32
        surf.blit(
            self._font_small.render(
                f"Step {step}  |  Episode return: {episode_return:+.1f}  |  "
                f"Projects: {projects}  |  Mentor sessions: {mentorship}",
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
        y += line + 6

        surf.blit(self._font_small.render("Skill", True, (220, 222, 228)), (20, y))
        self._bar(surf, 130, y, 520, 24, skill / 100.0, (45, 48, 55), (80, 140, 220))
        surf.blit(
            self._font_small.render(f"{skill:.0f}/100", True, (230, 232, 238)),
            (660, y + 3),
        )
        y += 36

        surf.blit(self._font_small.render("Confidence", True, (220, 222, 228)), (20, y))
        self._bar(surf, 130, y, 520, 24, confidence / 100.0, (45, 48, 55), (120, 200, 140))
        surf.blit(
            self._font_small.render(f"{confidence:.0f}/100", True, (230, 232, 238)),
            (660, y + 3),
        )
        y += 44

        act_label = ACTION_LABELS.get(last_action, "?")
        surf.blit(
            self._font.render(f"Last action → {act_label}", True, (255, 215, 120)),
            (20, y),
        )
        y += 40

        chart_y = y
        chart_h = 130
        pygame.draw.rect(
            surf, (36, 40, 48), (20, chart_y, self.width - 40, chart_h), border_radius=8
        )
        surf.blit(
            self._font_small.render("Skill growth over time", True, (180, 184, 192)),
            (30, chart_y + 8),
        )
        if len(skill_history) >= 2:
            pts = skill_history[-min(70, len(skill_history)) :]
            xs = np.linspace(35, self.width - 55, num=len(pts))
            scale_y = chart_h - 40
            base = chart_y + chart_h - 16
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
                (40, chart_y + 55),
            )

        if self.demo_overlay:
            panel_y = chart_y + chart_h + 14
            self._draw_demo_panel(surf, 20, panel_y, self.width - 40, 108)

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


TrafficRenderer = CodetyRenderer
