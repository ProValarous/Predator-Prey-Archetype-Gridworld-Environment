"""
⚠️ CORE MODULE – DO NOT MODIFY ⚠️

Defines the base Agent class used by GridWorldEnv.

This class is intentionally minimal:
- identity
- movement primitives
- rendering
- metadata

Students must NOT modify this file.
All learning, reward shaping, and observation logic must live elsewhere.
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import colorsys


class Agent(gym.Env):
    """
    Base Agent for the GridWorld environment.

    =========================
    STABLE CONTRACT
    =========================
    Public attributes (read-only usage expected):
    - agent_name : str
    - agent_type : str
    - agent_team : str | int
    - agent_speed : int
    - stamina : int
    - _agent_location : np.ndarray (2,)

    Public methods:
    - _get_obs(global_obs: dict | None) -> dict
    - _get_info() -> dict
    - _draw_agent(canvas, pix_square_size) -> None

    =========================
    DESIGN PHILOSOPHY
    =========================
    - Agents do NOT decide actions
    - Agents do NOT compute rewards
    - Agents do NOT know the environment
    - Agents expose identity + state only
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agent_type: str, agent_team, agent_name: str):
        # --- identity ---
        self.agent_type: str = agent_type
        self.agent_team = agent_team
        self.agent_name: str = agent_name

        # --- movement parameters ---
        if self.agent_type == "predator":
            self.agent_speed = 1
        elif self.agent_type == "prey":
            self.agent_speed = 3
        else:
            self.agent_speed = 1

        # stamina is a pure state variable
        self.stamina: int = 10

        # --- action space ---
        self.action_space = spaces.Discrete(5)
        self._actions_to_directions = self._action_to_direction()

        # --- spatial state ---
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._start_location = np.array([0, 0], dtype=np.int32)

        # rendering helpers
        self.total_subteams: int = 5

        if not pygame.font.get_init():
            pygame.font.init()
        self._font_cache = {}

    # ------------------------------------------------------------------
    # Action primitives
    # ------------------------------------------------------------------
    def _action_to_direction(self) -> dict:
        """
        Maps discrete actions to unit grid directions.

        0: Right
        1: Up
        2: Left
        3: Down
        4: Noop
        """
        return {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------
    def _get_obs(self, global_obs: Optional[dict] = None) -> dict:
        """
        Returns the agent's observation payload.

        Students MUST NOT override this.
        Custom observations must be built at the environment level.

        Returns:
            dict with keys:
            - "local": np.ndarray [x, y]
            - "global": dict | None
        """
        return {
            "local": self._agent_location.copy(),
            "global": global_obs,
        }

    # ------------------------------------------------------------------
    # Info interface
    # ------------------------------------------------------------------
    def _get_info(self) -> dict:
        """
        Returns diagnostic metadata for logging and evaluation.
        """
        return {
            "name": self.agent_name,
            "type": self.agent_type,
            "team": self.agent_team,
            "speed": self.agent_speed,
            "stamina": self.stamina,
        }

    # ------------------------------------------------------------------
    # Team parsing utilities
    # ------------------------------------------------------------------
    def _parse_team(self) -> Tuple[str, int]:
        """
        Parses agent_team into (base_type, sub_id).

        Supported formats:
        - 3
        - "predator_2"
        - "2"
        """
        base = str(self.agent_type).lower()
        sub_id = 1

        team = self.agent_team
        if team is None:
            return base, sub_id

        try:
            if isinstance(team, int):
                return base, max(1, team)
            if isinstance(team, str):
                if "_" in team:
                    t, sid = team.split("_", 1)
                    return t.lower(), max(1, int(sid))
                if team.isdigit():
                    return base, max(1, int(team))
                return team.lower(), 1
        except Exception:
            return base, 1

        return base, 1

    # ------------------------------------------------------------------
    # Color helpers
    # ------------------------------------------------------------------
    def get_agent_color(self) -> Tuple[int, int, int]:
        """
        Computes a deterministic RGB color based on agent type and subteam.
        """
        base_type, sub_id = self._parse_team()

        base_hues = {
            "predator": 0.0,
            "prey": 120.0,
            "other": 240.0,
        }

        hue = base_hues.get(base_type, 0.0) / 360.0
        total = max(1, int(self.total_subteams))

        frac = ((sub_id - 1) % total) / max(1, total - 1)
        sat = 0.5 + 0.4 * frac
        val = 1.0 - 0.25 * frac

        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return int(r * 255), int(g * 255), int(b * 255)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _star_points(center, outer_r, inner_r, points=5) -> List[Tuple[int, int]]:
        cx, cy = center
        pts = []
        step = math.pi / points
        angle = -math.pi / 2

        for i in range(points * 2):
            r = outer_r if i % 2 == 0 else inner_r
            pts.append((
                int(cx + r * math.cos(angle)),
                int(cy + r * math.sin(angle)),
            ))
            angle += step

        return pts

    def _shape_for_subteam(self, sub_id: int) -> str:
        shapes = ["circle", "square", "triangle", "star", "diamond"]
        return shapes[(sub_id - 1) % len(shapes)]

    def _get_font(self, size: int) -> pygame.font.Font:
        if size not in self._font_cache:
            self._font_cache[size] = pygame.font.SysFont(None, size)
        return self._font_cache[size]

    def _draw_agent(self, canvas: pygame.Surface, pix_square_size: float) -> None:
        """
        Renders the agent on the grid.

        This method is PURELY visual.
        """
        base_type, sub_id = self._parse_team()
        color = self.get_agent_color()

        cx, cy = ((self._agent_location + 0.5) * pix_square_size).astype(int)
        radius = max(2, int(pix_square_size / 3))
        shape = self._shape_for_subteam(sub_id)

        if shape == "circle":
            pygame.draw.circle(canvas, color, (cx, cy), radius)
        elif shape == "square":
            pygame.draw.rect(canvas, color, pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2))
        elif shape == "triangle":
            pygame.draw.polygon(canvas, color, [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ])
        elif shape == "diamond":
            pygame.draw.polygon(canvas, color, [
                (cx, cy - radius),
                (cx - radius, cy),
                (cx, cy + radius),
                (cx + radius, cy),
            ])
        elif shape == "star":
            pts = self._star_points((cx, cy), radius, int(radius * 0.45))
            pygame.draw.polygon(canvas, color, pts)

        # label
        label = self.agent_name[:5].upper()
        font = self._get_font(max(8, int(radius * 0.8)))
        text = font.render(label, True, (0, 0, 0))
        rect = text.get_rect(center=(cx, cy))
        canvas.blit(text, rect)
