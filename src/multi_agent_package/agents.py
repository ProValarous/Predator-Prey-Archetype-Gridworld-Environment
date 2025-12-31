"""
Agent module for the Predator-Prey Gridworld Environment.

This module defines the :class:`Agent` class, which represents individual
agents (predators, prey, or custom types) in a multi-agent GridWorld.

Classes
-------
Agent
    A multi-agent GridWorld agent with customizable type, team, and rendering.

Example
-------
>>> from multi_agent_package.agents import Agent
>>> predator = Agent("predator", "predator_1", "Hunter")
>>> prey = Agent("prey", "prey_1", "Runner")
>>> print(predator.agent_speed)
1
>>> print(prey.agent_speed)
3
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
    A simple agent class for a multi-agent GridWorld environment.

    See Also
    --------
    GridWorldEnv : The environment that manages multiple Agent instances.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agent_type: str, agent_team, agent_name: str):
        """
        A simple agent class for a multi-agent GridWorld environment.

        Attributes
        ----------
        agent_type : str
            Base type (e.g., "predator", "prey", "other").
        agent_team : Union[str, int]
            Subteam identifier. Can be an int (e.g., 3) or string like "predator_3".
        agent_name : str
            Human-readable name / unique id for the agent.
        agent_speed : int
            Movement speed. Predator=1, Prey=3, Other=1.
        stamina : int
            Energy resource (default: 10).
        action_space : gymnasium.spaces.Discrete
            Discrete action space with 5 actions.

        Notes
        -----
        - Rendering is implemented using pygame primitives.
        - The class provides helper methods to select colors and shapes per subteam.
        - Actions: 0=Right, 1=Up, 2=Left, 3=Down, 4=Noop

        Examples
        --------
        >>> agent = Agent("predator", "predator_1", "Hunter")
        >>> agent.agent_speed
        1
        >>> agent.action_space
        Discrete(5)

        See Also
        --------
        GridWorldEnv : The environment that manages multiple Agent instances.
        """
        self.agent_type: str = agent_type
        self.agent_team = agent_team
        self.agent_name: str = agent_name

        # Gameplay attributes
        if self.agent_type == "predator":
            self.agent_speed = 1
        elif self.agent_type == "prey":
            self.agent_speed = 3
        else:
            self.agent_speed = 1

        self.stamina = 10

        # Action space (Right, Up, Left, Down, Noop)
        self.action_space = self._make_action_space()
        self._actions_to_directions = self._action_to_direction()

        # Agent location as numpy array: [x, y] (grid coordinates)
        self._agent_location = np.array([0, 0])
        self._start_location = np.array([0, 0])

        # Optional: set this externally to help color spacing; fallback will be used.
        self.total_subteams: int = getattr(self, "total_subteams", 5)

        # Pygame font initialization & small font cache (avoid recreating fonts every frame)
        if not pygame.font.get_init():
            pygame.font.init()
        self._font_cache = {}  # maps font_size -> pygame.font.Font

    # -------------------------
    # Basic helpers / spaces
    # -------------------------
    def _make_action_space(self) -> spaces.Discrete:
        """Return discrete action space (5 actions)."""
        return spaces.Discrete(5)

    def _action_to_direction(self) -> dict:
        """Map action indices to unit direction vectors (numpy arrays)."""
        return {
            0: np.array([1, 0]),  # Right
            1: np.array([0, 1]),  # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]),  # Down
            4: np.array([0, 0]),  # Noop
        }

    def _get_obs(self, global_obs: Optional[dict] = None) -> dict:
        """
        Get the current observation for this agent.

        Parameters
        ----------
        global_obs : dict, optional
            Global observation from the environment (e.g., other agent positions).

        Returns
        -------
        dict
            Observation with keys:

            - ``"local"``: Agent's position as numpy array [x, y]
            - ``"global"``: Passed global observation or None

        Examples
        --------
        >>> agent = Agent("prey", "prey_1", "P1")
        >>> agent._agent_location = np.array([3, 4])
        >>> obs = agent._get_obs()
        >>> obs["local"]
        array([3, 4])
        """
        return {"local": self._agent_location, "global": global_obs}

    def _get_info(self) -> dict:
        """
        Get metadata information about this agent.

        Returns
        -------
        dict
            Information containing:

            - ``"name"``: Agent's display name
            - ``"type"``: Agent type (predator/prey/other)
            - ``"team"``: Team identifier
            - ``"speed"``: Movement speed
            - ``"stamina"``: Current stamina

        Examples
        --------
        >>> agent = Agent("predator", "pred_1", "Hunter")
        >>> info = agent._get_info()
        >>> info["speed"]
        1
        """
        return {
            "name": self.agent_name,
            "type": self.agent_type,
            "team": self.agent_team,
            "speed": self.agent_speed,
            "stamina": self.stamina,
        }

    # -------------------------
    # Team / subteam parsing
    # -------------------------
    def _parse_team(self) -> Tuple[str, int]:
        """
        Parse `self.agent_team` (or fallback to `self.agent_type`) and return:
        (base_type, sub_id)

        Acceptable formats:
          - agent_team = 3                -> base_type from agent_type, sub_id = 3
          - agent_team = "predator_2"     -> base_type = "predator", sub_id = 2
          - agent_team = "2"              -> sub_id = 2, base_type from agent_type
          - otherwise sub_id defaults to 1
        """
        # Preferred: parse from self.agent_team if provided
        team_val = self.agent_team

        base = str(self.agent_type or "").lower()
        sub_id = 1

        if team_val is None:
            return base, sub_id

        # If agent_team is numeric or numeric string, use it as sub_id
        try:
            if isinstance(team_val, (int, np.integer)):
                sub_id = int(team_val)
                return base, max(1, sub_id)
            if isinstance(team_val, str):
                # If string has underscore, assume format "type_k"
                if "_" in team_val:
                    parts = team_val.split("_", 1)
                    base = parts[0].lower() or base
                    try:
                        sub_id = int(parts[1])
                    except ValueError:
                        sub_id = 1
                    return base, max(1, sub_id)
                # if just numeric string
                if team_val.isdigit():
                    sub_id = int(team_val)
                    return base, max(1, sub_id)
                # otherwise treat it as a named team (no numeric id) -> sub_id stays 1
                return team_val.lower(), 1
        except Exception:
            # fall back gracefully
            return base, 1

        return base, 1

    # -------------------------
    # Color generation
    # -------------------------
    def get_agent_color(self, agent_team: Optional[str] = None) -> Tuple[int, int, int]:
        """
        Compute RGB color based on agent type and subteam.

        Colors use HSV color space:

        - Predator: Red hue (0°)
        - Prey: Green hue (120°)
        - Other: Blue hue (240°)

        Parameters
        ----------
        agent_team : str or int, optional
            Override team for color calculation. If None, uses self.agent_team.

        Returns
        -------
        tuple of (int, int, int)
            RGB values in range [0, 255].

        Examples
        --------
        >>> predator = Agent("predator", "predator_1", "P1")
        >>> r, g, b = predator.get_agent_color()
        >>> r > g  # Red-dominant for predators
        True
        """
        # Decide which team string to parse
        original_team = agent_team if agent_team is not None else self.agent_team
        # parse_team expects self.agent_team, so temporarily set and restore if needed
        old_team = self.agent_team
        if agent_team is not None:
            self.agent_team = agent_team

        base_type, sub_id = self._parse_team()

        # restore
        if agent_team is not None:
            self.agent_team = old_team

        # Base hues in HSV (range 0.0 - 1.0)
        base_hues = {
            "predator": 0.0 / 360.0,  # red
            "prey": 120.0 / 360.0,  # green
            "other": 240.0 / 360.0,  # blue
        }
        hue = base_hues.get(base_type.lower(), 0.0)

        # Determine total_subteams (allow external override on instance)
        total = getattr(self, "total_subteams", 1)
        try:
            total = int(total)
        except Exception:
            total = 5
        total = max(1, total)

        # Adaptive spread: fewer teams -> larger sat/val differences
        max_sat_spread = 0.4
        max_val_spread = 0.25
        spread_factor = min(1.0, 2.0 / float(total))  # 2 teams => full spread

        sat_min = 0.5
        sat_max = sat_min + max_sat_spread * spread_factor
        val_max = 1.0
        val_min = val_max - max_val_spread * spread_factor

        # Map sub_id into 0..1 fraction across total subteams
        fraction = ((sub_id - 1) % total) / max(1, total - 1)
        sat = sat_min + (sat_max - sat_min) * fraction
        val = val_max - (val_max - val_min) * fraction

        # Convert HSV -> RGB (0..1 floats) then scale to 0..255 ints
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
        return int(r_f * 255), int(g_f * 255), int(b_f * 255)

    # -------------------------
    # Shape helpers
    # -------------------------
    @staticmethod
    def _star_points(
        center: Tuple[int, int], outer_r: float, inner_r: float, points: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Compute polygon points for a star (useful for pygame.draw.polygon).

        Parameters
        ----------
        center : (x, y)
        outer_r : float
        inner_r : float
        points : int
            Number of star points (default 5).
        """
        cx, cy = center
        pts: List[Tuple[int, int]] = []
        angle_step = math.pi / points  # half step
        start_angle = -math.pi / 2  # start at top
        for i in range(points * 2):
            r = outer_r if (i % 2 == 0) else inner_r
            angle = start_angle + i * angle_step
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((int(round(x)), int(round(y))))
        return pts

    def _shape_for_subteam(self, sub_id: int) -> str:
        """
        Deterministic mapping from subteam id to a shape.

        Cycles through: circle, square, triangle, star, diamond.
        """
        shapes = ["circle", "square", "triangle", "star", "diamond"]
        return shapes[(sub_id - 1) % len(shapes)]

    # -------------------------
    # Rendering
    # -------------------------
    def _get_font(self, font_size: int) -> pygame.font.Font:
        """Return a cached pygame font instance for given size (or create one)."""
        if font_size not in self._font_cache:
            self._font_cache[font_size] = pygame.font.SysFont(None, font_size)
        return self._font_cache[font_size]

    def _render_label(
        self, canvas: pygame.Surface, center: Tuple[int, int], label: str, max_dim: int
    ) -> None:
        """
        Draw a black label centered at `center` and scaled to fit within `max_dim`.
        This mutates the canvas (pygame Surface).
        """
        text_color = (0, 0, 0)
        font_size = max(8, int(max_dim * 0.5))  # heuristic start size
        font = self._get_font(font_size)
        surf = font.render(label, True, text_color)

        # shrink to fit if necessary
        while (
            surf.get_width() > max_dim or surf.get_height() > max_dim
        ) and font_size > 6:
            font_size -= 1
            font = self._get_font(font_size)
            surf = font.render(label, True, text_color)

        text_rect = surf.get_rect(center=center)
        canvas.blit(surf, text_rect)

    def _draw_agent(self, canvas: pygame.Surface, pix_square_size: float) -> None:
        """
        Draw the agent on a pygame canvas.

        Parameters
        ----------
        canvas : pygame.Surface
            Surface to draw on.
        pix_square_size : float
            Size of one grid cell in pixels.

        Notes
        -----
        Draws a colored shape (based on subteam) with a centered label.
        """
        # Ensure pygame font system is ready
        if not pygame.font.get_init():
            pygame.font.init()

        # Parse team/type for color and shape decisions
        base_type, sub_id = self._parse_team()
        color = self.get_agent_color()  # uses self.agent_team internally

        # Compute center in pixels. support numpy arrays or tuples
        cx_f, cy_f = (self._agent_location + 0.5) * pix_square_size
        cx, cy = int(round(cx_f)), int(round(cy_f))
        center = (cx, cy)

        # Radius for shapes
        radius = max(2, int(pix_square_size / 3))

        # Determine shape
        shape = self._shape_for_subteam(sub_id)

        # Draw shape
        if shape == "circle":
            pygame.draw.circle(canvas, color, center, radius)
        elif shape == "square":
            rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            # Try rounded rect if supported
            try:
                pygame.draw.rect(canvas, color, rect, border_radius=max(0, radius // 4))
            except TypeError:
                pygame.draw.rect(canvas, color, rect)
        elif shape == "triangle":
            pts = [
                (cx, cy - radius),
                (cx - radius, cy + radius),
                (cx + radius, cy + radius),
            ]
            pygame.draw.polygon(canvas, color, pts)
        elif shape == "diamond":
            pts = [
                (cx, cy - radius),
                (cx - radius, cy),
                (cx, cy + radius),
                (cx + radius, cy),
            ]
            pygame.draw.polygon(canvas, color, pts)
        elif shape == "star":
            outer_r = radius
            inner_r = max(1, int(radius * 0.45))
            pts = self._star_points(center, outer_r, inner_r)
            pygame.draw.polygon(canvas, color, pts)

        # Label: prefer agent_name; fallback to base type abbreviation
        full_label = str(getattr(self, "agent_name", "") or "").strip()
        if full_label:
            # Use first word or initials, up to 5 chars
            parts = [p for p in full_label.split() if p]
            if parts:
                if len(parts) == 1:
                    label = parts[0][:5].upper()
                else:
                    label = (parts[0][0] + parts[1][0]).upper()
            else:
                label = full_label[:5].upper()
        else:
            label = base_type[:3].upper() if base_type else "A"

        # Put label inside shape; size it relative to radius
        max_dim = int(radius * 1.6)
        self._render_label(canvas, center, label, max_dim)
