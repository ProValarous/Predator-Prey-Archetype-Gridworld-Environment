"""
Agent module for the Predator-Prey Gridworld Environment.

This module defines the :class:`Agent` class, which represents individual
agents (predators, prey, or custom types) in a multi-agent GridWorld.

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
    A multi-agent GridWorld agent with customizable type, team, and rendering.

    This class represents an individual agent in the Predator-Prey GridWorld
    environment. Each agent has a type (predator/prey/other), belongs to a
    team/subteam, and can be rendered with distinct colors and shapes.

    The class extends ``gymnasium.Env`` to maintain compatibility with
    standard RL interfaces, though the primary stepping logic is handled
    by the parent :class:`GridWorldEnv`.

    Attributes
    ----------
    agent_type : str
        Base role of the agent:

        - ``"predator"``: Hunter role, speed=1
        - ``"prey"``: Evader role, speed=3
        - ``"other"``: Custom role, speed=1

    agent_team : Union[str, int]
        Team/subteam identifier used for grouping and color differentiation.
        Accepts multiple formats:

        - Integer: ``3`` → subteam 3
        - String with underscore: ``"predator_2"`` → type "predator", subteam 2
        - Numeric string: ``"2"`` → subteam 2

    agent_name : str
        Human-readable unique identifier. Used in logging, observations,
        and rendered labels.

    agent_speed : int
        Movement speed multiplier. Set automatically based on ``agent_type``:

        - Predator: 1
        - Prey: 3
        - Other: 1

    stamina : int
        Energy resource for the agent (default: 10). Available for
        extended game mechanics.

    action_space : gymnasium.spaces.Discrete
        Discrete action space with 5 actions:

        - 0: Right ``[+1, 0]``
        - 1: Up ``[0, +1]``
        - 2: Left ``[-1, 0]``
        - 3: Down ``[0, -1]``
        - 4: Noop ``[0, 0]``

    total_subteams : int
        Number of subteams for color spacing calculations (default: 5).

    Notes
    -----
    - Rendering uses pygame primitives with HSV-based color generation.
    - Shapes cycle through: circle, square, triangle, star, diamond.
    - Font rendering is cached to avoid recreation overhead each frame.

    Examples
    --------
    Create a predator agent:

    >>> agent = Agent("predator", "predator_1", "Hunter")
    >>> agent.agent_speed
    1
    >>> agent.action_space
    Discrete(5)

    Create a prey agent:

    >>> prey = Agent("prey", "prey_2", "Runner")
    >>> prey.agent_speed
    3

    Get agent color:

    >>> r, g, b = agent.get_agent_color()
    >>> r > g  # Red-dominant for predators
    True

    See Also
    --------
    GridWorldEnv : The environment that manages multiple Agent instances.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agent_type: str, agent_team, agent_name: str):
        """
        Initialize an Agent instance.

        Creates a new agent with the specified type, team, and name.
        Speed is automatically set based on agent type.

        Parameters
        ----------
        agent_type : str
            Base role for the agent. Determines speed and color:

            - ``"predator"``: Hunter role, speed=1, red color family
            - ``"prey"``: Evader role, speed=3, green color family
            - ``"other"``: Custom role, speed=1, blue color family

        agent_team : Union[str, int]
            Team/subteam identifier. Determines color variation and shape
            within the same agent type. Accepted formats:

            - ``int``: Direct subteam ID (e.g., ``3``)
            - ``str``: Format ``"type_id"`` (e.g., ``"predator_2"``)
            - ``str``: Numeric string (e.g., ``"2"``)

        agent_name : str
            Unique display name for the agent. Used in:

            - Observation dictionaries (as identifier)
            - Info dictionaries (metadata)
            - Rendered labels on the agent shape
            - Logging and debugging

        Raises
        ------
        None
            This method does not raise exceptions. Invalid agent types
            default to ``"other"`` behavior with speed=1.

        Examples
        --------
        Create a predator:

        >>> predator = Agent("predator", "predator_1", "Hunter")
        >>> predator.agent_type
        'predator'
        >>> predator.agent_speed
        1

        Create a prey with integer team:

        >>> prey = Agent("prey", 2, "Runner")
        >>> prey.agent_team
        2
        >>> prey.agent_speed
        3

        Create a custom agent:

        >>> other = Agent("other", "observer_1", "Watcher")
        >>> other.agent_speed
        1

        Notes
        -----
        The following attributes are initialized:

        - ``agent_speed``: Set based on ``agent_type``
        - ``stamina``: Set to 10
        - ``action_space``: ``Discrete(5)``
        - ``_agent_location``: ``[0, 0]``
        - ``_start_location``: ``[0, 0]``
        - ``total_subteams``: 5 (for color spacing)

        See Also
        --------
        _get_info : Get agent metadata as dictionary.
        _get_obs : Get agent observation.
        get_agent_color : Get RGB color based on type and team.
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
        """
        Create the discrete action space.

        Returns
        -------
        gymnasium.spaces.Discrete
            Discrete space with 5 actions.

        Notes
        -----
        Actions: 0=Right, 1=Up, 2=Left, 3=Down, 4=Noop.
        """
        return spaces.Discrete(5)

    def _action_to_direction(self) -> dict:
        """
        Map action indices to direction vectors.

        Returns
        -------
        dict
            Mapping of action index to numpy direction vector.

        Examples
        --------
        >>> agent._actions_to_directions[0]
        array([1, 0])  # Right
        """
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
        Parse team identifier into base type and subteam ID.

        Returns
        -------
        tuple of (str, int)
            ``(base_type, sub_id)`` where sub_id >= 1.

        Notes
        -----
        Parsing rules:

        - Integer ``3`` → ``(agent_type, 3)``
        - String ``"predator_2"`` → ``("predator", 2)``
        - Numeric string ``"2"`` → ``(agent_type, 2)``

        Examples
        --------
        >>> agent = Agent("predator", "predator_2", "P1")
        >>> agent._parse_team()
        ('predator', 2)
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

        Parameters
        ----------
        agent_team : str or int, optional
            Override team for color calculation.

        Returns
        -------
        tuple of (int, int, int)
            RGB values in range [0, 255].

        Notes
        -----
        Color mapping: predator=red, prey=green, other=blue.
        Subteams get varied saturation/brightness.

        Examples
        --------
        >>> predator = Agent("predator", "predator_1", "P1")
        >>> r, g, b = predator.get_agent_color()
        >>> r > g  # Red-dominant
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
        Compute polygon vertices for a star shape.

        Parameters
        ----------
        center : tuple of (int, int)
            Center coordinates (x, y) in pixels.
        outer_r : float
            Radius to outer points.
        inner_r : float
            Radius to inner points.
        points : int, default=5
            Number of star points.

        Returns
        -------
        list of tuple of (int, int)
            Vertices for pygame.draw.polygon.

        Examples
        --------
        >>> pts = Agent._star_points((100, 100), 20, 10, 5)
        >>> len(pts)
        10
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
        Get shape name for a subteam ID.

        Parameters
        ----------
        sub_id : int
            Subteam identifier (1-indexed).

        Returns
        -------
        str
            Shape: ``"circle"``, ``"square"``, ``"triangle"``, ``"star"``, or ``"diamond"``.

        Notes
        -----
        Shapes cycle: 1=circle, 2=square, 3=triangle, 4=star, 5=diamond, 6=circle...

        Examples
        --------
        >>> agent._shape_for_subteam(1)
        'circle'
        >>> agent._shape_for_subteam(4)
        'star'
        """
        shapes = ["circle", "square", "triangle", "star", "diamond"]
        return shapes[(sub_id - 1) % len(shapes)]

    # -------------------------
    # Rendering
    # -------------------------
    def _get_font(self, font_size: int) -> pygame.font.Font:
        """
        Get a cached pygame font instance.

        Parameters
        ----------
        font_size : int
            Font size in points.

        Returns
        -------
        pygame.font.Font
            Cached font instance.

        Notes
        -----
        Fonts are cached to avoid recreation overhead during rendering.
        """
        if font_size not in self._font_cache:
            self._font_cache[font_size] = pygame.font.SysFont(None, font_size)
        return self._font_cache[font_size]

    def _render_label(
        self, canvas: pygame.Surface, center: Tuple[int, int], label: str, max_dim: int
    ) -> None:
        """
        Render a text label centered at the given position.

        Parameters
        ----------
        canvas : pygame.Surface
            Surface to draw on.
        center : tuple of (int, int)
            Center position (x, y) for the label.
        label : str
            Text to render.
        max_dim : int
            Maximum dimension; font shrinks to fit.

        Notes
        -----
        Mutates the canvas by blitting text onto it.
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
        Draws a colored shape (based on type/subteam) with a centered label.
        Shape is determined by :meth:`_shape_for_subteam`.
        Color is determined by :meth:`get_agent_color`.
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
