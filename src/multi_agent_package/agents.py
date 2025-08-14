import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import colorsys
import math

class Agent(gym.Env):
    """
    A simple agent class for use within a multi-agent GridWorld environment.
    Each agent has a type (predator/prey/other), unique name, speed, and a location on the grid.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agent_type, agent_name):
        """
        Initializes the agent.

        Args:
            agent_type (str): The role of the agent (e.g., 'predator', 'prey').
            agent_name (str): Unique identifier for the agent.
        """
        self.agent_type = agent_type
        self.agent_name = agent_name

        # Assign speed based on agent type
        if self.agent_type == 'predator':
            self.agent_speed = 1
        elif self.agent_type == 'prey':
            self.agent_speed = 3
        else:
            self.agent_speed = 1

        self.stamina = 10

        # Define agent action space (Right, Up, Left, Down)
        self.action_space = self._make_action_space()
        self._actions_to_directions = self._action_to_direction()

        # Initial location on the grid
        self._agent_location = np.array([0, 0])

    def _make_action_space(self):
        """Defines discrete action space with 4 directions."""
        return spaces.Discrete(5)

    def _action_to_direction(self):
        """Maps action indices to direction vectors."""
        return {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1]) ,  # Down
            4: np.array([0, 0])   # Noop
        }

    def _get_obs(self, global_obs=None):
        """
        Returns the agent's local and global observation.

        Args:
            global_obs (dict): Optional global state info.

        Returns:
            dict: Local and global observation.
        """
        return {
            'local': self._agent_location,
            'global': global_obs
        }

    def _get_info(self):
        """Returns metadata about the agent (name, speed)."""
        return {
            "name": self.agent_name,
            "type": self.agent_type,
            "speed": self.agent_speed,
            "stamina": self.stamina
        }
        
    def get_agent_color(self, agent_type_str):
        """
        Returns an RGB color for the agent based on its base type and subteam number.
        Example: predator_1, predator_2, prey_1...
        """
        # --- parse base type and subteam id ---
        if "_" in agent_type_str:
            base_type, sub_id_str = agent_type_str.split("_", 1)
            try:
                sub_id = int(sub_id_str)
            except ValueError:
                sub_id = 1
        else:
            base_type = agent_type_str
            sub_id = 1

        # --- assign hue based on base type ---
        base_hues = {
            "predator": 0 / 360,    # red
            "prey": 120 / 360,      # green
            "other": 240 / 360      # blue
        }
        hue = base_hues.get(base_type.lower(), 0 / 360)

        # Determine total_subteams (allow override on the agent instance)
        total_subteams = getattr(self, "total_subteams", 5)
        if total_subteams < 1:
            total_subteams = 1

        # Adaptive spread: fewer teams → bigger saturation/brightness difference
        max_sat_spread = 0.4  # maximum possible spread in saturation
        max_val_spread = 0.25 # maximum possible spread in brightness
        spread_factor = min(1.0, 2 / total_subteams)  # 2 teams = full spread, more teams = smaller

        sat_min = 0.5
        sat_max = sat_min + max_sat_spread * spread_factor
        val_max = 1.0
        val_min = val_max - max_val_spread * spread_factor

        # Position in range (sub_id parsed again)
        fraction = ((sub_id - 1) % total_subteams) / max(1, total_subteams - 1)
        sat = sat_min + (sat_max - sat_min) * fraction
        val = val_max - (val_max - val_min) * fraction

        # HSV → RGB
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))


    def _star_points(self,center, outer_r, inner_r, points=5):
        """Return list of points for a star polygon centered at center."""
        cx, cy = center
        pts = []
        angle_step = math.pi / points  # half-step
        start_angle = -math.pi / 2  # start at top
        for i in range(points * 2):
            r = outer_r if i % 2 == 0 else inner_r
            angle = start_angle + i * angle_step
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((int(round(x)), int(round(y))))
        return pts


    def _draw_agent(self, canvas, pix_square_size):
        """
        Draws the agent as a colored shape (circle/square/triangle/star/diamond)
        and a centered, appropriately sized label (uses self.agent_name). Text color is black.
        """
        # Ensure font system initialized
        if not pygame.font.get_init():
            pygame.font.init()

        # parse agent_type for subteam id
        agent_type_str = str(getattr(self, "agent_type", "") or "")
        if "_" in agent_type_str:
            base_type, sub_id_str = agent_type_str.split("_", 1)
            try:
                sub_id = int(sub_id_str)
            except ValueError:
                sub_id = 1
        else:
            base_type = agent_type_str
            sub_id = 1

        # get color (uses self.total_subteams if set)
        color = self.get_agent_color(agent_type_str)

        # compute center (handle numpy arrays or tuples)
        cx_f, cy_f = (self._agent_location + 0.5) * pix_square_size
        cx, cy = int(round(cx_f)), int(round(cy_f))

        # radius used to size shapes
        radius = max(2, int(pix_square_size / 3))

        # choose shape based on sub_id (cycle through shapes)
        shapes = ["circle", "square", "triangle", "star", "diamond"]
        shape_id = (sub_id - 1) % len(shapes)
        shape = shapes[shape_id]

        # Draw chosen shape centered on (cx, cy)
        if shape == "circle":
            pygame.draw.circle(canvas, color, (cx, cy), radius)
        elif shape == "square":
            rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            # rounded rectangle if available; border_radius works with SDL2
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
            center = (cx, cy)
            pts = self._star_points(center, outer_r, inner_r)
            pygame.draw.polygon(canvas, color, pts)

        # --- label text (centered inside shape), up to 5 chars from agent_name ---
        text_color = (0, 0, 0)
        full_label = str(getattr(self, "agent_name", "") or "").strip()
        parts = [p for p in full_label.split() if p]
        if parts:
            short_label = (parts[0][:5] if len(parts) == 1 else (parts[0][0] + parts[1][0])).upper()
        else:
            # fallback show first 3 chars of base type
            short_label = (base_type[:3].upper() if base_type else "A")

        # fit text inside shape: aim for <= ~1.6 * radius
        max_dim = int(radius * 1.6)
        font_size = max(8, int(radius * 1.0))
        font = pygame.font.SysFont(None, font_size)
        surf = font.render(short_label, True, text_color)

        while (surf.get_width() > max_dim or surf.get_height() > max_dim) and font_size > 6:
            font_size -= 1
            font = pygame.font.SysFont(None, font_size)
            surf = font.render(short_label, True, text_color)

        text_rect = surf.get_rect(center=(cx, cy))
        canvas.blit(surf, text_rect)