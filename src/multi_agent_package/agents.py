import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import colorsys

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
    
    import colorsys

    def get_agent_color(self,agent_type_str):
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

        # --- vary saturation and brightness for contrast ---
        # alternate between lighter & darker pastel tones
        total_subteams = 5
        
        # Adaptive spread: fewer teams → bigger saturation/brightness difference
        max_sat_spread = 0.4  # maximum possible spread in saturation
        max_val_spread = 0.25 # maximum possible spread in brightness
        spread_factor = min(1.0, 2 / total_subteams)  # 2 teams = full spread, more teams = smaller

        sat_min = 0.5
        sat_max = sat_min + max_sat_spread * spread_factor
        val_max = 1.0
        val_min = val_max - max_val_spread * spread_factor

        # Position in range
        fraction = (sub_id - 1) / max(1, total_subteams - 1)
        sat = sat_min + (sat_max - sat_min) * fraction
        val = val_max - (val_max - val_min) * fraction

        # HSV → RGB
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))


    def _draw_agent(self, canvas, pix_square_size):
        """
        Draws the agent as a colored circle and a centered, appropriately sized label
        (uses self.agent_type). Text color is black.
        """

        # Ensure font system initialized (do once in __init__ ideally)
        if not pygame.font.get_init():
            pygame.font.init()

        # --- circle color ---
        # if getattr(self, "agent_type", "")[:7] == "predator" or getattr(self, "agent_name", "")[:2] == 'PD':
        #     color = (255, 0, 0)
        # elif getattr(self, "agent_type", "")[:7] == "prey" or getattr(self, "agent_name", "")[:2] == 'PY':
        #     color = (0, 255, 0)
        # else:
        #     color = (0, 0, 255)

        # 1. Get color based on agent_type (e.g., predator_1, prey_3)
        
        color = self.get_agent_color(self.agent_type)

        # compute pixel center (handle numpy array or tuple)
        cx_f, cy_f = (self._agent_location + 0.5) * pix_square_size
        center = (int(round(cx_f)), int(round(cy_f)))

        radius = max(2, int(pix_square_size / 3))
        pygame.draw.circle(canvas, color, center, radius)

        # text specifics
        text_color = (0, 0, 0)  # forced black
        full_label = str(getattr(self, "agent_name", "") or "").strip()
        # short label (use first 5 letters or initials)
        parts = [p for p in full_label.split() if p]
        if parts:
            short_label = (parts[0][:5] if len(parts) == 1 else (parts[0][0] + parts[1][0])).upper()
        else:
            short_label = "A"

        # Heuristic: if cell is big enough, attempt to draw full label below circle.
        MIN_CELL_FOR_FULL_LABEL = 100 #pix_square_size # tweakable
        padding = 2

        if pix_square_size >= MIN_CELL_FOR_FULL_LABEL and len(full_label) <= 12:
            # render full label below the circle and make it fit horizontally in the cell
            max_width = int(pix_square_size - 4)
            font_size = max(10, int(pix_square_size / 4))
            font = pygame.font.SysFont(None, font_size)
            surf = font.render(full_label, True, text_color)

            while surf.get_width() > max_width and font_size > 6:
                font_size -= 1
                font = pygame.font.SysFont(None, font_size)
                surf = font.render(full_label, True, text_color)

            text_rect = surf.get_rect()
            # place it centered horizontally, just below the circle
            text_rect.midtop = (center[0], center[1] + radius + padding)
            # clamp vertically inside cell
            cell_top = int(round(cy_f - pix_square_size / 2))
            cell_bottom = int(round(cy_f + pix_square_size / 2))
            if text_rect.top < cell_top:
                text_rect.top = cell_top + padding
            if text_rect.bottom > cell_bottom:
                text_rect.bottom = cell_bottom - padding

            canvas.blit(surf, text_rect)
            return

        # Otherwise, draw short_label INSIDE the circle and size so it fits comfortably.
        # Aim for text width/height <= ~1.6 * radius (so it sits inside with some padding).
        max_dim = int(radius * 1.6)
        font_size = max(8, int(radius * 1.0))  # start guess
        font = pygame.font.SysFont(None, font_size)
        surf = font.render(short_label, True, text_color)

        # shrink until it fits (or until minimum size)
        while (surf.get_width() > max_dim or surf.get_height() > max_dim) and font_size > 6:
            font_size -= 1
            font = pygame.font.SysFont(None, font_size)
            surf = font.render(short_label, True, text_color)

        text_rect = surf.get_rect(center=center)
        canvas.blit(surf, text_rect)
