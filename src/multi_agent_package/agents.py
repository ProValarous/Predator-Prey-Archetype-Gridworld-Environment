import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

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
            self.agent_speed = 9

        # Define agent action space (Right, Up, Left, Down)
        self.action_space = self._make_action_space()
        self._actions_to_directions = self._action_to_direction()

        # Initial location on the grid
        self._agent_location = np.array([0, 0])

    def _make_action_space(self):
        """Defines discrete action space with 4 directions."""
        return spaces.Discrete(4)

    def _action_to_direction(self):
        """Maps action indices to direction vectors."""
        return {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])   # Down
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
            "speed": self.agent_speed
        }

    def _draw_agent(self, canvas, pix_square_size):
        """
        Draws the agent as a colored circle on the Pygame canvas.

        Args:
            canvas (pygame.Surface): The surface to draw on.
            pix_square_size (float): Size of a single grid cell in pixels.
        """
        if self.agent_type == "predator":
            color = (255, 0, 0)  # Red
        elif self.agent_type == "prey":
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Blue

        pygame.draw.circle(
            canvas,
            color,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )