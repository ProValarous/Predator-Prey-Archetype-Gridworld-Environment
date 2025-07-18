import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from multi_agent_package.agents import Agent

class GridWorldEnv(gym.Env):
    """
    A multi-agent grid world environment using Gymnasium and Pygame.
    Each agent moves in a grid world, interacting with others and avoiding obstacles.

    Attributes:
        agents (list): List of agent instances.
        render_mode (str): 'human' or 'rgb_array'.
        size (int): Size of the grid.
        perc_num_obstacle (int): % of grid cells as obstacles.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agents: list, render_mode=None, size=5, perc_num_obstacle=30):
        # Basic environment parameters
        self.size = size
        self.window_size = 600
        self.perc_num_obstacle = perc_num_obstacle
        self._num_obstacles = int((self.perc_num_obstacle / 100) * (self.size * self.size))

        # Define action space (Right, Up, Left, Down)
        self.action_space = self._make_action_space()

        # Initialize agents
        self.agents = agents
        self.num_agents = {'total': len(self.agents)}
        for ag in self.agents:
            if str(ag.agent_type) in self.num_agents:
                self.num_agents[str(ag.agent_type)] += 1
            else:
                self.num_agents[str(ag.agent_type)] = 1

        # Set rendering mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _make_observation_space(self):
        """Creates the observation space for each agent as their position."""
        observation_space = spaces.Dict({})
        for ag in self.agents:
            observation_space[ag.agent_name] = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)
        return observation_space

    def _make_action_space(self):
        """Defines 4 discrete actions for each agent."""
        return spaces.Discrete(5)

    def _get_obs(self):
        """Gets current observations: relative distances to other agents."""
        obs = {}
        for ag in self.agents:
            dist = {}
            for ag2 in self.agents:
                if ag.agent_name != ag2.agent_name:
                    dist[ag2.agent_name] = self._dist_func(ag, ag2)
            obs[ag.agent_name] = ag._get_obs({'dist': dist})
        return obs

    def _get_info(self):
        """Collects debugging/info dictionary from each agent."""
        return {ag.agent_name: ag._get_info() for ag in self.agents}

    def reset(self, seed=None, options=None, start_location="random"):
        """Resets environment and randomly assigns agent positions."""
        self._agents_location = []
        for ag in self.agents:
            ag._start_location = self._initialize_start_location(loc=start_location)
            for aj in self.agents:
                aj._start_location = self._initialize_start_location(loc=start_location)
                if aj.agent_name != ag.agent_name and np.array_equal(ag._start_location, aj._start_location):
                    aj._start_location = self._initialize_start_location(loc=start_location)
            self._agents_location.append(ag._start_location)

        self._obstacle_location = self._initialize_obstacle()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _initialize_start_location(self, loc="random"):
        """Generates random or fixed start location for agents."""
        if loc == 'random':
            return self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            return np.array(loc)

    def _initialize_obstacle(self):
        """Placeholder for obstacle initialization."""
        return []

    def _dist_func(self, agent1, agent2):
        """Euclidean distance between two agents."""
        return int(np.linalg.norm(agent1._agent_location - agent2._agent_location))

    def _get_reward(self):
        """Assigns reward to each agent. Default = 0 for all."""
        get_obs = self._get_obs()
        dist_coeff = 10
        dist_lst = []
        for ag in get_obs:
            get_obs[ag]['global']
        return {ag.agent_name: 0 for ag in self.agents}

    def step(self, action: dict):
        """
        Executes environment step using actions per agent.
        Returns updated obs, reward, info and done flags.
        """
        agents_mdp = {}
        for ag in self.agents:
            steps = ag._get_info()['speed']
            if steps <= ag.stamina:
                for _ in range(steps):
                    direction = ag._actions_to_directions[action[ag.agent_name]]
                    ag._agent_location = np.clip(ag._agent_location + direction, 0, self.size - 1)
            else:
                print("no action")
                direction = ag._actions_to_directions[4]
                ag._agent_location = np.clip(ag._agent_location + direction, 0, self.size - 1)
            if action[ag.agent_name] == 4 or steps > ag.stamina:
                ag.stamina += 1
                print("in + 1")
            else:
                ag.stamina -= steps
                print("in -",steps)
            print(ag._get_info()["stamina"])


        agents_mdp['obs'] = self._get_obs()
        agents_mdp['reward'] = self._get_reward()
        agents_mdp['terminated'] = False
        agents_mdp['trunc'] = False
        agents_mdp['info'] = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return agents_mdp

    def _render_frame(self):
        """Renders the environment frame using Pygame."""

        pygame.display.set_caption("Simulation")

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for ag in self.agents:
            ag._draw_agent(canvas, pix_square_size)

        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render(self):
        """Public render interface."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """Closes Pygame rendering session."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
