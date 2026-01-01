"""
⚠️ CORE MODULE – DO NOT MODIFY ⚠️

This file defines the base GridWorld environment.
All environment dynamics, physics, and episode control live here.

Students must NOT edit this file.
All research variation must be implemented via:
- configs/
- observations/
- rewards/
"""

from typing import List, Dict, Optional, Tuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from multi_agent_package.core.agent import Agent


class GridWorldEnv(gym.Env):
    """
    Multi-agent GridWorld environment.

    =========================
    AGENT CONTRACT (STABLE)
    =========================
    Each Agent instance must provide:

    Attributes:
    - agent_name : str
    - agent_type : str
    - agent_team : str
    - agent_speed : int
    - stamina : int
    - _agent_location : np.ndarray shape (2,)

    Methods:
    - _get_obs(global_obs: dict) -> dict
    - _get_info() -> dict
    - _draw_agent(canvas, pix_square_size) -> None

    =========================
    EXTENSION POINTS
    =========================
    Students may extend behavior ONLY via:
    - self.reward_fn (callable)
    - self.observation_builder (callable)

    These must be pure functions and must NOT:
    - modify env state
    - move agents
    - change dynamics
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        agents: List[Agent],
        render_mode: Optional[str] = None,
        size: int = 5,
        perc_num_obstacle: float = 30.0,
        window_size: int = 600,
        seed: Optional[int] = None,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # --- environment parameters ---
        self.size = int(size)
        self.window_size = int(window_size)
        self.perc_num_obstacle = float(perc_num_obstacle)
        self._num_obstacles = int((self.perc_num_obstacle / 100.0) * self.size * self.size)

        # deterministic RNG
        self.rng = np.random.default_rng(seed)

        # agents
        self.agents: List[Agent] = agents
        self.action_space = spaces.Discrete(5)

        # rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # state
        self._agents_location: List[np.ndarray] = []
        self._obstacle_location: List[np.ndarray] = []

        # episode bookkeeping
        self._episode_steps = 0
        self._captures_total = 0
        self._captures_this_step = 0
        self._captured_agents: List[str] = []

        # behavior flags (configurable)
        self.allow_cell_sharing: bool = True
        self.block_agents_by_obstacles: bool = True

        # extension hooks (students plug logic here)
        self.reward_fn = None
        self.observation_builder = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._episode_steps = 0
        self._captures_total = 0
        self._captures_this_step = 0
        self._captured_agents.clear()
        self._agents_location.clear()

        # assign unique start positions
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.rng.shuffle(all_positions)

        for ag in self.agents:
            pos = np.array(self.rng.choice(all_positions), dtype=np.int32)
            ag._agent_location = pos.copy()
            ag._start_location = pos.copy()
            self._agents_location.append(pos.copy())

        self._obstacle_location = self._initialize_obstacles(
            avoid={tuple(p) for p in self._agents_location}
        )

        return self._build_observations(), self._get_info()

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    def _initialize_obstacles(self, avoid: set) -> List[np.ndarray]:
        cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in avoid
        ]
        self.rng.shuffle(cells)
        chosen = cells[: self._num_obstacles]
        return [np.array(c, dtype=np.int32) for c in chosen]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _default_observations(self) -> Dict[str, Dict]:
        obs = {}

        for ag in self.agents:
            dist_agents = {}
            dist_obstacles = {}

            for other in self.agents:
                if other.agent_name != ag.agent_name:
                    dist_agents[other.agent_name] = int(
                        np.linalg.norm(ag._agent_location - other._agent_location)
                    )

            for i, obs_pos in enumerate(self._obstacle_location):
                dist_obstacles[f"obstacle_{i}"] = int(
                    np.linalg.norm(ag._agent_location - obs_pos)
                )

            obs[ag.agent_name] = ag._get_obs(
                {
                    "dist_agents": dist_agents,
                    "dist_obstacles": dist_obstacles,
                }
            )

        return obs

    def _build_observations(self) -> Dict[str, Dict]:
        if self.observation_builder is None:
            return self._default_observations()
        return self.observation_builder(self)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def base_reward(self) -> Dict[str, float]:
        rewards = {}

        capture_reward = 100.0
        obstacle_penalty = 200.0
        step_cost = 5.0

        obstacle_positions = {tuple(o) for o in self._obstacle_location}
        captured = set(self._captured_agents)

        for ag in self.agents:
            r = 0.0

            if ag.agent_type.startswith("predator"):
                if ag.agent_name in captured:
                    r += capture_reward
                r -= step_cost

            if ag.agent_type.startswith("prey"):
                if ag.agent_name in captured:
                    r -= capture_reward

            if tuple(ag._agent_location) in obstacle_positions:
                r -= obstacle_penalty

            rewards[ag.agent_name] = r

        return rewards

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, int]) -> Dict[str, object]:
        self._captures_this_step = 0
        self._captured_agents.clear()

        # movement
        for ag in self.agents:
            a = action.get(ag.agent_name, 4)
            direction = ag._actions_to_directions[a]
            candidate = np.clip(ag._agent_location + direction, 0, self.size - 1)

            if self.block_agents_by_obstacles:
                if tuple(candidate) in map(tuple, self._obstacle_location):
                    continue

            ag._agent_location = candidate.copy()

        # capture detection
        pos_map: Dict[Tuple[int, int], List[Agent]] = {}
        for ag in self.agents:
            pos_map.setdefault(tuple(ag._agent_location), []).append(ag)

        for agents_here in pos_map.values():
            predators = [a for a in agents_here if a.agent_type.startswith("predator")]
            preys = [a for a in agents_here if a.agent_type.startswith("prey")]

            if predators and preys:
                for prey in preys:
                    self._captures_this_step += 1
                    self._captured_agents.append(prey.agent_name)
                for p in predators:
                    self._captured_agents.append(p.agent_name)

        self._captures_total += self._captures_this_step
        self._episode_steps += 1

        # rewards
        rewards = self.base_reward()

        if self.reward_fn is not None:
            custom = self.reward_fn(self)
            for k in rewards:
                rewards[k] += custom.get(k, 0.0)

        terminated = self._captures_total >= 1

        if self.render_mode == "human":
            self._render_frame()

        return {
            "obs": self._build_observations(),
            "reward": rewards,
            "terminated": terminated,
            "trunc": False,
            "info": self._get_info(),
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def _get_info(self) -> Dict[str, Dict]:
        return {ag.agent_name: ag._get_info() for ag in self.agents}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_frame(self) -> Optional[np.ndarray]:

        if self.render_mode != "human":
          return None
        
       

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            self.clock = pygame.time.Clock()

        pygame.event.pump()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        # obstacles
        for obs in self._obstacle_location:
            x = int((obs[0] + 0.5) * pix)
            y = int((obs[1] + 0.5) * pix)
            r = max(2, int(pix / 4))
            pygame.draw.rect(canvas, (50, 50, 50), pygame.Rect(x - r, y - r, 2 * r, 2 * r))

        # agents
        for ag in self.agents:
            ag._draw_agent(canvas, pix)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(pygame.surfarray.pixels3d(canvas), (1, 0, 2))

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def render(self):
        """
        Gym-compatible render entry point.
        """
        if self.render_mode != "human":
            return None

        return self._render_frame()
