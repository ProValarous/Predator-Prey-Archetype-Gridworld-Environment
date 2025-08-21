import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from multi_agent_package.agents import Agent


class GridWorldEnv(gym.Env):
    """
    Multi-agent GridWorld environment.

    Each `Agent` instance must implement the methods/properties used here
    (e.g. `_agent_location`, `_draw_agent`, `_get_info`, `_get_obs`).

    Design notes
    --------------
    - This environment is intentionally minimal and focuses on grid positioning
      and visualization. Game logic (collisions, goals, team-rewards, etc.) should
      be implemented in wrapper methods or by extending this class.
    - Rendering uses pygame; call `reset(render_mode='human')` or set
      `render_mode='human'` on construction to enable an interactive window.

    Parameters
    ----------
    agents : List[Agent]
        List of agent instances participating in the environment.
    render_mode : Optional[str]
        'human' or 'rgb_array' or None. When 'human', a pygame window will be
        created on `reset` / `_render_frame`.
    size : int
        Number of grid cells on one side (grid is size x size).
    perc_num_obstacle : float
        Percentage (0-100) of grid cells that should be obstacles.
    window_size : int
        Pixel size of the pygame rendering window (square).
    seed : Optional[int]
        Seed for the internal RNG (numpy Generator).
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

        self.size = int(size)
        self.window_size = int(window_size)
        self.perc_num_obstacle = float(perc_num_obstacle)
        self._num_obstacles = int((self.perc_num_obstacle / 100.0) * (self.size * self.size))

        # RNG for reproducible placement
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # action / observation spaces (per-agent) are simple Discrete/Box for now
        self.action_space = self._make_action_space()

        # Agents
        self.agents: List[Agent] = agents
        # Compute counts per type
        self.num_agents: Dict[str, int] = {"total": len(self.agents)}
        for ag in self.agents:
            key = str(ag.agent_type)
            self.num_agents[key] = self.num_agents.get(key, 0) + 1

        # rendering state
        self.render_mode = render_mode
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Internal state
        self._agents_location: List[np.ndarray] = []
        self._obstacle_location: List[np.ndarray] = []

    # -------------------------
    # Spaces / observations
    # -------------------------
    def _make_observation_space(self) -> spaces.Dict:
        """Create a Dict observation space mapping agent_name -> Box(position)."""
        obs_space = spaces.Dict({})
        for ag in self.agents:
            obs_space.spaces[ag.agent_name] = spaces.Box(
                low=0,
                high=self.size - 1,
                shape=(2,),
                dtype=np.int32,
            )
        return obs_space

    def _make_action_space(self) -> spaces.Discrete:
        """Discrete actions: Right, Up, Left, Down, Noop"""
        return spaces.Discrete(5)

    def _get_obs(self) -> Dict[str, Dict]:
        """Return current observations for every agent.

        The returned structure is a dict mapping agent_name -> agent._get_obs(...)
        where each agent receives a small 'global' context containing distances
        to all other agents and all obstacles (from `self._obstacle_location`).
        """
        obs: Dict[str, Dict] = {}
        for ag in self.agents:
            distances: Dict[str, int] = {}
            obstacle_distances: Dict[str, int] = {}

            # Distances to other agents
            for ag2 in self.agents:
                if ag.agent_name != ag2.agent_name:
                    distances[ag2.agent_name] = self._dist_func(ag, ag2)

            # Distances to obstacles (indexed) from self._obstacle_location
            if getattr(self, "_obstacle_location", None):
                for idx, obstacle in enumerate(self._obstacle_location):
                    # obstacle expected as (x, y) np.ndarray or tuple
                    try:
                        dist = int(np.linalg.norm(ag._agent_location - np.asarray(obstacle)))
                    except Exception:
                        ox, oy = obstacle  # fallback if needed
                        dist = int(np.linalg.norm(ag._agent_location - np.array([ox, oy])))
                    obstacle_distances[f"obstacle_{idx}"] = dist

            obs[ag.agent_name] = ag._get_obs({
                "dist_agents": distances,
                "dist_obstacles": obstacle_distances,
            })
        return obs

    def _get_info(self) -> Dict[str, Dict]:
        """Return per-agent info dict for logging/debugging."""
        return {ag.agent_name: ag._get_info() for ag in self.agents}

    # -------------------------
    # Reset / initialization
    # -------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, start_location: str = "random") -> Tuple[Dict, Dict]:
        """Reset environment and return (observation, info).

        Parameters
        ----------
        seed : Optional[int]
            If provided, reseed the internal RNG for deterministic resets.
        start_location : str or tuple
            'random' or an explicit location or function to generate starts.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._agents_location = []

        # Assign unique start positions for agents by sampling without replacement
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.rng.shuffle(all_positions)

        for i, ag in enumerate(self.agents):
            pos = np.array(all_positions[i], dtype=np.int32)
            ag._agent_location = pos.copy()
            ag._start_location = pos.copy()
            self._agents_location.append(pos.copy())

        # Obstacles (avoid agent starts)
        self._obstacle_location = self._initialize_obstacle(avoid=set((tuple(p) for p in self._agents_location)))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _initialize_obstacle(self, avoid: Optional[set] = None) -> List[np.ndarray]:
        """Place obstacles randomly on the grid, avoiding positions in `avoid`.

        Returns a list of coordinates (numpy arrays).
        """
        avoid = avoid or set()
        cells = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in avoid]
        self.rng.shuffle(cells)
        chosen = cells[: max(0, min(len(cells), self._num_obstacles))]
        return [np.array(c, dtype=np.int32) for c in chosen]

    # -------------------------
    # Utilities
    # -------------------------
    def _dist_func(self, agent1: Agent, agent2: Agent) -> int:
        """Euclidean distance (rounded to int) between two agents."""
        return int(np.linalg.norm(agent1._agent_location - agent2._agent_location))

    def _get_reward(self) -> Dict[str, float]:
        """
        Tuned reward system:
        - Predators: big bonus for catching prey, scaled reward for getting closer.
        - Prey: big penalty for being caught, scaled reward for distance from predators.
        - Obstacles: stronger penalty to avoid camping.
        - Small living penalty for prey to encourage movement.
        """
        rewards = {}
        catch_distance = 0
        obstacle_penalty = -10
        catch_reward = 50
        distance_reward_scale = 5.0  # scaling for distance-based rewards
        prey_living_penalty = -0.5

        predator_positions = [ag._agent_location for ag in self.agents if ag.agent_type.startswith("predator")]
        prey_positions = [ag._agent_location for ag in self.agents if ag.agent_type.startswith("prey")]

        # Optional: Max distance in the grid for normalization
        grid_size = self.size  # assuming self.grid_shape = (rows, cols)

        for ag in self.agents:
            reward = 0.0

            if ag.agent_type.startswith("predator"):
                distances = [np.linalg.norm(ag._agent_location - prey_pos) for prey_pos in prey_positions]
                if distances:
                    min_distance = min(distances)
                    # Normalize and scale
                    norm_reward = (grid_size - min_distance) / grid_size
                    reward += norm_reward * distance_reward_scale
                    if min_distance <= catch_distance:
                        reward += catch_reward

            elif ag.agent_type.startswith("prey"):
                distances = [np.linalg.norm(ag._agent_location - pred_pos) for pred_pos in predator_positions]
                if distances:
                    min_distance = min(distances)
                    # Normalize and scale
                    norm_reward = min_distance / grid_size
                    reward += norm_reward * distance_reward_scale
                    if min_distance <= catch_distance:
                        reward -= catch_reward
                # Small time penalty to push movement
                reward += prey_living_penalty

            # Obstacle penalty
            if any(np.array_equal(ag._agent_location, obs) for obs in self._obstacle_location):
                reward += obstacle_penalty

            rewards[ag.agent_name] = reward

        return rewards

    def compute_cumulative_reward(self,
                                rewards: Dict[str, float],
                                name_to_type: Optional[Dict[str, str]] = None,
                                eps: float = 1e-6) -> float:
        """
        Compute the cumulative score according to the formula:

            cumulative_reward = (1 / mean_diff) * (sum_prey_rewards + sum_predator_rewards)

        Where:
        - rewards: dict mapping agent_name -> scalar reward (as returned by _get_reward()).
        - name_to_type: optional mapping {agent_name: "prey"|"predator"|...}. If provided,
            it is used to determine which agents are prey vs predator.
            If not provided, a simple name-based heuristic is used:
            - if 'prey' in name (case-insensitive) => prey
            - elif 'predator' in name (case-insensitive) => predator
            - elif name startswith 'py' => prey
            - elif name startswith 'pd' => predator
            - otherwise the agent is ignored in prey/predator aggregations
        - eps: small value to avoid division by zero.

        Returns
        -------
        float
            The computed cumulative_reward (float). If there are no predators or no preys
            in the reward dict, returns 0.0 (can't compute meaningful mean difference).
        """
        if not rewards:
            return 0.0

        # If a mapping is provided, use it; else infer using heuristic from agent_name strings.
        def infer_type(agent_name: str) -> Optional[str]:
            if name_to_type and agent_name in name_to_type:
                return name_to_type[agent_name].lower()
            ln = agent_name.lower()
            if "prey" in ln:
                return "prey"
            if "predator" in ln:
                return "predator"
            if ln.startswith("py"):
                return "prey"
            if ln.startswith("pd"):
                return "predator"
            return None  # unknown / ignore

        prey_values = []
        pred_values = []

        for name, val in rewards.items():
            typ = infer_type(name)
            if typ == "prey":
                prey_values.append(float(val))
            elif typ == "predator":
                pred_values.append(float(val))
            else:
                # ignore other agent types (or unknown names) for this metric
                continue

        # If either group is empty, we cannot compute a meaningful mean-difference; return 0.
        if len(prey_values) == 0 or len(pred_values) == 0:
            return 0.0

        sum_prey = sum(prey_values)
        sum_pred = sum(pred_values)
        mean_prey = sum_prey / len(prey_values)
        mean_pred = sum_pred / len(pred_values)

        mean_diff = abs(mean_pred - mean_prey)
        denom = max(mean_diff, eps)  # avoid division by zero
        cumulative = (1.0 / denom) * (sum_prey + sum_pred)

        return float(cumulative)

    # -------------------------
    # Step function
    # -------------------------
    def step(self, action: Dict[str, int]) -> Dict[str, object]:
        """Apply actions for every agent and return a multi-agent dict.

        Parameters
        ----------
        action : Dict[str,int]
            Mapping from agent_name -> action index (0..4).

        Returns
        -------
        Dict with keys:
            - 'obs': observation dict
            - 'reward': reward dict
            - 'terminated': bool (episode-level)
            - 'trunc': bool (episode-level)
            - 'info': info dict
        """
        # Validate input
        if not isinstance(action, dict):
            raise ValueError("`action` must be a dict mapping agent_name -> action_idx")

        for ag in self.agents:
            # default noop if missing
            a = action.get(ag.agent_name, 4)
            steps = ag._get_info().get("speed", getattr(ag, "agent_speed", 1))

            # movement limited by stamina
            if steps <= ag.stamina:
                for _ in range(int(steps)):
                    direction = ag._actions_to_directions.get(int(a), np.array([0, 0]))
                    ag._agent_location = np.clip(ag._agent_location + direction, 0, self.size - 1)
                # consume stamina
                ag.stamina = max(0, ag.stamina - int(steps))
            else:
                # cannot move full speed -> noop and regain a bit of stamina
                direction = ag._actions_to_directions.get(4, np.array([0, 0]))
                ag._agent_location = np.clip(ag._agent_location + direction, 0, self.size - 1)
                ag.stamina = min(ag.stamina + 1, 100)

    

        agents_mdp = {
            "obs": self._get_obs(),
            "reward": self._get_reward(),
            "terminated": False,
            "trunc": False,
            "info": self._get_info(),
        }

        if self.render_mode == "human":
            self._render_frame()

        return agents_mdp

    # -------------------------
    # Rendering
    # -------------------------
    def _render_frame(self) -> Optional[np.ndarray]:
        """Render the current grid to a pygame surface (and display if human).

        Returns an RGB array if render_mode != 'human'.
        """
        # Initialize pygame window and clock lazily
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # white background
        pix_square_size = float(self.window_size) / float(self.size)

        # Draw obstacles
        for obs in self._obstacle_location:
            ox = int((obs[0] + 0.5) * pix_square_size)
            oy = int((obs[1] + 0.5) * pix_square_size)
            r = max(2, int(pix_square_size / 4))
            pygame.draw.rect(canvas, (50, 50, 50), pygame.Rect(ox - r, oy - r, r * 2, r * 2))

        # Draw agents
        for ag in self.agents:
            ag._draw_agent(canvas, pix_square_size)

        # Draw grid lines (thin black)
        line_color = (0, 0, 0)
        for x in range(self.size + 1):
            pos = int(round(pix_square_size * x))
            pygame.draw.line(canvas, line_color, (0, pos), (self.window_size, pos), width=1)
            pygame.draw.line(canvas, line_color, (pos, 0), (pos, self.window_size), width=1)

        if self.render_mode == "human":
            # Blit to window and update display
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None

        # Return RGB array for 'rgb_array' mode
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Public render API. Use mode='rgb_array' to get an image array."""
        if mode is not None:
            prev = self.render_mode
            self.render_mode = mode
            result = self._render_frame()
            self.render_mode = prev
            return result

        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self) -> None:
        """Close pygame resources if they were created."""
        if self.window is not None:
            try:
                pygame.display.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass
            self.window = None
            self.clock = None
