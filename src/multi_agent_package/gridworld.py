"""
GridWorld environment module for the Predator-Prey scenario.

This module implements a multi-agent grid-based environment where
predator agents attempt to capture prey agents. The environment follows
the Gymnasium interface for compatibility with standard RL tooling.

Classes
-------
GridWorldEnv
    Multi-agent GridWorld environment for Predator-Prey scenarios.

Example
-------
>>> from multi_agent_package.agents import Agent
>>> from multi_agent_package.gridworld import GridWorldEnv
>>> agents = [
...     Agent("predator", "predator_1", "P1"),
...     Agent("prey", "prey_1", "R1"),
... ]
>>> env = GridWorldEnv(agents=agents, size=5, render_mode="human")
>>> obs, info = env.reset(seed=42)
>>> env.close()
"""

import math
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from multi_agent_package.agents import Agent


class GridWorldEnv(gym.Env):
    """
    Multi-agent GridWorld environment for Predator-Prey scenarios.

    This environment implements a discrete grid world where predator agents
    attempt to capture prey agents. It follows the Gymnasium interface for
    compatibility with standard RL libraries.

    Design notes
    --------------
    - This environment is intentionally minimal and focuses on grid positioning
      and visualization. Game logic (collisions, goals, team-rewards, etc.) should
      be implemented in wrapper methods or by extending this class.
    - Rendering uses pygame; call `reset(render_mode='human')` or set
      `render_mode='human'` on construction to enable an interactive window.

    Attributes
    ----------
    size : int
        Grid dimensions (size × size).
    agents : List[Agent]
        List of agents participating in the environment.
    num_agents : Dict[str, int]
        Count of agents by type (e.g., ``{"total": 4, "predator": 2, "prey": 2}``).
    action_space : gymnasium.spaces.Discrete
        Discrete action space with 5 actions:

        - 0: Right ``[+1, 0]``
        - 1: Up ``[0, +1]``
        - 2: Left ``[-1, 0]``
        - 3: Down ``[0, -1]``
        - 4: Noop ``[0, 0]``

    render_mode : str or None
        Current rendering mode (``'human'``, ``'rgb_array'``, or ``None``).
    window_size : int
        Pygame window size in pixels.
    perc_num_obstacle : float
        Percentage of grid cells that are obstacles.

    Notes
    -----
    - Each ``Agent`` instance must implement ``_agent_location``, ``_draw_agent``,
      ``_get_info``, and ``_get_obs`` methods.
    - Game logic (captures, rewards) is implemented in the ``step()`` method.
    - Rendering uses pygame; set ``render_mode='human'`` to enable visualization.
    - The environment is intentionally minimal for interpretable MARL research.

    Examples
    --------
    Create and run environment:

    >>> agents = [Agent("predator", "pred_1", "P1"), Agent("prey", "prey_1", "R1")]
    >>> env = GridWorldEnv(agents=agents, size=8, render_mode="human")
    >>> obs, info = env.reset(seed=42)
    >>> result = env.step({"P1": 0, "R1": 1})
    >>> env.close()

    See Also
    --------
    Agent : The agent class used within this environment.
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
        """
        Initialize the GridWorld environment.

        Parameters
        ----------
        agents : List[Agent]
            List of Agent instances that will participate in the environment.
        render_mode : str, optional
            How to render the environment:

            - ``'human'``: Display pygame window
            - ``'rgb_array'``: Return pixel array
            - ``None``: No rendering (fastest for training)

        size : int, default=5
            Grid dimensions (size x size cells).
        perc_num_obstacle : float, default=30.0
            Percentage of cells to fill with obstacles (0-100).
        window_size : int, default=600
            Pygame window size in pixels.
        seed : int, optional
            Random seed for reproducible obstacle and agent placement.

        Raises
        ------
        AssertionError
            If ``render_mode`` is not None, 'human', or 'rgb_array'.

        Examples
        --------
        >>> agents = [Agent("predator", "pred_1", "P1")]
        >>> env = GridWorldEnv(agents=agents, size=10, seed=42)
        >>> env.size
        10

        Notes
        -----
        The following attributes are initialized:

        - ``_agents_location``: Empty list (populated on reset)
        - ``_obstacle_location``: Empty list (populated on reset)
        - ``_captures_total``: 0
        - ``_captured_agents``: Empty list
        - ``window``, ``clock``: None (lazy initialization)

        See Also
        --------
        reset : Initialize agent and obstacle positions.
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.size = int(size)
        self.window_size = int(window_size)
        self.perc_num_obstacle = float(perc_num_obstacle)
        self._num_obstacles = int(
            (self.perc_num_obstacle / 100.0) * (self.size * self.size)
        )

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

        # capture bookkeeping
        self._captures_total: int = 0  # cumulative captures since last reset
        self._captures_this_step: int = (
            0  # number of captures that happened in the current step
        )
        self._captured_agents: List[str] = (
            []
        )  # list of agent_names involved in last-step captures

    # -------------------------
    # Spaces / observations
    # -------------------------
    def _make_observation_space(self) -> spaces.Dict:
        """
        Create observation space mapping agent names to position spaces.

        Returns
        -------
        gymnasium.spaces.Dict
            Dictionary space with agent names as keys.
        """
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
        """
        Create discrete action space (5 actions).

        Returns
        -------
        gymnasium.spaces.Discrete
            Discrete space with 5 actions.

        Notes
        -----
        Actions: 0=Right, 1=Up, 2=Left, 3=Down, 4=Noop.
        """
        return spaces.Discrete(5)

    def _get_obs(self) -> Dict[str, Dict]:
        """
        Get current observations for all agents.

        Returns
        -------
        Dict[str, Dict]
            Mapping of agent_name → observation dict containing:

            - ``"local"``: Agent's position as numpy array
            - ``"global"``: Distances to other agents and obstacles

        Examples
        --------
        >>> obs = env._get_obs()
        >>> obs["P1"]["local"]
        array([2, 3])
        """
        obs: Dict[str, Dict] = {}
        for ag in self.agents:
            distances: Dict[str, int] = {}
            obstacle_distances: Dict[str, int] = {}

            def _dist_func(agent1: Agent, agent2: Agent) -> int:
                """Euclidean distance (rounded to int) between two agents."""
                return int(
                    np.linalg.norm(agent1._agent_location - agent2._agent_location)
                )

            # Distances to other agents
            for ag2 in self.agents:
                if ag.agent_name != ag2.agent_name:
                    distances[ag2.agent_name] = _dist_func(ag, ag2)

            # Distances to obstacles (indexed) from self._obstacle_location
            if getattr(self, "_obstacle_location", None):
                for idx, obstacle in enumerate(self._obstacle_location):
                    # obstacle expected as (x, y) np.ndarray or tuple
                    try:
                        dist = int(
                            np.linalg.norm(ag._agent_location - np.asarray(obstacle))
                        )
                    except Exception:
                        ox, oy = obstacle  # fallback if needed
                        dist = int(
                            np.linalg.norm(ag._agent_location - np.array([ox, oy]))
                        )
                    obstacle_distances[f"obstacle_{idx}"] = dist

            obs[ag.agent_name] = ag._get_obs(
                {
                    "dist_agents": distances,
                    "dist_obstacles": obstacle_distances,
                }
            )
        return obs

    def _get_info(self) -> Dict[str, Dict]:
        """
        Get metadata information for all agents.

        Returns
        -------
        Dict[str, Dict]
            Mapping of agent_name → info dict from each agent's ``_get_info()``.
        """
        return {ag.agent_name: ag._get_info() for ag in self.agents}

    # -------------------------
    # Reset / initialization
    # -------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        start_location: str = "random",
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to start a new episode.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible initialization.
        options : dict, optional
            Additional options (reserved for future use).
        start_location : str, default="random"
            Agent placement strategy: ``"random"`` or ``"fixed"``.

        Returns
        -------
        observations : Dict[str, Dict]
            Initial observations for each agent.
        info : Dict[str, Dict]
            Initial metadata for each agent.

        Examples
        --------
        >>> obs, info = env.reset(seed=42)
        >>> obs["P1"]["local"]
        array([1, 3])

        Notes
        -----
        Reset sequence:

        1. Reseed RNG if seed provided
        2. Reset capture counters
        3. Place obstacles randomly
        4. Place agents in remaining cells
        5. Return initial observations
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._agents_location = []

        self._captures_total = 0
        self._captures_this_step = 0
        self._captured_agents = []

        # Assign unique start positions for agents by sampling without replacement
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.rng.shuffle(all_positions)

        for i, ag in enumerate(self.agents):
            pos = np.array(random.choice(all_positions), dtype=np.int32)
            ag._agent_location = pos.copy()
            ag._start_location = pos.copy()
            self._agents_location.append(pos.copy())
            # print(f"Agent '{ag.agent_name}', start location: {ag._agent_location}")

        # Obstacles (avoid agent starts)
        self._obstacle_location = self._initialize_obstacle(
            avoid=set((tuple(p) for p in self._agents_location))
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _initialize_obstacle(
        self, avoid: Optional[set] = None
    ) -> List[np.ndarray]:  # fix randomness
        """
        Initialize obstacle positions on the grid.

        Parameters
        ----------
        avoid : set, optional
            Set of (x, y) tuples where obstacles cannot be placed.

        Returns
        -------
        List[np.ndarray]
            List of [x, y] obstacle coordinates.
        """
        avoid = avoid or set()
        cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in avoid
        ]
        self.rng.shuffle(cells)
        chosen = cells[: max(0, min(len(cells), self._num_obstacles))]
        return [np.array(c, dtype=np.int32) for c in chosen]

    # -------------------------
    # Potential-based reward shaping
    # -------------------------
    def _distance_potential(
        self, agent_positions: Dict[str, tuple[int, int]], weight: float
    ) -> Dict[str, float]:
        """
        Calculate distance-based potential for reward shaping.

        Parameters
        ----------
        agent_positions : Dict[str, Tuple[int, int]]
            Mapping of agent_name → (x, y) position.
        weight : float
            Scaling factor for potential values.

        Returns
        -------
        Dict[str, float]
            Mapping of agent_name → potential value.

        Notes
        -----
        Used for potential-based reward shaping to provide denser rewards.
        Predators: closer to prey = higher potential.
        Prey: farther from predators = higher potential.
        """
        dist_potential: Dict[str, float] = {}

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Separate predator and prey positions
        predator_positions = [
            pos for name, pos in agent_positions.items() if name.startswith("predator")
        ]
        prey_positions = [
            pos for name, pos in agent_positions.items() if name.startswith("prey")
        ]

        for name, pos in agent_positions.items():
            r = 0.0
            if name.startswith("predator"):
                # Reward closer distance (negative shaping for being far)
                if prey_positions:
                    nearest_dist = min(
                        manhattan_dist(pos, prey_pos) for prey_pos in prey_positions
                    )
                    r -= weight * nearest_dist

            elif name.startswith("prey"):
                # Reward being farther away
                if predator_positions:
                    nearest_dist = min(
                        manhattan_dist(pos, pred_pos) for pred_pos in predator_positions
                    )
                    r += weight * nearest_dist

            dist_potential[name] = r

        return dist_potential

    def potential_reward(self, state) -> Dict[str, float]:
        """Calculate the Potential-Based Reward Shaping (PBRS) value for the obs.

        !!! obs: agent local positions only - 1 prey, 1 pred !!!

        This function computes a shaping reward based on the distances between predators and preys.
        The reward is designed to encourage predators to move closer to preys and preys to move away from predators.

        Parameters
        ----------
        state : object
            Current environment state.


        Returns
        -------
        Dict[str, float]
            The computed PBRS value for the current state for each agent.
            {'agent_name': PBRS value, ...}
        """
        pbrs_value = {ag.agent_name: 0.0 for ag in self.agents}

        # Compute distance-based potential rewards
        dist_potential = self._distance_potential(state, weight=0.0)

        for ag in self.agents:
            pbrs_value[ag.agent_name] += dist_potential[ag.agent_name]

        return pbrs_value

    def base_reward(self) -> Dict[str, float]:  # name changed to base-reward
        """
        Computes the reward for each agent in the gridworld environment based on (states, not obs):
        1. Capture:
            - Predators receive a large positive reward for capturing a prey.
            - Preys receive an equivalent large negative reward when captured.
        2. Step cost:
            - Predators incur a small negative penalty each step (to encourage efficiency).
        3. Obstacle hit:
            - Both predators and preys incur a penalty when moving into an obstacle.

        Returns
        -------
        Dict[str, float]
            Mapping of agent_name → base reward.

        Notes
        -----
        Reward structure:

        - Predator capture: ``+10.0``
        - Predator timestep: ``-0.01``
        - Prey survival: ``+0.1``
        - Prey captured: ``-10.0`
        """

        rewards: Dict[str, float] = {}

        # base rewards parameter
        capture_reward = 100.0
        obstacle_hit_penalty = 200
        step_cost = 5

        predators = [ag for ag in self.agents if ag.agent_type.startswith("predator")]
        preys = [ag for ag in self.agents if ag.agent_type.startswith("prey")]

        captured_set = set(getattr(self, "_captured_agents", []))
        obstacle_positions = set(
            tuple(obs.astype(int)) for obs in getattr(self, "_obstacle_location", [])
        )

        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        for ag in self.agents:
            r = 0.0

            # Capture rewards/penalties
            if ag.agent_type.startswith("predator"):
                if ag.agent_name in captured_set:
                    r += capture_reward
                r -= step_cost
            elif ag.agent_type.startswith("prey"):
                if ag.agent_name in captured_set:
                    r -= capture_reward

            # Obstacle penalty: if agent is currently on an obstacle cell
            if tuple(ag._agent_location) in obstacle_positions:
                r -= obstacle_hit_penalty

            rewards[ag.agent_name] = r

        return rewards

    def step(self, action: Dict[str, int]) -> Dict[str, object]:
        """
        Execute one timestep of the environment.

        Parameters
        ----------
        action : Dict[str, int]
            Mapping of agent_name → action index (0-4).

        Returns
        -------
        Dict[str, object]
            Dictionary containing:

            - ``"obs"``: New observations for each agent
            - ``"reward"``: Rewards for each agent
            - ``"done"``: Whether episode ended (all prey captured)
            - ``"truncated"``: Whether episode was cut short
            - ``"info"``: Additional information

        Examples
        --------
        >>> result = env.step({"P1": 0, "R1": 1})
        >>> result["done"]
        False

        Notes
        -----
        Step execution order:

        1. Process each agent's action
        2. Validate moves (bounds, obstacles)
        3. Check for captures
        4. Calculate rewards
        5. Check termination
        6. Return results
        """
        # --- preserve previous positions for PBRS & diagnostics ---
        self._prev_agents_location = [
            pos.copy() if hasattr(pos, "copy") else np.array(pos, dtype=np.int32)
            for pos in getattr(self, "_agents_location", [])
        ]

        # ensure an episode step counter exists (used for annealing shaping)
        if not hasattr(self, "_episode_steps"):
            self._episode_steps = 0

        # Behaviour flags (default safe values)
        allow_sharing = bool(getattr(self, "allow_cell_sharing", True))
        block_by_obstacle = bool(getattr(self, "block_agents_by_obstacles", True))

        # Validate input
        if not isinstance(action, dict):
            raise ValueError("`action` must be a dict mapping agent_name -> action_idx")

        # Build current occupancy sets (tuples) from the tracked agent locations
        agent_positions = {
            ag.agent_name: tuple(
                np.array(getattr(ag, "_agent_location", ag._agent_location)).astype(int)
            )
            for ag in self.agents
        }
        occupied_by_agents = set(agent_positions.values())

        obstacle_positions = set(
            tuple(obs.astype(int)) for obs in getattr(self, "_obstacle_location", [])
        )

        # bookkeeping for diagnostics
        self._last_step_sharing = False
        self._last_collisions = (
            []
        )  # list of tuples (agent_name, target_pos, blocked_by)

        # Process movements: each agent may move up to `steps` micro-steps (to account for speed).
        working_positions = {
            ag.agent_name: np.array(ag._agent_location, dtype=int) for ag in self.agents
        }

        # Determine maximum micro-steps to run (max speed among agents)
        max_micro_steps = 0
        agent_requested_steps = {}
        for ag in self.agents:
            steps = ag._get_info().get("speed", getattr(ag, "agent_speed", 1))
            try:
                steps = int(steps)
            except Exception:
                steps = 1
            agent_requested_steps[ag.agent_name] = steps
            if steps > max_micro_steps:
                max_micro_steps = steps

        # run micro-step loop
        for micro in range(max_micro_steps):
            # prepare a snapshot of occupied cells at the start of this micro-step
            occupied_snapshot = set(tuple(pos) for pos in working_positions.values())

            for ag in self.agents:
                name = ag.agent_name
                requested = agent_requested_steps.get(name, 1)
                # only move if this agent still has micro-steps remaining
                if micro >= requested:
                    continue

                # fetch action (default noop)
                a = action.get(name, 4)
                direction = ag._actions_to_directions.get(int(a), np.array([0, 0]))
                # compute tentative new position
                candidate = np.clip(
                    working_positions[name] + direction, 0, self.size - 1
                ).astype(int)
                candidate_t = tuple(candidate)

                blocked_by = None
                blocked = False

                # Check obstacle blocking if requested
                if block_by_obstacle and (candidate_t in obstacle_positions):
                    blocked = True
                    blocked_by = "obstacle"

                # Check agent blocking (only if sharing not allowed)
                if (not allow_sharing) and (candidate_t in occupied_snapshot):
                    # if the only occupant is this agent itself (i.e. moving zero), allow
                    if candidate_t != tuple(working_positions[name]):
                        blocked = True
                        blocked_by = "agent"

                if blocked:
                    # don't update working_positions[name] (agent stays put)
                    # record collision for diagnostics
                    self._last_collisions.append((name, candidate_t, blocked_by))
                    continue

                # commit movement to working positions
                occupied_snapshot.discard(
                    tuple(working_positions[name])
                )  # free previous pos
                working_positions[name] = candidate
                occupied_snapshot.add(candidate_t)

        # After micro-steps complete, apply the working positions to agents and update stamina
        for ag in self.agents:
            name = ag.agent_name
            new_pos = working_positions[name]
            ag._agent_location = new_pos.copy()
            # consume stamina as before: if agent asked for steps <= stamina, consume; else they regain
            requested = agent_requested_steps.get(name, 1)
            if requested <= ag.stamina:
                ag.stamina = max(0, ag.stamina - int(requested))
            else:
                # cannot move full speed -> noop behaviour was applied micro-step wise; regain small amount
                ag.stamina = min(ag.stamina + 1, 100)

        # detect sharing: if multiple agents occupy same tuple position, flag it
        pos_counts = {}
        for ag in self.agents:
            pos_t = tuple(ag._agent_location)
            pos_counts[pos_t] = pos_counts.get(pos_t, 0) + 1
        self._last_step_sharing = any(cnt > 1 for cnt in pos_counts.values())

        # update tracked current agent positions (used next call as previous)
        self._agents_location = [ag._agent_location.copy() for ag in self.agents]

        # -------------------------
        # CAPTURE DETECTION
        # -------------------------
        # Reset per-step capture bookkeeping
        self._captures_this_step = 0
        self._captured_agents = []

        # Map positions -> agents in that position
        pos_to_agents: Dict[tuple, list] = {}
        for ag in self.agents:
            pos_t = tuple(np.array(ag._agent_location, dtype=int))
            pos_to_agents.setdefault(pos_t, []).append(ag)

        # For each occupied cell, if at least one predator AND at least one prey are present -> capture
        for pos, agents_here in pos_to_agents.items():
            predators_here = [
                ag for ag in agents_here if ag.agent_type.startswith("predator")
            ]
            preys_here = [ag for ag in agents_here if ag.agent_type.startswith("prey")]
            if predators_here and preys_here:
                # Count each prey captured individually. Change logic here if you prefer
                # to count one capture per step or per cell.
                for prey_ag in preys_here:
                    self._captures_this_step += 1
                    # record the prey and participating predators
                    self._captured_agents.append(prey_ag.agent_name)
                    for p in predators_here:
                        self._captured_agents.append(p.agent_name)

        # Update cumulative captures (per-episode)
        self._captures_total = self._captures_total + self._captures_this_step

        # increment step counter
        self._episode_steps += 1

        # build mdp return; terminate episode if a capture happened this step
        terminated_flag = self._captures_total > 1

        agents_mdp = {
            "obs": self._get_obs(),
            "reward": self.base_reward(),
            "terminated": terminated_flag,
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
        """
        Render current state using pygame.

        Returns
        -------
        np.ndarray or None
            Pixel array if ``render_mode='rgb_array'``, else None.

        Notes
        -----
        Draws grid lines, obstacles (gray), and agents (via ``_draw_agent``).
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
            pygame.draw.rect(
                canvas, (50, 50, 50), pygame.Rect(ox - r, oy - r, r * 2, r * 2)
            )

        # Draw agents
        for ag in self.agents:
            ag._draw_agent(canvas, pix_square_size)

        # Draw grid lines (thin black)
        line_color = (0, 0, 0)
        for x in range(self.size + 1):
            pos = int(round(pix_square_size * x))
            pygame.draw.line(
                canvas, line_color, (0, pos), (self.window_size, pos), width=1
            )
            pygame.draw.line(
                canvas, line_color, (pos, 0), (pos, self.window_size), width=1
            )

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
        """
        Public render method following Gymnasium interface.

        Parameters
        ----------
        mode : str, optional
            Override render mode (``'human'`` or ``'rgb_array'``).

        Returns
        -------
        np.ndarray or None
            Pixel array if mode is ``'rgb_array'``, else None.
        """
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
        """
        Clean up environment resources.

        Notes
        -----
        Closes pygame window and releases resources.
        Safe to call multiple times.
        """
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
