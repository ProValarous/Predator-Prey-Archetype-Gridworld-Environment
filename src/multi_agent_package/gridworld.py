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

        # capture bookkeeping
        self._captures_total: int = 0         # cumulative captures since last reset
        self._captures_this_step: int = 0     # number of captures that happened in the current step
        self._captured_agents: List[str] = [] # list of agent_names involved in last-step captures


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
    
        self._captures_total = 0
        self._captures_this_step = 0
        self._captured_agents = []


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
        Improved reward with:
        - capture reward (sparse, large)
        - per-step distance shaping (PBRS when previous positions are available)
        - difference-reward approximation for predators (helps credit assignment)
        - adjacency bonus for surrounding the prey
        - time penalty for prey (discourage stalling)
        - obstacle penalty for stepping on obstacles
        This implementation is defensive: if previous agent positions aren't available
        (e.g. first step), PBRS contribution is zero.
        """
        rewards: Dict[str, float] = {}

        # --- hyperparams (tune these) ---
        # BALANCED preset (recommended starting point)
        catch_distance = 0
        catch_reward = 600.0
        obstacle_penalty = -5.0
        prey_time_penalty = -0.02
        time_penalty_predator = -0.005
        distance_reward_scale = 0.5
        pbrs_weight_start = 0.6
        pbrs_anneal_steps = 50_000       # number of env steps to slowly zero PBRS
        diff_reward_weight = 0.15
        adjacency_bonus = 0.25
        clip_min, clip_max = -1200.0, 1200.0
        gamma = 0.95
        lambda_dist = 1.0                    # scaling inside potential

        # Defensive: compute current step count for annealing (if not present, treat as 0)
        steps = int(getattr(self, "_episode_steps", 0))
        pbrs_weight = pbrs_weight_start * max(0.0, 1.0 - (steps / float(max(1, pbrs_anneal_steps))))

        predators = [ag for ag in self.agents if ag.agent_type.startswith("predator")]
        preys = [ag for ag in self.agents if ag.agent_type.startswith("prey")]

        # grid max euclidean distance
        max_dist = math.sqrt((self.size - 1) ** 2 + (self.size - 1) ** 2) or 1.0

        # build quick lookup for current positions
        cur_pos = {ag.agent_name: np.array(ag._agent_location, dtype=float) for ag in self.agents}
        prev_pos_list = getattr(self, "_prev_agents_location", None)
        prev_pos = None
        if prev_pos_list:
            # prev positions stored as list in same order as self.agents during reset/step
            prev_pos = {ag.agent_name: np.array(pp, dtype=float) for ag, pp in zip(self.agents, prev_pos_list)}

        # Precompute per-predator "closeness" to prey (higher => better for predator team)
        # closeness = (max_dist - min_dist) / max_dist in [0..1]
        predator_closeness = {}
        for p in predators:
            if preys:
                dists = [float(np.linalg.norm(p._agent_location - q._agent_location)) for q in preys]
                min_d = float(min(dists))
                predator_closeness[p.agent_name] = (max_dist - min_d) / max_dist
            else:
                predator_closeness[p.agent_name] = 0.0

        # team score used for difference reward approx
        team_score = sum(predator_closeness.values()) if predators else 0.0

        # For predators: precompute team scores without each predator for difference reward
        team_score_without = {}
        for p in predators:
            s_without = sum(v for k, v in predator_closeness.items() if k != p.agent_name)
            team_score_without[p.agent_name] = s_without

        # Compute rewards per agent
        for ag in self.agents:
            r = 0.0

            # --- Predator logic ---
            if ag.agent_type.startswith("predator"):
                if preys:
                    # closeness to nearest prey
                    dists = [float(np.linalg.norm(ag._agent_location - q._agent_location)) for q in preys]
                    min_d = float(min(dists))
                    norm_closeness = (max_dist - min_d) / max_dist  # 0..1
                    # simple dense shaping (normalized)
                    r += distance_reward_scale * norm_closeness

                    # adjacency bonus (encourages surrounding / cooperating)
                    # use Manhattan adjacency to encourage grid adjacency behaviour
                    for q in preys:
                        manh = int(np.sum(np.abs(ag._agent_location - q._agent_location)))
                        if 0 < manh <= 1:
                            r += adjacency_bonus
                            break

                    # capture
                    if min_d <= catch_distance:
                        r += catch_reward

                    # difference reward approx (team contribution)
                    # D_i ≈ team_score - team_score_without_i  (how much this predator adds to overall closeness)
                    D_i = team_score - team_score_without.get(ag.agent_name, 0.0)
                    r += diff_reward_weight * D_i

                    # tiny per-step cost to discourage infinite wandering
                    r += time_penalty_predator

            # --- Prey logic ---
            elif ag.agent_type.startswith("prey"):
                if predators:
                    dists = [float(np.linalg.norm(ag._agent_location - p._agent_location)) for p in predators]
                    min_d = float(min(dists))
                    # farther -> larger reward for prey
                    r += distance_reward_scale * (min_d / max_dist)
                    # capture penalty
                    if min_d <= catch_distance:
                        r -= catch_reward
                # small time penalty to discourage stalling
                r += prey_time_penalty

            # --- obstacle penalty ---
            if getattr(self, "_obstacle_location", None):
                if any(np.array_equal(ag._agent_location, obs) for obs in self._obstacle_location):
                    r += obstacle_penalty

            # --- PBRS shaping based on previous state (if available) ---
            # Potential per-agent: phi = -lambda_dist * (min_dist / max_dist)  (lower is better for predator)
            pbrs = 0.0
            if prev_pos is not None:
                # current potential
                if ag.agent_type.startswith("predator"):
                    if preys:
                        curr_min = min(float(np.linalg.norm(cur_pos[ag.agent_name] - np.array(q._agent_location))) for q in preys)
                        phi_curr = -lambda_dist * (curr_min / max_dist)
                    else:
                        phi_curr = 0.0
                elif ag.agent_type.startswith("prey"):
                    if predators:
                        curr_min = min(float(np.linalg.norm(cur_pos[ag.agent_name] - np.array(p._agent_location))) for p in predators)
                        # for prey, larger distance is better -> flip sign so higher potential when farther
                        phi_curr = +lambda_dist * (curr_min / max_dist)
                    else:
                        phi_curr = 0.0
                else:
                    phi_curr = 0.0

                # previous potential (if we have previous position for that agent)
                if ag.agent_name in prev_pos:
                    if ag.agent_type.startswith("predator"):
                        if preys:
                            prev_min = min(float(np.linalg.norm(prev_pos[ag.agent_name] - np.array(q._agent_location))) for q in preys)
                            phi_prev = -lambda_dist * (prev_min / max_dist)
                        else:
                            phi_prev = 0.0
                    elif ag.agent_type.startswith("prey"):
                        if predators:
                            prev_min = min(float(np.linalg.norm(prev_pos[ag.agent_name] - np.array(p._agent_location))) for p in predators)
                            phi_prev = +lambda_dist * (prev_min / max_dist)
                        else:
                            phi_prev = 0.0
                    else:
                        phi_prev = 0.0

                    # shaping reward F = gamma * phi(curr) - phi(prev)
                    pbrs = (gamma * phi_curr - phi_prev) * pbrs_weight
                    r += pbrs

            # --- final clipping and bookkeeping ---
            # clip for numerical stability
            r = float(np.clip(r, clip_min, clip_max))
            rewards[ag.agent_name] = r

        return rewards



    def step(self, action: Dict[str, int]) -> Dict[str, object]:
        """Apply actions for every agent and return a multi-agent dict.

        Behaviour flags (set on the env instance; defaults used if missing):
        - self.allow_cell_sharing (bool, default True): allow multiple agents in same cell.
        - self.block_agents_by_obstacles (bool, default False): treat obstacles as blocking (can't move onto them).
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
            ag.agent_name: tuple(np.array(getattr(ag, "_agent_location", ag._agent_location)).astype(int))
            for ag in self.agents
        }
        occupied_by_agents = set(agent_positions.values())

        obstacle_positions = set(tuple(obs.astype(int)) for obs in getattr(self, "_obstacle_location", []))

        # bookkeeping for diagnostics
        self._last_step_sharing = False
        self._last_collisions = []  # list of tuples (agent_name, target_pos, blocked_by)

        # Process movements: each agent may move up to `steps` micro-steps (to account for speed).
        working_positions = {ag.agent_name: np.array(ag._agent_location, dtype=int) for ag in self.agents}

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
                candidate = np.clip(working_positions[name] + direction, 0, self.size - 1).astype(int)
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
                occupied_snapshot.discard(tuple(working_positions[name]))  # free previous pos
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
            predators_here = [ag for ag in agents_here if ag.agent_type.startswith("predator")]
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
        terminated_flag = False # bool(self._captures_this_step > 0)

        agents_mdp = {
            "obs": self._get_obs(),
            "reward": self._get_reward(),
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
