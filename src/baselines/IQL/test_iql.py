"""
Clean, robust tester / visualizer for tabular IQL Q-tables.

This script attempts to be compatible with the trainer that saves Q-tables as a .npz
or checkpoint. It tries multiple loading strategies and multiple ways to compute the
state index so it will work even if the training used a compact joint-state index
based on (own_cell, discretized_distance_to_other).

Usage
-----
python test_iql_cleaned.py --file baselines/IQL/iql_qs.npz --size 8 --episodes 5 --pause 0.05

Notes
-----
- If `baselines.IQL.utils.global_joint_state_index` is available and the observation
  contains the expected `obs[agent]['global']['dist_agents']` information, the
  tester will use that to compute state indices (preferred; matches trainer).
- If not available, it falls back to a simple position-only indexing (x*size + y).
- The script is defensive about different key names inside the loaded .npz/checkpoint
  (e.g., agent names vs. 'Q_<agent_name>').

"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, Optional

import numpy as np

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

# Try to import helper utilities from your baselines. They are optional here but
# preferred because they keep test behavior identical to the trainer.
try:
    from baselines.IQL.utils import load_checkpoint, global_joint_state_index  # type: ignore
except Exception:
    load_checkpoint = None  # type: ignore
    global_joint_state_index = None  # type: ignore


LOGGER = logging.getLogger("test_iql")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


def make_agents() -> tuple[Agent, Agent]:
    """Create a prey and predator Agent instance.

    Adjust the constructor arguments if your Agent signature differs.
    """
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def try_load_qs(file_path: str) -> Dict[str, np.ndarray]:
    """Attempt to load Q-tables from a file.

    Supports:
    - Using `load_checkpoint` (if available) which may return a (Qs, meta) pair
    - A plain .npz file where arrays are saved under agent-name keys or "Q_<agent>" keys

    Returns
    -------
    Dict[str, np.ndarray]: mapping from agent_name -> Q-table array
    """
    # 1) If load_checkpoint helper exists, try it first (keeps compatibility with trainer checkpoint format)
    if load_checkpoint is not None:
        try:
            Qs, meta = load_checkpoint(file_path)
            if isinstance(Qs, dict) and all(isinstance(v, np.ndarray) for v in Qs.values()):
                LOGGER.info("Loaded checkpoint using helper: %s", file_path)
                return Qs
        except FileNotFoundError:
            raise
        except Exception:
            LOGGER.debug("load_checkpoint failed, falling back to np.load", exc_info=True)

    # 2) Fallback to np.load on the .npz file
    with np.load(file_path, allow_pickle=False) as data:
        files = list(data.files)
        LOGGER.info("NPZ contents: %s", files)
        qs: Dict[str, np.ndarray] = {}
        # Heuristics: keys could be agent names (e.g., "prey_1" or "prey"), or "Q_<agent_name>"
        for key in files:
            arr = data[key]
            # Accept only 2D arrays for Q-tables
            if arr.ndim == 2:
                # normalize key names: strip leading 'Q_' if present
                name = key[2:] if key.startswith("Q_") else key
                qs[name] = arr.astype(np.float32)
        if not qs:
            raise RuntimeError(f"No valid Q-table arrays found in '{file_path}'. Keys: {files}")
        return qs


def _pos_only_index(pos: np.ndarray, size: int) -> int:
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def get_state_index_for_agent(
    agent: Agent,
    obs: dict,
    other_agent_name: str,
    grid_size: int,
    use_joint_index: bool,
) -> int:
    """Compute the state index compatible with training-time indexing.

    If `use_joint_index` is True and the observation contains the required 'global'/'dist_agents'
    field and the helper `global_joint_state_index` is available, that will be used.
    Otherwise a fallback (position-only) index is used.
    """
    local = obs.get(agent.agent_name, {}).get("local")
    if local is None:
        local = getattr(agent, "_agent_location", np.array([0, 0]))

    # try joint-state index if available
    if use_joint_index and global_joint_state_index is not None:
        global_info = obs.get(agent.agent_name, {}).get("global", {})
        dist_agents = (global_info or {}).get("dist_agents", {})
        try:
            return int(global_joint_state_index(np.asarray(local, dtype=int), dist_agents, other_agent_name, grid_size))
        except Exception:
            LOGGER.debug("global_joint_state_index failed; falling back to pos-only", exc_info=True)

    # fallback
    return int(_pos_only_index(np.asarray(local, dtype=int), grid_size))


def choose_action_for_agent(agent: Agent, q_table: np.ndarray, s_idx: int, rng: np.random.Generator) -> int:
    """Policy for testing/visualization. Keeps the original behaviour: prey is fixed/random,
    predator picks argmax over Q-values.

    You can easily change this to use stochastic policies or to break ties randomly.
    """
    # guard index range
    if s_idx < 0 or s_idx >= q_table.shape[0]:
        # Out-of-range state index — choose a random valid action
        LOGGER.warning("State index %d out of range for q_table with shape %s; choosing random action", s_idx, q_table.shape)
        return int(rng.integers(0, q_table.shape[1]))

    if getattr(agent, "agent_type", "") == "prey":
        # original script used action 4 as the prey's behavior: keep this deterministic default
        # you can change to rng.integers(0, q_table.shape[1]) to make prey random
        return int(4)
    # predator: greedy
    return int(int(np.argmax(q_table[s_idx])))


def run_test(
    q_file: str = "baselines/IQL/iql_qs.npz",
    size: int = 8,
    episodes: int = 3,
    max_steps: int = 250,
    pause: float = 0.05,
) -> None:
    qs = try_load_qs(q_file)
    LOGGER.info("Loaded Q-tables for agents: %s", list(qs.keys()))

    prey, predator = make_agents()
    agents = [prey, predator]
    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode="human", size=size, perc_num_obstacle=10)
    rng = np.random.default_rng(0)

    # decide whether joint-state indexing is possible/likely
    use_joint_index = global_joint_state_index is not None

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            LOGGER.info("Test episode %d/%d", ep, episodes)

            for t in range(1, max_steps + 1):
                actions: Dict[str, int] = {}

                for ag in agents:
                    other_name = next(o for o in agents if o.agent_name != ag.agent_name).agent_name
                    s_idx = get_state_index_for_agent(ag, obs, other_name, size, use_joint_index)

                    # Find a matching Q-table for this agent (different key heuristics)
                    # Keys are normalized to agent_name in qs (try exact match first)
                    q_table = None
                    if ag.agent_name in qs:
                        q_table = qs[ag.agent_name]
                    else:
                        # try agent prefixes (e.g., 'prey' may be stored as 'prey_1' etc.)
                        for k in qs.keys():
                            if k.startswith(ag.agent_name):
                                q_table = qs[k]
                                break
                    if q_table is None:
                        # try 'Q_<agent_name>' key
                        if f"Q_{ag.agent_name}" in qs:
                            q_table = qs[f"Q_{ag.agent_name}"]

                    if q_table is None:
                        raise RuntimeError(f"No Q-table found for agent '{ag.agent_name}'. Available keys: {list(qs.keys())}")

                    a = choose_action_for_agent(ag, q_table, s_idx, rng)
                    actions[ag.agent_name] = a

                mgp = env.step(actions)
                obs = mgp["obs"]

                # Print reward dict for visibility (can be noisy)
                print(mgp.get("reward", {}))

                time.sleep(pause)

                if mgp.get("terminated", False):
                    print(f"Capture at step {t} (episode {ep})")
                    break

            # small inter-episode pause
            time.sleep(0.25)

    finally:
        try:
            env.close()
        except Exception:
            LOGGER.debug("Failed to close env cleanly", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test IQL-trained agents")
    p.add_argument("--file", type=str, default="baselines/IQL/iql_checkpoint.npz")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_test(q_file=args.file, size=args.size, episodes=args.episodes, max_steps=args.max_steps, pause=args.pause)
