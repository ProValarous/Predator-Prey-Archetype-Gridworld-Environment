"""
Simplified tester / visualizer for tabular IQL Q-tables (discrete grid state only).

This version assumes that the Q-table was trained with predator's state = cell index (x*size + y).
Prey is fixed to action 4 (no-op).

Usage:
python test_iql_cleaned.py --file baselines/IQL/iql_qs.npz --size 8 --episodes 5 --pause 0.05
"""

import argparse
import logging
import time
from typing import Dict

import numpy as np

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

LOGGER = logging.getLogger("test_iql")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


def make_agents() -> tuple[Agent, Agent]:
    prey = Agent("prey", 1, "prey")
    predator = Agent("predator", 1, "predator")
    return prey, predator


def try_load_qs(file_path: str) -> Dict[str, np.ndarray]:
    with np.load(file_path, allow_pickle=False) as data:
        qs: Dict[str, np.ndarray] = {}
        for key in data.files:
            arr = data[key]
            if arr.ndim == 2:
                name = key[2:] if key.startswith("Q_") else key
                qs[name] = arr.astype(np.float32)
        if not qs:
            raise RuntimeError(f"No valid Q-table arrays found in '{file_path}'. Keys: {list(data.files)}")
        return qs

def state_index_from_obs(obs: dict, predator: Agent, prey: Agent, size: int) -> int:
    """
    Build state index using predator + prey positions.
    Maps (pred_x, pred_y, prey_x, prey_y) -> unique integer index.
    """
    pos_pred = obs[predator.agent_name]["local"]
    pos_prey = obs[prey.agent_name]["local"]

    pred_x, pred_y = int(pos_pred[0]), int(pos_pred[1])
    prey_x, prey_y = int(pos_prey[0]), int(pos_prey[1])

    return (
        pred_x * size * size * size
        + pred_y * size * size
        + prey_x * size
        + prey_y
    )


def choose_action(agent: Agent, q_table: np.ndarray, s_idx: int) -> int:
    if getattr(agent, "agent_type", "") == "prey":
        return 4  # fixed no-op
    if s_idx < 0 or s_idx >= q_table.shape[0]:
        LOGGER.warning("State index %d out of range; choosing random", s_idx)
        return int(np.random.randint(0, q_table.shape[1]))
    return int(np.argmax(q_table[s_idx]))


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
    env = GridWorldEnv(agents=agents, render_mode="human", size=size, perc_num_obstacle=10)

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            LOGGER.info("Test episode %d/%d", ep, episodes)

            for t in range(1, max_steps + 1):
                actions: Dict[str, int] = {}

                # build joint state index once (predator + prey)
                s_idx = state_index_from_obs(obs, predator, prey, size)

                for ag in agents:
                    if ag.agent_type.startswith("prey"):
                        # Prey is fixed policy (noop), no Q-table needed
                        a = env.action_space.sample()
                        actions[ag.agent_name] = a
                        continue

                    # Predator uses Q-table
                    q_table = qs.get(ag.agent_name)
                    if q_table is None:
                        # fall back to key matching
                        for k in qs.keys():
                            if k.startswith(ag.agent_name) or k.endswith(ag.agent_name):
                                q_table = qs[k]
                                break
                    if q_table is None:
                        raise RuntimeError(f"No Q-table for agent '{ag.agent_name}'. Keys: {list(qs.keys())}")

                    a = choose_action(ag, q_table, s_idx)
                    actions[ag.agent_name] = a

                mgp = env.step(actions)
                obs = mgp["obs"]
                print(mgp.get("reward", {}))
                time.sleep(pause)

                if mgp.get("terminated", False):
                    print(f"Capture at step {t} (episode {ep})")
                    break

            time.sleep(0.25)

    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test IQL-trained agents (discrete state)")
    p.add_argument("--file", type=str, default="baselines/IQL/iql_qs.npz")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--pause", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    run_test(q_file=args.file, size=args.size, episodes=args.episodes, max_steps=args.max_steps, pause=args.pause)
