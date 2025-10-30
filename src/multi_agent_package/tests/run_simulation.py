"""
GridWorld demo runner — cleaned, modular and documented version of your simulation script.

Features
- Agent factory (build_agents) separated from simulation runner
- Auto-detection of total subteams to set `agent.total_subteams`
- Pluggable policy interface (default: random policy using env.action_space)
- Command-line options for steps, grid size, obstacle percentage, render mode, and policy
- Robust printing using optional helper functions if `multi_agent_package.helpers.helper` provides them
- Proper resource cleanup and logging

Usage
-----
python gridworld_demo_cleaned.py --steps 200 --size 8 --obst 10 --mode human --policy random

"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from multi_agent_package.agents import Agent
from multi_agent_package.gridworld import GridWorldEnv

# Optional helpers for pretty-printing step info
try:
    from multi_agent_package.helpers.helper import print_action, print_mgp_info  # type: ignore
except Exception:
    print_action = None  # type: ignore
    print_mgp_info = None  # type: ignore


LOGGER = logging.getLogger("gridworld_demo")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


# ----------------------
# Agent factory
# ----------------------


def build_agents() -> List[Agent]:
    """Create and return a list of Agent instances for the demo.

    Customize or replace this factory to configure different scenarios.

    Returns
    -------
    List[Agent]
        List of agents in the environment.
    """
    # Example agents with team IDs and human-friendly names
    agent101 = Agent("predator", 1, "PY101_Tom")
    agent102 = Agent("prey", 2, "PY102_Garfield")

    agent201 = Agent("prey", 3, "PY201_Jerry")
    agent202 = Agent("predator", 1, "PD202_Stuart")

    return [agent201, agent202, agent101, agent102]


# ----------------------
# Utilities
# ----------------------


def assign_total_subteams(agents: List[Agent]) -> None:
    """Auto-detect total subteams per base type and set `agent.total_subteams`.

    This helps the environment render distinct colors/shapes for subteams.
    """
    counts: Dict[str, int] = {}
    for ag in agents:
        try:
            base, sub_id = ag._parse_team()
            sub_id = int(sub_id)
        except Exception:
            base = ag.agent_type
            sub_id = int(ag.agent_team) if isinstance(ag.agent_team, int) else 1
        counts[base] = max(counts.get(base, 0), sub_id)

    for ag in agents:
        try:
            base, _ = ag._parse_team()
        except Exception:
            base = ag.agent_type
        ag.total_subteams = counts.get(base, 1)


# ----------------------
# Policies
# ----------------------


def random_policy(
    agents: List[Agent],
    obs: Dict[str, dict],
    rng: np.random.Generator,
    env: GridWorldEnv,
) -> Dict[str, int]:
    """Return a random action for each agent using the environment action_space."""
    return {ag.agent_name: int(env.action_space.sample()) for ag in agents}


def noop_policy(
    agents: List[Agent],
    obs: Dict[str, dict],
    rng: np.random.Generator,
    env: GridWorldEnv,
) -> Dict[str, int]:
    """Return a 'no-op' (4) action for each agent. Useful for debugging.

    Note: ensure action 4 is a valid no-op in your environment.
    """
    return {ag.agent_name: 4 for ag in agents}


# ----------------------
# Simulation runner
# ----------------------


def run_simulation(
    agents: List[Agent],
    size: int = 10,
    perc_num_obstacle: float = 10.0,
    render_mode: str = "human",
    steps: int = 100,
    seed: int = 0,
    pause: float = 0.01,
) -> None:
    """Run a simple simulation loop.

    Parameters
    ----------
    agents : list[Agent]
        Agents participating in the environment.
    size : int
        Grid size (size x size).
    perc_num_obstacle : float
        Percent of grid occupied by obstacles.
    render_mode : str
        'human' for pygame window or 'rgb_array' to get frames returned.
    steps : int
        Number of simulation steps to run.
    seed : int
        RNG seed for reproducibility.
    policy : callable
        Function that maps (agents, obs, rng, env) -> actions dict.
    pause : float
        Sleep time between frames when `render_mode=='human'`.
    """
    assign_total_subteams(agents)

    env = GridWorldEnv(
        agents=agents,
        render_mode=render_mode,
        size=size,
        perc_num_obstacle=perc_num_obstacle,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    for t in range(steps):
        actions = {
            agents[0].agent_name: env.action_space.sample(),
            agents[1].agent_name: env.action_space.sample(),
            agents[2].agent_name: env.action_space.sample(),
            agents[3].agent_name: env.action_space.sample(),
        }

        mgp_tuple = env.step(actions)

        # Pretty-print step info if helpers exist
        if print_mgp_info is not None:
            try:
                print_mgp_info(mgp_tuple, t, obs, actions)
            except Exception:
                LOGGER.debug("print_mgp_info failed", exc_info=True)
        else:
            # Minimal fallback print
            print(f"Step {t}: actions={actions}, reward={mgp_tuple.get('reward')}")

        # Optional print of helper metric fields
        print(f"Total captures this episode: {getattr(env, '_captures_total', 'N/A')}")
        print(f"Captures this step : {getattr(env, '_captures_this_step', 'N/A')}")

        if mgp_tuple.get("terminated"):
            print("Episode terminated at step", t)
            break

        obs = mgp_tuple["obs"]

        if render_mode == "human":
            time.sleep(pause)

    try:
        env.close()
    except Exception:
        LOGGER.debug("Failed to close environment cleanly", exc_info=True)
    print("Environment closed.")


# ----------------------
# CLI
# ----------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GridWorldEnv demo")
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of simulation steps"
    )
    parser.add_argument("--size", type=int, default=8, help="Grid size (NxN)")
    parser.add_argument(
        "--obst", type=float, default=10.0, help="Percentage of obstacles"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.01,
        help="Pause between frames when running in human mode",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    agents = build_agents()

    run_simulation(
        agents=agents,
        size=args.size,
        perc_num_obstacle=args.obst,
        render_mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
