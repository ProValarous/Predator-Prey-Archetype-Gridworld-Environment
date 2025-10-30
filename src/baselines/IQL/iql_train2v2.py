"""
Tabular IQL trainer for multi-agent predator-prey
(supports 2 predators + 2 preys).

This module implements independent Q-learning (IQL) with a tabular
state-action representation. Each agent maintains its own Q-table.
The joint state is encoded by concatenating each agent's cell index
(x * G + y) as digits in base n_cells. The number of states grows as
(n_cells) ** n_agents and can become very large; use small grids.

Features
- Per-agent Q-tables (independent learners)
- Epsilon-greedy exploration
- Optional potential-based shaping if the environment provides
    potential_reward
- TensorBoard and wandb logging for episode metrics
- Periodic checkpointing of per-agent Q-tables

Usage
    cd src
    python -m baselines.IQL.iql_train2v2 --episodes 20000 --size 6
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

from .utils import create_experiment_dir, write_experiment_md

LOGGER = logging.getLogger("iql_trainer")


# ---------------- wandb (optional) ----------------
wandb.init(project="MARL-Predator-Prey-Project")
wandb_path = "baselines/IQL/logs/"
wandb.tensorboard.patch(root_logdir=wandb_path)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


# ---------------- utilities ----------------
def joint_state_index(positions: List[Tuple[int, int]], grid_size: int) -> int:
    """Encode a list of (x,y) positions into a single integer index.

    Each agent's cell index = x * grid_size + y. The combined index treats
    those cell indices as digits in base `n_cells`.
    """
    n_cells = grid_size * grid_size
    idx = 0
    for cell_pos in positions:
        cell_index = int(cell_pos[0]) * grid_size + int(cell_pos[1])
        idx = idx * n_cells + cell_index
    return int(idx)


def make_agents(num_predators: int = 2, num_preys: int = 2) -> List[Agent]:
    """
    Create a list of Agent objects: num_preys prey and num_predators predator.

    Agent naming convention: prey_1, prey_2, predator_1, predator_2, ...
    """
    agents: List[Agent] = []
    for i in range(1, num_preys + 1):
        agents.append(
            Agent(
                agent_name=f"prey_{i}",
                agent_team=i,
                agent_type="prey",
            )
        )
    for i in range(1, num_predators + 1):
        agents.append(
            Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator")
        )
    return agents


def make_env_and_meta(
    agents: List[Agent], grid_size: int, seed: int
) -> Tuple[GridWorldEnv, int, int]:
    """Create the GridWorld env and return (env, n_states, n_actions).

    n_states is the joint state space size: (grid_size * grid_size) ** n_agents
    """
    env = GridWorldEnv(
        agents=agents,
        render_mode=None,
        size=grid_size,
        perc_num_obstacle=10,
        seed=seed,
    )

    n_cells = grid_size * grid_size
    n_states = n_cells ** len(agents)
    n_actions = env.action_space.n

    return env, n_states, n_actions


def init_q_tables(
    agent_names: List[str],
    n_states: int,
    n_actions: int,
) -> Dict[str, np.ndarray]:
    """Initialize per-agent Q-tables

    Each Q-table is a zero-initialized array of shape (n_states, n_actions)
    """
    q_tables: Dict[str, np.ndarray] = {}
    for name in agent_names:
        q_tables[name] = np.zeros((n_states, n_actions), dtype=np.float32)
    return q_tables


def epsilon_greedy_action(
    q_row: np.ndarray, n_actions: int, rng: np.random.Generator, eps: float
) -> int:
    """
    Returns a random action based on the value of epsilon
    """
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    return int(int(np.argmax(q_row)))


def save_q_table(path: str, Q: np.ndarray) -> None:
    """
    Exports Q table as an .npz file
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, Q=Q)
    LOGGER.info("Saved Q-table -> %s", path)


# ---------------- training ----------------
def train(
    episodes: int = 5000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.99,
    save_path: str = "baselines/IQL/",
    grid_size: int = 8,
    num_predators: int = 2,
    num_preys: int = 2,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    agents = make_agents(num_predators=num_predators, num_preys=num_preys)
    agent_names = [ag.agent_name for ag in agents]

    env, n_states, n_actions = make_env_and_meta(agents, grid_size, seed)

    # Initialize Q tables per agent
    Qs = init_q_tables(agent_names, n_states, n_actions)

    # Prepare save paths per agent
    save_path_Q = {
        name: os.path.join(os.path.dirname(save_path) or ".", name, "iql_q_table.npz")
        for name in agent_names
    }

    eps = eps_start

    # bookkeeping for TensorBoard
    per_agent_rewards: Dict[str, List[float]] = {name: [] for name in agent_names}
    captures_per_ep: List[int] = []
    episode_lengths: List[int] = []

    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")

    # create an experiment folder (checkpoints + logs) and use it
    exp_dir, checkpoints_dir, logs_dir = create_experiment_dir(
        base=os.path.dirname(save_path) or ".", name="iql_run"
    )

    # write a short README.md describing this run
    params = {
        "command": f"python -m baselines.IQL.iql_train2v2\
                    --size {grid_size} --episodes {episodes}",
        "seed": seed,
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "num_predators": num_predators,
        "num_preys": num_preys,
    }
    write_experiment_md(exp_dir, params)

    log_dir = os.path.join(logs_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # prepare per-agent checkpoint paths under the experiment checkpoints dir
    save_path_Q = {
        name: os.path.join(checkpoints_dir, name, "iql_q_table.npz")
        for name in agent_names
    }

    window = 100

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()

        # initialize per-episode accumulators
        total_reward = {name: 0.0 for name in agent_names}
        ep_len = max_steps

        for t in range(max_steps):
            # build ordered positions list matching agent order
            # used to create Qs
            positions = [tuple(obs[name]["local"]) for name in agent_names]
            s = joint_state_index(positions, grid_size)

            # select actions for every agent (independent epsilon-greedy)
            actions: Dict[str, int] = {}
            chosen_actions: Dict[str, int] = {}
            for name in agent_names:
                q_row = Qs[name][s]
                a = epsilon_greedy_action(q_row, n_actions, rng, eps)
                chosen_actions[name] = a
                actions[name] = a

            mgp = env.step(actions)
            next_obs, rewards = mgp["obs"], mgp["reward"]

            # accumulate rewards and prepare next-state index
            next_positions = []
            for name in agent_names:
                next_positions.append(tuple(next_obs[name]["local"]))
            s2 = joint_state_index(next_positions, grid_size)

            # potential shaping
            # use dicts; fall back to zero per-agent if not provided
            current_state = {n: obs[n]["local"] for n in agent_names}
            next_state = {n: next_obs[n]["local"] for n in agent_names}
            potential_fn = getattr(env, "potential_reward", None)
            if callable(potential_fn):
                current_pot = potential_fn(current_state)
                next_pot = potential_fn(next_state)
            else:
                current_pot = {n: 0.0 for n in agent_names}
                next_pot = {n: 0.0 for n in agent_names}

            for name in agent_names:
                r = rewards[name]
                total_reward[name] += r

                # Q update (IQL)
                current_q = Qs[name][s, chosen_actions[name]]
                next_q_max = float(np.max(Qs[name][s2]))
                target = (
                    r + gamma * next_pot[name] - current_pot[name] + gamma * next_q_max
                )
                Qs[name][s, chosen_actions[name]] = Qs[name][
                    s, chosen_actions[name]
                ] + alpha * (target - current_q)

            if mgp.get("terminated", False):
                ep_len = t + 1
                break

            obs = next_obs

        # episode bookkeeping
        for name in agent_names:
            per_agent_rewards[name].append(total_reward[name])
            # writer.add_scalar(
            #   f"episode/total_reward/{name}",
            #   total_reward[name],
            #   ep
            # )

        episode_lengths.append(ep_len)

        captures_this_episode = int(getattr(env, "_captures_total", 0))
        captures_per_ep.append(captures_this_episode)

        # TensorBoard scalars (one-per-episode)
        writer.add_scalar("episode/length", ep_len, ep)
        # writer.add_scalar("episode/captures", captures_this_episode, ep)

        # running means per-agent
        for name in agent_names:
            if per_agent_rewards[name]:
                mean_reward_running = float(np.mean(per_agent_rewards[name][-window:]))
            else:
                mean_reward_running = 0.0
            writer.add_scalar(f"mean/{name}/reward", mean_reward_running, ep)

        if captures_per_ep:
            mean_captures_running = float(np.mean(captures_per_ep[-window:]))
        else:
            mean_captures_running = 0.0

        writer.add_scalar("mean/captures", mean_captures_running, ep)

        # decay epsilon periodically
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)
            avg_str = ", ".join(
                f"{name}={np.mean(per_agent_rewards[name][-100:]):.2f}"
                for name in agent_names
            )

            LOGGER.info(
                "Ep %d | eps=%.3f | avg(last100) %s | " "mean captures(last100)=%.2f",
                ep,
                eps,
                avg_str,
                mean_captures_running,
            )

        # periodic flush & save
        if ep % 10 == 0:
            writer.flush()
        if ep % 1000 == 0:
            for name in agent_names:
                save_q_table(save_path_Q[name], Qs[name])

    # final save
    for name in agent_names:
        save_q_table(save_path_Q[name], Qs[name])

    writer.close()
    LOGGER.info("Training done. Final epsilon=%.3f", eps)


# ---------------- CLI ----------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train multi-agent IQL tabular")
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="baselines/IQL/")
    p.add_argument("--predators", type=int, default=2, help="Number of predators")
    p.add_argument("--preys", type=int, default=2, help="Number of preys")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    train(
        episodes=args.episodes,
        grid_size=args.size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        save_path=args.save_path,
        num_predators=args.predators,
        num_preys=args.preys,
    )
