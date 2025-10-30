"""
Central Q-Learning (tabular CQL) trainer (cleaned + bug fixes).

Key fixes / improvements
- Removed nested function definitions inside the main loop.
- Fixed joint-action / per-agent action conversions and ensured the `actions`
  dict passed to `env.step` matches agent naming/order.
- Defensive memory check before allocating the joint Q-table (helps avoid
  huge allocations that crash the process). If table is too large, an error
  is raised with actionable advice.
- Consistent and clear variable names; modular helper functions.
- Proper handling of potential-based shaping when `env.potential_reward` is
  present (expects a dict agent_name -> potential) with safe fallbacks.
- TensorBoard logging and periodic checkpoint saving.

Usage
-----
python cql_train_cleaned.py --episodes 20000 --size 8 --alpha 0.25 --gamma 0.95

"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except Exception:  # pragma: no cover - optional
    wandb = None

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

LOGGER = logging.getLogger("cql_trainer")


################ wandb setup (optional) #######################
if wandb is not None:
    try:
        wandb.init(project="MARL-Predator-Prey-Project")
    except Exception as e:
        LOGGER.warning("WandB init failed: %s", e)


# -------------------- helpers --------------------


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def joint_state_index(positions: List[Tuple[int, int]], grid_size: int) -> int:
    """Encode a list of (x,y) positions into a single integer index.

    Each agent cell index = x * grid_size + y. The combined index treats
    those cell indices as digits in base `n_cells`.
    """
    n_cells = grid_size * grid_size
    idx = 0
    for cell_pos in positions:
        cell_index = int(cell_pos[0]) * grid_size + int(cell_pos[1])
        idx = idx * n_cells + cell_index
    return int(idx)


def joint_actions_to_index(actions: List[int], n_actions: int) -> int:
    """Convert list of per-agent actions into flat joint-action index."""
    idx = 0
    for a in actions:
        idx = idx * n_actions + int(a)
    return int(idx)


def index_to_joint_actions(idx: int, n_agents: int, n_actions: int) -> List[int]:
    """Convert flat joint-action index back to per-agent actions (same order)."""
    acts = [0] * n_agents
    for i in range(n_agents - 1, -1, -1):
        acts[i] = int(idx % n_actions)
        idx //= n_actions
    return acts


def make_agents(num_predators: int = 2, num_preys: int = 2) -> List[Agent]:
    agents: List[Agent] = []
    for i in range(1, num_preys + 1):
        agents.append(Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey"))
    for i in range(1, num_predators + 1):
        agents.append(
            Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator")
        )
    return agents


def make_env_and_meta(
    agents: List[Agent], grid_size: int, seed: int
) -> Tuple[GridWorldEnv, int, int]:
    env = GridWorldEnv(
        agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=10, seed=seed
    )
    n_cells = grid_size * grid_size

    # total joint states = n_cells ** n_agents (may be very large)
    n_states = n_cells ** len(agents)
    n_actions = env.action_space.n
    return env, n_states, n_actions


def estimate_table_bytes(n_states: int, n_joint_actions: int, dtype=np.float32) -> int:
    return int(n_states) * int(n_joint_actions) * np.dtype(dtype).itemsize


def init_joint_q_table(
    n_states: int, n_joint_actions: int, max_bytes: int | None = None
) -> np.ndarray:
    """Create joint Q table; optionally check memory requirement first."""
    needed = estimate_table_bytes(n_states, n_joint_actions)
    if max_bytes is not None and needed > max_bytes:
        raise MemoryError(
            f"Joint Q-table requires {needed/(1024**3):.2f} GiB > allowed {max_bytes/(1024**3):.2f} GiB"
        )
    return np.zeros((n_states, n_joint_actions), dtype=np.float32)


def save_q_table(path: str, Q: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, central=Q)
    LOGGER.info("Saved joint Q-table -> %s", path)


# -------------------- training loop --------------------


def train(
    episodes: int = 5000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.99,
    save_path: str = "baselines/CQL",
    grid_size: int = 8,
    num_predators: int = 1,
    num_preys: int = 1,
    seed: int = 0,
    max_table_bytes: int | None = None,
):
    """Train a central (joint) Q-table using tabular Q-learning.

    Parameters
    - max_table_bytes: if provided, refuse to allocate a table larger than this.
    """
    rng = np.random.default_rng(seed)

    agents = make_agents(num_predators=num_predators, num_preys=num_preys)
    agent_names = [ag.agent_name for ag in agents]
    n_agents = len(agent_names)

    env, n_states, n_actions = make_env_and_meta(agents, grid_size, seed)

    n_agents = len(agent_names)

    # joint-action space size
    n_joint_actions = n_actions**n_agents

    # memory safety check (use a conservative default if not provided)
    if max_table_bytes is None:
        # set default max to 8 GiB for safety on typical dev machines
        max_table_bytes = 16 * 1024**3

    LOGGER.info(
        "Allocating joint Q-table: states=%d, joint_actions=%d",
        n_states,
        n_joint_actions,
    )
    Q = init_joint_q_table(n_states, n_joint_actions, max_bytes=max_table_bytes)

    save_path_Q = os.path.join(
        os.path.dirname(save_path) or ".", "central_cql_q_table.npz"
    )

    eps = eps_start

    rewards_per_ep = {name: [] for name in agent_names}
    episode_lengths: List[int] = []
    captures_per_ep: List[int] = []

    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(os.path.dirname(save_path) or ".", "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    window = 100

    # Prepare index->action tensor shape for later use
    action_shape = (n_actions,) * n_agents

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_len = max_steps

        total_reward_per_agent = {name: 0.0 for name in agent_names}

        for t in range(max_steps):
            # compute joint-state index
            positions = [tuple(obs[name]["local"]) for name in agent_names]
            s = joint_state_index(positions, grid_size)

            # select actions by marginalizing the joint-Q over others
            flat_row = Q[s]
            if flat_row.size != n_joint_actions:
                raise ValueError(
                    "Unexpected joint-Q row length: %d != %d"
                    % (flat_row.size, n_joint_actions)
                )

            q_tensor = flat_row.reshape(
                action_shape
            )  # shape: (n_actions, n_actions, ...)

            # compute marginal per-agent action-values by averaging over other axes
            q_vals_per_agent = []
            for i in range(n_agents):
                axes_to_avg = tuple(j for j in range(n_agents) if j != i)
                q_vals_i = q_tensor.mean(axis=axes_to_avg)
                q_vals_per_agent.append(q_vals_i)

            # epsilon-greedy per-agent
            chosen_actions: List[int] = []
            for i in range(n_agents):
                if rng.random() < eps:
                    a_i = int(rng.integers(0, n_actions))
                else:
                    row = np.asarray(q_vals_per_agent[i])
                    best = float(np.max(row))
                    best_actions = (
                        np.flatnonzero(np.isclose(row, best)).astype(int).tolist()
                    )
                    a_i = int(rng.choice(best_actions))
                chosen_actions.append(a_i)

            joint_idx = joint_actions_to_index(chosen_actions, n_actions)

            # build actions dict for env.step
            actions = {
                agents[i].agent_name: int(chosen_actions[i]) for i in range(n_agents)
            }

            mgp = env.step(actions)
            next_obs, rewards = mgp["obs"], mgp["reward"]

            # accumulate per-agent rewards and compute central reward
            central_r = 0.0
            current_pot_sum = 0.0
            next_pot_sum = 0.0

            # potential-based shaping: try to obtain dicts from env.potential_reward
            try:
                current_state = {n: obs[n]["local"] for n in agent_names}
                next_state = {n: next_obs[n]["local"] for n in agent_names}
                current_pot = env.potential_reward(current_state)
                next_pot = env.potential_reward(next_state)
            except Exception:
                current_pot = {n: 0.0 for n in agent_names}
                next_pot = {n: 0.0 for n in agent_names}

            for name in agent_names:
                r = float(rewards.get(name, 0.0))
                total_reward_per_agent[name] += r
                central_r += r

                # accumulate potentials (defensive: missing keys => 0)
                current_pot_sum += float(current_pot.get(name, 0.0))
                next_pot_sum += float(next_pot.get(name, 0.0))

            # next state index
            next_positions = [tuple(next_obs[name]["local"]) for name in agent_names]
            s2 = joint_state_index(next_positions, grid_size)

            # CQL update (centralized TD update)
            td_target = (
                central_r
                + (gamma * next_pot_sum)
                - current_pot_sum
                + gamma * np.max(Q[s2])
            )
            td_error = td_target - Q[s, joint_idx]
            Q[s, joint_idx] += alpha * td_error

            if mgp.get("terminated", False):
                ep_len = t + 1
                break

            obs = next_obs

        # end episode
        for name in agent_names:
            rewards_per_ep[name].append(total_reward_per_agent[name])

        episode_lengths.append(ep_len)
        captures_this_episode = int(getattr(env, "_captures_total", 0))
        captures_per_ep.append(captures_this_episode)

        # TensorBoard logging
        writer.add_scalar("episode/length", ep_len, ep)
        writer.add_scalar("episode/captures", captures_this_episode, ep)

        for name in agent_names:
            writer.add_scalar(
                f"episode/total_reward/{name}", float(total_reward_per_agent[name]), ep
            )
            mean_reward_running = (
                float(np.mean(rewards_per_ep[name][-window:]))
                if rewards_per_ep[name]
                else 0.0
            )
            writer.add_scalar(f"mean/{name}/reward", mean_reward_running, ep)

        mean_captures_running = (
            float(np.mean(captures_per_ep[-window:])) if captures_per_ep else 0.0
        )
        writer.add_scalar("mean/captures", mean_captures_running, ep)

        # epsilon decay and logs
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)
            avg_per_agent = {
                name: (
                    np.mean(rewards_per_ep[name][-100:])
                    if rewards_per_ep[name]
                    else 0.0
                )
                for name in agent_names
            }
            LOGGER.info(
                "Ep %d | eps=%.3f | averages(last100)=%s | mean captures(last100)=%.2f",
                ep,
                eps,
                ", ".join([f"{n}={v:.2f}" for n, v in avg_per_agent.items()]),
                mean_captures_running,
            )

        if ep % 10 == 0:
            writer.flush()

        if ep % 1000 == 0:
            try:
                save_q_table(save_path_Q, Q)
            except Exception as e:
                LOGGER.warning("Failed saving checkpoint at ep %d: %s", ep, e)

    # final save and cleanup
    try:
        save_q_table(save_path_Q, Q)
    except Exception as e:
        LOGGER.warning("Final save failed: %s", e)

    writer.close()
    LOGGER.info("Training finished. Final epsilon=%.3f", eps)


# ---------------- CLI ----------------


def parse_args():
    p = argparse.ArgumentParser("Train central CQL (tabular)")
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--size", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="baselines/CQL/")
    p.add_argument("--predators", type=int, default=2)
    p.add_argument("--preys", type=int, default=2)
    p.add_argument(
        "--max-table-gb",
        type=float,
        default=16.0,
        help="Max allowed joint-Q memory in GiB before aborting",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    max_table_bytes = int(args.max_table_gb * 1024**3) if args.max_table_gb else None
    try:
        train(
            episodes=args.episodes,
            grid_size=args.size,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
            save_path=args.save_path,
            num_predators=args.predators,
            num_preys=args.preys,
            max_table_bytes=max_table_bytes,
        )
    except MemoryError as me:
        LOGGER.error("MemoryError: %s", me)
        sys.exit(2)
    except Exception as e:
        LOGGER.exception("Training failed: %s", e)
        sys.exit(1)
