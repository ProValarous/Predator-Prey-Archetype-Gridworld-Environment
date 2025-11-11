"""
Hybrid trainer: allow choosing IQL or CQL separately for predators and preys.

- Team-level choice via CLI: --predator-algo {iql,cql} --prey-algo {iql,cql}
- Keeps data structures dictionary-based so it's easy to extend and inspect.
- Supports mixtures: e.g. predators use CQL (centralized among predators) while
  preys use IQL (independent tabular Qs), and vice-versa. If both teams use CQL
  the script builds two separate central Q-tables (one per team).

Design notes
- The "state" used for indexing Qs is the full joint-state (positions of all
  agents) encoded exactly like in the original trainers (base-n_cells digits).
  This keeps the effect of other agents' positions visible to all learners.
- IQL agents keep per-agent Q tables indexed by full joint-state s.
- For each team that chose CQL we allocate a central Q over the full
  joint-state but joint-actions only cover that team's agents.
- Updates and action selection follow the previous scripts' logic with shaping
  fallback and epsilon-greedy behaviour.

Usage example
--------------
python mixed_trainer.py --predator-algo cql --prey-algo iql\
                        --episodes 20000 --size 6

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
except Exception:  # optional
    wandb = None

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

LOGGER = logging.getLogger("mixed_trainer")


# ----------------- utility helpers -----------------


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def joint_state_index(positions: List[Tuple[int, int]], grid_size: int) -> int:
    """Encode a list of (x,y) positions into a single integer index.

    Same encoding as the original trainers: treat each agent's cell index
    (x*grid_size + y) as a digit in base `n_cells`.
    """
    n_cells = grid_size * grid_size
    idx = 0
    for cell_pos in positions:
        cell_index = int(cell_pos[0]) * grid_size + int(cell_pos[1])
        idx = idx * n_cells + cell_index
    return int(idx)


def joint_actions_to_index(actions: List[int], n_actions: int) -> int:
    idx = 0
    for a in actions:
        idx = idx * n_actions + int(a)
    return int(idx)


def index_to_joint_actions(idx: int, n_agents: int, n_actions: int) -> List[int]:
    acts = [0] * n_agents
    for i in range(n_agents - 1, -1, -1):
        acts[i] = int(idx % n_actions)
        idx //= n_actions
    return acts


def estimate_table_bytes(n_states: int, n_joint_actions: int, dtype=np.float32) -> int:
    return int(n_states) * int(n_joint_actions) * np.dtype(dtype).itemsize


def init_joint_q_table(
    n_states: int, n_joint_actions: int, max_bytes: int | None = None
) -> np.ndarray:
    needed = estimate_table_bytes(n_states, n_joint_actions)
    if max_bytes is not None and needed > max_bytes:
        raise MemoryError(
            f"Joint Q-table requires {needed / (1024**3):.2f} GiB "
            f"> allowed {max_bytes / (1024**3):.2f} GiB"
        )
    return np.zeros((n_states, n_joint_actions), dtype=np.float32)


def init_q_tables(
    agent_names: List[str], n_states: int, n_actions: int
) -> Dict[str, np.ndarray]:
    return {
        name: np.zeros((n_states, n_actions), dtype=np.float32) for name in agent_names
    }


def save_q_table(path: str, Q: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, Q=Q)
    LOGGER.info("Saved Q-table -> %s", path)


def epsilon_greedy_action(
    q_row: np.ndarray, n_actions: int, rng: np.random.Generator, eps: float
) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    return int(int(np.argmax(q_row)))


# ----------------- environment / agents -----------------


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
    n_states = n_cells ** len(agents)
    n_actions = env.action_space.n
    return env, n_states, n_actions


# ----------------- mixed training -----------------


def train(
    episodes: int = 5000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.99,
    save_path: str = "baselines/mixed/",
    grid_size: int = 8,
    num_predators: int = 2,
    num_preys: int = 2,
    predator_algo: str = "iql",
    prey_algo: str = "cql",
    seed: int = 0,
    max_table_bytes: int | None = None,
) -> None:
    rng = np.random.default_rng(seed)

    agents = make_agents(num_predators=num_predators, num_preys=num_preys)
    agent_names = [ag.agent_name for ag in agents]

    env, n_states, n_actions = make_env_and_meta(agents, grid_size, seed)

    # mapping: agent_name -> algorithm
    agent_algo: Dict[str, str] = {}
    group_agents: Dict[str, List[str]] = {"prey": [], "predator": []}
    for ag in agents:
        if ag.agent_type == "prey":
            agent_algo[ag.agent_name] = prey_algo.lower()
            group_agents["prey"].append(ag.agent_name)
        else:
            agent_algo[ag.agent_name] = predator_algo.lower()
            group_agents["predator"].append(ag.agent_name)

    # Create IQL per-agent Q-tables for all agents that chose IQL
    iql_agent_names = [n for n, algo in agent_algo.items() if algo == "iql"]
    if iql_agent_names:
        Q_iql: Dict[str, np.ndarray] = init_q_tables(
            iql_agent_names,
            n_states,
            n_actions,
        )
    else:
        Q_iql: Dict[str, np.ndarray] = {}

    # For each team that chose CQL create a central Q-table
    # that covers full joint-state
    cql_groups = {
        g: group_agents[g]
        for g in ("prey", "predator")
        if (g == "prey" and prey_algo == "cql")
        or (g == "predator" and predator_algo == "cql")
    }
    Q_cql: Dict[str, np.ndarray] = {}
    for group_name, names in cql_groups.items():
        n_group_actions = n_actions ** len(names)
        if max_table_bytes is None:
            # default 16 GiB limit per group
            max_table_bytes_group = 16 * 1024**3
        else:
            max_table_bytes_group = max_table_bytes
        LOGGER.info(
            "Allocating central Q for group '%s' " "(agents=%d, joint_actions=%d)",
            group_name,
            len(names),
            n_group_actions,
        )
        Q_cql[group_name] = init_joint_q_table(
            n_states, n_group_actions, max_bytes=max_table_bytes_group
        )

    # Prepare save paths
    save_dir = os.path.dirname(save_path) or "."
    save_paths = {
        name: os.path.join(save_dir, name, "iql_q_table.npz")
        for name in iql_agent_names
    }
    for group_name in Q_cql.keys():
        save_paths[f"central_{group_name}"] = os.path.join(
            save_dir, f"central_{group_name}_cql_q_table.npz"
        )

    eps = eps_start

    # bookkeeping
    per_agent_rewards: Dict[str, List[float]] = {}
    for name in agent_names:
        per_agent_rewards[name] = []
    captures_per_ep: List[int] = []
    episode_lengths: List[int] = []

    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(save_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    window = 100

    # Precompute group action shapes for marginalization
    group_action_shapes: Dict[str, tuple] = {}
    for group_name, names in cql_groups.items():
        group_action_shapes[group_name] = (n_actions,) * len(names)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_len = max_steps
        total_reward = {name: 0.0 for name in agent_names}

        for t in range(max_steps):
            positions = [tuple(obs[name]["local"]) for name in agent_names]
            s = joint_state_index(positions, grid_size)

            # Choose actions for IQL agents (independent)
            actions: Dict[str, int] = {}

            for name in iql_agent_names:
                q_row = Q_iql[name][s]
                a = epsilon_greedy_action(q_row, n_actions, rng, eps)
                actions[name] = a

            # Choose actions for each CQL group by marginalizing central Q
            for group_name, names in cql_groups.items():
                Qg = Q_cql[group_name]
                flat_row = Qg[s]
                n_group_actions = flat_row.size
                expected = math.prod(group_action_shapes[group_name])
                if n_group_actions != expected:
                    raise ValueError(
                        f"Central Q row size mismatch for group {group_name}:"
                        f" {n_group_actions} != {expected}"
                    )

                q_tensor = flat_row.reshape(group_action_shapes[group_name])

                # compute marginal per-agent action-values by
                # averaging over other axes
                q_vals_per_agent: List[np.ndarray] = []
                n_group_agents = len(names)
                for i in range(n_group_agents):
                    axes_to_avg = tuple(j for j in range(n_group_agents) if j != i)
                    q_vals_i = q_tensor.mean(axis=axes_to_avg)
                    q_vals_per_agent.append(q_vals_i)

                # epsilon-greedy per-agent within the group
                chosen_actions_group: List[int] = []
                for i, name in enumerate(names):
                    if rng.random() < eps:
                        a_i = int(rng.integers(0, n_actions))
                    else:
                        row = np.asarray(q_vals_per_agent[i])
                        best = float(np.max(row))
                        best_actions = (
                            np.flatnonzero(np.isclose(row, best)).astype(int).tolist()
                        )
                        a_i = int(rng.choice(best_actions))
                    chosen_actions_group.append(a_i)
                    actions[name] = a_i

            # As a final safety (shouldn't happen), ensure
            # actions provided for all agents
            for name in agent_names:
                if name not in actions:
                    # fallback to random
                    actions[name] = int(rng.integers(0, n_actions))

            mgp = env.step(actions)
            next_obs, rewards = mgp["obs"], mgp["reward"]

            # compute potentials defensively
            try:
                current_state = {n: obs[n]["local"] for n in agent_names}
                next_state = {n: next_obs[n]["local"] for n in agent_names}
                current_pot = env.potential_reward(current_state)
                next_pot = env.potential_reward(next_state)
            except Exception:
                current_pot = {n: 0.0 for n in agent_names}
                next_pot = {n: 0.0 for n in agent_names}

            # accumulate rewards and perform updates
            next_positions = [tuple(next_obs[name]["local"]) for name in agent_names]
            s2 = joint_state_index(next_positions, grid_size)

            # 1) Update IQL agents
            for name in iql_agent_names:
                r = float(rewards.get(name, 0.0))
                total_reward[name] += r
                # IQL update (same shaped TD used previously)
                td_target = (
                    r
                    + (gamma * float(np.max(Q_iql[name][s2])))
                    + (gamma * next_pot.get(name, 0.0))
                    - current_pot.get(name, 0.0)
                )
                td_error = td_target - Q_iql[name][s, actions[name]]
                Q_iql[name][s, actions[name]] += alpha * td_error

            # 2) Update each CQL group's central Q using group's central reward
            for group_name, names in cql_groups.items():
                # central reward is sum of rewards for agents in this group
                central_r = 0.0
                current_pot_sum = 0.0
                next_pot_sum = 0.0
                for name in names:
                    r = float(rewards.get(name, 0.0))
                    total_reward[name] += r
                    central_r += r
                    current_pot_sum += float(current_pot.get(name, 0.0))
                    next_pot_sum += float(next_pot.get(name, 0.0))

                # compute joint index of chosen actions for this group
                chosen_actions_group = [actions[name] for name in names]
                joint_idx = joint_actions_to_index(chosen_actions_group, n_actions)

                Qg = Q_cql[group_name]
                # TD target for central group (uses group's Q at s2)
                td_target = (
                    central_r
                    + (gamma * next_pot_sum)
                    - current_pot_sum
                    + gamma * np.max(Qg[s2])
                )
                td_error = td_target - Qg[s, joint_idx]
                Qg[s, joint_idx] += alpha * td_error

            if mgp.get("terminated", False):
                ep_len = t + 1
                break

            obs = next_obs

        # end of episode bookkeeping
        for name in agent_names:
            per_agent_rewards[name].append(total_reward[name])

        episode_lengths.append(ep_len)
        captures_this_episode = int(getattr(env, "_captures_total", 0))
        captures_per_ep.append(captures_this_episode)

        writer.add_scalar("episode/length", ep_len, ep)
        writer.add_scalar("episode/captures", captures_this_episode, ep)

        for name in agent_names:
            writer.add_scalar(
                f"episode/total_reward/{name}", float(total_reward[name]), ep
            )
            mean_reward_running = (
                float(np.mean(per_agent_rewards[name][-window:]))
                if per_agent_rewards[name]
                else 0.0
            )
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
                [
                    f"{name}={np.mean(per_agent_rewards[name][-100:]):.2f}"
                    for name in agent_names
                ]
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
            # save IQLs
            for name, Q in Q_iql.items():
                save_q_table(save_paths[name], Q)
            # save central Qs
            for group_name, Q in Q_cql.items():
                save_q_table(save_paths[f"central_{group_name}"], Q)

    # final save
    for name, Q in Q_iql.items():
        save_q_table(save_paths[name], Q)
    for group_name, Q in Q_cql.items():
        save_q_table(save_paths[f"central_{group_name}"], Q)

    writer.close()
    LOGGER.info("Training done. Final epsilon=%.3f", eps)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Hybrid IQL/CQL trainer")
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="baselines/mixed/")
    p.add_argument("--predators", type=int, default=2)
    p.add_argument("--preys", type=int, default=2)
    p.add_argument(
        "--predator-algo",
        type=str,
        default="cql",
        choices=["iql", "cql"],
        help="Algorithm for predators",
    )
    p.add_argument(
        "--prey-algo",
        type=str,
        default="iql",
        choices=["iql", "cql"],
        help="Algorithm for preys",
    )
    p.add_argument(
        "--max-table-gb",
        type=float,
        default=16.0,
        help="Max allowed joint-Q memory in GiB " "before aborting (per central table)",
    )
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    if args.max_table_gb:
        max_table_bytes = int(args.max_table_gb * 1024**3)
    else:
        max_table_bytes = None
    try:
        if wandb is not None:
            try:
                wandb.init(project="MARL-Predator-Prey-Project")
                wandb.tensorboard.patch(root_logdir=args.save_path)
            except Exception as e:
                LOGGER.warning("WandB init failed: %s", e)

        train(
            episodes=args.episodes,
            grid_size=args.size,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
            save_path=args.save_path,
            num_predators=args.predators,
            num_preys=args.preys,
            predator_algo=args.predator_algo,
            prey_algo=args.prey_algo,
            max_table_bytes=max_table_bytes,
        )
    except MemoryError as me:
        LOGGER.error("MemoryError: %s", me)
        sys.exit(2)
    except Exception as e:
        LOGGER.exception("Training failed: %s", e)
        sys.exit(1)
