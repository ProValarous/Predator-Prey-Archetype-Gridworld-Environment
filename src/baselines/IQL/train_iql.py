"""
Train tabular IQL (Independent Q-Learning) for 1 predator and 1 prey using
the GridWorldEnv's reward function (env._get_reward) and termination logic.

This variant uses the environment's `obs[agent]['global']['dist_agents']`
to build a compact joint state: (agent_pos, discretized_distance_to_other).
Note: `dist_agents` provides distance only (no direction).
"""

import argparse
import os
import time
from typing import Dict, Tuple, Optional

import math
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent


def make_agents() -> Tuple[Agent, Agent]:
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    """Encode (x,y) into a single integer state index (row-major)."""
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def global_joint_state_index(
    own_pos: np.ndarray,
    dist_agents: Dict[str, float],
    other_name: str,
    size: int,
) -> int:
    """
    Create a compact joint-state index using:
      - own position (ax,ay) encoded as a_idx = ax*size + ay
      - distance to other agent (from dist_agents[other_name]) discretized to integer bins

    Returns index in range [0, n_cells * (max_dist+1) - 1].
    """
    # own cell index
    a_idx = state_index(own_pos, size)

    # estimate maximum possible Euclidean distance on grid and number of bins
    max_dist = math.ceil(math.sqrt(2) * (size - 1))
    # read distance (fallback to max_dist if missing)
    d = None
    if dist_agents is not None:
        try:
            d = float(dist_agents.get(other_name, max_dist))
        except Exception:
            d = max_dist
    else:
        d = max_dist

    # discretize/clamp distance to integer bin [0..max_dist]
    dist_bin = int(min(max_dist, max(0, int(round(d)))))

    return a_idx * (max_dist + 1) + dist_bin


def train(
    episodes: int = 5000,
    max_steps: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.9,
    grid_size: int = 8,
    save_path: str = "baselines/IQL/iql_qs.npz",
    seed: int = 0,
):
    prey, predator = make_agents()
    agents = [prey, predator]

    # Set total_subteams for color / shape logic (not required for training)
    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=10, seed=seed)
    rng = np.random.default_rng(seed)

    # n_cells for own position; distance bins determined by grid geometry
    n_cells = grid_size * grid_size
    max_dist = math.ceil(math.sqrt(2) * (grid_size - 1))
    n_distance_bins = max_dist + 1

    # total states = own_cell * distance_bin
    n_states = n_cells * n_distance_bins
    n_actions = env.action_space.n  # expected 5

    # Q-tables
    Qs: Dict[str, np.ndarray] = {
        prey.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
        predator.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
    }

    # Stats
    prey_episode_totals = []
    predator_episode_totals = []
    eps = eps_start
    capture_count = 0

    # Make save/log dirs
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # TensorBoard writer (safe timestamp format)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(save_dir or ".", "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs -> {log_dir}")

    # ---- training loop ----
    for ep in range(1, episodes + 1):
        obs, info = env.reset()

        # per-episode raw reward accumulators
        ep_agent_totals: Dict[str, float] = {ag.agent_name: 0.0 for ag in agents}

        for t in range(max_steps):
            # Build state indices using obs[agent]['global']['dist_agents']
            s_idx: Dict[str, int] = {}
            pos_map: Dict[str, np.ndarray] = {}

            # read positions (from obs[].get('local')) for own_pos fallback
            for ag in agents:
                # get local pos if present
                local = obs.get(ag.agent_name, {}).get("local", None)
                if local is None:
                    # fallback to agent internal state
                    local = getattr(ag, "_agent_location", np.array([0, 0]))
                pos_map[ag.agent_name] = np.asarray(local, dtype=int)

            # compute s_idx for each agent using the distance provided in global
            for ag in agents:
                other = next(o for o in agents if o.agent_name != ag.agent_name)
                # dist_agents dict is under obs[ag]['global']['dist_agents']
                global_info = obs.get(ag.agent_name, {}).get("global", {})
                dist_agents = global_info.get("dist_agents", {}) if global_info is not None else {}
                s_idx[ag.agent_name] = global_joint_state_index(
                    pos_map[ag.agent_name], dist_agents, other.agent_name, grid_size
                )

            # epsilon-greedy action selection
            actions: Dict[str, int] = {}
            for ag in agents:
                q_row = Qs[ag.agent_name][s_idx[ag.agent_name]]
                if rng.random() < eps:
                    a = int(rng.integers(0, n_actions))
                else:
                    a = int(np.argmax(q_row))
                actions[ag.agent_name] = a

            # step environment
            mgp = env.step(actions)
            next_obs = mgp["obs"]
            rewards_from_env = mgp["reward"]

            # accumulate raw rewards
            for ag in agents:
                ep_agent_totals[ag.agent_name] += float(rewards_from_env.get(ag.agent_name, 0.0))

            # Compute next-state indices using next_obs
            s_next_idx: Dict[str, int] = {}
            next_pos_map: Dict[str, np.ndarray] = {}
            for ag in agents:
                local = next_obs.get(ag.agent_name, {}).get("local", None)
                if local is None:
                    local = getattr(ag, "_agent_location", np.array([0, 0]))
                next_pos_map[ag.agent_name] = np.asarray(local, dtype=int)

            for ag in agents:
                other = next(o for o in agents if o.agent_name != ag.agent_name)
                global_info_next = next_obs.get(ag.agent_name, {}).get("global", {})
                dist_agents_next = global_info_next.get("dist_agents", {}) if global_info_next is not None else {}
                s_next_idx[ag.agent_name] = global_joint_state_index(
                    next_pos_map[ag.agent_name], dist_agents_next, other.agent_name, grid_size
                )

            # IQL updates (per-agent)
            for ag in agents:
                s = s_idx[ag.agent_name]
                a = int(actions[ag.agent_name])
                r = float(rewards_from_env.get(ag.agent_name, 0.0))
                s2 = s_next_idx[ag.agent_name]
                qvals = Qs[ag.agent_name]
                td_target = r + gamma * np.max(qvals[s2])
                td_error = td_target - qvals[s, a]
                qvals[s, a] += alpha * td_error

            # termination check
            if mgp.get("terminated", False):
                capture_count += 1
                break

            obs = next_obs

        # episode finished: aggregate totals (by name heuristics)
        ep_prey_total = sum(v for n, v in ep_agent_totals.items() if "prey" in n.lower() or n.lower().startswith("py"))
        ep_pred_total = sum(v for n, v in ep_agent_totals.items() if "predator" in n.lower() or n.lower().startswith("pd"))

        prey_episode_totals.append(ep_prey_total)
        predator_episode_totals.append(ep_pred_total)

        # decay epsilon periodically
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)

        # running stats
        mean_prey = float(np.mean(prey_episode_totals)) if prey_episode_totals else 0.0
        mean_pred = float(np.mean(predator_episode_totals)) if predator_episode_totals else 0.0
        mean_diff = abs(mean_pred - mean_prey)

        # log to TensorBoard
        writer.add_scalar("total/prey", ep_prey_total, ep)
        writer.add_scalar("total/predator", ep_pred_total, ep)
        writer.add_scalar("mean/prey", mean_prey, ep)
        writer.add_scalar("mean/predator", mean_pred, ep)
        writer.add_scalar("mean_diff/pred_minus_prey", mean_diff, ep)

        if ep % 100 == 0:
            print(
                f"Episode {ep}/{episodes} | eps={eps:.4f} | captures={capture_count} | "
                f"last_prey_total={ep_prey_total:.3f} | last_pred_total={ep_pred_total:.3f} | "
                f"mean_prey={mean_prey:.3f} | mean_pred={mean_pred:.3f} | mean_diff(pred-prey)={mean_diff:.3f}"
            )

        if ep % 10 == 0:
            writer.flush()

    # Save Q-tables
    np.savez(save_path, **{name: Q for name, Q in Qs.items()})
    print(f"Training finished. Saved Qs to '{save_path}'. Total captures: {capture_count}")

    # Final summary
    mean_prey = float(np.mean(prey_episode_totals)) if prey_episode_totals else 0.0
    mean_pred = float(np.mean(predator_episode_totals)) if predator_episode_totals else 0.0
    mean_diff = mean_pred - mean_prey
    total_prey_reward = float(np.sum(prey_episode_totals))
    total_pred_reward = float(np.sum(predator_episode_totals))

    print("FINAL SUMMARY:")
    print(f"  Episodes run: {episodes}")
    print(f"  Total prey reward (sum over episodes): {total_prey_reward:.3f}")
    print(f"  Total predator reward (sum over episodes): {total_pred_reward:.3f}")
    print(f"  Mean prey reward per episode: {mean_prey:.3f}")
    print(f"  Mean predator reward per episode: {mean_pred:.3f}")
    print(f"  Mean difference (predator - prey): {mean_diff:.3f}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train IQL (tabular) predator-prey")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )
