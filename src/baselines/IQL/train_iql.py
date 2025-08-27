"""
Train tabular IQL (Independent Q-Learning) for 1 predator and 1 prey using
the GridWorldEnv's reward function (env._get_reward) and termination logic.

TensorBoard logs are written alongside the saved Qs:
    <save_dir>/logs/<timestamp>/

Run from project root:
    python src/baselines/IQL/train_iql.py --episodes 5000 --size 8

View logs:
    tensorboard --logdir src/baselines/IQL/logs
"""
import argparse
import os
import time
from typing import Dict, Tuple, Optional

import numpy as np

# from tensorflow.summary import create_file_writer  # if using TF directly
from torch.utils.tensorboard import SummaryWriter

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent


def make_agents() -> Tuple[Agent, Agent]:
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    """Encode (x,y) into a single integer state index."""
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


# def compute_cumulative_reward_from_episode_rewards(
#     episode_rewards: Dict[str, float],
#     name_to_type: Optional[Dict[str, str]] = None,
#     eps: float = 1e-6,
# ) -> float:
#     """Compute cumulative reward for an episode using only the per-agent reward dict.

#     cumulative_reward = (1 / mean_diff) * (sum_prey_rewards + sum_predator_rewards)
#     with mean_diff = |mean_pred - mean_prey|. Returns 0.0 if either group is missing.
#     """
#     if not episode_rewards:
#         return 0.0

#     def infer_type(name: str) -> Optional[str]:
#         if name_to_type and name in name_to_type:
#             return name_to_type[name].lower()
#         ln = name.lower()
#         if "prey" in ln:
#             return "prey"
#         if "predator" in ln:
#             return "predator"
#         if ln.startswith("py"):
#             return "prey"
#         if ln.startswith("pd"):
#             return "predator"
#         return None

#     prey_vals = []
#     pred_vals = []
#     for name, val in episode_rewards.items():
#         t = infer_type(name)
#         if t == "prey":
#             prey_vals.append(float(val))
#         elif t == "predator":
#             pred_vals.append(float(val))

#     if len(prey_vals) == 0 or len(pred_vals) == 0:
#         return 0.0

#     sum_prey = sum(prey_vals)
#     sum_pred = sum(pred_vals)
#     mean_prey = sum_prey / len(prey_vals)
#     mean_pred = sum_pred / len(pred_vals)

#     mean_diff = abs(mean_pred - mean_prey)
#     denom = max(mean_diff, eps)
#     cumulative = (1.0 / denom) * (sum_prey + sum_pred)
#     return float(cumulative)


def train(
    episodes: int = 5000,
    max_steps: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
    grid_size: int = 8,
    save_path: str = "baselines/IQL/iql_qs.npz",
    seed: int = 0,
):
    prey, predator = make_agents()
    agents = [prey, predator]

    # Set total_subteams for color/shape logic (not required for training)
    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=10, seed=seed)

    rng = np.random.default_rng(seed)

    n_states = grid_size * grid_size
    n_actions = env.action_space.n  # expected 5

    # Tabular Q per agent
    Qs: Dict[str, np.ndarray] = {
        prey.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
        predator.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
    }

    # Stats tracking
    prey_episode_totals = []
    predator_episode_totals = []
    cumulative_episode_totals = []

    eps = eps_start
    capture_count = 0

    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Prepare TensorBoard writer
    timestamp = time.strftime("%d-%m-%Y -- %H-%M-%S")
    log_dir = os.path.join(save_dir or ".", "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs -> {log_dir}")


    for ep in range(1, episodes + 1):
        obs, info = env.reset()

        # per-episode accumulators per agent (use agent names as keys)
        ep_agent_totals: Dict[str, float] = {ag.agent_name: 0.0 for ag in agents}

        # step loop
        for t in range(max_steps):
            # state indices for each agent
            s_idx = {ag.agent_name: state_index(obs[ag.agent_name]["local"], grid_size) for ag in agents}

            # select actions epsilon-greedy (independent)
            actions = {}
            for ag in agents:
                q_row = Qs[ag.agent_name][s_idx[ag.agent_name]]
                if rng.random() < eps:
                    a = int(rng.integers(0, n_actions))
                else:
                    a = int(np.argmax(q_row))
                actions[ag.agent_name] = a

            # step using env (we will use env's reward)
            mgp = env.step(actions)
            next_obs = mgp["obs"]
            rewards_from_env = mgp["reward"]

            # accumulate per-agent rewards for this episode
            for ag in agents:
                ep_agent_totals[ag.agent_name] += float(rewards_from_env.get(ag.agent_name, 0.0))

            # Q update (IQL: each agent treats others as environment)
            s_next_idx = {ag.agent_name: state_index(next_obs[ag.agent_name]["local"], grid_size) for ag in agents}
            for ag in agents:
                s = s_idx[ag.agent_name]
                a = int(actions[ag.agent_name])
                r = float(rewards_from_env.get(ag.agent_name, 0.0))
                s2 = s_next_idx[ag.agent_name]
                qvals = Qs[ag.agent_name]
                td_target = r + gamma * np.max(qvals[s2])
                td_error = td_target - qvals[s, a]
                qvals[s, a] += alpha * td_error

            # termination handled by env
            if mgp.get("terminated", False):
                capture_count += 1
                break

            obs = next_obs

        # episode finished: aggregate per-type totals and compute cumulative metric
        ep_prey_total = sum(v for n, v in ep_agent_totals.items() if "prey" in n.lower() or n.lower().startswith("py"))
        ep_pred_total = sum(v for n, v in ep_agent_totals.items() if "predator" in n.lower() or n.lower().startswith("pd"))

        prey_episode_totals.append(ep_prey_total)
        predator_episode_totals.append(ep_pred_total)

        # compute cumulative reward from the episode-level per-agent totals
        # cumulative = compute_cumulative_reward_from_episode_rewards(ep_agent_totals)
        # cumulative_episode_totals.append(cumulative)

        # decay epsilon
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)

        # running stats
        mean_prey = float(np.mean(prey_episode_totals)) if prey_episode_totals else 0.0
        mean_pred = float(np.mean(predator_episode_totals)) if predator_episode_totals else 0.0
        mean_diff = abs(mean_pred - mean_prey)
        # mean_cumulative = float(np.mean(cumulative_episode_totals)) if cumulative_episode_totals else 0.0

        # TensorBoard logging (per-episode scalars)
        writer.add_scalar("total/prey", ep_prey_total, ep)
        writer.add_scalar("total/predator", ep_pred_total, ep)
        writer.add_scalar("mean/prey", mean_prey, ep)
        writer.add_scalar("mean/predator", mean_pred, ep)
        writer.add_scalar("mean_diff/pred_minus_prey", mean_diff, ep)
        # writer.add_scalar("cumulative/episode", cumulative, ep)

        if ep % 100 == 0:
            print(
                f"Episode {ep}/{episodes} | eps={eps:.4f} | captures={capture_count} | "
                f"last_prey_total={ep_prey_total:.3f} | last_pred_total={ep_pred_total:.3f} | "
                f"mean_prey={mean_prey:.3f} | mean_pred={mean_pred:.3f} | mean_diff(pred-prey)={mean_diff:.3f}"
            )

        # flush periodically so logs are available during long runs
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
    mean_cumulative = float(np.mean(cumulative_episode_totals)) if cumulative_episode_totals else 0.0

    print("FINAL SUMMARY:")
    print(f"  Episodes run: {episodes}")
    print(f"  Total prey reward (sum over episodes): {total_prey_reward:.3f}")
    print(f"  Total predator reward (sum over episodes): {total_pred_reward:.3f}")
    print(f"  Mean prey reward per episode: {mean_prey:.3f}")
    print(f"  Mean predator reward per episode: {mean_pred:.3f}")
    print(f"  Mean difference (predator - prey): {mean_diff:.3f}")
    print(f"  Mean cumulative_reward per episode: {mean_cumulative:.3f}")

    # finalize writer
    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train IQL (tabular) predator-prey")
    parser.add_argument("--episodes", type=int, default=25000)
    parser.add_argument("--size", type=int, default=8)
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
