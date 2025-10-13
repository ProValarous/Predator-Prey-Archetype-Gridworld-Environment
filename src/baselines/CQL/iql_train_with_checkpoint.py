"""
Central Q-Learning (tabular CQL) trainer for predator-prey with TensorBoard metrics:
 - episode length (per episode)
 - episode total reward (per episode) (per-agent)
 - episode captures (per episode)
 - running means (window=100) for reward and captures

Usage
-----
python cql_train_with_tb_metrics.py --episodes 20000 --size 8 --alpha 0.25 --gamma 0.95

This keeps the original script's structure and style but replaces independent Q-tables
with a central (joint) Q-table over joint-state and joint-action. The prey remains a
fixed-policy agent (noop=4) by default, but the central Q supports general joint-actions.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

LOGGER = logging.getLogger("cql_trainer")

################ wandb setup #######################
wandb.init(project="MARL-Predator-Prey-Project")

wandb_path = "baselines/CQL/logs/"
wandb.tensorboard.patch(root_logdir=wandb_path)
###################################################


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


def make_env_and_meta(agents: List[Agent], grid_size: int, seed: int) -> Tuple[GridWorldEnv, int, int]:
    env = GridWorldEnv(agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=10, seed=seed)

    # State space = predator (x,y) + prey (x,y)
    n_states = (grid_size * grid_size) * (grid_size * grid_size)
    n_actions = env.action_space.n
    return env, n_states, n_actions


def init_joint_q_table(n_states: int, n_joint_actions: int) -> np.ndarray:
    return np.zeros((n_states, n_joint_actions), dtype=np.float32)


def joint_action_to_index(joint_action: List[int], n_actions: int) -> int:
    # Interpret joint action as a base-n_actions integer. Order is agent order in `agents`.
    idx = 0
    for a in joint_action:
        idx = idx * n_actions + int(a)
    return int(idx)


def index_to_joint_action(idx: int, n_agents: int, n_actions: int) -> List[int]:
    # Recover joint action in agent order
    joint = [0] * n_agents
    for i in range(n_agents - 1, -1, -1):
        joint[i] = int(idx % n_actions)
        idx //= n_actions
    return joint


def save_q_table(path: str, Q: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, central=Q)
    LOGGER.info("Saved joint Q-table -> %s", path)


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
    seed: int = 0,
    prey_fixed_action: int = 4,  # keep prey fixed (noop) by default
):
    rng = np.random.default_rng(seed)

    predator = Agent(agent_name="predator", agent_team=1, agent_type="predator")
    prey = Agent(agent_name="prey", agent_team=2, agent_type="prey")

    agents = [predator, prey]
    n_agents = len(agents)

    env, n_states, n_actions = make_env_and_meta(agents, grid_size, seed)

    # Joint-action space size
    n_joint_actions = n_actions ** n_agents

    # Central joint Q-table: shape (n_states, n_joint_actions)
    Q = init_joint_q_table(n_states, n_joint_actions)

    save_path_Q = os.path.join(os.path.dirname(save_path) or ".", "central_iql_q_table.npz")

    eps = eps_start

    # Per-agent episode reward history
    rewards_per_ep = {ag.agent_name: [] for ag in agents}
    episode_lengths = []
    captures_per_ep = []

    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(os.path.dirname(save_path) or ".", "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    window = 100

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_len = max_steps

        # reset per-episode cumulative rewards
        total_reward_per_agent = {ag.agent_name: 0.0 for ag in agents}

        for t in range(max_steps):
            # build state index from predator + prey positions
            pos_pred = obs[predator.agent_name]["local"]
            pos_prey = obs[prey.agent_name]["local"]

            pred_x, pred_y = int(pos_pred[0]), int(pos_pred[1])
            prey_x, prey_y = int(pos_prey[0]), int(pos_prey[1])

            s = (
                pred_x * grid_size * grid_size * grid_size
                + pred_y * grid_size * grid_size
                + prey_x * grid_size
                + prey_y
            )

            # --- Centralized action selection ---
            # If prey is fixed, we only search over predator's actions: joint_idx = pred_action * n_actions + prey_fixed_action
            if prey_fixed_action is not None:
                # compute argmax over predator actions using joint Q values where prey action = prey_fixed_action
                joint_indices = [pred_a * n_actions + prey_fixed_action for pred_a in range(n_actions)]
                q_values_for_pred = Q[s, joint_indices]

                if rng.random() < eps:
                    pred_action = int(rng.integers(0, n_actions))
                else:
                    pred_action = int(int(np.argmax(q_values_for_pred)))

                joint_action = [pred_action, prey_fixed_action]
                joint_idx = joint_action_to_index(joint_action, n_actions)

            else:
                # general joint epsilon-greedy over full joint action space
                if rng.random() < eps:
                    joint_idx = int(rng.integers(0, n_joint_actions))
                else:
                    joint_idx = int(np.argmax(Q[s]))
                joint_action = index_to_joint_action(joint_idx, n_agents, n_actions)

            # split joint action to per-agent actions
            actions = {agents[i].agent_name: int(joint_action[i]) for i in range(n_agents)}

            mgp = env.step(actions)
            next_obs, rewards = mgp["obs"], mgp["reward"]

            # accumulate per-agent rewards
            for ag in agents:
                r = float(rewards.get(ag.agent_name, 0.0))
                total_reward_per_agent[ag.agent_name] += r

            # next state
            pos_pred_next = next_obs[predator.agent_name]["local"]
            pos_prey_next = next_obs[prey.agent_name]["local"]

            pred_x2, pred_y2 = int(pos_pred_next[0]), int(pos_pred_next[1])
            prey_x2, prey_y2 = int(pos_prey_next[0]), int(pos_prey_next[1])

            s2 = (
                pred_x2 * grid_size * grid_size * grid_size
                + pred_y2 * grid_size * grid_size
                + prey_x2 * grid_size
                + prey_y2
            )

            # potential-based shaping: sum potentials across agents for central update
            current_potential = env.potential_reward({agents[0].agent_name: pos_pred, agents[1].agent_name: pos_prey})
            next_potential = env.potential_reward({agents[0].agent_name: pos_pred_next, agents[1].agent_name: pos_prey_next})

            current_potential_sum = sum(float(v) for v in current_potential.values())
            next_potential_sum = sum(float(v) for v in next_potential.values())

            # central reward = sum of per-agent rewards
            central_r = sum(float(rewards.get(ag.agent_name, 0.0)) for ag in agents)

            # update joint Q
            Q[s, joint_idx] += alpha * (
                central_r + (gamma * next_potential_sum) - current_potential_sum + gamma * np.max(Q[s2]) - Q[s, joint_idx]
            )

            if mgp.get("terminated", False):
                ep_len = t + 1
                break

            obs = next_obs

        # episode bookkeeping
        for ag in agents:
            rewards_per_ep[ag.agent_name].append(total_reward_per_agent[ag.agent_name])

        episode_lengths.append(ep_len)

        # captures for this episode (env._captures_total resets on env.reset())
        captures_this_episode = int(getattr(env, "_captures_total", 0))
        captures_per_ep.append(captures_this_episode)

        # TensorBoard logging
        # episode length (single scalar)
        writer.add_scalar("episode/length", ep_len, ep)

        # per-agent total reward and running mean
        for ag in agents:
            writer.add_scalar(f"episode/{ag.agent_name}/total_reward", float(total_reward_per_agent[ag.agent_name]), ep)
            mean_reward_running = float(np.mean(rewards_per_ep[ag.agent_name][-window:])) if rewards_per_ep[ag.agent_name] else 0.0
            writer.add_scalar(f"mean/{ag.agent_name}/reward", mean_reward_running, ep)

        # captures
        writer.add_scalar("episode/captures", captures_this_episode, ep)
        mean_captures_running = float(np.mean(captures_per_ep[-window:])) if captures_per_ep else 0.0
        writer.add_scalar("mean/captures", mean_captures_running, ep)

        # epsilon decay every 100 episodes
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)
            avg = {ag.agent_name: np.mean(rewards_per_ep[ag.agent_name][-100:]) if len(rewards_per_ep[ag.agent_name]) >= 1 else 0.0 for ag in agents}
            LOGGER.info(
                "Ep %d | eps=%.3f | Predator avg reward(last100)=%.2f | Prey avg reward(last100)=%.2f | mean captures(last100)=%.2f",
                ep,
                eps,
                avg.get(predator.agent_name, 0.0),
                avg.get(prey.agent_name, 0.0),
                mean_captures_running,
            )

        if ep % 10 == 0:
            writer.flush()

        if ep % 1000 == 0:
            save_q_table(save_path_Q, Q)

    writer.close()
    LOGGER.info("Training done. Final epsilon=%.3f", eps)


def parse_args():
    p = argparse.ArgumentParser("Train predator-prey (centralized tabular CQL)")
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="baselines/CQL/")
    p.add_argument("--prey-fixed-action", type=int, default=4, help="If set, forces prey action to this value every step (keeps prey policy fixed). Set to -1 for no forcing.")
    return p.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    prey_fixed = args.prey_fixed_action if args.prey_fixed_action >= 0 else None
    train(
        episodes=args.episodes,
        grid_size=args.size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        save_path=args.save_path,
        prey_fixed_action=prey_fixed,
    )
