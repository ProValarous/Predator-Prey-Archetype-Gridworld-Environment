"""
Train predator-prey (tabular IQL) with TensorBoard metrics for:
- episode length (per episode)
- episode total reward (per episode)
- episode captures (per episode)
- running means (window=100) for reward and captures

Usage
-----
python iql_train_with_tb_metrics.py --episodes 20000 \
                                    --size 8 \
                                    --alpha 0.25 \
                                    --gamma 0.95

The training behavior matches the original script: one predator learns
via tabular Q-learning and the prey uses a fixed policy. This variant
logs the requested TensorBoard scalars under the save-path/logs
timestamp directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

LOGGER = logging.getLogger("iql_trainer")

# ----------------- wandb setup --------------------
wandb.init(project="MARL-Predator-Prey-Project", sync_tensorboard=True)

# renamed to follow pep8 naming convention of global variables
WANDB_PATH = "baselines/IQL/logs/"
wandb.tensorboard.patch(root_logdir=WANDB_PATH)
# --------------------------------------------------


def setup_logging(level=logging.INFO):
    """
    Configures the logging settings for the application
    """

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )


def make_env_and_meta(
        predator: Agent, prey: Agent, grid_size: int, seed: int
) -> Tuple[GridWorldEnv, int, int]:
    """
    Creates a GridWorld environment with a predator and prey agent, and returns the environment along with the number of possible states and actions.
    Args:
        predator (Agent): The predator agent.
        prey (Agent): The prey agent.
        grid_size (int): The size of the grid.
        seed (int): Random seed for environment reproducibility.
    Returns:
        Tuple[GridWorldEnv, int, int]: The environment, number of states, and number of actions.
    """
    env = GridWorldEnv(
        agents=[prey, predator],
        render_mode=None,
        size=grid_size,
        perc_num_obstacle=10,
        seed=seed
    )

    # State space = predator (x,y) + prey (x,y)
    n_states = (grid_size * grid_size) * (grid_size * grid_size)
    n_actions = env.action_space.n
    return env, n_states, n_actions


def init_q_table(n_states: int, n_actions: int) -> np.ndarray:
    """
    Initializes a Q-table for a reinforcement learning agent.
    """
    return np.zeros((n_states, n_actions), dtype=np.float32)


def epsilon_greedy_action(
        q_row: np.ndarray, n_actions: int, rng: np.random.Generator, eps: float
) -> int:
    """
    Selects an action using the epsilon-greedy strategy.
    With probability `eps`, selects a random action; otherwise, selects the action with the highest Q-value.
    Args:
        q_row (np.ndarray): Array of Q-values for available actions.
        n_actions (int): Total number of possible actions.
        rng (np.random.Generator): Random number generator.
        eps (float): Probability of choosing a random action.
    Returns:
        int: Selected action index.
    """
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(q_row))


def save_q_table(path: str, Q: np.ndarray):
    """
    Saves Q Table in the form of a NumPy .npz file at the specified path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, predator=Q)
    LOGGER.info("Saved Q-table -> %s", path)


def train(
    episodes: int = 5000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.1,
    eps_decay: float = 0.99,
    save_path: str = "baselines/IQL",
    grid_size: int = 8,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    predator = Agent(
        agent_name="predator", agent_team=1, agent_type="predator"
    )
    prey = Agent(agent_name="prey", agent_team=2, agent_type="prey")

    agents = [predator, prey]

    # prey is fixed: will always choose noop (action=4)
    env, n_states, n_actions = make_env_and_meta(
        predator, prey, grid_size, seed)

    Q = {}
    save_path_Q = {}
    for ag in agents:
        Q[ag.agent_name] = init_q_table(n_states, n_actions)
        save_path_Q[ag.agent_name] = os.path.join(
            os.path.dirname(save_path), str(ag.agent_name), "iql_q_table.npz"
        )

    eps = eps_start
    rewards_per_ep = []
    captures_per_ep = []
    episode_lengths = []

    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(
        os.path.dirname(save_path) or ".",
        "logs",
        timestamp
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Before training loop
    rewards_per_ep = {ag.agent_name: [] for ag in agents}

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        ep_len = max_steps

        for t in range(max_steps):
            # --- build state index from predator + prey positions ---
            pos_pred = obs[predator.agent_name]["local"]
            pos_prey = obs[prey.agent_name]["local"]

            pred_x, pred_y = int(pos_pred[0]), int(pos_pred[1])
            prey_x, prey_y = int(pos_prey[0]), int(pos_prey[1])

            s = (
                pred_x * grid_size ** 3
                + pred_y * grid_size ** 2
                + prey_x * grid_size
                + prey_y
            )

            # predator/prey chooses action
            a = {}
            actions = {}
            for ag in agents:
                a[ag.agent_name] = epsilon_greedy_action(
                    Q[ag.agent_name][s], n_actions, rng, eps
                )
                actions[ag.agent_name] = a[ag.agent_name]

            # actions[prey.agent_name] = 4  # prey no-op

            # a = epsilon_greedy_action(Q[s], n_actions, rng, eps) <-- old code

            # old code:
            # actions = {
            #   predator.agent_name: a,
            #   prey.agent_name: env.action_space.sample()
            # }
            # prey: static or random or policy

            # step
            mgp = env.step(actions)
            next_obs, rewards = mgp["obs"], mgp["reward"]

            r = {}
            total_reward = {ag.agent_name: 0.0 for ag in agents}
            for ag in agents:
                r[ag.agent_name] = float(rewards[ag.agent_name])
                total_reward[ag.agent_name] += r[ag.agent_name]

            # old code:
            # r_prey = float(rewards.get(prey.agent_name, 0.0))
            # total_reward_pred += r_pred
            # total_reward_prey += r_prey

            # --- next state index ---
            pos_pred_next = next_obs[predator.agent_name]["local"]
            pos_prey_next = next_obs[prey.agent_name]["local"]

            pred_x2, pred_y2 = int(pos_pred_next[0]), int(pos_pred_next[1])
            prey_x2, prey_y2 = int(pos_prey_next[0]), int(pos_prey_next[1])

            current_state = {predator.agent_name: pos_pred,
                             prey.agent_name: pos_prey}
            next_state = {
                predator.agent_name: pos_pred_next,
                prey.agent_name: pos_prey_next}

            s2 = (pred_x2 * grid_size * grid_size * grid_size
                  + pred_y2 * grid_size * grid_size
                  + prey_x2 * grid_size
                  + prey_y2)

            current_potential = env.potential_reward(current_state)
            next_potential = env.potential_reward(next_state)

            # old code:
            # current_potential_prey = env.potential_reward(current_state).get(
            #     predator.agent_name, 0.0
            # )
            # next_potential_prey = env.potential_reward(next_state).get(
            #     predator.agent_name, 0.0
            # )

            # update Q
            for ag in agents:
                Q[ag.agent_name][s, a[ag.agent_name]] += (
                    alpha * (
                        r[ag.agent_name]
                        + (gamma * next_potential[ag.agent_name])
                        - current_potential[ag.agent_name]
                        + gamma * np.max(Q[ag.agent_name][s2])
                        - Q[ag.agent_name][s, a[ag.agent_name]]
                    )
                )

            if mgp.get("terminated", False):
                ep_len = t + 1
                break

            obs = next_obs

        # episode bookkeeping
        # After each episode
        for ag in agents:
            rewards_per_ep[ag.agent_name].append(total_reward[ag.agent_name])
        # rewards_per_ep.append(total_reward)
        episode_lengths.append(ep_len)

        # captures: env._captures_total is cumulative since last reset;
        # since we reset at episode start,
        # it represents captures in this episode
        captures_this_episode = int(getattr(env, "_captures_total", 0))
        captures_per_ep.append(captures_this_episode)

        # TensorBoard logging: per-episode scalars
        for ag in agents:
            writer.add_scalar("episode/length", ep_len, ep)
            # writer.add_scalar(
            #   "episode/total_reward",
            #   total_reward[ag.agent_name], ep
            # )
            # writer.add_scalar("episode/captures", captures_this_episode, ep)

            # Running means (window=100)
            window = 100
            mean_reward_running = (
                float(np.mean(rewards_per_ep[ag.agent_name][-window:]))
                if rewards_per_ep[ag.agent_name]
                else 0.0
            )
            writer.add_scalar(
                f"mean/{ag.agent_name}/reward", mean_reward_running, ep)

        mean_captures_running = float(
            np.mean(captures_per_ep[-window:])) if captures_per_ep else 0.0
        writer.add_scalar("episode/captures", captures_this_episode, ep)
        writer.add_scalar("mean/captures", mean_captures_running, ep)

        # decay epsilon every 100 episodes
        if ep % 100 == 0:
            eps = max(eps_end, eps * eps_decay)
            avg = {}
            for ag in agents:
                avg[ag.agent_name] = (
                    np.mean(rewards_per_ep[ag.agent_name][-100:])
                    if len(rewards_per_ep[ag.agent_name]) >= 1
                    else 0.0
                )
            avg_pred = avg.get(predator.agent_name, 0.0)
            avg_prey = avg.get(prey.agent_name, 0.0)
            LOGGER.info(
                "Ep %d | eps=%.3f | Predator avg reward(last100)=%.2f | "
                "Prey avg reward(last100)=%.2f | mean captures(last100)=%.2f",
                ep, eps, avg_pred, avg_prey, mean_captures_running
            )

        # flush periodically
        if ep % 10 == 0:
            writer.flush()

        if ep % 1000 == 0:
            for ag in agents:
                save_q_table(save_path_Q[ag.agent_name], Q[ag.agent_name])

    writer.close()
    LOGGER.info("Training done. Final epsilon=%.3f", eps)


def parse_args():
    """
    Parses command-line arguments for training a predator-prey model
    """
    p = argparse.ArgumentParser(
        "Train predator-prey (1 predator learns, prey fixed)"
    )
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="baselines/IQL/")
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
    )
