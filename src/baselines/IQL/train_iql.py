"""
train_iql.py

Train two agents (1 predator, 1 prey) using Independent Q-Learning (tabular).
Saves learned Q-tables to `iql_qs.npz`.

Usage:
    python train_iql.py
"""
import numpy as np
import argparse
from typing import Dict, Tuple

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent


def make_agents() -> Tuple[Agent, Agent]:
    # Agent(agent_type, agent_team, agent_name)
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    """Encode (x,y) into a single integer state index."""
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def train(
    episodes: int = 5000,
    max_steps: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.9995,
    grid_size: int = 8,
    save_path: str = "iql_qs.npz",
    seed: int = 0,
):
    # Create agents and env
    prey, predator = make_agents()
    agents = [prey, predator]

    env = GridWorldEnv(agents=agents, render_mode=None, size=grid_size, perc_num_obstacle=10)
    # set total_subteams so coloring/shape logic stays consistent (optional)
    for ag in agents:
        ag.total_subteams = 1

    rng = np.random.default_rng(seed)

    n_states = grid_size * grid_size
    n_actions = env.action_space.n  # should be 5

    # Tabular Q per agent: shape (n_states, n_actions)
    Qs: Dict[str, np.ndarray] = {
        prey.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
        predator.agent_name: np.zeros((n_states, n_actions), dtype=np.float32),
    }

    eps = eps_start
    print_every = max(1, episodes // 10)
    capture_count = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False

        # step loop
        for t in range(max_steps):
            # Build current state indices
            s_idx = {}
            for ag in agents:
                s_idx[ag.agent_name] = state_index(obs[ag.agent_name]["local"], grid_size)

            # Select actions (epsilon-greedy independent)
            actions = {}
            for ag in agents:
                q = Qs[ag.agent_name][s_idx[ag.agent_name]]
                if rng.random() < eps:
                    a = rng.integers(0, n_actions)
                else:
                    a = int(np.argmax(q))
                actions[ag.agent_name] = a

            # Step environment
            mgp = env.step(actions)
            next_obs = mgp["obs"]

            # Compute rewards ourselves (simple shaping):
            # - predator gets +10 for capture, prey gets -10 on being captured
            # - small step penalty to encourage finishing (-0.01)
            # Determine capture: same cell
            prey_pos = next_obs[prey.agent_name]["local"]
            predator_pos = next_obs[predator.agent_name]["local"]
            captured = np.array_equal(prey_pos, predator_pos)

            rewards = {}
            if captured:
                rewards[predator.agent_name] = 10.0
                rewards[prey.agent_name] = -10.0
            else:
                rewards[predator.agent_name] = -0.01
                rewards[prey.agent_name] = -0.01

            # Compute next state idx
            s_next_idx = {
                ag.agent_name: state_index(next_obs[ag.agent_name]["local"], grid_size)
                for ag in agents
            }

            # Q-updates (IQL: each agent treats others as part of environment)
            for ag in agents:
                s = s_idx[ag.agent_name]
                a = actions[ag.agent_name]
                r = rewards[ag.agent_name]
                s2 = s_next_idx[ag.agent_name]
                qvals = Qs[ag.agent_name]
                td_target = r + gamma * np.max(qvals[s2])
                td_error = td_target - qvals[s, a]
                qvals[s, a] += alpha * td_error

            # If capture -> end episode
            if captured:
                capture_count += 1
                break

            obs = next_obs

        # Epsilon decay
        eps = max(eps_end, eps * eps_decay)

        if ep % print_every == 0 or ep == 1:
            print(f"Ep {ep}/{episodes} eps={eps:.3f} captures={capture_count}")

    # Save Q-tables
    np.savez(save_path, **{name: Q for name, Q in Qs.items()})
    print(f"Training finished. Saved Qs to '{save_path}'. Total captures: {capture_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train IQL (tabular) for predator-prey")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        grid_size=args.size,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )
