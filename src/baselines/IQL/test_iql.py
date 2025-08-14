"""
test_iql.py

Load trained Q-tables saved by train_iql.py and run the environment in human mode
so you can visually watch agent behavior.

Usage:
    python test_iql.py --file iql_qs.npz --size 8 --episodes 5
"""
import numpy as np
import argparse
import time
from typing import Dict

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent


def make_agents() -> tuple:
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def run_test(q_file: str = "iql_qs.npz", size: int = 8, episodes: int = 5, max_steps: int = 200, pause: float = 0.05):
    # load Qs
    data = np.load(q_file)
    # Expect keys: prey_1, predator_1
    Qs: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    print("Loaded Q keys:", list(Qs.keys()))

    prey, predator = make_agents()
    agents = [prey, predator]

    # ensure agent.total_subteams consistent (optional)
    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode="human", size=size, perc_num_obstacle=10)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            print(f"Test Episode {ep+1}/{episodes}")
            for t in range(max_steps):
                # greedy actions from Q
                actions = {}
                for ag in agents:
                    s_idx = state_index(obs[ag.agent_name]["local"], size)
                    q = Qs[ag.agent_name]
                    a = int(np.argmax(q[s_idx]))
                    actions[ag.agent_name] = a

                mgp = env.step(actions)
                obs = mgp["obs"]

                # show frame
                time.sleep(pause)

                # stop if capture
                prey_pos = obs[prey.agent_name]["local"]
                predator_pos = obs[predator.agent_name]["local"]
                if np.array_equal(prey_pos, predator_pos):
                    print(f"Capture at step {t+1}")
                    break

            time.sleep(0.5)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test IQL-trained agents")
    parser.add_argument("--file", type=str, default="iql_qs.npz")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.05)
    args = parser.parse_args()

    run_test(q_file=args.file, size=args.size, episodes=args.episodes, pause=args.pause)
