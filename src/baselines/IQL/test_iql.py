"""
Test (visualize) the trained Qs saved by `train_iql.py` using env rewards.

Run:
    python src/baselines/IQL/test_iql.py --file src/baselines/IQL/iql_qs.npz --size 8 --episodes 5
"""
import argparse
import time
import numpy as np
from typing import Dict

from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent


def make_agents():
    prey = Agent("prey", 1, "prey_1")
    predator = Agent("predator", 1, "predator_1")
    return prey, predator


def state_index(pos: np.ndarray, size: int) -> int:
    x, y = int(pos[0]), int(pos[1])
    return x * size + y


def run_test(q_file: str = "baselines/IQL/iql_qs.npz", size: int = 8, episodes: int = 3, max_steps: int = 200, pause: float = 0.05):
    data = np.load(q_file)
    Qs: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    print("Loaded Q keys:", list(Qs.keys()))

    prey, predator = make_agents()
    agents = [prey, predator]

    for ag in agents:
        ag.total_subteams = 1

    env = GridWorldEnv(agents=agents, render_mode="human", size=size, perc_num_obstacle=10)

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            print(f"Test Episode {ep+1}/{episodes}")
            for t in range(max_steps):
                actions = {}
                for ag in agents:
                    s_idx = state_index(obs[ag.agent_name]["local"], size)
                    q = Qs[ag.agent_name]
                    a = int(np.argmax(q[s_idx]))
                    actions[ag.agent_name] = a

                mgp = env.step(actions)
                obs = mgp["obs"]
                print(mgp['reward'])
                time.sleep(pause)

                # termination if capture
                if mgp.get("terminated", False):
                    print(f"Capture at step {t+1} (episode {ep+1})")
                    break

            time.sleep(0.5)
    finally:
        env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test IQL-trained agents")
    parser.add_argument("--file", type=str, default="baselines/IQL/iql_qs.npz")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.05)
    args = parser.parse_args()

    run_test(q_file=args.file, size=args.size, episodes=args.episodes, pause=args.pause)
