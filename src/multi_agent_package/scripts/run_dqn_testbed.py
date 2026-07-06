"""
Hardcoded DQN test bed.

This is intentionally imperative for now:
- build the environment directly
- attach the plugins directly
- instantiate DQN directly

The goal is to validate the mechanics before wiring DQN into the config path.
"""

import argparse
import logging

import baselines  # noqa: F401 - triggers auto-registration
from baselines.DQN.dqn import DQN
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.registry import (
    get_action_space,
    get_observation_builder,
    get_reward_function,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


def build_testbed_environment(render_mode=None):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="predator_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]

    env = GridWorldEnv(
        agents=agents,
        size=8,
        perc_num_obstacle=10,
        render_mode=render_mode,
        seed=0,
        allow_cell_sharing=True,
        block_agents_by_obstacles=True,
        capture_threshold=1,
        max_steps=100,
    )

    observation_builder = get_observation_builder("default")
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode

    def combined_reward(env_instance):
        total = {ag.agent_name: 0.0 for ag in env_instance.agents}
        reward_fn = get_reward_function("base")
        reward_values = reward_fn.compute(env_instance)
        for agent_name in total:
            total[agent_name] += reward_values.get(agent_name, 0.0)
        return total

    env.reward_fn = combined_reward
    env.action_space_plugin = get_action_space("discrete_5")
    return env


def main():
    parser = argparse.ArgumentParser("Run hardcoded DQN test bed")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="trained_dqn.pkl")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the pygame window during train/eval.",
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = build_testbed_environment(render_mode=render_mode)
    config = {
        "episodes": args.episodes,
        "epsilon": 1.0 if args.mode == "train" else 0.0,
        "epsilon_decay": 0.99,
        "min_epsilon": 0.05,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 5000,
        "learning_rate": 1e-3,
        "hidden_dim": 128,
        "target_update_interval": 5,
    }

    if args.mode == "train":
        algo = DQN(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for eval mode")
        algo = DQN.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()


if __name__ == "__main__":
    main()
