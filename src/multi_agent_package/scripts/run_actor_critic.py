# src/multi_agent_package/scripts/run_actor_critic.py
"""
Train or evaluate the Actor-Critic baseline using configs/experiment_actor_critic.yaml.

Usage:
    cd src
    python -m multi_agent_package.scripts.run_actor_critic                      # train
    python -m multi_agent_package.scripts.run_actor_critic --mode eval \\
        --load-path trained_actor_critic.pkl
"""

import argparse
import logging

import baselines  # noqa: F401 — triggers auto-registration
from baselines.AC.actor_critic import ActorCritic
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

EXPERIMENT_FILE = "experiment_actor_critic.yaml"


def main():
    p = argparse.ArgumentParser("Run Actor-Critic experiment")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--save-path", default="trained_actor_critic.pkl")
    p.add_argument("--load-path", default=None)
    p.add_argument(
        "--render",
        action="store_true",
        help="Enable pygame window during eval (requires a display)",
    )
    args = p.parse_args()

    configs = load_all_configs(args.config_dir, EXPERIMENT_FILE)

    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = "human" if args.render else None

    env = build_environment(configs)
    algo_params = configs["experiment"]["experiment"]["algorithm"].get("params", {})

    if args.mode == "eval":
        # AC exploration comes from sampling the stochastic
        # policy. For evaluation we want the greedy (argmax) action instead.
        algo_params = dict(algo_params, greedy_eval=True)

    if args.mode == "train":
        algo = ActorCritic(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = ActorCritic.load(env, algo_params, args.load_path)
        print(algo.evaluate())

    env.close()


if __name__ == "__main__":
    main()
