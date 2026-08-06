# plug-and-play/scripts/run_actor_critic.py
"""
Train or evaluate the Actor-Critic baseline using
plug-and-play/configs/experiment_actor_critic.yaml.

Usage:
    python plug-and-play/scripts/run_actor_critic.py                      # train
    python plug-and-play/scripts/run_actor_critic.py --mode eval \\
        --load-path trained_actor_critic.pkl
"""

import argparse
import logging

import ppage.baselines  # noqa: F401 — triggers auto-registration
from ppage.baselines.AC.actor_critic import ActorCritic
from run_from_config import (
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
    p.add_argument("--config-dir", default="plug-and-play/configs")
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
