# src/multi_agent_package/scripts/run_cql.py
"""
Train or evaluate CQL using configs/experiment_cql.yaml.

Usage:
    cd src
    python -m multi_agent_package.scripts.run_cql                      # train
    python -m multi_agent_package.scripts.run_cql --mode eval \\
        --load-path trained_cql.pkl
"""

import argparse
import logging

import baselines  # noqa: F401 — triggers auto-registration
from baselines.CQL.cql import CQL
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

EXPERIMENT_FILE = "experiment_cql.yaml"


def main():
    p = argparse.ArgumentParser("Run CQL experiment")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--save-path", default="trained_cql.pkl")
    p.add_argument("--load-path", default=None)
    p.add_argument("--render", action="store_true",
                   help="Enable pygame window during eval (requires a display)")
    args = p.parse_args()

    configs = load_all_configs(args.config_dir, EXPERIMENT_FILE)

    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = "human" if args.render else None

    env = build_environment(configs)
    algo_params = configs["experiment"]["experiment"]["algorithm"].get("params", {})

    if args.mode == "train":
        algo = CQL(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = CQL.load(env, algo_params, args.load_path)
        algo.evaluate()

    env.close()


if __name__ == "__main__":
    main()
