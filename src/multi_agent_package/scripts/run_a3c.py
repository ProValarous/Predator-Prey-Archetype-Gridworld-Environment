# src/multi_agent_package/scripts/run_a3c.py
"""
Train or evaluate A3C using configs/experiment_a3c.yaml.

Usage:
    cd src
    python -m multi_agent_package.scripts.run_a3c                      # train
    python -m multi_agent_package.scripts.run_a3c --mode eval \\
        --load-path trained_a3c.pkl

A3C spawns worker processes, so this script (like any A3C entry point)
must only start them from inside an `if __name__ == "__main__":` guard --
required on Windows/spawn to avoid each worker re-importing this module
and recursively spawning its own workers.
"""

import argparse
import logging

import baselines  # noqa: F401 — triggers auto-registration
from baselines.A3C.a3c import A3C
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

EXPERIMENT_FILE = "experiment_a3c.yaml"


class EnvFactory:
    """Picklable env builder: A3C workers run in separate processes (Windows
    always uses the 'spawn' start method, which pickles Process args), and a
    lambda closing over `configs` would not survive that. A class instance
    holding only the plain, picklable `configs` dict does."""

    def __init__(self, configs: dict):
        self.configs = configs

    def __call__(self):
        return build_environment(self.configs)


def main():
    p = argparse.ArgumentParser("Run A3C experiment")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--save-path", default="trained_a3c.pkl")
    p.add_argument("--load-path", default=None)
    args = p.parse_args()

    configs = load_all_configs(args.config_dir, EXPERIMENT_FILE)

    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = None  # no pygame across processes

    env = build_environment(configs)
    algo_params = dict(
        configs["experiment"]["experiment"]["algorithm"].get("params", {})
    )
    algo_params["env_fn"] = EnvFactory(configs)

    if args.mode == "train":
        algo = A3C(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = A3C.load(env, algo_params, args.load_path)
        print(algo.evaluate())

    env.close()


if __name__ == "__main__":
    main()
