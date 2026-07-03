"""
Train or evaluate DQN using configs/experiment_dqn.yaml.
"""

import argparse
import logging
import sys
from pathlib import Path

# ensure src/ is on sys.path so 'baselines' and 'multi_agent_package' are importable
# when this script is run directly (e.g. python3 src/.../run_dqn.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import baselines  # noqa: F401 — triggers auto-registration
from baselines.DQN.dqn import DQN
from multi_agent_package.scripts.run_from_config import build_environment, load_all_configs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

EXPERIMENT_FILE = "experiment_dqn.yaml"


def main():
    parser = argparse.ArgumentParser("Run DQN experiment")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--save-path", default="trained_dqn.pkl")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--render", action="store_true",
                        help="Enable pygame window during eval (requires a display)")
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir, EXPERIMENT_FILE)

    if args.mode == "eval":
        configs["env"]["env"]["render_mode"] = "human" if args.render else None

    env = build_environment(configs)
    algo_params = configs["experiment"]["experiment"].get("algorithm", {}).get("params", {})

    if args.mode == "train":
        algo = DQN(env, algo_params)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = DQN.load(env, algo_params, args.load_path)
        algo.evaluate()

    env.close()


if __name__ == "__main__":
    main()