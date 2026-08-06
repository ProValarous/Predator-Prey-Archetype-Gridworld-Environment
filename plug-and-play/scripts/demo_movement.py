# plug-and-play/scripts/demo_movement.py
"""
No-training movement-type demo.

Builds the environment from a config directory and drives it with
uniformly random actions (no trained policy -- there's nothing to load)
in a live pygame window, so an audience can see an action plugin's raw
movement geometry -- or, for a speed config, the effect of differing
agent_speed values -- without waiting on DQN training.

Usage (run from the repository root; --config-dir is resolved from there):
    python plug-and-play/scripts/demo_movement.py --config-dir CONFIG_DIR

CONFIG_DIR is one of the plug-and-play/configs/ demo sets: demo_plus,
demo_diagonal, or demo_speed.
"""

import argparse

import numpy as np

from run_from_config import (
    build_environment,
    load_all_configs,
)


def main():
    p = argparse.ArgumentParser("No-training movement-type demo (random actions)")
    p.add_argument("--config-dir", required=True)
    p.add_argument("--experiment-file", default="experiment_dqn.yaml")
    p.add_argument(
        "--steps", type=int, default=200, help="number of logical env steps to run"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="seed for random action selection"
    )
    args = p.parse_args()

    configs = load_all_configs(args.config_dir, args.experiment_file)
    configs["env"]["env"]["render_mode"] = "human"
    env = build_environment(configs)

    obs, _ = env.reset()
    agent_ids = list(obs.keys())
    n_actions = env.action_space_plugin.n_actions
    rng = np.random.default_rng(args.seed)

    for _ in range(args.steps):
        actions = {aid: int(rng.integers(n_actions)) for aid in agent_ids}
        out = env.step(actions)
        if out["terminated"] or out["truncated"]:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
