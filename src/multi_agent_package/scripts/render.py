"""
Render a single episode in the pygame window.

By default the agents act randomly (seeded for reproducibility); pass
--load-path to render a trained policy instead. This replaces the old behavior
where render.py ran a full training experiment. Run from the repository root.
"""

import argparse

import numpy as np

import baselines  # noqa: F401 — triggers algorithm auto-registration
from baselines.registry import get as get_algorithm
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)


def render_episode(
    config_dir: str = "configs",
    experiment_file: str = "experiment.yaml",
    load_path: str = None,
    seed: int = None,
) -> None:
    configs = load_all_configs(config_dir, experiment_file)
    configs["env"]["env"]["render_mode"] = "human"
    if seed is not None:
        configs["env"]["env"]["seed"] = seed

    env = build_environment(configs)

    algo = None
    if load_path:
        algo_cfg = configs["experiment"]["experiment"]["algorithm"]
        algo = get_algorithm(algo_cfg["name"]).load(
            env, algo_cfg.get("params", {}), load_path
        )
        if hasattr(algo, "epsilon"):
            algo.epsilon = 0.0  # greedy

    rng = np.random.default_rng(configs["env"]["env"].get("seed"))
    n_actions = env.action_space_plugin.n_actions

    obs, _ = env.reset()
    done = False
    while not done:
        if algo is not None:
            actions = algo.select_actions(obs)
        else:
            actions = {aid: int(rng.integers(n_actions)) for aid in obs}
        step_out = env.step(actions)
        obs = step_out["obs"]
        done = step_out["terminated"] or step_out["truncated"]

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser("Render a single episode (requires a display)")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--experiment-file", default="experiment.yaml")
    p.add_argument(
        "--load-path",
        default=None,
        help="Trained checkpoint to render; omit to drive agents randomly.",
    )
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    render_episode(args.config_dir, args.experiment_file, args.load_path, args.seed)
