"""
Evaluation script.

Loads env + algorithm from YAML configs, runs evaluation episodes,
and prints episode-level statistics collected from env.step() outputs.
"""

import logging
import numpy as np
from collections import defaultdict

import baselines  # noqa: F401 — triggers auto-registration
from baselines.registry import get as get_algorithm
from multi_agent_package.scripts.run_from_config import (
    load_all_configs,
    build_environment,
)

LOGGER = logging.getLogger("evaluate")


def evaluate(
    config_dir: str = "configs", episodes: int = 10, load_path: str = None
) -> dict:
    configs = load_all_configs(config_dir)
    configs["env"]["env"]["render_mode"] = None  # headless evaluation

    env = build_environment(configs)

    algo_cfg = configs["experiment"]["experiment"]["algorithm"]
    algo_cls = get_algorithm(algo_cfg["name"])
    algo_params = algo_cfg.get("params", {})

    if load_path:
        algo = algo_cls.load(env, algo_params, load_path)
    else:
        algo = algo_cls(env, algo_params)
        LOGGER.warning(
            "No --load-path given: evaluating an UNTRAINED %s (random / zero-"
            "initialised policy). Pass --load-path to evaluate a trained model.",
            algo_cfg["name"],
        )

    # greedy evaluation regardless of the configured training epsilon
    if hasattr(algo, "epsilon"):
        algo.epsilon = 0.0

    episode_lengths = []
    agent_returns = defaultdict(list)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_len = 0
        ep_reward = defaultdict(float)

        while not done:
            actions = algo.select_actions(obs)
            step_out = env.step(actions)
            obs = step_out["obs"]
            rewards = step_out["reward"]
            done = step_out["terminated"] or step_out["truncated"]
            ep_len += 1
            for agent_id, r in rewards.items():
                ep_reward[agent_id] += float(r)

        episode_lengths.append(ep_len)
        for agent_id, total in ep_reward.items():
            agent_returns[agent_id].append(total)

        LOGGER.info(
            "Episode %d/%d | length=%d | %s",
            ep + 1,
            episodes,
            ep_len,
            ", ".join(f"{k}={v:.2f}" for k, v in ep_reward.items()),
        )

    env.close()

    summary = {
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
    }
    for agent_id, returns in agent_returns.items():
        summary[f"mean_return_{agent_id}"] = float(np.mean(returns))

    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    p = argparse.ArgumentParser("Evaluate a trained policy from YAML configs")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--load-path",
        default=None,
        help="Trained checkpoint (.pkl) to evaluate. Without it, an untrained "
        "policy is evaluated and a warning is printed.",
    )
    args = p.parse_args()

    result = evaluate(args.config_dir, episodes=args.episodes, load_path=args.load_path)
    print("\n=== Evaluation Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v:.3f}")
