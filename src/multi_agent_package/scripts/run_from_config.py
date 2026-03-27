"""
Main experiment launcher.

Responsibilities:
- Load YAML configs
- Build environment
- Attach observation + reward plug-ins
- Instantiate algorithm via registry
- Launch training

NO learning logic lives here.
"""

import yaml
from pathlib import Path

# Force baseline auto-registration
import baselines
from baselines.registry import get as get_algorithm

from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.core.agent import Agent
from multi_agent_package.registry import (
    get_observation_builder,
    get_reward_function,
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]


# -------------------------------------------------
# YAML Loader
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = "configs") -> dict:
    base = REPO_ROOT / config_dir

    return {
        "env": load_yaml(base / "env.yaml"),
        "agents": load_yaml(base / "agents.yaml"),
        "observations": load_yaml(base / "observations.yaml"),
        "rewards": load_yaml(base / "rewards.yaml"),
        "experiment": load_yaml(base / "dqn_experiment.yaml"),
    }


# -------------------------------------------------
# Agent Builder
# -------------------------------------------------

def build_agents(agent_cfg: dict):
    agents = []

    predators = agent_cfg["agents"]["predators"]
    preys = agent_cfg["agents"]["preys"]

    for i in range(predators["count"]):
        ag = Agent(
            agent_type="predator",
            agent_team=f"predator_{i+1}",
            agent_name=f"predator_{i+1}",
        )
        ag.agent_speed = predators["speed"]
        ag.stamina = predators["stamina"]
        agents.append(ag)

    for i in range(preys["count"]):
        ag = Agent(
            agent_type="prey",
            agent_team=f"prey_{i+1}",
            agent_name=f"prey_{i+1}",
        )
        ag.agent_speed = preys["speed"]
        ag.stamina = preys["stamina"]
        agents.append(ag)

    return agents


# -------------------------------------------------
# Environment Builder
# -------------------------------------------------

def build_environment(configs: dict) -> GridWorldEnv:
    env_cfg = configs["env"]
    agent_cfg = configs["agents"]
    obs_cfg = configs["observations"]
    reward_cfg = configs["rewards"]

    agents = build_agents(agent_cfg)

    env = GridWorldEnv(
        agents=agents,
        size=env_cfg["env"]["size"],
        perc_num_obstacle=env_cfg["env"]["obstacle_percentage"],
        render_mode=env_cfg["env"]["render_mode"],
        window_size=env_cfg["env"]["window_size"],
        seed=env_cfg["env"]["seed"],
    )

    # -----------------------------
    # Attach Observation Wrapper
    # -----------------------------
    obs_type = obs_cfg["observations"]["type"]
    obs_params = obs_cfg["observations"].get("params", {})

    observation_builder = get_observation_builder(
        obs_type,
        **obs_params,
    )

    env.observation_builder = observation_builder.build

    # -----------------------------
    # Attach Reward Wrapper(s)
    # -----------------------------
    reward_fns = []

    if reward_cfg["rewards"]["base"]["enabled"]:
        reward_fns.append(get_reward_function("base"))

    for r in reward_cfg["rewards"].get("shaping", []):
        reward_fns.append(
            get_reward_function(
                r["name"],
                weight=r.get("weight", 1.0),
            )
        )

    def combined_reward(env_instance):
        total = {ag.agent_name: 0.0 for ag in env_instance.agents}

        for rf in reward_fns:
            r = rf.compute(env_instance)
            for k in total:
                total[k] += r.get(k, 0.0)

        return total

    env.reward_fn = combined_reward

    return env


# -------------------------------------------------
# Main Entry
# -------------------------------------------------

def main(config_dir: str = "configs"):
    configs = load_all_configs(config_dir)

    env = build_environment(configs)

    algo_cfg = configs["experiment"]["algorithm"]

    algo_name = algo_cfg["name"]
    algo_params = algo_cfg.get("params", {})

    algo_cls = get_algorithm(algo_name)
    algorithm = algo_cls(env, algo_params)

    algorithm.train()

    env.close()


if __name__ == "__main__":
    main("configs")