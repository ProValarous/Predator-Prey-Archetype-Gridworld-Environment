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
    get_action_space,
)
from multi_agent_package.wrappers.speed import SpeedWrapper

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


def load_all_configs(
    config_dir: str = "configs",
    experiment_file: str = "experiment.yaml",
) -> dict:
    base = REPO_ROOT / config_dir

    return {
        "env": load_yaml(base / "env.yaml"),
        "agents": load_yaml(base / "agents.yaml"),
        "observations": load_yaml(base / "observations.yaml"),
        "rewards": load_yaml(base / "rewards.yaml"),
        "actions": load_yaml(base / "actions.yaml"),
        "experiment": load_yaml(base / experiment_file),
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
    action_cfg = configs["actions"]

    agents = build_agents(agent_cfg)

    dynamics = env_cfg["env"].get("dynamics", {})
    termination = env_cfg["env"].get("termination", {})

    env = GridWorldEnv(
        agents=agents,
        size=env_cfg["env"]["size"],
        perc_num_obstacle=env_cfg["env"]["obstacle_percentage"],
        render_mode=env_cfg["env"]["render_mode"],
        window_size=env_cfg["env"]["window_size"],
        seed=env_cfg["env"]["seed"],
        allow_cell_sharing=dynamics.get("allow_cell_sharing", True),
        block_agents_by_obstacles=dynamics.get("block_agents_by_obstacles", True),
        capture_threshold=termination.get("capture_threshold", 1),
        max_steps=termination.get("max_steps", None),
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
    env.observation_encoder = observation_builder.encode

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

    # -----------------------------
    # Attach Action Space
    # -----------------------------
    action_type = action_cfg["actions"]["type"]
    action_params = action_cfg["actions"].get("params", {})

    env.action_space_plugin = get_action_space(action_type, **action_params)

    # -----------------------------
    # Speed wrapper (must wrap last: it proxies unknown attributes to the
    # inner env via __getattr__, so observation_encoder/action_space_plugin
    # must already be attached above for the proxy to expose them)
    # -----------------------------
    env = SpeedWrapper(env)

    return env


# -------------------------------------------------
# Main Entry
# -------------------------------------------------

def main(config_dir: str = "configs"):
    configs = load_all_configs(config_dir)

    env = build_environment(configs)

    algo_cfg = configs["experiment"]["experiment"]["algorithm"]

    algo_name = algo_cfg["name"]
    algo_params = algo_cfg.get("params", {})

    algo_cls = get_algorithm(algo_name)
    algorithm = algo_cls(env, algo_params)

    algorithm.train()

    env.close()


if __name__ == "__main__":
    main("configs")