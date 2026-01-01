"""
Main experiment launcher.

Loads YAML configs, builds environment, attaches observation + reward logic,
and runs a simple rollout loop.

NO learning logic lives here.
"""

import yaml
from pathlib import Path
import sys

# Resolve repo root robustly
REPO_ROOT = Path(__file__).resolve().parents[3]

from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.core.agent import Agent
from multi_agent_package.registry import (
    get_observation_builder,
    get_reward_function,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def run(config_dir: str = "configs"):
    base = REPO_ROOT / config_dir

    env_cfg = load_yaml(base / "env.yaml")
    agent_cfg = load_yaml(base / "agents.yaml")
    obs_cfg = load_yaml(base / "observations.yaml")
    reward_cfg = load_yaml(base / "rewards.yaml")
    exp_cfg = load_yaml(base / "experiment.yaml")

    agents = build_agents(agent_cfg)

    env = GridWorldEnv(
        agents=agents,
        size=env_cfg["env"]["size"],
        perc_num_obstacle=env_cfg["env"]["obstacle_percentage"],
        render_mode=env_cfg["env"]["render_mode"],
        window_size=env_cfg["env"]["window_size"],
        seed=env_cfg["env"]["seed"],
    )

    # attach observation builder
    obs_type = obs_cfg["observations"]["type"]
    obs_params = obs_cfg["observations"].get("params", {})
    env.observation_builder = get_observation_builder(obs_type, **obs_params).build

    # attach reward functions
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

    def combined_reward(env):
        total = {ag.agent_name: 0.0 for ag in env.agents}
        for rf in reward_fns:
            r = rf.compute(env)
            for k in total:
                total[k] += r.get(k, 0.0)
        return total

    env.reward_fn = combined_reward

    episodes = exp_cfg["experiment"]["runtime"]["episodes"]

    for ep in range(episodes):
        obs, info = env.reset()
        done = False


        if env.render_mode == "human":
            env.render()

        while not done:
            actions = {ag.agent_name: ag.action_space.sample() for ag in env.agents}
            step_out = env.step(actions)
            done = step_out["terminated"]

    env.close()


if __name__ == "__main__":
    run("configs")
