"""
Reproducibility guard: the project's core promise is that a fixed seed plus a
fixed configuration yields an identical trajectory. These tests reproduce a
seeded episode (and a config-built environment) twice and assert byte-for-byte
identical outcomes, so any accidental global randomness or state leak is caught
in CI.
"""

import numpy as np

from ppage.core.agent import Agent
from ppage.core.gridworld import GridWorldEnv
from ppage.rewards.base_reward import BaseReward


def _make_env(seed):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    env = GridWorldEnv(
        agents=agents,
        size=6,
        perc_num_obstacle=20,
        render_mode=None,
        seed=seed,
        max_steps=40,
    )
    env.reward_fn = BaseReward(weight=1.0).compute
    return env


def _run_trajectory(seed, actions_seq):
    env = _make_env(seed)
    env.reset()
    trace = []
    for actions in actions_seq:
        out = env.step(actions)
        positions = tuple(
            (int(a._agent_location[0]), int(a._agent_location[1])) for a in env.agents
        )
        rewards = tuple(sorted((k, round(v, 6)) for k, v in out["reward"].items()))
        trace.append((positions, rewards, out["terminated"], out["truncated"]))
        if out["terminated"] or out["truncated"]:
            break
    return trace


ACTIONS = [
    {"pred_1": 0, "prey_1": 4},
    {"pred_1": 1, "prey_1": 2},
    {"pred_1": 2, "prey_1": 1},
    {"pred_1": 3, "prey_1": 0},
] * 8


def test_same_seed_reproduces_reset():
    a, b = _make_env(2024), _make_env(2024)
    a.reset()
    b.reset()
    # identical obstacle layout and identical agent start positions
    assert [tuple(o) for o in a._obstacle_location] == [
        tuple(o) for o in b._obstacle_location
    ]
    for ag_a, ag_b in zip(a.agents, b.agents):
        np.testing.assert_array_equal(ag_a._agent_location, ag_b._agent_location)


def test_same_seed_reproduces_full_trajectory():
    t1 = _run_trajectory(2024, ACTIONS)
    t2 = _run_trajectory(2024, ACTIONS)
    assert t1 == t2
    assert len(t1) > 1  # the episode actually ran multiple steps


def test_build_environment_is_seed_deterministic():
    from run_from_config import (
        load_all_configs,
        build_environment,
    )

    def first_obs():
        cfg = load_all_configs(experiment_file="experiment_iql.yaml")
        cfg["env"]["env"]["render_mode"] = None
        cfg["env"]["env"]["seed"] = 999
        env = build_environment(cfg)
        obs, _ = env.reset()
        return {k: np.asarray(v["local"]) for k, v in obs.items()}

    o1, o2 = first_obs(), first_obs()
    assert set(o1) == set(o2)
    for k in o1:
        np.testing.assert_array_equal(o1[k], o2[k])
