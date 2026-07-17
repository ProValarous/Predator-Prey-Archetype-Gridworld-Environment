"""
Shared pytest fixtures for all architecture layers.
"""

import sys
import os
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_agent_package.core.agent import Agent  # noqa: E402
from multi_agent_package.core.gridworld import GridWorldEnv  # noqa: E402

# ------------------------------------------------------------------
# Agent fixtures
# ------------------------------------------------------------------


@pytest.fixture
def predator():
    return Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1")


@pytest.fixture
def prey():
    return Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1")


@pytest.fixture
def two_predators_one_prey():
    return [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="predator", agent_team="predator_2", agent_name="pred_2"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]


@pytest.fixture
def one_predator_one_prey():
    return [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]


# ------------------------------------------------------------------
# Environment fixtures
# ------------------------------------------------------------------


@pytest.fixture
def small_env(one_predator_one_prey):
    """Tiny env: 5×5, no obstacles, deterministic seed."""
    env = GridWorldEnv(
        agents=one_predator_one_prey,
        size=5,
        perc_num_obstacle=0,
        render_mode=None,
        seed=42,
    )
    return env


@pytest.fixture
def env_3agents(two_predators_one_prey):
    """5×5 env with 2 predators and 1 prey."""
    env = GridWorldEnv(
        agents=two_predators_one_prey,
        size=5,
        perc_num_obstacle=0,
        render_mode=None,
        seed=0,
    )
    return env


@pytest.fixture
def obstacle_env(one_predator_one_prey):
    """Env with ~40% obstacles."""
    env = GridWorldEnv(
        agents=one_predator_one_prey,
        size=8,
        perc_num_obstacle=40,
        render_mode=None,
        seed=7,
    )
    return env


# ------------------------------------------------------------------
# Baseline config fixtures
# ------------------------------------------------------------------


@pytest.fixture
def iql_config():
    return {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "seed": 0,
    }


@pytest.fixture
def cql_config():
    return {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "seed": 0,
    }


@pytest.fixture
def mixed_config():
    return {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "predator_algo": "cql",
        "prey_algo": "iql",
        "seed": 0,
    }


@pytest.fixture
def dqn_config():
    return {
        "hidden_layers": [8, 8],
        "learning_rate": 0.01,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "buffer_size": 200,
        "batch_size": 8,
        "min_replay_size": 8,
        "target_update_interval": 5,
        "episodes": 3,
        "log_interval": 1,
        "verbose": False,
        "seed": 0,
    }


@pytest.fixture
def dqn_env(one_predator_one_prey):
    """5×5 env with local_only observation + discrete_5 actions wired for DQN."""
    from multi_agent_package.observations.local_only import LocalOnlyObservation
    from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
    from multi_agent_package.rewards.base_reward import BaseReward

    env = GridWorldEnv(
        agents=one_predator_one_prey,
        size=5,
        perc_num_obstacle=0,
        render_mode=None,
        seed=42,
    )
    observation_builder = LocalOnlyObservation()
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode
    env.action_space_plugin = DiscreteActionSpace()
    # base reward now enters through the plugin pipeline, not gridworld.step();
    # attach it so training sees the capture/step-cost signal (issue #32)
    env.reward_fn = BaseReward(weight=1.0).compute
    return env
