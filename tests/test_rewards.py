"""
Tests for all three reward functions.
"""

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.rewards.base_reward import BaseReward
from multi_agent_package.rewards.predator_distance import PredatorDistanceReward
from multi_agent_package.rewards.survival_reward import SurvivalReward


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(n_pred=1, n_prey=1, size=6, seed=0):
    agents = []
    for i in range(1, n_pred + 1):
        agents.append(Agent(agent_type="predator", agent_team=f"predator_{i}", agent_name=f"pred_{i}"))
    for i in range(1, n_prey + 1):
        agents.append(Agent(agent_type="prey", agent_team=f"prey_{i}", agent_name=f"prey_{i}"))
    env = GridWorldEnv(agents=agents, size=size, perc_num_obstacle=0, render_mode=None, seed=seed)
    env.reset()
    return env


# ------------------------------------------------------------------
# BaseReward
# ------------------------------------------------------------------

class TestBaseReward:
    def test_returns_all_agent_keys(self):
        env = make_env()
        reward_fn = BaseReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert set(rewards.keys()) == {"pred_1", "prey_1"}

    def test_step_cost_applied_to_predator(self):
        env = make_env()
        obs, _ = env.reset()
        env.step({ag.agent_name: 4 for ag in env.agents})
        reward_fn = BaseReward(weight=1.0)
        rewards = reward_fn.compute(env)
        # -5 step cost, no capture, no obstacle
        assert rewards["pred_1"] == pytest.approx(-5.0)

    def test_weight_scales_reward(self):
        env = make_env()
        obs, _ = env.reset()
        env.step({ag.agent_name: 4 for ag in env.agents})
        reward_fn = BaseReward(weight=2.0)
        rewards = reward_fn.compute(env)
        assert rewards["pred_1"] == pytest.approx(-10.0)

    def test_values_are_floats(self):
        env = make_env()
        obs, _ = env.reset()
        env.step({ag.agent_name: 4 for ag in env.agents})
        reward_fn = BaseReward(weight=1.0)
        rewards = reward_fn.compute(env)
        for v in rewards.values():
            assert isinstance(v, float)


# ------------------------------------------------------------------
# PredatorDistanceReward
# ------------------------------------------------------------------

class TestPredatorDistanceReward:
    def test_returns_all_agent_keys(self):
        env = make_env()
        reward_fn = PredatorDistanceReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert set(rewards.keys()) == {"pred_1", "prey_1"}

    def test_predator_reward_is_negative(self):
        """Predator should receive negative reward proportional to distance."""
        env = make_env()
        env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
        env.agents[1]._agent_location = np.array([3, 4], dtype=np.int32)
        reward_fn = PredatorDistanceReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["pred_1"] < 0

    def test_prey_reward_is_zero(self):
        env = make_env()
        reward_fn = PredatorDistanceReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["prey_1"] == pytest.approx(0.0)

    def test_closer_predator_gets_less_negative_reward(self):
        env = make_env()
        env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
        env.agents[1]._agent_location = np.array([5, 5], dtype=np.int32)
        reward_fn = PredatorDistanceReward(weight=1.0)
        far_reward = reward_fn.compute(env)["pred_1"]

        env.agents[1]._agent_location = np.array([1, 0], dtype=np.int32)
        close_reward = reward_fn.compute(env)["pred_1"]

        assert close_reward > far_reward

    def test_weight_scales_predator_reward(self):
        env = make_env()
        env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
        env.agents[1]._agent_location = np.array([3, 0], dtype=np.int32)

        reward_fn_1 = PredatorDistanceReward(weight=1.0)
        reward_fn_2 = PredatorDistanceReward(weight=2.0)
        r1 = reward_fn_1.compute(env)["pred_1"]
        r2 = reward_fn_2.compute(env)["pred_1"]
        assert r2 == pytest.approx(r1 * 2)

    def test_no_prey_gives_zero_to_all(self):
        agents = [Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1")]
        env = GridWorldEnv(agents=agents, size=5, perc_num_obstacle=0, render_mode=None, seed=0)
        env.reset()
        reward_fn = PredatorDistanceReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["pred_1"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# SurvivalReward
# ------------------------------------------------------------------

class TestSurvivalReward:
    def test_returns_all_agent_keys(self):
        env = make_env()
        reward_fn = SurvivalReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert set(rewards.keys()) == {"pred_1", "prey_1"}

    def test_prey_gets_positive_reward(self):
        env = make_env()
        reward_fn = SurvivalReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["prey_1"] == pytest.approx(1.0)

    def test_predator_gets_zero(self):
        env = make_env()
        reward_fn = SurvivalReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["pred_1"] == pytest.approx(0.0)

    def test_weight_scales_prey_reward(self):
        env = make_env()
        reward_fn = SurvivalReward(weight=5.0)
        rewards = reward_fn.compute(env)
        assert rewards["prey_1"] == pytest.approx(5.0)

    def test_multiple_prey_all_get_reward(self):
        env = make_env(n_pred=1, n_prey=2)
        reward_fn = SurvivalReward(weight=1.0)
        rewards = reward_fn.compute(env)
        assert rewards["prey_1"] == pytest.approx(1.0)
        assert rewards["prey_2"] == pytest.approx(1.0)
