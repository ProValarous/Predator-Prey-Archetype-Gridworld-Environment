"""
Tests for IQL (Independent Q-Learning).
"""

import os
import tempfile

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from baselines.IQL.iql import IQL


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_env(seed=42):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    return GridWorldEnv(agents=agents, size=5, perc_num_obstacle=0, render_mode=None, seed=seed)


def base_config(**overrides):
    cfg = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestIQLInit:
    def test_agent_ids_discovered(self):
        env = make_env()
        algo = IQL(env, base_config())
        assert set(algo.agent_ids) == {"pred_1", "prey_1"}

    def test_q_tables_created_per_agent(self):
        env = make_env()
        algo = IQL(env, base_config())
        assert set(algo.q_tables.keys()) == {"pred_1", "prey_1"}

    def test_q_tables_start_empty(self):
        env = make_env()
        algo = IQL(env, base_config())
        for table in algo.q_tables.values():
            assert len(table) == 0

    def test_config_values_applied(self):
        env = make_env()
        algo = IQL(env, base_config(alpha=0.5, gamma=0.9, epsilon=0.2))
        assert algo.alpha == 0.5
        assert algo.gamma == 0.9
        assert algo.epsilon == 0.2


# ------------------------------------------------------------------
# State encoding
# ------------------------------------------------------------------

class TestIQLEncodeState:
    def test_returns_hashable(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs, _ = env.reset()
        s = algo._encode_state(obs["pred_1"])
        hash(s)  # must not raise

    def test_same_obs_same_encoding(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs, _ = env.reset()
        s1 = algo._encode_state(obs["pred_1"])
        s2 = algo._encode_state(obs["pred_1"])
        assert s1 == s2

    def test_different_obs_gives_different_encoding(self):
        """Two obs dicts with different local positions must encode differently."""
        env = make_env()
        algo = IQL(env, base_config())
        obs_a = {"local": np.array([0, 0]), "global": None}
        obs_b = {"local": np.array([3, 4]), "global": None}
        assert algo._encode_state(obs_a) != algo._encode_state(obs_b)

    def test_numpy_array_encoded(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs = {"local": np.array([2, 3]), "global": None}
        s = algo._encode_state(obs)
        assert isinstance(s, tuple)

    def test_nested_dict_encoded(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs = {"local": np.array([0, 0]), "global": {"dist": {"other": 5}}}
        s = algo._encode_state(obs)
        hash(s)  # must not raise


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------

class TestIQLSelectActions:
    def test_returns_dict_for_all_agents(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = IQL(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for a in actions.values():
                assert 0 <= a < 5

    def test_greedy_selects_best_action(self):
        env = make_env()
        algo = IQL(env, base_config(epsilon=0.0))
        obs, _ = env.reset()
        s = algo._encode_state(obs["pred_1"])
        # Manually set action 3 as best
        algo.q_tables["pred_1"][s][3] = 100.0
        actions = algo.select_actions(obs)
        assert actions["pred_1"] == 3

    def test_epsilon_one_gives_random(self):
        """With epsilon=1.0, all actions are random. Distribution should vary."""
        env = make_env()
        algo = IQL(env, base_config(epsilon=1.0, seed=0))
        obs, _ = env.reset()
        action_set = set()
        for _ in range(50):
            actions = algo.select_actions(obs)
            action_set.add(actions["pred_1"])
        assert len(action_set) > 1


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

class TestIQLTrain:
    def test_q_tables_populated_after_training(self):
        env = make_env()
        algo = IQL(env, base_config(episodes=10))
        algo.train()
        for table in algo.q_tables.values():
            assert len(table) > 0

    def test_epsilon_decays(self):
        env = make_env()
        algo = IQL(env, base_config(epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.0, episodes=5))
        initial_eps = algo.epsilon
        algo.train()
        assert algo.epsilon < initial_eps

    def test_epsilon_respects_min(self):
        env = make_env()
        algo = IQL(env, base_config(epsilon=1.0, epsilon_decay=0.1, min_epsilon=0.05, episodes=20))
        algo.train()
        assert algo.epsilon >= 0.05

    def test_td_update_changes_q_value(self):
        """After one step, at least one Q-value should differ from zero."""
        env = make_env()
        algo = IQL(env, base_config(episodes=1, epsilon=1.0))
        algo.train()
        all_values = [
            v
            for table in algo.q_tables.values()
            for arr in table.values()
            for v in arr
        ]
        assert any(v != 0.0 for v in all_values)


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------

class TestIQLPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = IQL(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_q_tables(self):
        env = make_env()
        algo = IQL(env, base_config(episodes=5))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = IQL.load(make_env(), base_config(), path)
            for aid in algo.agent_ids:
                # states learned should be a subset of loaded tables
                assert len(algo2.q_tables[aid]) >= len(algo.q_tables[aid])
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = IQL(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = IQL.load(make_env(), base_config(epsilon=0.0), path)
            algo2.evaluate(episodes=1)  # must not raise
        finally:
            os.unlink(path)
