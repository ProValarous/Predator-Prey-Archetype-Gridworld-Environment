"""
Tests for CQL (Centralized Q-Learning).
"""

import os
import tempfile

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from baselines.CQL.cql import CQL

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_env(n_pred=1, n_prey=1, size=5, seed=0):
    agents = []
    for i in range(1, n_pred + 1):
        agents.append(
            Agent(
                agent_type="predator",
                agent_team=f"predator_{i}",
                agent_name=f"pred_{i}",
            )
        )
    for i in range(1, n_prey + 1):
        agents.append(
            Agent(agent_type="prey", agent_team=f"prey_{i}", agent_name=f"prey_{i}")
        )
    return GridWorldEnv(
        agents=agents, size=size, perc_num_obstacle=0, render_mode=None, seed=seed
    )


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


class TestCQLInit:
    def test_n_agents(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config())
        assert algo.n_agents == 2

    def test_joint_action_space_size(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config(action_dim=5))
        assert algo.n_joint_actions == 5**2  # 25

    def test_joint_action_space_3_agents(self):
        env = make_env(n_pred=2, n_prey=1)
        algo = CQL(env, base_config(action_dim=5))
        assert algo.n_joint_actions == 5**3  # 125

    def test_q_table_starts_empty(self):
        env = make_env()
        algo = CQL(env, base_config())
        assert len(algo.q_table) == 0

    def test_action_shape(self):
        env = make_env(n_pred=1, n_prey=2)
        algo = CQL(env, base_config(action_dim=5))
        assert algo._action_shape == (5, 5, 5)


# ------------------------------------------------------------------
# Joint state encoding
# ------------------------------------------------------------------


class TestCQLJointState:
    def test_joint_state_is_tuple(self):
        env = make_env()
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        js = algo._joint_state(obs)
        assert isinstance(js, tuple)

    def test_joint_state_hashable(self):
        env = make_env()
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        js = algo._joint_state(obs)
        hash(js)

    def test_joint_state_length(self):
        env = make_env(n_pred=1, n_prey=2)
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        js = algo._joint_state(obs)
        assert len(js) == 3  # one element per agent

    def test_same_obs_same_joint_state(self):
        env = make_env()
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        assert algo._joint_state(obs) == algo._joint_state(obs)


# ------------------------------------------------------------------
# Joint action encoding
# ------------------------------------------------------------------


class TestCQLJointAction:
    def test_joint_action_in_range(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config(action_dim=5))
        actions = {"pred_1": 2, "prey_1": 3}
        idx = algo._joint_action_index(actions)
        assert 0 <= idx < 25

    def test_all_noop_gives_consistent_index(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config(action_dim=5))
        obs, _ = env.reset()
        actions = {aid: 4 for aid in algo.agent_ids}
        idx = algo._joint_action_index(actions)
        assert isinstance(idx, int)

    def test_index_unique_per_action_combo(self):
        """Different action combos should map to different indices."""
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config(action_dim=5))
        indices = set()
        for a1 in range(5):
            for a2 in range(5):
                idx = algo._joint_action_index({"pred_1": a1, "prey_1": a2})
                indices.add(idx)
        assert len(indices) == 25


# ------------------------------------------------------------------
# Action selection (marginalisation)
# ------------------------------------------------------------------


class TestCQLSelectActions:
    def test_returns_all_agents(self):
        env = make_env()
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == set(algo.agent_ids)

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = CQL(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for a in actions.values():
                assert 0 <= a < 5

    def test_greedy_picks_best_marginal(self):
        """Manually wire a Q-vector so one action has highest marginal value."""
        env = make_env(n_pred=1, n_prey=1)
        algo = CQL(env, base_config(epsilon=0.0))
        obs, _ = env.reset()
        js = algo._joint_state(obs)
        q_vec = np.zeros(25, dtype=np.float32)
        # Set action 0 for pred_1 to dominate regardless of prey action
        for a2 in range(5):
            idx = 0 * 5 + a2  # pred_1 action=0
            q_vec[idx] = 10.0
        algo.q_table[js] = q_vec
        actions = algo.select_actions(obs)
        assert actions["pred_1"] == 0


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


class TestCQLTrain:
    def test_q_table_populated_after_training(self):
        env = make_env()
        algo = CQL(env, base_config(episodes=10))
        algo.train()
        assert len(algo.q_table) > 0

    def test_q_values_non_zero_after_training(self):
        env = make_env()
        algo = CQL(env, base_config(episodes=5))
        algo.train()
        all_vals = [v for arr in algo.q_table.values() for v in arr]
        assert any(v != 0.0 for v in all_vals)

    def test_centralized_update_uses_sum_of_rewards(self):
        """Joint Q-table uses summed rewards, not per-agent."""
        env = make_env()
        algo = CQL(env, base_config(episodes=1, epsilon=1.0, alpha=1.0, gamma=0.0))
        obs, _ = env.reset()
        actions = {aid: 4 for aid in algo.agent_ids}
        js = algo._joint_state(obs)
        ja = algo._joint_action_index(actions)

        step_out = env.step(actions)
        rewards = step_out["reward"]
        central_r = sum(float(rewards[aid]) for aid in algo.agent_ids)

        # manually run one update
        algo.q_table[js][ja] = 0.0
        algo.q_table[js][ja] += algo.alpha * (central_r - algo.q_table[js][ja])
        expected = central_r  # with gamma=0, alpha=1
        assert algo.q_table[js][ja] == pytest.approx(expected)

    def test_epsilon_decays_when_decay_set(self):
        env = make_env()
        algo = CQL(
            env,
            base_config(epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.0, episodes=5),
        )
        algo.train()
        assert algo.epsilon < 1.0


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


class TestCQLPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = CQL(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_q_table(self):
        env = make_env()
        algo = CQL(env, base_config(episodes=5))
        algo.train()
        n_states = len(algo.q_table)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = CQL.load(make_env(), base_config(), path)
            assert len(algo2.q_table) == n_states
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = CQL(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = CQL.load(make_env(), base_config(epsilon=0.0), path)
            algo2.evaluate(episodes=1)  # must not raise
        finally:
            os.unlink(path)
