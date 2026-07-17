"""
Tests for MixedTrainer (per-team CQL/IQL assignment).
"""

import os
import tempfile

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.rewards.base_reward import BaseReward
from baselines.MIXED.mix_train import MixedTrainer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_env(seed=0):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="predator", agent_team="predator_2", agent_name="pred_2"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    env = GridWorldEnv(
        agents=agents, size=6, perc_num_obstacle=0, render_mode=None, seed=seed
    )
    # base reward now enters through the plugin pipeline, not gridworld.step();
    # attach it so training sees the capture/step-cost signal (issue #32)
    env.reward_fn = BaseReward(weight=1.0).compute
    return env


def base_config(predator_algo="cql", prey_algo="iql", **overrides):
    cfg = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "predator_algo": predator_algo,
        "prey_algo": prey_algo,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# Team partitioning
# ------------------------------------------------------------------


class TestMixedTeamPartitioning:
    def test_predators_identified(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        assert set(algo._predators) == {"pred_1", "pred_2"}

    def test_prey_identified(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        assert set(algo._prey) == {"prey_1"}

    def test_team_lookup_predators(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        assert algo._team_of["pred_1"] == "predator"
        assert algo._team_of["pred_2"] == "predator"

    def test_team_lookup_prey(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        assert algo._team_of["prey_1"] == "prey"


# ------------------------------------------------------------------
# Q-table structure based on config
# ------------------------------------------------------------------


class TestMixedQTableStructure:
    def test_cql_predators_get_shared_table(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(predator_algo="cql", prey_algo="iql"))
        assert "predator" in algo._cql_tables
        assert "prey" not in algo._cql_tables

    def test_iql_prey_gets_individual_tables(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(predator_algo="cql", prey_algo="iql"))
        assert "prey_1" in algo._iql_tables
        assert "pred_1" not in algo._iql_tables
        assert "pred_2" not in algo._iql_tables

    def test_iql_predators_iql_prey(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(predator_algo="iql", prey_algo="iql"))
        assert "pred_1" in algo._iql_tables
        assert "pred_2" in algo._iql_tables
        assert "prey_1" in algo._iql_tables
        assert len(algo._cql_tables) == 0

    def test_cql_both_teams(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(predator_algo="cql", prey_algo="cql"))
        assert "predator" in algo._cql_tables
        assert "prey" in algo._cql_tables
        assert len(algo._iql_tables) == 0

    def test_cql_joint_action_size_for_predator_team(self):
        """2 predators with 5 actions → 5^2 = 25 joint actions."""
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="cql", prey_algo="iql", action_dim=5)
        )
        assert algo._cql_n_joint["predator"] == 25

    def test_cql_joint_action_size_for_prey_team(self):
        """1 prey with 5 actions → 5^1 = 5 joint actions."""
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="iql", prey_algo="cql", action_dim=5)
        )
        assert algo._cql_n_joint["prey"] == 5


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------


class TestMixedSelectActions:
    def test_returns_all_agents(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "pred_2", "prey_1"}

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = MixedTrainer(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for a in actions.values():
                assert 0 <= a < 5


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


class TestMixedTrain:
    def test_iql_tables_populated_after_training(self):
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="iql", prey_algo="iql", episodes=10)
        )
        algo.train()
        for aid, table in algo._iql_tables.items():
            assert len(table) > 0, f"IQL table for {aid} empty after training"

    def test_cql_table_populated_for_predator_team(self):
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="cql", prey_algo="iql", episodes=10)
        )
        algo.train()
        assert len(algo._cql_tables["predator"]) > 0

    def test_iql_table_populated_for_prey(self):
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="cql", prey_algo="iql", episodes=10)
        )
        algo.train()
        assert len(algo._iql_tables["prey_1"]) > 0

    def test_mixed_all_cql_tables_populated(self):
        env = make_env()
        algo = MixedTrainer(
            env, base_config(predator_algo="cql", prey_algo="cql", episodes=10)
        )
        algo.train()
        assert len(algo._cql_tables["predator"]) > 0
        assert len(algo._cql_tables["prey"]) > 0

    def test_cql_reward_uses_team_sum(self):
        """CQL Q-value = sum of team rewards, not individual (alpha=1, gamma=0)."""
        from unittest.mock import patch

        env = make_env()
        algo = MixedTrainer(
            env,
            base_config(
                predator_algo="cql",
                prey_algo="iql",
                alpha=1.0,
                gamma=0.0,
                epsilon=0.0,
                episodes=1,
            ),
        )

        # Inject a fixed one-step episode with known per-agent rewards.
        # pred_1=+10, pred_2=+20 → team sum must be 30, not 10 or 20.
        fake_obs = {
            ag.agent_name: {"local": np.array([0, 0]), "global": None}
            for ag in env.agents
        }

        def fake_step(actions):
            return {
                "obs": fake_obs,
                "reward": {"pred_1": 10.0, "pred_2": 20.0, "prey_1": 3.0},
                "terminated": True,  # one-step episode → done=True, no bootstrap
                "truncated": False,
                "info": {},
            }

        with patch.object(env, "step", side_effect=fake_step):
            algo.train()

        # With alpha=1, gamma=0, done=True:
        #   Q(s,a) = 0 + 1.0 * (central_r + 0 - 0) = central_r = 30.0
        non_zero = [
            q
            for q_vec in algo._cql_tables["predator"].values()
            for q in q_vec
            if q != 0.0
        ]
        assert len(non_zero) > 0, "No Q-values updated — training did not run"
        assert all(
            q == pytest.approx(30.0) for q in non_zero
        ), f"Expected team sum 30.0; got {non_zero}"


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


class TestMixedPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_tables(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(episodes=5))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = MixedTrainer.load(make_env(), base_config(), path)
            # CQL table should have same number of states
            if algo._cql_tables:
                for tk in algo._cql_tables:
                    assert len(algo2._cql_tables[tk]) == len(algo._cql_tables[tk])
            if algo._iql_tables:
                for aid in algo._iql_tables:
                    assert len(algo2._iql_tables[aid]) == len(algo._iql_tables[aid])
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = MixedTrainer(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = MixedTrainer.load(make_env(), base_config(epsilon=0.0), path)
            algo2.evaluate(episodes=1)  # must not raise
        finally:
            os.unlink(path)
