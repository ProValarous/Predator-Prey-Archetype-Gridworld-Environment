"""
Tests for DQN (Independent Deep Q-Networks).
"""

import os
import tempfile

import numpy as np
import pytest

from baselines.DQN.dqn import DQN
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.observations.local_only import LocalOnlyObservation


def make_env(seed=0):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    env = GridWorldEnv(
        agents=agents,
        size=5,
        perc_num_obstacle=0,
        render_mode=None,
        seed=seed,
    )
    env.observation_builder = LocalOnlyObservation().build
    return env


def base_config(**overrides):
    cfg = {
        "alpha": 0.001,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "batch_size": 2,
        "buffer_size": 100,
        "target_update_interval": 5,
        "hidden_layers": [32, 32],
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


class TestDQNInit:
    def test_agent_ids_discovered(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.agent_ids) == {"pred_1", "prey_1"}

    def test_networks_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.q_networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.target_networks.keys()) == {"pred_1", "prey_1"}

    def test_replay_buffers_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.replay_buffers.keys()) == {"pred_1", "prey_1"}


class TestDQNEncoding:
    def test_encode_observation_returns_numeric_vector(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        vector = algo._encode_observation(obs["pred_1"])
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert vector.ndim == 1

    def test_vector_length_equals_obs_dim(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        vector = algo._encode_observation(obs["pred_1"])
        assert vector.shape[0] == algo.obs_dim

    def test_obs_dim_auto_computed(self):
        # obs_dim = 2 + max_other_agents*5 + max_visible_obstacles*3
        # 5 values per agent: rel_x, rel_y, dist, is_prey, speed
        env = make_env()
        algo = DQN(env, base_config(max_other_agents=1, max_visible_obstacles=0))
        assert algo.obs_dim == 2 + 1 * 5 + 0 * 3  # = 7

    def test_same_observation_same_encoding(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        encoded_a = algo._encode_observation(obs["pred_1"])
        encoded_b = algo._encode_observation(obs["pred_1"])
        assert np.array_equal(encoded_a, encoded_b)

    def test_structured_encoding_local_radius_obs(self):
        # verify that the structured encoder correctly places values at known indices
        # for a local_radius observation with 1 visible agent and no obstacles
        # per-agent slot layout: rel_x, rel_y, dist, is_prey, speed (5 values)
        env = make_env()
        algo = DQN(env, base_config(max_other_agents=1, max_visible_obstacles=0))
        obs = {
            "local": np.array([3, 4]),
            "visible_agents": {
                "prey_1": {"rel_pos": (2, -1), "dist": 3, "type": "prey", "speed": 1},
            },
            "visible_obstacles": {},
            "radius": 3,
        }
        vector = algo._encode_observation(obs)
        assert vector[0] == pytest.approx(3.0)   # self_x
        assert vector[1] == pytest.approx(4.0)   # self_y
        assert vector[2] == pytest.approx(2.0)   # prey rel_x
        assert vector[3] == pytest.approx(-1.0)  # prey rel_y
        assert vector[4] == pytest.approx(3.0)   # prey dist
        assert vector[5] == pytest.approx(1.0)   # is_prey = 1.0
        assert vector[6] == pytest.approx(1.0)   # speed = 1


class TestDQNSelectActions:
    def test_returns_dict_for_all_agents(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for action in actions.values():
                assert 0 <= action < 5


class TestDQNTrain:
    def test_replay_buffers_populated_after_training(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=5, epsilon=1.0))
        algo.train()
        for buffer in algo.replay_buffers.values():
            assert len(buffer) > 0

    def test_epsilon_decays(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=5, epsilon=1.0, epsilon_decay=0.5))
        initial_eps = algo.epsilon
        algo.train()
        assert algo.epsilon < initial_eps


class TestDQNPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = DQN.load(make_env(), base_config(epsilon=0.0), path)
            algo2.evaluate(episodes=1)
        finally:
            os.unlink(path)