"""
Tests for the vanilla DQN baseline.
"""

import os
import tempfile

import numpy as np

import baselines
from baselines.DQN.dqn import DQN
from baselines.registry.algorithm_registry import get
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.registry import get_action_space, get_observation_builder, get_reward_function


def make_env(seed=42):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    env = GridWorldEnv(agents=agents, size=5, perc_num_obstacle=0, render_mode=None, seed=seed)
    observation_builder = get_observation_builder("default")
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode
    env.reward_fn = lambda env_instance: get_reward_function("base").compute(env_instance)
    env.action_space_plugin = get_action_space("discrete_5")
    return env


def base_config(**overrides):
    cfg = {
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "episodes": 5,
        "batch_size": 2,
        "buffer_size": 100,
        "learning_rate": 1e-3,
        "hidden_dim": 32,
        "target_update_interval": 2,
        "min_replay_size": 2,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


class TestDQNInit:
    def test_registry_has_dqn(self):
        assert get("dqn") is DQN

    def test_agent_ids_discovered(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.agent_ids) == {"pred_1", "prey_1"}

    def test_action_dim_comes_from_action_plugin(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert algo.action_dim == env.action_space_plugin.n_actions

    def test_policy_and_target_networks_created_per_agent(self):
        env = make_env()
        algo = DQN(env, base_config())
        assert set(algo.policy_networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.target_networks.keys()) == {"pred_1", "prey_1"}


class TestDQNActionSelection:
    def test_returns_dict_for_all_agents(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}

    def test_actions_are_in_valid_range(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for action in actions.values():
                assert 0 <= action < env.action_space_plugin.n_actions

    def test_greedy_action_uses_highest_q_value(self):
        env = make_env()
        algo = DQN(env, base_config(epsilon=0.0))
        obs, _ = env.reset()
        tensor = algo._obs_to_tensor(obs["pred_1"])
        with np.errstate(all="ignore"):
            q_values = algo.policy_networks["pred_1"](tensor)
        best_action = int(np.argmax(q_values.detach().cpu().numpy()[0]))
        actions = algo.select_actions(obs)
        assert actions["pred_1"] == best_action

    def test_encoder_produces_numeric_vector(self):
        env = make_env()
        algo = DQN(env, base_config())
        obs, _ = env.reset()
        encoded = algo._encode_observation(obs["pred_1"])
        assert encoded.dtype == np.float32
        assert encoded.ndim == 1
        assert encoded.shape[0] == algo.state_dim


class TestDQNTraining:
    def test_training_populates_replay_buffers(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=3, epsilon=1.0))
        algo.train()
        for buffer in algo.replay_buffers.values():
            assert len(buffer) > 0

    def test_target_networks_match_policy_at_init(self):
        env = make_env()
        algo = DQN(env, base_config())
        for agent_id in algo.agent_ids:
            for key, value in algo.policy_networks[agent_id].state_dict().items():
                target_value = algo.target_networks[agent_id].state_dict()[key]
                assert np.array_equal(value.cpu().numpy(), target_value.cpu().numpy())


class TestDQNPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=2, epsilon=1.0))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_networks(self):
        env = make_env()
        algo = DQN(env, base_config(episodes=2, epsilon=1.0))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            loaded = DQN.load(make_env(), base_config(epsilon=0.0), path)
            assert set(loaded.policy_networks.keys()) == {"pred_1", "prey_1"}
        finally:
            os.unlink(path)
