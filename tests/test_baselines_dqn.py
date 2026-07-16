"""
Tests for the DQN baseline: ReplayBuffer, QNetwork, and the DQN algorithm.
"""

import pytest
import torch

from baselines.DQN.replay_buffer import ReplayBuffer
from baselines.DQN.q_network import QNetwork
from baselines.DQN.curve_recorder import CurveRecorder

# ------------------------------------------------------------------
# ReplayBuffer
# ------------------------------------------------------------------


class TestReplayBuffer:
    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0, state_dim=4)

    def test_rejects_non_positive_state_dim(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=10, state_dim=0)

    def test_len_starts_at_zero(self):
        buf = ReplayBuffer(capacity=10, state_dim=2)
        assert len(buf) == 0

    def test_push_increments_length(self):
        buf = ReplayBuffer(capacity=10, state_dim=2)
        buf.push([0, 0], 1, 1.0, [1, 0], False)
        assert len(buf) == 1

    def test_length_caps_at_capacity(self):
        buf = ReplayBuffer(capacity=3, state_dim=2, seed=0)
        for i in range(10):
            buf.push([i, i], 0, 0.0, [i, i], False)
        assert len(buf) == 3

    def test_sample_more_than_stored_raises(self):
        buf = ReplayBuffer(capacity=10, state_dim=2)
        buf.push([0, 0], 0, 0.0, [0, 0], False)
        with pytest.raises(ValueError):
            buf.sample(2)

    def test_sample_returns_correct_shapes(self):
        buf = ReplayBuffer(capacity=10, state_dim=3, seed=0)
        for i in range(5):
            buf.push([i, i, i], i % 2, float(i), [i + 1, i + 1, i + 1], i == 4)

        states, actions, rewards, next_states, dones = buf.sample(4)
        assert states.shape == (4, 3)
        assert next_states.shape == (4, 3)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

    def test_wraparound_overwrites_oldest(self):
        buf = ReplayBuffer(capacity=2, state_dim=1, seed=0)
        buf.push([0], 0, 0.0, [0], False)
        buf.push([1], 0, 0.0, [1], False)
        buf.push([2], 0, 0.0, [2], False)  # overwrites slot 0
        states, *_ = buf.sample(2)
        assert 0.0 not in states.flatten()


# ------------------------------------------------------------------
# QNetwork
# ------------------------------------------------------------------


class TestQNetwork:
    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError):
            QNetwork(0, [8], 5)

    def test_rejects_non_positive_output_dim(self):
        with pytest.raises(ValueError):
            QNetwork(4, [8], 0)

    def test_rejects_empty_hidden_layers(self):
        with pytest.raises(ValueError):
            QNetwork(4, [], 5)

    def test_rejects_non_positive_hidden_layer(self):
        with pytest.raises(ValueError):
            QNetwork(4, [8, 0], 5)

    def test_forward_output_shape(self):
        net = QNetwork(4, [8, 8], 5)
        x = torch.zeros((3, 4))
        out = net(x)
        assert out.shape == (3, 5)

    def test_variable_depth_hidden_layers(self):
        net = QNetwork(2, [16, 8, 4], 3)
        out = net(torch.zeros((1, 2)))
        assert out.shape == (1, 3)


# ------------------------------------------------------------------
# CurveRecorder
# ------------------------------------------------------------------


class TestCurveRecorder:
    def test_column_order_and_header(self, tmp_path):
        path = tmp_path / "curves.csv"
        rec = CurveRecorder(str(path), ["pred_1", "prey_1"])
        rec.close()
        header = path.read_text().strip().splitlines()[0]
        assert header == "episode,epsilon,pred_1_reward,prey_1_reward,pred_1_loss,prey_1_loss"

    def test_record_rounds_and_writes_row(self, tmp_path):
        path = tmp_path / "curves.csv"
        rec = CurveRecorder(str(path), ["pred_1"])
        rec.record(1, 0.123456, {"pred_1": 2.98765}, {"pred_1": [1.0, 2.0]})
        rec.close()
        row = path.read_text().strip().splitlines()[1]
        # epsilon->4dp, reward->4dp, mean loss (1.5)->6dp
        assert row == "1,0.1235,2.9876,1.5"

    def test_warmup_episode_writes_empty_loss_cell(self, tmp_path):
        path = tmp_path / "curves.csv"
        rec = CurveRecorder(str(path), ["pred_1"])
        rec.record(1, 0.5, {"pred_1": -3.0}, {"pred_1": []})
        rec.close()
        row = path.read_text().strip().splitlines()[1]
        assert row == "1,0.5,-3.0,"

    def test_creates_missing_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "sub" / "curves.csv"
        rec = CurveRecorder(str(path), ["pred_1"])
        rec.close()
        assert path.exists()


# ------------------------------------------------------------------
# DQN
# ------------------------------------------------------------------


class TestDQNInit:
    def test_infers_state_dim_and_action_dim(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        assert algo.state_dim == 2  # local_only encodes [x, y]
        assert algo.action_dim == 5  # discrete_5

    def test_one_learner_per_agent(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        assert set(algo.q_networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.target_networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.replay_buffers.keys()) == {"pred_1", "prey_1"}

    def test_target_starts_synced_with_policy(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        for aid in algo.agent_ids:
            policy_sd = algo.q_networks[aid].state_dict()
            target_sd = algo.target_networks[aid].state_dict()
            for key in policy_sd:
                assert torch.equal(policy_sd[key], target_sd[key])

    def test_missing_observation_encoder_raises(
        self, one_predator_one_prey, dqn_config
    ):
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        from baselines.DQN.dqn import DQN

        env = GridWorldEnv(
            agents=one_predator_one_prey,
            size=5,
            perc_num_obstacle=0,
            render_mode=None,
            seed=0,
        )
        env.action_space_plugin = DiscreteActionSpace()
        # no observation_builder/observation_encoder attached
        with pytest.raises(ValueError):
            DQN(env, dqn_config)

    def test_action_dim_mismatch_raises(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["action_dim"] = 9
        with pytest.raises(ValueError):
            DQN(dqn_env, dqn_config)

    def test_action_dim_matching_config_is_accepted(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["action_dim"] = 5
        algo = DQN(dqn_env, dqn_config)
        assert algo.action_dim == 5

    def test_buffer_smaller_than_batch_warns(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["buffer_size"] = 2
        dqn_config["batch_size"] = 8
        with pytest.warns(UserWarning):
            DQN(dqn_env, dqn_config)


class TestDQNSelectActions:
    def test_epsilon_one_is_fully_random(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["epsilon"] = 1.0
        algo = DQN(dqn_env, dqn_config)
        obs, _ = dqn_env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}
        for a in actions.values():
            assert 0 <= a < algo.action_dim

    def test_epsilon_zero_is_deterministic(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["epsilon"] = 0.0
        algo = DQN(dqn_env, dqn_config)
        obs, _ = dqn_env.reset()
        first = algo.select_actions(obs)
        second = algo.select_actions(obs)
        assert first == second


class TestDQNTrain:
    def test_trains_without_error(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        algo.train()
        assert all(len(buf) > 0 for buf in algo.replay_buffers.values())

    def test_epsilon_decays(self, dqn_env, dqn_config):
        from baselines.DQN.dqn import DQN

        dqn_config["epsilon"] = 1.0
        dqn_config["epsilon_decay"] = 0.9
        dqn_config["min_epsilon"] = 0.0
        algo = DQN(dqn_env, dqn_config)
        algo.train()
        assert algo.epsilon < 1.0

    def test_writes_curve_csv_when_configured(self, dqn_env, dqn_config, tmp_path):
        from baselines.DQN.dqn import DQN

        csv_path = tmp_path / "curves.csv"
        dqn_config["curves_path"] = str(csv_path)
        algo = DQN(dqn_env, dqn_config)
        algo.train()
        assert csv_path.exists()
        rows = csv_path.read_text().strip().splitlines()
        assert len(rows) == dqn_config["episodes"] + 1  # header + one row per episode


class TestDQNPersistence:
    def test_save_creates_file(self, dqn_env, dqn_config, tmp_path):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        path = tmp_path / "dqn.pkl"
        algo.save(str(path))
        assert path.exists()

    def test_load_restores_weights(self, dqn_env, dqn_config, tmp_path):
        from baselines.DQN.dqn import DQN

        algo = DQN(dqn_env, dqn_config)
        algo.train()
        path = tmp_path / "dqn.pkl"
        algo.save(str(path))

        loaded = DQN.load(dqn_env, dqn_config, str(path))
        for aid in algo.agent_ids:
            orig_sd = algo.q_networks[aid].state_dict()
            loaded_sd = loaded.q_networks[aid].state_dict()
            for key in orig_sd:
                assert torch.equal(orig_sd[key], loaded_sd[key])

    def test_loaded_model_matches_original_greedy_actions(
        self, dqn_env, dqn_config, tmp_path
    ):
        from baselines.DQN.dqn import DQN

        dqn_config["epsilon"] = 0.0
        algo = DQN(dqn_env, dqn_config)
        algo.train()
        path = tmp_path / "dqn.pkl"
        algo.save(str(path))

        loaded_config = dict(dqn_config)
        loaded_config["epsilon"] = 0.0
        loaded = DQN.load(dqn_env, loaded_config, str(path))

        obs, _ = dqn_env.reset()
        assert algo.select_actions(obs) == loaded.select_actions(obs)
