"""
Tests for the Actor-Critic baseline: ActorCriticNetwork and the
one-step online ActorCritic algorithm (Sutton & Barto, Algorithm 13.5).
"""

import pytest
import torch

from baselines.AC.network import ActorCriticNetwork

# ------------------------------------------------------------------
# ActorCriticNetwork
# ------------------------------------------------------------------


class TestActorCriticNetwork:
    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError):
            ActorCriticNetwork(0, [8], 5)

    def test_rejects_non_positive_action_dim(self):
        with pytest.raises(ValueError):
            ActorCriticNetwork(4, [8], 0)

    def test_rejects_empty_hidden_layers(self):
        with pytest.raises(ValueError):
            ActorCriticNetwork(4, [], 5)

    def test_rejects_non_positive_hidden_layer(self):
        with pytest.raises(ValueError):
            ActorCriticNetwork(4, [8, 0], 5)

    def test_forward_output_shapes(self):
        net = ActorCriticNetwork(4, [8, 8], 5)
        logits, value = net(torch.zeros((3, 4)))
        assert logits.shape == (3, 5)
        assert value.shape == (3,)

    def test_variable_depth_hidden_layers(self):
        net = ActorCriticNetwork(2, [16, 8, 4], 3)
        logits, value = net(torch.zeros((1, 2)))
        assert logits.shape == (1, 3)
        assert value.shape == (1,)


# ------------------------------------------------------------------
# ActorCritic
# ------------------------------------------------------------------


class TestActorCriticInit:
    def test_infers_state_dim_and_action_dim(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        assert algo.state_dim == 2  # local_only encodes [x, y]
        assert algo.action_dim == 5  # discrete_5

    def test_one_learner_per_agent(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        assert set(algo.networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.optimizers.keys()) == {"pred_1", "prey_1"}

    def test_missing_observation_encoder_raises(
        self, one_predator_one_prey, ac_config
    ):
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        from baselines.AC.actor_critic import ActorCritic

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
            ActorCritic(env, ac_config)

    def test_action_dim_mismatch_raises(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        ac_config["action_dim"] = 9
        with pytest.raises(ValueError):
            ActorCritic(dqn_env, ac_config)

    def test_action_dim_matching_config_is_accepted(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        ac_config["action_dim"] = 5
        algo = ActorCritic(dqn_env, ac_config)
        assert algo.action_dim == 5


class TestActorCriticSelectActions:
    def test_returns_valid_actions_for_all_agents(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        obs, _ = dqn_env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}
        for a in actions.values():
            assert 0 <= a < algo.action_dim

    def test_seeded_construction_is_reproducible(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        ac_config["seed"] = 7
        algo_a = ActorCritic(dqn_env, ac_config)
        algo_b = ActorCritic(dqn_env, ac_config)
        obs, _ = dqn_env.reset()

        torch.manual_seed(123)
        first = algo_a.select_actions(obs)
        torch.manual_seed(123)
        second = algo_b.select_actions(obs)
        assert first == second


class TestActorCriticUpdate:
    def test_update_returns_float_loss(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        agent_id = algo.agent_ids[0]
        obs, _ = dqn_env.reset()
        state = algo._encode_observation(obs[agent_id])

        loss = algo._update(agent_id, state, 0, 1.0, state, terminal=False, discount=1.0)
        assert isinstance(loss, float)

    def test_update_changes_network_weights(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        agent_id = algo.agent_ids[0]
        obs, _ = dqn_env.reset()
        state = algo._encode_observation(obs[agent_id])

        before = {
            k: v.clone() for k, v in algo.networks[agent_id].state_dict().items()
        }
        algo._update(agent_id, state, 0, 1.0, state, terminal=False, discount=1.0)
        after = algo.networks[agent_id].state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_terminal_update_does_not_raise(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        agent_id = algo.agent_ids[0]
        obs, _ = dqn_env.reset()
        state = algo._encode_observation(obs[agent_id])

        loss = algo._update(agent_id, state, 0, 1.0, state, terminal=True, discount=1.0)
        assert isinstance(loss, float)


class TestActorCriticTrain:
    def test_trains_without_error_and_updates_weights(self, dqn_env, ac_config):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        agent_id = algo.agent_ids[0]
        before = {
            k: v.clone() for k, v in algo.networks[agent_id].state_dict().items()
        }

        algo.train()

        after = algo.networks[agent_id].state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)

    def test_writes_curve_csv_when_configured(self, dqn_env, ac_config, tmp_path):
        from baselines.AC.actor_critic import ActorCritic

        csv_path = tmp_path / "curves.csv"
        ac_config["curves_path"] = str(csv_path)
        algo = ActorCritic(dqn_env, ac_config)
        algo.train()

        assert csv_path.exists()
        rows = csv_path.read_text().strip().splitlines()
        assert len(rows) == ac_config["episodes"] + 1  # header + one row per episode

        expected_header = (
            "episode,"
            + ",".join(f"{aid}_reward" for aid in algo.agent_ids)
            + ","
            + ",".join(f"{aid}_loss" for aid in algo.agent_ids)
        )
        assert rows[0] == expected_header


class TestActorCriticPersistence:
    def test_save_creates_file(self, dqn_env, ac_config, tmp_path):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        path = tmp_path / "ac.pkl"
        algo.save(str(path))
        assert path.exists()

    def test_load_restores_weights(self, dqn_env, ac_config, tmp_path):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        algo.train()
        path = tmp_path / "ac.pkl"
        algo.save(str(path))

        loaded = ActorCritic.load(dqn_env, ac_config, str(path))
        for aid in algo.agent_ids:
            orig_sd = algo.networks[aid].state_dict()
            loaded_sd = loaded.networks[aid].state_dict()
            for key in orig_sd:
                assert torch.equal(orig_sd[key], loaded_sd[key])

    def test_loaded_model_matches_original_sampled_actions(
        self, dqn_env, ac_config, tmp_path
    ):
        from baselines.AC.actor_critic import ActorCritic

        algo = ActorCritic(dqn_env, ac_config)
        algo.train()
        path = tmp_path / "ac.pkl"
        algo.save(str(path))

        loaded = ActorCritic.load(dqn_env, ac_config, str(path))

        obs, _ = dqn_env.reset()
        # Reseed identically right before each call so the comparison isolates
        # weight-equality -- sampling itself is stochastic, so without this
        # both instances would draw from different points in the RNG stream.
        torch.manual_seed(999)
        first = algo.select_actions(obs)
        torch.manual_seed(999)
        second = loaded.select_actions(obs)
        assert first == second
