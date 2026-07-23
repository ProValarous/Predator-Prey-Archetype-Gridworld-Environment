# tests/test_baselines_a2c.py
"""
Tests for A2C (Advantage Actor-Critic baseline).
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.observations.relative import RelativeObservation
from baselines.A2C.a2c import A2C
from baselines.A2C.actor_network import ActorNetwork
from baselines.A2C.critic_network import CriticNetwork


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(n_pred=1, n_prey=1, size=5, seed=0, perc_obstacle=0, max_steps=50):
    """
    Builds a small env wired with the 'relative' observation builder and
    its encode() method attached as observation_encoder -- the exact
    contract A2C (and DQN) expect. max_steps is capped so a badly-behaved
    untrained policy can't make a test hang on an episode that never
    terminates.
    """
    agents = []
    for i in range(1, n_pred + 1):
        agents.append(Agent(agent_type="predator", agent_team=f"predator_{i}", agent_name=f"pred_{i}"))
    for i in range(1, n_prey + 1):
        agents.append(Agent(agent_type="prey", agent_team=f"prey_{i}", agent_name=f"prey_{i}"))

    env = GridWorldEnv(
        agents=agents, size=size, perc_num_obstacle=perc_obstacle,
        render_mode=None, seed=seed, max_steps=max_steps,
    )
    observation_builder = RelativeObservation(
        include_agents=True, include_obstacles=False, include_walls=False,
        distance_type="manhattan",
    )
    env.observation_builder = observation_builder.build
    env.observation_encoder = observation_builder.encode
    return env


def expected_state_dim(env):
    """Computes the state_dim independently of A2C, straight from the
    observation builder, so tests never hardcode a magic number that
    would silently go stale if the encoder's padding scheme changes."""
    obs, _ = env.reset()
    first_agent = next(iter(obs))
    encoded = env.observation_encoder(obs[first_agent], env)
    return np.asarray(encoded, dtype=np.float32).reshape(-1).shape[0]


def base_config(**overrides):
    cfg = {
        "hidden_layers": [8, 8],
        "gamma": 0.99,
        "n_steps": 4,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "actor_learning_rate": 0.01,
        "critic_learning_rate": 0.01,
        "episodes": 3,
        "seed": 0,
        "grad_clip": 5.0,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# ActorNetwork -- tested in isolation from the env
# ------------------------------------------------------------------

class TestActorNetworkValidation:
    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError):
            ActorNetwork(input_dim=0, hidden_layers=[8], output_dim=5)

    def test_rejects_non_positive_output_dim(self):
        with pytest.raises(ValueError):
            ActorNetwork(input_dim=2, hidden_layers=[8], output_dim=0)

    def test_rejects_empty_hidden_layers(self):
        with pytest.raises(ValueError):
            ActorNetwork(input_dim=2, hidden_layers=[], output_dim=5)

    def test_rejects_negative_hidden_layer_size(self):
        with pytest.raises(ValueError):
            ActorNetwork(input_dim=2, hidden_layers=[8, -4], output_dim=5)

    def test_forward_output_shape(self):
        net = ActorNetwork(input_dim=2, hidden_layers=[8, 8], output_dim=5)
        out = net(torch.zeros((1, 2)))
        assert out.shape == (1, 5)

    def test_distribution_probs_sum_to_one(self):
        net = ActorNetwork(input_dim=2, hidden_layers=[8], output_dim=5)
        dist = net.distribution(torch.tensor([[0.1, -0.2]]))
        total = dist.probs.sum(dim=1).item()
        assert abs(total - 1.0) < 1e-5

    def test_distribution_entropy_is_positive_at_init(self):
        """A freshly initialized network shouldn't already be a degenerate
        (zero-entropy) policy."""
        net = ActorNetwork(input_dim=2, hidden_layers=[8], output_dim=5)
        dist = net.distribution(torch.tensor([[0.0, 0.0]]))
        assert dist.entropy().item() > 0.0

    def test_sample_returns_valid_action_index(self):
        net = ActorNetwork(input_dim=2, hidden_layers=[8], output_dim=5)
        dist = net.distribution(torch.tensor([[0.3, -0.1]]))
        action = dist.sample()
        assert 0 <= int(action.item()) < 5


# ------------------------------------------------------------------
# CriticNetwork -- tested in isolation from the env
# ------------------------------------------------------------------

class TestCriticNetworkValidation:
    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError):
            CriticNetwork(input_dim=0, hidden_layers=[8])

    def test_rejects_empty_hidden_layers(self):
        with pytest.raises(ValueError):
            CriticNetwork(input_dim=2, hidden_layers=[])

    def test_forward_output_shape(self):
        net = CriticNetwork(input_dim=2, hidden_layers=[8, 8])
        out = net(torch.zeros((1, 2)))
        assert out.shape == (1, 1)

    def test_can_output_negative_values(self):
        """Linear output layer -- V(s) must be free to go negative, this
        env hands out large negative penalties."""
        net = CriticNetwork(input_dim=1, hidden_layers=[4])
        with torch.no_grad():
            net.model[-1].bias.fill_(-5.0)
            net.model[-1].weight.fill_(0.0)
        out = net(torch.zeros((1, 1)))
        assert out.item() < 0


# ------------------------------------------------------------------
# A2C initialization
# ------------------------------------------------------------------

class TestA2CInit:
    def test_agent_ids_discovered(self):
        env = make_env()
        algo = A2C(env, base_config())
        assert set(algo.agent_ids) == {"pred_1", "prey_1"}

    def test_state_dim_matches_encoder_output(self):
        env = make_env()
        expected = expected_state_dim(env)
        algo = A2C(env, base_config())
        assert algo.state_dim == expected

    def test_action_dim_matches_env_when_unset_in_config(self):
        env = make_env()
        algo = A2C(env, base_config())
        assert algo.action_dim == 5  # discrete_5: right/up/left/down/noop

    def test_actors_and_critics_created_per_agent(self):
        env = make_env()
        algo = A2C(env, base_config())
        assert set(algo.actors.keys()) == {"pred_1", "prey_1"}
        assert set(algo.critics.keys()) == {"pred_1", "prey_1"}

    def test_optimizers_created_per_agent(self):
        env = make_env()
        algo = A2C(env, base_config())
        assert set(algo.actor_optimizers.keys()) == {"pred_1", "prey_1"}
        assert set(algo.critic_optimizers.keys()) == {"pred_1", "prey_1"}

    def test_invalid_action_dim_raises(self):
        env = make_env()
        with pytest.raises(ValueError):
            A2C(env, base_config(action_dim=4))  # env actually has 5 actions

    def test_invalid_n_steps_raises(self):
        env = make_env()
        with pytest.raises(ValueError):
            A2C(env, base_config(n_steps=0))

    def test_missing_observation_encoder_raises(self):
        env = make_env()
        env.observation_encoder = None
        with pytest.raises(ValueError):
            A2C(env, base_config())


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------

class TestA2CSelectActions:
    def test_returns_action_for_every_agent(self):
        env = make_env()
        algo = A2C(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = A2C(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        for a in actions.values():
            assert 0 <= a < algo.action_dim

    def test_greedy_eval_is_deterministic(self):
        """With greedy_eval=True, repeated calls on the same observation
        must return the same action every time (argmax, not sampling)."""
        env = make_env()
        algo = A2C(env, base_config(greedy_eval=True))
        obs, _ = env.reset()
        first = algo.select_actions(obs)
        for _ in range(5):
            again = algo.select_actions(obs)
            assert again == first

    def test_select_actions_does_not_build_gradient_graph(self):
        """select_actions is the eval-facing method -- it must not leave
        tensors requiring grad lying around (it's wrapped in no_grad)."""
        env = make_env()
        algo = A2C(env, base_config())
        obs, _ = env.reset()
        state = algo._encode_observation(obs["pred_1"])
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            value = algo.critics["pred_1"](state_t)
        assert not value.requires_grad


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

class TestA2CTrain:
    def test_actor_weights_change_after_training(self):
        env = make_env(max_steps=10)
        algo = A2C(env, base_config(episodes=5, n_steps=3))
        before = [p.clone() for p in algo.actors["pred_1"].parameters()]
        algo.train()
        after = list(algo.actors["pred_1"].parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed

    def test_critic_weights_change_after_training(self):
        env = make_env(max_steps=10)
        algo = A2C(env, base_config(episodes=5, n_steps=3))
        before = [p.clone() for p in algo.critics["pred_1"].parameters()]
        algo.train()
        after = list(algo.critics["pred_1"].parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed

    def test_train_steps_counter_increases(self):
        env = make_env(max_steps=10)
        algo = A2C(env, base_config(episodes=3, n_steps=3))
        assert algo._train_steps == 0
        algo.train()
        assert algo._train_steps > 0

    def test_partial_final_rollout_still_updates(self):
        """If an episode ends before n_steps transitions accumulate, the
        partial rollout must still trigger an update (bootstrapping from 0
        at the terminal state), not silently get dropped."""
        env = make_env(max_steps=2)  # forces very short episodes
        algo = A2C(env, base_config(episodes=2, n_steps=100))
        algo.train()
        assert algo._train_steps > 0

    def test_curves_csv_written(self):
        env = make_env(max_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            curves_path = os.path.join(tmp, "curves.csv")
            algo = A2C(env, base_config(episodes=2, n_steps=3, curves_path=curves_path))
            algo.train()
            assert os.path.isfile(curves_path)
            with open(curves_path) as fh:
                lines = fh.readlines()
            assert len(lines) == 3  # header + 2 episodes


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

class TestA2CPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = A2C(env, base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a2c.pkl")
            algo.save(path)
            assert os.path.isfile(path)

    def test_load_restores_actor_and_critic_weights(self):
        env = make_env()
        algo = A2C(env, base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a2c.pkl")
            algo.save(path)

            loaded = A2C.load(env, base_config(), path)
            for aid in algo.agent_ids:
                for p_before, p_after in zip(
                    algo.actors[aid].parameters(), loaded.actors[aid].parameters()
                ):
                    torch.testing.assert_close(p_before, p_after)
                for p_before, p_after in zip(
                    algo.critics[aid].parameters(), loaded.critics[aid].parameters()
                ):
                    torch.testing.assert_close(p_before, p_after)

    def test_loaded_model_can_evaluate(self):
        env = make_env(max_steps=10)
        algo = A2C(env, base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a2c.pkl")
            algo.save(path)
            loaded = A2C.load(env, base_config(greedy_eval=True), path)
            loaded.evaluate(episodes=2, max_steps=10)  # should not raise
