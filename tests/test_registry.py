"""
Tests for observation registry and reward registry.
"""

import pytest

from multi_agent_package.registry.observation_registry import (
    get_observation_builder,
    register_observation,
    _OBSERVATION_REGISTRY,
)
from multi_agent_package.registry.reward_registry import (
    get_reward_function,
    register_reward,
    _REWARD_REGISTRY,
)
from multi_agent_package.observations.base import ObservationBuilder
from multi_agent_package.rewards.base import RewardFunction

# ------------------------------------------------------------------
# Observation registry
# ------------------------------------------------------------------


class TestObservationRegistry:
    def test_default_registered(self):
        builder = get_observation_builder("default")
        assert builder is not None

    def test_local_only_registered(self):
        builder = get_observation_builder("local_only")
        assert builder is not None

    def test_local_radius_registered(self):
        builder = get_observation_builder("local_radius", radius=3)
        assert builder is not None

    def test_absolute_registered(self):
        builder = get_observation_builder("absolute")
        assert builder is not None

    def test_relative_registered(self):
        builder = get_observation_builder("relative")
        assert builder is not None

    def test_all_five_keys_present(self):
        expected = {"default", "local_only", "local_radius", "absolute", "relative"}
        assert expected.issubset(set(_OBSERVATION_REGISTRY.keys()))

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError):
            get_observation_builder("nonexistent_builder")

    def test_returned_instance_is_observation_builder(self):
        builder = get_observation_builder("default")
        assert isinstance(builder, ObservationBuilder)

    def test_params_forwarded_to_builder(self):
        builder = get_observation_builder(
            "local_radius", radius=5, include_agents=False
        )
        assert builder.params.get("radius") == 5
        assert builder.params.get("include_agents") is False

    def test_register_custom_observation(self):
        class DummyObs(ObservationBuilder):
            def build(self, env):
                return {}

            def encode(self, observation, env):
                return self._vector([])

        register_observation("_test_obs_custom", DummyObs)
        builder = get_observation_builder("_test_obs_custom")
        assert isinstance(builder, DummyObs)
        # cleanup
        del _OBSERVATION_REGISTRY["_test_obs_custom"]

    def test_register_non_subclass_raises_type_error(self):
        with pytest.raises(TypeError):
            register_observation("_bad", object)


# ------------------------------------------------------------------
# Reward registry
# ------------------------------------------------------------------


class TestRewardRegistry:
    def test_base_registered(self):
        fn = get_reward_function("base")
        assert fn is not None

    def test_predator_distance_registered(self):
        fn = get_reward_function("predator_distance")
        assert fn is not None

    def test_survival_registered(self):
        fn = get_reward_function("survival")
        assert fn is not None

    def test_all_three_keys_present(self):
        expected = {"base", "predator_distance", "survival"}
        assert expected.issubset(set(_REWARD_REGISTRY.keys()))

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError):
            get_reward_function("nonexistent_reward")

    def test_returned_instance_is_reward_function(self):
        fn = get_reward_function("base")
        assert isinstance(fn, RewardFunction)

    def test_weight_applied(self):
        fn = get_reward_function("base", weight=3.5)
        assert fn.weight == pytest.approx(3.5)

    def test_default_weight_is_one(self):
        fn = get_reward_function("base")
        assert fn.weight == pytest.approx(1.0)

    def test_register_custom_reward(self):
        class DummyReward(RewardFunction):
            def compute(self, env):
                return {}

        register_reward("_test_reward_custom", DummyReward)
        fn = get_reward_function("_test_reward_custom")
        assert isinstance(fn, DummyReward)
        # cleanup
        del _REWARD_REGISTRY["_test_reward_custom"]

    def test_register_non_subclass_raises_type_error(self):
        with pytest.raises(TypeError):
            register_reward("_bad", object)


# ------------------------------------------------------------------
# Algorithm registry (separate module, tested via baselines import)
# ------------------------------------------------------------------


class TestAlgorithmRegistry:
    def test_all_four_algorithms_registered(self):
        import baselines  # noqa: F401 (trigger registrations)
        from baselines.registry.algorithm_registry import list_algorithms

        algos = list_algorithms()
        assert "iql" in algos
        assert "cql" in algos
        assert "mixed" in algos
        assert "dqn" in algos

    def test_get_returns_class(self):
        import baselines  # noqa: F401 (trigger registrations)
        from baselines.registry.algorithm_registry import get
        from baselines.IQL.iql import IQL
        from baselines.CQL.cql import CQL
        from baselines.MIXED.mix_train import MixedTrainer
        from baselines.DQN.dqn import DQN

        assert get("iql") is IQL
        assert get("cql") is CQL
        assert get("mixed") is MixedTrainer
        assert get("dqn") is DQN

    def test_unknown_algorithm_raises_value_error(self):
        from baselines.registry.algorithm_registry import get

        with pytest.raises(ValueError):
            get("nonexistent_algo")


# ------------------------------------------------------------------
# Action registry
# ------------------------------------------------------------------


class TestActionRegistry:
    def test_discrete_5_registered(self):
        from multi_agent_package.registry.action_registry import get_action_space

        sp = get_action_space("discrete_5")
        assert sp is not None

    def test_cross_registered(self):
        from multi_agent_package.registry.action_registry import get_action_space

        sp = get_action_space("cross")
        assert sp.n_actions == 5

    def test_speed_discrete_5_registered(self):
        from multi_agent_package.registry.action_registry import get_action_space

        sp = get_action_space("speed_discrete_5")
        assert sp is not None

    def test_speed_discrete_5_has_to_moves(self):
        from multi_agent_package.registry.action_registry import get_action_space

        sp = get_action_space("speed_discrete_5")
        assert hasattr(sp, "to_moves") and callable(sp.to_moves)

    def test_unknown_action_space_raises_key_error(self):
        from multi_agent_package.registry.action_registry import get_action_space

        with pytest.raises(KeyError):
            get_action_space("nonexistent_action")


# ------------------------------------------------------------------
# ActionSpace.is_noop() — base-class default, all concrete spaces
# ------------------------------------------------------------------

class TestActionSpaceIsNoop:
    def test_discrete_noop_action(self):
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        sp = DiscreteActionSpace()
        assert sp.is_noop(4) is True

    def test_discrete_movement_action(self):
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        sp = DiscreteActionSpace()
        for a in range(4):
            assert sp.is_noop(a) is False

    def test_cross_noop_action(self):
        from multi_agent_package.actions.cross_actions import CrossActionSpace
        sp = CrossActionSpace()
        assert sp.is_noop(4) is True

    def test_cross_movement_action(self):
        from multi_agent_package.actions.cross_actions import CrossActionSpace
        sp = CrossActionSpace()
        for a in range(4):
            assert sp.is_noop(a) is False

    def test_speed_discrete_noop_action(self):
        from multi_agent_package.actions.speed_discrete import SpeedDiscreteActionSpace
        sp = SpeedDiscreteActionSpace()
        assert sp.is_noop(4) is True

    def test_speed_discrete_movement_action(self):
        from multi_agent_package.actions.speed_discrete import SpeedDiscreteActionSpace
        sp = SpeedDiscreteActionSpace()
        for a in range(4):
            assert sp.is_noop(a) is False
