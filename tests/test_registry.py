"""
Tests for observation registry, reward registry, and action registry.
"""

import numpy as np
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
        builder = get_observation_builder("local_radius", radius=5, include_agents=False)
        assert builder.params.get("radius") == 5
        assert builder.params.get("include_agents") is False

    def test_register_custom_observation(self):
        class DummyObs(ObservationBuilder):
            def build(self, env):
                return {}

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
    def test_all_three_algorithms_registered(self):
        import baselines  # trigger registrations
        from baselines.registry.algorithm_registry import list_algorithms, get

        algos = list_algorithms()
        assert "iql" in algos
        assert "cql" in algos
        assert "dqn" in algos
        assert "mixed" in algos

    def test_get_returns_class(self):
        import baselines
        from baselines.registry.algorithm_registry import get
        from baselines.IQL.iql import IQL
        from baselines.CQL.cql import CQL
        from baselines.DQN.dqn import DQN
        from baselines.MIXED.mix_train import MixedTrainer

        assert get("iql") is IQL
        assert get("cql") is CQL
        assert get("dqn") is DQN
        assert get("mixed") is MixedTrainer

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
# SpeedDiscreteActionSpace unit tests
# ------------------------------------------------------------------

class TestSpeedDiscreteActionSpace:
    def _make(self):
        from multi_agent_package.actions.speed_discrete import SpeedDiscreteActionSpace
        return SpeedDiscreteActionSpace()

    def test_is_subclass_of_discrete_action_space(self):
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        sp = self._make()
        assert isinstance(sp, DiscreteActionSpace)

    def test_noop_returns_empty_list(self):
        sp = self._make()
        assert sp.to_moves(4, 3, 9999) == []

    def test_full_speed_when_stamina_sufficient(self):
        sp = self._make()
        moves = sp.to_moves(0, 3, 9999)
        assert len(moves) == 3

    def test_stamina_caps_moves_below_speed(self):
        sp = self._make()
        moves = sp.to_moves(0, 3, 2)
        assert len(moves) == 2

    def test_zero_stamina_returns_empty_list(self):
        sp = self._make()
        assert sp.to_moves(0, 3, 0) == []

    def test_direction_vectors_are_correct(self):
        sp = self._make()
        moves = sp.to_moves(0, 2, 9999)  # action 0 = RIGHT = [1, 0]
        assert all(np.array_equal(m, np.array([1, 0])) for m in moves)

    def test_speed_1_returns_one_move(self):
        sp = self._make()
        assert len(sp.to_moves(1, 1, 9999)) == 1

    def test_all_non_noop_actions_return_moves(self):
        sp = self._make()
        for action in range(4):  # 0-3 are movement actions
            assert len(sp.to_moves(action, 2, 9999)) == 2
