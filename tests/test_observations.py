"""
Tests for all five observation builders.
"""

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.observations.default import DefaultObservation
from multi_agent_package.observations.local_only import LocalOnlyObservation
from multi_agent_package.observations.absolute import AbsoluteObservation
from multi_agent_package.observations.relative import RelativeObservation
from multi_agent_package.observations.local_radius import LocalRadiusObservation


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(size=6, perc_obstacle=0, seed=1):
    agents = [
        Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
        Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
    ]
    env = GridWorldEnv(agents=agents, size=size, perc_num_obstacle=perc_obstacle, render_mode=None, seed=seed)
    env.reset()
    return env


def assert_no_env_mutation(env, builder):
    """Builder must not change any agent location."""
    locations_before = [ag._agent_location.copy() for ag in env.agents]
    builder.build(env)
    for ag, loc_before in zip(env.agents, locations_before):
        np.testing.assert_array_equal(ag._agent_location, loc_before)


# ------------------------------------------------------------------
# DefaultObservation
# ------------------------------------------------------------------

class TestDefaultObservation:
    def setup_method(self):
        self.env = make_env()
        self.builder = DefaultObservation()

    def test_returns_all_agents(self):
        obs = self.builder.build(self.env)
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_each_obs_has_local_and_global(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert "local" in o
            assert "global" in o

    def test_global_has_dist_agents(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert "dist_agents" in o["global"]

    def test_no_self_in_dist_agents(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert aid not in o["global"]["dist_agents"]

    def test_distances_non_negative(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            for d in o["global"]["dist_agents"].values():
                assert d >= 0

    def test_does_not_mutate_env(self):
        assert_no_env_mutation(self.env, self.builder)


# ------------------------------------------------------------------
# LocalOnlyObservation
# ------------------------------------------------------------------

class TestLocalOnlyObservation:
    def setup_method(self):
        self.env = make_env()
        self.builder = LocalOnlyObservation()

    def test_returns_all_agents(self):
        obs = self.builder.build(self.env)
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_obs_has_local_key(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert "local" in o

    def test_local_is_ndarray_len2(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert isinstance(o["local"], np.ndarray)
            assert o["local"].shape == (2,)

    def test_does_not_mutate_env(self):
        assert_no_env_mutation(self.env, self.builder)


# ------------------------------------------------------------------
# AbsoluteObservation
# ------------------------------------------------------------------

class TestAbsoluteObservation:
    def setup_method(self):
        self.env = make_env()
        self.builder = AbsoluteObservation()

    def test_returns_all_agents(self):
        obs = self.builder.build(self.env)
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_obs_has_position_key(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            # Should include some form of absolute position
            assert "position" in o or "local" in o or "abs_pos" in o

    def test_does_not_mutate_env(self):
        assert_no_env_mutation(self.env, self.builder)


# ------------------------------------------------------------------
# RelativeObservation
# ------------------------------------------------------------------

class TestRelativeObservation:
    def setup_method(self):
        self.env = make_env()
        self.builder = RelativeObservation()

    def test_returns_all_agents(self):
        obs = self.builder.build(self.env)
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_does_not_mutate_env(self):
        assert_no_env_mutation(self.env, self.builder)

    def test_relative_offsets_are_symmetric(self):
        """pred_1 sees prey_1 at +offset; prey_1 must see pred_1 at -offset."""
        obs = self.builder.build(self.env)
        pred_to_prey = obs["pred_1"]["agents"]["prey_1"]["rel_pos"]
        prey_to_pred = obs["prey_1"]["agents"]["pred_1"]["rel_pos"]
        np.testing.assert_array_equal(pred_to_prey, -prey_to_pred)


# ------------------------------------------------------------------
# LocalRadiusObservation
# ------------------------------------------------------------------

class TestLocalRadiusObservation:
    def setup_method(self):
        self.env = make_env(size=8)
        self.builder = LocalRadiusObservation(radius=3)

    def test_returns_all_agents(self):
        obs = self.builder.build(self.env)
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_each_obs_has_expected_keys(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert "local" in o
            assert "visible_agents" in o
            assert "visible_obstacles" in o
            assert "radius" in o

    def test_radius_stored_correctly(self):
        obs = self.builder.build(self.env)
        for o in obs.values():
            assert o["radius"] == 3

    def test_visible_agents_within_radius(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            ax, ay = o["local"]
            for other_name, info in o["visible_agents"].items():
                rx, ry = info["rel_pos"]
                manhattan = abs(rx) + abs(ry)
                assert manhattan <= 3

    def test_no_self_in_visible_agents(self):
        obs = self.builder.build(self.env)
        for aid, o in obs.items():
            assert aid not in o["visible_agents"]

    def test_does_not_mutate_env(self):
        assert_no_env_mutation(self.env, self.builder)

    def test_agents_outside_radius_not_visible(self):
        """Place agents at max distance; with radius=1 they should see nobody."""
        builder = LocalRadiusObservation(radius=1)
        env = make_env(size=8)
        env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
        env.agents[1]._agent_location = np.array([7, 7], dtype=np.int32)
        obs = builder.build(env)
        for aid, o in obs.items():
            assert len(o["visible_agents"]) == 0

    def test_encode_is_numeric_and_fixed_length(self):
        obs = self.builder.build(self.env)
        encoded_lengths = set()

        for o in obs.values():
            encoded = self.builder.encode(o, self.env)
            assert isinstance(encoded, np.ndarray)
            assert encoded.dtype == np.float32
            assert encoded.ndim == 1
            assert np.isfinite(encoded).all()
            encoded_lengths.add(encoded.shape[0])

        assert len(encoded_lengths) == 1


class TestObservationEncoding:
    @pytest.mark.parametrize(
        "builder_cls,builder_kwargs,env_size",
        [
            (DefaultObservation, {}, 6),
            (LocalOnlyObservation, {}, 6),
            (AbsoluteObservation, {}, 6),
            (RelativeObservation, {}, 6),
            (LocalRadiusObservation, {"radius": 3}, 8),
        ],
    )
    def test_encode_returns_numeric_vectors(self, builder_cls, builder_kwargs, env_size):
        env = make_env(size=env_size)
        builder = builder_cls(**builder_kwargs)
        obs = builder.build(env)

        lengths = set()
        for agent_obs in obs.values():
            encoded = builder.encode(agent_obs, env)
            assert isinstance(encoded, np.ndarray)
            assert encoded.dtype == np.float32
            assert encoded.ndim == 1
            assert np.isfinite(encoded).all()
            lengths.add(encoded.shape[0])

        assert len(lengths) == 1
