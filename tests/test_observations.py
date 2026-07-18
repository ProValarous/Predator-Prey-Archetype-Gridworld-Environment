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
    env = GridWorldEnv(
        agents=agents,
        size=size,
        perc_num_obstacle=perc_obstacle,
        render_mode=None,
        seed=seed,
    )
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

    def test_encode_slot_is_identity_stable(self):
        # A given prey must occupy the same fixed slot regardless of which prey
        # is currently within radius (issue #34). Two different prey at the same
        # position must therefore produce DIFFERENT encodings, not identical
        # ones (which was the old present-first-packing bug).
        builder = LocalRadiusObservation(radius=2, include_obstacles=False)
        agents = [
            Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
            Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
            Agent(agent_type="prey", agent_team="prey_2", agent_name="prey_2"),
        ]
        env = GridWorldEnv(
            agents=agents, size=9, perc_num_obstacle=0, render_mode=None, seed=0
        )
        env.reset()
        pred, p1, p2 = env.agents
        # Case A: only prey_1 within radius, at (1,0)
        pred._agent_location = np.array([0, 0], dtype=np.int32)
        p1._agent_location = np.array([1, 0], dtype=np.int32)
        p2._agent_location = np.array([8, 8], dtype=np.int32)
        enc_a = builder.encode(builder.build(env)["pred_1"], env)
        # Case B: only prey_2 within radius, at the same (1,0)
        p1._agent_location = np.array([8, 8], dtype=np.int32)
        p2._agent_location = np.array([1, 0], dtype=np.int32)
        enc_b = builder.encode(builder.build(env)["pred_1"], env)
        assert not np.array_equal(enc_a, enc_b)


class TestLocalRadiusObservationSlotStability:
    """
    Regression test for the identity-slot bug: an agent's feature slot in
    the encoded vector must stay at the same index regardless of which
    other agents are currently visible. Visibility should only zero/non-zero
    a slot, never shift another agent into it.
    """

    def _make_three_agent_env(self, size=20, seed=1):
        agents = [
            Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
            Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
            Agent(agent_type="prey", agent_team="prey_2", agent_name="prey_2"),
        ]
        env = GridWorldEnv(
            agents=agents,
            size=size,
            perc_num_obstacle=0,
            render_mode=None,
            seed=seed,
        )
        env.reset()
        return env

    def _slot_index(self, env, observer_idx, target_name):
        """Index in the encoded vector where target_name's block starts."""
        agent_names_sorted = sorted(a.agent_name for a in env.agents)
        pos = agent_names_sorted.index(target_name)
        # 2 floats for local pos + 1 for radius, then 5 floats per agent slot
        return 3 + pos * 5

    def test_slot_index_stable_as_visibility_changes(self):
        builder = LocalRadiusObservation(radius=3)
        env = self._make_three_agent_env()

        observer = env.agents[0]  # pred_1
        other_far = env.agents[1]  # prey_1
        other_near = env.agents[2]  # prey_2

        observer._agent_location = np.array([0, 0], dtype=np.int32)
        other_near._agent_location = np.array([1, 0], dtype=np.int32)  # visible
        other_far._agent_location = np.array([15, 15], dtype=np.int32)  # not visible

        obs = builder.build(env)
        encoded_step1 = builder.encode(obs[observer.agent_name], env)

        near_slot = self._slot_index(env, 0, other_near.agent_name)
        far_slot = self._slot_index(env, 0, other_far.agent_name)

        # prey_2 (near) should be present (presence flag == 1.0)
        assert encoded_step1[near_slot] == 1.0
        # prey_1 (far) should be absent (presence flag == 0.0)
        assert encoded_step1[far_slot] == 0.0

        # Now move prey_1 into radius and prey_2 out of radius.
        other_near._agent_location = np.array([15, 15], dtype=np.int32)  # now far
        other_far._agent_location = np.array([1, 0], dtype=np.int32)  # now near

        obs = builder.build(env)
        encoded_step2 = builder.encode(obs[observer.agent_name], env)

        # Slot indices must be identical to step 1 -- only the flag flips.
        assert encoded_step2[near_slot] == 0.0  # prey_2 now absent, same slot
        assert encoded_step2[far_slot] == 1.0  # prey_1 now present, same slot

        # Vector length must not change between steps either.
        assert encoded_step1.shape == encoded_step2.shape

    def test_encoded_length_includes_self_slot(self):
        """state_dim now includes one always-zero self slot per agent."""
        builder = LocalRadiusObservation(radius=3)
        env = self._make_three_agent_env()
        obs = builder.build(env)
        encoded = builder.encode(obs[env.agents[0].agent_name], env)

        n_agents = len(env.agents)
        n_obstacles = len(env._obstacle_location)
        expected_len = 3 + n_agents * 5 + n_obstacles * 4
        assert encoded.shape[0] == expected_len

        # The observer's own slot must always be all zeros.
        self_slot = self._slot_index(env, 0, env.agents[0].agent_name)
        assert list(encoded[self_slot:self_slot + 5]) == [0.0, 0.0, 0.0, 0.0, 0.0]


# ------------------------------------------------------------------
# encode() contract: every observation builder must produce a fixed-
# length, finite, numeric vector -- this is what DQN relies on.
# ------------------------------------------------------------------


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
    def test_encode_returns_numeric_vectors(
        self, builder_cls, builder_kwargs, env_size
    ):
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
