"""
Tests for multi_agent_package.core.agent.Agent
"""

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent


class TestAgentIdentity:
    def test_attributes_stored(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1")
        assert ag.agent_name == "pred_1"
        assert ag.agent_type == "predator"
        assert ag.agent_team == "predator_1"

    def test_predator_speed(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        assert ag.agent_speed == 1

    def test_prey_speed(self):
        ag = Agent(agent_type="prey", agent_team="prey_1", agent_name="q")
        assert ag.agent_speed == 3

    def test_unknown_type_defaults_speed(self):
        ag = Agent(agent_type="other", agent_team="team_1", agent_name="x")
        assert ag.agent_speed == 1

    def test_action_space_size(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        assert ag.action_space.n == 5

    def test_initial_location_zero(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        np.testing.assert_array_equal(ag._agent_location, [0, 0])

    def test_initial_stamina(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        assert ag.stamina == 10


class TestActionDirections:
    def setup_method(self):
        self.ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        self.dirs = self.ag._actions_to_directions

    def test_right_action(self):
        np.testing.assert_array_equal(self.dirs[0], [1, 0])

    def test_up_action(self):
        np.testing.assert_array_equal(self.dirs[1], [0, 1])

    def test_left_action(self):
        np.testing.assert_array_equal(self.dirs[2], [-1, 0])

    def test_down_action(self):
        np.testing.assert_array_equal(self.dirs[3], [0, -1])

    def test_noop_action(self):
        np.testing.assert_array_equal(self.dirs[4], [0, 0])

    def test_all_five_actions_present(self):
        assert set(self.dirs.keys()) == {0, 1, 2, 3, 4}


class TestGetObs:
    def test_obs_keys(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        obs = ag._get_obs()
        assert "local" in obs
        assert "global" in obs

    def test_local_is_copy(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="p")
        obs = ag._get_obs()
        obs["local"][0] = 99
        # mutation of obs should not affect agent's internal state
        np.testing.assert_array_equal(ag._agent_location, [0, 0])

    def test_global_forwarded(self):
        ag = Agent(agent_type="prey", agent_team="prey_1", agent_name="q")
        payload = {"key": "val"}
        obs = ag._get_obs(global_obs=payload)
        assert obs["global"] == payload

    def test_global_none_by_default(self):
        ag = Agent(agent_type="prey", agent_team="prey_1", agent_name="q")
        obs = ag._get_obs()
        assert obs["global"] is None


class TestGetInfo:
    def test_info_keys(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1")
        info = ag._get_info()
        assert set(info.keys()) == {"name", "type", "team", "speed", "stamina"}

    def test_info_values(self):
        ag = Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1")
        info = ag._get_info()
        assert info["name"] == "pred_1"
        assert info["type"] == "predator"
        assert info["speed"] == 1
        assert info["stamina"] == 10
