import numpy as np
import pytest

from multi_agent_package.actions.cross_actions import CrossActionSpace
from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
from multi_agent_package.actions.speed_discrete import SpeedDiscreteActionSpace
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv


class TestCrossActionSpace:
    def setup_method(self):
        self.space = CrossActionSpace()

    def test_n_actions(self):
        assert self.space.n_actions == 5

    def test_gymnasium_space_matches_n_actions(self):
        assert self.space.gymnasium_space.n == self.space.n_actions

    @pytest.mark.parametrize(
        "action,expected",
        [
            (0, [1, 1]),
            (1, [-1, 1]),
            (2, [-1, -1]),
            (3, [1, -1]),
            (4, [0, 0]),
        ],
    )
    def test_direction_vectors(self, action, expected):
        np.testing.assert_array_equal(self.space.to_direction(action), expected)

    def test_no_cardinal_moves_present(self):
        """Every non-noop direction must have both components non-zero."""
        for action in range(4):
            d = self.space.to_direction(action)
            assert d[0] != 0 and d[1] != 0

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            self.space.to_direction(99)

    def test_dtype_is_int32(self):
        for a in range(5):
            assert self.space.to_direction(a).dtype == np.int32


class TestCrossActionWiredToEnv:
    def make_env(self, seed=0):
        agents = [
            Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
            Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
        ]
        env = GridWorldEnv(
            agents=agents, size=8, perc_num_obstacle=0, render_mode=None, seed=seed
        )
        env.action_space_plugin = CrossActionSpace()
        return env

    def test_agent_moves_diagonally(self):
        env = self.make_env()
        env.reset()
        pred = env.agents[0]
        pred._agent_location = np.array([3, 3], dtype=np.int32)
        env.step({"pred_1": 0, "prey_1": 4})  # pred: NE, prey: noop
        np.testing.assert_array_equal(pred._agent_location, [4, 4])

    def test_diagonal_clips_at_boundary(self):
        env = self.make_env()
        env.reset()
        pred = env.agents[0]
        pred._agent_location = np.array([7, 7], dtype=np.int32)  # corner, size=8
        env.step({"pred_1": 0, "prey_1": 4})  # NE would go to [8,8] -> clipped
        np.testing.assert_array_equal(pred._agent_location, [7, 7])

    def test_step_returns_valid_structure_with_cross_actions(self):
        env = self.make_env()
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert set(out.keys()) == {"obs", "reward", "terminated", "truncated", "info"}


class TestSpeedDiscreteActionSpace:
    def _make(self):
        return SpeedDiscreteActionSpace()

    def test_is_subclass_of_discrete_action_space(self):
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
