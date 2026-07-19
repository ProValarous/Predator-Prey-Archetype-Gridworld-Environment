"""
Tests for multi_agent_package.core.gridworld.GridWorldEnv
"""

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_env(size=5, n_pred=1, n_prey=1, perc_obstacle=0, seed=42, **kwargs):
    agents = []
    for i in range(1, n_pred + 1):
        agents.append(
            Agent(
                agent_type="predator",
                agent_team=f"predator_{i}",
                agent_name=f"pred_{i}",
            )
        )
    for i in range(1, n_prey + 1):
        agents.append(
            Agent(agent_type="prey", agent_team=f"prey_{i}", agent_name=f"prey_{i}")
        )
    return GridWorldEnv(
        agents=agents,
        size=size,
        perc_num_obstacle=perc_obstacle,
        render_mode=None,
        seed=seed,
        **kwargs,
    )


# ------------------------------------------------------------------
# Reset postconditions
# ------------------------------------------------------------------


class TestReset:
    def test_returns_obs_and_info(self):
        env = make_env()
        out = env.reset()
        assert isinstance(out, tuple) and len(out) == 2

    def test_obs_has_all_agent_keys(self):
        env = make_env(n_pred=1, n_prey=1)
        obs, _ = env.reset()
        assert set(obs.keys()) == {"pred_1", "prey_1"}

    def test_info_has_all_agent_keys(self):
        env = make_env(n_pred=1, n_prey=1)
        _, info = env.reset()
        assert set(info.keys()) == {"pred_1", "prey_1"}

    def test_agent_locations_in_bounds(self):
        env = make_env(size=5)
        env.reset()
        for ag in env.agents:
            assert 0 <= ag._agent_location[0] < 5
            assert 0 <= ag._agent_location[1] < 5

    def test_no_duplicate_start_positions(self):
        env = make_env(n_pred=2, n_prey=2, size=6)
        env.reset()
        positions = [tuple(ag._agent_location) for ag in env.agents]
        assert len(positions) == len(set(positions))

    def test_episode_step_counter_reset(self):
        env = make_env()
        env.reset()
        assert env._episode_steps == 0

    def test_captures_reset(self):
        env = make_env()
        env.reset()
        assert env._captures_total == 0

    def test_seed_determinism(self):
        env_a = make_env(seed=7)
        env_b = make_env(seed=7)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        for aid in obs_a:
            np.testing.assert_array_equal(obs_a[aid]["local"], obs_b[aid]["local"])


# ------------------------------------------------------------------
# Step output structure
# ------------------------------------------------------------------


class TestStepStructure:
    def test_step_keys(self):
        env = make_env()
        obs, _ = env.reset()
        actions = {aid: 4 for aid in obs}  # noop
        out = env.step(actions)
        assert set(out.keys()) == {"obs", "reward", "terminated", "truncated", "info"}

    def test_obs_all_agents(self):
        env = make_env(n_pred=1, n_prey=1)
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert set(out["obs"].keys()) == {"pred_1", "prey_1"}

    def test_reward_all_agents(self):
        env = make_env(n_pred=1, n_prey=1)
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert set(out["reward"].keys()) == {"pred_1", "prey_1"}

    def test_terminated_is_bool(self):
        env = make_env()
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert isinstance(out["terminated"], bool)

    def test_truncated_is_bool(self):
        env = make_env()
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert isinstance(out["truncated"], bool)

    def test_episode_step_increments(self):
        env = make_env()
        obs, _ = env.reset()
        env.step({aid: 4 for aid in obs})
        assert env._episode_steps == 1


# ------------------------------------------------------------------
# Reward is supplied only through reward_fn (base reward is a plugin)
# ------------------------------------------------------------------


class TestStepCost:
    def test_step_applies_no_reward_without_reward_fn(self):
        # gridworld.step() supplies no reward on its own; all reward comes
        # through reward_fn. This is what makes the base reward (a plugin) have
        # a single application path and prevents double-counting (issue #32).
        env = make_env(perc_obstacle=0)
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert set(out["reward"]) == set(obs)
        assert all(v == 0.0 for v in out["reward"].values())

    def test_base_reward_plugin_applies_step_cost_once(self):
        # With the BaseReward plugin as the reward_fn, the predator's -5 step
        # cost is applied exactly once (not -10).
        from multi_agent_package.rewards.base_reward import BaseReward

        env = make_env(perc_obstacle=0)
        env.reward_fn = BaseReward(weight=1.0).compute
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        assert out["reward"]["pred_1"] == pytest.approx(-5.0)


# ------------------------------------------------------------------
# Obstacle mechanics
# ------------------------------------------------------------------


class TestObstacles:
    def test_obstacle_count(self):
        env = make_env(size=10, perc_obstacle=20)
        env.reset()
        expected = int(0.20 * 10 * 10)
        assert len(env._obstacle_location) == expected

    def test_obstacles_not_at_agent_positions(self):
        env = make_env(size=8, perc_obstacle=30, seed=1)
        env.reset()
        agent_positions = {tuple(ag._agent_location) for ag in env.agents}
        for obs in env._obstacle_location:
            assert tuple(obs) not in agent_positions

    def test_cell_sharing_disabled_blocks_same_role(self):
        # Two predators cannot stack when allow_cell_sharing=False.
        env = make_env(n_pred=2, n_prey=1, perc_obstacle=0, allow_cell_sharing=False)
        env.reset()
        p1, p2 = env.agents[0], env.agents[1]
        p1._agent_location = np.array([2, 2], dtype=np.int32)
        p2._agent_location = np.array([3, 2], dtype=np.int32)
        env.agents[2]._agent_location = np.array([0, 0], dtype=np.int32)
        # pred_2 moves LEFT (action 2) onto pred_1's cell -> blocked
        env.step({"pred_1": 4, "pred_2": 2, "prey_1": 4})
        assert tuple(p2._agent_location) != tuple(p1._agent_location)

    def test_cell_sharing_disabled_still_allows_capture(self):
        # Cross-role overlap (capture) is allowed even with sharing disabled.
        env = make_env(
            n_pred=1, n_prey=1, perc_obstacle=0, allow_cell_sharing=False,
            capture_threshold=1,
        )
        env.reset()
        pred, prey = env.agents[0], env.agents[1]
        pred._agent_location = np.array([2, 2], dtype=np.int32)
        prey._agent_location = np.array([1, 2], dtype=np.int32)
        out = env.step({"pred_1": 2, "prey_1": 4})  # predator LEFT onto prey
        assert "prey_1" in env._captured_this_step
        assert out["terminated"]

    def test_block_by_obstacle(self):
        """Agent cannot move onto obstacle when block_agents_by_obstacles=True."""
        import numpy as np

        env = make_env(perc_obstacle=0, seed=42, block_agents_by_obstacles=True)
        env.reset()
        agent = env.agents[0]
        # Place agent at (0,0) and obstacle at (1,0) — i.e., directly to the right
        agent._agent_location = np.array([0, 0], dtype=np.int32)
        env._obstacle_location = [np.array([1, 0], dtype=np.int32)]
        # Step: all agents noop except predator tries to go right (action 0)
        actions = {ag.agent_name: (0 if ag == agent else 4) for ag in env.agents}
        env.step(actions)
        # Agent should still be at (0,0), blocked by the obstacle
        assert tuple(agent._agent_location) == (0, 0)


# ------------------------------------------------------------------
# Capture mechanics
# ------------------------------------------------------------------


class TestCapture:
    def test_capture_on_same_cell(self):
        env = make_env(perc_obstacle=0, seed=0)
        env.reset()
        pred = next(a for a in env.agents if a.agent_type == "predator")
        prey = next(a for a in env.agents if a.agent_type == "prey")
        # Force both agents onto the same cell
        pred._agent_location = np.array([2, 2], dtype=np.int32)
        prey._agent_location = np.array([3, 2], dtype=np.int32)
        # step predator right (into prey cell would require setup)
        # Just confirm capture when already on same cell after location override
        pred._agent_location = np.array([3, 2], dtype=np.int32)
        out = env.step({ag.agent_name: 4 for ag in env.agents})
        assert prey.agent_name in env._captured_agents or out["terminated"]

    def test_no_capture_without_overlap(self):
        env = make_env(size=10, perc_obstacle=0, seed=0)
        obs, _ = env.reset()
        # Spread agents far apart
        env.agents[0]._agent_location = np.array([0, 0], dtype=np.int32)
        env.agents[1]._agent_location = np.array([9, 9], dtype=np.int32)
        out = env.step({aid: 4 for aid in obs})
        assert not out["terminated"]

    def test_capture_threshold_not_met(self):
        """threshold=2, 2 predators + 1 prey: one capture is not enough to terminate."""
        agents = [
            Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
            Agent(agent_type="predator", agent_team="predator_2", agent_name="pred_2"),
            Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
            Agent(agent_type="prey", agent_team="prey_2", agent_name="prey_2"),
        ]
        env = GridWorldEnv(
            agents=agents,
            size=6,
            perc_num_obstacle=0,
            render_mode=None,
            seed=0,
            capture_threshold=2,
        )
        env.reset()
        pred = next(a for a in env.agents if a.agent_name == "pred_1")
        prey = next(a for a in env.agents if a.agent_name == "prey_1")
        # Force one predator onto one prey cell — only 1 capture, threshold is 2
        pred._agent_location = prey._agent_location.copy()
        out = env.step({ag.agent_name: 4 for ag in env.agents})
        assert not out["terminated"]  # 1 capture < threshold of 2


# ------------------------------------------------------------------
# Truncation
# ------------------------------------------------------------------


class TestTruncation:
    def test_truncation_at_max_steps(self):
        env = make_env(perc_obstacle=0, max_steps=3)
        obs, _ = env.reset()
        actions = {aid: 4 for aid in obs}
        for _ in range(2):
            out = env.step(actions)
            assert not out["truncated"]
        out = env.step(actions)
        assert out["truncated"]

    def test_no_truncation_without_max_steps(self):
        env = make_env(perc_obstacle=0, max_steps=None)
        obs, _ = env.reset()
        for _ in range(20):
            out = env.step({aid: 4 for aid in obs})
            assert not out["truncated"]


# ------------------------------------------------------------------
# Extension hooks
# ------------------------------------------------------------------


class TestExtensionHooks:
    def test_custom_reward_fn_added(self):
        env = make_env(perc_obstacle=0)
        env.reward_fn = lambda e: {ag.agent_name: 100.0 for ag in e.agents}
        obs, _ = env.reset()
        out = env.step({aid: 4 for aid in obs})
        # step() applies no base reward itself, so only the custom fn counts
        assert out["reward"]["pred_1"] == pytest.approx(100.0)

    def test_custom_observation_builder_called(self):
        env = make_env(perc_obstacle=0)
        called = []

        def custom_builder(e):
            called.append(True)
            return {ag.agent_name: {"custom": True} for ag in e.agents}

        env.observation_builder = custom_builder
        obs, _ = env.reset()
        assert called  # builder was invoked during reset
