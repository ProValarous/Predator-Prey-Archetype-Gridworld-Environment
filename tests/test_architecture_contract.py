"""
Architecture-contract tests.

These encode the rules from CONTRIBUTING.md's "Golden Rule": students extend
the system through the ObservationBuilder / RewardFunction / ActionSpace
plugin base classes and the registries, and plugins never mutate core env or
agent state. They complement (not replace) test_registry.py's per-registry
unit tests and test_integration.py's end-to-end baseline training tests.

Modifying core/gridworld.py or core/agent.py directly is enforced separately
by the `core-guard` job in .github/workflows/ci.yaml (a diff-based PR check),
not here — a hash/content check in pytest would just have to be updated by
hand every time a maintainer legitimately touches core/.
"""

import glob
import os

import pytest

from multi_agent_package.actions.base import ActionSpace
from multi_agent_package.observations.base import ObservationBuilder
from multi_agent_package.registry import get_observation_builder, get_reward_function
from multi_agent_package.registry.action_registry import _ACTION_REGISTRY
from multi_agent_package.registry.observation_registry import _OBSERVATION_REGISTRY
from multi_agent_package.registry.reward_registry import _REWARD_REGISTRY
from multi_agent_package.rewards.base import RewardFunction

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_DIRS = sorted(
    os.path.relpath(p, _REPO_ROOT)
    for p in glob.glob(os.path.join(_REPO_ROOT, "configs", "*"))
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "experiment_dqn.yaml"))
)


# ------------------------------------------------------------------
# Plugin base classes must stay abstract — a plugin author can only
# participate by subclassing and implementing the required methods.
# ------------------------------------------------------------------


class TestPluginBaseClassesAreAbstract:
    def test_observation_builder_is_abstract(self):
        with pytest.raises(TypeError):
            ObservationBuilder()

    def test_reward_function_is_abstract(self):
        with pytest.raises(TypeError):
            RewardFunction()

    def test_action_space_is_abstract(self):
        with pytest.raises(TypeError):
            ActionSpace()


# ------------------------------------------------------------------
# Every plugin actually wired into a registry must be a real subclass
# of the matching base class, not just a duck-typed lookalike.
# ------------------------------------------------------------------


class TestAllRegisteredPluginsConformToContract:
    @pytest.mark.parametrize("name,cls", sorted(_OBSERVATION_REGISTRY.items()))
    def test_observation_plugin_is_subclass(self, name, cls):
        assert issubclass(cls, ObservationBuilder), (
            f"observation plugin '{name}' ({cls.__name__}) must inherit "
            "from ObservationBuilder"
        )

    @pytest.mark.parametrize("name,cls", sorted(_REWARD_REGISTRY.items()))
    def test_reward_plugin_is_subclass(self, name, cls):
        assert issubclass(
            cls, RewardFunction
        ), f"reward plugin '{name}' ({cls.__name__}) must inherit from RewardFunction"

    @pytest.mark.parametrize("name,cls", sorted(_ACTION_REGISTRY.items()))
    def test_action_plugin_is_subclass(self, name, cls):
        assert issubclass(
            cls, ActionSpace
        ), f"action plugin '{name}' ({cls.__name__}) must inherit from ActionSpace"


# ------------------------------------------------------------------
# Observations/rewards are documented as read-only (base.py: "Do NOT
# modify env state"). Verify no shipped plugin moves agents as a
# side effect of build()/compute().
# ------------------------------------------------------------------


class TestPluginsDoNotMutateEnvState:
    @staticmethod
    def _agent_positions(env):
        return {ag.agent_name: tuple(ag._agent_location) for ag in env.agents}

    @pytest.mark.parametrize("obs_name", sorted(_OBSERVATION_REGISTRY))
    def test_observation_build_does_not_move_agents(self, obs_name, small_env):
        small_env.reset()
        before = self._agent_positions(small_env)
        builder = get_observation_builder(obs_name)
        builder.build(small_env)
        builder.build(small_env)
        assert self._agent_positions(small_env) == before

    @pytest.mark.parametrize("reward_name", sorted(_REWARD_REGISTRY))
    def test_reward_compute_does_not_move_agents(self, reward_name, small_env):
        small_env.reset()
        before = self._agent_positions(small_env)
        reward_fn = get_reward_function(reward_name)
        reward_fn.compute(small_env)
        assert self._agent_positions(small_env) == before


# ------------------------------------------------------------------
# Full config-driven pipeline, swept across every shipped config set
# (configs/dqn_1v1, dqn_speed1, dqn_speed2, dqn_speed3, ...). A new
# config directory that breaks the pipeline should fail here instead
# of only surfacing when someone happens to run it manually.
# ------------------------------------------------------------------


class TestConfigPipelineEndToEnd:
    @pytest.mark.parametrize("config_dir", _CONFIG_DIRS)
    def test_pipeline_runs_end_to_end(self, config_dir):
        from multi_agent_package.scripts.run_from_config import (
            build_environment,
            load_all_configs,
        )

        configs = load_all_configs(
            config_dir=config_dir, experiment_file="experiment_dqn.yaml"
        )
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        obs, info = env.reset()
        assert obs is not None
        assert info is not None

        for _ in range(5):
            actions = {ag.agent_name: 0 for ag in env.agents}
            result = env.step(actions)
            assert {"obs", "reward", "terminated", "truncated", "info"} <= result.keys()
            if result["terminated"] or result["truncated"]:
                break

        env.close()
