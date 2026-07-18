"""
Integration tests: config loading, environment building, and end-to-end training.
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


class TestLoadAllConfigs:
    def test_loads_five_sections(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert set(configs.keys()) == {
            "env",
            "agents",
            "observations",
            "rewards",
            "actions",
            "experiment",
        }

    def test_env_section_has_size(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "size" in configs["env"]["env"]

    def test_agents_section_has_predators_and_preys(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "predators" in configs["agents"]["agents"]
        assert "preys" in configs["agents"]["agents"]

    def test_observations_has_type(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "type" in configs["observations"]["observations"]

    def test_rewards_has_base(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "base" in configs["rewards"]["rewards"]

    def test_experiment_has_algorithm(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "algorithm" in configs["experiment"]["experiment"]

    def test_algorithm_has_name(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs()
        assert "name" in configs["experiment"]["experiment"]["algorithm"]

    def test_iql_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs(experiment_file="experiment_iql.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "iql"

    def test_cql_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs(experiment_file="experiment_cql.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "cql"

    def test_mixed_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs(experiment_file="experiment_mixed.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "mixed"

    def test_dqn_experiment_file(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs(experiment_file="experiment_dqn.yaml")
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "dqn"


# ------------------------------------------------------------------
# build_agents
# ------------------------------------------------------------------


class TestBuildAgents:
    def test_correct_agent_count(self):
        from multi_agent_package.scripts.run_from_config import (
            build_agents,
            load_all_configs,
        )

        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        pred_count = configs["agents"]["agents"]["predators"]["count"]
        prey_count = configs["agents"]["agents"]["preys"]["count"]
        assert len(agents) == pred_count + prey_count

    def test_agent_types_correct(self):
        from multi_agent_package.scripts.run_from_config import (
            build_agents,
            load_all_configs,
        )

        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        pred_count = configs["agents"]["agents"]["predators"]["count"]
        types = [a.agent_type for a in agents]
        assert types[:pred_count] == ["predator"] * pred_count

    def test_agent_names_unique(self):
        from multi_agent_package.scripts.run_from_config import (
            build_agents,
            load_all_configs,
        )

        configs = load_all_configs()
        agents = build_agents(configs["agents"])
        names = [a.agent_name for a in agents]
        assert len(names) == len(set(names))


# ------------------------------------------------------------------
# build_environment
# ------------------------------------------------------------------


class TestBuildEnvironment:
    def _load_and_build(self, experiment_file="experiment_iql.yaml"):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )

        # Override render_mode to None so no display is needed
        configs = load_all_configs(experiment_file=experiment_file)
        configs["env"]["env"]["render_mode"] = None
        return build_environment(configs)

    def test_returns_gridworld_env(self):
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.wrappers.speed import SpeedWrapper

        env = self._load_and_build()
        # build_environment wraps the raw GridWorldEnv in SpeedWrapper to honor
        # per-agent agent_speed; the wrapper proxies everything else through.
        assert isinstance(env, SpeedWrapper)
        assert isinstance(env.env, GridWorldEnv)

    def test_observation_builder_attached(self):
        env = self._load_and_build()
        assert env.observation_builder is not None
        assert callable(env.observation_builder)

    def test_reward_fn_attached(self):
        env = self._load_and_build()
        assert env.reward_fn is not None
        assert callable(env.reward_fn)

    def test_env_resets_successfully(self):
        env = self._load_and_build()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) > 0

    def test_env_steps_successfully(self):
        env = self._load_and_build()
        obs, _ = env.reset()
        actions = {aid: 4 for aid in obs}
        out = env.step(actions)
        assert "obs" in out
        assert "reward" in out


# ------------------------------------------------------------------
# Base reward enters only through the plugin pipeline (issue #32)
# ------------------------------------------------------------------


class TestBaseRewardIsPluginDriven:
    def _build(self, base_enabled):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )

        configs = load_all_configs(experiment_file="experiment_iql.yaml")
        configs["env"]["env"]["render_mode"] = None
        configs["rewards"]["rewards"]["base"]["enabled"] = base_enabled
        return build_environment(configs)

    def _first_step_reward(self, env):
        obs, _ = env.reset()
        return env.step({aid: 4 for aid in obs})["reward"]

    def test_base_enabled_adds_step_cost(self):
        # With base enabled, a NOOP predator incurs the negative base step
        # cost on top of shaping; disabling base removes exactly that
        # component. This is the single-application-path guarantee of #32.
        on = self._first_step_reward(self._build(True))
        off = self._first_step_reward(self._build(False))
        preds = [k for k in on if k.startswith("pred")]
        assert preds, "expected predator agents in the IQL config"
        for k in preds:
            assert on[k] < off[k], f"base reward not applied for {k}"

    def test_base_disabled_leaves_only_shaping(self):
        # Disabling base must not zero out shaping (prey survival shaping stays).
        off = self._first_step_reward(self._build(False))
        prey = [v for k, v in off.items() if k.startswith("prey")]
        assert prey and all(v > 0 for v in prey)


# ------------------------------------------------------------------
# End-to-end training: IQL
# ------------------------------------------------------------------


class TestEndToEndIQL:
    def test_iql_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )
        import baselines  # noqa: F401
        from baselines.IQL.iql import IQL

        configs = load_all_configs(experiment_file="experiment_iql.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3  # keep test fast

        algo = IQL(env, params)
        algo.train()

        for table in algo.q_tables.values():
            assert len(table) > 0


# ------------------------------------------------------------------
# End-to-end training: CQL
# ------------------------------------------------------------------


class TestEndToEndCQL:
    def test_cql_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )
        import baselines  # noqa: F401
        from baselines.CQL.cql import CQL

        configs = load_all_configs(experiment_file="experiment_cql.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3

        algo = CQL(env, params)
        algo.train()
        assert len(algo.q_table) > 0


# ------------------------------------------------------------------
# End-to-end training: Mixed
# ------------------------------------------------------------------


class TestEndToEndMixed:
    def test_mixed_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )
        import baselines  # noqa: F401
        from baselines.MIXED.mix_train import MixedTrainer

        configs = load_all_configs(experiment_file="experiment_mixed.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3

        algo = MixedTrainer(env, params)
        algo.train()

        # At least one table must be populated
        all_tables = list(algo._iql_tables.values()) + list(algo._cql_tables.values())
        assert any(len(t) > 0 for t in all_tables)


# ------------------------------------------------------------------
# End-to-end training: DQN (3v3, default config)
# ------------------------------------------------------------------


class TestEndToEndDQN:
    def test_dqn_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )
        import baselines  # noqa: F401
        from baselines.DQN.dqn import DQN

        configs = load_all_configs(experiment_file="experiment_dqn.yaml")
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3
        params["curves_path"] = None  # no file output during tests

        algo = DQN(env, params)
        algo.train()

        assert all(len(buffer) > 0 for buffer in algo.replay_buffers.values())


# ------------------------------------------------------------------
# End-to-end training: DQN (1v1 config, predator 2x prey speed)
# ------------------------------------------------------------------


class TestEndToEndDQN1v1:
    def test_dqn_1v1_config_loads(self):
        from multi_agent_package.scripts.run_from_config import load_all_configs

        configs = load_all_configs(
            config_dir="configs/dqn_1v1",
            experiment_file="experiment_dqn.yaml",
        )
        assert configs["agents"]["agents"]["predators"]["count"] == 1
        assert configs["agents"]["agents"]["preys"]["count"] == 1
        assert configs["agents"]["agents"]["predators"]["speed"] == 2
        assert configs["experiment"]["experiment"]["algorithm"]["name"] == "dqn"

    def test_dqn_1v1_trains_without_error(self):
        from multi_agent_package.scripts.run_from_config import (
            load_all_configs,
            build_environment,
        )
        import baselines  # noqa: F401
        from baselines.DQN.dqn import DQN

        configs = load_all_configs(
            config_dir="configs/dqn_1v1",
            experiment_file="experiment_dqn.yaml",
        )
        configs["env"]["env"]["render_mode"] = None
        env = build_environment(configs)

        params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
        params["episodes"] = 3
        params["curves_path"] = None  # no file output during tests

        algo = DQN(env, params)
        algo.train()

        assert len(algo.agent_ids) == 2  # 1 predator + 1 prey
        assert all(len(buffer) > 0 for buffer in algo.replay_buffers.values())


# ------------------------------------------------------------------
# SpeedWrapper stamina and sub-step behaviour
# ------------------------------------------------------------------


class TestSpeedWrapper:
    def _make_wrapped(
        self, pred_speed=2, pred_stamina=9999, prey_speed=1, prey_stamina=9999
    ):
        from multi_agent_package.core.agent import Agent
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.observations.local_only import LocalOnlyObservation
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        from multi_agent_package.wrappers.speed import SpeedWrapper

        agents = [
            Agent(agent_type="predator", agent_team="pred", agent_name="pred_1"),
            Agent(agent_type="prey", agent_team="prey", agent_name="prey_1"),
        ]
        agents[0].agent_speed = pred_speed
        agents[0].stamina = pred_stamina
        agents[1].agent_speed = prey_speed
        agents[1].stamina = prey_stamina

        env = GridWorldEnv(
            agents=agents, size=5, perc_num_obstacle=0, render_mode=None, seed=0
        )
        env.observation_builder = LocalOnlyObservation().build
        env.action_space_plugin = DiscreteActionSpace()
        return SpeedWrapper(env)

    def test_max_stamina_read_from_agent(self):
        wrapped = self._make_wrapped(pred_stamina=50, prey_stamina=80)
        assert wrapped._max_stamina["pred_1"] == 50
        assert wrapped._max_stamina["prey_1"] == 80

    def test_stamina_restored_on_reset(self):
        wrapped = self._make_wrapped(pred_speed=3, pred_stamina=10)
        wrapped.reset()
        wrapped._stamina["pred_1"] = 0  # drain it
        wrapped.reset()
        assert wrapped._stamina["pred_1"] == 10

    def test_stamina_depletes_by_speed_each_step(self):
        wrapped = self._make_wrapped(pred_speed=3, pred_stamina=9999)
        wrapped.reset()
        before = wrapped._stamina["pred_1"]
        wrapped.step({"pred_1": 0, "prey_1": 4})  # pred moves, prey NOOPs
        # predator took 3 sub-steps → 3 stamina deducted
        assert wrapped._stamina["pred_1"] == before - 3

    def test_noop_costs_zero_stamina(self):
        wrapped = self._make_wrapped(pred_speed=3, pred_stamina=9999)
        wrapped.reset()
        before = wrapped._stamina["pred_1"]
        wrapped.step({"pred_1": 4, "prey_1": 4})  # both NOOP
        assert wrapped._stamina["pred_1"] == before

    def test_stamina_cap_limits_sub_steps(self):
        # with stamina=2 and speed=3, predator gets only 2 sub-steps (not 3)
        wrapped = self._make_wrapped(pred_speed=3, pred_stamina=2)
        wrapped.reset()
        wrapped._stamina["pred_1"] = 2  # ensure known start value
        wrapped.step({"pred_1": 0, "prey_1": 4})
        assert wrapped._stamina["pred_1"] == 0  # depleted by min(3, 2) = 2

    def test_speed1_fast_path_no_stamina_deduction(self):
        wrapped = self._make_wrapped(
            pred_speed=1, pred_stamina=9999, prey_speed=1, prey_stamina=9999
        )
        assert wrapped._max_speed == 1
        wrapped.reset()
        before = wrapped._stamina["pred_1"]
        wrapped.step({"pred_1": 0, "prey_1": 4})
        # fast path bypasses sub-step loop entirely; stamina unchanged
        assert wrapped._stamina["pred_1"] == before
