"""
Tests for the A3C baseline: SharedAdam, the worker-loop logic, and the
A3C algorithm itself.

Most tests call `_worker_loop` directly (single process, real
multiprocessing.Value/Lock objects work fine without an actual spawned
process) to exercise the full rollout/bootstrap/loss/Hogwild-update logic
without process-spawn overhead. Exactly one test (`TestA3CTrain`) spawns
real worker processes end-to-end, since that's the only way to catch
pickling/spawn-specific failures the direct-call tests can't see.
"""

import pytest
import torch
import torch.multiprocessing as mp

from baselines.AC.network import ActorCriticNetwork
from baselines.A3C.shared_adam import SharedAdam
from baselines.A3C.a3c import _worker_loop


class _PicklableEnvFactory:
    """Module-level, picklable env builder -- required for the real-spawn
    test since a lambda/closure will not survive pickling under 'spawn'."""

    def __call__(self):
        from multi_agent_package.core.agent import Agent
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.observations.local_only import LocalOnlyObservation
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        from multi_agent_package.rewards.base_reward import BaseReward

        agents = [
            Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
            Agent(agent_type="prey", agent_team="prey_1", agent_name="prey_1"),
        ]
        env = GridWorldEnv(
            agents=agents,
            size=5,
            perc_num_obstacle=0,
            render_mode=None,
            seed=42,
            max_steps=20,
        )
        ob = LocalOnlyObservation()
        env.observation_builder = ob.build
        env.observation_encoder = ob.encode
        env.action_space_plugin = DiscreteActionSpace()
        env.reward_fn = BaseReward(weight=1.0).compute
        return env


# ------------------------------------------------------------------
# SharedAdam
# ------------------------------------------------------------------


class TestSharedAdam:
    def test_state_tensors_are_shared(self):
        net = ActorCriticNetwork(4, [8], 5)
        opt = SharedAdam(net.parameters(), lr=1e-3)
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state[p]
                assert state["step"].is_shared()
                assert state["exp_avg"].is_shared()
                assert state["exp_avg_sq"].is_shared()

    def test_step_runs_without_error(self):
        net = ActorCriticNetwork(4, [8], 5)
        opt = SharedAdam(net.parameters(), lr=1e-3)
        logits, value = net(torch.randn(2, 4))
        loss = logits.sum() + value.sum()
        loss.backward()
        opt.step()  # should not raise


# ------------------------------------------------------------------
# _worker_loop (direct call, single process -- no real spawning)
# ------------------------------------------------------------------


class TestWorkerLoopLogic:
    def _build_globals(self, state_dim, hidden_layers, action_dim, agent_ids, lr=0.01):
        networks = {}
        optimizers = {}
        for aid in agent_ids:
            net = ActorCriticNetwork(state_dim, hidden_layers, action_dim)
            net.share_memory()
            networks[aid] = net
            optimizers[aid] = SharedAdam(net.parameters(), lr=lr)
        return networks, optimizers

    def test_updates_shared_network_weights(self, dqn_env, a3c_config):
        agent_ids = ["pred_1", "prey_1"]
        state_dim = 2  # local_only encodes [x, y]
        action_dim = 5
        networks, optimizers = self._build_globals(
            state_dim, a3c_config["hidden_layers"], action_dim, agent_ids
        )
        before = {
            aid: {k: v.clone() for k, v in net.state_dict().items()}
            for aid, net in networks.items()
        }

        episode_counter = mp.Value("i", 0)
        counter_lock = mp.Lock()

        _worker_loop(
            worker_id=0,
            agent_ids=agent_ids,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=a3c_config["hidden_layers"],
            global_networks=networks,
            optimizers=optimizers,
            env_fn=lambda: dqn_env,
            gamma=a3c_config["gamma"],
            n_steps=a3c_config["n_steps"],
            entropy_coef=a3c_config["entropy_coef"],
            value_coef=a3c_config["value_coef"],
            grad_clip=a3c_config["grad_clip"],
            episode_counter=episode_counter,
            max_episodes=2,
            counter_lock=counter_lock,
            csv_lock=None,
            csv_path=None,
            log_interval=1,
            verbose=False,
            seed=0,
        )

        assert episode_counter.value == 2
        for aid, net in networks.items():
            after = net.state_dict()
            assert any(not torch.equal(before[aid][k], after[k]) for k in after)

    def test_normalize_returns_updates_shared_network_weights(self, dqn_env, a3c_config):
        agent_ids = ["pred_1", "prey_1"]
        state_dim = 2
        action_dim = 5
        networks, optimizers = self._build_globals(
            state_dim, a3c_config["hidden_layers"], action_dim, agent_ids
        )
        before = {
            aid: {k: v.clone() for k, v in net.state_dict().items()}
            for aid, net in networks.items()
        }

        episode_counter = mp.Value("i", 0)
        counter_lock = mp.Lock()

        _worker_loop(
            worker_id=0,
            agent_ids=agent_ids,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=a3c_config["hidden_layers"],
            global_networks=networks,
            optimizers=optimizers,
            env_fn=lambda: dqn_env,
            gamma=a3c_config["gamma"],
            n_steps=a3c_config["n_steps"],
            entropy_coef=a3c_config["entropy_coef"],
            value_coef=a3c_config["value_coef"],
            grad_clip=a3c_config["grad_clip"],
            episode_counter=episode_counter,
            max_episodes=2,
            counter_lock=counter_lock,
            csv_lock=None,
            csv_path=None,
            log_interval=1,
            verbose=False,
            seed=0,
            normalize_returns=True,
            return_norm_decay=0.99,
        )

        assert episode_counter.value == 2
        for aid, net in networks.items():
            after = net.state_dict()
            assert any(not torch.equal(before[aid][k], after[k]) for k in after)

    def test_writes_csv_rows(self, dqn_env, a3c_config, tmp_path):
        agent_ids = ["pred_1", "prey_1"]
        state_dim = 2
        action_dim = 5
        networks, optimizers = self._build_globals(
            state_dim, a3c_config["hidden_layers"], action_dim, agent_ids
        )
        csv_path = tmp_path / "a3c_curves.csv"
        # Header written manually here to mirror what A3C.train() does before
        # spawning workers -- _worker_loop always opens csv_path in append mode.
        with open(csv_path, "w", newline="") as f:
            f.write(
                "episode,worker,"
                + ",".join(f"{a}_reward" for a in agent_ids)
                + ","
                + ",".join(f"{a}_loss" for a in agent_ids)
                + ","
                + ",".join(f"{a}_entropy" for a in agent_ids)
                + "\n"
            )

        episode_counter = mp.Value("i", 0)
        counter_lock = mp.Lock()
        csv_lock = mp.Lock()

        _worker_loop(
            worker_id=0,
            agent_ids=agent_ids,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=a3c_config["hidden_layers"],
            global_networks=networks,
            optimizers=optimizers,
            env_fn=lambda: dqn_env,
            gamma=a3c_config["gamma"],
            n_steps=a3c_config["n_steps"],
            entropy_coef=a3c_config["entropy_coef"],
            value_coef=a3c_config["value_coef"],
            grad_clip=a3c_config["grad_clip"],
            episode_counter=episode_counter,
            max_episodes=2,
            counter_lock=counter_lock,
            csv_lock=csv_lock,
            csv_path=str(csv_path),
            log_interval=1,
            verbose=False,
            seed=0,
        )

        rows = csv_path.read_text().strip().splitlines()
        assert len(rows) == 3  # header + 2 episode rows
        assert rows[0].startswith("episode,worker,")


# ------------------------------------------------------------------
# A3C (BaseAlgorithm interface)
# ------------------------------------------------------------------


class TestA3CInit:
    def test_infers_state_dim_and_action_dim(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        assert algo.state_dim == 2
        assert algo.action_dim == 5

    def test_one_learner_per_agent(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        assert set(algo.networks.keys()) == {"pred_1", "prey_1"}
        assert set(algo.optimizers.keys()) == {"pred_1", "prey_1"}

    def test_actor_weight_decay_scoped_to_policy_head_only(self, dqn_env, a3c_config):
        """actor_weight_decay must land only on policy_head's param group --
        the trunk and value_head are shared with the critic across every
        worker, so decaying them too would regularize value estimation, not
        just the policy (see docs/algorithms/a3c.md)."""
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        a3c_config["actor_weight_decay"] = 0.001
        algo = A3C(dqn_env, a3c_config)
        agent_id = algo.agent_ids[0]
        network = algo.networks[agent_id]
        optimizer = algo.optimizers[agent_id]

        policy_head_params = set(id(p) for p in network.policy_head.parameters())
        other_params = set(id(p) for p in network.parameters()) - policy_head_params

        seen = set()
        for group in optimizer.param_groups:
            group_param_ids = set(id(p) for p in group["params"])
            if group_param_ids == policy_head_params:
                assert group["weight_decay"] == 0.001
            elif group_param_ids == other_params:
                assert group["weight_decay"] == 0
            else:
                pytest.fail("Unexpected optimizer param group composition")
            seen |= group_param_ids
        assert seen == policy_head_params | other_params

    def test_default_actor_weight_decay_is_zero(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        assert algo.actor_weight_decay == 0.0

    def test_default_normalize_returns_is_false(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        assert algo.normalize_returns is False

    def test_normalize_returns_config_is_parsed(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        a3c_config["normalize_returns"] = True
        a3c_config["return_norm_decay"] = 0.95
        algo = A3C(dqn_env, a3c_config)
        assert algo.normalize_returns is True
        assert algo.return_norm_decay == 0.95

    def test_missing_env_fn_raises(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        with pytest.raises(ValueError):
            A3C(dqn_env, a3c_config)

    def test_non_callable_env_fn_raises(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = "not a callable"
        with pytest.raises(ValueError):
            A3C(dqn_env, a3c_config)

    def test_missing_observation_encoder_raises(
        self, one_predator_one_prey, a3c_config
    ):
        from multi_agent_package.core.gridworld import GridWorldEnv
        from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
        from baselines.A3C.a3c import A3C

        env = GridWorldEnv(
            agents=one_predator_one_prey,
            size=5,
            perc_num_obstacle=0,
            render_mode=None,
            seed=0,
        )
        env.action_space_plugin = DiscreteActionSpace()
        a3c_config["env_fn"] = lambda: env
        with pytest.raises(ValueError):
            A3C(env, a3c_config)

    def test_action_dim_mismatch_raises(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        a3c_config["action_dim"] = 9
        with pytest.raises(ValueError):
            A3C(dqn_env, a3c_config)

    def test_non_positive_n_steps_raises(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        a3c_config["n_steps"] = 0
        with pytest.raises(ValueError):
            A3C(dqn_env, a3c_config)

    def test_non_positive_num_workers_raises(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        a3c_config["num_workers"] = 0
        with pytest.raises(ValueError):
            A3C(dqn_env, a3c_config)


class TestA3CSelectActions:
    def test_returns_valid_actions_for_all_agents(self, dqn_env, a3c_config):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        obs, _ = dqn_env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == {"pred_1", "prey_1"}
        for a in actions.values():
            assert 0 <= a < algo.action_dim


class TestA3CTrain:
    def test_trains_end_to_end_with_real_worker_processes(self, a3c_config):
        """The one test in this file that spawns real OS processes --
        exercises pickling of env_fn/networks/optimizers and the actual
        'spawn' start method, which the direct _worker_loop tests above
        cannot catch."""
        from baselines.A3C.a3c import A3C

        env_fn = _PicklableEnvFactory()
        env = env_fn()

        a3c_config["env_fn"] = env_fn
        a3c_config["episodes"] = 4
        algo = A3C(env, a3c_config)

        before = {
            aid: {k: v.clone() for k, v in net.state_dict().items()}
            for aid, net in algo.networks.items()
        }

        algo.train()

        for aid, net in algo.networks.items():
            after = net.state_dict()
            assert any(not torch.equal(before[aid][k], after[k]) for k in after)

        env.close()

    def test_normalize_returns_trains_end_to_end_with_real_worker_processes(
        self, a3c_config
    ):
        """normalize_returns adds two new args threaded through mp.Process --
        this is the only way to catch a pickling/spawn-argument-order bug in
        that plumbing, since the direct _worker_loop tests call it in-process
        with keyword args and wouldn't notice a positional mismatch."""
        from baselines.A3C.a3c import A3C

        env_fn = _PicklableEnvFactory()
        env = env_fn()

        a3c_config["env_fn"] = env_fn
        a3c_config["episodes"] = 4
        a3c_config["normalize_returns"] = True
        algo = A3C(env, a3c_config)

        before = {
            aid: {k: v.clone() for k, v in net.state_dict().items()}
            for aid, net in algo.networks.items()
        }

        algo.train()

        for aid, net in algo.networks.items():
            after = net.state_dict()
            assert any(not torch.equal(before[aid][k], after[k]) for k in after)

        env.close()

    def test_csv_header_matches_worker_written_rows(self, a3c_config, tmp_path):
        """train() writes the CSV header itself, separately from the rows
        _worker_loop appends -- a header/row column mismatch here would slip
        past every other test since none of them read the header back."""
        from baselines.A3C.a3c import A3C

        env_fn = _PicklableEnvFactory()
        env = env_fn()

        csv_path = tmp_path / "a3c_train_curves.csv"
        a3c_config["env_fn"] = env_fn
        a3c_config["episodes"] = 2
        a3c_config["curves_path"] = str(csv_path)
        algo = A3C(env, a3c_config)
        algo.train()

        rows = csv_path.read_text().strip().splitlines()
        expected_header = (
            "episode,worker,"
            + ",".join(f"{aid}_reward" for aid in algo.agent_ids)
            + ","
            + ",".join(f"{aid}_loss" for aid in algo.agent_ids)
            + ","
            + ",".join(f"{aid}_entropy" for aid in algo.agent_ids)
        )
        assert rows[0] == expected_header
        for data_row in rows[1:]:
            assert len(data_row.split(",")) == len(expected_header.split(","))

        env.close()


class TestA3CPersistence:
    def test_save_creates_file(self, dqn_env, a3c_config, tmp_path):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        path = tmp_path / "a3c.pkl"
        algo.save(str(path))
        assert path.exists()

    def test_load_restores_weights(self, dqn_env, a3c_config, tmp_path):
        from baselines.A3C.a3c import A3C

        a3c_config["env_fn"] = lambda: dqn_env
        algo = A3C(dqn_env, a3c_config)
        path = tmp_path / "a3c.pkl"
        algo.save(str(path))

        loaded = A3C.load(dqn_env, a3c_config, str(path))
        for aid in algo.agent_ids:
            orig_sd = algo.networks[aid].state_dict()
            loaded_sd = loaded.networks[aid].state_dict()
            for key in orig_sd:
                assert torch.equal(orig_sd[key], loaded_sd[key])
