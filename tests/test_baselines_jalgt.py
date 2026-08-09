"""
Tests for JAL-GT (Joint-Action Learning with Game Theory), instantiated as
correlated Q-learning -- see docs/algorithms/jal-gt.md and Algorithm 7 in
Albrecht, Christianos & Tuyls (2024), Section 6.2.
"""

import os
import tempfile

import numpy as np
import pytest

from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.rewards.base_reward import BaseReward
from baselines.JALGT.jal_gt import JALGT

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_env(n_pred=1, n_prey=1, size=5, seed=0):
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
    env = GridWorldEnv(
        agents=agents, size=size, perc_num_obstacle=0, render_mode=None, seed=seed
    )
    env.reward_fn = BaseReward(weight=1.0).compute
    return env


def base_config(**overrides):
    cfg = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 1.0,
        "min_epsilon": 0.0,
        "action_dim": 5,
        "episodes": 5,
        "seed": 0,
    }
    cfg.update(overrides)
    return cfg


def is_correlated_equilibrium(probs, q_values, agent_ids, action_dim, atol=1e-6):
    """Independent, from-scratch check of Eq. 4.19 -- does NOT reuse any of
    JALGT's own LP-construction code, so it can actually catch bugs in it
    rather than agreeing with itself. `probs` is a length-n_joint_actions
    distribution over joint actions (index = a_0*K^(n-1) + ... + a_{n-1},
    agent_ids[0] most significant); `q_values` is {agent_id: reward vector}.
    """
    n_agents = len(agent_ids)
    n_joint = action_dim**n_agents

    def component(idx, pos):
        power = action_dim ** (n_agents - 1 - pos)
        return (idx // power) % action_dim

    def deviate(idx, pos, new_action):
        power = action_dim ** (n_agents - 1 - pos)
        old = component(idx, pos)
        return idx + (new_action - old) * power

    for pos, aid in enumerate(agent_ids):
        qi = q_values[aid]
        for a_prime in range(action_dim):
            for a_double_prime in range(action_dim):
                if a_prime == a_double_prime:
                    continue
                lhs = 0.0
                rhs = 0.0
                for ja in range(n_joint):
                    if component(ja, pos) != a_prime:
                        continue
                    lhs += probs[ja] * qi[ja]
                    rhs += probs[ja] * qi[deviate(ja, pos, a_double_prime)]
                if lhs < rhs - atol:
                    return False
    return True


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestJALGTInit:
    def test_n_agents(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config())
        assert algo.n_agents == 2

    def test_joint_action_space_size(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5))
        assert algo.n_joint_actions == 5**2

    def test_one_q_table_per_agent(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config())
        assert set(algo.q_tables.keys()) == set(algo.agent_ids)

    def test_q_tables_start_empty(self):
        env = make_env()
        algo = JALGT(env, base_config())
        for table in algo.q_tables.values():
            assert len(table) == 0

    def test_q_tables_are_independent_objects(self):
        """Each agent's table must be its own defaultdict, not accidentally
        the same shared object (a classic dict-comprehension-with-shared-
        mutable-default bug)."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config())
        js = algo._joint_state(env.reset()[0])
        algo.q_tables["pred_1"][js][0] = 99.0
        assert algo.q_tables["prey_1"][js][0] == 0.0


# ------------------------------------------------------------------
# Joint state encoding
# ------------------------------------------------------------------


class TestJALGTJointState:
    def test_joint_state_is_tuple(self):
        env = make_env()
        algo = JALGT(env, base_config())
        obs, _ = env.reset()
        assert isinstance(algo._joint_state(obs), tuple)

    def test_joint_state_hashable(self):
        env = make_env()
        algo = JALGT(env, base_config())
        obs, _ = env.reset()
        hash(algo._joint_state(obs))


# ------------------------------------------------------------------
# Joint action encoding / decoding
# ------------------------------------------------------------------


class TestJALGTJointAction:
    def test_index_unique_per_action_combo(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5))
        indices = set()
        for a1 in range(5):
            for a2 in range(5):
                idx = algo._joint_action_index({"pred_1": a1, "prey_1": a2})
                indices.add(idx)
        assert len(indices) == 25

    def test_encode_decode_roundtrip(self):
        env = make_env(n_pred=1, n_prey=2)
        algo = JALGT(env, base_config(action_dim=5))
        for a1 in range(5):
            for a2 in range(5):
                for a3 in range(5):
                    actions = {"pred_1": a1, "prey_1": a2, "prey_2": a3}
                    idx = algo._joint_action_index(actions)
                    assert algo._decode_joint_action(idx) == actions

    def test_component_matches_decode(self):
        env = make_env(n_pred=1, n_prey=2)
        algo = JALGT(env, base_config(action_dim=5))
        actions = {"pred_1": 3, "prey_1": 1, "prey_2": 4}
        idx = algo._joint_action_index(actions)
        for pos, aid in enumerate(algo.agent_ids):
            assert algo._component(idx, pos) == actions[aid]

    def test_deviate_changes_only_target_agent(self):
        env = make_env(n_pred=1, n_prey=2)
        algo = JALGT(env, base_config(action_dim=5))
        actions = {"pred_1": 3, "prey_1": 1, "prey_2": 4}
        idx = algo._joint_action_index(actions)
        deviated_idx = algo._deviate(idx, 1, 2)  # prey_1's action -> 2
        deviated = algo._decode_joint_action(deviated_idx)
        assert deviated == {"pred_1": 3, "prey_1": 2, "prey_2": 4}


# ------------------------------------------------------------------
# Correlated-equilibrium LP -- ground-truth checks against the book
# ------------------------------------------------------------------


class TestCorrelatedEquilibriumLP:
    def test_prisoners_dilemma_forces_mutual_defection(self):
        """Exact payoff matrix from Figure 6.8(a) in the book: C,C=-1,-1;
        C,D=-5,0; D,C=0,-5; D,D=-3,-3. D strictly dominates C for both
        agents regardless of the other's action, so ANY correlated
        equilibrium must put probability 1 on (D,D) -- a deterministic,
        exactly-checkable ground truth (unlike Chicken below, no ambiguity)."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=2))
        # action 0 = C, action 1 = D; index = a_i*2 + a_j
        Ri = np.array([-1.0, -5.0, 0.0, -3.0])  # (C,C) (C,D) (D,C) (D,D)
        Rj = np.array([-1.0, 0.0, -5.0, -3.0])
        q_values = {algo.agent_ids[0]: Ri, algo.agent_ids[1]: Rj}
        c, A_ub, b_ub, A_eq, b_eq = algo._build_correlated_equilibrium_lp(q_values)
        from scipy.optimize import linprog

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
        )
        assert res.success
        # index 3 = (D,D)
        assert res.x[3] == pytest.approx(1.0, abs=1e-6)
        assert res.x[0] == pytest.approx(0.0, abs=1e-6)

    def test_chicken_game_solution_is_a_valid_correlated_equilibrium(self):
        """Chicken matrix game (Figure 4.3): S,S=0,0; S,L=7,2; L,S=2,7;
        L,L=6,6. The book's own worked example (Section 4.6) gives one
        specific valid correlated equilibrium with total welfare 10
        (pi_c(L,L)=pi_c(S,L)=pi_c(L,S)=1/3, expected return 5 to each
        agent). Our LP explicitly maximizes total welfare (Eq. 4.20) over
        the SAME feasible set of correlated equilibria that example belongs
        to, so its optimum must be >= 10 -- a robust, book-grounded lower
        bound that doesn't require matching the book's specific numbers
        (which were illustrative, not claimed to be welfare-optimal)."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=2))
        # action 0 = S, action 1 = L; index = a_i*2 + a_j
        Ri = np.array([0.0, 7.0, 2.0, 6.0])  # (S,S) (S,L) (L,S) (L,L)
        Rj = np.array([0.0, 2.0, 7.0, 6.0])
        q_values = {algo.agent_ids[0]: Ri, algo.agent_ids[1]: Rj}
        c, A_ub, b_ub, A_eq, b_eq = algo._build_correlated_equilibrium_lp(q_values)
        from scipy.optimize import linprog

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
        )
        assert res.success
        probs = res.x
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert (probs >= -1e-9).all()

        welfare = float(np.dot(probs, Ri) + np.dot(probs, Rj))
        assert welfare >= 10.0 - 1e-6  # at least as good as the book's own example

        assert is_correlated_equilibrium(probs, q_values, algo.agent_ids, action_dim=2)

    def test_solve_stage_game_returns_valid_distribution(self):
        """End-to-end through the actual method (not just the raw LP
        builder), on an arbitrary populated Q-table."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5))
        obs, _ = env.reset()
        js = algo._joint_state(obs)
        rng = np.random.default_rng(0)
        for aid in algo.agent_ids:
            algo.q_tables[aid][js] = rng.normal(size=25)

        probs = algo._solve_stage_game(js)
        assert probs.shape == (25,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert (probs >= -1e-9).all()

        q_values = {aid: algo.q_tables[aid][js] for aid in algo.agent_ids}
        assert is_correlated_equilibrium(probs, q_values, algo.agent_ids, 5)


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------


class TestMarginalWeight:
    def test_rejects_invalid_marginal_weight(self):
        env = make_env(n_pred=1, n_prey=1)
        with pytest.raises(ValueError):
            JALGT(env, base_config(marginal_weight=-0.1))
        with pytest.raises(ValueError):
            JALGT(env, base_config(marginal_weight=1.1))

    def test_default_weight_is_zero_and_is_a_no_op(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5))
        assert algo.marginal_weight == 0.0
        q_values = {
            aid: np.random.default_rng(0).normal(size=25) for aid in algo.agent_ids
        }
        smoothed = algo._marginalized_q_values(q_values)
        for aid in algo.agent_ids:
            assert smoothed[aid] is q_values[aid]  # same object, not even copied

    def test_weight_one_replaces_with_pure_marginal(self):
        """At weight=1.0, every joint action sharing agent i's own action a_i
        must get EXACTLY the mean of agent i's raw Q over all joint actions
        with that a_i -- computed independently here, not by calling the
        method under test with different inputs."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5, marginal_weight=1.0))
        pred_id, prey_id = algo.agent_ids
        rng = np.random.default_rng(1)
        qi = rng.normal(size=25)
        q_values = {pred_id: qi, prey_id: np.zeros(25)}

        smoothed = algo._marginalized_q_values(q_values)

        # independent, brute-force expected marginal: predator is position 0
        # (most significant digit), so its own action = joint_idx // 5
        for joint_idx in range(25):
            own_action = joint_idx // 5
            same_own_action = [j for j in range(25) if j // 5 == own_action]
            expected = np.mean([qi[j] for j in same_own_action])
            assert smoothed[pred_id][joint_idx] == pytest.approx(expected)

    def test_intermediate_weight_is_a_convex_combination(self):
        env = make_env(n_pred=1, n_prey=1)
        algo_raw = JALGT(env, base_config(action_dim=5, marginal_weight=0.0))
        algo_full = JALGT(env, base_config(action_dim=5, marginal_weight=1.0))
        algo_half = JALGT(env, base_config(action_dim=5, marginal_weight=0.5))
        rng = np.random.default_rng(2)
        qi = rng.normal(size=25)
        q_values = {algo_half.agent_ids[0]: qi, algo_half.agent_ids[1]: np.zeros(25)}

        raw = algo_raw._marginalized_q_values(dict(q_values))
        full = algo_full._marginalized_q_values(dict(q_values))
        half = algo_half._marginalized_q_values(dict(q_values))

        aid = algo_half.agent_ids[0]
        expected_half = 0.5 * raw[aid] + 0.5 * full[aid]
        assert half[aid] == pytest.approx(expected_half)


class TestQInit:
    def test_rejects_invalid_q_init(self):
        env = make_env(n_pred=1, n_prey=1)
        with pytest.raises(ValueError):
            JALGT(env, base_config(q_init="bogus"))

    def test_default_q_init_is_zero(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5))
        assert algo.q_init == "zero"
        js = algo._joint_state(env.reset()[0])
        assert np.array_equal(algo.q_tables["pred_1"][js], np.zeros(25))

    def test_random_q_init_produces_nonzero_rows(self):
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5, q_init="random", seed=1))
        js = algo._joint_state(env.reset()[0])
        row = algo.q_tables["pred_1"][js]
        assert row.shape == (25,)
        assert not np.array_equal(row, np.zeros(25))

    def test_random_q_init_draws_independently_per_state_and_agent(self):
        """A fresh draw for every never-before-seen (agent, joint_state) pair
        -- not the same row reused everywhere, and not shared across agents
        (same independent-object requirement as the zero-init default)."""
        env = make_env(n_pred=1, n_prey=1)
        algo = JALGT(env, base_config(action_dim=5, q_init="random", seed=2))
        js1 = algo._joint_state(env.reset()[0])
        row_pred = algo.q_tables["pred_1"][js1]
        row_prey = algo.q_tables["prey_1"][js1]
        assert not np.array_equal(row_pred, row_prey)

        js2 = js1 + (("marker", 1),)  # a distinct, never-visited joint state
        row_pred_2 = algo.q_tables["pred_1"][js2]
        assert not np.array_equal(row_pred, row_pred_2)

    def test_q_init_scale_controls_draw_magnitude(self):
        env = make_env(n_pred=1, n_prey=1)
        algo_small = JALGT(
            env, base_config(action_dim=5, q_init="random", q_init_scale=0.001, seed=3)
        )
        algo_large = JALGT(
            env, base_config(action_dim=5, q_init="random", q_init_scale=10.0, seed=3)
        )
        js = algo_small._joint_state(env.reset()[0])
        small_row = algo_small.q_tables["pred_1"][js]
        large_row = algo_large.q_tables["pred_1"][js]
        assert np.abs(small_row).max() < np.abs(large_row).max()


class TestJALGTSelectActions:
    def test_returns_all_agents(self):
        env = make_env()
        algo = JALGT(env, base_config())
        obs, _ = env.reset()
        actions = algo.select_actions(obs)
        assert set(actions.keys()) == set(algo.agent_ids)

    def test_actions_in_valid_range(self):
        env = make_env()
        algo = JALGT(env, base_config())
        obs, _ = env.reset()
        for _ in range(10):
            actions = algo.select_actions(obs)
            for a in actions.values():
                assert 0 <= a < 5

    def test_pure_exploration_never_solves_lp(self):
        """epsilon=1.0 means every agent always explores -- select_actions
        should short-circuit before ever building/solving the LP."""
        env = make_env()
        algo = JALGT(env, base_config(epsilon=1.0))
        obs, _ = env.reset()

        def _boom(*a, **k):
            raise AssertionError("LP should not be solved when fully exploring")

        algo._solve_stage_game = _boom
        for _ in range(10):
            algo.select_actions(obs)  # must not raise


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


class TestJALGTTrain:
    def test_q_tables_populated_after_training(self):
        env = make_env()
        algo = JALGT(env, base_config(episodes=5))
        algo.train()
        for table in algo.q_tables.values():
            assert len(table) > 0

    def test_q_values_non_zero_after_training(self):
        env = make_env()
        algo = JALGT(env, base_config(episodes=5))
        algo.train()
        all_vals = [
            v for t in algo.q_tables.values() for arr in t.values() for v in arr
        ]
        assert any(v != 0.0 for v in all_vals)

    def test_agents_learn_their_own_reward_not_a_shared_signal(self):
        """Unlike CQL, predator and prey Q-tables must diverge -- they're
        valued by each agent's own (very different, often opposed) reward,
        not a summed central one."""
        env = make_env()
        algo = JALGT(env, base_config(episodes=15, epsilon=1.0, epsilon_decay=1.0))
        algo.train()
        pred_vals = np.concatenate([arr for arr in algo.q_tables["pred_1"].values()])
        prey_vals = np.concatenate([arr for arr in algo.q_tables["prey_1"].values()])
        assert not np.allclose(pred_vals, prey_vals)

    def test_terminal_cuts_bootstrap_but_truncation_does_not(self):
        """Same convention as CQL/MixedTrainer/IQL (and the #57 lesson from
        A2C): only a true `terminated` zeroes the bootstrap value."""
        env = make_env(n_pred=1, n_prey=1, size=5)
        algo = JALGT(env, base_config(episodes=1, alpha=1.0, gamma=0.5, epsilon=1.0))
        obs, _ = env.reset()
        actions = {aid: 4 for aid in algo.agent_ids}
        js = algo._joint_state(obs)
        ja = algo._joint_action_index(actions)

        step_out = env.step(actions)
        next_obs = step_out["obs"]
        rewards = step_out["reward"]
        js_next = algo._joint_state(next_obs)

        # seed the next state's Q-values so the bootstrap term is non-zero
        # if it were (wrongly) used
        for aid in algo.agent_ids:
            algo.q_tables[aid][js_next][:] = 100.0

        for terminal in (True, False):
            for aid in algo.agent_ids:
                algo.q_tables[aid][js][ja] = 0.0
            r = float(rewards[algo.agent_ids[0]])
            if terminal:
                next_value = 0.0
            else:
                probs = algo._solve_stage_game(js_next)
                next_value = float(
                    np.dot(probs, algo.q_tables[algo.agent_ids[0]][js_next])
                )
            expected = r + algo.gamma * next_value
            algo.q_tables[algo.agent_ids[0]][js][ja] += algo.alpha * (
                expected - algo.q_tables[algo.agent_ids[0]][js][ja]
            )
            if terminal:
                assert algo.q_tables[algo.agent_ids[0]][js][ja] == pytest.approx(r)
            else:
                assert algo.q_tables[algo.agent_ids[0]][js][ja] != pytest.approx(r)

    def test_epsilon_decays_when_decay_set(self):
        env = make_env()
        algo = JALGT(
            env,
            base_config(epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.0, episodes=5),
        )
        algo.train()
        assert algo.epsilon < 1.0


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


class TestJALGTPersistence:
    def test_save_creates_file(self):
        env = make_env()
        algo = JALGT(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_restores_all_agents_q_tables(self):
        env = make_env()
        algo = JALGT(env, base_config(episodes=5))
        algo.train()
        n_states = {aid: len(t) for aid, t in algo.q_tables.items()}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = JALGT.load(make_env(), base_config(), path)
            for aid, n in n_states.items():
                assert len(algo2.q_tables[aid]) == n
        finally:
            os.unlink(path)

    def test_load_restores_epsilon(self):
        """The #66/DQN lesson: persist enough state to safely resume, not
        just the raw Q-values -- epsilon (the exploration schedule) too."""
        env = make_env()
        algo = JALGT(
            env,
            base_config(episodes=5, epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.0),
        )
        algo.train()
        saved_epsilon = algo.epsilon
        assert saved_epsilon < 1.0
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = JALGT.load(make_env(), base_config(epsilon=1.0), path)
            assert algo2.epsilon == pytest.approx(saved_epsilon)
        finally:
            os.unlink(path)

    def test_load_instance_can_evaluate(self):
        env = make_env()
        algo = JALGT(env, base_config(episodes=3))
        algo.train()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            algo.save(path)
            algo2 = JALGT.load(make_env(), base_config(epsilon=0.0), path)
            algo2.evaluate(episodes=1)  # must not raise
        finally:
            os.unlink(path)
