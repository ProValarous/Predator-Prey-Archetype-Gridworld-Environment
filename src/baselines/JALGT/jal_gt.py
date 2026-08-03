# src/baselines/JALGT/jal_gt.py

"""
Joint-Action Learning with Game Theory (JAL-GT) — Correlated Q-learning.

Implements Algorithm 7 from Albrecht, Christianos & Tuyls (2024), *Multi-Agent
Reinforcement Learning: Foundations and Modern Approaches*, Section 6.2, using
correlated equilibrium (Greenwald & Hall 2003, Section 6.2.3) as the solution
concept. See docs/algorithms/jal-gt.md for the full derivation and why
correlated equilibrium was chosen over minimax (this environment is not
zero-sum, per GridWorldEnv.base_reward()) and Nash (its convergence conditions
are essentially never satisfiable in practice, and even epsilon-Nash is
PPAD-complete).

Each agent keeps its OWN joint-action value table Q_i(s, a), valued by its own
reward (not a shared centralized signal like CQL's). At each visited state,
the set of tables induces a normal-form "stage game" Gamma_s -- Gamma_s,i(a) =
Q_i(s, a) -- which is solved via the correlated-equilibrium linear program
(Eq. 4.20-4.23) to get a joint policy for that state. Action selection and the
temporal-difference bootstrap target both come from solving this LP.

One deliberate simplification vs. the book's literal Algorithm 7: the pseudocode
is written from a single controlled agent's perspective, which (if every agent
ran its own independent copy) would mean every agent redundantly maintains a
synchronized copy of every OTHER agent's Q_j too. Structured the same way this
repo's other joint-action learners are (see MixedTrainer, CQL) -- one trainer
centrally owning every agent's table directly -- there is only ever one ground
truth per Q_i, so that redundancy is simply unnecessary, not something being
approximated away.

Another deliberate deviation: Algorithm 7's pseudocode doesn't distinguish a
true terminal state from an episode-ending timeout (it assumes absorbing
states with 0 reward thereafter). This implementation follows this repo's
established, more correct convention instead (see CQL, MixedTrainer, IQL):
only a true `terminated` cuts the bootstrap value to 0; a `truncated` timeout
does not.

Usage (config-driven):
    experiment.yaml -> algorithm: jalgt

Usage (standalone CLI):
    cd src
    python -m baselines.JALGT.jal_gt --episodes 1000 --size 6 --predators 1 --preys 1
"""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from numpy.random import default_rng
from scipy.optimize import linprog

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("jalgt")


class JALGT(BaseAlgorithm):
    """
    Joint-Action Learning with Game Theory, instantiated with correlated
    equilibrium (correlated Q-learning).

    Per-agent Q-tables : joint_state -> Q-vector of length action_dim^n_agents,
                          one table per agent, each valued by that agent's own
                          reward.
    Joint state  : tuple of every agent's encoded observation (ordered by
                   agent_ids) -- identical encoding to CQL/IQL/MixedTrainer.
    Joint action : integer encoding all agents' actions in base action_dim,
                   agent_ids[0] most significant -- identical encoding to CQL.
    Stage game   : at any visited joint state s, Gamma_s,i(a) = Q_i(s, a) for
                   every agent i. Solved via the correlated-equilibrium LP
                   (Eq. 4.20-4.23) with a social-welfare objective (maximize
                   the sum of agents' expected rewards) to pick among the
                   possibly many equilibria -- the same selection rule the
                   book uses in its own worked example.
    """

    def __init__(self, env, config: dict):
        super().__init__(env, config)

        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.min_epsilon = config.get("min_epsilon", 0.05)
        self.action_dim = config.get("action_dim", 5)
        self.episodes = config.get("episodes", 1000)

        self.rng = default_rng(config.get("seed", None))

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())
        self.n_agents = len(self.agent_ids)
        self.n_joint_actions = self.action_dim**self.n_agents

        # One joint-action Q-table PER AGENT (not a single shared table like
        # CQL) -- each valued by that agent's own reward, since the whole
        # point of solving an equilibrium is that agents' payoffs can conflict.
        self.q_tables: dict = {
            aid: defaultdict(
                lambda n=self.n_joint_actions: np.zeros(n, dtype=np.float64)
            )
            for aid in self.agent_ids
        }

        # Precomputed structure for the correlated-equilibrium LP's deviation
        # constraints (Eq. 4.21) -- which joint actions belong to each
        # (agent position, recommended action) group, and where each such
        # joint action lands if that agent deviates to each alternate action.
        # This depends only on n_agents/action_dim, never on Q-values, so it
        # is built ONCE here rather than rebuilt on every LP solve -- solving
        # the stage game happens twice per environment step during training,
        # and rebuilding this from scratch every time was the dominant cost
        # in an unvectorized first version (see docs/algorithms/jal-gt.md's
        # Scalability section for the measured ~45x-slower-than-CQL finding
        # this closes).
        self._all_joint_actions = np.arange(self.n_joint_actions)
        self._pos_components = [
            self._component(self._all_joint_actions, pos)
            for pos in range(self.n_agents)
        ]
        self._pos_masks = [
            [self._pos_components[pos] == a_prime for a_prime in range(self.action_dim)]
            for pos in range(self.n_agents)
        ]
        self._deviated_indices = [
            [
                self._deviate(self._all_joint_actions, pos, a)
                for a in range(self.action_dim)
            ]
            for pos in range(self.n_agents)
        ]

    # ------------------------------------------------------------------
    # State encoding (wrapper-compatible, same approach as CQL/IQL)
    # ------------------------------------------------------------------

    def _encode_state(self, obs_dict) -> tuple:
        def _convert(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, _convert(v)) for k, v in obj.items()))
            if isinstance(obj, (list, tuple)):
                return tuple(_convert(v) for v in obj)
            if hasattr(obj, "tolist"):
                val = obj.tolist()
                if isinstance(val, (list, tuple)):
                    return tuple(_convert(v) for v in val)
                return val
            return obj

        return _convert(obs_dict)

    def _joint_state(self, observations: dict) -> tuple:
        """Encode all agents' observations into a single hashable joint state."""
        return tuple(self._encode_state(observations[aid]) for aid in self.agent_ids)

    # ------------------------------------------------------------------
    # Joint action encoding / decoding
    # ------------------------------------------------------------------

    def _joint_action_index(self, actions: dict) -> int:
        """Convert per-agent action dict -> single joint-action integer
        (agent_ids[0] is the most significant base-action_dim digit)."""
        idx = 0
        for aid in self.agent_ids:
            idx = idx * self.action_dim + int(actions[aid])
        return idx

    def _decode_joint_action(self, idx: int) -> dict:
        """Inverse of _joint_action_index: joint-action integer -> per-agent
        action dict."""
        digits = []
        remaining = idx
        for _ in range(self.n_agents):
            remaining, digit = divmod(remaining, self.action_dim)
            digits.append(digit)
        digits.reverse()
        return {aid: digits[i] for i, aid in enumerate(self.agent_ids)}

    def _component(self, idx, pos: int):
        """The action of the agent at position `pos` (0 = agent_ids[0],
        the most significant digit) within joint-action index `idx`. `idx`
        may be a scalar int or a numpy array -- pure arithmetic (//, %), so
        it vectorizes over an array of indices with no code change, which is
        exactly how __init__ precomputes _pos_components/_pos_masks/
        _deviated_indices for every joint action at once."""
        power = self.action_dim ** (self.n_agents - 1 - pos)
        return (idx // power) % self.action_dim

    def _deviate(self, idx, pos: int, new_action: int):
        """Joint-action index identical to `idx` except the agent at
        position `pos` plays `new_action` instead. O(1) per element:
        exploits the place-value structure of _joint_action_index directly
        rather than decoding and re-encoding the whole tuple. `idx` may be a
        scalar int or a numpy array, same as _component."""
        old_action = self._component(idx, pos)
        power = self.action_dim ** (self.n_agents - 1 - pos)
        return idx + (new_action - old_action) * power

    # ------------------------------------------------------------------
    # Stage game: build + solve the correlated-equilibrium LP
    # ------------------------------------------------------------------

    def _build_correlated_equilibrium_lp(self, q_values: dict) -> tuple:
        """Builds the correlated-equilibrium LP (Eq. 4.20-4.23) for the stage
        game induced by q_values = {agent_id: Q-vector of length
        n_joint_actions}. Returns (c, A_ub, b_ub, A_eq, b_eq) in the form
        scipy.optimize.linprog expects (minimize c.x s.t. A_ub x <= b_ub,
        A_eq x = b_eq, x >= 0).

        Objective: maximize sum_a sum_i x_a * Q_i(a) (social welfare, Eq. 4.20)
        == minimize -sum_a x_a * (sum_i Q_i(a)).

        Constraints (Eq. 4.21): for every agent i and every pair of actions
        (a', a'') with a' != a'', no agent gains by unilaterally deviating
        from a recommended action a' to a'':
            sum_{a: a_i=a'} x_a Q_i(a) >= sum_{a: a_i=a'} x_a Q_i(a'', a_{-i})
        Rearranged into linprog's <= 0 form. Rows where a'=a'' are omitted --
        they reduce to the trivial 0 >= 0 and don't affect feasibility or the
        optimum, only LP size.

        Vectorized using the deviation structure precomputed once in
        __init__ (_pos_masks, _deviated_indices): that structure depends
        only on n_agents/action_dim, never on q_values, so each row here is
        one numpy boolean-mask assignment over all joint actions at once
        rather than a Python-level loop over every individual joint action.
        """
        n = self.n_joint_actions
        total_reward = np.zeros(n, dtype=np.float64)
        for aid in self.agent_ids:
            total_reward += q_values[aid]
        c = -total_reward

        rows = []
        for pos, aid in enumerate(self.agent_ids):
            qi = q_values[aid]
            for a_prime in range(self.action_dim):
                mask = self._pos_masks[pos][a_prime]
                if not mask.any():
                    continue
                for a_double_prime in range(self.action_dim):
                    if a_prime == a_double_prime:
                        continue
                    deviated = self._deviated_indices[pos][a_double_prime]
                    row = np.zeros(n, dtype=np.float64)
                    row[mask] = qi[deviated[mask]] - qi[self._all_joint_actions[mask]]
                    rows.append(row)

        A_ub = np.array(rows, dtype=np.float64)
        b_ub = np.zeros(len(rows), dtype=np.float64)
        A_eq = np.ones((1, n), dtype=np.float64)
        b_eq = np.array([1.0])
        return c, A_ub, b_ub, A_eq, b_eq

    def _solve_stage_game(self, joint_state: tuple) -> np.ndarray:
        """Solves Gamma_{joint_state} via the correlated-equilibrium LP and
        returns a probability vector of length n_joint_actions."""
        q_values = {aid: self.q_tables[aid][joint_state] for aid in self.agent_ids}
        c, A_ub, b_ub, A_eq, b_eq = self._build_correlated_equilibrium_lp(q_values)
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
        )
        if not res.success:
            raise RuntimeError(
                "Correlated equilibrium LP failed to solve (should always be "
                f"feasible -- Aumann 1974): {res.message}"
            )

        # An LP solver's output can carry tiny floating-point noise (a
        # near-zero negative entry, a sum off from 1 by ~1e-12) -- harmless
        # to the equilibrium itself, but rng.choice(p=...) requires
        # probabilities that are non-negative and sum to (very nearly) 1.
        probs = np.clip(res.x, 0.0, None)
        total = probs.sum()
        if total <= 0:
            probs = np.full(self.n_joint_actions, 1.0 / self.n_joint_actions)
        else:
            probs = probs / total
        return probs

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        joint_s = self._joint_state(observations)

        # Each agent independently rolls its own epsilon check (same
        # convention as CQL's select_actions), overriding its own component
        # of the joint sample with an independent random action when it
        # explores. If every agent is exploring this step, skip solving the
        # LP entirely -- a real optimization early in training when epsilon
        # is near 1.0 and most steps don't need the equilibrium at all.
        explore = {aid: self.rng.random() < self.epsilon for aid in self.agent_ids}
        if all(explore.values()):
            return {
                aid: int(self.rng.integers(self.action_dim)) for aid in self.agent_ids
            }

        probs = self._solve_stage_game(joint_s)
        joint_idx = int(self.rng.choice(self.n_joint_actions, p=probs))
        joint_action = self._decode_joint_action(joint_idx)

        actions = {}
        for aid in self.agent_ids:
            if explore[aid]:
                actions[aid] = int(self.rng.integers(self.action_dim))
            else:
                actions[aid] = joint_action[aid]
        return actions

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = {aid: 0.0 for aid in self.agent_ids}

            while not done:
                actions = self.select_actions(obs)
                step_out = self.env.step(actions)
                next_obs = step_out["obs"]
                rewards = step_out["reward"]
                done = step_out["terminated"] or step_out["truncated"]
                # Only a true terminal state cuts the bootstrap; a timeout
                # truncation does not (see module docstring).
                terminal = bool(step_out["terminated"])

                joint_s = self._joint_state(obs)
                joint_s_next = self._joint_state(next_obs)
                joint_a = self._joint_action_index(actions)

                if terminal:
                    next_values = {aid: 0.0 for aid in self.agent_ids}
                else:
                    # ONE solve of Gamma_{s'} is reused for every agent's
                    # bootstrap target -- Value_i and Value_j differ only in
                    # which agent's Q-vector the SAME equilibrium is dotted
                    # against (Eq. 6.11), not in the equilibrium itself.
                    next_probs = self._solve_stage_game(joint_s_next)
                    next_values = {
                        aid: float(np.dot(next_probs, self.q_tables[aid][joint_s_next]))
                        for aid in self.agent_ids
                    }

                for aid in self.agent_ids:
                    r = float(rewards[aid])
                    ep_reward[aid] += r
                    q_current = self.q_tables[aid][joint_s][joint_a]
                    td_error = r + self.gamma * next_values[aid] - q_current
                    self.q_tables[aid][joint_s][joint_a] += self.alpha * td_error

                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 100 == 0:
                reward_str = ", ".join(f"{k}={v:.2f}" for k, v in ep_reward.items())
                LOGGER.info(
                    "Episode %d/%d | eps=%.3f | joint_states=%d | %s",
                    ep + 1,
                    self.episodes,
                    self.epsilon,
                    len(next(iter(self.q_tables.values()))),
                    reward_str,
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "q_tables": {aid: dict(t) for aid, t in self.q_tables.items()},
            "agent_ids": self.agent_ids,
            "n_joint_actions": self.n_joint_actions,
            # Persisted so a resumed run keeps its exploration schedule
            # rather than silently restarting it from epsilon=1.0.
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info(
            "Saved -> %s (%d joint states/agent)",
            path,
            len(next(iter(self.q_tables.values()))),
        )

    @classmethod
    def load(cls, env, config: dict, path: str) -> "JALGT":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)
        for aid, table in payload["q_tables"].items():
            if aid in instance.q_tables:
                instance.q_tables[aid].update(table)
        instance.epsilon = payload.get("epsilon", instance.epsilon)
        LOGGER.info("Loaded from %s", path)
        return instance


if __name__ != "__main__":
    register("jalgt", JALGT)


# ------------------------------------------------------------------
# CLI  —  python -m baselines.JALGT.jal_gt [--mode train|eval] [options]
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging

    from multi_agent_package.core.gridworld import GridWorldEnv
    from multi_agent_package.core.agent import Agent

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    p = argparse.ArgumentParser("JAL-GT (Correlated Q-learning) — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument(
        "--size",
        type=int,
        default=6,
        help="Grid size. Keep small — the correlated-equilibrium LP has "
        "action_dim^n_agents variables.",
    )
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--min-epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_jalgt.pkl")
    p.add_argument("--load-path", type=str, default=None)
    p.add_argument(
        "--render",
        action="store_true",
        help="Open pygame window during eval (requires a display)",
    )
    args = p.parse_args()

    agents = []
    for i in range(1, args.preys + 1):
        agents.append(Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey"))
    for i in range(1, args.predators + 1):
        agents.append(
            Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator")
        )

    render = "human" if (args.mode == "eval" and args.render) else None
    env = GridWorldEnv(
        agents=agents,
        render_mode=render,
        size=args.size,
        perc_num_obstacle=10,
        seed=args.seed,
    )

    config = {
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon if args.mode == "train" else 0.0,
        "epsilon_decay": args.epsilon_decay,
        "min_epsilon": args.min_epsilon,
        "episodes": args.episodes,
        "seed": args.seed,
    }

    if args.mode == "train":
        algo = JALGT(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = JALGT.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
