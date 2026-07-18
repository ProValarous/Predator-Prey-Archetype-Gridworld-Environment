# src/baselines/MIXED/mix_train.py

"""
MixedTrainer — assign CQL or IQL independently to predators and prey.

- IQL team : each agent in the team keeps its own independent Q-table.
- CQL team : all agents in the team share one centralized Q-table keyed on
             the joint team state and joint team action. Updates use the sum
             of the team's rewards as the central learning signal.

This means predators can learn centrally (coordinating via a shared table)
while prey learn independently, or vice-versa — any combination is valid.

Usage (config-driven via run_from_config):
    experiment.yaml → algorithm: mixed
    params: {predator_algo: cql, prey_algo: iql, ...}

Usage (standalone CLI):
    cd src
    python -m baselines.MIXED.mix_train \
        --predator-algo cql --prey-algo iql --episodes 1000
"""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("mixed")

_PREDATOR_TEAM = "predator"
_PREY_TEAM = "prey"


class MixedTrainer(BaseAlgorithm):
    """
    Per-team algorithm assignment: predators and prey can use different algorithms.

    predator_algo / prey_algo: "iql" or "cql"
    - iql : independent — one Q-table per agent in the team
    - cql : centralized — one shared Q-table over joint team state-action space;
            action selection marginalises the joint Q over other team members
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
        self.predator_algo = config.get("predator_algo", "cql").lower()
        self.prey_algo = config.get("prey_algo", "iql").lower()

        self.rng = default_rng(config.get("seed", None))

        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())

        # Partition agents into teams
        self._predators = [
            ag.agent_name
            for ag in self.env.agents
            if ag.agent_type.startswith(_PREDATOR_TEAM)
        ]
        self._prey = [
            ag.agent_name
            for ag in self.env.agents
            if not ag.agent_type.startswith(_PREDATOR_TEAM)
        ]

        # Build team lookup: agent_name → team_key
        self._team_of: dict[str, str] = {}
        for aid in self._predators:
            self._team_of[aid] = _PREDATOR_TEAM
        for aid in self._prey:
            self._team_of[aid] = _PREY_TEAM

        self._algo_of: dict[str, str] = {
            _PREDATOR_TEAM: self.predator_algo,
            _PREY_TEAM: self.prey_algo,
        }

        # ------------------------------------------------------------------
        # Q-table structures
        # ------------------------------------------------------------------

        # IQL tables: one per agent
        self._iql_tables: dict[str, defaultdict] = {}

        # CQL tables: one per team
        self._cql_tables: dict[str, defaultdict] = {}
        self._cql_team_ids: dict[str, list[str]] = {}  # team → ordered agent list
        self._cql_n_joint: dict[str, int] = {}  # team → n_joint_actions
        self._cql_action_shape: dict[str, tuple] = {}  # team → reshape tuple

        for team_key, members, algo in [
            (_PREDATOR_TEAM, self._predators, self.predator_algo),
            (_PREY_TEAM, self._prey, self.prey_algo),
        ]:
            if algo == "iql":
                for aid in members:
                    self._iql_tables[aid] = defaultdict(
                        lambda: np.zeros(self.action_dim)
                    )
            else:  # cql
                n_joint = self.action_dim ** len(members)
                self._cql_team_ids[team_key] = list(members)
                self._cql_n_joint[team_key] = n_joint
                self._cql_action_shape[team_key] = (self.action_dim,) * len(members)
                self._cql_tables[team_key] = defaultdict(
                    lambda nj=n_joint: np.zeros(nj, dtype=np.float32)
                )

    # ------------------------------------------------------------------
    # State encoding (wrapper-safe, identical to IQL)
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

    def _team_joint_state(self, observations: dict, team_key: str) -> tuple:
        return tuple(
            self._encode_state(observations[aid])
            for aid in self._cql_team_ids[team_key]
        )

    def _team_joint_action_index(self, actions: dict, team_key: str) -> int:
        idx = 0
        for aid in self._cql_team_ids[team_key]:
            idx = idx * self.action_dim + int(actions[aid])
        return idx

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        actions = {}

        for team_key, members, algo in [
            (_PREDATOR_TEAM, self._predators, self.predator_algo),
            (_PREY_TEAM, self._prey, self.prey_algo),
        ]:
            if algo == "iql":
                for aid in members:
                    s = self._encode_state(observations[aid])
                    if self.rng.random() < self.epsilon:
                        actions[aid] = int(self.rng.integers(self.action_dim))
                    else:
                        actions[aid] = int(np.argmax(self._iql_tables[aid][s]))
            else:  # cql — marginalise joint Q
                joint_s = self._team_joint_state(observations, team_key)
                q_vals = self._cql_tables[team_key][joint_s]
                q_tensor = q_vals.reshape(self._cql_action_shape[team_key])
                team_members = self._cql_team_ids[team_key]
                n = len(team_members)

                for i, aid in enumerate(team_members):
                    if self.rng.random() < self.epsilon:
                        actions[aid] = int(self.rng.integers(self.action_dim))
                    else:
                        axes = tuple(j for j in range(n) if j != i)
                        q_marginal = q_tensor.mean(axis=axes) if axes else q_vals
                        actions[aid] = int(np.argmax(q_marginal))

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
                # truncation does not.
                terminal = bool(step_out["terminated"])

                for aid in self.agent_ids:
                    ep_reward[aid] += float(rewards[aid])

                # IQL updates — per agent
                for aid, table in self._iql_tables.items():
                    s = self._encode_state(obs[aid])
                    s_next = self._encode_state(next_obs[aid])
                    a = actions[aid]
                    r = float(rewards[aid])
                    q_current = table[s][a]
                    q_next_max = 0.0 if terminal else float(np.max(table[s_next]))
                    td_error = r + self.gamma * q_next_max - q_current
                    table[s][a] += self.alpha * td_error

                # CQL updates — per team (centralized)
                for team_key, table in self._cql_tables.items():
                    joint_s = self._team_joint_state(obs, team_key)
                    joint_s_next = self._team_joint_state(next_obs, team_key)
                    joint_a = self._team_joint_action_index(actions, team_key)

                    # central reward = sum of team members' rewards
                    central_r = sum(
                        float(rewards[aid]) for aid in self._cql_team_ids[team_key]
                    )

                    q_current = table[joint_s][joint_a]
                    q_next_max = 0.0 if terminal else float(np.max(table[joint_s_next]))
                    td_error = central_r + self.gamma * q_next_max - q_current
                    table[joint_s][joint_a] += self.alpha * td_error

                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 100 == 0:
                reward_str = ", ".join(f"{k}={v:.2f}" for k, v in ep_reward.items())
                LOGGER.info(
                    "Episode %d/%d | eps=%.3f | %s",
                    ep + 1,
                    self.episodes,
                    self.epsilon,
                    reward_str,
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "iql_tables": {aid: dict(t) for aid, t in self._iql_tables.items()},
            "cql_tables": {tk: dict(t) for tk, t in self._cql_tables.items()},
            "cql_team_ids": self._cql_team_ids,
            "algo_of": self._algo_of,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved -> %s", path)

    @classmethod
    def load(cls, env, config: dict, path: str) -> "MixedTrainer":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)
        for aid, table in payload["iql_tables"].items():
            if aid in instance._iql_tables:
                instance._iql_tables[aid].update(table)
        for tk, table in payload["cql_tables"].items():
            if tk in instance._cql_tables:
                instance._cql_tables[tk].update(table)
        LOGGER.info("Loaded from %s", path)
        return instance


if __name__ != "__main__":
    register("mixed", MixedTrainer)


# ------------------------------------------------------------------
# CLI  —  python -m baselines.MIXED.mix_train [--mode train|eval] [options]
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

    p = argparse.ArgumentParser("MixedTrainer — per-team CQL (centralized) / IQL")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--predator-algo", type=str, default="cql", choices=["iql", "cql"])
    p.add_argument("--prey-algo", type=str, default="iql", choices=["iql", "cql"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--min-epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_mixed.pkl")
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
        "predator_algo": args.predator_algo,
        "prey_algo": args.prey_algo,
        "episodes": args.episodes,
        "seed": args.seed,
    }

    if args.mode == "train":
        algo = MixedTrainer(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = MixedTrainer.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
