# src/baselines/CQL/cql.py

"""
Tabular Centralized Q-Learning (CQL).

All agents share a single Q-table indexed by the joint state (tuple of all
agents' encoded observations) and the joint action (single integer encoding
all agents' individual actions).

Action selection marginalises the joint Q-values over other agents' actions
to obtain per-agent Q-values, then acts epsilon-greedily on those.

The TD target uses the sum of all agents' rewards as the central learning
signal — the table tries to maximise collective return.

State encoding: recursive hashable tuple, identical to IQL (wrapper-safe).
Q-table backend: defaultdict so only visited joint states are stored.

Usage (config-driven):
    experiment.yaml → algorithm: cql

Usage (standalone CLI):
    cd src
    python -m baselines.CQL.cql --episodes 1000 --size 6 --predators 1 --preys 1
"""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("cql")


class CQL(BaseAlgorithm):
    """
    Tabular Centralized Q-Learning.

    Single shared Q-table: joint_state → Q-vector of length action_dim^n_agents.
    Joint state  : tuple of every agent's encoded observation (ordered by agent_ids).
    Joint action : integer encoding all agents' actions in base action_dim.
    Central reward: sum of all agents' rewards for each transition.
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

        # shape for reshaping the Q-vector into a per-agent tensor
        self._action_shape = (self.action_dim,) * self.n_agents

        # single shared Q-table: joint_state (tuple) → np.zeros(n_joint_actions)
        self.q_table: defaultdict = defaultdict(
            lambda: np.zeros(self.n_joint_actions, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # State encoding (wrapper-compatible, same approach as IQL)
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
        """Convert per-agent action dict → single joint-action integer."""
        idx = 0
        for aid in self.agent_ids:
            idx = idx * self.action_dim + int(actions[aid])
        return idx

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        joint_s = self._joint_state(observations)
        q_vals = self.q_table[joint_s]  # (n_joint_actions,)
        q_tensor = q_vals.reshape(self._action_shape)  # (a_dim, a_dim, ...)

        actions = {}
        for i, aid in enumerate(self.agent_ids):
            if self.rng.random() < self.epsilon:
                actions[aid] = int(self.rng.integers(self.action_dim))
            else:
                # marginalise over all other agents' action axes
                axes = tuple(j for j in range(self.n_agents) if j != i)
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

                joint_s = self._joint_state(obs)
                joint_s_next = self._joint_state(next_obs)
                joint_a = self._joint_action_index(actions)

                # central reward = sum of all agents' rewards
                central_r = 0.0
                for aid in self.agent_ids:
                    r = float(rewards[aid])
                    ep_reward[aid] += r
                    central_r += r

                # centralized TD update on the shared Q-table
                q_current = self.q_table[joint_s][joint_a]
                q_next_max = 0.0 if done else float(np.max(self.q_table[joint_s_next]))
                td_error = central_r + self.gamma * q_next_max - q_current
                self.q_table[joint_s][joint_a] += self.alpha * td_error

                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 100 == 0:
                reward_str = ", ".join(f"{k}={v:.2f}" for k, v in ep_reward.items())
                LOGGER.info(
                    "Episode %d/%d | eps=%.3f | joint_states=%d | %s",
                    ep + 1,
                    self.episodes,
                    self.epsilon,
                    len(self.q_table),
                    reward_str,
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "q_table": dict(self.q_table),
            "agent_ids": self.agent_ids,
            "n_joint_actions": self.n_joint_actions,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved -> %s  (%d joint states)", path, len(self.q_table))

    @classmethod
    def load(cls, env, config: dict, path: str) -> "CQL":
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)
        instance.q_table.update(payload["q_table"])
        LOGGER.info("Loaded from %s  (%d joint states)", path, len(instance.q_table))
        return instance


if __name__ != "__main__":
    register("cql", CQL)


# ------------------------------------------------------------------
# CLI  —  python -m baselines.CQL.cql [--mode train|eval] [options]
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

    p = argparse.ArgumentParser("Tabular CQL (Centralized) — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument(
        "--size",
        type=int,
        default=6,
        help="Grid size. Keep small — joint state space is size^(2*n_agents).",
    )
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--min-epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_cql.pkl")
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
        algo = CQL(env, config)
        algo.train()
        algo.save(args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = CQL.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
