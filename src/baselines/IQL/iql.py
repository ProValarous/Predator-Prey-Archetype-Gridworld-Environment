# src/baselines/IQL/iql.py

import logging
import os
import pickle

import numpy as np
from collections import defaultdict
from numpy.random import default_rng

from baselines.base import BaseAlgorithm
from baselines.registry.algorithm_registry import register

LOGGER = logging.getLogger("iql")


class IQL(BaseAlgorithm):
    """
    Tabular Independent Q-Learning.

    One Q-table per agent. State is encoded as a hashable tuple from the raw
    observation dict — no assumption about observation structure, fully
    wrapper-compatible.
    """

    def __init__(self, env, config):
        super().__init__(env, config)

        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.episodes = config.get("episodes", 500)
        self.epsilon_decay = config.get("epsilon_decay", 1.0)
        self.min_epsilon = config.get("min_epsilon", 0.01)
        self.action_dim = config.get("action_dim", 5)

        # seeded RNG for reproducible exploration
        self.rng = default_rng(config.get("seed", None))

        # discover agent IDs from an initial reset
        initial_obs, _ = self.env.reset()
        self.agent_ids = list(initial_obs.keys())

        # per-agent Q-tables backed by defaultdict (supports arbitrary states)
        self.q_tables = {
            agent_id: defaultdict(lambda: np.zeros(self.action_dim))
            for agent_id in self.agent_ids
        }

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _encode_state(self, obs_dict) -> tuple:
        """Convert observation dict into a hashable tuple (wrapper-safe)."""

        def _convert(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, _convert(v)) for k, v in obj.items()))
            if isinstance(obj, (list, tuple)):
                return tuple(_convert(v) for v in obj)
            if hasattr(obj, "tolist"):
                val = obj.tolist()
                # numpy array → list; numpy scalar → Python int/float
                if isinstance(val, (list, tuple)):
                    return tuple(_convert(v) for v in val)
                return val
            return obj

        return _convert(obs_dict)

    # ------------------------------------------------------------------
    # Action selection (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def select_actions(self, observations: dict) -> dict:
        actions = {}
        for agent_id, obs in observations.items():
            state = self._encode_state(obs)
            if self.rng.random() < self.epsilon:
                action = int(self.rng.integers(self.action_dim))
            else:
                action = int(np.argmax(self.q_tables[agent_id][state]))
            actions[agent_id] = action
        return actions

    # ------------------------------------------------------------------
    # Training loop (BaseAlgorithm interface)
    # ------------------------------------------------------------------

    def train(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = {agent_id: 0.0 for agent_id in self.agent_ids}

            while not done:
                actions = self.select_actions(obs)

                step_out = self.env.step(actions)
                next_obs = step_out["obs"]
                rewards = step_out["reward"]
                done = step_out["terminated"] or step_out["truncated"]

                for agent_id in self.agent_ids:
                    s = self._encode_state(obs[agent_id])
                    a = actions[agent_id]
                    r = float(rewards[agent_id])
                    ep_reward[agent_id] += r
                    s_next = self._encode_state(next_obs[agent_id])

                    q_current = self.q_tables[agent_id][s][a]
                    q_next_max = (
                        0.0 if done
                        else float(np.max(self.q_tables[agent_id][s_next]))
                    )
                    td_error = r + self.gamma * q_next_max - q_current
                    self.q_tables[agent_id][s][a] += self.alpha * td_error

                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 100 == 0:
                reward_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in ep_reward.items()
                )
                LOGGER.info(
                    "Episode %d/%d | eps=%.3f | %s",
                    ep + 1, self.episodes, self.epsilon, reward_str,
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize Q-tables to a pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {agent_id: dict(table) for agent_id, table in self.q_tables.items()}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        LOGGER.info("Saved Q-tables -> %s", path)

    @classmethod
    def load(cls, env, config: dict, path: str) -> "IQL":
        """Load Q-tables from a pickle file; returns a ready-to-evaluate instance."""
        instance = cls(env, config)
        with open(path, "rb") as fh:
            payload: dict = pickle.load(fh)
        for agent_id, table in payload.items():
            instance.q_tables[agent_id].update(table)
        LOGGER.info(
            "Loaded Q-tables from %s (agents: %s)", path, list(payload.keys())
        )
        return instance


if __name__ != "__main__":
    register("iql", IQL)


# ------------------------------------------------------------------
# CLI  —  python -m baselines.IQL.iql [--mode train|eval] [options]
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

    p = argparse.ArgumentParser("Tabular IQL — train or evaluate")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--predators", type=int, default=1)
    p.add_argument("--preys", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--min-epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="trained_iql.pkl")
    p.add_argument("--load-path", type=str, default=None,
                   help="pkl file to load (required for --mode eval)")
    p.add_argument("--render", action="store_true",
                   help="Open pygame window during eval (requires a display)")
    args = p.parse_args()

    agents = []
    for i in range(1, args.preys + 1):
        agents.append(Agent(agent_name=f"prey_{i}", agent_team=i, agent_type="prey"))
    for i in range(1, args.predators + 1):
        agents.append(Agent(agent_name=f"predator_{i}", agent_team=i, agent_type="predator"))

    render = "human" if (args.mode == "eval" and args.render) else None
    env = GridWorldEnv(agents=agents, render_mode=render,
                       size=args.size, perc_num_obstacle=10, seed=args.seed)

    config = {
        "alpha": args.alpha, "gamma": args.gamma,
        "epsilon": args.epsilon if args.mode == "train" else 0.0,
        "epsilon_decay": args.epsilon_decay, "min_epsilon": args.min_epsilon,
        "episodes": args.episodes, "seed": args.seed,
    }

    if args.mode == "train":
        algo = IQL(env, config)
        algo.train()
        algo.save(args.save_path)
        LOGGER.info("Saved -> %s", args.save_path)
    else:
        if not args.load_path:
            raise SystemExit("--load-path is required for --mode eval")
        algo = IQL.load(env, config, args.load_path)
        algo.evaluate(episodes=args.episodes)

    env.close()
