# src/baselines/base.py

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class BaseAlgorithm(ABC):
    """
    Stable learning interface.
    Algorithms treat env as a black box.
    """

    def __init__(self, env, config: dict):
        self.env = env
        self.config = config

    @abstractmethod
    def select_actions(self, observations: dict) -> dict:
        """
        observations: {agent_name: obs_dict}
        returns: {agent_name: action_int}
        """

    @abstractmethod
    def train(self):
        pass

    def evaluate(self, episodes: int = 5, max_steps: int = 500) -> dict:
        """Run greedy-ish evaluation episodes and return summary metrics.

        Returns a dict with the number of episodes, mean episode length, and
        mean per-agent return. Callers (the run_* scripts) print this; the
        method no longer discards its results.
        """
        episode_lengths = []
        agent_returns = defaultdict(list)

        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            ep_reward = defaultdict(float)
            while not done and steps < max_steps:
                actions = self.select_actions(obs)
                step_out = self.env.step(actions)
                obs = step_out["obs"]
                for agent_id, r in step_out["reward"].items():
                    ep_reward[agent_id] += float(r)
                done = step_out["terminated"] or step_out["truncated"]
                steps += 1
            episode_lengths.append(steps)
            for agent_id, total in ep_reward.items():
                agent_returns[agent_id].append(total)

        summary = {
            "episodes": episodes,
            "mean_episode_length": (
                float(np.mean(episode_lengths)) if episode_lengths else 0.0
            ),
        }
        for agent_id, returns in agent_returns.items():
            summary[f"mean_return_{agent_id}"] = float(np.mean(returns))
        return summary
