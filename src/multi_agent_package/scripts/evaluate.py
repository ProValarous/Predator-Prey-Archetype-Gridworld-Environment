"""
Evaluation script.

Collects basic episode-level statistics.
"""

import numpy as np
from collections import defaultdict

from multi_agent_package.scripts.run_from_config import run


class Metrics:
    def __init__(self):
        self.episode_lengths = []
        self.capture_counts = []

    def log_episode(self, env):
        self.episode_lengths.append(env._episode_steps)
        self.capture_counts.append(env._captures_total)

    def summary(self):
        return {
            "mean_episode_length": float(np.mean(self.episode_lengths)),
            "mean_captures": float(np.mean(self.capture_counts)),
        }


if __name__ == "__main__":
    metrics = Metrics()
    run("configs")
    print(metrics.summary())
