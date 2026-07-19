"""
Wrapper for GridWorldEnv.base_reward().

This allows base rewards to be toggled via config.
"""

from multi_agent_package.rewards.base import RewardFunction


class BaseReward(RewardFunction):
    """
    Uses the environment's canonical base reward.
    """

    def compute(self, env):
        base = env.base_reward()
        return {k: self.weight * v for k, v in base.items()}
