"""
Survival reward for prey agents.

Rewards prey for staying alive over time.
"""

from multi_agent_package.rewards.base import RewardFunction


class SurvivalReward(RewardFunction):
    """
    Parameters:
    - weight: float
    """

    def compute(self, env):
        rewards = {}

        for ag in env.agents:
            if ag.agent_type.startswith("prey"):
                rewards[ag.agent_name] = self.weight
            else:
                rewards[ag.agent_name] = 0.0

        return rewards
