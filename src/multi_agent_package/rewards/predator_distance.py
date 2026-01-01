"""
Predator distance shaping reward.

Encourages predators to move closer to the nearest prey.
"""

from multi_agent_package.rewards.base import RewardFunction


class PredatorDistanceReward(RewardFunction):
    """
    Parameters:
    - weight: float
    """

    def compute(self, env):
        rewards = {}

        predators = [a for a in env.agents if a.agent_type.startswith("predator")]
        preys = [a for a in env.agents if a.agent_type.startswith("prey")]

        for ag in env.agents:
            rewards[ag.agent_name] = 0.0

        if not preys:
            return rewards

        for pred in predators:
            px, py = pred._agent_location
            dists = [
                abs(px - prey._agent_location[0]) + abs(py - prey._agent_location[1])
                for prey in preys
            ]
            nearest = min(dists)
            rewards[pred.agent_name] = -self.weight * nearest

        return rewards
