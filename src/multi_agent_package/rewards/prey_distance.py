"""
Prey distance shaping reward.

Encourages prey to move away from the nearest predator. Deliberately NOT a
negation of PredatorDistanceReward -- use a different weight than
predator_distance's, or their shaping terms sum to exactly zero on every
step and the correlated-equilibrium LP loses its welfare signal (see
docs/algorithms/jal-gt.md).
"""

from multi_agent_package.rewards.base import RewardFunction


class PreyDistanceReward(RewardFunction):
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

        if not predators:
            return rewards

        for prey in preys:
            px, py = prey._agent_location
            dists = [
                abs(px - pred._agent_location[0]) + abs(py - pred._agent_location[1])
                for pred in predators
            ]
            nearest = min(dists)
            rewards[prey.agent_name] = self.weight * nearest

        return rewards
