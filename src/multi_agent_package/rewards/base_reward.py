"""
BaseReward plugin: the canonical capture / obstacle / step-cost signal.

This is the ONLY path by which the base reward enters an experiment.
GridWorldEnv.step() applies no reward on its own; run_from_config adds this
plugin to the reward pipeline when rewards.base.enabled is true, and toggling
that flag is how the base reward is enabled or disabled. Because there is a
single application path, the base reward can no longer be double-counted
(issue #32).

Custom run scripts that assemble their own reward_fn must include this plugin
if they want the base signal; otherwise the environment produces no reward.
"""

from multi_agent_package.rewards.base import RewardFunction


class BaseReward(RewardFunction):
    """
    Uses the environment's canonical base reward.
    """

    def compute(self, env):
        base = env.base_reward()
        return {k: self.weight * v for k, v in base.items()}
