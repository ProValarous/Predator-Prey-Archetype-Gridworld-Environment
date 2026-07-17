"""
Wrapper for GridWorldEnv.base_reward().

DEPRECATED / DO NOT USE VIA reward_fn.

base_reward() is applied directly by GridWorldEnv.step(), gated by the
include_base_reward constructor flag (wired from rewards.yaml -> base.enabled
in run_from_config.build_environment()). Toggle the base reward through that
flag, not by adding this plugin to the reward_fn pipeline.

If you add get_reward_function("base") into the reward_fns list in
build_environment(), you will double-count every capture/death/obstacle event.
"""

from multi_agent_package.rewards.base import RewardFunction


class BaseReward(RewardFunction):
    """
    Uses the environment's canonical base reward.
    """

    def compute(self, env):
        base = env.base_reward()
        return {k: self.weight * v for k, v in base.items()}
