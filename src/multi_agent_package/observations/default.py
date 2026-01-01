"""
Default observation builder.

This reproduces the original GridWorldEnv observation behavior.
"""

from multi_agent_package.observations.base import ObservationBuilder


class DefaultObservation(ObservationBuilder):
    """
    Uses GridWorldEnv's internal default observation logic.
    """

    def build(self, env):
        # delegate to env's canonical method
        return env._default_observations()
