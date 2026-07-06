"""
Default observation builder.

This reproduces the original GridWorldEnv observation behavior.
"""

import numpy as np

from multi_agent_package.observations.base import ObservationBuilder


class DefaultObservation(ObservationBuilder):
    """
    Uses GridWorldEnv's internal default observation logic.
    """

    def build(self, env):
        # delegate to env's canonical method
        return env._default_observations()

    def encode(self, observation: dict, env) -> np.ndarray:
        values = []

        values.extend(self._vector(observation.get("local", [])).tolist())

        global_obs = observation.get("global") or {}
        dist_agents = global_obs.get("dist_agents", {})
        for name in sorted(dist_agents.keys()):
            values.append(float(dist_agents[name]))

        dist_obstacles = global_obs.get("dist_obstacles", {})
        for name in sorted(dist_obstacles.keys()):
            values.append(float(dist_obstacles[name]))

        return np.asarray(values, dtype=np.float32)
