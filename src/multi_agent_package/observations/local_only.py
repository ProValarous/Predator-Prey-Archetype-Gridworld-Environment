"""
Local-only observation.

Agents observe ONLY their own position.
"""

import numpy as np

from multi_agent_package.observations.base import ObservationBuilder


class LocalOnlyObservation(ObservationBuilder):
    """
    Observation = agent's own location only.
    """

    def build(self, env):
        obs = {}

        for ag in env.agents:
            obs[ag.agent_name] = {
                "local": ag._agent_location.copy(),
                "global": None,
            }

        return obs

    def encode(self, observation: dict, env) -> np.ndarray:
        return self._vector(observation.get("local", []))
