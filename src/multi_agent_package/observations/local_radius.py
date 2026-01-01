"""
Radius-based partial observability.

Agents see other agents and obstacles within a Manhattan radius.
"""

import numpy as np
from multi_agent_package.observations.base import ObservationBuilder


class LocalRadiusObservation(ObservationBuilder):
    """
    Parameters (from YAML):
    - radius: int
    - include_agents: bool
    - include_obstacles: bool
    """

    def build(self, env):
        radius = int(self.params.get("radius", 3))
        include_agents = bool(self.params.get("include_agents", True))
        include_obstacles = bool(self.params.get("include_obstacles", True))

        obs = {}

        for ag in env.agents:
            ax, ay = ag._agent_location
            local_agents = {}
            local_obstacles = {}

            if include_agents:
                for other in env.agents:
                    if other.agent_name == ag.agent_name:
                        continue
                    ox, oy = other._agent_location
                    d = abs(ax - ox) + abs(ay - oy)
                    if d <= radius:
                        local_agents[other.agent_name] = {
                            "rel_pos": (ox - ax, oy - ay),
                            "dist": d,
                            "type": other.agent_type,
                        }

            if include_obstacles:
                for i, obs_pos in enumerate(env._obstacle_location):
                    ox, oy = obs_pos
                    d = abs(ax - ox) + abs(ay - oy)
                    if d <= radius:
                        local_obstacles[f"obstacle_{i}"] = {
                            "rel_pos": (ox - ax, oy - ay),
                            "dist": d,
                        }

            obs[ag.agent_name] = {
                "local": ag._agent_location.copy(),
                "visible_agents": local_agents,
                "visible_obstacles": local_obstacles,
                "radius": radius,
            }

        return obs
