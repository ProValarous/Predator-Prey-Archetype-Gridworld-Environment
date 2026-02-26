"""
Absolute observation builder.

All positions in world/grid coordinates. Sees ALL entities.
"""

import numpy as np
from typing import Dict, Any
from multi_agent_package.observations.base import ObservationBuilder


class AbsoluteObservation(ObservationBuilder):
    """
    Fully absolute observation - world frame.
    
    No radius filtering. All positions in grid coordinates.
    """

    def build(self, env) -> Dict[str, Dict[str, Any]]:
        include_agents = self.params.get("include_agents", True)
        include_obstacles = self.params.get("include_obstacles", True)
        distance_type = self.params.get("distance_type", "euclidean")

        obs = {}

        for ag in env.agents:
            agent_pos = ag._agent_location.copy()
            ax, ay = agent_pos

            agent_obs = {
                "local": {
                    "pos": agent_pos,              # ABSOLUTE world position
                    "type": ag.agent_type,
                    "team": ag.agent_team,
                    "speed": ag.agent_speed,
                }
            }

            if include_agents:
                agents_obs = {}
                for other in env.agents:
                    if other.agent_name == ag.agent_name:
                        continue

                    other_pos = other._agent_location.copy()
                    ox, oy = other_pos

                    if distance_type == "manhattan":
                        dist = abs(ox - ax) + abs(oy - ay)
                    else:
                        dist = float(np.sqrt((ox - ax)**2 + (oy - ay)**2))

                    agents_obs[other.agent_name] = {
                        "pos": other_pos,          # ABSOLUTE position
                        "dist": dist,
                        "type": other.agent_type,
                        "team": other.agent_team,
                    }

                agent_obs["agents"] = agents_obs

            if include_obstacles:
                obstacles_obs = {}
                for i, obs_pos in enumerate(env._obstacle_location):
                    obs_copy = obs_pos.copy()
                    ox, oy = obs_copy

                    if distance_type == "manhattan":
                        dist = abs(ox - ax) + abs(oy - ay)
                    else:
                        dist = float(np.sqrt((ox - ax)**2 + (oy - ay)**2))

                    obstacles_obs[f"obstacle_{i}"] = {
                        "pos": obs_copy,           # ABSOLUTE position
                        "dist": dist,
                    }

                agent_obs["obstacles"] = obstacles_obs

            obs[ag.agent_name] = agent_obs

        return obs