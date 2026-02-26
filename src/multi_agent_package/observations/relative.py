"""
Relative observation builder.

All positions relative to observing agent. Agent is always at origin.
"""

import numpy as np
from typing import Dict, Any
from multi_agent_package.observations.base import ObservationBuilder


class RelativeObservation(ObservationBuilder):
    """
    Fully egocentric observation - agent frame.
    
    No radius filtering. Agent always at [0,0].
    """

    def build(self, env) -> Dict[str, Dict[str, Any]]:
        include_agents = self.params.get("include_agents", True)
        include_obstacles = self.params.get("include_obstacles", True)
        include_walls = self.params.get("include_walls", False)
        distance_type = self.params.get("distance_type", "manhattan")

        obs = {}

        for ag in env.agents:
            ax, ay = ag._agent_location

            agent_obs = {
                "local": {
                    "pos": np.array([0, 0]),       # ALWAYS origin
                    "type": ag.agent_type,
                    "team": ag.agent_team,
                    "speed": ag.agent_speed,
                }
            }

            # --- ALL other agents---
            if include_agents:
                agents_obs = {}
                for other in env.agents:
                    if other.agent_name == ag.agent_name:
                        continue

                    ox, oy = other._agent_location
                    rel_x = ox - ax
                    rel_y = oy - ay

                    if distance_type == "manhattan":
                        dist = abs(rel_x) + abs(rel_y)
                    else:
                        dist = float(np.sqrt(rel_x**2 + rel_y**2))

                    agents_obs[other.agent_name] = {
                        "rel_pos": np.array([rel_x, rel_y]),  # RELATIVE offset
                        "dist": dist,
                        "type": other.agent_type,
                        "team": other.agent_team,
                    }

                agent_obs["agents"] = agents_obs

            # --- ALL obstacles ---
            if include_obstacles:
                obstacles_obs = {}
                for i, obs_pos in enumerate(env._obstacle_location):
                    ox, oy = obs_pos
                    rel_x = ox - ax
                    rel_y = oy - ay

                    if distance_type == "manhattan":
                        dist = abs(rel_x) + abs(rel_y)
                    else:
                        dist = float(np.sqrt(rel_x**2 + rel_y**2))

                    obstacles_obs[f"obstacle_{i}"] = {
                        "rel_pos": np.array([rel_x, rel_y]),  # RELATIVE offset
                        "dist": dist,
                    }

                agent_obs["obstacles"] = obstacles_obs

            # --- Wall distances (optional) ---
            if include_walls:
                agent_obs["walls"] = {
                    "left": ax,
                    "right": env.size - 1 - ax,
                    "up": ay,
                    "down": env.size - 1 - ay,
                }

            obs[ag.agent_name] = agent_obs

        return obs