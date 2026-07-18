"""
Relative observation builder.

All positions relative to observing agent. Agent is always at origin.
"""

from typing import Dict, Any

import numpy as np

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
                    "pos": np.array([0, 0]),  # ALWAYS origin
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

    def encode(self, observation: dict, env) -> np.ndarray:
        include_agents = bool(self.params.get("include_agents", True))
        include_obstacles = bool(self.params.get("include_obstacles", True))
        include_walls = bool(self.params.get("include_walls", False))

        values = []
        local = observation.get("local", {})
        values.extend(self._vector(local.get("pos", [0.0, 0.0])).tolist())
        values.append(float(self._agent_type_id(local.get("type"))))
        team_base, team_index = self._team_features(local.get("team"))
        values.append(float(team_base))
        values.append(float(team_index))
        values.append(float(local.get("speed", 0.0)))

        agents_obs = observation.get("agents", {}) if include_agents else {}
        max_agents = max(0, len(env.agents) - 1)
        for name in sorted(agents_obs.keys()):
            entry = agents_obs[name]
            values.extend(self._vector(entry.get("rel_pos", [0.0, 0.0])).tolist())
            values.append(float(entry.get("dist", 0.0)))
            values.append(float(self._agent_type_id(entry.get("type"))))
            team_base, team_index = self._team_features(entry.get("team"))
            values.append(float(team_base))
            values.append(float(team_index))

        missing_agents = max(0, max_agents - len(agents_obs))
        values.extend([0.0] * (missing_agents * 6))

        obstacles_obs = observation.get("obstacles", {}) if include_obstacles else {}
        max_obstacles = len(env._obstacle_location)
        for name in sorted(obstacles_obs.keys(), key=lambda k: int(k.split("_")[1])):
            entry = obstacles_obs[name]
            values.extend(self._vector(entry.get("rel_pos", [0.0, 0.0])).tolist())
            values.append(float(entry.get("dist", 0.0)))

        missing_obstacles = max(0, max_obstacles - len(obstacles_obs))
        values.extend([0.0] * (missing_obstacles * 3))

        if include_walls:
            values.extend(
                [
                    float(observation.get("walls", {}).get("left", 0.0)),
                    float(observation.get("walls", {}).get("right", 0.0)),
                    float(observation.get("walls", {}).get("up", 0.0)),
                    float(observation.get("walls", {}).get("down", 0.0)),
                ]
            )
        else:
            values.extend([0.0, 0.0, 0.0, 0.0])

        return np.asarray(values, dtype=np.float32)
