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

    def encode(self, observation: dict, env) -> np.ndarray:
        radius = float(observation.get("radius", self.params.get("radius", 3)))
        include_agents = bool(self.params.get("include_agents", True))
        include_obstacles = bool(self.params.get("include_obstacles", True))

        values = []
        values.extend(self._vector(observation.get("local", [0.0, 0.0])).tolist())
        values.append(radius)

        visible_agents = observation.get("visible_agents", {}) if include_agents else {}
        # Fixed slot per agent identity: iterate over every agent name in a
        # stable order and write zeros for any not currently within radius. The
        # observing agent never appears in visible_agents, so its own slot stays
        # zero. This keeps each identity in the same slot regardless of which
        # others are visible, instead of packing visible agents first (issue
        # #34: the old present-first packing made a slot mean different agents
        # on different steps, silently corrupting learning).
        for name in sorted(a.agent_name for a in env.agents):
            entry = visible_agents.get(name)
            if entry is None:
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                values.append(1.0)
                values.extend(self._vector(entry.get("rel_pos", [0.0, 0.0])).tolist())
                values.append(float(entry.get("dist", 0.0)))
                values.append(float(self._agent_type_id(entry.get("type"))))

        visible_obstacles = (
            observation.get("visible_obstacles", {}) if include_obstacles else {}
        )
        # Obstacle indices are assigned once per reset() and never reordered
        # within an episode, so "obstacle_i" is a stable identity too.
        for i in range(len(env._obstacle_location)):
            entry = visible_obstacles.get(f"obstacle_{i}")
            if entry is None:
                values.extend([0.0, 0.0, 0.0, 0.0])
            else:
                values.append(1.0)
                values.extend(self._vector(entry.get("rel_pos", [0.0, 0.0])).tolist())
                values.append(float(entry.get("dist", 0.0)))

        return np.asarray(values, dtype=np.float32)
