# Guide: Writing a Custom Observation Builder

Full working example from scratch to registered and testable.

---

## What you're building

An observation builder gives each agent a view of the world at each step. You control exactly what information agents receive — positions, distances, types, obstacles, or anything derived from the env state.

---

## Step 1 — Create the file

```
src/multi_agent_package/observations/my_obs.py
```

```python
import numpy as np
from multi_agent_package.observations.base import ObservationBuilder


class MyObservation(ObservationBuilder):
    """
    Example: each agent sees its own position plus the Manhattan
    distance to every other agent, grouped by type.
    
    Params (from YAML):
        max_dist: int — distances clamped to this value (default 99)
    """

    def build(self, env) -> dict:
        max_dist = int(self.params.get("max_dist", 99))
        obs = {}

        for ag in env.agents:
            ax, ay = ag._agent_location

            # distances to other agents, split by type
            pred_dists = []
            prey_dists = []

            for other in env.agents:
                if other.agent_name == ag.agent_name:
                    continue
                ox, oy = other._agent_location
                d = min(abs(ax - ox) + abs(ay - oy), max_dist)
                if other.agent_type.startswith("predator"):
                    pred_dists.append(d)
                else:
                    prey_dists.append(d)

            obs[ag.agent_name] = {
                "local": ag._agent_location.copy(),  # own [x, y]
                "pred_dists": pred_dists,             # list of ints
                "prey_dists": prey_dists,
            }

        return obs
```

**Rules:**
- Always iterate `env.agents` — never hardcode agent names
- Never write to `env` or any `agent._agent_location`
- Return exactly one key per agent in `env.agents`
- All values must be hashable by `IQL._encode_state()` — use lists/ints/tuples, not sets or custom objects

---

## Step 2 — Register it

Open `src/multi_agent_package/registry/observation_registry.py`:

```python
from multi_agent_package.observations.my_obs import MyObservation

_OBSERVATION_REGISTRY = {
    "default":      DefaultObservation,
    "local_only":   LocalOnlyObservation,
    "local_radius": LocalRadiusObservation,
    "absolute":     AbsoluteObservation,
    "relative":     RelativeObservation,
    "my_obs":       MyObservation,          # ← add this line
}
```

---

## Step 3 — Export it (optional but good practice)

Open `src/multi_agent_package/observations/__init__.py`, add:

```python
from multi_agent_package.observations.my_obs import MyObservation
__all__ = [..., "MyObservation"]
```

---

## Step 4 — Configure it

```yaml
# configs/observations.yaml
observations:
  type: my_obs
  params:
    max_dist: 10
```

---

## Step 5 — Test it

```python
from multi_agent_package.core.agent import Agent
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.observations.my_obs import MyObservation

agents = [
    Agent(agent_type="predator", agent_team="predator_1", agent_name="pred_1"),
    Agent(agent_type="prey",     agent_team="prey_1",     agent_name="prey_1"),
]
env = GridWorldEnv(agents=agents, size=6, perc_num_obstacle=0, render_mode=None, seed=0)
env.reset()

builder = MyObservation(max_dist=10)
obs = builder.build(env)

assert set(obs.keys()) == {"pred_1", "prey_1"}
assert "local" in obs["pred_1"]
assert isinstance(obs["pred_1"]["prey_dists"], list)

# Verify no mutation
loc_before = env.agents[0]._agent_location.copy()
builder.build(env)
assert (env.agents[0]._agent_location == loc_before).all()

print("Observation output:", obs)
```

---

## Verifying IQL compatibility

`IQL._encode_state()` recursively converts the observation dict to a hashable tuple. Your observation is compatible if:

```python
from baselines.IQL.iql import IQL

env.observation_builder = builder.build
env.reset()
obs, _ = env.reset()

algo = IQL(env, {"episodes": 1, "epsilon": 1.0, "seed": 0})
s = algo._encode_state(obs["pred_1"])
hash(s)  # must not raise
```

If `hash(s)` raises `TypeError: unhashable type`, your builder returned a non-hashable value (e.g., a numpy array directly at the top level — wrap it in a list or use `.tolist()`).

---

## Checklist

- [ ] Inherits from `ObservationBuilder`
- [ ] Returns one entry per agent in `env.agents`
- [ ] No writes to `env` or agent state
- [ ] All values serializable to hashable tuple
- [ ] Registered in `observation_registry.py` with matching YAML key
- [ ] Manual test confirms no mutation
- [ ] `hash(algo._encode_state(obs[aid]))` succeeds
